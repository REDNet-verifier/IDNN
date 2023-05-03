import torch
import torch.nn as nn
import torch.nn.functional as F
import cupy as cp
from cupyx.scipy.sparse import csr_matrix, hstack, vstack
import onnx
from onnx import helper, shape_inference
from auto_LiRPA import * #BoundedTensor, BoundedModule
from auto_LiRPA.operators import *

from math import floor
import numpy as np
from cupy.sparse import identity
from onnx2torch import convert

#import gc
#import sys
#import nvidia_smi


#from pympler.asizeof import asizeof

class Node:
    def __init__(self):
        self._input = []
        self._shape = None
        self._name  = None
        self._temp = None
        self._output = []
        self._width = None
        
    def set_name(self, name):
        self._name = name
        
    def set_shape(self, shape):
        self._shape = shape
        self._width = shape.numel()
        
    def append_input(self, input):
        self._input.append(input)
        
        
        
class SumLinear(Node):
    def __init__(self):
        super().__init__()
        
    def append_input(self, weight, bias, input):
        self._input.append((weight, bias, input))

    def forward(self):        
        self._temp = 0
        for input in self._input:
            w = input[0]
            b = input[1]
            x = input[2]._temp            
            self._temp += w@x+b
        
        #print(self._temp)
        
    def forward_bound(self):
        lb = 0
        ub = 0
        for input in self._input:
            w = input[0]
            b = input[1]
            
            if isinstance(w, cp.ndarray):
                w0 = cp.minimum(w, 0)
                w1 = cp.maximum(w, 0)
            else:
                w0 = w.minimum(0)
                w1 = w.maximum(0)
                
            xl, xu = input[2]._temp            
            
            lb += w0@xu + w1@xl + b
            ub += w0@xl + w1@xu + b                  
        
        self._temp = (lb, ub)
        
            
class ReLU(Node):
    def __init__(self):
        super().__init__()
        self._lower_bound = None
        self._upper_bound = None
        
    def forward(self):
        input = self._input[0]        
        self._temp =  cp.maximum(input._temp, 0)
        #print(self._temp)
        

    def forward_bound(self):
        input = self._input[0]
        ind = cp.concatenate((self._unstable, self._activate))
        
        lb = cp.maximum(input._temp[0], 0)
        ub = cp.maximum(input._temp[1], 0)
        
        self._temp = (lb, ub)
        
    def compute_unstable_neurons(self, input_lbs: torch.Tensor, input_ubs: torch.Tensor):
        # input bounds in flatten mode
        
        input_lbs = input_lbs.view(-1)
        input_ubs = input_ubs.view(-1)
        self._activate = cp.asarray(((input_lbs>=0)&(input_ubs>0)).nonzero().view(-1), dtype = np.int32)
        self._unstable = cp.asarray(((input_lbs<0)&(input_ubs>0)).nonzero().view(-1), dtype = np.int32)   
        #if self._activate.shape[0] == 0 and self._unstable.shape[0] == 0:
        #    self._activate = cp.asarray([0], dtype = np.int32)
        
        
        
class Input(Node):
    def __init__(self, shape):
        super().__init__()
        self.set_shape(shape)     

    def compute_unstable_neurons(self, input_lbs: torch.Tensor, input_ubs: torch.Tensor):        
        input_lbs = input_lbs.view(-1)
        input_ubs = input_ubs.view(-1)
        self._activate = cp.asarray(((input_lbs>=0)&(input_ubs>0)).nonzero().view(-1), dtype = np.int32)
        self._unstable = cp.asarray(((input_lbs<0)&(input_ubs>0)).nonzero().view(-1), dtype = np.int32)    
        #print(self._unstable.shape, self._activate.shape, self._width)
        if self._activate.shape[0] == 0 and self._unstable.shape[0] == 0:
            self._activate = cp.asarray([0], dtype = np.int32)

        
class Output(Node):
    def __init__(self, shape):
        super().__init__()
        self.set_shape(shape)    

        
class Network:
    def __init__(self):
        self._output = None
        self._input = []
        self._layers = []

        self._name_to_layer = {}
        self._output_to_layer = {}
        self._relu_names = []

        self._node_id = 0
    
        self._dtype = None
        self._device = None
    
        self._bound_propagation_time = None
        self._final_name = None
        
    def print(self, s):
        size = 0
        for layer in self._layers:
            if isinstance(layer, ReLU):
                #print(layer._name,"----",type(layer), "size:", layer._width)        
                size += layer._width
            #else: 
            #    print(layer._name,"----",type(layer))        
        
        print("The number of neurons(",s,"):",size)
    
    def preprocess(self, layer):
        #self._layers.append(layer)        
        self.preprocess_deactivate_first(layer)
        
        
        
    def transform_to_torch(self):
        layers = []
        layers.append(nn.Flatten())
        self._output._input[0]._output=[self._output]
        for layer in self._layers:
            if isinstance(layer, ReLU):
                layers.append(nn.ReLU(inplace=False))
            else:
                x = nn.Linear(layer._input[0][2]._width, layer._output[0]._width)
                x.weight = nn.Parameter(torch.tensor(layer._input[0][0], dtype = self._dtype))
                x.bias = nn.Parameter(torch.tensor(layer._input[0][1], dtype = self._dtype))
                layers.append(x)
        net = nn.Sequential(*layers)
        print(net)
        return net

    
    
    def forward(self, x):
        x = cp.asarray(x).ravel()
        self._input[0]._temp = x
        for layer in self._layers:
            layer.forward()
        
        print(type(self._output), self._output._input[0]._temp)
    
    def forward_bound(self, lb, ub):
        lb = cp.asarray(lb).ravel()
        ub = cp.asarray(ub).ravel()
        
        self._input[0]._temp = (lb, ub)
 
        for layer in self._layers:           
            layer.forward_bound()
            
        print("bounds: ",self._output._input[0]._temp)


    def read_onnx_to_pytorch(self, onnx_path, data_ub, data_lb):
        model = convert(onnx_path).cuda()
        data_lb, data_ub = data_lb.cuda(), data_ub.cuda()
        data = (data_ub+data_lb)/2
        
        boundnet = BoundedModule(model, torch.zeros_like(data), bound_opts={"conv_mode": "patches"})
        ptb = PerturbationLpNorm(eps=None, x_L=data_lb, x_U=data_ub)
        bound_input = BoundedTensor(data, ptb).to(data_lb.device)        
    
        return self.read_IDNN(boundnet, bound_input)
        
    def read_onnx_to_onnx(self, onnx_path, data_ub, data_lb):
        model = convert(onnx_path)
        data_lb, data_ub = data_lb.cuda(), data_ub.cuda()
        data = (data_ub+data_lb)/2
        boundnet = BoundedModule(model, torch.zeros_like(image), bound_opts={"conv_mode": "patches"})
        ptb = PerturbationLpNorm(norm='inf', eps=None, x_L=data_lb, x_U=data_ub)
        bound_input = BoundedTensor(data, ptb).to(data_lb.device)        
    
        return read_IDNN(boundnet, bound_input)

    def read_pytorch(self, model, data_lb, data_ub):
        data_lb, data_ub = data_lb.cuda(), data_ub.cuda()
        data = (data_ub+data_lb)/2
        model = model.cuda()
        
        boundnet = BoundedModule(model, torch.zeros_like(data), bound_opts={"conv_mode": "patches"}).cuda()
        ptb = PerturbationLpNorm(eps=None, x_L=data_lb, x_U=data_ub)
        bound_input = BoundedTensor(data, ptb).to(data_lb.device)        
        
        return self.read_IDNN(boundnet, bound_input)
        
    
    def init_node(self, layer, x, inputs):
        x.set_name(layer.name)
        self._name_to_layer[layer.name] = x
        self._layers.append(x)
        x.set_shape(layer.output_shape)        
        for input in inputs:
            if not self._redundant_layer[input.name]:
                x._input.append(self._name_to_layer[input.name])
        #print(type(layer))
        if isinstance(x, ReLU):
            l = layer.inputs[0].lower.detach()
            u = layer.inputs[0].upper.detach()    
            x.compute_unstable_neurons(l, u)
            #print(x, "unstable:", x._unstable.shape[0], "stable", x._activate.shape[0], "width", x._width)
        
    def remove_dead_layer(self, layer):
        if isinstance(layer, Input):
            return
        self._out_degree[layer._name] -= 1 
        if self._out_degree[layer._name] == 0:
            if layer._name not in self._dead_layer:
                self._dead_layer[layer._name] = False
                for input in layer._input:
                    self.remove_dead_layer(input)
                    
    def init_DAG(self, layers):
        self._name_to_bound_layer = {}
        for layer in layers:
            self._name_to_bound_layer[layer.name] = layer
            
            redundant = True
            if isinstance(layer, BoundInput):
                redundant = False
                if isinstance(layer, BoundParams) or isinstance(layer, BoundBuffers):
                    redundant = True
            else:
                for input in layer.inputs:
                    if self._redundant_layer[input.name] == False:
                        redundant = False
                        break        
            self._redundant_layer[layer.name] = redundant
            
            if redundant:
                continue
            elif isinstance(layer, BoundInput):
                x = Input(layer.output_shape)
                x.set_name(layer.name)
                self._name_to_layer[layer.name] = x
                self._input.append(x)
            elif isinstance(layer, BoundRelu):
                self.init_node(layer, ReLU(), layer.inputs)
            else: 
                self.init_node(layer, SumLinear(), layer.inputs) 

        self._node_id = len(layers)
        x = Output(layer.output_shape)
        x.set_name("output"+str(self._node_id))
        self._node_id +=1
        self._name_to_layer[x._name] = x
        self._output = x
        x.append_input(self._name_to_layer[self._final_name])   

        self.print("original network")

        self._out_degree = {}
        self._dead_layer = {}
        for layer in self._input:    
            self._out_degree[layer._name] = 0            
            
        for layer in self._layers:
            self._out_degree[layer._name] = 0
            
        for layer in self._layers:
            for input in layer._input:
                self._out_degree[input._name] += 1
                
        for layer in self._layers:
            if isinstance(layer, ReLU):
                if layer._unstable.shape[0]==0 and layer._activate.shape[0]==0:
                    if layer._name not in self._dead_layer:
                        self.remove_dead_layer(layer._input[0])
                        self._redundant_layer[layer._name] = True
                        self._constant[layer._name] = torch.zeros(layer._shape, dtype = self._dtype, device="cuda")
                    self._dead_layer[layer._name] = True

        
        for layer in layers:
            if not isinstance(layer, BoundInput) or self._redundant_layer[layer.name]:
                f = True
                
                for x in layer.inputs:
                    if not self._redundant_layer[x.name]:
                        f=False
                        break
                if f:
                    self._redundant_layer[layer.name] = True
                    if layer.name not in self._constant:
                        para = []
                        for x in layer.inputs:
                            para.append(self._constant[x.name])
                        self._constant[layer.name] = layer.forward(*para).clone().detach()    
        #print(len(self._constant), len(layers))
        
        self._new_layers = []
        for layer in self._layers:
            if not self._redundant_layer[layer._name] and (layer._name not in self._dead_layer):
                self._new_layers.append(layer)
                layer._input = []
        
        
        #print("final layer num:", len(self._new_layers), len(self._layers))
        
        self._layers = self._new_layers

    def init_network(self, layers):
        self._redundant_layer = {}
        self._constant = {}
        
        self.init_DAG(layers)        
        if self._output._input[0]._name in self._constant:
            ############### all intermediate layers have been removed #######################
            x = SumLinear()
            const = self._constant[self.output._input[0]._name]
            b = cp.array(const).ravel()
            w = csr_matrix(cp.zeros((self._input[0]._width, self._output._width), dtype = self._dtype))
            x._input.append((w, b, self._input[0]))
            
            x.set_name("linear"+str(self._node_id))
            x.set_shape(b.shape)
            self._node_id +=1
            self._name_to_layer[x._name] = x
            self._layers = [x]
            self.output._input = [x]
        else:
            for x in self._layers:
                layer = self._name_to_bound_layer[x._name]
                if isinstance(layer, BoundConv):
                    self.conv_node(layer)
                elif isinstance(layer, BoundRelu):
                    self.relu_node(layer)
                elif isinstance(layer, BoundAdd):
                    self.add_node(layer)      
                elif isinstance(layer, BoundFlatten) or isinstance(layer, BoundReshape):
                    self.flatten_node(layer)
                elif isinstance(layer, BoundLinear):
                    self.linear_node(layer)
                elif isinstance(layer, BoundSub):
                    self.sub_node(layer)
                elif isinstance(layer, BoundDiv):
                    self.div_node(layer)
                elif isinstance(layer, BoundConcat):
                    self.concate_node(layer)
                #elif isinstance(layer, BoundSplit):
                #    self.split_node(layer)
                elif isinstance(layer, BoundSqueeze) or isinstance(layer, BoundUnsqueeze):
                    self.flatten_node(layer)
                #elif isinstance(layer, BoundDropout) or isinstance(layer, BoundMatMul) or isinstance(layer, BoundBatchNormalization):
                #    assert False, "BoundDropout or BoundMatMul or BoundBatchNormalization Shouldn't exist in the selected benchmark!"
                else:
                    assert False, "To Do!!!!!!!!!!!!!!!! Missing Layer:" + type(layer)                
    
    def read_IDNN(self, boundnet: BoundedModule, bound_input: BoundedTensor):
        self._final_name = boundnet.final_name
        layers = boundnet._modules.values()
        #print(boundnet.__dict__.keys())
        start = time.time()
        self._dtype = torch.float32
        
        #lb, ub = boundnet.compute_bounds(x=(bound_input,), method='ibp')    
        lb, ub, aux_reference_bounds =  boundnet.init_slope((bound_input,), share_slopes=True, c=None, method = 'backward')
        end = time.time()
        self._bound_propagation_time = end-start
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!Network Reduction!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")         
        
        self.init_network(layers)
        self.preprocess_deactivate_second()
        d = bound_input.data
        self.reduction(boundnet,bound_input)
        #self.forward(d)
        end = time.time()
        self.print("reduced network")
        reduce_time = end-start
        bound_time = self._bound_propagation_time
        print("Reduction Time:",end-start, "which includes the bound propagation time:", self._bound_propagation_time)
        return self.transform_to_torch(), reduce_time, bound_time
        
    def linear_node(self, layer):
        x = self._name_to_layer[layer.name]
        inputs = layer.inputs
        
        weights = self._constant[inputs[1].name]
        bias = self._constant[inputs[2].name]
        
        if bias is None:           
            bias = torch.zeros(weights.shape[0]).to(weights)
        
        input = self._name_to_layer[inputs[0].name]
        x.append_input(csr_matrix(cp.asarray(weights)), cp.asarray(bias).ravel(), input)
        self.preprocess(x)

    def split_node(self, layer):
        x = self._name_to_layer[layer.name]
        inputs = layer.inputs        
        input = self._name_to_layer[inputs[0].name]
        self.preprocess(x)
        ########To Do############
        
        
    def flatten_node(self, layer):
        x = self._name_to_layer[layer.name]
        inputs = layer.inputs        
        input = self._name_to_layer[inputs[0].name]
        
        num = input._width
        weight, bias = identity(num, dtype = np.float32, format="csr"), cp.zeros(num, dtype = np.float32)
        x.append_input(weight, bias, input)
        self.preprocess(x)
        
        
    def add_node(self, layer):
        x = self._name_to_layer[layer.name]
        inputs = layer.inputs        
        output_shape = layer.output_shape
        
        input1 = layer.inputs[0]
        input2 = layer.inputs[1]
        
        
        if input1.name in self._constant:
            value = self._constant[input1.name]

            input = self._name_to_layer[input2.name]
            num = input._width
            weight, bias = identity(num, dtype = np.float32, format="csr"), cp.zeros(num, dtype = np.float32)
            bias = (bias + cp.asarray(value).ravel()).ravel()
            x.append_input(weight, bias, input)
            
        elif input2.name in self._constant:
            value = self._constant[input2.name]
            
            input = self._name_to_layer[input1.name]
            num = input._width
            weight, bias = identity(num, dtype = np.float32, format="csr"), cp.zeros(num, dtype = np.float32)
            bias = (bias + cp.asarray(value).ravel()).ravel()
            x.append_input(weight, bias, input)
        else:
            for input in layer.inputs:
                input = self._name_to_layer[input.name]
                num = input._width
                weight, bias = identity(num, dtype = np.float32, format="csr"), cp.zeros(num, dtype = np.float32)
                x.append_input(weight, bias, input)                
    
        self.preprocess(x)
                
    def relu_node(self, layer):
        x = self._name_to_layer[layer.name]
        x._input.append(self._name_to_layer[layer.inputs[0].name])
        
        #print(x._width, len(x._unstable), len(x._activate))
        
        self.preprocess(x)
        
    def conv_node(self, layer):
        x = self._name_to_layer[layer.name]
        inputs = layer.inputs     
        
        input_shape = layer.input_shape
        
        weights = self._constant[inputs[1].name]
        if layer.has_bias:    
            bias = self._constant[inputs[2].name]
        else:
            bias = torch.zeros(weights.shape[0]).to(weights)
            
        dilations = layer.dilation
        groups = layer.groups
        strides = layer.stride
        pads = layer.padding
        
        kernel_shape = (weights.shape[2], weights.shape[3])
        
        input_channel, height, width = input_shape[1], input_shape[2], input_shape[3]
        output_h = floor((height + 2 * pads[0] - dilations[0] * (kernel_shape[0] - 1) - 1) / strides[0]+ 1)
        output_w = floor((width + 2 * pads[1] - dilations[1] * (kernel_shape[1] - 1)  - 1) / strides[1] + 1)
        
        pre_node_size = input_channel * height * width
        output_channel = weights.shape[0]
        node_size = output_channel * output_h * output_w
        #pre_neuro_id = np.arange(pre_node_size, dtype=np.int32).reshape(1, input_channel, height, width)
        #pre_neuro_id = np.pad(pre_neuro_id,((0,0),(0,0), (pads[0],pads[0]), (pads[1],pads[1])), constant_values=-1)
        pre_neuro_id = torch.arange(pre_node_size, dtype=torch.float32, device="cuda").view(input_channel, height, width)
        pre_neuro_id = F.pad(pre_neuro_id, (pads[0],pads[0],pads[1],pads[1]), "constant", pre_node_size)
        
        dilation_shape = dilations[0] * (kernel_shape[0] - 1) + 1, dilations[1] * (kernel_shape[1] - 1) + 1
        
        #print(pre_node_size, node_size)
        
        #print(weights.shape)        
        kernel = torch.zeros((output_channel, input_channel, dilation_shape[0], dilation_shape[1]), device="cuda", dtype = weights.dtype)
        #print(kernel.shape)

        if dilations[0] > 1:
            ind = (torch.ones((kernel_shape[0],kernel_shape[1]))>0).nonzero()
            ind1 = ind.clone()
            ind[:, 0] = ind[:, 0]*dilations[0]
            ind[:, 1] = ind[:, 1]*dilations[1]
            kernel[:, :, ind] = weights[:,:, ind1]
            #for i in range(kernel_shape[0]):
            #    for j in range(kernel_shape[1]):
            #        kernel[:, :, dilations[0] * i, dilations[1] * j] = weights[:, :, i, j]
        else:
            kernel = weights  
                
        #print(pre_neuro_id.shape)
        
        cols = F.unfold(pre_neuro_id, kernel_size=dilation_shape, stride=strides, padding=(0,0)).int().transpose(0, 1)
        b =  bias.view(-1,1).repeat(1, cols.shape[0]).view(-1)        

        values = kernel.contiguous().view(output_channel, -1)
        #print(cols.shape, values.shape,  kernel.shape)
        values = values.repeat(1, cols.shape[0]).view(-1)
        
        #To Do: use index selection to directly remove deactivate neurons
        
        
        cols = cols.repeat(output_channel, 1)
        rows = torch.arange(cols.shape[0], dtype = torch.int32, device = "cuda").view(-1,1).repeat(1,cols.shape[1]).view(-1)
        cols = cols.view(-1)
                
        '''
        cols = pre_neuro_id.unfold(2, dilation_shape[0], strides[0]).unfold(3, dilation_shape[1], strides[1])
        values = kernel.contiguous().view(output_channel, -1)
        cols = cols.permute(0,2,3,1,4,5).contiguous().view(-1,values.shape[1])
        
        b =  bias.view(-1,1).repeat(1, cols.shape[0]).view(-1)
        n = cols.shape[0]*output_channel
        #print(b.shape, cols.shape, n)
        rows = torch.arange(n, dtype = torch.int32, device = "cuda").view(-1,1).repeat(1,cols.shape[1]).view(-1)
        values = values.repeat(1, cols.shape[0]).view(-1)
        cols = cols.view(-1).repeat(output_channel, 1).view(-1)
        ind = (cols!=-1).nonzero()
        values = values[ind]
        rows = rows[ind]
        cols = cols[ind]
        '''
        #print(node_size, rows.shape, cols.shape, values.shape)
        
        #ind = cols!=-1
        #values = values[ind]
        #rows = rows[ind]
        #cols = cols[ind]
        
        out_bias = cp.asarray(b).ravel()
        out_weight = csr_matrix((cp.asarray(values).ravel(), (cp.asarray(rows).ravel(), cp.asarray(cols).ravel())), shape=(node_size, pre_node_size+1), dtype = np.float32)[:,:pre_node_size]
        
        input = self._name_to_layer[inputs[0].name]      
        
        x.append_input(out_weight, out_bias, input)
        #x.set_shape(layer.output_shape)        
        self.preprocess(x)
        del cols, values, b, rows
        
    def sub_node(self, layer):
        # Sub opereator computes: input1 - input2
        x = self._name_to_layer[layer.name]
        inputs = layer.inputs        
        input = self._name_to_layer[inputs[0].name]
        output_shape = layer.output_shape
        
        # print("sub layer.input_shape", input_shape)
        input1 = layer.inputs[0]
        input2 = layer.inputs[1]   
        if input2.name in self._constant:
            value = self._constant[input2.name]
            input = self._name_to_layer[input1.name]
            num = input._width
            weight, bias = identity(num, dtype = np.float32, format="csr"), cp.zeros(num, dtype = np.float32)
            bias = (bias.reshape(output_shape) - cp.asarray(value)).ravel()
            x.append_input(weight, bias, input)
        elif input1.name in self._constant:
            value = self._constant[input1.name]
            input = self._name_to_layer[input2.name]
            num = input._width
            weight, bias = -identity(num, dtype = np.float32, format="csr"), cp.zeros(num, dtype = np.float32)
            bias = (bias.reshape(output_shape) + cp.asarray(value)).ravel()
            x.append_input(weight, bias, input)
        else:
            for i, input in enumerate(layer.inputs):
                input = self._name_to_layer[input.name]
                num = input._width
                weight = identity(num, dtype = np.float32, format="csr")
                weight = weight if i == 0 else -weight
                bias = cp.zeros(num, dtype = np.float32)
                x.append_input(weight, bias, input)  

        self.preprocess(x)

    def div_node(self, layer):
        # Div opereator computes: input1 div input2
        x = self._name_to_layer[layer.name] 
        output_shape = layer.output_shape
        
        input1 = layer.inputs[0]
        input2 = layer.inputs[1]   
        if input2.name in self._constant:
            value =  self._constant[input2.name]
            input = self._name_to_layer[input1.name]
            num = input._width
            rows = cp.arange(num,dtype=np.int32)
            cols = cp.arange(num,dtype=np.int32)
            const = cp.divide(cp.full(output_shape, 1.0, dtype=np.float32), value).ravel()
            weight = csr_matrix((const, (rows, cols)), shape=(num, num), dtype = np.float32)
            x.append_input(weight, cp.zeros(num), input)         
        else:
            assert False, "We only support constant as denominator in Div() operator"

        self.preprocess(x)

    
    def concate_node(self, layer):
        ##To Do
        
        x = self._name_to_layer[layer.name]
        inputs = layer.inputs        
        output_shape = layer.output_shape
        x.set_shape(layer.output_shape)

        dim=int(layer.axis)

        width = 0
        for input1 in layer.inputs:       
            if input1.name not in self._constant:        
                input = self._name_to_layer[input1.name]
                width += input._width
            
        temp = 0
        b = cp.zeros(width, dtype = np.float32)
        for input1 in layer.inputs:
            input = self._name_to_layer[input1.name]
            num = input._width
            if input1.name not in self._constant:
                rows = cp.arange(num,dtype=np.int32) + temp
                cols = cp.arange(num,dtype=np.int32)
                const = cp.full(num, 1, dtype=np.float32)
                weight = csr_matrix((const, (rows, cols)), shape=(width, num), dtype = np.float32)
                bias = cp.zeros(width, dtype = np.float32)
                x.append_input(weight, bias, input) 
                temp += num
                del rows, cols, const
            else:
                b[temp:temp+num] = cp.asarray(self._constant[input1.name].clone().detach()).ravel()
        
        weight, bias, input = x._input[0]
        x._input[0] = (weight, b, input)
                    
        self.preprocess(x)
        
        '''
        x = self._name_to_layer[layer.name]
        inputs = layer.inputs        
        output_shape = layer.output_shape
        x.set_shape(layer.output_shape)

        input1 = layer.inputs[0]
        input2 = layer.inputs[1]
        dim=int(layer.axis)
        # print(input1)
        # print(input2)
        # print(dim)
        
        input = self._name_to_layer[input1.name]
        num = input._width
        rows = cp.arange(num,dtype=np.int32)
        cols = cp.arange(num,dtype=np.int32)
        const = cp.full(num, 1, dtype=np.float32)
        weight = csr_matrix((const, (rows, cols)), shape=(2*num, num), dtype = np.float32)
        bias = cp.zeros(2*num, dtype = np.float32)
        x.append_input(weight, bias, input) 
        del rows, cols, const

        input = self._name_to_layer[input2.name]
        num = input._width
        rows = cp.arange(num,dtype=np.int32) + num
        cols = cp.arange(num,dtype=np.int32)
        const = cp.full(num, 1, dtype=np.float32)
        weight = csr_matrix((const, (rows, cols)), shape=(2*num, num), dtype = np.float32)
        bias = cp.zeros(2*num, dtype = np.float32)
        x.append_input(weight, bias, input) 

        self.preprocess(x)
       
        del rows, cols, const
         '''
        
    def print_network(self):
        for input in self._input:
            print("%%%%%%%%Input layer%%%%%%%%%", input)
            print("input layer shape",input._shape)
            print("input layer links to", input._output[0])
        print("len of self._layers", len(self._layers))
        for layer in self._layers:
            print("%%%%%%%%Intermediate layer%%%%%%%%%",layer)
            if isinstance(layer, SumLinear):
                print("linear layer shape", layer._shape)
                for input in layer._input:
                    print("input weight shape", input[0]._shape)
                    print("input bias shape", input[1]._shape)
                    print("input layer", input[2])
                for output in layer._output:
                    print("output layer", output)
            else:
                print("relu shape", layer._shape)
                for input in layer._input:
                    print("relu input layer", input)
                for output in layer._output:
                    print("relu output layer", output)
                
    
    def back(self, x):        
        if isinstance(x, SumLinear):
            for input1 in x._input:
                y = input1[2] 
                if isinstance(y, SumLinear):
                    self.back(y)
            pre_input = []
            pos = {}
            for input1 in x._input:
                w1, b1, y = input1[0], input1[1], input1[2]
                
                if isinstance(y, SumLinear):
                    for input2 in y._input:
                        w2, b2, z = input2
                        w3 = w1@w2
                        b3 = w1@b2
                        if z._name in pos:
                            j = pos[z._name]
                            input3 = pre_input[j]
                            w3 = w3 + input3[0]
                            b3 = b3 + input3[1]
                            pre_input[j] = (w3,b3,z)
                            #if isinstance(z, ReLU):
                            #    self._out_degree[z._name] -= 1
                        else:
                            pos[z._name] = len(pre_input)
                            pre_input.append((w3,b3,z))
                        k = pos[z._name]
                    
                    (w3,b3,z) = pre_input[k] 
                    pre_input[k] = (w3,b1+b3,z) ## the bias b1 can only be added to one input
                else:
                    if y._name in pos:
                        j = pos[y._name]
                        input3 = pre_input[j]
                        w1 = w1+input3[0]
                        b1 = b1+input3[1]
                        pre_input[j] = (w1,b1,y)
                        #if isinstance(y, ReLU):
                        #    self._out_degree[y._name] -= 1
                    else:
                        pos[y._name] = len(pre_input)
                        pre_input.append(input1)
                    
                
            x._input = pre_input
             
            
            
    
    def get_shift_matrix(self, h, w, shift):
        values = cp.ones(w)
        cols = cp.arange(w)
        rows = cols + shift
                
        return csr_matrix((values, (rows, cols)), shape=(h, w), dtype = np.float32)
                
    
    
    def backsubstite(self, x):        
        if isinstance(x, ReLU) or isinstance(x, Output):
            y = x._input[0]
            self.back(y)
            if len(y._input)>1:
                b = 0
                w = []
                shift = []
                n = 0
                for input in y._input:
                    b += input[1]
                    w.append(input[0])
                    shift.append(n)
                    n += input[0].shape[1]                
                    
                w = hstack(w).tocsr()
                size = w.shape[1]
                
                z = SumLinear()
                z.set_name("node"+str(self._node_id))
                self._node_id += 1

                x._input[0] = z
                
                r = ReLU()
                r.set_name("node"+str(self._node_id))
                r._width = n
                self._node_id += 1
                
                z.append_input(w, b, r)
                self._new_layers.insert(0,z)
                self._new_layers.insert(0,r)
                
                rz = SumLinear()
                rz.set_name("node"+str(self._node_id))
                self._node_id += 1
                r.append_input(rz)
                                
                r._activate = cp.asarray([], dtype = np.int32)
                r._unstable = cp.asarray([], dtype = np.int32)

                for i, input in enumerate(y._input):
                    w = self.get_shift_matrix(size, input[0].shape[1], shift[i])
                    b = cp.zeros(size, dtype = np.float32)
                    
                    i2 = input[2]
                    if isinstance(i2, ReLU) and self._out_degree[i2._name]<2:
                        v = i2._input[0]
                        r._activate = cp.concatenate((r._activate, i2._activate + shift[i]))
                        r._unstable = cp.concatenate((r._unstable, i2._unstable + shift[i]))
                        #i2._temp =True
                    elif isinstance(i2, ReLU):
                        r._activate = cp.concatenate((r._activate, i2._activate + shift[i]))
                        r._activate = cp.concatenate((r._activate, i2._unstable + shift[i]))
                        v = i2
                    else:
                        r._activate = cp.concatenate((r._activate, i2._activate + shift[i]))
                        r._unstable = cp.concatenate((r._unstable, i2._unstable + shift[i]))                    
                        v = i2
                        
                    rz.append_input(w, b, v)    
                    self._out_degree[i2._name] -= 1
                    
                return r
            else:
                if not isinstance(y, SumLinear):
                    if not isinstance(y, Input):
                        self._new_layers.insert(0,y)
                    return y
                
                self._new_layers.insert(0,y)
                z = y._input[0][2]
                if not isinstance(z, Input):
                    self._new_layers.insert(0,z)
                return z
                
        return None     
    
    
    
    def back_all_linear_nodes(self):
        self._new_layers = []
        self._out_degree = {}
        
        for layer in self._layers:
            self._out_degree[layer._name] = 0
        
        for layer in self._layers:
            if isinstance(layer, ReLU):
                x = layer._input[0]
                self.back(x)
                self._new_layers.append(x)
                self._new_layers.append(layer)

        x = self._output._input[0]
        if isinstance(x, SumLinear):
            self.back(x)
            self._new_layers.append(x)
            

        self._layers = self._new_layers
        self._name_to_layer = {}
        self._name_to_layer[self._input[0]._name] = self._input[0]
        self._name_to_layer[self._output._name] = self._output
        for layer in self._layers:
            self._name_to_layer[layer._name] = layer
    
    
    def compute_out_degree(self, x, old):
        if x._name not in old:
            self._out_degree[x._name] += 1
            old[x._name] = True
            if isinstance(x, SumLinear):
                for input in x._input:
                    #print(input[2]._name, type(input[2]), type(self._name_to_bound_layer[input[2]._name]))
                    self.compute_out_degree(input[2], old)
    
    def simply(self):
        self.clear_redundant_relu()

        #self.print()
        #self.back_all_linear_nodes()
        
        self._out_degree = {}
        self._new_layers = []        
       
        for layer in self._layers:
            self._out_degree[layer._name] = 0   
            #layer._temp = False
        
        for layer in self._input:
            self._out_degree[layer._name] = 0        
            #layer._temp = False
        
        self.compute_out_degree(self._output._input[0], {})
        for layer in self._layers:
            if isinstance(layer, ReLU):
                self.compute_out_degree(layer._input[0], {})
        
        '''
        self._out_degree[self._output._input[0]._name] += 1
        for layer in self._layers:
            if isinstance(layer, SumLinear):
                for input in layer._input:
                    self._out_degree[input[2]._name] +=1
            else:
                self._out_degree[layer._input[0]._name] +=1
        '''
        #for layer in self._layers:
        #    if isinstance(layer, ReLU):
        #        print("------------------------", self._out_degree[layer._name])
        
        '''
        i = 0
        self._new_layers.append(self._layers[0])
        self._new_layers.append(self._layers[1])
        for layer in self._layers:
            if isinstance(layer, ReLU):
                if i>0:
                    self.back(layer._input[0])
                    self._new_layers.append(layer._input[0])
                    self._new_layers.append(layer)
                else:
                    self._new_layers.append(layer)

                i += 1
            elif i<0:
                self._new_layers.append(layer)
            
        if isinstance(self._output._input[0], SumLinear):
            self._new_layers.append(self._output._input[0])
        '''
                
        r = self._output
        while r is not None and not isinstance(r, Input):
            r = self.backsubstite(r)
        
        self._layers = self._new_layers
        self._name_to_layer = {}
        self._name_to_layer[self._input[0]._name] = self._input[0]
        self._name_to_layer[self._output._name] = self._output
        for layer in self._layers:
            self._name_to_layer[layer._name] = layer
            
        return         
        
        
    def clear_redundant_relu(self):
        layers = []
        # to do        

    
    def preprocess_deactivate_first(self, layer):
        if isinstance(layer, ReLU):
            if layer._name != self._final_name:
                an = layer._activate.shape[0]
                un = layer._unstable.shape[0]
                ind = cp.concatenate((layer._unstable, layer._activate))
                x = layer._input[0]
                if isinstance(x, SumLinear):
                    if an == 0 and un == 0:
                         ind = cp.asarray([0], dtype = np.int32)
                    for i in range(len(x._input)):
                        xw, xb, xl = x._input[i]
                        xw = xw[ind, :]
                        xb = xb[ind]
                        x._input[i] = (xw, xb, xl)
        elif isinstance(layer, SumLinear):
            x = layer
            for i in range(len(x._input)):
                xw, xb, y = x._input[i]
                if isinstance(y, ReLU):
                    an = y._activate.shape[0]
                    un = y._unstable.shape[0]
                    ind = cp.concatenate((y._unstable, y._activate))
                    if an == 0 and un == 0:
                         ind = cp.asarray([0], dtype = np.int32)
                    xw = xw[:,ind]
                    x._input[i] = (xw, xb, y)

    def preprocess_deactivate_second(self):
        for layer in self._layers:
            layer._output = []
        
        self._input[0]._output = []
        
        
        for layer in self._layers:
            if isinstance(layer, ReLU):
                y = layer._input[0]
                y._output.append((layer, 0))
            else:
                for i, input in enumerate(layer._input):
                    y = input[2]
                    y._output.append((layer, i))
        
        for pos, layer in enumerate(self._layers):
            if isinstance(layer, ReLU) and len(layer._output)>0:
                redundant = False
                if layer._activate.shape[0]==0 and layer._unstable.shape[0]==0:
                    layer._activate = cp.asarray([0], dtype = np.int32)
                    redundant = True
                an = layer._activate.shape[0]
                un = layer._unstable.shape[0]
                ind = cp.concatenate((layer._unstable, layer._activate))
                layer._activate = cp.arange(an, dtype=np.int32) + un
                layer._unstable = cp.arange(un, dtype=np.int32)
                
                layer._width = an+un

                if layer._unstable.shape[0]==0:
                    x = SumLinear()
                    x.set_name(layer._name)
                    self._name_to_layer[layer._name] = x
                    self._layers[pos] = x

                    if redundant:
                        x.set_shape(torch.zeros(1).shape)
                        weights, bias = csr_matrix((x._width, x._width), dtype = np.float32), cp.zeros(x._width, dtype = np.float32)           
                    else:
                        x.set_shape(torch.zeros(layer._activate.shape[0]).shape)
                        weights, bias = identity(x._width, dtype = np.float32, format="csr"), cp.zeros(x._width, dtype = np.float32)

                    x.append_input(weights, bias, layer._input[0])
                    x._output = layer._output
                    for (y, i) in layer._output:
                        yw, yb, yl = y._input[i]
                        y._input[i] = (yw, yb, x)           
        

    
    def reduce_deactivate(self):
        for layer in self._layers:
            layer._output = []
        
        self._input[0]._output = []
        
        
        for layer in self._layers:
            if isinstance(layer, ReLU):
                y = layer._input[0]
                y._output.append((layer, 0))
            else:
                for i, input in enumerate(layer._input):
                    y = input[2]
                    y._output.append((layer, i))
                
        
        for pos, layer in enumerate(self._layers):
            if isinstance(layer, ReLU) and len(layer._output)>0:
                redundant = False
                if layer._activate.shape[0]==0 and layer._unstable.shape[0]==0:
                    layer._activate = cp.asarray([0], dtype = np.int32)
                    redundant = True
                an = layer._activate.shape[0]
                un = layer._unstable.shape[0]
                ind = cp.concatenate((layer._unstable, layer._activate))
                layer._activate = cp.arange(an, dtype=np.int32) + un
                layer._unstable = cp.arange(un, dtype=np.int32)
                
                layer._width = an+un

                x = layer._input[0]
                for i in range(len(x._input)):
                    xw, xb, xl = x._input[i]
                    xw = xw[ind, :]
                    xb = xb[ind]
                    x._input[i] = (xw, xb, xl)
                    
                for (y, i) in layer._output:
                    yw, yb, yl = y._input[i]
                    yw = yw[:, ind]
                    y._input[i] = (yw, yb, yl)
            
            
                if layer._unstable.shape[0]==0:
                    x = SumLinear()
                    x.set_name(layer._name)
                    self._name_to_layer[layer._name] = x
                    self._layers[pos] = x

                    if redundant:
                        x.set_shape(torch.zeros(1).shape)
                        weights, bias = csr_matrix((x._width, x._width), dtype = np.float32), cp.zeros(x._width, dtype = np.float32)           
                    else:
                        x.set_shape(torch.zeros(layer._activate.shape[0]).shape)
                        weights, bias = identity(x._width, dtype = np.float32, format="csr"), cp.zeros(x._width, dtype = np.float32)

                    x.append_input(weights, bias, layer._input[0])
                    x._output = layer._output
                    for (y, i) in layer._output:
                        yw, yb, yl = y._input[i]
                        y._input[i] = (yw, yb, x)           

            
        
    def reduce_activate(self):
        #Note that after reduce deactivate, the indices of unstable neurons are smaller than that of activate neurons
        self.reduce_deactivate()
        size = {}
        size[self._input[0]._name] = self._input[0]._width        
        
        for layer in self._layers:
            layer._output = []
        
        self._input[0]._output = []
                    
        for layer in self._layers:
            if isinstance(layer, ReLU):
                layer._input[0]._output.append(layer)
                size[layer._name] = layer._width
            else:
                layer._input[0][2]._output.append(layer)        
        
        size[self._input[0]._name] = self._input[0]._width
        size[self._output._name] = self._output._width    
        
        #print(size)
        while size:
            #print(size)
            name = min(size, key=size.get)
            layer = self._name_to_layer[name]
            size.pop(name)
            width = layer._width
            
            if isinstance(layer, Output):
                z = layer._input[0]
                if isinstance(z, SumLinear):
                    y = z._input[0][2]
                    if y._activate.shape[0]>width:
                        x = y._input[0]
                        self.update_activate_right(x, y, z, width)
                        size[y._name] = y._width
            elif isinstance(layer,Input):
                x = layer._output[0]
                if isinstance(x, SumLinear):
                    y = x._output[0]
                    if y._activate.shape[0]>width:
                        if len(y._output)>0:
                            z = y._output[0]
                            self.update_activate_left(x, y, z, width)
                            size[y._name] = y._width            
            
            else:
                #right
                
                z = layer._input[0]
                
                if isinstance(z, SumLinear):
                    y = z._input[0][2]
                    if isinstance(y, ReLU) and y._activate.shape[0]>width:
                        x = y._input[0]
                        if isinstance(x, SumLinear):
                            self.update_activate_right(x, y, z, width)
                            size[y._name] = y._width
                
                #left
                if len(layer._output)>0:
                    x = layer._output[0]
                    if isinstance(x, SumLinear) and len(x._output)>0:
                            y = x._output[0]
                            if y._activate.shape[0]>width and len(y._output)>0:
                                    z = y._output[0]                                    
                                    self.update_activate_left(x, y, z, width)
                                    size[y._name] = y._width      
                
                
        for layer in self._layers:
            if isinstance(layer, SumLinear):
                for i, input in enumerate(layer._input):
                    w, b, x = input
                    if not isinstance(w, cp.ndarray):
                        w = w.toarray()
                    layer._input[i] = (w,b,x)    
        
        
    def update_activate_right(self, x, y, z, width):
        activate = y._activate
        unstable = y._unstable
        uwidth = unstable.shape[0]
        awidth = activate.shape[0]
        y._activate = cp.arange(width, dtype = np.int32) + uwidth
        
        w1 = x._input[0][0]
        b1 = x._input[0][1]
        
        w2 = z._input[0][0]
        b2 = z._input[0][1]
        
        w1u = w1[unstable,:]
        b1u = b1[unstable]
        
        w1a = w1[activate,:]
        b1a = b1[activate]
        
        w2u = w2[:, unstable]        
        w2a = w2[:, activate]
        
        
        w = w2a@w1a
        
        b2 += w2a@b1a        
        w2a = identity(width, dtype = w2.dtype, format="csr")
        if isinstance(w2u, cp.ndarray):
            w2u = csr_matrix(w2u)
        w2 = hstack([w2u, w2a]).tocsr()
        
        y._width = width+uwidth

        if isinstance(w1u, cp.ndarray) and isinstance(w, cp.ndarray):
            w1 = cp.vstack([w1u, w])
        else:
            if isinstance(w, cp.ndarray):
                w = csr_matrix(w)            
            if isinstance(w1u, cp.ndarray):
                w1u = csr_matrix(w1u)            
            w1 = vstack([w1u, w]).tocsr()
        
        
        b1[activate] = 0
        b1 = b1[:y._width]        
        
        x._input[0] = (w1,b1,x._input[0][2])
        z._input[0] = (w2, b2, y)
        

        
    def update_activate_left(self, x, y, z, width):
        activate = y._activate
        unstable = y._unstable
        uwidth = unstable.shape[0]
        awidth = activate.shape[0]
        y._activate = cp.arange(width, dtype = np.int32) + uwidth
        
        w1 = x._input[0][0]
        b1 = x._input[0][1]
        
        w2 = z._input[0][0]
        b2 = z._input[0][1]
        
        w1u = w1[unstable,:]
        b1u = b1[unstable]
        
        w1a = w1[activate,:]
        b1a = b1[activate]
        
        w2u = w2[:, unstable]        
        w2a = w2[:, activate]
        
        
        w = w2a@w1a
        
        b2 += w2a@b1a      
        y._width = width+uwidth
        b1[activate] = 0
        b1 = b1[:y._width]  
        
        w1a = identity(width, dtype = w1.dtype, format="csr")
        if isinstance(w1u, cp.ndarray):
            w1u = csr_matrix(w1u)
        w1 = vstack([w1u, w1a]).tocsr()
                
        if isinstance(w2u, cp.ndarray) and isinstance(w, cp.ndarray):
            w2 = cp.hstack([w2u, w])        
        else:
            if isinstance(w, cp.ndarray):
                w = csr_matrix(w)            
            if isinstance(w2u, cp.ndarray):
                w2u = csr_matrix(w2u)
                
            w2 = hstack([w2u, w]).tocsr()
      
        
        x._input[0] = (w1,b1,x._input[0][2])
        z._input[0] = (w2, b2, y)
        
        

    def repair_bias(self, lb, ub):
        for layer in self._layers:
            layer._output = []
        self._input[0]._output = []
                    
        for layer in self._layers:
            if isinstance(layer, ReLU):
                layer._input[0]._output.append(layer)
            else:
                layer._input[0][2]._output.append(layer)        
        
        lb = cp.asarray(lb.view(-1), np.float32).ravel()
        ub = cp.asarray(ub.view(-1), np.float32).ravel()
        
        self._input[0]._temp = (lb, ub)
 
        for layer in self._layers:           
            layer.forward_bound()
            lb, ub = layer._temp
            if isinstance(layer, SumLinear):
                if len(layer._output)>0:
                    y = layer._output[0]
                    if isinstance(y, ReLU) and len(y._output)>0:
                        z = y._output[0]
                        w, b, x = layer._input[0]
                        ind = y._activate[lb[y._activate]<0]
                                                 
                        add = lb[ind] -1
                        b[ind] -= add
                        layer._input[0] = (w, b, x)

                        lb[ind] -= add
                        ub[ind] -= add
                        layer._temp = (lb, ub)

                        w, b, v = z._input[0]
                        b += w[:, ind] @ add
                        z._input[0] = (w, b, v)

        
    def reduction(self, boundnet, bound_input):
        '''
        start = time.time()
        #lb, ub = boundnet.compute_bounds(x=(bound_input,), method='CROWN', share_slopes=True)         
        lb, ub, aux_reference_bounds =  boundnet.init_slope(
            (bound_input,), share_slopes=True, c=None, method = 'backward')
        end = time.time()
        self._bound_propagation_time = end-start
        #print("initial CROWN execution time:", end-start)
        #print("initial CROWN bounds:", lb, ub, lb.dtype)
        
        #lb, ub = boundnet.compute_bounds(x=(bound_input,), method='alpha-CROWN')         

        
        self._input[0].compute_unstable_neurons(bound_input.ptb.x_L,bound_input.ptb.x_U)
        
        for layer in boundnet.perturbed_optimizable_activations:
            l = layer.inputs[0].lower.detach()
            u = layer.inputs[0].upper.detach()    
            self._name_to_layer[layer.name].compute_unstable_neurons(l, u)
        '''
        
        #print(self._input[0]._width)
        #self.print("middle")            
        #self.reduce_deactivate()            
        self.simply()
        #self.print("mid")
        self.reduce_activate()
        self.repair_bias(bound_input.ptb.x_L, bound_input.ptb.x_U)
        return   
        
        