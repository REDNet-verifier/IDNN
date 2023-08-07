REDNet pre-processing
========

REDNet is a pre-processing technique that uses the state-of-the-art bound propagator to detect stable ReLU neurons and then delete them in a way that preserves network manner. It is instantiated on CROWN propagator in alpha-beta-CROWN ([alpha-beta-CROWN](https://github.com/Verified-Intelligence/alpha-beta-CROWN)). 


Requirements 
------------
Installation of alpha-beta-CROWN


Installation
------------
Install and setup alpha-beta-CROWN as follows:
```
git clone https://github.com/Verified-Intelligence/alpha-beta-CROWN.git
cd alpha-beta-CROWN
conda deactivate; conda env remove --name alpha-beta-crown
conda env create -f complete_verifier/environment.yml --name alpha-beta-crown
conda activate alpha-beta-crown
```

Clone the IDNN repository via git as follows:
```
git clone https://github.com/REDNet-verifier/IDNN
```

Copy the files and folder in IDNN under path alpha-beta-crown/complete_verifier:
```
cp -r IDNN/exp_setups complete_verifier/exp_setups/
cp IDNN/IDNN_A.py complete_verifier/IDNN_A.py
cp IDNN/IDNN_B.py complete_verifier/IDNN_B.py
cp IDNN/run_IDNN.py complete_verifier/run_IDNN.py
cp IDNN/export_reduced_onnx.py complete_verifier/export_reduced_onnx.py
```

Usage
-------------
To run the benchmarks, please follow the command in the respective .yaml; download the VNNCOMP2021/2022/ERAN networks accordingly; and update the network path in .yaml configuration file.

The configuration files for all the networks and tested properties can be found under folder **exp_setups**. The networks' names are abbreviated in the paper due to the limited space, we hereby provide the table to match the networks' names in the paper and in this repo.

| Network name in the paper  |   Network name in this repo  |
| :------------------------- | :--------------------------- |
| M\_256x6   | mnist\_256x6 |
| M\_ConvMed   | mnist\_convmed |
| M\_ConvBig   | mnist\_convbig |
| M\_SkipNet   | mnist\_skipnet |
| C\_8\_255Simp   | cifar2020\_8\_255 |
| C\_WideKW   | cifar\_wide |
| C\_ConvBig   | cifar\_convbig |
| C\_Resnet4b  | resnet4b |
| C\_ResnetA   | resnetA |
| C\_ResnetB   | resnetB |
| C\_100\_Med   | cifar100\_med |
| C\_100\_Large  | cifar100\_large |

Example of running network C\_8\_255Simp:
```
conda activate alpha-beta-crown
cd complete_verifier
python abcrown.py --config exp_setups/cifar2020_8_255.yaml # running original abcrown
python run_IDNN.py --config exp_setups/cifar2020_8_255.yaml # running abcrown with reduced network
```

To export the reduced network in onnx networks:
```
python export_reduced_onnx.py --config exp_setups/cifar2020_8_255.yaml
```