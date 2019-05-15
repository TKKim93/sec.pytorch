# sec.pytorch
This repository is currently being prepared.

This is the pytorch implementation of the ECCV 2016 paper 'Seed, Expand and Constrain: Three Principles for Weakly-Supervised Image Segmentation' ([paper](https://arxiv.org/abs/1603.06098)).

### Acknowledgment
The implementation heavily refers to the python implementation of DeepLab-ResNet ([isht7/pytorch-deeplab-resnet](https://github.com/isht7/pytorch-deeplab-resnet)) and the public code of Seed, Expand, Constrain ([https://github.com/kolesman/SEC](https://github.com/kolesman/SEC)).

### Dependency preparation
1. Python packages:
```bash
      $ pip install -r python-dependencies.txt
      $ conda install -c conda-forge opencv
      $ conda install -c conda-forge tensorboardx
```
If you have an issue with numpy.core.multiarray, remove the currently installed numpy from your virtual environment and re-install with the follwing line:
```bash
      $ pip install -U numpy
```
2. Build the Fully connected CRF wrapper:

Install the Eigen3 package and link the installed custum Eigen3 folder to '/usr/local/include/Eigen'. Then
```bash
      $ pip install CRF/
```
3. Install PyTorch.
```bash
      $ conda install pytorch=0.4.1 torchvision cuda80 -c pytorch
```
### Data preparation
1. Prepare the [initial vgg16 model](https://drive.google.com/open?id=1oRPzan6-Zy7VVcopesRX2s4VxebOPb2_) pretrained on ImageNet.
```bash
      $ mkdir vgg16_20M
```

2. Prepare localization cues.
```bash
      $ cd localization_cues
      $ gzip -kd localization_cues/localization_cues.pickle.gz
```

3. Prepare dataset (e.g., PASCAL VOC 2012) and update the directory in train.py.

https://github.com/TKKim93/sec.pytorch/blob/6094f3a55a755f6159d01917e4a7d49bf389d891/train.py#L25
https://github.com/TKKim93/sec.pytorch/blob/6094f3a55a755f6159d01917e4a7d49bf389d891/train.py#L26
https://github.com/TKKim93/sec.pytorch/blob/6094f3a55a755f6159d01917e4a7d49bf389d891/train.py#L27
