# sec.pytorch
This repository is currently being prepared.

### Acknowledgment
The implementation is heavily refer to the python implementation of DeepLab-ResNet [isht7/pytorch-deeplab-resnet](https://github.com/isht7/pytorch-deeplab-resnet) and public code of Seed, Expand, Constrain [https://github.com/kolesman/SEC](https://github.com/kolesman/SEC)

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
