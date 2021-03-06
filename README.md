# Deformation-Monitoring-Dev
The source code of Paper: **Deep learning for localized rapid deformation detection and InSAR phase unwrapping**

Authors: Zhipeng Wu, *Student Member, IEEE*, Teng Wang, Yingjie Wang, Robert Wang, *Senior Member, IEEE*, Daqing Ge



### Introduction

This is the source code for training and testing PUNet/DDNet, implemented in the PyTorch framework version 1.8.0 based on Python 3.6.

**For code to generate the training dataset, see [InterferogramSimulator](https://github.com/Wu-Patrick/InterferogramSimulator).**


### Installation

Assume you have Python 3.6 installed.

1. Clone the repo:

   ~~~shell
   git clone https://github.com/Wu-Patrick/Deformation-Monitoring-Dev.git
   cd Deformation-Monitoring-Dev
   ~~~

2. Install dependencies:

   ~~~shell
   pip install -r requirements.txt
   ~~~

### Training

1. Input arguments: (see full input arguments via `python train.py --help`):

~~~shell
usage: train.py [-h] [--model MODEL] [--dataRootDir DATAROOTDIR]
                [--dataset DATASET] [--input_size INPUT_SIZE]
                [--num_workers NUM_WORKERS] [--num_channels NUM_CHANNELS]
                [--max_epochs MAX_EPOCHS] [--random_mirror RANDOM_MIRROR]
                [--lr LR] [--batch_size BATCH_SIZE] [--optim {sgd,adam}]
                [--poly_exp POLY_EXP] [--cuda CUDA] [--gpus GPUS]
                [--resume RESUME] [--savedir SAVEDIR] [--logFile LOGFILE]
~~~

2. Run：

~~~shell
python train.py
~~~

### Testing
1. Input arguments: (see full input arguments via `python test.py --help`):

~~~shell
usage: test.py [-h] [--model MODEL] [--dataRootDir DATAROOTDIR]
               [--dataset DATASET] [--num_workers NUM_WORKERS]
               [--batch_size BATCH_SIZE] [--checkpoint CHECKPOINT]
               [--cuda CUDA] [--gpus GPUS]
~~~

2. Run：

~~~shell
python test.py
~~~

### Citation
If you use this code, please cite the following:
~~~BibTeX
@ARTICLE{9583229,
  author={Wu, Zhipeng and Wang, Teng and Wang, Yingjie and Wang, Robert and Ge, Daqing},
  journal={IEEE Transactions on Geoscience and Remote Sensing}, 
  title={Deep Learning for the Detection and Phase Unwrapping of Mining-Induced Deformation in Large-Scale Interferograms}, 
  year={2022},
  volume={60},
  number={},
  pages={1-18},
  doi={10.1109/TGRS.2021.3121907}}
~~~

### Acknowledgement

[Python](https://www.python.org/), [PyTorch](https://pytorch.org/), [xiaoyufenfei](https://github.com/xiaoyufenfei/Efficient-Segmentation-Networks)

### Statement

The code can only be used for personal academic research testing.

