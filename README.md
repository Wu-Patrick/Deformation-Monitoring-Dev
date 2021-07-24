# Deformation-Monitoring-Dev
The source code of Paper: **Deep learning for localized rapid deformation detection and InSAR phase unwrapping**

Authors: Zhipeng Wu, *Student Member, IEEE*, Teng Wang, Yingjie Wang, Robert Wang, *Senior Member, IEEE*, Daqing Ge



### Introduction

This is the source code for training and testing PUNet/DDNet, which will be uploaded soon.

### Installation

The code was tested with Python 3.6.

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


### Acknowledgement

[Python](https://www.python.org/), [PyTorch](https://pytorch.org/), [xiaoyufenfei](https://github.com/xiaoyufenfei/Efficient-Segmentation-Networks)

### Statement

The code can only be used for personal academic research testing.

