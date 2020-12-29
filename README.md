# VRDL-Final Project
NCTU Selected Topics in Visual Recognition using Deep Learning Final Project.

## Global Wheat Detection
The topic is from [Kaggle](https://www.kaggle.com/c/global-wheat-detection/overview). It is about detecting wheat from images.

## Hardware
The following specs were used to create the original solution.
- Ubuntu 18.04 LTS
- Intel(R) Core(TM) i7-6700 CPU @ 3.40 GHz
- NVIDIA GeForce GTX TITAN X

## Installation
1. Using Anaconda is strongly recommended. {envs_name} is the new environment name which you should assign.
    ```shell
    conda create -n {envs_name} python=3.7
    conda activate {envs_name}
    ```
2. Install PyTorch and torchvision following the [official instructions](https://pytorch.org/), e.g.,
    ```shell
    conda install pytorch==1.5.0 torchvision==0.6.0 cudatoolkit=10.1 -c pytorch
    ```
   Note: Make sure that your compilation CUDA version and runtime CUDA version match.
   You can check the supported CUDA version for precompiled packages on the [PyTorch website](https://pytorch.org/).
3. Install requirements.
   ```shell
   pip install -r requirements.txt
   ```

## Dataset Preparation
You can download the dataset [here](https://www.kaggle.com/c/global-wheat-detection/data).

You can also download the pretrained models from [EfficientDet Pytorch](https://www.kaggle.com/mathurinache/efficientdet).

After downloading the data and pretrained models, the data directory is structured as:
```text
+- input
    +- efficientdet
        +- efficientdet_d0-d92fd44f.pth
        +- efficientdet_d1-4c7ebaf2.pth
        ...
    +- global-wheat-detection
        +- train
            +- 0a3cb453f.jpg
            +- 0a3ff84a7.jpg
            ...
        +- test
            +- 2fd875eaa.jpg
            +- 51b3e36ab.jpg
            ...
        train.csv
        sample_submission.csv
+- EfficientDetTool
+- omegaconf
+- weightedboxesfusion
train.py
make_submission.py
```

## Training
You can train the model by following:
```shell
python3 train.py
```

## Testing
You can test the model and make a csv submission file by following:
```shell
python3 make_submission.py
```
## Thanks For
