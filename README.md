# 2021_VRDL_HW2

## Github Link

2021_VRDL_HW2: https://github.com/LordHo/2021_VRDL_HW2.git

## Hardware

Intel(R) Core(TM) i7-9700F CPU @ 3.00GHz  
NVIDIA GeForce RTX 2080 Ti

## Environment

Window 10  
Pytorch 1.8.0  
Torchvision 0.9.0  
CUDA 11.1

To install correspond cuda and torch version:

```cmd
# CUDA 11.1
conda install pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 cudatoolkit=11.1 -c pytorch -c conda-forge
```

Other version Link: https://pytorch.org/get-started/previous-versions/

## Dataset

The Street View House Numbers: http://ufldl.stanford.edu/housenumbers/  
Class Competition: https://competitions.codalab.org/competitions/35888?secret_key=7e3231e6-358b-4f06-a528-0e3c8f9e328e

## Requirements

To install requirements:

```cmd
pip install -r requirements.txt
```

## Introduction

I use yolov5 [1] as my final result model. Besides, I also try RetinaNet [2] and Faster R-CNN [3] in torchvision model zoo. Yolov5 can easy to get started by the command provided by yolov5 office github. However, if you want to complete whole flow from data preprocessing, dataset and data loader construct, training model until final step- predicting testing data, the torchvision is great choice for you.

## Training

To train model with yolov5:

```cmd
python train.py --img 512 --batch 16 --epochs 60 --data data\yolov5\data.yaml --weights yolov5m.pt --workers 4
```

Can modify the image size, batch size and epochs that you want.

## Testing

To test model with testing data:

```cmd
python detect.py --weights runs/train/exp47/weights/best.pt --conf 0.2 --source data/test --save-txt --save-conf
```

## Model Performance

|       Model        | baseline | yolov5 | RetinaNet | Faster R-CNN |
| :----------------: | :------: | :----: | :-------: | :----------: |
|     Test (mAP)     |  0.3919  | 0.4034 |  0.2533   |    0.3333    |
|  Inference (sec)   |  0.2989  | 0.0365 |     -     |      -       |
| Training time (hr) |    -     |   1    |    10     |      10      |

## Best Model Weight

Yolov5: https://drive.google.com/file/d/1mxwvPyvY1WNXfovtYSQeyrGeLKgdLF34/view?usp=sharing  

## Benchmark Result
You can see the all testing image results and time costs on `inference.ipynb`.

## Reference

[1] Yolov5: https://github.com/ultralytics/yolov5  
[2] RetinaNet: https://arxiv.org/abs/1708.02002  
[3] Faster R-CNN: https://arxiv.org/abs/1506.01497
[4] Yolov4: https://arxiv.org/abs/2004.10934