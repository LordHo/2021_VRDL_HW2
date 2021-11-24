# 2021_VRDL_HW2

## Github Link

2021_VRDL_HW2: https://github.com/LordHo/2021_VRDL_HW2.git

## Environment

Window 10  
NVIDIA GeForce RTX 2080

Pytorch 1.8.0  
Torchvision 0.9.0  
CUDA 11.1

To install correspond cuda and torch version:

```cmd
# CUDA 11.1
conda install pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 cudatoolkit=11.1 -c pytorch -c conda-forge
```

Other version Link: https://pytorch.org/get-started/previous-versions/

## Requirements

To install requirements:

```cmd
pip install -r requirements.txt
```

## Introduction

I use yolov5 [1] as my final result model. Besides, I also try RetinaNet [2] and Faster R-CNN [3], and the performance of three models are in following table.
|      Model       | baseline | yolov5 | RetinaNet | Faster R-CNN |
| :--------------: | -------- | :----: | :-------: | :----------: |
|       Test       | 0.3919   | 0.4034 |  0.2533   |   0.333336   |
| Interfence (sec) | 0.2989   | 0.0365 |     -     |      -       |

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

## Best Model Weight

Yolov5: https://drive.google.com/file/d/1mxwvPyvY1WNXfovtYSQeyrGeLKgdLF34/view?usp=sharing  

## Reference

[1] Yolov5: https://github.com/ultralytics/yolov5  
[2] RetinaNet: https://arxiv.org/abs/1708.02002  
[3] Faster R-CNN: https://proceedings.neurips.cc/paper/2015/file/14bfa6bb14875e45bba028a21ed38046-Paper.pdf