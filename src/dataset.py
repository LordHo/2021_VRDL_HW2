import os
import glob
import torch
import numpy as np
import glob
from PIL import Image
from torch.utils.data.dataset import Dataset
from torchvision import transforms
from preprocess import TrainLabel


class TrainDataset(TrainLabel):
    def __init__(self, mat_path, image_dir):
        super().__init__(mat_path)
        self.image_dir = image_dir

        self.ToTensor = transforms.Compose([
            transforms.ToTensor()
        ])

    def __len__(self):
        # return len(os.listdir(self.image_dir))
        return len(glob.glob(os.path.join(self.image_dir, '*.png')))

    def __getitem__(self, index):
        image = self.__getImage(index)
        # print(type(image))
        target = self.__get_target(index)
        return self.ToTensor(image), target

    def __getImage(self, index):
        image_name = super().getName(index)
        image = Image.open(os.path.join(self.image_dir, image_name))
        return image

    def __get_box(self, bboxes, index):
        x, y = bboxes['left'][index], bboxes['top'][index]
        height, width = bboxes['height'][index], bboxes['width'][index]
        return [x, y, x+width, y+height]

    def __get_target(self, index):
        target = {}
        image_name, bboxes = super().__getitem__(index)
        labe_num = len(bboxes['label'])

        boxes = []
        labels = []
        for index in range(labe_num):
            target = {}
            boxes.append(self.__get_box(bboxes, index))
            labels.append(bboxes['label'][index])

        target["boxes"] = torch.as_tensor(boxes, dtype=torch.float32)
        target["labels"] = torch.as_tensor(labels, dtype=torch.int64)

        return target


class EvalDataset(Dataset):
    def __init__(self, image_dir):
        self.image_dir = image_dir
        self.image_names = sorted(
            glob.glob(os.path.join(self.image_dir, '*.png')), key=lambda x: int(os.path.basename(x).split('.')[0]))

        self.ToTensor = transforms.Compose([
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(sorted(glob.glob(os.path.join(self.image_dir, '*.png'))))

    def __getitem__(self, index):
        image = self.__getImage(index)
        return self.ToTensor(image), self.__getName(index)

    def __getName(self, index):
        return os.path.basename(self.image_names[index])

    def __getImage(self, index):
        image_name = self.__getName(index)
        image = Image.open(os.path.join(self.image_dir, image_name))
        return image
