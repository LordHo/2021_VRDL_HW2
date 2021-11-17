import os
import numpy as np
import h5py
from torch.utils.data import Dataset
from PIL import Image


class TrainLabel(Dataset):
    def __init__(self, mat_path, image_dir):
        self.file = h5py.File(mat_path, 'r')
        self.names = self.file['digitStruct']['name']
        self.bboxes = self.file['digitStruct']['bbox']

        self.image_dir = image_dir

    def __getitem__(self, index):
        image_name = self.__getName(index)
        image = self.__getImage(index)
        bboxes = self.__getBbox(index)
        return image_name, bboxes, image

    def __getName(self, index):
        name = ''.join([chr(v[0])
                       for v in self.file[(self.names[index][0])]])
        return name

    def __bboxHelper(self, attr):
        if len(attr) > 1:
            attr = [self.file[attr[j].item()][0][0] for j in range(len(attr))]
        else:
            attr = [attr[0][0]]
        return attr

    def __getBbox(self, index):
        bbox = {}
        bb = self.bboxes[index].item()
        bbox['height'] = self.__bboxHelper(self.file[bb]['height'])
        bbox['label'] = self.__bboxHelper(self.file[bb]['label'])
        bbox['left'] = self.__bboxHelper(self.file[bb]['left'])
        bbox['top'] = self.__bboxHelper(self.file[bb]['top'])
        bbox['width'] = self.__bboxHelper(self.file[bb]['width'])
        return bbox

    def __getImage(self, index):
        image_name = self.__getName(index)
        image = Image.open(os.path.join(self.image_dir, image_name))
        return image
