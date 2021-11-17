import os
from PIL import Image
from preprocess import TrainLabel


class TrainDataset(TrainLabel):
    def __init__(self, mat_path, image_dir):
        super().__init__(mat_path)
        self.image_dir = image_dir

    def __getitem__(self, index):
        image = self.__getImage(index)
        return super().__getitem__(index), image

    def __getImage(self, index):
        image_name = super().getName(index)
        image = Image.open(os.path.join(self.image_dir, image_name))
        return image
