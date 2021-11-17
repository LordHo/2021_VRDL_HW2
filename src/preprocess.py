import h5py
from torch.utils.data import Dataset


class TrainLabel(Dataset):
    def __init__(self, mat_path):
        self.file = h5py.File(mat_path, 'r')
        self.names = self.file['digitStruct']['name']
        self.bboxes = self.file['digitStruct']['bbox']

    def __getitem__(self, index):
        image_name = self.getName(index)
        bboxes = self.__getBbox(index)
        return image_name, bboxes

    def getName(self, index):
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
