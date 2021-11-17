import os
import numpy as np
from processTrainLabel import TrainLabel
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def load_mat(mat_path):
    image_dir = os.path.join('..', 'data', 'train')
    dataset = TrainLabel(mat_path, image_dir)
    count = 0
    for name, bbox, image in dataset:
        fig, ax = plt.subplots()
        ax.imshow(image)
        ax.set_title(
            ' '.join([name, ' '.join(np.array(bbox['label']).astype(str))]))
        for i in range(len(bbox['label'])):
            print(i)
            x, y = bbox['left'][i]+1, bbox['top'][i]+1
            height, width = bbox['height'][i], bbox['width'][i]
            print(x, y, height, width)
            rect = patches.Rectangle((x, y), width, height, linewidth=1,
                                     edgecolor='r', facecolor='none')
            ax.add_patch(rect)
        plt.show()
        if count > 5:
            break
        count += 1


def main():
    mat_path = os.path.join('..', 'data', 'train answer', 'digitStruct.mat')
    # mat_path = os.path.join('..', 'other resource', 'test_32x32.mat')
    load_mat(mat_path)


if __name__ == '__main__':
    main()
