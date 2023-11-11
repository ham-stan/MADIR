from torch.utils.data import Dataset
import numpy as np


class TrainsetImg(Dataset):
    def __init__(self, txt_path='data/train_data/train_img.txt', reshape=True, grid=4, transform=None, target_transform=None):
        fh = open(txt_path, 'r')
        imgs = []
        for line in fh:
            line = line.rstrip()
            imgs.append(line)
            self.imgs = imgs
            self.reshape = reshape
            self.grid = grid
            self.transform = transform
            self.target_transform = target_transform

    def __getitem__(self, index):
        fn = self.imgs[index]
        img = np.fromfile(fn, dtype='float32')
        w, h, c = 512, 512, 1
        img = img.reshape(w, h, c)
        if self.transform is not None:
            img = self.transform(img)
        if self.reshape:
            img = img.reshape(w//self.grid, h//self.grid, c*self.grid*self.grid)

        return img

    def __len__(self):
        return len(self.imgs)
