import random
from torch.utils.data import Dataset, DataLoader
import numpy as np


class TrainsetImg(Dataset):
    def __init__(self, txt_path,
                 transform=None, random_crop=False, random_flip=True):
        fh = open(txt_path, 'r')
        imgs = []
        for line in fh:
            line = line.rstrip()
            imgs.append(line)
            self.imgs = imgs

            self.transform = transform
            self.random_crop = random_crop
            self.random_flip = random_flip

    def __getitem__(self, index):
        fn = self.imgs[index]
        img = np.fromfile(fn, dtype='float32')
        w, h, c = 512, 512, 1
        img = img.reshape(w, h, c)
        if self.random_flip and random.random() < 0.5:
            img = img[:, ::-1]
        if self.transform is not None:
            img = self.transform(img)

        a = np.min(img)
        b = np.max(img)
        img = img - a / b - a
        return img

    def __len__(self):
        return len(self.imgs)


def load_data(data_dir='data/train_data/train_img.txt', batch_size=16, shuffle=True, ):
    data = TrainsetImg(txt_path=data_dir)
    loader = DataLoader(data, batch_size=batch_size, shuffle=shuffle, num_workers=1, drop_last=True)
    while True:
        yield from loader
# data = TrainsetImg()
# print(data.__len__())
# dalo = DataLoader(data, batch_size=16, shuffle=True)
# # batch = next(iter(dalo))
# # print(batch.size)
# for i, batch in enumerate(dalo):
#     print('-----------', i)
#     print(batch.shape)
#     print(batch[0].shape)
#     # print(batch[1].shape)
#     print('--------\n')
#     if i == 5:
#         break
