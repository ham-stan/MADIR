from torch.utils.data import Dataset, DataLoader
import numpy as np


class TrainsetImg(Dataset):
    def __init__(self, txt_path='data/train_data/train_img.txt', reshape=True, grid=4,
                 transform=None, target_transform=None):
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
            img = img.reshape(self.grid, w//self.grid, self.grid, h//self.grid)
            patches = np.zeros([self.grid*self.grid, w//self.grid, h//self.grid])
            i = 0
            for j in range(self.grid):
                for k in range(self.grid):
                    patches[i, :, :] = img[j, :, k, :]
                    i += 1
            img = patches
        return img  # c, w, h

    def __len__(self):
        return len(self.imgs)


# data = TrainsetImg()
# print(data.__len__())
# dalo = DataLoader(data, batch_size=8, shuffle=True)
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
