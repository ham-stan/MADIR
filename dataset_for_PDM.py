import random
from torch.utils.data import Dataset, DataLoader
import numpy as np
from get_mu_delta import mu, delta
from PIL import Image
import math


class TrainsetImg(Dataset):
    def __init__(self, txt_path, mean=mu, std=delta, resolution=512,
                 random_flip=True, norm=True, crop=True, random_crop=False):
        fh = open(txt_path, 'r')
        imgs = []
        for line in fh:
            line = line.rstrip()
            imgs.append(line)
            self.imgs = imgs
        self.mean = mean
        self.std = std
        self.resolution = resolution
        self.random_flip = random_flip
        self.norm = norm
        self.crop = crop
        self.random_crop = random_crop

    def __getitem__(self, index):
        fn = self.imgs[index]
        img = np.fromfile(fn, dtype='float32')
        w, h, c = 512, 512, 1
        img = img.reshape(w, h, c)
        if self.crop:
            if self.random_crop:
                img = random_crop_arr(img, self.resolution)
            else:
                img = center_crop_arr(img, self.resolution)
        if self.random_flip and random.random() < 0.5:
            img = img[:, ::-1]
        if self.norm:
            img = img - self.mean / self.std
        return np.transpose(img, [2, 0, 1])

    def __len__(self):
        return len(self.imgs)


def load_data(data_dir='data/train_data/train_img.txt', resolution=512, crop=True, batch_size=16, shuffle=True):
    data = TrainsetImg(txt_path=data_dir, resolution=resolution, crop=crop)
    loader = DataLoader(data, batch_size=batch_size, shuffle=shuffle, num_workers=1, drop_last=True)
    while True:
        yield from loader


def center_crop_arr(pil_image, image_size):
    # We are not on a new enough PIL to support the `reducing_gap`
    # argument, which uses BOX downsampling at powers of two first.
    # Thus, we do it by hand to improve downsample quality.
    # print(pil_image.size)
    # while min(*pil_image.size) >= 2 * image_size:
    #     pil_image = pil_image.resize(
    #         tuple(x // 2 for x in pil_image.size), resample=Image.BOX
    #     )
    #
    # scale = image_size / min(*pil_image.size)
    # pil_image = pil_image.resize(
    #     tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    # )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size]


def random_crop_arr(pil_image, image_size, min_crop_frac=0.8, max_crop_frac=1.0):
    min_smaller_dim_size = math.ceil(image_size / max_crop_frac)
    max_smaller_dim_size = math.ceil(image_size / min_crop_frac)
    smaller_dim_size = random.randrange(min_smaller_dim_size, max_smaller_dim_size + 1)

    # We are not on a new enough PIL to support the `reducing_gap`
    # argument, which uses BOX downsampling at powers of two first.
    # Thus, we do it by hand to improve downsample quality.
    while min(*pil_image.size) >= 2 * smaller_dim_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = smaller_dim_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = random.randrange(arr.shape[0] - image_size + 1)
    crop_x = random.randrange(arr.shape[1] - image_size + 1)
    return arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size]


# data = TrainsetImg(txt_path='data/train_data/train_img.txt')
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

# if __name__ == '__main__':
#
#     da = load_data()
#     for i in range(3):
#         batch = next(da)
#         print('-----------', i)
#         print(batch.shape)
#         print(batch[0].shape)
#         # print(batch[1].shape)
#         print('--------\n')
