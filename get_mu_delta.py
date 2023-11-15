from dataset import TrainsetImg
from torch.utils.data import DataLoader
import torch


def get_m_s():
    data = TrainsetImg(reshape=False)
    print(data.__len__())
    dataloader = DataLoader(data, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)
    mean = torch.zeros(1)
    std = torch.zeros(1)
    for x in dataloader:
        mean[0] += x[:, 0, :, :].mean()
        std[0] += x[:, 0, :, :].std()
    mean.div_(len(data))
    std.div_(len(data))
    print('mean:', mean.numpy(), 'std:', std.numpy())
    return mean.numpy(), std.numpy()


# m, s = get_m_s()  # mean: [-977.9869] std: [52.43935]# mean: [-977.9869] std: [52.43935]

mu, delta = -977.9869, 52.43935
