import torch
import pytorch_lightning as pl
from data.multi_mnist_dataset import MNIST

class MNISTLoader(pl.LightningDataModule):
    def __init__(self, batch_size: int, train_transform=None, test_transform=None, *args, **kwargs):
        super().__init__()
        self.batch_size = batch_size
        self.train_dataset = MNIST(mode='train', transform=train_transform, *args, **kwargs)
        self.val_dataset = MNIST(mode='val', transform=test_transform, *args, **kwargs)
        self.test_dataset = MNIST(mode='test', transform=test_transform, *args, **kwargs)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size = self.batch_size,
            num_workers = 4,
            shuffle = True
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size = 100,
            num_workers = 4,
            shuffle = False
        )
        
    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size = 100,
            num_workers = 4,
            shuffle = False
        )


# import torchvision
# from torchvision import transforms
# import matplotlib.pyplot as plt
# transform = torchvision.transforms.Compose([transforms.ToTensor(),
#                                             transforms.Normalize((0.1307,), (0.3081,)),
#                                             ])
# data = MNISTLoader(batch_size=256,
#                    train_transform=transform,
#                    test_transform=transform,
#                    file_path='MTL_Dataset/multi_mnist.pickle')
# batch = next(iter(data.train_dataloader()))
# print(batch[0].shape)
# ims = batch[0] # Tensor (batch_size, 1, 36, 36)
# labs_l = batch[1][:, 0]
# labs_r = batch[1][:, 1]
# f, axarr = plt.subplots(4, 8, figsize=(20, 10))
# for j in range(8):
#     for i in range(4):
#         axarr[i][j].imshow(ims[j*2+i].squeeze(0).numpy(), cmap='gray') 
#         axarr[i][j].set_title('{}_{}'.format(labs_l[j*2+i],labs_r[j*2+i]))
# plt.tight_layout()
# plt.show()