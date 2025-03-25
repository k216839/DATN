import matplotlib.pyplot as plt
import numpy as np
from torchvision import transforms
from data.multi_mnist_dataloader import MNISTLoader
def load_MultiMnist_data():

    train_transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize((0.1307,), (0.3081,))
                                        ])

    test_transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize((0.1307,), (0.3081,))
                                        ])
    data = MNISTLoader(batch_size=256,
                    train_transform=train_transform,
                    test_transform=test_transform,
                    file_path='MTL_Dataset/multi_mnist.pickle')
    train_loader, val_loader, test_loader = data.train_dataloader(), data.val_dataloader(), data.test_dataloader()
    print("Data loaded!")

    print("Show sample image...")
    # Get the first batch from the train loader
    batch = next(iter(train_loader))
    print(batch[0].shape)
    ims = batch[0] # Tensor (batch_size, 1, 36, 36)
    labs_l = batch[1][:, 0]
    labs_r = batch[1][:, 1]
    f, axarr = plt.subplots(4, 8, figsize=(20, 10))
    for j in range(8):
        for i in range(4):
            axarr[i][j].imshow(ims[j*2+i].squeeze(0).numpy(), cmap='gray') 
            axarr[i][j].set_title('{}_{}'.format(labs_l[j*2+i],labs_r[j*2+i]))
    plt.tight_layout()
    plt.show()

    return train_loader, val_loader, test_loader