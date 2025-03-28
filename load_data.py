import matplotlib.pyplot as plt
import numpy as np
import torchvision
from torchvision import transforms
from data.multi_mnist_dataloader import MNISTLoader
def load_MultiMnist_data():

    train_transform = torchvision.transforms.Compose([transforms.ToTensor(),
                                            transforms.Normalize((0.1307,), (0.3081,)),
                                           transforms.Resize((28, 28))])

    test_transform = torchvision.transforms.Compose([transforms.ToTensor(),
                                            transforms.Normalize((0.1307,), (0.3081,)),
                                           transforms.Resize((28, 28))])
    data = MNISTLoader(batch_size=256,
                    train_transform=train_transform,
                    test_transform=test_transform,
                    file_path='MTL_dataset/multi_mnist.pickle')
    train_loader, val_loader, test_loader = data.train_dataloader(), data.val_dataloader(), data.test_dataloader()
    print("Data loaded!")

    print("Show sample image...")
    # Get the first batch from the train loader
    images, targets = next(iter(train_loader))

    labs_l = targets[0].squeeze()  
    labs_r = targets[1].squeeze()  

    print(f"Image batch shape: {images.shape}")
    print(f"Left label batch shape: {labs_l.shape}")
    print(f"Right label batch shape: {labs_r.shape}")
    print(targets[0][0])
    # f, axarr = plt.subplots(4, 8, figsize=(20, 10))
    # for j in range(8):
    #     for i in range(4):
    #         axarr[i][j].imshow(images[j*2+i].squeeze(0).numpy(), cmap='gray') 
    #         axarr[i][j].set_title('{}_{}'.format(labs_l[j*2+i],labs_r[j*2+i]))
    # plt.tight_layout()
    # plt.show()

    # img = images[0]
    # plt.figure(figsize=(5, 5))
    # plt.imshow(img, cmap='gray')
    # plt.title(f"({targets[0][0].item()}, {targets[1][0].item()})")
    # plt.axis('off')
    # plt.show()

    return train_loader, val_loader, test_loader
if __name__ == '__main__':
    train_loader, val_loader, test_loader = load_MultiMnist_data()