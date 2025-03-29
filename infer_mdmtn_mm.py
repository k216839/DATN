from src.MDMTN_model_MM import SparseMonitoredMultiTaskNetwork_I
import torch
from src.utils.projectedOWL_utils import proxOWL
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
    return train_loader, val_loader, test_loader

if __name__ == '__main__':
    # Data
    train_loader, val_loader, test_loader = load_MultiMnist_data()

    images, targets = next(iter(train_loader))
    label_l = targets[0][0].item()
    label_r = targets[1][0].item()

    print(f"Image batch shape: {images.shape}")
    print(f"Left label batch shape: {targets[0].shape}")
    print(f"Right label batch shape: {targets[1].shape}")
    

    img = images[0]
    plt.figure(figsize=(5, 5))
    plt.imshow(img.squeeze(0), cmap='gray')
    plt.title(f"({label_l}, {label_r})")
    plt.axis('off')
    plt.show(block=False)
    plt.pause(2) 
    plt.close()

    # Model 
    GrOWL_parameters = {"tp": "spike", #"Dejiao", #"linear", 
                    "beta1": 0.8,  
                    "beta2": 0.2, 
                "proxOWL": proxOWL,
                "skip_layer": 1, # Skip layer with "1" neuron
                    "sim_preference": 0.7, 
                }
    GrOWL_parameters["max_layerSRate"] = 0.8
    model = SparseMonitoredMultiTaskNetwork_I(GrOWL_parameters, num_classes=[10, 10])
    model = torch.load("logs/MDMTN_MM_logs/MDMTN_model_MM_onek/model000.pth", map_location="cpu", weights_only=False)
    model.eval()

    # Predict
    outputs = model(images[0].unsqueeze(0))
    for i, out in enumerate(outputs):
        pred = out.argmax(dim=1).item()
        gt = label_l if i == 0 else label_r
        print(f"Task {i+1} | Predicted: {pred} | Ground Truth: {gt}")