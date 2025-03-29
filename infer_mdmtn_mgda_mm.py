from src.MDMTN_MGDA_model_MM import MDMTNmgda_MultiTaskNetwork_I
import torch
from src.utils.projectedOWL_utils import proxOWL
import matplotlib.pyplot as plt
import numpy as np
from torchvision import transforms
from data.multi_mnist_dataloader import MNISTLoader

def load_MultiMnist_data():

    train_transform = transforms.Compose([transforms.ToTensor(),
                                            transforms.Normalize((0.1307,), (0.3081,)),
                                           transforms.Resize((28, 28))])

    test_transform = transforms.Compose([transforms.ToTensor(),
                                            transforms.Normalize((0.1307,), (0.3081,)),
                                           transforms.Resize((28, 28))])
    data = MNISTLoader(batch_size=[256, 100],
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
    mod_params_mgda = {"batch_size": 256}
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    model = MDMTNmgda_MultiTaskNetwork_I(mod_params_mgda["batch_size"], device=device, static_a = [False, None])
    # model = torch.load("logs/MDMTN_MM_logs/MGDA_model_logs/model_states/29-03-2025--13-35-59/model_9model_task_1.pth", map_location="cpu", weights_only=False)
    model.load_model("logs/MDMTN_MM_logs/MGDA_model_logs/model_states/29-03-2025--13-35-59/model_9")
    model.eval()

    # Predict
    # outputs = model(images[0].unsqueeze(0))
    images = images.to(device)
    outputs = model(images.unsqueeze(-1))
    task1_outputs = outputs[:, :10]
    task2_outputs = outputs[:, 10:]

    task1_preds = task1_outputs.argmax(dim=1)
    task2_preds = task2_outputs.argmax(dim=1)
    for i in range(5):
        print(f"Image {i} | Task 1 → Pred: {task1_preds[i].item()} | GT: {targets[0][i].item()}  ||  Task 2 → Pred: {task2_preds[i].item()} | GT: {targets[1][i].item()}")