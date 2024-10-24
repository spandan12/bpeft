import torch
import torchvision
import numpy as np
import torchvision.transforms as transforms

cifar100 = np.load("/home/dp7972/dp7972/Prompt/vpt/ys_dataset/200.npy")
print(cifar100)

transform = transforms.Compose([
    transforms.ToTensor(),  # Convert images to tensors
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize pixel values
])

cifar100_dataset = torchvision.datasets.CIFAR100('./D_ALL/data/', download = True,
                                                    transform = transform)

sub = torch.utils.data.Subset(cifar100_dataset, cifar100)

dl = torch.utils.data.DataLoader(sub, batch_size = 200, shuffle = False)

for data, label in dl:
    # print(data.shape)
    print(label)