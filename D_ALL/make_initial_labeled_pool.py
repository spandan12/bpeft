import os
import torch
import torchvision
import numpy as np
# import os
# print(os.getcwd())
# import D_ALL
# from D_ALL.custom_transforms import get_transform

import torchvision.transforms as transforms



from torch.utils.data import Dataset

def initial_pool_cifar10_deep(num_classes=10, num_samples_per_class = 2):
    
    only_basic = transforms.Compose([transforms.ToTensor()])

    cifar10_dataset_tr = torchvision.datasets.CIFAR10(root='D_ALL/data', 
                                          train=True, download=True, 
                                          transform=only_basic)
    
    tr_data_loader = torch.utils.data.DataLoader(cifar10_dataset_tr, 
                                                 batch_size = 1, 
                                                 shuffle = False, drop_last = False)
    
    seen_labels = {}
    data_indices_to_save = []
    saved_pool = 0
    for idx, (data, label_tor) in enumerate(tr_data_loader):
        label = label_tor.item()
        

        if label in seen_labels and seen_labels[label] >= num_samples_per_class:
            continue
        else:
            if label not in seen_labels:
                seen_labels[label] = 1
            else:
                seen_labels[label] += 1
            data_indices_to_save.append(idx)
            saved_pool += 1
            print(idx, label)
            if saved_pool >= num_samples_per_class*num_classes:
                break
    
    print("Indices: ", data_indices_to_save)
    print("classes: ", seen_labels)
    cifar10_initial_pool_array = np.array(data_indices_to_save)
    save_name = f'cifar10_{num_samples_per_class*num_classes}labels.npy'
    np.save("D_ALL/References/ys_dataset/" + save_name, cifar10_initial_pool_array)
    return save_name

def main():
    name = initial_pool_cifar10_deep(num_classes=10, num_samples_per_class=2)
    print("CWD: ",os.getcwd(), "name: ", name)

if __name__=='__main__':
    main()