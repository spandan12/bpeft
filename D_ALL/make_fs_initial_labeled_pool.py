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

def initial_pool_datasets(num_classes=10, num_samples_per_class = 2, dataset_name = 'cifar10'):
    data_path = "D_ALL/fsDataset"
    if not os.path.exists(data_path):
        os.mkdir(data_path)
    
    if not os.path.exists(data_path + "/" + dataset_name):
        os.mkdir(data_path + "/" + dataset_name)
    


    only_basic = transforms.Compose([transforms.ToTensor()])


    if dataset_name == 'cifar10':
        dataset_tr = torchvision.datasets.CIFAR10(root='D_ALL/data', 
                                            train=True, download=True, 
                                            transform=only_basic)
    # elif dataset_name == 'svhn'
    elif dataset_name == 'cifar100':
        dataset_tr = torchvision.datasets.CIFAR100('./D_ALL/data/', 
                                                        download = True,train=True,
                                                        transform = only_basic)
    
    tr_data_loader = torch.utils.data.DataLoader(dataset_tr, 
                                                 batch_size = 1, 
                                                 shuffle = False, drop_last = False)
    
    saved_pool = 0
    seen_labels, data_indices_to_save = {}, {}
    for idx, (_, label_tor) in enumerate(tr_data_loader):
        
        label = label_tor.item()
        
        if label in seen_labels and seen_labels[label] >= num_samples_per_class:
            continue
        
        else:
            if label not in seen_labels:
                seen_labels[label] = 1
                data_indices_to_save[label] = [idx]
            else:
                seen_labels[label] += 1
                data_indices_to_save[label] = data_indices_to_save[label] + [idx]
            
            saved_pool += 1
            if saved_pool >= num_samples_per_class*num_classes:
                break
    
    print("classes: ", seen_labels)

    val_max = 10
    for val_shot in [1, 2, 5, val_max]:
        save_data_list = []
        for k in data_indices_to_save.keys():
            class_k_data = data_indices_to_save[k]
            save_data_list += class_k_data[:val_shot]
        
        save_name = f'{dataset_name}_{val_shot}_shot_val'
        print("Save name: ", save_name)
        np.save(f"{data_path}/{dataset_name}/" + save_name, save_data_list)


    for tr_shot in [1,2,3,4,5,10,20,50]:
        save_data_list = []
        for k in data_indices_to_save.keys():
            class_k_data = data_indices_to_save[k]
            save_data_list += class_k_data[val_max:val_max + tr_shot]
        
        save_name = f'{dataset_name}_{tr_shot}_shot_tr'
        print("Save name: ", save_name)
        np.save(f"{data_path}/{dataset_name}/" + save_name, save_data_list)


    print("Done")
    return save_name

def main():
    # name = initial_pool_datasets(num_classes=10, num_samples_per_class=60, dataset_name = "cifar10")
    # print("CWD: ",os.getcwd(), "name: ", name)

    name = initial_pool_datasets(num_classes=100, num_samples_per_class=60, dataset_name = "cifar100")
    print("CWD: ",os.getcwd(), "name: ", name)

if __name__=='__main__':
    main()