import torch
import torchvision
import numpy as np
# import os
# print(os.getcwd())
# import D_ALL
# from D_ALL.custom_transforms import get_transform

import torchvision.transforms as transforms

def get_tr_transform_full(mean, std):
    resize_dim = 256
    crop_dim = 224
    return transforms.Compose([# transforms.RandomHorizontalFlip(),
                                   # transforms.RandomCrop(32, padding=4),
                                    transforms.Resize(resize_dim),
                                    transforms.RandomCrop(crop_dim),
                                    transforms.RandomHorizontalFlip(0.5),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean, std)])
def get_transform(mean, std, train=True):
    resize_dim = 256
    crop_dim = 224
    if train:
        return transforms.Compose([# transforms.RandomHorizontalFlip(),
                                   # transforms.RandomCrop(32, padding=4),
                                    transforms.Resize(resize_dim),
                                    transforms.RandomCrop(crop_dim),
                                    transforms.RandomHorizontalFlip(0.5),
                                    # transforms.ToTensor(),
                                    transforms.Normalize(mean, std)])
    else:
        return transforms.Compose([
                                    transforms.Resize(resize_dim),
                                    transforms.CenterCrop(crop_dim),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean, std)])

class TempClass():
    def __init__(self, name = "cifar100") -> None:
        self.name = name

from torch.utils.data import Dataset
class Custom_train_loader(Dataset):
    def __init__(self, data, label, train_transform, dataset_name = "cifar100"):
        self.data = data
        self.label = label
        self.dataset = TempClass(dataset_name)
        self.dataset.name = dataset_name
        self.num_samples = len(label)
        print("Num samples: ", self.num_samples)
        self.apply_transform = train_transform
    
    def __len__(self):
        if self.num_samples>200:
            return 2560
        return 1280
    
    def __getitem__(self, idx):
        ds = self.data[idx%self.num_samples]
        label = self.label[idx%self.num_samples]
        ds = self.apply_transform(ds)
        return (ds, label)

def get_cifar100_transforms():
    mean, std = {}, {}
    mean['cifar100'] = [x / 255 for x in [129.3, 124.1, 112.4]]
    std['cifar100'] = [x / 255 for x in [68.2,  65.4,  70.4]]
    test_transform = get_transform(mean['cifar100'], std['cifar100'], train=False)
    train_transform = get_transform(mean['cifar100'], std['cifar100'], train=True)
    train_transform_full = get_tr_transform_full(mean['cifar100'], std['cifar100'])
    return train_transform, test_transform, train_transform_full


def get_loaders_svhn_deep(al_eval = False, al_cycle = 0, 
                          output_dir_al='results', cfg=None):

    if not cfg:
        print("Pass CFG")
        raise NotImplementedError
    
    # mean, std = {}, {}
    # mean['svhn'] = [x / 255 for x in [129.3, 124.1, 112.4]]
    # std['svhn'] = [x / 255 for x in [68.2,  65.4,  70.4]]
    # test_transform = get_transform(mean['svhn'], std['svhn'], train=False)
    train_transform, test_transform, tr_transform_full = get_cifar100_transforms()
    

    sel_sample_ID=cfg.SOLVER.STRATEGY
    init_pool = cfg.SOLVER.INIT_POOL
    each_round = cfg.SOLVER.SEL_AL
    

    svhn = np.load("./D_ALL/References/ys_dataset/svhn10_num_labels_200.npy")
    sub_data_size = cfg.SOLVER.INIT_POOL

    print("Data indices Length: ", len(svhn)) # cifar100)
    assert len(svhn)==sub_data_size
    
    only_basic = transforms.Compose([transforms.ToTensor()])

    svhn_dataset_tr = torchvision.datasets.SVHN('./D_ALL/data/', 
                                                        download = True,
                                                        split='train',
                                                        transform = only_basic)
    

    full_tr_data_loader = torch.utils.data.DataLoader(svhn_dataset_tr, batch_size = 64, 
                                               shuffle = True, drop_last = False)

    sub = torch.utils.data.Subset(svhn_dataset_tr, svhn)

    fs_tr_data_loader = torch.utils.Data.DataLoader(sub, batch_size = sub_data_size, shuffle = False, drop_last = False)

    tr_image, tr_label = next(iter(fs_tr_data_loader))
    print(tr_label)
    print(tr_image.shape)
    train_transform = get_transform(mean['svhn'], std['svhn'], train=True)

    train_loader_high = Custom_train_loader(tr_image, tr_label,train_transform, dataset_name="svhn")
    train_loader = torch.utils.data.DataLoader(train_loader_high, batch_size = 64, 
                                               shuffle = True,drop_last = False)

    
    test_dataset = torchvision.datasets.SVHN('./D_ALL/data/', download = True,
                                                        split = 'test',
                                                        transform = test_transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size = 256, 
                                              shuffle = False, drop_last = False)
    val_loader = train_loader 

    train_loader.dataset.name = 'svhn'
    val_loader.dataset.name = 'svhn'
    test_loader.dataset.name = 'svhn'
    full_tr_data_loader.dataset.name = 'svhn'

    return train_loader, val_loader, test_loader, full_tr_data_loader

def get_loaders_cif100_deep(cfg = None):
    if not cfg:
        print("Pass CFG")
        raise NotImplementedError
    
    train_transform, test_transform, tr_transform_full = get_cifar100_transforms()

    cifar100 = np.load("./D_ALL/References/ys_dataset/200.npy")
    sub_data_size = cfg.SOLVER.INIT_POOL

    print("Data indices: ", len(cifar100), cifar100)
    assert len(cifar100)==sub_data_size
    
    only_basic = transforms.Compose([transforms.ToTensor()])
    cifar100_dataset_tr = torchvision.datasets.CIFAR100('./D_ALL/data/', 
                                                        download = True,
                                                        train=True,
                                                        transform = only_basic)
    
    full_tr_data_loader = torch.utils.data.DataLoader(cifar100_dataset_tr, batch_size = 64, 
                                               shuffle = True, drop_last = False)
    
    sub = torch.utils.data.Subset(cifar100_dataset_tr, cifar100)

    all_tr_data_loader = torch.utils.data.DataLoader(sub, batch_size = sub_data_size, 
                                               shuffle = False, drop_last = False)
    tr_image, tr_label = next(iter(all_tr_data_loader))

    train_loader_high = Custom_train_loader(tr_image, tr_label, train_transform, dataset_name="cifar100")
    train_loader = torch.utils.data.DataLoader(train_loader_high, batch_size = 64, 
                                               shuffle = True,drop_last = False)

    
    test_dataset = torchvision.datasets.CIFAR10('./D_ALL/data/', download = True,
                                                        train=False,
                                                        transform = test_transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size = 256, 
                                              shuffle = False, drop_last = False)
    val_loader = train_loader

    full_cifar100_dataset_tr = torchvision.datasets.CIFAR10('./D_ALL/data/', 
                                                        download = True,
                                                        train=True,
                                                        transform = tr_transform_full)
    
    full_tr_data_loader = torch.utils.data.DataLoader(full_cifar100_dataset_tr, batch_size = 64, 
                                               shuffle = True, drop_last = False)


    train_loader.dataset.name = 'cifar100'
    val_loader.dataset.name = 'cifar100'
    test_loader.dataset.name = 'cifar100'
    full_tr_data_loader.dataset.name = 'cifar100'

    return train_loader, val_loader, test_loader, full_tr_data_loader


def get_loaders_svhn_deep(al_eval = False, al_cycle = 0, 
                          output_dir_al='results', cfg=None):

    mean, std = {}, {}
    mean['svhn'] = [x / 255 for x in [129.3, 124.1, 112.4]]
    std['svhn'] = [x / 255 for x in [68.2,  65.4,  70.4]]
    test_transform = get_transform(mean['svhn'], std['svhn'], train=False)
    
    if al_eval:
        #For selecting samples next time
        ds = torchvision.datasets.SVHN('./D_ALL/data/', 
                                                download = True,
                                                split="train",
                                                transform = test_transform)
        return  torch.utils.data.DataLoader(ds, batch_size = 512, 
                                               shuffle = False,
                                               drop_last = False,)
    
    print("Training Phase. AL Cycle: ", al_cycle)
    
    if not cfg:
        print("Pass CFG")
        raise NotImplementedError

    sel_sample_ID=cfg.SOLVER.STRATEGY
    init_pool = cfg.SOLVER.INIT_POOL
    each_round = cfg.SOLVER.SEL_AL
    

    if al_cycle <= 0:
        svhn = np.load("./D_ALL/References/ys_dataset/svhn10_num_labels_200.npy")
    elif al_cycle > 0:
        num_samples_old = init_pool+each_round*al_cycle
        data_path = f"{output_dir_al}/{sel_sample_ID}_after_{al_cycle}_{num_samples_old}.npy"
        svhn = np.load(f"{data_path}")
    print("Data indices: ", len(svhn), svhn)
    
    only_basic = transforms.Compose([transforms.ToTensor()])

    svhn_dataset_tr = torchvision.datasets.SVHN('./D_ALL/data/', 
                                                        download = True,
                                                        split='train',
                                                        transform = only_basic)
    
    sub = torch.utils.data.Subset(svhn_dataset_tr, svhn)
    sub_data_size = int(init_pool + each_round * al_cycle)

    all_tr_data_loader = torch.utils.data.DataLoader(sub, batch_size = sub_data_size, 
                                               shuffle = False, drop_last = False)
    tr_image, tr_label = next(iter(all_tr_data_loader))
    print(tr_label)
    print(tr_image.shape)
    train_transform = get_transform(mean['svhn'], std['svhn'], train=True)

    train_loader_high = Custom_train_loader(tr_image, tr_label,train_transform, dataset_name="svhn")
    train_loader = torch.utils.data.DataLoader(train_loader_high, batch_size = 64, 
                                               shuffle = True,drop_last = False)

    
    test_dataset = torchvision.datasets.SVHN('./D_ALL/data/', download = True,
                                                        split = 'test',
                                                        transform = test_transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size = 256, 
                                              shuffle = False, drop_last = False)
    val_loader = train_loader 

    train_loader.dataset.name = 'svhn'
    val_loader.dataset.name = 'svhn'
    test_loader.dataset.name = 'svhn'

    return train_loader, val_loader, test_loader


def get_loaders_cifar10_deep(output_dir_al='results', cfg=None):
    
    size = 224
    train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.Resize(size),
            transforms.RandomHorizontalFlip(),
            # transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
    test_transform = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    

    train_ds = torchvision.datasets.CIFAR10(root='./D_ALL/data', 
                                        train=True, download=True, 
                                        transform=test_transform)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size = 512, 
                                            shuffle = False,
                                            drop_last = False,)
 
    
    if not cfg:
        print("Pass CFG")
        raise NotImplementedError

    sel_sample_ID=cfg.SOLVER.STRATEGY
    init_pool = cfg.SOLVER.INIT_POOL
    each_round = cfg.SOLVER.SEL_AL

    if al_cycle <= 0:
        cifar10 = np.load("./D_ALL/References/ys_dataset/cifar10_20labels.npy")
    elif al_cycle > 0:
        num_samples_old = init_pool+each_round*al_cycle
        data_path = f"{output_dir_al}/{sel_sample_ID}_after_{al_cycle}_{num_samples_old}.npy"
        cifar10 = np.load(f"{data_path}")
    print("Data indices: ", len(cifar10), cifar10)
    
    only_basic = transforms.Compose([transforms.ToTensor()])

    cifar10_dataset_tr = torchvision.datasets.CIFAR10(root='./D_ALL/data', 
                                                      train=True, download=True,
                                                      transform=only_basic)
    
    sub = torch.utils.data.Subset(cifar10_dataset_tr, cifar10)
    sub_data_size = int(init_pool + each_round * al_cycle)

    all_tr_data_loader = torch.utils.data.DataLoader(sub, batch_size = sub_data_size, 
                                               shuffle = False, drop_last = False)
    tr_image, tr_label = next(iter(all_tr_data_loader))
    print(tr_label)
    print(tr_image.shape)
    train_loader_high = Custom_train_loader(tr_image, tr_label,train_transform, dataset_name="cifar10")
    train_loader = torch.utils.data.DataLoader(train_loader_high, batch_size = 64, 
                                               shuffle = True,drop_last = False)

    
    test_dataset = torchvision.datasets.CIFAR10(root='./D_ALL/data', 
                                          train=False, download=True, 
                                          transform=test_transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size = 256, 
                                              shuffle = False, drop_last = False)
    val_loader = train_loader 

    train_loader.dataset.name = 'cifar10'
    val_loader.dataset.name = 'cifar10'
    test_loader.dataset.name = 'cifar10'

    return train_loader, val_loader, test_loader

def main():
    get_loaders_cifar10_deep(al_eval=True)

if __name__=='__main__':
    main()