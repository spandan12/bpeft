import torch
import torchvision
import numpy as np
import os
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
def get_transform(mean, std):
    resize_dim = 256
    crop_dim = 224
    train_transform = transforms.Compose([# transforms.RandomHorizontalFlip(),
                                   # transforms.RandomCrop(32, padding=4),
                                    transforms.Resize(resize_dim),
                                    transforms.RandomCrop(crop_dim),
                                    transforms.RandomHorizontalFlip(0.5),
                                    # transforms.ToTensor(),
                                    transforms.Normalize(mean, std)])
    val_transform = transforms.Compose([
                                    transforms.Resize(resize_dim),
                                    transforms.CenterCrop(crop_dim),
                                    transforms.Normalize(mean, std)])
    test_transform = transforms.Compose([
                                    transforms.Resize(resize_dim),
                                    transforms.CenterCrop(crop_dim),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean, std)])
    return train_transform, val_transform, test_transform

class TempClass():
    def __init__(self, name = "cifar100") -> None:
        self.name = name

from torch.utils.data import Dataset
class Custom_train_loader(Dataset):
    def __init__(self, sub_dl, train_transform, cfg):

        dataset_name = cfg.DATA.NAME

        all_tr_data_loader = torch.utils.data.DataLoader(sub_dl, batch_size = len(sub_dl),shuffle = False, drop_last = False)
        data, label = next(iter(all_tr_data_loader))
        

        self.data = data
        self.label = label
        self.dataset = TempClass(dataset_name)

        self.mult = cfg.SOLVER.FSTR
        self.dataset.name = dataset_name
        self.num_samples = len(label)
        print("Num samples: ", self.num_samples)
        self.apply_transform = train_transform
    
    def __len__(self):
        # return 100
        return 1280*self.mult
    
    def __getitem__(self, idx):
        ds = self.data[idx%self.num_samples]
        label = self.label[idx%self.num_samples]
        ds = self.apply_transform(ds)
        return (ds, label)

def get_cifar100_transforms():
    mean, std = {}, {}
    mean['cifar100'] = [x / 255 for x in [129.3, 124.1, 112.4]]
    std['cifar100'] = [x / 255 for x in [68.2,  65.4,  70.4]]
    train_transform, _, test_transform = get_transform(mean['cifar100'], std['cifar100'])
    train_transform_full = get_tr_transform_full(mean['cifar100'], std['cifar100'])
    return train_transform, test_transform, train_transform_full

def get_svhn_loader_as_ood():
    mean, std = {}, {}
    mean['svhn'] = [x / 255 for x in [129.3, 124.1, 112.4]]
    std['svhn'] = [x / 255 for x in [68.2,  65.4,  70.4]]
    _,_, test_transform = get_transform(mean['svhn'], std['svhn'])

    test_dataset = torchvision.datasets.SVHN('./D_ALL/data/', download = True,
                                                        split = 'test',
                                                        transform = test_transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size = 256, 
                                              shuffle = False, drop_last = False)

    test_loader.dataset.name = 'svhn'

    return test_loader                                          

def get_loaders_svhn_deep(al_eval = False, al_cycle = 0, 
                          output_dir_al='results', cfg=None):

    mean, std = {}, {}
    mean['svhn'] = [x / 255 for x in [129.3, 124.1, 112.4]]
    std['svhn'] = [x / 255 for x in [68.2,  65.4,  70.4]]
    _,_, test_transform = get_transform(mean['svhn'], std['svhn'])
    
    if al_eval:
        #For selecting samples next time
        ds = torchvision.datasets.SVHN('./D_ALL/data/', 
                                                download = True,
                                                split="train",
                                                transform = test_transform)
        return  torch.utils.data.DataLoader(ds, batch_size = 512, 
                                               shuffle = False,
                                               drop_last = False,)

    
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
    train_transform, val_transform, _ = get_transform(mean['svhn'], std['svhn'], train=True)

    train_loader_high = Custom_train_loader(tr_image, tr_label,train_transform, cfg)
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


def get_cifar10_tr_test_data(train_transform, test_transform):
    train_dataset = torchvision.datasets.CIFAR10(root='./D_ALL/data', train=True, download=True, transform=train_transform)
    test_dataset = torchvision.datasets.CIFAR10(root='./D_ALL/data', train=False, download=True, transform=test_transform)
    return train_dataset, test_dataset

def get_cifar10_train_val_test_transform():
    size = 224
    train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.Resize(size),
            transforms.RandomHorizontalFlip(),
            # transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
    val_transform = transforms.Compose([
        transforms.Resize(size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        # transforms.ColorJitter(),
        # transforms.GaussianBlur(kernel_size=5),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    test_transform = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    return train_transform, val_transform, test_transform

def get_loaders_cifar10_deep( cfg=None):
    if not cfg:
        print("Pass CFG")
        raise NotImplementedError

    only_basic = transforms.Compose([transforms.ToTensor()])

    train_transform, val_transform, test_transform = get_cifar10_train_val_test_transform()
    train_dataset, test_dataset = get_cifar10_tr_test_data(train_transform, test_transform)
    dataset_tr_ref_sub, _ = get_cifar10_tr_test_data(only_basic, only_basic)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = 64, shuffle = True, drop_last = False,)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size = 256, shuffle = False, drop_last = False)


    fs_tr_indx = np.load(f"{os.getcwd()}/D_ALL/fsDataset/{cfg.DATA.NAME}/{cfg.DATA.NAME}_{cfg.SOLVER.FSTR}_shot_tr.npy")
    fs_val_indx = np.load(f"{os.getcwd()}/D_ALL/fsDataset/{cfg.DATA.NAME}/{cfg.DATA.NAME}_{cfg.SOLVER.FSVAL}_shot_val.npy")
    
    sub_tr = torch.utils.data.Subset(dataset_tr_ref_sub, fs_tr_indx)
    sub_val = torch.utils.data.Subset(dataset_tr_ref_sub, fs_val_indx)

    fs_train_loader_cl = Custom_train_loader(sub_tr,train_transform, cfg)
    fs_val_loader_cl = Custom_train_loader(sub_val,val_transform, cfg) #val

    fs_train_loader = torch.utils.data.DataLoader(fs_train_loader_cl, batch_size = 64, shuffle = True,drop_last = False)
    fs_val_loader = torch.utils.data.DataLoader(fs_val_loader_cl, batch_size = 64, shuffle = False,drop_last = False)

    fs_train_loader.dataset.name = cfg.DATA.NAME
    fs_val_loader.dataset.name = cfg.DATA.NAME
    train_loader.dataset.name = cfg.DATA.NAME
    test_loader.dataset.name = cfg.DATA.NAME

    return fs_train_loader, fs_val_loader, test_loader, train_loader

def get_cifar100_tr_test_data(train_transform, test_transform):
    train_dataset = torchvision.datasets.CIFAR100('./D_ALL/data/', download = True, train=True, transform = train_transform)
    test_dataset = torchvision.datasets.CIFAR100('./D_ALL/data/', download = True, train=False, transform = test_transform)
    return train_dataset, test_dataset
    

def get_loaders_cif100_deep(cfg = None):
    if not cfg:
        print("Pass CFG")
        raise NotImplementedError
    
    only_basic = transforms.Compose([transforms.ToTensor()])
    
    #TODO LATER
    train_transform, test_transform, tr_transform_full = get_cifar100_transforms()
    val_transform = train_transform

    train_dataset, test_dataset = get_cifar100_tr_test_data(train_transform, test_transform)
    dataset_tr_ref_sub, _ = get_cifar100_tr_test_data(only_basic, only_basic)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = 64, shuffle = True, drop_last = False,)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size = 256, shuffle = False, drop_last = False)

    fs_tr_indx = np.load(f"./D_ALL/fsDataset/{cfg.DATA.NAME}/{cfg.DATA.NAME}_{cfg.SOLVER.FSTR}_shot_tr.npy")
    fs_val_indx = np.load(f"./D_ALL/fsDataset/{cfg.DATA.NAME}/{cfg.DATA.NAME}_{cfg.SOLVER.FSVAL}_shot_val.npy")
    
    sub_tr = torch.utils.data.Subset(dataset_tr_ref_sub, fs_tr_indx)
    sub_val = torch.utils.data.Subset(dataset_tr_ref_sub, fs_val_indx)

    fs_train_loader_cl = Custom_train_loader(sub_tr,train_transform, cfg)
    fs_val_loader_cl = Custom_train_loader(sub_val,val_transform, cfg) #val

    fs_train_loader = torch.utils.data.DataLoader(fs_train_loader_cl, batch_size = 64, shuffle = True,drop_last = False)
    fs_val_loader = torch.utils.data.DataLoader(fs_val_loader_cl, batch_size = 64, shuffle = False,drop_last = False)

    fs_train_loader.dataset.name = cfg.DATA.NAME
    fs_val_loader.dataset.name = cfg.DATA.NAME
    train_loader.dataset.name = cfg.DATA.NAME
    test_loader.dataset.name = cfg.DATA.NAME

    return fs_train_loader, fs_val_loader, test_loader, train_loader


def main():
    get_loaders_cifar10_deep()

if __name__=='__main__':
    main()