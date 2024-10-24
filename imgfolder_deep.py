import bisect
import os
import os.path

from PIL import Image
import numpy as np
import copy
from itertools import accumulate

import torch
import torch.utils.data as data
from torchvision import datasets, transforms

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def find_classes(dir):
    classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)


def make_dataset(dir, class_to_idx, file_list):
    images = []
    # print('here')
    dir = os.path.expanduser(dir)
    set_files = [line.rstrip('\n') for line in open(file_list)]
    for target in sorted(os.listdir(dir)):
        # print(target)
        d = os.path.join(dir, target)
        if not os.path.isdir(d):
            continue

        for root, _, fnames in sorted(os.walk(d)):
            for fname in sorted(fnames):
                if is_image_file(fname):
                    dir_file = target + '/' + fname
                    # print(dir_file)
                    if dir_file in set_files:
                        path = os.path.join(root, fname)
                        item = (path, class_to_idx[target])
                        images.append(item)
    return images


def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)


class ImageFolderTrainVal(datasets.ImageFolder):
    def __init__(self, root, files_list, transform=None, target_transform=None,
                 loader=default_loader, classes=None, class_to_idx=None, imgs=None):
        """
        :param root: root path of the dataset
        :param files_list: list of filenames to include in this dataset
        :param classes: classes to include, based on subdirs of root if None
        :param class_to_idx: overwrite class to idx mapping
        :param imgs: list of image paths (under root)
        """
        if classes is None:
            assert class_to_idx is None
            classes, class_to_idx = find_classes(root)
        elif class_to_idx is None:
            class_to_idx = {classes[i]: i for i in range(len(classes))}
        print("Creating Imgfolder with root: {}".format(root))
        imgs = make_dataset(root, class_to_idx, files_list) if imgs is None else imgs
        if len(imgs) == 0:
            raise (RuntimeError("Found 0 images in subfolders of: {}\nSupported image extensions are: {}".
                                format(root, ",".join(IMG_EXTENSIONS))))
        self.root = root
        self.samples = imgs
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader


class ImageFolder_Subset(ImageFolderTrainVal):
    """
    Wrapper of ImageFolderTrainVal, subsetting based on indices.
    """

    def __init__(self, dataset, indices):
        self.__dict__ = copy.deepcopy(dataset).__dict__
        self.indices = indices  # Extra

    def __getitem__(self, idx):
        return super().__getitem__(self.indices[idx])  # Only return from subset

    def __len__(self):
        return len(self.indices)
    

class ImageFolderSSL_Subset(ImageFolderTrainVal):
    """
    Wrapper of ImageFolderTrainVal, subsetting based on indices.
    """

    def __init__(self, dataset, indices, strong_transform=None):
        self.__dict__ = copy.deepcopy(dataset).__dict__
        self.indices = indices  # Extra
        self.strong_transform = strong_transform

    def __getitem__(self, idx):
        img, label = super().__getitem__(self.indices[idx])  # Only return from subset
        if not self.strong_transform:
            return (img, label)
        else:
            return (self.strong_transform[0](img), self.strong_transform[1](img), label)

    def __len__(self):
        return len(self.indices)



class ImagePathlist(data.Dataset):
    """
    Adapted from: https://github.com/pytorch/vision/issues/81
    Load images from a list with paths (no labels).
    """

    def __init__(self, imlist, targetlist=None, root='', transform=None, loader=default_loader):
        self.imlist = imlist
        self.targetlist = targetlist
        self.root = root
        self.transform = transform
        self.loader = loader

    def __getitem__(self, index):
        impath = self.imlist[index]

        img = self.loader(os.path.join(self.root, impath))
        if self.transform is not None:
            img = self.transform(img)

        if self.targetlist is not None:
            target = self.targetlist[index]
            return img, target
        else:
            return img

    def __len__(self):
        return len(self.imlist)


def random_split(dataset, lengths):
    """
    Creates ImageFolder_Subset subsets from the dataset, by altering the indices.
    :param dataset:
    :param lengths:
    :return: array of ImageFolder_Subset objects
    """
    assert sum(lengths) == len(dataset)
    indices = torch.randperm(sum(lengths))
    return [ImageFolder_Subset(dataset, indices[offset - length:offset]) for offset, length in
            zip(accumulate(lengths), lengths)]





def get_transforms(split, size=224):
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    if size == 448:
        resize_dim = 512
        crop_dim = 448
    elif size == 224:
        resize_dim = 256
        crop_dim = 224
    elif size == 384:
        resize_dim = 438
        crop_dim = 384
    if split == "train":
        transform = transforms.Compose(
            [
                transforms.Resize(resize_dim),
                transforms.RandomCrop(crop_dim),
                transforms.RandomHorizontalFlip(0.5),
                # tv.transforms.RandomResizedCrop(224, scale=(0.2, 1.0)),
                # tv.transforms.RandomHorizontalFlip(),
                # tv.transforms.ToTensor(),
                # normalize,
            ]
        )
    else:
        transform = transforms.Compose(
            [
                transforms.Resize(resize_dim),
                transforms.CenterCrop(crop_dim),
                # transforms.ToTensor(),
                # normalize,
            ]
        )
    return transform



def divide_into_tasks(root_path, task_count=1):
    nb_classes_task = 200 
    #/home/dp7972/dp7972/Prompt/vpt/D_ALL/data/tiny-imagenet-200/wnids.txt
    file_path = os.path.join(root_path, "classes.txt")
    lines = [line.rstrip('\n') for line in open(file_path)]
    assert len(lines) == 200, "Should have 200 classes, but {} lines in classes.txt".format(len(lines))
    subsets = ['train', 'val']
    img_paths = {s: [] for s in subsets + ['classes', 'class_to_idx']}
    print("img_paths:", img_paths)
    for subset in subsets:
        classes = lines
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        if len(img_paths['classes']) == 0:
                img_paths['classes'].extend(classes)
        img_paths['class_to_idx'] = class_to_idx

        # Make subset dataset dir for each task
        for class_index in range(200):
            target = lines[class_index]
            src_path = os.path.join(root_path, subset, target, 'images')
            imgs = [(os.path.join(src_path, f), class_to_idx[target]) for f in os.listdir(src_path)
                        if os.path.isfile(os.path.join(src_path, f))]  # (label_idx, path)
            img_paths[subset].extend(imgs)
            # print("img_paths:", img_paths)
    return img_paths

def get_tiny_dset():
    root = "./D_ALL/data/tiny-imagenet-200"
    img_paths = divide_into_tasks(root)

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    sufx_transf = [transforms.ToTensor(), normalize, ]
    sufx_transf_val = [transforms.ToTensor(), normalize, ]
    # modify the transform 
    train_transf = transforms.Compose([get_transforms("train")] + sufx_transf)
    val_transf = transforms.Compose([get_transforms("test")] + sufx_transf_val)
    train_dataset = ImageFolderTrainVal(root, None, transform=train_transf, classes=img_paths['classes'],
                                        class_to_idx=img_paths['class_to_idx'], imgs=img_paths['train'])

    eval_tr_dataset = ImageFolderTrainVal(root, None, transform=val_transf, classes=img_paths['classes'],
                                        class_to_idx=img_paths['class_to_idx'], imgs=img_paths['train'])

    val_dataset = ImageFolderTrainVal(root, None, transform=val_transf, classes=img_paths['classes'],
                                        class_to_idx=img_paths['class_to_idx'], imgs=img_paths['val'])

    return train_dataset, eval_tr_dataset, val_dataset


train_dataset, eval_tr_dataset, val_dataset = get_tiny_dset()
print(len(train_dataset), len(eval_tr_dataset), len(val_dataset))