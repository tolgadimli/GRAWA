import os
import time

import torch
import torch.distributed as dist
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import torch.utils.data as udata
import torchvision.models
from models import PyramidNet, Wide_ResNet, VGG16, ResNet20, ResNet56, densenet121
import random

# ---- Usefull Utilities ----
def mkdir(path):
    '''create a single empty directory if it didn't exist
    Parameters: path (str) -- a single directory path'''
    if not os.path.exists(path):
        os.makedirs(path)

def mkdirs(paths):
    '''create empty directories if they don't exist
    Parameters: paths (str list) -- a list of directory paths'''
    rmdirs(paths)
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)

def rmdirs(paths):
    if os.path.exists(paths):
        for file in os.listdir(paths): 
            file_path = os.path.join(paths, file)
            if os.path.isfile(file_path):
                os.remove(file_path)
        os.rmdir(paths)

def tensor2text(tensor):
    '''convert tensor's contents to readable texts'''
    tensor_text = ""
    tensor_type = tensor.type()
    int_tensor_dict = ['torch.cuda.CharTensor', 'torch.cuda.IntTensor']
    float_tensor_dict = ['torch.cuda.FloatTensor', 'torch.cuda.DoubleTensor']
    for var in tensor:
        if tensor_type in float_tensor_dict:
            tensor_text += "%.4f "%var.item()
        elif tensor_type in int_tensor_dict:
            tensor_text += "%d "%var.item()
        else:
            raise ValueError(var, "Undefined tensor type to print")
    return '[' + tensor_text + ']'

# ---- PyTorch Utilities ----
def get_data_loader(args):
    '''return train and test dataloader'''
    # [normalization]
    normalize = None
    if args.dataset == 'CIFAR10' or args.dataset == 'CIFAR100':

        if args.dataset == 'CIFAR10':
            print('CIFAR10 normalization')
            # https://github.com/kuangliu/pytorch-cifar/blob/master/main.py
            normalize = transforms.Normalize((0.4914, 0.4822, 0.4465),
                                            (0.2023, 0.1994, 0.2010))
        else:
            print('CIFAR100 normalization')
            normalize = transforms.Normalize((0.5071, 0.4867, 0.4408),
                                            (0.2675, 0.2565, 0.2761)) 
    
        # [transformation]
        if args.model == 'resnet20':
            print("RESNET20 TRANSFORMATIONS ARE USED.")
            transformTR = transforms.Compose([
                            transforms.RandomHorizontalFlip(),
                            transforms.RandomCrop(28),
                            transforms.ToTensor(),
                            normalize,
                        ])
            transformTE = transforms.Compose([
                            transforms.CenterCrop(28),
                            transforms.ToTensor(),
                            normalize,
                        ])

        elif args.model == 'vgg16' or args.model == 'pyramidnet' or args.model =='wideresnet' or \
                                                            args.model == 'resnet56' or args.model == 'densenet':
            print("VGG16+PYRAMID+WIDERESNET_RESNET56 TRANSFORMATIONS ARE USED.")
            transformTR = transforms.Compose([
                            transforms.RandomHorizontalFlip(),
                            transforms.RandomCrop(size= 32, padding = 4),
                            transforms.ToTensor(),
                            normalize,
                        ])
            transformTE = transforms.Compose([
                            transforms.ToTensor(),
                            normalize,
                        ])            

        # [dataset]
        train_dataset = dsets.__dict__[args.dataset](root=args.datadir,
                                                    train=True,
                                                    transform=transformTR,
                                                    download=False)

        test_dataset = dsets.__dict__[args.dataset](root=args.datadir,
                                                    train=False,
                                                    transform=transformTE,
                                                    download=False)

        # [data loader] arguments
        # when 'num_loader' > 0 the loading hangs before each epoch which seems to be a strange bug
        # 'num_loaders = 0' means loading in main process and it works fine
        # raised question by me: https://discuss.pytorch.org/t/training-hangs-for-a-second-at-the-beginning-of-each-epoch/39304
        num_loaders = 0
        is_pin_mem = True # This will be forced to be True in distributed training

    elif args.dataset == 'ImageNet':
        traindir = os.path.join(args.datadir, 'train')
        valdir = os.path.join(args.datadir, 'val')

        # if not os.path.exists(traindir):
        #     os.mkdir(traindir)

        # if not os.path.exists(valdir):
        #     os.mkdir(valdir)
        # [normalization]
                                                    
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                          std=[0.229, 0.224, 0.225])

        # [dataset]
        train_dataset = dsets.ImageFolder(
                traindir,
                transforms.Compose([
                    transforms.RandomResizedCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalize,
                ]))

        test_dataset = dsets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]))
        num_loaders = 4
        is_pin_mem = True # This will be forced to be True in distributed training

    # [data loader] set up
    train_sampler = None
    train_sampler = udata.distributed.DistributedSampler(train_dataset, args.world_size, dist.get_rank())
    if args.random_sampler:
        train_loader = udata.DataLoader(dataset=train_dataset,
                                        batch_size=args.batch_size,
                                        num_workers=num_loaders,
                                        shuffle=True,
                                        pin_memory=is_pin_mem,
                                        drop_last=False)
    else:
        train_loader = udata.DataLoader(dataset=train_dataset,
                                        batch_size=args.batch_size,
                                        num_workers=num_loaders,
                                        shuffle=False,
                                        pin_memory=is_pin_mem,
                                        sampler=train_sampler,
                                        drop_last=False)
    
    test_loader  = udata.DataLoader(dataset=test_dataset,
                                    batch_size=args.batch_size,
                                    num_workers=num_loaders,
                                    shuffle=False,
                                    pin_memory=is_pin_mem)

    return train_sampler, train_dataset, train_loader, test_dataset, test_loader

def get_model(args):
    '''return pre-*defined* model'''
    if args.model == 'vgg16':  
        return VGG16() 
    elif args.model == 'resnet20':  
        return ResNet20() 
    
    elif args.model == 'resnet56' and args.dataset == 'CIFAR10':
        return ResNet56(num_classes=10) 
    elif args.model == 'resnet56' and args.dataset == 'CIFAR100':
        return ResNet56(num_classes=100) 
    
    elif args.model == 'pyramidnet' and args.dataset == 'CIFAR10':
        return PyramidNet(110, 270, 10)
    elif args.model == 'pyramidnet' and args.dataset == 'CIFAR100':
        return PyramidNet(110, 270, 100)
    
    elif args.model == 'densenet' and args.dataset == 'CIFAR10':
        return densenet121(num_class=10)
    elif args.model == 'densenet' and args.dataset == 'CIFAR100':
        return densenet121(num_class=100)
    
    elif args.model == 'wideresnet' and args.dataset == 'CIFAR10':
        print('Using WideResNet')
        return Wide_ResNet(28, 10, num_classes=10)
    elif args.model == 'wideresnet' and args.dataset == 'CIFAR100':
        print('Using WideResNet')
        return Wide_ResNet(28, 10, num_classes=100)
    
    elif args.model == 'resnet50':
        return torchvision.models.resnet50(pretrained=False)
    else:
        raise ValueError("wrong model name!")     

class AverageMeter(object):
    '''computes and stores the average and current value'''
    def __init__(self, running_size=10):
        self.running_avg = 0
        self.running_size = running_size
        self.running_vec = torch.zeros(running_size)
        self.reset()

    def reset(self):
        self.running_index = 0
        self.running_vec.zero_()
        self.running_avg = 0

    def update(self, val, n=1):
        if self.running_index >= self.running_size:
            self.running_index = 0
        self.running_vec[self.running_index] = val
        self.running_index += 1
        self.running_avg = self.running_vec.mean().item()


def get_all_trainset(args):
    if args.dataset == 'CIFAR10' or args.dataset == 'CIFAR100':

        if args.dataset == 'CIFAR10':
            print('CIFAR10 normalization')
            # https://github.com/kuangliu/pytorch-cifar/blob/master/main.py
            normalize = transforms.Normalize((0.4914, 0.4822, 0.4465),
                                            (0.2023, 0.1994, 0.2010))
        else:
            print('CIFAR100 normalization')
            normalize = transforms.Normalize((0.5071, 0.4867, 0.4408),
                                            (0.2675, 0.2565, 0.2761)) 
        # [transformation]
        if args.model == 'resnet20':
            transformTR = transforms.Compose([
                            # transforms.RandomHorizontalFlip(),
                            # transforms.RandomCrop(28),
                            transforms.CenterCrop(28),
                            transforms.ToTensor(),
                            normalize,
                        ])

        elif args.model == 'vgg16' or args.model == 'pyramidnet' or args.model =='wideresnet' or \
                                                        args.model == 'resnet56' or args.model == 'densenet':
            transformTR = transforms.Compose([
                            # transforms.RandomHorizontalFlip(),
                            # transforms.RandomCrop(size = 32, padding = 4),
                            transforms.ToTensor(),
                            normalize,
                        ])        

        # [dataset]
        train_dataset = dsets.__dict__[args.dataset](root=args.datadir,
                                                    train=True,
                                                    transform=transformTR,
                                                    download=False)

    elif args.dataset == 'ImageNet':

        traindir = os.path.join(args.datadir, 'train')
                                                    
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                          std=[0.229, 0.224, 0.225])

        # [dataset]
        train_dataset = dsets.ImageFolder(
                traindir,
                transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    normalize,
                ]))            

    return train_dataset
                                                


def get_subdataloader( args, train_dataset ):

    indexes = random.sample( range(0, len(train_dataset)), args.size ) #number of batches

    subtrainset = torch.utils.data.Subset(train_dataset, indexes)
    subtrainloader = torch.utils.data.DataLoader(subtrainset, batch_size = args.batch_size,
                                                        shuffle=False)
    return subtrainloader