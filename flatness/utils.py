import torch
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import torch.utils.data as udata
import torch.nn.functional as F
import random

from models import VGG16, ResNet20, ResNet56, PyramidNet, densenet121, Wide_ResNet


def accuracy(output, target, topk=(1,)):
  with torch.no_grad():
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
      correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
      res.append(correct_k.mul_(100.0 / batch_size))
  return res


class AverageMeter(object):
  """Computes and stores the average and current value"""
  def __init__(self):
    self.reset()

  def reset(self):
    self.val = 0
    self.avg = 0
    self.sum = 0
    self.count = 0

  def update(self, val, n=1):
    self.val = val
    self.sum += val * n
    self.count += n
    self.avg = self.sum / self.count

def get_model(model_name, dataset_name_name = 'CIFAR10'):
    '''return pre-*defined* model'''

    if dataset_name_name == 'CIFAR10':
        num_classes = 10
    else:
        num_classes = 100
    
    if model_name == 'vgg16':  
        return VGG16() 
    elif model_name == 'resnet20':  
        return ResNet20() 
    elif model_name == 'resnet56':
        return ResNet56(num_classes=num_classes) 

    elif model_name == 'pyramidnet':
        return PyramidNet(110, 270, num_classes)
    elif model_name == 'densenet':
        return densenet121(num_class=num_classes)
    elif model_name == 'wideresnet':
        return Wide_ResNet(28, 10, num_classes=num_classes)

    else:
        raise ValueError("wrong model name!")     
    


def get_data_loader(dataset_name, model, bs=128):
    '''return train and test dataloader'''
    # [normalization]

    if dataset_name == 'CIFAR10':
        print('CIFAR10 normalization')
        normalize = transforms.Normalize((0.4914, 0.4822, 0.4465),
                                        (0.2023, 0.1994, 0.2010))
    else:
        print('CIFAR100 normalization')
        normalize = transforms.Normalize((0.5071, 0.4867, 0.4408),
                                        (0.2675, 0.2565, 0.2761)) 

    # [transformation]
    if model == 'resnet20':
        print("RESNET20 TRANSFORMATIONS ARE USED.")
        transform = transforms.Compose([
                        transforms.CenterCrop(28),
                        transforms.ToTensor(),
                        normalize,
                    ])
    else:
        transform = transforms.Compose([
                        transforms.ToTensor(),
                        normalize,
                    ])            

    # [dataset_name]
    train_dataset = dsets.__dict__[dataset_name](root='dataset',
                                                train=True,
                                                transform=transform,
                                                download=True)

    test_dataset= dsets.__dict__[dataset_name](root='dataset',
                                                train=False,
                                                transform=transform,
                                                download=True)


    random.seed(31)
    indexes = random.sample( range(0, len(train_dataset)), bs*10 ) #number of batches
    subtrainset = torch.utils.data.Subset(train_dataset, indexes)


    train_loader = udata.DataLoader(dataset=subtrainset,
                                    batch_size=bs,
                                    shuffle=False,
                                    drop_last=False)

    test_loader  = udata.DataLoader(dataset=test_dataset,
                                batch_size=bs,
                                shuffle=False,
                                drop_last=False)
    

    return train_loader, test_loader



def get_model_eval(model, dataloader, device):

    model.eval()
    with torch.no_grad():
        loss, correct = 0, 0
        for (data,target) in dataloader:
            data, target = data.to(device), target.to(device)

            output = model(data)
            loss += F.cross_entropy(output, target, reduction='mean').item()
            _, predicted = torch.max(output.data, 1)
            correct += (predicted.to(device) == target).sum().item()
    
    loss = loss/ len(dataloader)
    error = ( 1 - correct/ len(dataloader.dataset) )* 100

    return loss, error