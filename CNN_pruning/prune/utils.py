import torch
import os
import random
import numpy as np
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from model.resnet_cifar10_manual import resnet20
from collections import OrderedDict
import timm
from model.resnet_imagenet import resnet50
import sys
import detectors
from model.mobilenet import mobilenet
IHTPATH = './Lagrangian-Heuristic'
sys.path.append(IHTPATH)


def model_factory(arch,dset_path,pretrained=True, force_model_path = False, model_path = '../WoodFisher/checkpoints/resnet20_cifar10.pt'):

    if arch == 'resnet20_cifar10':

        if not force_model_path:
            new_state_trained = torch.load('./checkpoints/resnet20_cifar10.pt', map_location=torch.device('cpu'))
        else:
            new_state_trained = torch.load(model_path, map_location=torch.device('cpu'))
        
        model = resnet20()
        if pretrained:
            model.load_state_dict(new_state_trained, strict=False)

        test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

        train_random_transforms=True

        if train_random_transforms:
            train_transform = transforms.Compose([
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                ])
        else:
            train_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ])

        train_dataset = datasets.CIFAR10(root=dset_path, train=True, download=True,transform=train_transform)
        test_dataset = datasets.CIFAR10(root=dset_path, train=False, download=True,transform=test_transform)

        criterion = torch.nn.functional.cross_entropy

        modules_to_prune = []
        for name, param in model.named_parameters():
            #print("name is {} and shape of param is {} \n".format(name, param.shape))
            layer_name,param_name = '.'.join(name.split('.')[:-1]),name.split('.')[-1]
            if param_name == 'bias':
                    continue
            if 'conv' in layer_name or 'fc' in layer_name:
                modules_to_prune.append(name)

        model.name = "resnet20_cifar10"
        return model,train_dataset,test_dataset,criterion,modules_to_prune

    elif arch == 'resnet50_cifar100':

        model = timm.create_model("resnet50_cifar100", pretrained=True)  
        
        # Apply the test transformation
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
    
        train_random_transforms = True
        if train_random_transforms:
            train_transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ])
        else:
            train_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ])
    
        # Load the CIFAR10 dataset
        train_dataset = datasets.CIFAR100(root=dset_path, train=True, download=True, transform=train_transform)
        test_dataset = datasets.CIFAR100(root=dset_path, train=False, download=True, transform=test_transform)
    
        # Define the criterion
        criterion = torch.nn.functional.cross_entropy
    
        # Identify modules to prune
        modules_to_prune = []
        for name, param in model.named_parameters():
            layer_name, param_name = '.'.join(name.split('.')[:-1]), name.split('.')[-1]
            if param_name == 'bias':
                continue
            if 'conv' in layer_name or 'fc' in layer_name:
                modules_to_prune.append(name)

        model.name = "resnet50_cifar100"
        return model, train_dataset, test_dataset, criterion, modules_to_prune

    elif arch == 'mobilenetv1':
        model = mobilenet()
        train_dataset,test_dataset = imagenet_get_datasets(dset_path)

        criterion = torch.nn.functional.cross_entropy

        modules_to_prune = []
        for name, layer in model.named_modules():
            if isinstance(layer, torch.nn.Conv2d) or isinstance(layer, torch.nn.Linear):
                modules_to_prune.append(name+'.weight')

        if pretrained:
            path = './checkpoints/MobileNetV1-Dense-STR.pth'
            state_trained = torch.load(path,map_location=torch.device('cpu'))['state_dict']
            new_state_trained = model.state_dict()
            for k in state_trained:
                key = k[7:]
                if key in new_state_trained:
                    new_state_trained[key] = state_trained[k].view(new_state_trained[key].size())
                else:
                    print('Missing key',key)
            model.load_state_dict(new_state_trained,strict=False)

        model.name = "MobileNet"
        return model,train_dataset,test_dataset,criterion,modules_to_prune
    
    elif arch == 'resnet50_cifar10':

        model = timm.create_model("resnet50_cifar10", pretrained=True)  # Set pretrained=False since you're loading local weights

        # Apply the test transformation
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
    
        train_random_transforms = True
        if train_random_transforms:
            train_transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ])
        else:
            train_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ])
    
        # Load the CIFAR10 dataset
        train_dataset = datasets.CIFAR10(root=dset_path, train=True, download=True, transform=train_transform)
        test_dataset = datasets.CIFAR10(root=dset_path, train=False, download=True, transform=test_transform)
    
        # Define the criterion
        criterion = torch.nn.functional.cross_entropy
    
        # Identify modules to prune
        modules_to_prune = []
        for name, param in model.named_parameters():
            layer_name, param_name = '.'.join(name.split('.')[:-1]), name.split('.')[-1]
            if param_name == 'bias':
                continue
            if 'conv' in layer_name or 'fc' in layer_name:
                modules_to_prune.append(name)

        model.name = "resnet50_cifar100"
        return model, train_dataset, test_dataset, criterion, modules_to_prune

    
    elif arch == 'resnet50_imagenet':
        model = resnet50()
        train_dataset,test_dataset = imagenet_get_datasets(dset_path)
        criterion = torch.nn.functional.cross_entropy

        modules_to_prune = []
        for name, param in model.named_parameters():
            
            #print("name is {} and shape of param is {} \n".format(name, param.shape))
            layer_name,param_name = '.'.join(name.split('.')[:-1]),name.split('.')[-1]
            if param_name == 'bias':
                    continue
            if 'conv' in layer_name or 'fc' in layer_name:
                modules_to_prune.append(name)
        if pretrained:
            
            if not force_model_path:
                path = './checkpoints/resnet50_imagenet1k_v1.pth'
            else:
                path = model_path

            #path = 'checkpoints/resnet50-19c8e357.pth'
            # path = 'checkpoints/resnet50-19c8e357.pth'
            
            state_trained = torch.load(path,map_location=torch.device('cpu'))['state_dict']

            # print(state_trained.keys())
            
            # print(state_trained.keys())
            new_state_trained = model.state_dict()
            for k in state_trained:
                key = k[7:]
                if key in new_state_trained:
                    new_state_trained[key] = state_trained[k].view(new_state_trained[key].size())
                else:
                    print('Missing key',key)
            model.load_state_dict(new_state_trained,strict=True)
            
            #model.load_state_dict(torch.load(path))
            model.name = "resnet50_imagenet"
            return model,train_dataset,test_dataset,criterion,modules_to_prune


def imagenet_get_datasets(data_dir):

    train_dir = os.path.join(data_dir, 'train')
    test_dir = os.path.join(data_dir, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])

    train_transform = [
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip()
    ]
    
    train_transform += [
        transforms.ToTensor(),
        normalize,
    ]
    train_transform = transforms.Compose(train_transform)

    train_dataset = datasets.ImageFolder(train_dir, train_transform)

    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])

    test_dataset = datasets.ImageFolder(test_dir, test_transform)

    return train_dataset, test_dataset

@torch.no_grad()
def get_pvec(model, params):
    state_dict = model.state_dict()
    return torch.cat([
        state_dict[p].reshape(-1) for p in params
    ])

@torch.no_grad()
def get_sparsity(model, params):
    pvec = get_pvec(model,params)
    return (pvec == 0).float().mean()

@torch.no_grad()
def get_blocklist(model,params,block_size):
    i_w = 0
    block_list = [0]
    state_dict = model.state_dict()
    for p in params:
        param_size = np.prod(state_dict[p].shape)
        if param_size <block_size:
            block_list.append(i_w+param_size)
        else:
            num_block = int(param_size/block_size)
            block_subdiag = list(range(i_w,i_w+param_size+1,int(param_size/num_block))) 
            block_subdiag[-1] = i_w+param_size
            block_list += block_subdiag   
        i_w += param_size
    return block_list

@torch.no_grad()
def set_pvec(w, model, params,device, nhwc=False):
    state_dict = model.state_dict()
    i = 0
    for p in params:
        count = state_dict[p].numel()
        if type(w) ==  torch.Tensor :
            state_dict[p] = w[i:(i + count)].reshape(state_dict[p].shape)
        else:
            state_dict[p] = torch.Tensor(w[i:(i + count)]).to(device).reshape(state_dict[p].shape)
        i += count
    model.load_state_dict(state_dict)


@torch.no_grad()
def zero_grads(model):
    for p in model.parameters():
        p.grad = None

def set_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

def compute_metrics(model, dataloader, criterion=None, device='cpu', memory_device = 'cpu', verbose=False, n_samples=100000, seed=42):
    torch.manual_seed(seed)
    correct = 0
    total = 0
    avg_loss = 0
    total_samples = 0
    
    # Move model to the specified device
    model.to(device)
    
    with torch.no_grad():
        for i, data in enumerate(dataloader):
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            
            # Calculate outputs by running images through the network
            outputs = model(images)
            
            # Compute accuracy
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Compute loss if criterion is provided
            if criterion is not None:
                loss = criterion(outputs, labels).item()
                avg_loss += loss
            
            if verbose and i % 10 == 0:
                print(f'Processed {total} samples, accuracy so far: {100 * correct / total:.2f}%')
            
            total_samples += labels.size(0)
            del images, labels, outputs
            
            if total_samples >= n_samples:
                break
    
    accuracy = 100 * correct / total
    avg_loss /= (i + 1) if criterion is not None else 1

    model.to(memory_device)
    if criterion is not None:
        return accuracy, avg_loss
    else:
        return accuracy

