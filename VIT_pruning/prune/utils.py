import torch
import sys
import numpy as np
import os
import random
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from collections import OrderedDict
import json
import torch.distributed as dist
from models.ViT import VisionTransformer, ViT_L_16_Weights, ViT_B_16_Weights, ViT_H_14_Weights, _vision_transformer 
from torchvision.transforms._presets import ImageClassification, InterpolationMode 
from torchvision.datasets import (
    OxfordIIITPet,
    StanfordCars,
)    
import torch.nn as nn
from PIL import Image
from transformers import ViTFeatureExtractor

def flatten_tensor_list(tensors):
    flattened = []
    for tensor in tensors:
        flattened.append(tensor.view(-1))
    return torch.cat(flattened, 0)


def print_parameters(model):
    for name, param in model.named_parameters(): 
        print(name, param.shape)

def load_model(path, model):
    tmp = torch.load(path, map_location='cpu')
    if 'state_dict' in tmp:
        tmp = tmp['state_dict']
    if 'model' in tmp:
        tmp = tmp['model']
    for k in list(tmp.keys()):
        if 'module.' in k:
            tmp[k.replace('module.', '')] = tmp[k]
            del tmp[k]
    model.load_state_dict(tmp)


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

def imagenet_get_datasets_ViT(data_dir):
    train_dir = os.path.join(data_dir, 'train')
    test_dir = os.path.join(data_dir, 'val')
    
    # Apply ImageClassification preset transforms with crop size of 224 for both train and test datasets 
    transform = ImageClassification(
            crop_size=224)
    
    # Use the same transform for both training and testing datasets
    train_dataset = datasets.ImageFolder(train_dir, transform=transform)
    test_dataset = datasets.ImageFolder(test_dir, transform=transform)

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
def get_gvec(model, params):
    named_parameters = dict(model.named_parameters())
    return torch.cat([
        named_parameters[p].grad.reshape(-1) for p in params
    ])
@torch.no_grad()
def get_gvec1(model, params):
    named_parameters = dict(model.named_parameters())
    return torch.cat([
        named_parameters[p].grad1.reshape(named_parameters[p].grad1.shape[0],-1) for p in params
    ],dim=1)

@torch.no_grad()
def get_gps_vec(model, params):
    named_parameters = dict(model.named_parameters())
    return torch.cat([
        named_parameters[p].grad_sample.reshape(named_parameters[p].grad_sample.shape[0],-1) for p in params
    ],dim=1)
    
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


def compute_acc5(model,dataloader,device='cpu',verbose=False):
    correct = 0
    total = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    i = 0
    with torch.no_grad():
        for data in dataloader:
            i+=1
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            images=images
            labels=labels
            # calculate outputs by running images through the network
            outputs = model(images)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            if verbose and i%10 == 0:
                print(total,correct)

            del images,labels,outputs

    return 100 * correct / total


def model_factory(arch,dset_path,pretrained=True, force_model_path = False, model_path = '', viz_mode = False):

    if arch == 'vit_b_16_cifar10':
        if pretrained:
            weights = ViT_B_16_Weights.IMAGENET1K_V1  # Use ImageNet weights if available, adjust if CIFAR-10 specific weights are needed
        else:
            weights = None
    
        model = _vision_transformer(
            patch_size=16,
            num_layers=12,
            num_heads=12,
            hidden_dim=768,
            mlp_dim=3072,
            weights=weights,
            progress=True
        )

        train_transform = transforms.Compose(
                    [
                        transforms.RandomResizedCrop(
                            (224, 224),
                            scale=(0.08, 1),
                        ),
                        transforms.RandAugment(0, 9),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)),
                        transforms.RandomErasing(p=0),
                    ]
                )
        
        test_transform = transforms.Compose([
            transforms.Resize((224, 224)),            # Resize directly for test set
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))  # Normalize CIFAR-10 stats
        ])
    

        train_dataset = datasets.CIFAR10(root=dset_path, train=True, download=True, transform=train_transform)
        test_dataset = datasets.CIFAR10(root=dset_path, train=False, download=True, transform=test_transform)

        img, label = train_dataset[0]  # Access the first image from the training dataset
        print(f"Image size: {img.size()}")  # Print the size of the image tensor

        criterion = torch.nn.functional.cross_entropy
    
        modules_to_prune = []
        for name, param in model.named_parameters():
            layer_name, param_name = '.'.join(name.split('.')[:-1]), name.split('.')[-1]
            if param_name == 'bias':
                continue
            if 'conv' in layer_name or 'fc' in layer_name or 'encoder' in layer_name:
                modules_to_prune.append(name)
    
        model.name = 'vit_b_16_cifar10'
        return model, train_dataset, test_dataset, criterion, modules_to_prune

    elif arch == 'vit_b_16_cifar10_pretrain':
        
        model, train_dataset, test_dataset, criterion, modules_to_prune = get_vit_b_16_cifar10(pretrained=True, dset_path='./data')
        
        return model, train_dataset, test_dataset, criterion, modules_to_prune

    elif arch == 'vit_b_16_cifar100':
        if pretrained:
            weights = ViT_B_16_Weights.IMAGENET1K_V1  # Use ImageNet weights if available, adjust if CIFAR-100 specific weights are needed
        else:
            weights = None
    
        model = _vision_transformer(
            patch_size=16,
            num_layers=12,
            num_heads=12,
            hidden_dim=768,
            mlp_dim=3072,
            weights=weights,
            progress=True,
        )

    
        # Data augmentation and transformation for CIFAR-100
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop((224, 224), scale=(0.08, 1)),  # Resize + random crop for training
            transforms.RandAugment(0, 9),                              # Apply RandAugment
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5071, 0.4867, 0.4408),        # CIFAR-100 mean/std normalization
                                 std=(0.2675, 0.2565, 0.2761)),
            transforms.RandomErasing(p=0)  # Random Erasing (optional)
        ])
        
        test_transform = transforms.Compose([
            transforms.Resize((224, 224)),  # Resize to 224x224 for testing
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5071, 0.4867, 0.4408),        # CIFAR-100 mean/std normalization
                                 std=(0.2675, 0.2565, 0.2761))
        ])
    
        # Load CIFAR-100 dataset
        train_dataset = datasets.CIFAR100(root=dset_path, train=True, download=True, transform=train_transform)
        test_dataset = datasets.CIFAR100(root=dset_path, train=False, download=True, transform=test_transform)
    
        img, label = train_dataset[0]  # Access the first image from the training dataset
        print(f"Image size: {img.size()}")  # Print the size of the image tensor
    
        # Define the criterion
        criterion = torch.nn.functional.cross_entropy
    
        # Identify modules to prune
        modules_to_prune = []
        for name, param in model.named_parameters():
            layer_name, param_name = '.'.join(name.split('.')[:-1]), name.split('.')[-1]
            if param_name == 'bias':
                continue
            if 'conv' in layer_name or 'fc' in layer_name or 'encoder' in layer_name:
                modules_to_prune.append(name)
    
        model.name = 'vit_b_16_cifar100'
        return model, train_dataset, test_dataset, criterion, modules_to_prune
    
    elif arch == 'vit_l_16':

        # Check if pretrained weights should be used
        if pretrained:
            weights = ViT_L_16_Weights.IMAGENET1K_SWAG_LINEAR_V1  # or any other weights you prefer
        else:
            weights = None
    
        # Define the model architecture using _vision_transformer function
        model = _vision_transformer(
            patch_size=16,
            num_layers=24,
            num_heads=16,
            hidden_dim=1024,
            mlp_dim=4096,
            weights=weights,
            progress=True,
        )
    
        # Load the ImageNet datasets
        train_dataset, test_dataset = imagenet_get_datasets_ViT(dset_path)
    
        # Specify the loss criterion
        criterion = torch.nn.functional.cross_entropy
    
        # Identify the modules to prune
        modules_to_prune = []
        for name, param in model.named_parameters():
            layer_name, param_name = '.'.join(name.split('.')[:-1]), name.split('.')[-1]
            if param_name == 'bias':
                continue  # Skip bias parameters
    
            # Consider layers that are conv, fc, or encoder layers for pruning
            if 'conv' in layer_name or 'fc' in layer_name or 'encoder' in layer_name:
                modules_to_prune.append(name)
    
        # Set the model name
        model.name = 'vit_l_16'
    
        # Return model, datasets, criterion, and modules to prune
        return model, train_dataset, test_dataset, criterion, modules_to_prune

    elif arch == 'vit_b_16':

        # Check if pretrained weights should be used
        if pretrained:
            weights = ViT_B_16_Weights.IMAGENET1K_V1  # Use ImageNet weights if available, adjust if CIFAR-100 specific weights are needed
        else:
            weights = None
    
        model = _vision_transformer(
            patch_size=16,
            num_layers=12,
            num_heads=12,
            hidden_dim=768,
            mlp_dim=3072,
            weights=weights,
            progress=True,
        )

        # Load the ImageNet datasets
        train_dataset, test_dataset = imagenet_get_datasets_ViT(dset_path)
    
        # Specify the loss criterion
        criterion = torch.nn.functional.cross_entropy
    
        # Identify the modules to prune
        modules_to_prune = []
        for name, param in model.named_parameters():
            layer_name, param_name = '.'.join(name.split('.')[:-1]), name.split('.')[-1]
            if param_name == 'bias':
                continue  # Skip bias parameters
    
            # Consider layers that are conv, fc, or encoder layers for pruning
            if 'conv' in layer_name or 'fc' in layer_name or 'encoder' in layer_name:
                modules_to_prune.append(name)
    
        # Set the model name
        model.name = 'vit_l_16'
    
        # Return model, datasets, criterion, and modules to prune
        return model, train_dataset, test_dataset, criterion, modules_to_prune
    
    elif arch == 'vit_h_14':

        # Check if pretrained weights should be used
        if pretrained:
            weights = ViT_H_14_Weights.IMAGENET1K_SWAG_LINEAR_V1  # or any other weights you prefer
        else:
            weights = None
    
        # Define the model architecture using _vision_transformer function
        model = _vision_transformer(
            patch_size=14,
            num_layers=32,
            num_heads=16,
            hidden_dim=1280,
            mlp_dim=5120,
            weights=weights,
            progress=True,
             viz_mode = viz_mode
        )
    
        # Load the ImageNet datasets
        train_dataset, test_dataset = imagenet_get_datasets_ViT(dset_path)
    
        # Specify the loss criterion
        criterion = torch.nn.functional.cross_entropy
    
        # Identify the modules to prune
        modules_to_prune = []
        for name, param in model.named_parameters():
            layer_name, param_name = '.'.join(name.split('.')[:-1]), name.split('.')[-1]
            if param_name == 'bias':
                continue  # Skip bias parameters
    
            # Consider layers that are conv, fc, or encoder layers for pruning
            if 'conv' in layer_name or 'fc' in layer_name or 'encoder' in layer_name:
                modules_to_prune.append(name)
    
        # Set the model name
        model.name = 'vit_h_14'
    
        # Return model, datasets, criterion, and modules to prune
        return model, train_dataset, test_dataset, criterion, modules_to_prune


def get_vit_b_16_cifar10(pretrained=False, dset_path='./data', model_path='./checkpoints/vit_model_cifar10_2.pth'):
    """
    Initializes the Vision Transformer model for CIFAR-10 with optional pretrained weights,
    prepares the CIFAR-10 datasets, defines the loss criterion, and identifies modules to prune.

    Args:
        pretrained (bool): Whether to load pretrained weights.
        dset_path (str): Path to download/load the CIFAR-10 dataset.
        model_path (str): Path to the pretrained model's .pth file.

    Returns:
        model (nn.Module): Initialized Vision Transformer model.
        train_dataset (Dataset): CIFAR-10 training dataset.
        test_dataset (Dataset): CIFAR-10 test dataset.
        criterion (nn.Module): Loss function.
        modules_to_prune (list): List of modules to consider for pruning.
    """
    if pretrained:
        # Load the source model
        model_data = torch.load(model_path)
        source_state_dict = model_data['model'] if 'model' in model_data else model_data

        # Initialize the new state dict
        target_state_dict = {}

        # Collect data for concatenation of attention weights and biases
        attention_weights = {}
        attention_biases = {}

        # Initialize the new state dict
        target_state_dict = {}
        
        # Collect data for concatenation of attention weights and biases
        attention_weights = {}
        attention_biases = {}
        
        for source_key in source_state_dict.keys():
            if source_key == 'vit.embeddings.cls_token':
                target_key = 'class_token'  # Adjusted to match the target model's key
                target_state_dict[target_key] = source_state_dict[source_key]
            elif source_key == 'vit.embeddings.position_embeddings':
                target_key = 'encoder.pos_embedding'
                target_state_dict[target_key] = source_state_dict[source_key]
            elif source_key == 'vit.embeddings.patch_embeddings.projection.weight':
                target_key = 'conv_proj.weight'
                target_state_dict[target_key] = source_state_dict[source_key]
            elif source_key == 'vit.embeddings.patch_embeddings.projection.bias':
                target_key = 'conv_proj.bias'
                target_state_dict[target_key] = source_state_dict[source_key]
            elif source_key.startswith('vit.encoder.layer.'):
                # Extract layer number and the rest of the key
                rest = source_key[len('vit.encoder.layer.'):]
                layer_num_str, rest = rest.split('.', 1)
                layer_num = int(layer_num_str)
                rest_of_key = rest
                target_key_prefix = f'encoder.layers.encoder_layer_{layer_num}.'
        
                if rest_of_key == 'layernorm_before.weight':
                    target_key = target_key_prefix + 'ln_1.weight'  # Adjusted key
                    target_state_dict[target_key] = source_state_dict[source_key]
                elif rest_of_key == 'layernorm_before.bias':
                    target_key = target_key_prefix + 'ln_1.bias'  # Adjusted key
                    target_state_dict[target_key] = source_state_dict[source_key]
                elif rest_of_key == 'layernorm_after.weight':
                    target_key = target_key_prefix + 'ln_2.weight'  # Adjusted key
                    target_state_dict[target_key] = source_state_dict[source_key]
                elif rest_of_key == 'layernorm_after.bias':
                    target_key = target_key_prefix + 'ln_2.bias'  # Adjusted key
                    target_state_dict[target_key] = source_state_dict[source_key]
                elif rest_of_key == 'attention.attention.query.weight':
                    attention_weights.setdefault(layer_num, {})['query'] = source_state_dict[source_key]
                elif rest_of_key == 'attention.attention.key.weight':
                    attention_weights.setdefault(layer_num, {})['key'] = source_state_dict[source_key]
                elif rest_of_key == 'attention.attention.value.weight':
                    attention_weights.setdefault(layer_num, {})['value'] = source_state_dict[source_key]
                elif rest_of_key == 'attention.attention.query.bias':
                    attention_biases.setdefault(layer_num, {})['query'] = source_state_dict[source_key]
                elif rest_of_key == 'attention.attention.key.bias':
                    attention_biases.setdefault(layer_num, {})['key'] = source_state_dict[source_key]
                elif rest_of_key == 'attention.attention.value.bias':
                    attention_biases.setdefault(layer_num, {})['value'] = source_state_dict[source_key]
                elif rest_of_key == 'attention.output.dense.weight':
                    target_key = target_key_prefix + 'self_attention.out_proj.weight'  # Adjusted key
                    target_state_dict[target_key] = source_state_dict[source_key]
                elif rest_of_key == 'attention.output.dense.bias':
                    target_key = target_key_prefix + 'self_attention.out_proj.bias'  # Adjusted key
                    target_state_dict[target_key] = source_state_dict[source_key]
                elif rest_of_key == 'intermediate.dense.weight':
                    target_key = target_key_prefix + 'mlp.0.weight'  # Adjusted key
                    target_state_dict[target_key] = source_state_dict[source_key]
                elif rest_of_key == 'intermediate.dense.bias':
                    target_key = target_key_prefix + 'mlp.0.bias'  # Adjusted key
                    target_state_dict[target_key] = source_state_dict[source_key]
                elif rest_of_key == 'output.dense.weight':
                    target_key = target_key_prefix + 'mlp.3.weight'  # Adjusted key
                    target_state_dict[target_key] = source_state_dict[source_key]
                elif rest_of_key == 'output.dense.bias':
                    target_key = target_key_prefix + 'mlp.3.bias'  # Adjusted key
                    target_state_dict[target_key] = source_state_dict[source_key]
                else:
                    print(f"Unknown key in layer: {source_key}")
            elif source_key == 'vit.layernorm.weight':
                target_key = 'encoder.ln.weight'  # Adjusted key
                target_state_dict[target_key] = source_state_dict[source_key]
            elif source_key == 'vit.layernorm.bias':
                target_key = 'encoder.ln.bias'  # Adjusted key
                target_state_dict[target_key] = source_state_dict[source_key]
            elif source_key == 'classifier.weight':
                target_key = 'heads.head.weight'  # Corrected mapping
                target_state_dict[target_key] = source_state_dict[source_key]
            elif source_key == 'classifier.bias':
                target_key = 'heads.head.bias'  # Corrected mapping
                target_state_dict[target_key] = source_state_dict[source_key]
            else:
                print(f"Unknown key: {source_key}")

        # Concatenate query, key, and value weights and biases for attention
        for layer_num in attention_weights.keys():
            target_key_prefix = f'encoder.layers.encoder_layer_{layer_num}.self_attention.'
            qw = attention_weights[layer_num]['query']
            kw = attention_weights[layer_num]['key']
            vw = attention_weights[layer_num]['value']
            # Concatenate weights along dimension 0
            in_proj_weight = torch.cat([qw, kw, vw], dim=0)

            print(target_key_prefix + 'in_proj_module.in_proj_weight')
            target_state_dict[target_key_prefix + 'in_proj_module.in_proj_weight'] = in_proj_weight
            target_state_dict[target_key_prefix + 'in_proj_weight'] = in_proj_weight
        
            qb = attention_biases[layer_num]['query']
            kb = attention_biases[layer_num]['key']
            vb = attention_biases[layer_num]['value']
            # Concatenate biases along dimension 0
            in_proj_bias = torch.cat([qb, kb, vb], dim=0)

            target_state_dict[target_key_prefix + 'in_proj_bias'] = in_proj_bias
            target_state_dict[target_key_prefix + 'in_proj_module.in_proj_bias'] = in_proj_bias


        # Initialize the model
        model = _vision_transformer(
            patch_size=16,
            num_layers=12,
            num_heads=12,
            hidden_dim=768,
            mlp_dim=3072,
            weights=None,  # We'll load weights manually
            progress=True
        )

        # Adjust the classifier layer if needed
        if 'heads.head.weight' in target_state_dict:
            source_num_classes = target_state_dict['heads.head.weight'].shape[0]
            target_num_classes = model.heads.head.out_features

            if source_num_classes != target_num_classes:
                print(f"Adjusting classifier from {target_num_classes} to {source_num_classes} classes.")
                model.heads.head = nn.Linear(in_features=768, out_features=source_num_classes)
                # It's essential to initialize the new classifier layer's weights
                nn.init.trunc_normal_(model.heads.head.weight, std=.02)
                if model.heads.head.bias is not None:
                    nn.init.zeros_(model.heads.head.bias)
        else:
            print("Warning: 'heads.head.weight' not found in the converted state_dict.")

        # Load the state dict
        missing_keys, unexpected_keys = model.load_state_dict(target_state_dict, strict=False)
        if missing_keys:
            print("Missing keys:", missing_keys)
        if unexpected_keys:
            print("Unexpected keys:", unexpected_keys)

    else:
        # Initialize the model without pretrained weights
        model = _vision_transformer(
            patch_size=16,
            num_layers=12,
            num_heads=12,
            hidden_dim=768,
            mlp_dim=3072,
            weights=None,
            progress=True
        )

    # # Define transforms
    # train_transform = transforms.Compose([
    #     transforms.RandomResizedCrop(224, scale=(0.08, 1.0)),
    #     transforms.RandAugment(num_ops=2, magnitude=9),
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), 
    #                          std=(0.2023, 0.1994, 0.2010)),
    #     transforms.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3)),
    # ])

    # test_transform = transforms.Compose([
    #     transforms.Resize(224),
    #     transforms.CenterCrop(224),
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), 
    #                          std=(0.2023, 0.1994, 0.2010)),
    # ])

    # # Load datasets
    # train_dataset = datasets.CIFAR10(root=dset_path, train=True, download=True, transform=train_transform)
    # test_dataset = datasets.CIFAR10(root=dset_path, train=False, download=True, transform=test_transform)

    feature_extractor = ViTFeatureExtractor.from_pretrained('aaraki/vit-base-patch16-224-in21k-finetuned-cifar10')
    
    # Define a custom transform that uses the feature extractor
    def feature_extractor_transform(image):
        # Apply the feature extractor and return the transformed image as a tensor
        inputs = feature_extractor(images=image, return_tensors="pt")
        return inputs['pixel_values'].squeeze(0)  # Remove batch dimension
    
    # Create a wrapper transform to convert images to PIL before applying the extractor
    class FeatureExtractorTransform:
        def __call__(self, img):
            if not isinstance(img, Image.Image):
                img = Image.fromarray(img)
            return feature_extractor_transform(img)
    
    # Load the CIFAR-10 dataset with the custom transform
    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=FeatureExtractorTransform())
    test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=FeatureExtractorTransform())
    

    # Display first image size
    img, label = train_dataset[0]
    print(f"Image size: {img.size()}")  # Should be torch.Size([3, 224, 224])

    # Define the loss criterion
    criterion = nn.CrossEntropyLoss()

    # Identify modules to prune (Conv2d and Linear layers)
    modules_to_prune = []
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            modules_to_prune.append((module, 'weight'))

    # Assign a name to the model for identification
    model.name = 'vit_b_16_cifar10'

    return model, train_dataset, test_dataset, criterion, modules_to_prune
