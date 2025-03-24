import torch
import torch.nn as nn
from collections import OrderedDict
import numpy as np

def extract_conv_layer(layer_string):
    """
    Extract the appropriate convolutional layer based on a numerical index in the layer string.
    Example: For 'model.1.3.weight', it returns '1.3' which can be used to access the Conv2d layer.
    """
    # Split the layer string by '.' to break down the components
    parts = layer_string.split('.')

    print(layer_string)
    
    # Search for common patterns like 'conv1', 'conv2', 'conv3', or 'downsample'
    if 'downsample' in layer_string:
        return 'downsample'
    elif 'conv1' in layer_string:
        return 'conv1'
    elif 'conv2' in layer_string:
        return 'conv2'
    elif 'conv3' in layer_string:
        return 'conv3'
    
    elif 'fc' in layer_string:
        return 'fc'
    
    # If the layer string has a numeric pattern like 'model.1.3.weight'
    # Return the indexed path as a string (e.g., '1.3')
    try:
        # Extract and return the second and third parts of the layer string if they are numbers
        if parts[1].isdigit() and parts[2].isdigit():
            return f"{parts[2]}"
    except IndexError:
        # Handle cases where the string doesn't have enough parts
        pass
    
    # Return None if no valid conv layer is found
    return None
    
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
        
def get_block_number_for_param(layerparams, param):
    # Initialize variables
    current_block = None
    block_counter = 0
    
    for i, layer_param in enumerate(layerparams):
        # Split the parameter string by dots
        parts = layer_param.split('.')
        # Extract the block identifier (e.g., 'layer1.0')
        block_identifier = '.'.join(parts[:2])
        
        # Increment block_counter when encountering a new block
        if block_identifier != current_block:
            block_counter += 1
            current_block = block_identifier
        
        # Check if the current layer_param matches the input param
        if layer_param == param:
            return block_counter
    
    # Return None if the param is not found in layerparams
    return None

def zero_grads(model):
    for param in model.parameters():
        param.grad = None

def get_pvec(model, params):
    vec = []
    for name, param in model.named_parameters():
        if name in params:
            vec.append(param.view(-1))
    return torch.cat(vec)

def set_pvec(vec, model, params, device):
    pointer = 0
    for name, param in model.named_parameters():
        if name in params:
            num_param = param.numel()
            param.data = vec[pointer:pointer + num_param].view_as(param).data.to(device)
            pointer += num_param


def get_blocks(model):
    
    block_list = []
    child_list = list(model.named_children())

    block_list.append(("",nn.Sequential(OrderedDict([
      ('conv1', child_list[0][1]),
      ('bn1', child_list[1][1]),
      ('relu', child_list[2][1])
      ]))))

    for i in range(3,6):
        for name, child in child_list[i][1].named_children():
            block_list.append((child_list[i][0]+"."+name,child))

    block_list.append(("",nn.Sequential(OrderedDict([('avgpool',child_list[6][1]),('fc',child_list[7][1])]))))
        
    return block_list

class FinalBlockWithFCOutput(nn.Module):
    def __init__(self, avgpool, fc):
        super(FinalBlockWithFCOutput, self).__init__()
        self.avgpool = avgpool
        self.fc = fc
        self.fc_output = None

    def forward(self, x):
        x = self.avgpool(x)
        self.fc_output = x.view(x.size(0), -1)
        return self.fc(self.fc_output)

def replace_func(model, block_number, new_block):
    # Case 1: ResNet50 (general ImageNet version)
    if model.name == "resnet50_imagenet":
        return replace_block_in_resnet50(model, block_number, new_block)

    elif model.name == "resnet50_cifar100":
        return replace_block_in_resnet50_cifar(model, block_number, new_block)
    
    # Case 2: ResNetCifar (CIFAR-10 ResNet variant)
    elif model.name == "resnet20_cifar10":
        return replace_block_in_resnet_cifar(model, block_number, new_block)

    # Case 3: MobileNetV1 (general ImageNet version)
    elif model.name == "MobileNet":
        return replace_block_in_mobilenet(model, block_number, new_block)

    # Default case (for unsupported models, raise an exception or handle gracefully)
    else:
        raise ValueError(f"Unsupported model name: {model.name}")


def replace_block_in_resnet50(model, block_number, new_block):
    block_count = 0
    
    # Conv1, bn1, relu, and maxpool
    if block_count == block_number:
        model.conv1 = new_block
        return model

    block_count += 1

    # Layer1 (3 bottleneck blocks)
    for i in range(3):
        if block_count == block_number:
            model.layer1[i] = new_block
            return model
        block_count += 1

    # Layer2 (4 bottleneck blocks)
    for i in range(4):
        if block_count == block_number:
            model.layer2[i] = new_block
            return model
        block_count += 1

    # Layer3 (6 bottleneck blocks)
    for i in range(6):
        if block_count == block_number:
            model.layer3[i] = new_block
            return model
        block_count += 1

    # Layer4 (3 bottleneck blocks)
    for i in range(3):
        if block_count == block_number:
            model.layer4[i] = new_block
            return model
        block_count += 1

    # Final block (avgpool and fc)
   
    if block_count == block_number:
        if isinstance(new_block, nn.Sequential):
            model.avgpool = new_block[0]
            model.fc = new_block[-1]
        else:
            raise ValueError("For the final block, new_block should be nn.Sequential containing avgpool and fc")
        return model
    block_count += 1
    raise ValueError(f"Block number {block_number} not found in the model")

def replace_block_in_mobilenet(model, block_number, new_block):
    """
    Replace the block at the given block_number with new_block in a MobileNet model.
    """
    # Access the model's Sequential layers
    block_count = 0
    
    # Iterate through the layers of the model
    for i, layer in enumerate(model.model):
        if block_count == block_number:
            model.model[i] = new_block  # Replace the entire Sequential block
            return model
        block_count += 1

    # Handle case where block_number is beyond the number of blocks in the model
    raise ValueError(f"Block number {block_number} not found in the MobileNet model")


def replace_block_in_resnet_cifar(model, block_number, new_block):
    block_count = 0
    child_list = list(model.named_children())
    
    # Skip initial_block
    block_count += 1

    for i in range(3, 6):  # Iterate over layer1, layer2, layer3
        layer = child_list[i][1]
        for name, child in layer.named_children():
            if block_count == block_number:
                setattr(layer, name, new_block)  # Replace the block
                return model
            block_count += 1

    # Check if it's the final_block (avgpool and fc)
    if block_count == block_number:
        if isinstance(new_block, nn.Sequential):
            model.avgpool = new_block[0]
            model.fc = new_block[2]
        else:
            raise ValueError("For the final block, new_block should be nn.Sequential containing avgpool and fc")
        return model
    
    raise ValueError(f"Block number {block_number} not found in the model")

def replace_block_in_resnet50_cifar(model, block_number, new_block):
    block_count = 0
    
    # Conv1, bn1, relu (no maxpool, Identity instead)
    if block_count == block_number:
        model.conv1 = new_block
        return model

    block_count += 1

    # Layer1 (3 bottleneck blocks)
    for i in range(3):
        if block_count == block_number:
            model.layer1[i] = new_block
            return model
        block_count += 1

    # Layer2 (4 bottleneck blocks)
    for i in range(4):
        if block_count == block_number:
            model.layer2[i] = new_block
            return model
        block_count += 1

    # Layer3 (6 bottleneck blocks)
    for i in range(6):
        if block_count == block_number:
            model.layer3[i] = new_block
            return model
        block_count += 1

    # Layer4 (3 bottleneck blocks)
    for i in range(3):
        if block_count == block_number:
            model.layer4[i] = new_block
            return model
        block_count += 1

    # Final block (avgpool and fc)
    if block_count == block_number:
        if isinstance(new_block, nn.Sequential):
            model.avgpool = new_block[0]
            model.fc = new_block[-1]
        else:
            raise ValueError("For the final block, new_block should be nn.Sequential containing avgpool and fc")
        return model

    block_count += 1
    raise ValueError(f"Block number {block_number} not found in the model")
    

def get_submodel(model, num_blocks, block_number):
    # Case 1: ResNet50 (general ImageNet version)
    if model.name == "resnet50_imagenet":
        return get_submodel_resnet50(model, num_blocks, block_number)

    if model.name == "MobileNet":
        return get_submodel_mobilenetv1(model, num_blocks, block_number)
        
    # Case 2: ResNetCifar (CIFAR-10 ResNet variant)
    elif model.name == "resnet20_cifar10":
        return get_submodel_resnet_cifar(model, num_blocks, block_number)

    # Case 3: resnet50_cifar10 (custom CIFAR-10 ResNet50 variant)
    elif model.name == "resnet50_cifar10" or model.name == "resnet50_cifar100":
        return get_submodel_resnet50_cifar(model, num_blocks, block_number)

    # Default case (for unsupported models, raise an exception or handle gracefully)
    else:
        raise ValueError(f"Unsupported model name: {model.name}")


def get_submodel_resnet50(model, num_blocks, block_number):
    block_list = []
    child_list = list(model.named_children())
    
    # If block_number is 0, include the initial block
    if block_number == 0:
        initial_block = nn.Sequential(OrderedDict([
            ('conv1', child_list[0][1]),
            ('bn1', child_list[1][1]),
            ('relu', child_list[2][1]),
            ('maxpool', child_list[3][1])
        ]))
        block_list.append(("initial_block", initial_block))
    
    block_count = 0
    # Count blocks in layer1, layer2, layer3, layer4
    total_blocks = sum(len(layer) for layer in [child_list[4][1], child_list[5][1], child_list[6][1], child_list[7][1]])
    
    # Iterate through layer1, layer2, layer3, layer4
    for layer_name, layer in list(model.named_children())[4:8]:
        for block_name, block in layer.named_children():
            full_block_name = f"{layer_name}_{block_name}"
            
            block_count += 1
            if block_count >= block_number:
                block_list.append((full_block_name, block))
            
            if block_count >= num_blocks + block_number - 1 and num_blocks < total_blocks:
                submodel = nn.Sequential(OrderedDict(block_list))
                return submodel
    
    # If all blocks are included, add the final block with fc_output
    if num_blocks >= total_blocks:
        final_block = nn.Sequential(OrderedDict([
        ('avgpool', model.avgpool),
        ('flatten', nn.Flatten()),
        ('fc', model.fc)]))
    
        # Add this final block to the block_list
        block_list.append(("final_block", final_block))
        
    submodel = nn.Sequential(OrderedDict(block_list))
    return submodel


def get_submodel_resnet_cifar(model, num_blocks, block_number):
    block_list = []
    child_list = list(model.named_children())
    
    # If block_number is 0, include the initial block
    if block_number == 0:
        initial_block = nn.Sequential(OrderedDict([
            ('conv1', child_list[0][1]),
            ('bn1', child_list[1][1]),
            ('relu', child_list[2][1])
        ]))
        initial_block.prune_flag = False
        initial_block.first_block = False
        block_list.append(("initial_block", initial_block))
    
    block_count = 0
    total_blocks = sum(len(child_list[i][1]) for i in range(3, 6))
    
    for i in range(3, 6):
        for name, child in child_list[i][1].named_children():
            block_name = child_list[i][0] + "_" + name  # Replace dot with underscore
            
            block_count += 1
            
            if block_count >= block_number:
                block_list.append((block_name, child))
            
            if block_count >= num_blocks + block_number - 1 and num_blocks < total_blocks:
                submodel = nn.Sequential(OrderedDict(block_list))
                return submodel

    # If all blocks are included, add the final block with fc_output
    if block_count >= min(total_blocks, block_number + num_blocks - 1):
        # Create an nn.Sequential block for the final block with avgpool and fc
        final_block = nn.Sequential(OrderedDict([
        ('avgpool', model.avgpool),
        ('flatten', nn.Flatten()),
        ('fc', model.fc)]))
    
        # Add this final block to the block_list
        block_list.append(("final_block", final_block))
    
    submodel = nn.Sequential(OrderedDict(block_list))
    
    return submodel


def get_submodel_resnet50_cifar(model, num_blocks, block_number):
    block_list = []
    child_list = list(model.named_children())

    # If block_number is 0, include the initial block
    if block_number == 0:
        initial_block = nn.Sequential(OrderedDict([
            ('conv1', child_list[0][1]),  # CIFAR-10 specific conv1 (3x3 kernel, stride 1)
            ('bn1', child_list[1][1]),
            ('act1', child_list[2][1]),   # relu
            ('maxpool', nn.Identity())    # No maxpool for CIFAR-10
        ]))
        block_list.append(("initial_block", initial_block))
    
    block_count = 0
    # Use a list to track the number of blocks in each layer, skipping non-Sequential layers
    layer_block_map = [
        len(layer) if isinstance(layer, nn.Sequential) else 0 
        for layer in [child_list[3][1], child_list[4][1], child_list[5][1], child_list[6][1], child_list[7][1]]
    ]
    
    # Total number of blocks in the model
    total_blocks = sum(layer_block_map)

    # Iterate through the layers
    for layer_idx, layer_num in enumerate([3, 4, 5, 6, 7]):
        layer_name = child_list[layer_num][0]
        layer_blocks = child_list[layer_num][1]
        
        if isinstance(layer_blocks, nn.Sequential):
            for block_name, block in layer_blocks.named_children():
                full_block_name = f"{layer_name}_{block_name}"  # Adjust naming convention

                block_count += 1
                if block_count >= block_number:
                    block_list.append((full_block_name, block))
                
                # Use max to ensure we handle cases where block_number + num_blocks exceeds total_blocks
                if block_count >= max(total_blocks, block_number + num_blocks - 1):
                    submodel = nn.Sequential(OrderedDict(block_list))
                    return submodel

    # Handle the case where the requested blocks include the final blocks (avgpool and fc)
    if block_count >= min(total_blocks, block_number + num_blocks - 1):
        final_block = nn.Sequential(OrderedDict([
            ('avgpool', model.global_pool), 
            ('fc', model.fc)  # Fully connected layer
        ]))
        block_list.append(("final_block", final_block))
    
    submodel = nn.Sequential(OrderedDict(block_list))
    return submodel



def get_submodel_vit(model, num_blocks, block_number):
    block_list = []
    child_list = list(model.named_children())

    block_count = 0
    total_blocks = len(child_list[1][1].layers)  # Assuming child_list[1] is the encoder with transformer layers


    # If block_number is 0, include the initial embedding block (e.g., patch embedding or conv_proj)
    if block_number == 0:
        initial_block = nn.Sequential(OrderedDict([
            ('conv_proj', child_list[0][1]),  # Typically the initial conv projection or patch embedding layer
            ('pos_embed', child_list[1][1].pos_embed)  # Positional embedding if it exists separately
        ]))
        initial_block.prune_flag = False
        initial_block.first_block = False
        block_list.append(("initial_block", initial_block))

    block_count += 1
    # Iterate through transformer encoder layers
    for i, (layer_name, encoder_layer) in enumerate(child_list[1][1].layers.named_children()):
        full_block_name = f"encoder_layer_{i}"
       
        if block_count >= block_number:
            # Keep the entire encoder block intact, including LayerNorm layers
            block_list.append((full_block_name, encoder_layer))
        
        if block_count >= num_blocks + block_number - 1 and num_blocks < total_blocks:
            submodel = nn.Sequential(OrderedDict(block_list))
            return submodel

        block_count += 1
        
    # If all blocks are included, add the final classification head
    if num_blocks >= total_blocks:
        final_block = nn.Sequential(OrderedDict([
            ('head', child_list[2][1])  # Assuming child_list[2] is the classification head
        ]))
        block_list.append(("final_block", final_block))

    submodel = nn.Sequential(OrderedDict(block_list))
    
    return submodel

def get_submodel_mobilenetv1(model, num_blocks, block_number):
    block_list = []
    child_list = list(model.model.named_children())  # Assuming the MobileNet model is wrapped in 'model'

    # If block_number is 0, include the initial block (conv, bn, relu)
    if block_number == 0:
        initial_block = nn.Sequential(OrderedDict([
            ('conv1', child_list[0][1]),
            ('bn1', child_list[0][1][1]),
            ('relu1', child_list[0][1][2])
        ]))
        block_list.append(("initial_block", initial_block))
    
    block_count = 0
    total_blocks = len(child_list)  # Total number of layers

    # Iterate through layers starting from the first depthwise conv block
    for idx, (layer_name, layer) in enumerate(child_list):
        if idx >= block_number:
            block_count += 1
            full_block_name = f"block_{idx}"
            block_list.append((full_block_name, layer))
        
        if block_count >= num_blocks:
            submodel = nn.Sequential(OrderedDict(block_list))
            return submodel

    # If all blocks are included, add the final pooling and FC layers
    final_block = nn.Sequential(OrderedDict([
        ('avgpool', model.model[14]),  # The AvgPool2d layer
        ('flatten', nn.Flatten()),  # Flatten before FC
        ('fc', model.fc)
    ]))
    
    block_list.append(("final_block", final_block))
    submodel = nn.Sequential(OrderedDict(block_list))
    return submodel
    
def get_submodel_blocks(submodel):
    block_list = []
    child_list = list(submodel.named_children())
    
    for name, child in child_list:
        if isinstance(child, nn.Sequential):
            block_list.append((name, child))
        else:
            block_list.append((name, nn.Sequential(child)))
    
    return block_list

def find_module(net, params_to_prune, name=''):

    children = list(net.named_children())
    if not children:
        if name+".weight" == params_to_prune:
            return True, net
        else:
            return False, None 
    for child_name, child in children:
        if name:
            output_flag, net = find_module(child, params_to_prune, name="{}.{}".format(name, child_name))
        else:
            output_flag, net = find_module(child, params_to_prune, name=child_name)
        if output_flag:
            return True, net
    return False, None

def find_all_module(net, params_to_prune, name='', prune_list = []):

    children = list(net.named_children())
    
    if not children:
        if name+".weight" in params_to_prune:
            prune_list.append(name+".weight")
            
    for child_name, child in children:
        if name:
            find_all_module(child, params_to_prune, name="{}.{}".format(name, child_name), prune_list=prune_list)
        else:
            find_all_module(child, params_to_prune, name=child_name, prune_list=prune_list)

    return prune_list

@torch.no_grad()
def get_pvec(model, params):
    state_dict = model.state_dict()
    return torch.cat([
        state_dict[p].reshape(-1) for p in params
    ])

def getinput_hook(self, module, input, output):
    self.input_buffer.append(input[0].detach())
    return

def MP_MN(w_var, groups, N):

    # Ensure w_var is a torch tensor
    if not isinstance(w_var, torch.Tensor):
        w_var = torch.tensor(w_var)
    
    # Iterate over each group
    for output_channel in range(w_var.shape[1]):
        for group in groups:
            # Extract the values from w_var for the current group
            group_values = w_var[group, output_channel]

            # Find the indices of the top N elements in absolute value
            _, top_indices = torch.topk(torch.abs(group_values), N)

            # Create a mask where only the top N indices are True
            mask = torch.zeros_like(group_values, dtype=torch.bool)
            mask[top_indices] = True

            # Set values not in the top N to zero
            group_values[~mask] = 0

            # Place the processed group back into w_var
            w_var[:, output_channel][group] = group_values
    
    return w_var.cpu().numpy()

def generate_M_sized_groups(d_in, k_h, k_w, M):
    overall_groups = set()
    groups = []
    
    num_vars = int(d_in * k_h * k_w)
    
    for i in range(num_vars):
        if i not in overall_groups:
            group_i = [i + k_h * k_w * j for j in range(M) if i + k_h * k_w * j < num_vars]
            if len(group_i) == M:
                groups.append(group_i)
                overall_groups.update(group_i)
    
    if not groups:
        raise ValueError(f"Unable to generate any valid groups of size {M} with the given parameters.")
    
    return groups