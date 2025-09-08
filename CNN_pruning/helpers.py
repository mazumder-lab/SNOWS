import torch
import torch.nn as nn
from collections import OrderedDict
import numpy as np

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
        
def replace_block_in_vit(model, block_number, new_block):
    block_count = 0

    # Replace the initial patch embedding or convolutional projection layer
    if block_number == block_count:
        model.conv_proj = new_block.conv_proj
        model.pos_embed = new_block.pos_embed
        return model
    
    block_count += 1

    # Replace transformer encoder layers
    for i in range(len(model.encoder.layers)):
        if block_count == block_number:
            model.encoder.layers[i] = new_block
            return model
        block_count += 1

    # Replace the final classification head
    if block_count == block_number:
        if isinstance(new_block, nn.Sequential):
            model.head = new_block.head
        else:
            raise ValueError("For the final block, new_block should be an nn.Sequential containing the head.")
        return model

    raise ValueError(f"Block number {block_number} not found in the model")


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


# def get_submodel_blocks(submodel):
#     block_list = []
#     child_list = list(submodel.named_children())
    
#     for name, child in child_list:
#         if isinstance(child, nn.Sequential):
#             block_list.append((name, child))
#         else:
# #             block_list.append((name, nn.Sequential(child)))
    
# #     return block_list

# def find_module(net, params_to_prune, name=''):

#     children = list(net.named_children())
#     if not children:
#         if name+".weight" == params_to_prune:
#             return True, net
#         else:
#             return False, None 
#     for child_name, child in children:
#         if name:
#             output_flag, net = find_module(child, params_to_prune, name="{}.{}".format(name, child_name))
#         else:
#             output_flag, net = find_module(child, params_to_prune, name=child_name)
#         if output_flag:
#             return True, net
#     return False, None

# def find_all_module(net, params_to_prune, name='', prune_list = []):

#     children = list(net.named_children())
    

#     if not children:
#         if name+".weight" in params_to_prune:
#             prune_list.append(name+".weight")
            
#     for child_name, child in children:
#         if name:
#             find_all_module(child, params_to_prune, name="{}.{}".format(name, child_name), prune_list=prune_list)
#         else:
#             find_all_module(child, params_to_prune, name=child_name, prune_list=prune_list)

#     return prune_list

@torch.no_grad()
def get_pvec(model, params):
    state_dict = model.state_dict()
    return torch.cat([
        state_dict[p].reshape(-1) for p in params
    ])

def getinput_hook(self, module, input, output):
    self.input_buffer.append(input[0].detach())
    return
