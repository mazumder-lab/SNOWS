import torch
import torch.nn as nn
import numpy as np
import copy
import time
from torch.utils.data import DataLoader
import sys
IHTPATH = './Algs'
sys.path.append(IHTPATH)
from group_prunealg_vit import get_blocks, find_module, find_all_module, solve_for_W_given_Z, make_nm_sparse
from helpers import replace_block_in_vit, zero_grads, get_pvec, compute_metrics


class LayerPruner:

    def __init__(self, model, params, train_dataset, test_dataloader, nsamples, criterion, gradseed, device, algo, layers_to_prune):
        self.model = model
        self.params = params
        self.train_dataset = train_dataset
        self.test_dataloader = test_dataloader
        self.criterion = criterion
        self.nsamples = nsamples
        self.device = device
        self.algo = algo
        self.gradseed = gradseed
        self.layerparams = None  # To be set in get_params
        self.size_list = []
        self.layers_to_prune = layers_to_prune

    def update_model(self, new_w):
        set_pvec(new_w, self.model, self.params, self.device)

    def getinput_hook(self, module, input, output):
        self.input_buffer = []
        self.input_buffer.append(input[0].detach())

    def get_size(self, use_layer=True):
        size_list = []
        ignore_bias = True
        for name, param in self.model.named_parameters():
            param_name = name.split('.')[-1]
            if ignore_bias and param_name == 'bias':
                continue
            if use_layer:
                if self.layerparams and name not in self.layerparams:
                    continue
            else:
                if self.params and name not in self.params:
                    continue
            size_list.append(param.shape)
        self.size_list = size_list

    def get_params(self):
        if 'vit' in self.model.name:
            self.datasize = [3, 224, 224]
            self.layerparams = get_vit_layers(self.model, self.layers_to_prune)


    def prune_NM(self, N=2, M=4, newton_k_step=3, newton_steps = 5, batch_size = 128, max_CG_iterations = 30, batching=True, w_warm = False, device='cuda:0', save_memory=False):
        memory_device = 'cpu' if save_memory else device

        zero_grads(self.model)
        self.model.eval()
        self.get_params()
        self.get_size()

        original_weight = get_pvec(self.model, self.layerparams)
        w_layer = original_weight.to(memory_device).detach().cpu().numpy()
        w_prune = np.copy(w_layer)

        torch.manual_seed(self.gradseed)
        torch.cuda.manual_seed(self.gradseed)
        torch.cuda.manual_seed_all(self.gradseed)
        np.random.seed(self.gradseed)

        train_dataloader = DataLoader(self.train_dataset, batch_size=1, shuffle=True, num_workers=10, pin_memory=True)
        xdata = torch.zeros([self.nsamples] + self.datasize).to(memory_device)

        for i, batch in enumerate(train_dataloader):
            xdata_i, _ = batch
            xdata_i = xdata_i.to(memory_device)
            xdata[i] = xdata_i
            if (i + 1) % self.nsamples == 0:
                break

        xdata2 = copy.deepcopy(xdata)

        initial_accuracy, initial_loss = compute_metrics(self.model, self.test_dataloader, criterion=nn.CrossEntropyLoss(), device='cuda', memory_device=memory_device, verbose=False, n_samples=10000, seed=42)
        print("Initial Accuracy, Initial Loss")
        print(initial_accuracy)
        print(initial_loss)

        accuracies = {}
        losses = {}

        self.model.to(memory_device)
        model_update = copy.deepcopy(self.model)

        block_list = get_blocks(copy.deepcopy(self.model))
        total_blocks = len(block_list) - 1
        block_count = 0


        for name, block in block_list:
            block = block.to(memory_device)

            with torch.no_grad():
                prune_list = find_all_module(block, self.layerparams, name, [])

                if not prune_list:
                    if name == 'conv_proj':
                        if batching:
                            xdata = forward_pass_in_batches_single(block, xdata, batch_size, device='cuda')
                        else:
                            xdata = block(xdata)
                        n, hidden_dim, height, width = xdata.shape
                        seq_length = height * width
                        xdata = xdata.reshape(n, hidden_dim, seq_length).permute(0, 2, 1)
                        xdata2 = copy.deepcopy(xdata)
                    else:

                        if batching:
                            xdata, xdata2 = forward_pass_in_batches(block, block, xdata, xdata2, batch_size, device='cuda')
                        else:
                            xdata = block(xdata)
                            xdata2 = block(xdata2)
                        
                    block_count += 1
                    continue

            block_update = copy.deepcopy(block)

            forward_pass_in_batches_no_return(block_update, xdata, batch_size, device, xdata.size(0))

            for cur_i in range(len(prune_list)):

                cur_module_name = prune_list[cur_i]
                prune_flag, prune_module_update = find_module(block_update, cur_module_name, name)
                prune_flag, prune_module = find_module(block, cur_module_name, name)

                if w_warm:
                    hook_handle = prune_module_update.register_forward_hook(self.getinput_hook)
                    self.input_buffer = []
                    forward_pass_in_batches_no_return(block_update, xdata, batch_size, device, xdata.size(0))
                    hook_handle.remove()
    
                    
                    hook_handle = prune_module.register_forward_hook(self.getinput_hook)
                    self.input_buffer = []
                    forward_pass_in_batches_no_return(block, xdata2, batch_size, device, xdata.size(0))
                    hook_handle.remove()

                if "in_proj" in cur_module_name:
                    k = newton_k_step
                    w_var = prune_module.in_proj_weight
                elif "mlp" in cur_module_name:
                    k = 1 # force mlps to 1 
                    w_var = prune_module.weight
                elif "out_proj" in cur_module_name:
                    k = newton_k_step
                    w_var = prune_module.weight

                param_size = w_var.shape


                if self.algo == "MP":
                    prune_thirds = [0, 1, 2]
                    w_sol = make_nm_sparse(w_var, N, M, prune_thirds)

                elif self.algo == "SNOWS":

                    w_var = torch.tensor(w_var).to(device)
                    prune_thirds = [0, 1, 2]
                    w_sol = make_nm_sparse(w_var, N, M, prune_thirds)

                    mask = (torch.abs(w_sol) > 0).float()
                    total_params = torch.prod(torch.tensor(param_size)).item()

                    w_sol = solve_for_W_given_Z(
                        Z=mask,
                        model=self.model,
                        model_update=model_update,
                        xdata=xdata,
                        block_number=block_count,
                        k_step=newton_k_step,
                        total_blocks=total_blocks,
                        device=device,
                        N=N,
                        M=M,
                        w_warm=w_sol,
                        vit_layer=cur_module_name,
                        vit_name=name,
                        newton_steps=newton_steps,
                        block_list=block_list,
                        batch_size=batch_size,
                        CG_iterations = max_CG_iterations
                    )

                    model_update = model_update.to(memory_device)
                    self.model = self.model.to(memory_device)

                else:
                    w_sol = w_var

                w_sol = w_sol.to(memory_device)
    
                w_sol_shape = w_sol.shape
                start_data = time.time()

                state_dict = block_update.state_dict()

                if cur_module_name.startswith(name + "."):
                    param_local = copy.deepcopy(cur_module_name[len(name + "."):])
                else:
                    param_local = copy.deepcopy(cur_module_name)
                
                original_shape = state_dict[param_local].shape

                # Assign pruned weights back to the model
                state_dict[param_local].data = w_sol.data

                if cur_module_name.startswith(name + "."):
                    param_local = copy.deepcopy(cur_module_name[len(name + "."):])
                else:
                    param_local = copy.deepcopy(cur_module_name)
    
                original_shape = state_dict[param_local].shape
    
                param_value = torch.nn.Parameter(torch.Tensor(w_sol))
                
                module_name_parts = param_local.split('.')
                
                current_module = block_update
                for part in module_name_parts[:-1]:
                    current_module = getattr(current_module, part)
                
                setattr(current_module, module_name_parts[-1], param_value)
    
                if param_local == 'self_attention.in_proj_weight':
                    in_proj_module = getattr(current_module, 'in_proj_module')
                    setattr(in_proj_module, 'in_proj_weight', param_value)
                    block_update.self_attention.in_proj_weight = param_value
    
                elif 'mlp' in param_local:
                    mlp_idx = int(extract_mlp_number(cur_module_name))
                    block_update.mlp[mlp_idx].weight = param_value

                elif 'out_proj' in param_local:
                    block_update.self_attention.out_proj.weight = param_value
    
                model_update = replace_block_in_vit(model_update, block_count, block_update)

                model_state_dict = model_update.state_dict()

                model_state_dict[cur_module_name].data = w_sol.data
                
                # Optionally re-load the updated state_dict back into the model (if needed)
                model_update.load_state_dict(model_state_dict)

    
                # # Record OOS metrics for each element in prune_list
                accuracy, loss = compute_metrics(model_update, self.test_dataloader, criterion=nn.CrossEntropyLoss(), device='cuda:0', memory_device = memory_device, verbose=False, n_samples=10000, seed=42)

                # accuracy, loss = None, None

                if cur_module_name not in accuracies:
                    accuracies[cur_module_name] = []
                    losses[cur_module_name] = []


                accuracies[cur_module_name].append(accuracy)
                losses[cur_module_name].append(loss)

                print(f"Cur Module Name: {cur_module_name} | OOS Loss: {loss}, OOS Acc: {accuracy}")

            xdata, xdata2 = forward_pass_in_batches(block_update, block, xdata, xdata2, batch_size, device='cuda')
            block_count += 1


            # accuracy, loss = compute_metrics(model_update, self.test_dataloader, criterion=nn.CrossEntropyLoss(), device='cuda:0', memory_device=memory_device, verbose=False, n_samples=10000, seed=42)

            if name not in accuracies:
                accuracies[name] = []
                losses[name] = []

            accuracies[name].append(accuracy)
            losses[name].append(loss)

            print(f"Block Name: {name} | OOS Loss: {loss}, OOS Acc: {accuracy}")

        final_acc, final_loss = compute_metrics(model_update, self.test_dataloader, criterion=nn.CrossEntropyLoss(), device='cuda:0', memory_device=memory_device, verbose=False, n_samples=10000, seed=42)

        accuracies.append(final_acc)
        losses.append(final_loss)

        return model_update, accuracies, losses

        
def forward_pass_in_batches(block1, block2, xdata, xdata2, batch_size, device='cuda'):
    """
    Forward passes xdata through block1 and xdata2 through block2 in batches.
    
    Parameters:
    - block1: The first model/block to forward xdata through.
    - block2: The second model/block to forward xdata2 through.
    - xdata: The input tensor to be processed (on CPU).
    - xdata2: A copy of the input tensor to be processed (on CPU).
    - batch_size: The batch size to use for processing.
    - device: The device ('cpu' or 'cuda') to use for processing.
    
    Returns:
    - new_xdata: The processed xdata tensor (on CPU).
    - new_xdata2: The processed xdata2 tensor (on CPU).
    """
    new_xdata, new_xdata2 = None, None # Need to figure out shape first
    
    # Process xdata and xdata2 in smaller batches
    with torch.no_grad():
        for i in range(0, xdata.size(0), batch_size):
            # Create batches
            xdata_batch = xdata[i:i+batch_size].to(device)  # Move xdata batch to device
            xdata2_batch = xdata2[i:i+batch_size].to(device)  # Move xdata2 batch to device

            # Forward pass through block1 and block2
            xdata_batch_out = block1(xdata_batch)  # Forward pass xdata through block1
            xdata2_batch_out = block2(xdata2_batch)  # Forward pass xdata2 through block2

            # Initialize new_xdata and new_xdata2 based on the output size
            if new_xdata is None:
                new_xdata = torch.zeros((xdata.size(0), *xdata_batch_out.shape[1:])).to('cpu')
            if new_xdata2 is None:
                new_xdata2 = torch.zeros((xdata2.size(0), *xdata2_batch_out.shape[1:])).to('cpu')

            # Accumulate outputs into new_xdata and new_xdata2
            new_xdata[i:i+batch_size] = xdata_batch_out.to('cpu')
            new_xdata2[i:i+batch_size] = xdata2_batch_out.to('cpu')

    return new_xdata, new_xdata2

def forward_pass_in_batches_single(block, xdata, batch_size, device='cuda'):
    """
    Forward passes xdata through the provided block in batches.
    
    Parameters:
    - block: The model/block to forward xdata through.
    - xdata: The input tensor to be processed (on CPU).
    - batch_size: The batch size to use for processing.
    - device: The device ('cpu' or 'cuda') to use for processing.
    
    Returns:
    - new_xdata: The processed xdata tensor (on CPU).
    """
    new_xdata = None  # Initialize to None for later

    # Process xdata in smaller batches
    with torch.no_grad():
        for i in range(0, xdata.size(0), batch_size):
            # Create batches
            xdata_batch = xdata[i:i+batch_size].to(device)  # Move xdata batch to device
            
            # Forward pass through block
            xdata_batch_out = block(xdata_batch)  # Forward pass xdata through block

            # Initialize new_xdata based on the output size on first batch
            if new_xdata is None:
                new_xdata = torch.zeros((xdata.size(0), *xdata_batch_out.shape[1:])).to('cpu')

            # Accumulate outputs into new_xdata
            new_xdata[i:i+batch_size] = xdata_batch_out.to('cpu')

    return new_xdata

    
def forward_pass_in_batches_no_return(block, xdata, batch_size, device='cuda', max_examples=None):
    """
    Forward passes xdata through block in batches without returning the processed xdata.
    Ensures memory is freed after each batch is processed.
    
    Parameters:
    - block: The model/block to forward xdata through.
    - xdata: The input tensor to be processed (on CPU).
    - batch_size: The batch size to use for processing.
    - device: The device ('cpu' or 'cuda') to use for processing.
    - max_examples: The maximum number of examples to process (optional). If None, process all.
    
    Returns:
    None
    """
    total_examples = xdata.size(0)
    
    # If max_examples is specified, adjust total_examples accordingly
    if max_examples is not None:
        total_examples = min(total_examples, max_examples)
    
    # Process xdata in smaller batches
    with torch.no_grad():
        for i in range(0, total_examples, batch_size):
            # Create batches
            xdata_batch = xdata[i:i+batch_size].to(device)  # Move xdata batch to device
            
            # Forward pass through block
            block(xdata_batch)  # Forward pass xdata through block
            
            # Explicitly delete batch to free memory
            del xdata_batch
            torch.cuda.empty_cache()  # Clear GPU memory cache
            
            # If we've processed max_examples, exit the loop early
            if max_examples is not None and i + batch_size >= max_examples:
                break
    
    return None
    
def get_vit_layers(model, layers_to_prune = ['mlp', 'in_proj', 'out_proj']):
    layer_params = []
    for name, param in model.named_parameters():

        if 'mlp' in layers_to_prune:
            if "mlp" in name and "weight" in name:
                layer_params.append(name)

        if 'in_proj' in layers_to_prune:
            if 'in_proj_weight' in name:
                layer_params.append(name)

        if 'out_proj' in layers_to_prune:
            # Add check for out_proj layers
            if 'out_proj.weight' in name:
                layer_params.append(name)
            
    return layer_params


def extract_mlp_number(layer_string):
    # Split the string by '.'
    parts = layer_string.split('.')
    
    # Find the index of "mlp" in the split parts
    mlp_index = parts.index("mlp")
    
    # The MLP number is the element immediately after "mlp"
    mlp_number = parts[mlp_index + 1]
    
    return mlp_number

def calculate_model_sparsity(model):
    total_elements = 0
    zero_elements = 0
    
    for param in model.parameters():
        if param.requires_grad:  # Consider only trainable parameters
            num_elements = param.numel()  # Total number of elements
            num_zero_elements = torch.sum(param == 0).item()  # Number of zero elements

            total_elements += num_elements
            zero_elements += num_zero_elements
    
    # Compute sparsity level
    sparsity_level = zero_elements / total_elements
    
    return sparsity_level