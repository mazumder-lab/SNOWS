import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import copy
from collections import OrderedDict
from joblib import delayed, Parallel
import itertools
from helpers import get_submodel_vit, replace_block_in_vit
from torch.autograd import Variable, grad
import math
import torch.nn.functional as F
from models.ViT import InProjModule, MultiheadAttention
from torch.nn import GELU, LayerNorm
from torch.nn.modules.linear import NonDynamicallyQuantizableLinear
import psutil
import os
import gc
import time

def find_num_blocks_for_k_operations(blocks, start_block, cur_module, K):
    """Find the number of blocks needed to accumulate K operations, starting from cur_module in the first block."""
    total_operations = 0
    num_blocks = 0
    
    # Iterate over the blocks starting from start_block
    for block_idx in range(start_block, len(blocks)):  # Iterate over the provided list of blocks
        block = blocks[block_idx]  # Access the block from the list

        # If it's the first block, consider cur_module
        if block_idx == start_block:
            block_operations = count_operations_in_block(block, cur_module)
        else:
            block_operations = count_operations_in_block(block)
        
        total_operations += block_operations
        num_blocks += 1
        
        # Break if the accumulated operations exceed the target K
        if total_operations > K:
            break
    
    return num_blocks
    
def count_operations_in_block(block_tuple, cur_module=None):
    """Count the number of MultiheadAttention, MaskedLinear, GELU, and LayerNorm operations in a ViT block, starting from cur_module."""
    operation_count = 0
    count_flag = False  # Start counting once we hit cur_module
    
    # Extract the actual module from the tuple
    block_name, block = block_tuple
    
    for name, submodule in block.named_children():
        # Skip operations that occur before cur_module in the current block
        if cur_module is not None and not count_flag:
            if name == cur_module:
                count_flag = True  # Start counting after the current module
            continue
        
        # Count operations after the cur_module or if cur_module is None
        if count_flag or cur_module is None:
            # Check for MultiheadAttention and count it, but not its submodules
            if isinstance(submodule, MultiheadAttention):
                operation_count += 1  # Count the whole MultiheadAttention block
                print(f"Found MultiheadAttention: {submodule}")

            # Recursively count operations within Sequential blocks (e.g., MLP layers)
            elif isinstance(submodule, torch.nn.Sequential):
                for idx, layer in enumerate(submodule):
                    if isinstance(layer, (torch.nn.Linear, GELU, LayerNorm)):
                        operation_count += 1
                        print(f"Found {layer}")

    return operation_count


def register_hooks_on_vit_block(block, hook_list, outputs_list, cur_module=None, is_first_block=False, operation_count=0, K=0):
    """
    Register hooks on layers within a Vision Transformer block, 
    ignoring layers before cur_module if is_first_block is True.
    Stop when operation_count reaches K.
    """
    count_flag = not is_first_block  # Skip layers before cur_module only if it's the first block
    excluded_layers = (torch.nn.Dropout,)  # Exclude Dropout layers only

    for name, submodule in block.named_children():
        print(f"Processing: {name}, Current Module: {cur_module}")

        # If this is the first block and cur_module is provided, skip layers before cur_module
        if is_first_block and cur_module is not None and not count_flag:
            if name == cur_module:
                count_flag = True  # Start capturing hooks after reaching cur_module
                print(f"Reached cur_module: {cur_module}, now capturing subsequent layers.")

                hook_list.append(submodule.register_forward_hook(lambda m, i, o: capture_outputs_hook(m, i, o, outputs_list)))
                operation_count += 1

        # If count_flag is True, register hooks for subsequent layers
        if count_flag:
            # Register hook for MultiheadAttention
            if isinstance(submodule, MultiheadAttention):
                print(f"Hooked MultiheadAttention")
                hook_list.append(submodule.register_forward_hook(lambda m, i, o: capture_outputs_hook(m, i, o, outputs_list)))
                operation_count += 1

            # Traverse submodules for composite blocks (e.g., MLPBlock or Sequential)
            elif hasattr(submodule, 'named_children'):
                for child_name, child in submodule.named_children():
                    if isinstance(child, (torch.nn.Linear, GELU, LayerNorm)) and not isinstance(child, excluded_layers):
                        if operation_count > K:
                            break
                        print(f"Hooking child: {child_name}")
                        hook_list.append(child.register_forward_hook(lambda m, i, o: capture_outputs_hook(m, i, o, outputs_list)))
                        operation_count += 1

        # Stop when operation limit K is reached
        if operation_count > K:
            break

    print(f"Total operations hooked: {operation_count}")
    return operation_count


def capture_outputs_hook(module, input, output, outputs_list):
    """
    Capture the output of the hooked layer/module.
    If the output is a tuple (as with MultiheadAttention), extract the first element (the tensor).
    """
    if isinstance(output, tuple):
        print(f"Captured output from: {module.__class__.__name__} | Shape: {output[0].shape}")
        # Add the first element of the tuple to outputs_list, assuming that's the tensor
        outputs_list.append(output[0])
    elif isinstance(output, torch.Tensor):
        print(f"Captured output from: {module.__class__.__name__} | Shape: {output.shape}")
        outputs_list.append(output)
    else:
        print(f"Captured output from: {module.__class__.__name__} | Shape: Not a Tensor or Tuple")


                            

def print_memory_usage(message=""):
    """Print memory usage for debugging purposes."""
    print(f"[{message}] GPU memory allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
    print(f"[{message}] GPU memory reserved: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")
    print(f"[{message}] CPU memory used: {psutil.Process().memory_info().rss / 1024**2:.2f} MB")


def get_nested_attr(obj, attr):
    """Helper function to access nested attributes."""
    attributes = attr.split('.')
    for a in attributes:
        obj = getattr(obj, a)
    return obj

def extract_mlp_number(layer_string):
    # Split the string by '.'
    parts = layer_string.split('.')

    print(layer_string)
    # Find the index of "mlp" in the split parts
    mlp_index = parts.index("mlp")
    
    # The MLP number is the element immediately after "mlp"
    mlp_number = parts[mlp_index + 1]
    
    return mlp_number

def prune_and_forward(model_update, block_number, horizon, warm_w, mask, num_blocks=2, device='cuda:0', vit_layer="", vit_name=""):
    # Determine the appropriate submodel and replacement functions based on the model type
    get_submodel_func = get_submodel_vit  # Generic get_submodel function for ViT
    replace_func = replace_block_in_vit
    
    # Extract the specified submodel block
    sub_model_sparse = get_submodel_func(model_update, num_blocks, block_number)
    block_sp = sub_model_sparse[0]
    
    # Ensure `mask` is on the correct device
    mask = mask.to(device)

    _, mod = find_module(block_sp, vit_layer, vit_name)

    # Check if the vit_layer specifies 'mlp', 'in_proj', or 'out_proj' to decide the layer to prune
    if isinstance(mod, NonDynamicallyQuantizableLinear):
        # Apply warm weights and mask for out_proj layers
        mod.weight.data = warm_w * mask
        block_sp.self_attention.out_proj = mod
    
    elif isinstance(mod, torch.nn.Linear):
        # If 'mlp' is specified, handle MLP layers
        mod.weight.data = warm_w * mask
        setattr(block_sp, vit_name.split('.')[-1], mod)

    elif isinstance(mod, InProjModule):
        # Apply warm weights and mask for InProjModule
        mod.in_proj_weight.data = warm_w * mask
        block_sp.self_attention.in_proj_module = mod



    # Replace the block in the model with the updated block
    model_update = replace_func(model_update, block_number, block_sp)
    
    return block_sp, model_update


                            
def solve_for_W_given_Z(Z, 
                        model, 
                        model_update, 
                        xdata, 
                        block_number, 
                        k_step, 
                        total_blocks, 
                        device, 
                        N, 
                        M, 
                        w_warm, 
                        vit_layer, 
                        vit_name,
                        block_list,
                        print_results_every=10,
                        newton_steps=3,
                        CG_iterations=200,
                        batch_size=256):

    # Check and move model, model_update, and xdata to the device if necessary
    if not next(model.parameters()).is_cuda or next(model.parameters()).device != device:
        model.to(device)
    
    if not next(model_update.parameters()).is_cuda or next(model_update.parameters()).device != device:
        model_update.to(device)
        
    mask_output = Z.bool()

    # Functions for submodel extraction and replacement in ViT
    get_submodel_func = get_submodel_vit
    replace_func = replace_block_in_vit

    num_blocks = find_num_blocks_for_k_operations(block_list, block_number, vit_layer, k_step)

    mse_loss = nn.MSELoss()

    print("num blocks")
    print(num_blocks)

    # Process the vit_layer string to remove any unnecessary prefix
    vit_layer_parts = vit_layer.split('.')
    relative_vit_layer = '.'.join(vit_layer_parts[-1:])  # Keep only the last part, e.g., "self_attention.out_proj"

    sub_model_dense = get_submodel_func(model, num_blocks, block_number)
    sub_model_sparse = get_submodel_func(model_update, num_blocks, block_number)
    _, mod = find_module(sub_model_dense[0], vit_layer, vit_name)
    
    if isinstance(mod, InProjModule):
        module = "self_attention"
    elif isinstance(mod, NonDynamicallyQuantizableLinear):  # New case for out_proj layers
        module = "self_attention"
    elif isinstance(mod, torch.nn.Linear):
        module = "mlp"
    elif isinstance(mod, nn.Sequential) and any(isinstance(layer, torch.nn.Linear) for layer in mod):
        module = "mlp"
    else:
        module = "other"

    # Adjusted call to get_nested_attr and prune_and_forward
    block_sp, model_update = prune_and_forward(
        model_update=model_update,
        block_number=block_number,
        horizon=-1,
        warm_w=w_warm,
        mask=Z,  # This is the mask to be applied
        num_blocks=num_blocks,
        device=device,
        vit_layer=vit_layer,
        vit_name=vit_name
    )


    mu_t = float('inf')  # Initialize mu_t to a large value

    for i in range(newton_steps):

        print(xdata.size(0), batch_size)
        print(range(0, xdata.size(0) - batch_size, batch_size))
        for batch_start in range(0, xdata.size(0) - batch_size, batch_size):
            batch_end = batch_start + batch_size
            xdata_batch = xdata[batch_start:batch_end].to(device)

            # Reset outputs and hooks for each batch
            outputs_to_reconstruct = []
            dense_hook_handles = []
            sparse_hook_handles = []

            # Initialize the operation count for each batch
            operation_count = 0

            # Reattach hooks before each forward pass
            for j in range(num_blocks):
                block_dense = sub_model_dense[j]
                block_sparse = sub_model_sparse[j]

                if j == 0:
                    register_hooks_on_vit_block(block_dense, dense_hook_handles, outputs_to_reconstruct, cur_module=module, is_first_block=True, operation_count=operation_count, K=k_step)
                    operation_count = register_hooks_on_vit_block(block_sparse, sparse_hook_handles, outputs_to_reconstruct, cur_module=module, is_first_block=True, operation_count=operation_count, K=k_step)
                else:
                    register_hooks_on_vit_block(block_dense, dense_hook_handles, outputs_to_reconstruct, is_first_block=False, operation_count=operation_count, K=k_step)
                    operation_count = register_hooks_on_vit_block(block_sparse, sparse_hook_handles, outputs_to_reconstruct, is_first_block=False, operation_count=operation_count, K=k_step)

                print("Op count")
                print(operation_count)

                if operation_count > k_step:
                    break

            # Forward pass through dense and sparse models with current batch
            outputs_to_reconstruct.clear()
            _ = sub_model_dense(xdata_batch)
            ydata_dense = [output.to(device, non_blocking=True) for output in outputs_to_reconstruct]

            outputs_to_reconstruct.clear()
            _ = sub_model_sparse(xdata_batch)
            ydata_pruned = [output.to(device, non_blocking=True) for output in outputs_to_reconstruct]

            # Ensure min_length is non-zero
            min_length = min(len(ydata_dense), len(ydata_pruned))
            if min_length == 0:
                raise ValueError("No outputs were captured. Ensure hooks are correctly attached and used in every batch.")

            loss_pruned = sum(mse_loss(ydata_dense[i], ydata_pruned[i]) for i in range(k_step+1))
            
            # Perform Newton step
                        # Added case for out_proj layers
            if isinstance(mod, NonDynamicallyQuantizableLinear):
                _, sparse_out_proj = find_module(sub_model_sparse[0], vit_layer, vit_name)
                W = sparse_out_proj.weight
            
            elif isinstance(mod, torch.nn.Linear):
                _, sparse_mlp = find_module(sub_model_sparse[0], vit_layer, vit_name)
                W = sparse_mlp.weight
            
            elif isinstance(mod, InProjModule):
                _, sparse_in_proj = find_module(sub_model_sparse[0], vit_layer, vit_name)
                W = sparse_in_proj.in_proj_weight


            grad_W = torch.autograd.grad(loss_pruned, W, create_graph=True, retain_graph=True)[0]
            b = -grad_W[mask_output]
            full_newton_step = torch.zeros_like(W, device='cpu')
            newton_step = conjugate_gradient_sparse(lambda v: hessian_vector_product_chunks(grad_W, W, v, mask_output), b, max_iter=CG_iterations)
            full_newton_step[mask_output] = newton_step.to('cpu', non_blocking=True)
            full_newton_step = full_newton_step.to(device, non_blocking=True)

            # Perform Armijo Backtracking
            last_batch_xdata = xdata[xdata.size(0) - batch_size:xdata.size(0)].to(device)
            outputs_to_reconstruct.clear()
            _ = sub_model_dense(last_batch_xdata)
            last_batch_ydata_dense = [output.to(device, non_blocking=True) for output in outputs_to_reconstruct]

            outputs_to_reconstruct.clear()
            _ = sub_model_sparse(last_batch_xdata)
            last_batch_ydata_pruned = [output.to(device, non_blocking=True) for output in outputs_to_reconstruct]

            last_batch_loss_pruned = sum(mse_loss(last_batch_ydata_dense[i], last_batch_ydata_pruned[i]) for i in range(k_step+1))

            alpha = 1.0
            c = 1e-5
            max_backtracking_steps = 50

            W_original = W.clone()

            for bt in range(max_backtracking_steps):
                W_new = W_original + alpha * full_newton_step

                with torch.no_grad():
                    W.data = W_new

                outputs_to_reconstruct.clear()
                _ = sub_model_sparse(last_batch_xdata)
                last_batch_ydata_pruned = [output.to(device, non_blocking=True) for output in outputs_to_reconstruct]

                last_batch_loss_pruned_new = sum(mse_loss(last_batch_ydata_dense[i], last_batch_ydata_pruned[i]) for i in range(k_step+1)).item()

                print(last_batch_loss_pruned_new, last_batch_loss_pruned + c * alpha * torch.dot(grad_W.flatten(), full_newton_step.flatten()))
                if last_batch_loss_pruned_new <= last_batch_loss_pruned + c * alpha * torch.dot(grad_W.flatten(), full_newton_step.flatten()):
                    break
                else:
                    alpha *= 0.9

                if bt == max_backtracking_steps - 1:
                    with torch.no_grad():
                        W.data = W_original

            model_update = replace_func(model_update, block_number, sub_model_sparse[0])

            # Detach hooks for this batch
            for handle in dense_hook_handles + sparse_hook_handles:
                handle.remove()

            del full_newton_step, last_batch_xdata, last_batch_ydata_dense, last_batch_ydata_pruned
            torch.cuda.empty_cache()

        # Detach hooks after all batches
        for handle in dense_hook_handles + sparse_hook_handles:
            handle.remove()

        if alpha < 0.1:
            break

    sub_model_sparse = get_submodel_func(model_update, num_blocks, block_number)
    block_sp = sub_model_sparse[0]

    if isinstance(mod, NonDynamicallyQuantizableLinear):
        # Apply the mask to the out_proj layer
        weight_to_return = mod.weight.data * Z
    elif isinstance(mod, torch.nn.Linear):
        mlp_idx = int(extract_mlp_number(vit_layer))
        weight_to_return = block_sp.mlp[mlp_idx].weight.data * Z
    elif isinstance(mod, InProjModule):
        weight_to_return = mod.in_proj_weight * Z


    return weight_to_return

def conjugate_gradient_sparse(hvp_fn, b, tol=5e-4, max_iter=1000, lambda_reg=1e-4, window_size=100):
    # Initialize x to be zero everywhere
    x = torch.zeros_like(b)

    # Restrict r, p to non-zero elements of b
    r = b.clone()
    p = r.clone()

    rs_old = torch.sum(r * r)
    
    residuals = []  # List to store residuals for rolling mean calculation
    
    for i in range(max_iter):
        # Only compute Hessian-vector product on the sparse p
        Ap = hvp_fn(p) + lambda_reg * p  # Add Tikhonov regularization term
        
        alpha = rs_old / torch.sum(p * Ap)

        # Update x only at the masked locations
        x = x + alpha * p

        # Update residual
        r = r - alpha * Ap
        rs_new = torch.sum(r * r)

        # Calculate the current residual (root mean square of rs_new)
        residual = torch.sqrt(rs_new).item()
        residuals.append(residual)

        # Maintain a rolling window of residuals
        if len(residuals) > window_size:
            residuals.pop(0)
        
        # Calculate and print the rolling mean of residuals
        rolling_mean = sum(residuals) / len(residuals)

        if i % 10 == 0:
            print(residual)
        
        if residual < tol:
            break

        # Update p
        p = r + (rs_new / rs_old) * p
        rs_old = rs_new

        # Delete intermediate tensors to free memory
        del Ap, alpha, rs_new
        torch.cuda.empty_cache()

    # Delete remaining large tensors after loop to free memory
    del r, p, rs_old, residuals
    torch.cuda.empty_cache()

    return x


def hessian_vector_product_chunks(grad_W, W, vector, mask, max_chunk_size=5e4):
    device = W.device  # Get the device of W (either CPU or CUDA)
    
    vector = Variable(vector).to(device)
    full_vector = torch.zeros_like(W, device=device)
    full_vector[mask] = vector

    # Initialize an empty tensor to accumulate the results
    hvp = torch.zeros_like(W, device=device)

    # Calculate the total number of elements
    num_elements = mask.sum().item()

    # Determine the number of chunks based on max_chunk_size
    num_chunks = int(max(1, (num_elements + max_chunk_size - 1) // max_chunk_size))  # Number of chunks
    
    chunk_size = (num_elements + num_chunks - 1) // num_chunks  # Calculate chunk size

    for i in range(num_chunks):  # Loop over each chunk
        # Determine the range for the current chunk
        start = i * chunk_size
        end = min(start + chunk_size, num_elements)

        # Create a chunk mask on the same device as W
        chunk_mask = mask.clone().to(device)
        chunk_mask[mask] = torch.arange(num_elements, device=device) >= start
        chunk_mask[mask] = torch.arange(num_elements, device=device) < end

        # Compute the Hessian-vector product for this chunk
        chunk_hvp = torch.autograd.grad(
            torch.sum(grad_W * full_vector * chunk_mask), W, retain_graph=True
        )[0]

        # Accumulate the result in the corresponding positions
        hvp[chunk_mask] += chunk_hvp[chunk_mask]

    # Return only the elements corresponding to the mask
    return hvp[mask]

def hessian_vector_product(grad_W, W, vector, mask):
    # I use the chunked version now which 
    # Create a zero tensor of the same shape as W

    vector = Variable(vector)
    full_vector = torch.zeros_like(W)
    
    # Populate full_vector only at the masked locations
    full_vector[mask] = vector
    
    # Compute the Hessian-vector product using the full_vector
    hvp = torch.autograd.grad(torch.sum(grad_W * full_vector), W, retain_graph=False)[0]
    
    # Return only the elements corresponding to the mask
    return hvp[mask]



def find_module(net, params_to_prune, name=''):
    children = list(net.named_children())

    if not children:
        # Check if this is an InProjModule and the parameter is part of its name
        if isinstance(net, InProjModule) and params_to_prune.endswith(".in_proj_weight"):
            return True, net
        
        # Generic case for direct weight parameters
        if name + ".weight" == params_to_prune:
            return True, net
        else:
            return False, None 
    
    for child_name, child in children:
        # Build the full name for the current child
        if name:
            full_name = "{}.{}".format(name, child_name)
        else:
            full_name = child_name
        
        # Recursively call find_module on the child module
        output_flag, module = find_module(child, params_to_prune, name=full_name)
        if output_flag:
            return True, module

    return False, None


def find_all_module(net, params_to_prune, name='', prune_list=[]):
    children = list(net.named_children())

    # Check if this module has parameters that match params_to_prune
    for param_name, _ in net.named_parameters(recurse=False):
        full_param_name = f"{name}.{param_name}" if name else param_name
        if full_param_name in params_to_prune:
            prune_list.append(full_param_name)

    # Specifically handle InProjModule if it's encountered
    if isinstance(net, InProjModule):
        in_proj_weight_name = f"{name}.in_proj_weight" if name else "in_proj_weight"

        if in_proj_weight_name in params_to_prune:
            prune_list.append(in_proj_weight_name)
            
    # Specifically handle the out_proj layer
    if isinstance(net, NonDynamicallyQuantizableLinear):
        out_proj_weight_name = f"{name}.out_proj.weight" if name else "out_proj.weight"
        
        if out_proj_weight_name in params_to_prune:
            prune_list.append(out_proj_weight_name)

    # Recursively search through the children of the module
    for child_name, child in children:
        child_full_name = f"{name}.{child_name}" if name else child_name
        find_all_module(child, params_to_prune, name=child_full_name, prune_list=prune_list)

    return prune_list


def get_blocks(model):
    
    if 'vit' in model.name:

        block_list = []
        
        child_list = list(model.named_children())
        
        block_list.append(('conv_proj', child_list[0][1]))
        
        for i, layer in enumerate(child_list[1][1].layers):
            block_list.append((f'encoder.layers.encoder_layer_{i}', layer))
        
        block_list.append(('encoder.ln', child_list[1][1].ln))
        block_list.append(('heads', child_list[2][1]))
        
    return block_list  


def make_nm_sparse(weight, N, M, prune_thirds=[0, 1, 2]):
    """
    Makes the given 2D weight matrix N:M sparse along the row dimension for the specified thirds of rows.

    Note the thirds is useful because transformer weights are stacked into third (Q, K, V) so e.g [0, 1] will prune W_Q, W_K but not W_V.
    
    Args:
        weight (torch.Tensor): The 2D weight matrix to be sparsified.
        N (int): The number of non-zero elements to retain in each M block.
        M (int): The size of each block along the row dimension.
        prune_thirds (list): List of indices specifying which thirds of rows to prune. 
                             For example, [0, 1] prunes the first and second thirds of the rows.
        
    Returns:
        torch.Tensor: The pruned weight matrix with N:M sparsity applied.
    """
    assert weight.dim() == 2, "Weight matrix must be 2D."
    assert M > N, "M should be greater than N to achieve sparsity."
    assert all(third in [0, 1, 2] for third in prune_thirds), "prune_thirds must contain indices 0, 1, or 2."

    num_rows, num_cols = weight.size()
    third_size = num_rows // 3
    
    # Initialize a mask of ones (everything is initially retained)
    mask = torch.ones_like(weight)

    # Apply N:M sparsity to each specified third of rows
    for third in prune_thirds:
        start_row = third * third_size
        end_row = start_row + third_size

        for i in range(start_row, end_row):
            # Extract the entire row to prune
            row = weight[i].detach().abs()
            num_cols_to_prune = num_cols
            group_size = num_cols_to_prune // M
            remaining_cols = num_cols_to_prune % M

            # Reshape it into blocks of size M
            row_blocks = row[:group_size * M].reshape(-1, M)
            
            # Find the smallest values in each block that should be zeroed out
            _, indices = torch.topk(row_blocks, M - N, dim=1, largest=False)
            
            # Zero out the smallest values in each block
            mask_blocks = torch.ones_like(row_blocks)
            mask_blocks.scatter_(1, indices, 0)
            
            # Flatten the blocks back into a single row
            mask[i, :group_size * M] = mask_blocks.flatten()
            
            # If there are any remaining columns (not fitting into a full block)
            if remaining_cols > 0:
                # Sort the remaining part of the row and zero out the smallest elements if needed
                remaining_row = row[group_size * M:]
                _, remaining_indices = torch.topk(remaining_row, max(0, remaining_cols - N), largest=False)
                mask[i, group_size * M:group_size * M + remaining_cols].scatter_(0, remaining_indices, 0)
    
    # Apply the mask to the weight matrix
    return weight * mask

