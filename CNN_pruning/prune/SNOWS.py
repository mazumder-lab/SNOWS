import numpy as np
import torch
import torch.nn as nn
from collections import OrderedDict
from joblib import delayed, Parallel
import itertools
from helpers import get_submodel, replace_func
from torch.autograd import Variable, grad

def count_operations_in_block(block_tuple, cur_module=None):
    """Count the number of Conv2d, ReLU, Linear operations, and downsample layers in a block, starting from cur_module."""
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
        
        # Count operations after cur_module or if cur_module is None
        if count_flag or cur_module is None:
            if isinstance(submodule, (torch.nn.Conv2d, torch.nn.ReLU)):
                operation_count += 1

            # Check for downsample layers and count Conv2d inside downsample
            if hasattr(submodule, 'downsample') and isinstance(submodule.downsample, torch.nn.Sequential):
                for downsample_name, downsample_module in submodule.downsample.named_children():
                    if isinstance(downsample_module, torch.nn.Conv2d):
                        operation_count += 1

    return operation_count


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
        
        if total_operations > K:
            break
    
    return num_blocks


def capture_outputs_hook(module, input, output, output_list):
    """Hook function to capture outputs of layers and print the module type."""
    # Print the type of the module where the hook is registered
    # Append the output to the output list
    output_list.append(output)


def register_hooks_on_basicblock(block, hook_list, outputs_list, cur_module=None, is_first_block=False, operation_count=0, K=0):
    """Register hooks on layers within a BasicBlock, including downsample layers.
       Ignore layers before cur_module if is_first_block is True. Stop when operation_count reaches K."""
    count_flag = not is_first_block  # Skip layers before cur_module only if it's the first block

    for name, submodule in block.named_children():
        # If this is the first block and cur_module is provided, skip layers before cur_module
        if is_first_block and cur_module is not None and not count_flag:
            if name == cur_module:
                count_flag = True  # Start capturing hooks after reaching cur_module
            # continue

        # Register hooks on Conv2d, ReLU, Linear, and downsample layers
        if count_flag:
            if isinstance(submodule, (torch.nn.Conv2d, torch.nn.ReLU)):
                # Stop registering hooks if operation_count has reached K

                if operation_count > K:
                    break

                hook_list.append(submodule.register_forward_hook(lambda m, i, o: capture_outputs_hook(m, i, o, outputs_list)))
                operation_count += 1

            # Check for downsample layers and register hooks on Conv2d inside downsample
            if hasattr(submodule, 'downsample') and isinstance(submodule.downsample, torch.nn.Sequential):
                for downsample_name, downsample_module in submodule.downsample.named_children():
                    if isinstance(downsample_module, torch.nn.Conv2d):
                        hook_list.append(downsample_module.register_forward_hook(lambda m, i, o: capture_outputs_hook(m, i, o, outputs_list)))
                        operation_count += 1
                        if operation_count > K:
                            break

    return operation_count  # Return the updated operation count


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
                        conv_layer, 
                        block_list,
                        print_results_every=10,
                        newton_steps=3,
                        CG_iterations=500,
                        batch_size=1000,
                        batching=True,
                        stagnation_threshold=0.995,
                        max_steps = np.inf):  

    # Move model, model_update, and xdata to the device
    model = model.to(device)
    model_update = model_update.to(device)

    num_blocks = find_num_blocks_for_k_operations(block_list, block_number, conv_layer, k_step)
    
    # Prepare submodels containing only the necessary blocks
    mse_loss = nn.MSELoss()
    sub_model_sparse = get_submodel(model_update, num_blocks, block_number)
    sub_model_dense = get_submodel(model, num_blocks, block_number)

    if 'fc' in conv_layer:
        mod_num = -1
    else:
        mod_num = 0

    if not isinstance(w_warm, torch.Tensor):
        w_warm = getattr(sub_model_sparse[mod_num], conv_layer).weight

    block_sp = sub_model_sparse[mod_num]

    getattr(block_sp, conv_layer).weight.data = w_warm * Z # Apply the initial mask
    model_update = replace_func(model_update, block_number, block_sp)

    loss_history = []
    
    # Ensure the number of available blocks does not exceed actual number of layers
    num_blocks_available = min(len(sub_model_dense), num_blocks)

    for step in range(newton_steps):
        step_loss, total_batches = 0, 0
        
        # Set range for batching or full data
        batch_range = range(0, xdata.size(0), batch_size) if batching else [0]

        for batch_start in batch_range:
            batch_end = batch_start + batch_size if batching else xdata.size(0)
            xdata_batch = xdata[batch_start:batch_end].to(device)

            outputs_to_reconstruct = []
            dense_hook_handles, sparse_hook_handles = [], []

            # Use the helper function to register hooks
            operation_count = register_hooks_for_blocks(sub_model_dense, sub_model_sparse, conv_layer, k_step, num_blocks_available, outputs_to_reconstruct, dense_hook_handles, sparse_hook_handles)

            # Perform forward passes on dense and sparse submodels
            outputs_to_reconstruct.clear()
            _ = sub_model_dense(xdata_batch)
            ydata_dense = [output.to(device, non_blocking=True) for output in outputs_to_reconstruct]

            outputs_to_reconstruct.clear()
            _ = sub_model_sparse(xdata_batch)
            ydata_pruned = [output.to(device, non_blocking=True) for output in outputs_to_reconstruct]

            # Clear hooks after forward pass
            for handle in dense_hook_handles + sparse_hook_handles:
                handle.remove()

            min_length = min(len(ydata_dense), len(ydata_pruned))

            if min_length == 0:
                raise ValueError("No outputs were captured during forward pass. Ensure hooks are registered correctly.")

            if len(ydata_dense) != len(ydata_pruned):
                raise ValueError("Hooked outputs should be the same legnth. Something went wrong in hook registration.")
                
            loss_pruned = sum(mse_loss(ydata_dense[i], ydata_pruned[i]) for i in range(min_length))

            W = getattr(sub_model_sparse[mod_num], conv_layer).weight

            total_loss = loss_pruned
            current_loss = total_loss.item()

            if step == 0:
                loss_history.append(current_loss)
                
            loss_history.append(current_loss)

            grad_W = torch.autograd.grad(total_loss, W, create_graph=True, retain_graph=True)[0]*Z # Mask the gradient
            mask_output = Z.bool()
            b = -grad_W[mask_output]
            hvp_fn = hessian_vector_product_chunks
            full_newton_step = torch.zeros_like(W, device='cpu')
            newton_step, iters = conjugate_gradient_sparse(lambda v: hvp_fn(grad_W, W, v, mask_output), b, max_iter=CG_iterations)

            full_newton_step[mask_output] = newton_step.to('cpu', non_blocking=True)
            full_newton_step = full_newton_step.to(device, non_blocking=True)

            newton_step_norm = torch.norm(full_newton_step).item()

            # Delete variables to free memory before Armijo backtracking
            del xdata_batch, ydata_dense, ydata_pruned, total_loss
            torch.cuda.empty_cache()

            # Perform the Armijo step with hooks applied in the same way
            last_batch_xdata = xdata[xdata.size(0)-batch_size:xdata.size(0)].to(device, non_blocking=True)

            outputs_to_reconstruct = []
            dense_hook_handles, sparse_hook_handles = [], []

            operation_count = register_hooks_for_blocks(sub_model_dense, sub_model_sparse, conv_layer, k_step, num_blocks_available, outputs_to_reconstruct, dense_hook_handles, sparse_hook_handles)
            
            outputs_to_reconstruct.clear()
            _ = sub_model_dense(last_batch_xdata)
            last_batch_ydata_dense = [output.to(device, non_blocking=True) for output in outputs_to_reconstruct]

            outputs_to_reconstruct.clear()
            _ = sub_model_sparse(last_batch_xdata)
            last_batch_ydata_pruned = [output.to(device, non_blocking=True) for output in outputs_to_reconstruct]

            min_length = min(len(last_batch_ydata_dense), len(last_batch_ydata_pruned))
            
            if min_length == 0:
                raise ValueError("No outputs were captured during the Armijo step. Ensure hooks are registered correctly.")

            if len(last_batch_ydata_dense) != len(last_batch_ydata_pruned):
                raise ValueError("Hooked outputs should be the same legnth. Something went wrong in hook registration.")

            last_batch_loss_pruned = sum(mse_loss(last_batch_ydata_dense[i], last_batch_ydata_pruned[i]) for i in range(min_length))
            
            grad_W_val = torch.autograd.grad(last_batch_loss_pruned, W, create_graph=True, retain_graph=True)[0]

            del last_batch_ydata_pruned

            alpha = 1.0
            c = 1e-5
            max_backtracking_steps = 50

            W_original = getattr(sub_model_sparse[mod_num], conv_layer).weight.data.clone()

            sub_model_sparse = get_submodel(model_update, num_blocks, block_number)

            for bt in range(max_backtracking_steps):
                W_new = W_original + alpha * full_newton_step

                with torch.no_grad():
                    getattr(sub_model_sparse[mod_num], conv_layer).weight.data = W_new

                outputs_to_reconstruct.clear()
                _ = sub_model_sparse(last_batch_xdata)
                last_batch_ydata_pruned = [output.to(device, non_blocking=True) for output in outputs_to_reconstruct]

                last_batch_loss_pruned_new = sum(mse_loss(last_batch_ydata_dense[i], last_batch_ydata_pruned[i]) for i in range(min_length)).item()

                # Check if the Armijo condition is satisfied
                if last_batch_loss_pruned_new <= last_batch_loss_pruned + c * alpha * torch.dot(grad_W.flatten(), full_newton_step.flatten()):
                    break
                else:
                    # Reduce the step size and continue backtracking
                    alpha *= 0.9

                # If we reach the last backtracking step without improvement, reset to W_original
                if bt == max_backtracking_steps - 1:
                    with torch.no_grad():
                        getattr(sub_model_sparse[mod_num], conv_layer).weight.data = W_original

            with torch.no_grad():
                getattr(sub_model_sparse[mod_num], conv_layer).weight.data = W_new

            model_update = replace_func(model_update, block_number, sub_model_sparse[mod_num])

            for handle in dense_hook_handles + sparse_hook_handles:
                handle.remove()

            step_loss += current_loss
            total_batches += 1

            # Delete variables after Armijo backtracking
            del full_newton_step, last_batch_xdata, last_batch_ydata_dense
            torch.cuda.empty_cache()

            if len(loss_history) > 5 and all(loss_history[-i] / loss_history[-i-1] > stagnation_threshold for i in range(1, 6)):
                break

            if total_batches >= max_steps:
                return getattr(sub_model_sparse[mod_num], conv_layer).weight.data, current_loss

    return getattr(sub_model_sparse[mod_num], conv_layer).weight.data, current_loss


def register_hooks_for_blocks(sub_model_dense, sub_model_sparse, conv_layer, k_step, num_blocks, outputs_to_reconstruct, dense_hooks, sparse_hooks):
    operation_count = 0
    for i in range(min(len(sub_model_dense), num_blocks)):
        if i == 0:
            register_hooks_on_basicblock(sub_model_dense[i], dense_hooks, outputs_to_reconstruct, cur_module=conv_layer, is_first_block=True, operation_count=operation_count, K=k_step)
            operation_count = register_hooks_on_basicblock(sub_model_sparse[i], sparse_hooks, outputs_to_reconstruct, cur_module=conv_layer, is_first_block=True, operation_count=operation_count, K=k_step)
        else:
            register_hooks_on_basicblock(sub_model_dense[i], dense_hooks, outputs_to_reconstruct, is_first_block=False, operation_count=operation_count, K=k_step)
            operation_count = register_hooks_on_basicblock(sub_model_sparse[i], sparse_hooks, outputs_to_reconstruct, is_first_block=False, operation_count=operation_count, K=k_step)
        if operation_count > k_step:
            break
    return operation_count


def print_gpu_memory_usage():
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated()
        reserved = torch.cuda.memory_reserved()
        print(f"GPU Memory Usage: {allocated / (1024 ** 2):.2f} MB allocated, {reserved / (1024 ** 2):.2f} MB reserved")
    else:
        print("GPU is not available.")


    
def conjugate_gradient_sparse(hvp_fn, b, tol=1e-3, max_iter=5000, lambda_reg=1e-4):
    # Initialize x to be zero everywhere
    x = torch.zeros_like(b)

    # Restrict r, p to non-zero elements of b
    r = b.clone()
    p = r.clone()

    rs_old = torch.sum(r * r)

    for i in range(max_iter):
        # Only compute Hessian-vector product on the sparse p
        Ap = hvp_fn(p) + lambda_reg * p  # Add Tikhonov regularization term

        alpha = rs_old / torch.sum(p * Ap)

        # Update x only at the masked locations
        x = x + alpha * p

        # Update residual
        r = r - alpha * Ap
        rs_new = torch.sum(r * r)

        

        if torch.sqrt(rs_new) < tol:
            break

        # Update p
        p = r + (rs_new / rs_old) * p
        rs_old = rs_new

    return x, i + 1  # Return the solution x and the number of iterations




    
def hessian_vector_product(grad_W, W, vector, mask):
    # Create a zero tensor of the same shape as W

    vector = Variable(vector)
    full_vector = torch.zeros_like(W)
    
    # Populate full_vector only at the masked locations
    full_vector[mask] = vector
    
    # Compute the Hessian-vector product using the full_vector
    hvp = torch.autograd.grad(torch.sum(grad_W * full_vector), W, retain_graph=False)[0]
    
    # Return only the elements corresponding to the mask
    return hvp[mask]

def hessian_vector_product_chunks(grad_W, W, vector, mask, max_chunk_size=1e5):
    
    # gets the HVP in chunks to avoid peak memory usage being too high
    
    device = W.device  
    
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

        del chunk_hvp

    # Return only the elements corresponding to the mask
    return hvp[mask]


def prune_and_forward(model_update, block_number, horizon, warm_w, mask, num_blocks=2, device='cuda:0', conv_layer='conv2'):
    # Check if the model_update's name is "ResNet"
    sub_model_sparse = get_submodel(model_update, num_blocks, block_number)
    block_sp = sub_model_sparse[0]
    getattr(block_sp, conv_layer).weight.data = warm_w * mask
    model_update = replace_func(model_update, block_number, block_sp)
    
    return model_update




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

def print_memory_allocated(stage):
        # Only print if on GPU
        if torch.cuda.is_available():
            print(f"[{stage}] Memory Allocated: {torch.cuda.memory_allocated() / (1024 ** 2)} MB")


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
    
def get_blocks(model):
    
    # input copied model
    if model.name == 'resnet20_cifar10':
        
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

        block_list.append(("", nn.Sequential(OrderedDict([
          ('avgpool', model.avgpool),
          ('flatten', nn.Flatten()),
          ('fc', model.fc)
          ]))))
        
        # block_list.append(("",nn.Sequential(OrderedDict([('avgpool',child_list[6][1]),('flatten',nn.Flatten()),('fc',child_list[7][1])]))))

    if model.name == 'resnet50_imagenet':
        
        block_list = []
        child_list = list(model.named_children())
        
        block_list.append(("",nn.Sequential(OrderedDict([
          ('conv1', child_list[0][1]),
          ('bn1', child_list[1][1]),
          ('relu', child_list[2][1]),
          ('maxpool', child_list[3][1])
          ]))))
        
        for i in range(4,8):
            for name, child in child_list[i][1].named_children():
                block_list.append((child_list[i][0]+"."+name,child))
        
        block_list.append(("", nn.Sequential(OrderedDict([
          ('avgpool', model.avgpool),
          ('flatten', nn.Flatten()),
          ('fc', model.fc)
          ]))))

    if model.name == 'resnet50_cifar10' or model.name == 'resnet50_cifar100':

        block_list = []
        child_list = list(model.named_children())
        
        block_list.append(("", nn.Sequential(OrderedDict([
          ('conv1', child_list[0][1]),
          ('bn1', child_list[1][1]),
          ('relu', child_list[2][1]),
          ('maxpool', nn.Identity())  # No max pooling for CIFAR-10
          ]))))
        
        for i in range(4,8):
            for name, child in child_list[i][1].named_children():
                block_list.append((child_list[i][0]+"."+name, child))
        
        block_list.append(("", nn.Sequential(OrderedDict([
          ('global_pool', model.global_pool),
          ('fc', model.fc)
          ]))))

    if model.name == 'MobileNet':
        
        block_list = []
        child_list = list(model.named_children())
        child_name = child_list[0][0]
        child_list2 = list(child_list[0][1].named_children())
        for i in range(14):
            block_list.append((child_name+"."+child_list2[i][0],child_list2[i][1]))
            
        block_list.append(("",nn.Sequential(OrderedDict([('avgpool',child_list2[-1][1]),('flatten',nn.Flatten()),('fc',child_list[-1][1])]))))
        
        
    return block_list

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


def weight_update_unstr_torch(W, XTX, XTY):
    
    p, m = W.shape
        
    W = torch.tensor(W.astype(np.float64)) 
    XTX = torch.tensor(XTX.astype(np.float64))
    XTY = torch.tensor(XTY.astype(np.float64))
   
    W_sol = torch.zeros_like(W)
    for k in range(m):
       
       
        nzi = torch.nonzero(W[:,k], as_tuple=True)[0]
        if len(nzi) == 0:
            continue
        XTX_sub = XTX[nzi[:,None],nzi]
        XTY_sub = XTY[nzi,k]
        W_sol[nzi,k] = torch.linalg.inv(XTX_sub)@ XTY_sub
       
    return W_sol.cpu().numpy()


def calculate_objective_column(W, W_sol, XTX, XTY, col):
    objective = W_sol[:, col].T @ XTX @ W_sol[:, col] - 2 * W_sol[:, col].T @ XTY[:, col] + W[:, col].T @ XTX @ W[:, col]
    return 0.5 * objective

def forward_selection_column(W, W_MP, XTX, XTY, group_indices, N, col):
    p, m = W_MP.shape
    num_groups = len(group_indices)

    W_best = W_MP.copy()
    W_sol_best = weight_update_unstr_torch(W_best, XTX, XTY)
    obj_best = calculate_objective_column(W, W_sol_best, XTX, XTY, col)
    
    improved = False
    for group in range(num_groups):
        indices = group_indices[group]
        M = len(indices)
        
        current_pattern = np.nonzero(W_best[indices, col])[0]
        current_indices = [indices[i] for i in current_pattern]
        
        for pattern in itertools.combinations(indices, N):
            if set(pattern) == set(current_indices):
                continue
            
            W_candidate = W_best.copy()
            W_candidate[indices, col] = 0
            W_candidate[list(pattern), col] = 1
            
            W_sol_candidate = weight_update_unstr_torch_col(W_candidate, XTX, XTY, col)
            obj_candidate = calculate_objective_column(W, W_sol_candidate, XTX, XTY, col)
            
            if obj_candidate < obj_best:
                obj_best = obj_candidate
                W_best = W_candidate.copy()
                W_sol_best = W_sol_candidate.copy()
                improved = True

    return W_best[:, col], W_sol_best[:, col], obj_best, improved

def weight_update_unstr_torch_col(W, XTX, XTY, col):
    p, m = W.shape
    W = torch.tensor(W.astype(np.float64))
    XTX = torch.tensor(XTX.astype(np.float64))
    XTY = torch.tensor(XTY.astype(np.float64))
    W_sol = W.clone()
    
    nzi = torch.nonzero(W[:,col], as_tuple=True)[0]
    if len(nzi) > 0:
        XTX_sub = XTX[nzi[:,None],nzi]
        XTY_sub = XTY[nzi,col]
        W_sol[nzi,col] = torch.linalg.inv(XTX_sub) @ XTY_sub
    
    return W_sol.cpu().numpy()

def forward_selection(W, W_MP, XTX, XTY, group_indices, N, num_cycles=10):
    p, m = W_MP.shape
    results = []
    
    for cycle in range(num_cycles):
        # Parallelize the column-wise operation
        results = Parallel(n_jobs=-1)(delayed(forward_selection_column)(
            W, W_MP, XTX, XTY, group_indices, N, col) for col in range(m))
        
        # Aggregate results and check for improvement
        improved = any(result[3] for result in results)  # Check if any column was improved
        if not improved:
            break

    # Aggregate final results from all cycles
    W_best = np.column_stack([result[0] for result in results])
    W_sol_best = np.column_stack([result[1] for result in results])
    obj_best = sum(result[2] for result in results)  # Sum of objectives for all columns

    z_best = np.where(np.abs(W_sol_best) > 0, 1, 0)
    return W_sol_best, z_best


def generate_M_sized_groups(d_in, k_h, k_w, M):
    overall_groups = set()
    groups = []
    
    num_vars = int(d_in * k_h * k_w)

    for i in range(num_vars):
        # Check if the last index in the potential group is within the bounds
        if i not in overall_groups and i + k_h * k_w * (M - 1) < num_vars:
            group_i = [i + k_h * k_w * j for j in range(M)]
            groups.append(group_i)

            for j in group_i:
                overall_groups.add(j)
    
    return groups



def generate_M_sized_groups_fc(d_col, M):
    groups = []
    for i in range(0, d_col, M):
        groups.append(list(range(i, min(i + M, d_col))))
    return groups

def MP_unstr(w_var, p):
    """
    Keep the top p% of weights in the tensor w_var and set the rest to zero.
    
    Args:
    w_var (torch.Tensor): The weight tensor to prune.
    p (float): The percentage of top weights to keep (between 0 and 100).
    
    Returns:
    torch.Tensor: A tensor with only the top p% weights retained, others set to zero.
    """
    # Flatten the weight tensor
    w_var = torch.tensor(w_var)
    w_flat = w_var.flatten()
    
    # Calculate the threshold index
    k = int((1 - p / 100) * w_flat.numel())
    
    # Get the top k values (those that should be pruned)
    topk_values, _ = torch.topk(torch.abs(w_flat), k, largest=False)
    threshold = topk_values[-1]  # The threshold value
    
    # Create a mask for weights greater than the threshold
    mask = torch.abs(w_var) > threshold
    w_var = w_var * mask.float()
    # Return the pruned weight tensor
    return w_var.numpy()


# def iterative_pruning(W_init, sparsity_levels, solve_for_W_given_Z, param_size, model, model_update,
#                       newton_xdata, total_blocks, block_count, conv_layer, block_list,
#                       layerparams, k_step, device, M, groups, batching, batch_size, proj_batch_size, CG_iterations, projection_interval):

#     W = torch.clone(W_init)
#     total_params = np.prod(param_size)

#     model = model.to(device)
#     model_update = model_update.to(device)

#     # Initialize dictionaries to store loss and W for each sparsity pattern
#     loss_dict = {}
#     W_dict = {}

#     mse_loss = nn.MSELoss()

#     mask = (torch.abs(W) > 1e-7).float()

#     # Initialize tracking variables for support changes
#     support_changes = []  # List to store the number of changes in the support        

#     W_proj = W.clone()
#     for t, sparsity in enumerate(sparsity_levels):
#         print(f"Sparsity Level: {sparsity}")

#         # Step 1: Project W_init onto the current sparsity pattern
#         W = W_proj.reshape(param_size[0], -1).T
#         W_proj = torch.tensor(W).T.reshape(-1).reshape(param_size).to(device).to(torch.float32)
#         W_0 = torch.tensor(W).T.reshape(-1).reshape(param_size).to(device).to(torch.float32)

#         mask = (torch.abs(W_0) > 1e-7).float()

#         # Register hooks only once outside the projection loop
#         outputs_to_reconstruct = []
#         dense_hook_handles, sparse_hook_handles = [], []

#         num_blocks = min(k_step + 1, total_blocks - block_count)
#         sub_model_dense = get_submodel(model, num_blocks, block_count)
#         sub_model_sparse = get_submodel(model_update, num_blocks, block_count)

#         # Register hooks for the blocks
#         operation_count = 0
#         K = k_step  # Total number of operations to capture

#         for i in range(num_blocks):
#             block_dense = sub_model_dense[i]
#             block_sparse = sub_model_sparse[i]

#             if i == 0:
#                 # Register hooks on the first block, handling only relevant layers
#                 register_hooks_on_basicblock(block_dense, dense_hook_handles, outputs_to_reconstruct, cur_module=conv_layer, is_first_block=True, operation_count=operation_count, K=K)
#                 operation_count = register_hooks_on_basicblock(block_sparse, sparse_hook_handles, outputs_to_reconstruct, cur_module=conv_layer, is_first_block=True, operation_count=operation_count, K=K)
#             else:
#                 # Register hooks on subsequent blocks, stopping when operation_count reaches K
#                 register_hooks_on_basicblock(block_dense, dense_hook_handles, outputs_to_reconstruct, is_first_block=False, operation_count=operation_count, K=K)
#                 operation_count = register_hooks_on_basicblock(block_sparse, sparse_hook_handles, outputs_to_reconstruct, is_first_block=False, operation_count=operation_count, K=K)

#             if operation_count > K:
#                 break

#         # Forward passes for each batch
#         batch_range = range(0, proj_batch_size, newton_xdata.size(0))
#         for batch_start in batch_range:
#             batch_end = min(batch_start + proj_batch_size, newton_xdata.size(0))
#             xdata_batch = newton_xdata[batch_start:batch_end].to(device)

#             # Forward pass through dense model
#             outputs_to_reconstruct.clear()
#             _ = sub_model_dense(xdata_batch)
#             ydata_dense = [output.to(device, non_blocking=True) for output in outputs_to_reconstruct]

#             # Forward pass through sparse model
#             outputs_to_reconstruct.clear()
#             _ = sub_model_sparse(xdata_batch)
#             ydata_pruned = [output.to(device, non_blocking=True) for output in outputs_to_reconstruct]

#             # Compute initial loss for the projection
#             min_length = min(len(ydata_dense), len(ydata_pruned))
#             loss_pruned = sum(mse_loss(ydata_dense[i], ydata_pruned[i]) for i in range(min_length))
#             total_loss = loss_pruned

#             # Gradient computation for projection
#             grad_proj_W = torch.autograd.grad(total_loss, getattr(sub_model_sparse[0], conv_layer).weight, create_graph=True, retain_graph=True)[0]

#             steps = 100
#             prev_loss = float('inf')

#             initial_step = True
#             # Perform gradient-based projection for a fixed number of iterations
#             for proj_step in range(steps):
#                 hvp = hessian_vector_product_chunks(grad_proj_W, getattr(sub_model_sparse[0], conv_layer).weight, (W_proj - W_0)[mask.bool()], mask.bool(), max_chunk_size=1e5)

#                 g = 2 * hvp 

#                 max_norm = 10.0  # Adjust this value based on experimentation
#                 g_norm = torch.linalg.norm(g)
#                 if g_norm > max_norm:
#                     g = g * (max_norm / g_norm)
        

#                 # Update W_proj using gradient descent
#                 alpha = 1e-2
#                 W_proj[mask.bool()] = W_proj[mask.bool()] - alpha * g

                
#                 if (proj_step + 1) % projection_interval == 0 or proj_step == steps-1:
#                     W_proj = W_proj.reshape(param_size[0], -1).T.detach().cpu().numpy()
#                     W_proj = MP_MN(W_proj, groups=groups, N=sparsity)
#                     W_proj = torch.tensor(W_proj).T.reshape(-1).reshape(param_size).to(device).to(torch.float32)
#                     print("projection loss")
#                     print(torch.linalg.norm(g))
#                     print_sparsity(W_proj, f"Required: {sparsity}")

#                     current_loss = torch.linalg.norm(g)


#                     print(current_loss, prev_loss)
#                     print(proj_step)
#                     if current_loss > prev_loss:
#                         print(f"Projection loss increased at step {proj_step+1}. Breaking the loop.")
#                         break

#                     if initial_step:
#                         initial_step = False
#                     else:
#                         prev_loss = current_loss

                
#                 del g, hvp

#             mask_final = (torch.abs(W_proj) > 1e-7).float()

#             for handle in dense_hook_handles + sparse_hook_handles:
#                 handle.remove()
             
#             del xdata_batch, ydata_pruned, ydata_dense, grad_proj_W, total_loss, loss_pruned

#         # Step 2: Solve for the optimal W given the current mask Z
#         CG_iterations = 5000 if t == len(sparsity_levels) - 1 else 200
#         newton_steps = 1
#         W_proj, loss, _, _ = solve_for_W_given_Z(
#             Z=mask_final, model=model, model_update=model_update,
#             xdata=newton_xdata, block_number=block_count, k_step=k_step,
#             total_blocks=total_blocks, device=device, N=sparsity, M=M,
#             w_warm=W_proj, conv_layer=conv_layer, block_list=block_list,
#             batching=batching, batch_size=batch_size, CG_iterations=CG_iterations, newton_steps=newton_steps
#         )
#         del loss

#     return W_proj

def prune_and_update(w_var, pairs, groups, total_pairs, elimination_fraction=0.2, abs_largest = True):
#     """
#     Prunes the smallest non-zero elements across all (output channel, group_index) pairs
#     and returns the pruned weights, updated pairs, and updated mask.
#     """
#     mask = torch.ones_like(w_var, dtype=torch.float32)
#     smallest_elements = []

#     # Collect smallest non-zero elements from each (output channel, group_index) pair
#     for output_channel, group_idx in list(pairs):  # Use list(pairs) to iterate over a copy, allowing safe removal
#         group = groups[group_idx]
#         group_values = w_var[group, output_channel]  # Keep the structure

#         # Exclude zeros by setting them to infinity temporarily for the min operation
#         non_zero_mask = group_values != 0
        
#         if torch.any(non_zero_mask):
#             non_zero_values = torch.where(non_zero_mask, group_values, torch.tensor(float('inf')).to(group_values.device))

#             if abs_largest:
#                 # Find the minimum non-zero value and its index within the group
#                 min_value, min_idx = torch.min(torch.abs(non_zero_values), dim=0)
#             else:
#                 min_value, min_idx = torch.min(non_zero_values, dim=0)

#             # Handle the case where group_values might be multidimensional
#             if group_values.ndim > 1:
#                 min_idx = min_idx.item()  # Convert tensor index to scalar

#             # Record the minimum non-zero value along with the indices
#             smallest_elements.append((min_value.item(), group[min_idx], output_channel, group_idx))
#         else:
#             # Discard the pair if it doesn't pass the non_zero_mask condition
#             pairs.discard((output_channel, group_idx))

#     # Sort all collected elements by their magnitude
#     smallest_elements.sort(key=lambda x: x[0])

#     # Determine the number of (output channel, group_index) pairs to prune, rounding up
#     num_pairs_to_prune = max(1, math.ceil(total_pairs * elimination_fraction))

#     # Prune the smallest (output channel, group_index) pairs
#     for _, min_group_idx, output_channel, group_idx in smallest_elements[:num_pairs_to_prune]:
#         # Zero out the group values in the correct output channel
#         w_var[min_group_idx, output_channel] = 0
        
#         # Update the mask to reflect the pruning
#         mask[min_group_idx, output_channel] = 0
        
#         # Remove the pruned pair from the set of remaining pairs
#         pairs.discard((output_channel, group_idx))

#     return w_var, pairs, mask