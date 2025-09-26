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


# -----------------------------
# Small helpers
# -----------------------------

def get_nested_attr(obj, attr):
    parts = attr.split('.')
    for a in parts:
        obj = getattr(obj, a)
    return obj


def extract_mlp_number(layer_string):
    parts = layer_string.split('.')
    mlp_index = parts.index("mlp")
    return parts[mlp_index + 1]


def capture_outputs_hook(module, input, output, outputs_list):
    if isinstance(output, tuple):
        outputs_list.append(output[0])
    elif isinstance(output, torch.Tensor):
        outputs_list.append(output)


# -----------------------------
# Block & op counting
# -----------------------------

def count_operations_in_block(block_tuple, cur_module=None):
    op_count = 0
    count_flag = False
    _, block = block_tuple

    for name, submodule in block.named_children():
        if cur_module is not None and not count_flag:
            if name == cur_module:
                count_flag = True
            continue

        if count_flag or cur_module is None:
            if isinstance(submodule, MultiheadAttention):
                op_count += 1
            elif isinstance(submodule, torch.nn.Sequential):
                for layer in submodule:
                    if isinstance(layer, (torch.nn.Linear, GELU, LayerNorm)):
                        op_count += 1
    return op_count


def find_num_blocks_for_k_operations(blocks, start_block, cur_module, K):
    total_ops = 0
    num_blocks = 0
    for block_idx in range(start_block, len(blocks)):
        _, block = blocks[block_idx]
        block_ops = count_operations_in_block((None, block), cur_module if block_idx == start_block else None)
        total_ops += block_ops
        num_blocks += 1
        if total_ops > K:
            break
    return num_blocks


# -----------------------------
# Hook registration
# -----------------------------

def register_hooks_on_vit_block(block, hook_list, outputs_list, cur_module=None, is_first_block=False, operation_count=0, K=0):
    count_flag = not is_first_block
    excluded_layers = (torch.nn.Dropout,)

    for name, submodule in block.named_children():
        if is_first_block and cur_module is not None and not count_flag:
            if name == cur_module:
                count_flag = True
                hook_list.append(submodule.register_forward_hook(lambda m, i, o: capture_outputs_hook(m, i, o, outputs_list)))
                operation_count += 1

        if count_flag:
            if isinstance(submodule, MultiheadAttention):
                hook_list.append(submodule.register_forward_hook(lambda m, i, o: capture_outputs_hook(m, i, o, outputs_list)))
                operation_count += 1
            elif hasattr(submodule, 'named_children'):
                for _, child in submodule.named_children():
                    if isinstance(child, (torch.nn.Linear, GELU, LayerNorm)) and not isinstance(child, excluded_layers):
                        if operation_count > K:
                            break
                        hook_list.append(child.register_forward_hook(lambda m, i, o: capture_outputs_hook(m, i, o, outputs_list)))
                        operation_count += 1

        if operation_count > K:
            break
    return operation_count


# -----------------------------
# Module lookup
# -----------------------------

def find_module(net, params_to_prune, name=''):
    children = list(net.named_children())
    if not children:
        if isinstance(net, InProjModule) and params_to_prune.endswith(".in_proj_weight"):
            return True, net
        if name + ".weight" == params_to_prune:
            return True, net
        return False, None

    for child_name, child in children:
        full_name = f"{name}.{child_name}" if name else child_name
        found, module = find_module(child, params_to_prune, name=full_name)
        if found:
            return True, module
    return False, None


def find_all_module(net, params_to_prune, name='', prune_list=[]):
    for param_name, _ in net.named_parameters(recurse=False):
        full_param_name = f"{name}.{param_name}" if name else param_name
        if full_param_name in params_to_prune:
            prune_list.append(full_param_name)

    if isinstance(net, InProjModule):
        in_proj_weight_name = f"{name}.in_proj_weight" if name else "in_proj_weight"
        if in_proj_weight_name in params_to_prune:
            prune_list.append(in_proj_weight_name)

    if isinstance(net, NonDynamicallyQuantizableLinear):
        out_proj_weight_name = f"{name}.out_proj.weight" if name else "out_proj.weight"
        if out_proj_weight_name in params_to_prune:
            prune_list.append(out_proj_weight_name)

    for child_name, child in net.named_children():
        child_full_name = f"{name}.{child_name}" if name else child_name
        find_all_module(child, params_to_prune, name=child_full_name, prune_list=prune_list)

    return prune_list


# -----------------------------
# Model block listing (ViT)
# -----------------------------

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
    return []


# -----------------------------
# Pruning helpers
# -----------------------------

def prune_and_forward(model_update, block_number, horizon, warm_w, mask, num_blocks=2, device='cuda:0', vit_layer="", vit_name=""):
    get_submodel_func = get_submodel_vit
    replace_func = replace_block_in_vit

    sub_model_sparse = get_submodel_func(model_update, num_blocks, block_number)
    block_sp = sub_model_sparse[0]

    mask = mask.to(device)
    _, mod = find_module(block_sp, vit_layer, vit_name)

    if isinstance(mod, NonDynamicallyQuantizableLinear):
        mod.weight.data = warm_w * mask
        block_sp.self_attention.out_proj = mod
    elif isinstance(mod, torch.nn.Linear):
        mod.weight.data = warm_w * mask
        setattr(block_sp, vit_name.split('.')[-1], mod)
    elif isinstance(mod, InProjModule):
        mod.in_proj_weight.data = warm_w * mask
        block_sp.self_attention.in_proj_module = mod

    model_update = replace_func(model_update, block_number, block_sp)
    return block_sp, model_update


# -----------------------------
# HVP / CG
# -----------------------------

def hessian_vector_product_chunks(grad_W, W, vector, mask, max_chunk_size=5e4):
    device = W.device
    vector = Variable(vector).to(device)
    full_vector = torch.zeros_like(W, device=device)
    full_vector[mask] = vector

    hvp = torch.zeros_like(W, device=device)
    num_elements = mask.sum().item()

    num_chunks = int(max(1, (num_elements + max_chunk_size - 1) // max_chunk_size))
    chunk_size = (num_elements + num_chunks - 1) // num_chunks

    # precompute an index tensor for masked positions
    idx = torch.nonzero(mask.flatten(), as_tuple=False).flatten().to(device)

    for i in range(num_chunks):
        start = i * chunk_size
        end = min(start + chunk_size, num_elements)
        if start >= end:
            break

        cur_idx = idx[start:end]
        chunk_mask = torch.zeros_like(W, dtype=torch.bool, device=device).flatten()
        chunk_mask[cur_idx] = True
        chunk_mask = chunk_mask.view_as(W)

        chunk_hvp = torch.autograd.grad(
            torch.sum(grad_W * full_vector * chunk_mask), W, retain_graph=True
        )[0]
        hvp[chunk_mask] += chunk_hvp[chunk_mask]

    return hvp[mask]


def hessian_vector_product(grad_W, W, vector, mask):
    vector = Variable(vector)
    full_vector = torch.zeros_like(W)
    full_vector[mask] = vector
    hvp = torch.autograd.grad(torch.sum(grad_W * full_vector), W, retain_graph=False)[0]
    return hvp[mask]


def conjugate_gradient_sparse(hvp_fn, b, tol=5e-4, max_iter=1000, lambda_reg=1e-4, window_size=100):
    x = torch.zeros_like(b)
    r = b.clone()
    p = r.clone()
    rs_old = torch.sum(r * r)
    residuals = []

    for _ in range(max_iter):
        Ap = hvp_fn(p) + lambda_reg * p
        alpha = rs_old / torch.sum(p * Ap)
        x = x + alpha * p
        r = r - alpha * Ap
        rs_new = torch.sum(r * r)

        residual = torch.sqrt(rs_new).item()
        residuals.append(residual)
        if len(residuals) > window_size:
            residuals.pop(0)

        if residual < tol:
            break

        p = r + (rs_new / rs_old) * p
        rs_old = rs_new

        del Ap, alpha, rs_new
        torch.cuda.empty_cache()

    del r, p, rs_old, residuals
    torch.cuda.empty_cache()
    return x


# -----------------------------
# N:M mask application
# -----------------------------

def make_nm_sparse(weight, N, M, prune_thirds=[0, 1, 2]):
    assert weight.dim() == 2
    assert M > N
    assert all(third in [0, 1, 2] for third in prune_thirds)

    num_rows, num_cols = weight.size()
    third_size = num_rows // 3
    mask = torch.ones_like(weight)

    for third in prune_thirds:
        start_row = third * third_size
        end_row = start_row + third_size

        for i in range(start_row, end_row):
            row = weight[i].detach().abs()
            group_size = (num_cols) // M
            remaining_cols = num_cols % M

            row_blocks = row[:group_size * M].reshape(-1, M)
            _, indices = torch.topk(row_blocks, M - N, dim=1, largest=False)

            mask_blocks = torch.ones_like(row_blocks)
            mask_blocks.scatter_(1, indices, 0)
            mask[i, :group_size * M] = mask_blocks.flatten()

            if remaining_cols > 0:
                remaining_row = row[group_size * M:]
                _, remaining_indices = torch.topk(remaining_row, max(0, remaining_cols - N), largest=False)
                mask[i, group_size * M:group_size * M + remaining_cols].scatter_(0, remaining_indices, 0)

    return weight * mask


# -----------------------------
# Core: solve_for_W_given_Z
# -----------------------------

def solve_for_W_given_Z(
    Z,
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
    batch_size=256,
):
    # ensure devices
    if not next(model.parameters()).is_cuda or next(model.parameters()).device != device:
        model.to(device)
    if not next(model_update.parameters()).is_cuda or next(model_update.parameters()).device != device:
        model_update.to(device)

    mask_output = Z.bool()
    get_submodel_func = get_submodel_vit
    replace_func = replace_block_in_vit
    num_blocks = find_num_blocks_for_k_operations(block_list, block_number, vit_layer, k_step)
    mse_loss = nn.MSELoss()

    sub_model_dense = get_submodel_func(model, num_blocks, block_number)
    sub_model_sparse = get_submodel_func(model_update, num_blocks, block_number)
    _, mod = find_module(sub_model_dense[0], vit_layer, vit_name)

    if isinstance(mod, (InProjModule, NonDynamicallyQuantizableLinear)):
        module_key = "self_attention"
    elif isinstance(mod, (torch.nn.Linear, nn.Sequential)) and (
        isinstance(mod, torch.nn.Linear) or any(isinstance(layer, torch.nn.Linear) for layer in getattr(mod, "_modules", {}).values())
    ):
        module_key = "mlp"
    else:
        module_key = "other"

    # apply warm start + mask to sparse block
    _, model_update = prune_and_forward(
        model_update=model_update,
        block_number=block_number,
        horizon=-1,
        warm_w=w_warm,
        mask=Z,
        num_blocks=num_blocks,
        device=device,
        vit_layer=vit_layer,
        vit_name=vit_name,
    )
    sub_model_sparse = get_submodel_func(model_update, num_blocks, block_number)

    alpha = 1.0  # will be updated in loop

    for _ in range(newton_steps):
        for batch_start in range(0, xdata.size(0) - batch_size, batch_size):
            batch_end = batch_start + batch_size
            x_batch = xdata[batch_start:batch_end].to(device)

            outputs_to_reconstruct = []
            dense_hooks, sparse_hooks = [], []
            op_count = 0

            for j in range(num_blocks):
                block_dense = sub_model_dense[j]
                block_sparse = sub_model_sparse[j]

                if j == 0:
                    register_hooks_on_vit_block(block_dense, dense_hooks, outputs_to_reconstruct,
                                                cur_module=module_key, is_first_block=True,
                                                operation_count=op_count, K=k_step)
                    op_count = register_hooks_on_vit_block(block_sparse, sparse_hooks, outputs_to_reconstruct,
                                                           cur_module=module_key, is_first_block=True,
                                                           operation_count=op_count, K=k_step)
                else:
                    register_hooks_on_vit_block(block_dense, dense_hooks, outputs_to_reconstruct,
                                                is_first_block=False, operation_count=op_count, K=k_step)
                    op_count = register_hooks_on_vit_block(block_sparse, sparse_hooks, outputs_to_reconstruct,
                                                           is_first_block=False, operation_count=op_count, K=k_step)
                if op_count > k_step:
                    break

            # dense forward
            outputs_to_reconstruct.clear()
            _ = sub_model_dense(x_batch)
            y_dense = [t.to(device, non_blocking=True) for t in outputs_to_reconstruct]

            # sparse forward
            outputs_to_reconstruct.clear()
            _ = sub_model_sparse(x_batch)
            y_sparse = [t.to(device, non_blocking=True) for t in outputs_to_reconstruct]

            min_len = min(len(y_dense), len(y_sparse))
            if min_len == 0:
                for h in dense_hooks + sparse_hooks:
                    h.remove()
                raise ValueError("No outputs captured; check hooks.")

            loss_pruned = sum(mse_loss(y_dense[i], y_sparse[i]) for i in range(k_step + 1))

            # pick the weight tensor W for gradient/HVP
            if isinstance(mod, NonDynamicallyQuantizableLinear):
                _, sparse_out_proj = find_module(sub_model_sparse[0], vit_layer, vit_name)
                W = sparse_out_proj.weight
            elif isinstance(mod, torch.nn.Linear):
                _, sparse_mlp = find_module(sub_model_sparse[0], vit_layer, vit_name)
                W = sparse_mlp.weight
            elif isinstance(mod, InProjModule):
                _, sparse_in_proj = find_module(sub_model_sparse[0], vit_layer, vit_name)
                W = sparse_in_proj.in_proj_weight
            else:
                for h in dense_hooks + sparse_hooks:
                    h.remove()
                continue

            grad_W = torch.autograd.grad(loss_pruned, W, create_graph=True, retain_graph=True)[0]
            b = -grad_W[mask_output]
            full_newton_step = torch.zeros_like(W, device='cpu')
            newton_step = conjugate_gradient_sparse(
                lambda v: hessian_vector_product_chunks(grad_W, W, v, mask_output),
                b,
                max_iter=CG_iterations
            )
            full_newton_step[mask_output] = newton_step.to('cpu', non_blocking=True)
            full_newton_step = full_newton_step.to(device, non_blocking=True)

            # backtracking on a held-out recent batch
            last_batch = xdata[xdata.size(0) - batch_size:xdata.size(0)].to(device)
            outputs_to_reconstruct.clear()
            _ = sub_model_dense(last_batch)
            last_y_dense = [t.to(device, non_blocking=True) for t in outputs_to_reconstruct]

            outputs_to_reconstruct.clear()
            _ = sub_model_sparse(last_batch)
            last_y_sparse = [t.to(device, non_blocking=True) for t in outputs_to_reconstruct]

            last_loss = sum(mse_loss(last_y_dense[i], last_y_sparse[i]) for i in range(k_step + 1))
            alpha = 1.0
            c = 1e-5
            max_bt = 50
            W_original = W.clone()

            flat_dot = torch.dot(grad_W.flatten(), full_newton_step.flatten())

            for bt in range(max_bt):
                W_new = W_original + alpha * full_newton_step
                with torch.no_grad():
                    W.data = W_new

                outputs_to_reconstruct.clear()
                _ = sub_model_sparse(last_batch)
                last_y_sparse_new = [t.to(device, non_blocking=True) for t in outputs_to_reconstruct]
                last_loss_new = sum(mse_loss(last_y_dense[i], last_y_sparse_new[i]) for i in range(k_step + 1)).item()

                if last_loss_new <= last_loss + c * alpha * flat_dot:
                    break
                alpha *= 0.9

                if bt == max_bt - 1:
                    with torch.no_grad():
                        W.data = W_original

            model_update = replace_func(model_update, block_number, sub_model_sparse[0])

            for h in dense_hooks + sparse_hooks:
                h.remove()

            del full_newton_step, last_batch, last_y_dense, last_y_sparse, last_y_sparse_new
            torch.cuda.empty_cache()

        if alpha < 0.1:
            break

    sub_model_sparse = get_submodel_func(model_update, num_blocks, block_number)
    block_sp = sub_model_sparse[0]

    if isinstance(mod, NonDynamicallyQuantizableLinear):
        weight_to_return = mod.weight.data * Z
    elif isinstance(mod, torch.nn.Linear):
        mlp_idx = int(extract_mlp_number(vit_layer))
        weight_to_return = block_sp.mlp[mlp_idx].weight.data * Z
    elif isinstance(mod, InProjModule):
        weight_to_return = mod.in_proj_weight * Z
    else:
        weight_to_return = Z.to(W.device).clone()  # fallback, shape guard if ever reached

    return weight_to_return
