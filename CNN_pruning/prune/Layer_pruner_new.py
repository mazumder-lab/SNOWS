from .utils import *
import time
import torch.nn as nn
from torch.utils.data import DataLoader
import sys
import copy
import numpy as np

ALG_PATH = './SNOWS'
sys.path.append(ALG_PATH)
from SNOWS import (
    get_blocks,
    find_module,
    find_all_module,
    MP_MN,
    generate_M_sized_groups,
    weight_update_unstr_torch,
    forward_selection,
    solve_for_W_given_Z,
    MP_unstr,
    backward_selection_all_unstr
)

from helpers import extract_conv_layer, get_block_number_for_param


class LayerPruner:
    def __init__(
        self,
        model,
        params,
        train_dataset,
        train_dataloader,
        test_dataloader,
        nsamples,
        criterion,
        lambda_inv,
        gradseed,
        device,
        algo,
        custom_masks=None,
        custom_weights=None
    ):
        self.model = model
        self.params = params
        self.train_dataset = train_dataset
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.criterion = criterion
        self.nsamples = nsamples
        self.lambda_inv = lambda_inv
        self.device = device
        self.algo = algo
        self.grads = None
        self.results = dict()
        self.gradseed = gradseed
        self.scaled = False
        self.custom_masks = custom_masks
        self.custom_weights = custom_weights

    def update_model(self, new_w):
        set_pvec(new_w, self.model, self.params, self.device)

    def getinput_hook(self, module, input, output):
        self.input_buffer = []
        self.input_buffer.append(input[0].detach())

    def get_size(self, use_layer=True):
        size_list = []
        ignore_bias = True
        for name, param in self.model.named_parameters():
            layer_name, param_name = '.'.join(name.split('.')[:-1]), name.split('.')[-1]
            if ignore_bias and param_name == 'bias':
                continue
            if use_layer:
                if (not self.layerparams is None) and (not name in self.layerparams):
                    continue
                size_list.append(param.shape)
            else:
                if (not self.params is None) and (not name in self.params):
                    continue
                size_list.append(param.shape)
        self.size_list = size_list

    def get_params(self):
        if self.model.name == 'resnet20_cifar10':
            print("Setting up parameters for resnet20_cifar10")
            self.datasize = [3, 32, 32]
            self.layerparams = [
                'layer1.0.conv1.weight', 'layer1.0.conv2.weight',
                'layer1.1.conv1.weight', 'layer1.1.conv2.weight',
                'layer1.2.conv1.weight', 'layer1.2.conv2.weight',
                'layer2.0.conv1.weight', 'layer2.0.conv2.weight',
                'layer2.1.conv1.weight', 'layer2.1.conv2.weight',
                'layer2.2.conv1.weight', 'layer2.2.conv2.weight',
                'layer3.0.conv1.weight', 'layer3.0.conv2.weight',
                'layer3.1.conv1.weight', 'layer3.1.conv2.weight',
                'layer3.2.conv1.weight', 'layer3.2.conv2.weight'
            ]
        elif self.model.name == 'resnet50_imagenet':
            print("Setting up parameters for resnet50_imagenet")
            self.datasize = [3, 224, 224]
            self.layerparams = collect_layer_params(self.model)
            self.layerparams = [i for i in self.layerparams if "downsample" not in i]
            if 'conv1.weight' in self.layerparams:
                self.layerparams.remove('conv1.weight')
        elif self.model.name == 'MobileNet':
            print("Setting up parameters for MobileNet")
            self.datasize = [3, 224, 224]
            self.layerparams = [
                'model.1.0.weight', 'model.1.3.weight',
                'model.2.0.weight', 'model.2.3.weight',
                'model.3.0.weight', 'model.3.3.weight',
                'model.4.0.weight', 'model.4.3.weight',
                'model.5.0.weight', 'model.5.3.weight',
                'model.6.0.weight', 'model.6.3.weight',
                'model.7.0.weight', 'model.7.3.weight',
                'model.8.0.weight', 'model.8.3.weight',
                'model.9.0.weight', 'model.9.3.weight',
                'model.10.0.weight', 'model.10.3.weight',
                'model.11.0.weight', 'model.11.3.weight',
                'model.12.0.weight', 'model.12.3.weight',
                'model.13.0.weight', 'model.13.3.weight',
                "fc.weight"
            ]
        elif self.model.name == "resnet50_cifar10" or self.model.name == "resnet50_cifar100":
            print("Setting up parameters for resnet50_cifar10/cifar100")
            self.datasize = [3, 32, 32]
            self.layerparams = collect_layer_params(self.model)
            self.layerparams = [i for i in self.layerparams if "downsample" not in i]
            if 'conv1.weight' in self.layerparams:
                self.layerparams.remove('conv1.weight')

    def prune_NM(
        self,
        N=2,
        M=4,
        first_layer_flag=False,
        k_step=3,
        w_warm=True,
        device='cuda:0',
        save_memory=False,
        batching=True,
        batch_size=128,
        max_CG_iterations=500,
        mask_alg="MP",
        stagnation_threshold=0.995
    ):
        print("Starting N:M pruning")
        memory_device = 'cpu' if save_memory else device
        zero_grads(self.model)
        self.model.eval()
        self.get_params()
        self.get_size()
        original_weight = get_pvec(self.model, self.layerparams)

        if memory_device == 'cpu':
            w_layer = original_weight.cpu().numpy()
        else:
            w_layer = original_weight.to(memory_device).detach().cpu().numpy()

        w_prune = np.copy(w_layer)
        torch.manual_seed(self.gradseed)
        torch.cuda.manual_seed(self.gradseed)
        torch.cuda.manual_seed_all(self.gradseed)
        np.random.seed(self.gradseed)
        train_dataloader = DataLoader(self.train_dataset, batch_size=1, shuffle=True, num_workers=10, pin_memory=True)
        xdata = torch.zeros([self.nsamples] + self.datasize).to(memory_device)

        for i, batch in enumerate(train_dataloader):
            xdata_i, ydata_i = batch
            xdata_i = xdata_i.to(memory_device)
            xdata[i, :, :,] = xdata_i
            if (i + 1) % self.nsamples == 0:
                break

        xdata2 = copy.deepcopy(xdata)
        i_w = 0
        i_layer = 0
        block_list = get_blocks(copy.deepcopy(self.model))
        self.model = self.model.to(memory_device)
        model_update = copy.deepcopy(self.model).to(memory_device)
        update_dict = model_update.state_dict()
        accuracies = []
        losses = []
        layer_times = []
        layer_wise_loss = {}
        layer_wise_W = {}
        layer_wise_size = {}
        total_blocks = len(block_list) - 1

        print("Computing initial metrics")
        dense_accuracy, dense_loss = compute_metrics(self.model, self.test_dataloader, criterion=nn.CrossEntropyLoss(), device='cuda', verbose=False, n_samples=10000, seed=42)
        print(f"Initial OOS Acc, OOS Loss: {dense_accuracy, dense_loss}")
        accuracies.append(dense_accuracy)
        losses.append(dense_loss)

        for name, block in block_list:
            print(f"Processing block: {name}")
            start_data = time.time()
            block = block.to(memory_device)
            with torch.no_grad():
                prune_list = find_all_module(block, self.layerparams, name, [])
                if not prune_list:
                    if batching:
                        xdata, xdata2 = forward_pass_in_batches(block, block, xdata, xdata2, batch_size, device='cuda')
                    else:
                        xdata = block(xdata)
                        xdata2 = block(xdata2)
                    continue

            block_update = copy.deepcopy(block)
            if w_warm:
                print("Warming block weights")
                forward_pass_in_batches_no_return(block_update, xdata, batch_size, device, batch_size)

            for cur_i in range(len(prune_list)):
                cur_module = prune_list[cur_i]
                start_data = time.time()
                prune_flag, prune_module = find_module(block_update, cur_module, name)
                param_size = self.size_list[i_layer]

                if isinstance(prune_module, torch.nn.Conv2d):
                    d_out, d_in, k_h, k_w = param_size
                elif isinstance(prune_module, torch.nn.Linear):
                    d_row, d_col = param_size

                if w_warm or mask_alg != "MP":
                    hook_handle = prune_module.register_forward_hook(self.getinput_hook)
                    unfold = nn.Unfold(prune_module.kernel_size, dilation=prune_module.dilation, padding=prune_module.padding, stride=prune_module.stride)
                    self.input_buffer = []
                    forward_pass_in_batches_no_return(block_update, xdata, batch_size, device, self.nsamples)
                    inp = np.vstack([unfold(inss).permute([1, 0, 2]).flatten(1).to("cpu").numpy() for inss in self.input_buffer])
                    hook_handle.remove()
                    prune_flag, prune_module = find_module(block, cur_module, name)
                    hook_handle = prune_module.register_forward_hook(self.getinput_hook)
                    self.input_buffer = []
                    forward_pass_in_batches_no_return(block, xdata2, batch_size, device, self.nsamples)
                    inp2 = np.vstack([unfold(inss).permute([1, 0, 2]).flatten(1).to("cpu").numpy() for inss in self.input_buffer])
                    inss_2 = self.input_buffer[0]
                    hook_handle.remove()

                    count = np.prod(param_size)
                    w_var = np.copy(w_layer[i_w:i_w + count]).reshape(param_size[0], -1).T
                    XTX = inp @ inp.T / self.nsamples
                    XTX += self.lambda_inv * np.eye(XTX.shape[0]) * np.mean(np.diag(XTX))
                    XTY = (inp @ inp2.T) @ w_var / self.nsamples
                else:
                    count = np.prod(param_size)
                    w_var = np.copy(w_layer[i_w:i_w + count]).reshape(param_size[0], -1).T

                conv_layer = extract_conv_layer(cur_module)

                if self.algo == "MP":
                    print("Using MP algorithm (N:M)")
                    w_sol = MP_MN(
                        w_var,
                        generate_M_sized_groups(d_in, k_h, k_w, M) if isinstance(prune_module, torch.nn.Conv2d) else generate_M_sized_groups_fc(d_col, M),
                        N
                    )
                elif self.algo == "MP+":
                    print("Using MP+ algorithm (N:M)")
                    w_sol = MP_MN(
                        w_var,
                        generate_M_sized_groups(d_in, k_h, k_w, M) if isinstance(prune_module, torch.nn.Conv2d) else generate_M_sized_groups_fc(d_col, M),
                        N
                    )
                    w_sol = weight_update_unstr_torch(w_sol, XTX, XTY)
                elif self.algo == "SNOWS":
                    print("Using SNOWS algorithm (N:M)")
                    if mask_alg == "MP":
                        w_sol = MP_MN(
                            w_var,
                            generate_M_sized_groups(d_in, k_h, k_w, M) if isinstance(prune_module, torch.nn.Conv2d) else generate_M_sized_groups_fc(d_col, M),
                            N
                        )
                        mask = (torch.abs(torch.tensor(w_sol)) > 0).float()
                    elif mask_alg == "FS":
                        W_MP = MP_MN(
                            w_var,
                            generate_M_sized_groups(d_in, k_h, k_w, M) if isinstance(prune_module, torch.nn.Conv2d) else generate_M_sized_groups_fc(d_col, M),
                            N
                        )
                        W_MP = weight_update_unstr_torch(W_MP, XTX, XTY)
                        w_sol, _ = forward_selection(w_var, W_MP, XTX, XTY, generate_M_sized_groups(d_in, k_h, k_w, M) if isinstance(prune_module, torch.nn.Conv2d) else None, N, num_cycles=1)
                        w_sol = weight_update_unstr_torch(w_sol, XTX, XTY)
                    elif mask_alg == "Custom":
                        w_sol = self.custom_weights[cur_module]
                        mask = self.custom_masks[cur_module]
                        w_sol = w_sol * mask
                    else:
                        w_sol = MP_MN(
                            w_var,
                            generate_M_sized_groups(d_in, k_h, k_w, M) if isinstance(prune_module, torch.nn.Conv2d) else generate_M_sized_groups_fc(d_col, M),
                            N
                        )
                        mask = (torch.abs(torch.tensor(w_sol)) > 0).float()

                    if w_warm:
                        w_sol = weight_update_unstr_torch(w_sol, XTX, XTY)
                        del XTX, XTY

                    w_sol = torch.tensor(w_sol)
                    if mask_alg != "Custom":
                        w_sol = w_sol.T.reshape(-1).reshape(param_size).to(device).to(torch.float32)
                    mask = (torch.abs(w_sol) > 0).float()
                    block_count = get_block_number_for_param(self.layerparams, cur_module)
                    w_sol, _ = solve_for_W_given_Z(
                        Z=mask,
                        model=self.model,
                        model_update=model_update,
                        xdata=xdata,
                        block_number=block_count,
                        k_step=k_step,
                        total_blocks=total_blocks,
                        block_list=block_list,
                        device=device,
                        N=N,
                        M=M,
                        w_warm=w_sol,
                        conv_layer=conv_layer,
                        batch_size=batch_size,
                        batching=batching,
                        CG_iterations=max_CG_iterations,
                        stagnation_threshold=stagnation_threshold
                    )
                else:
                    w_sol = w_var

                w_sol_shape = w_sol.shape
                if w_sol_shape == param_size:
                    w_output = w_sol.detach().cpu().numpy()
                    w_prune[i_w:i_w + count] = w_output.flatten()
                else:
                    w_output = np.copy((w_sol.T).reshape(-1))
                    w_prune[i_w:i_w + count] = np.copy((w_sol.T).reshape(-1))

                i_w += count
                i_layer += 1
                state_dict = block_update.state_dict()
                if cur_module.startswith(name + "."):
                    param_local = copy.deepcopy(cur_module[len(name + "."):])
                else:
                    param_local = copy.deepcopy(cur_module)
                original_shape = state_dict[param_local].shape
                state_dict[param_local] = torch.Tensor(w_output).reshape(original_shape).to("cpu")
                block_update.load_state_dict(state_dict)
                hook_handle = prune_module.register_forward_hook(self.getinput_hook)

            accuracy, loss = compute_metrics(model_update, self.test_dataloader, criterion=nn.CrossEntropyLoss(), device='cuda', verbose=False, n_samples=10000, seed=42)
            print(f"Block {name} -> OOS Loss: {loss}, OOS Acc: {accuracy}")
            accuracies.append(accuracy)
            losses.append(loss)
            elapsed_time = time.time() - start_data
            layer_times.append(elapsed_time)
            start_data = time.time()
            with torch.no_grad():
                xdata, xdata2 = forward_pass_in_batches(block_update, block, xdata, xdata2, batch_size, device='cuda')
                state_dict = block_update.state_dict()
                update_list = find_all_module(block_update, self.params, name, [])
                for upd_i in range(len(update_list)):
                    upd_module = update_list[upd_i]
                    if upd_module.startswith(name + "."):
                        param_local = copy.deepcopy(upd_module[len(name + "."):])
                    else:
                        param_local = copy.deepcopy(upd_module)
                    update_dict[upd_module] = copy.deepcopy(state_dict[param_local])
            model_update.load_state_dict(update_dict)

        model_update.load_state_dict(update_dict)
        final_acc, final_loss = compute_metrics(model_update, self.test_dataloader, criterion=nn.CrossEntropyLoss(), device='cuda', verbose=False, n_samples=10000, seed=42)
        print(f"Final OOS Loss: {final_loss}, OOS Acc: {final_acc}")
        accuracies.append(final_acc)
        self.results['sparsity_true'] = calculate_model_sparsity(model_update)
        self.results['test_acc'] = final_acc
        self.results['test_loss'] = final_loss
        self.results['accuracies'] = accuracies
        self.results['losses'] = losses
        self.results['layer_times'] = layer_times

        return model_update, accuracies, losses, layer_times, layer_wise_loss, layer_wise_W, layer_wise_size

    def prune_unstructured_global(
        self,
        target_sparsity=90.0,
        max_layer_sparsity=95.0,
        first_layer_flag=False,
        k_step=3,
        w_warm=True,
        device='cuda:0',
        save_memory=False,
        batching=False,
        batch_size=128,
        max_CG_iterations=500,
        mask_alg="MP",
        stagnation_threshold=0.995,
        max_steps=np.inf
    ):
        print("Starting global unstructured pruning")
        memory_device = 'cpu' if save_memory else device
        zero_grads(self.model)
        self.model.eval()
        self.get_params()
        self.get_size()
        original_weight = get_pvec(self.model, self.layerparams)

        if memory_device == 'cpu':
            w_layer = original_weight.cpu().numpy()
        else:
            w_layer = original_weight.to(memory_device).detach().cpu().numpy()

        w_prune = np.copy(w_layer)
        torch.manual_seed(self.gradseed)
        torch.cuda.manual_seed(self.gradseed)
        torch.cuda.manual_seed_all(self.gradseed)
        np.random.seed(self.gradseed)
        train_dataloader = DataLoader(self.train_dataset, batch_size=1, shuffle=True, num_workers=10, pin_memory=True)
        xdata = torch.zeros([self.nsamples] + self.datasize).to(memory_device)

        for i, batch in enumerate(train_dataloader):
            xdata_i, ydata_i = batch
            xdata_i = xdata_i.to(memory_device)
            xdata[i, ...] = xdata_i
            if (i + 1) >= self.nsamples:
                break

        xdata2 = copy.deepcopy(xdata)
        total_weights_model = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        all_weights = []
        layer_weight_indices = {}
        layer_sizes = {}
        current_index = 0

        for name, param in self.model.named_parameters():
            if name in self.layerparams and 'weight' in name:
                weight = param.detach().cpu().numpy().flatten()
                all_weights.append(weight)
                layer_weight_indices[name] = (current_index, current_index + weight.size)
                layer_sizes[name] = weight.size
                current_index += weight.size

        if not all_weights:
            raise ValueError("No weights found in layerparams for pruning.")

        all_weights_concatenated = np.concatenate(all_weights)
        total_weights_in_layerparams = all_weights_concatenated.size
        total_prune = int(round((target_sparsity / 100.0) * total_weights_model))
        max_possible_prune = sum(int(round((max_layer_sparsity / 100.0) * layer_sizes[name])) for name in layer_sizes)

        if max_possible_prune < total_prune:
            print("Warning: Cannot reach target sparsity with current layerparams and per-layer sparsity caps.")
            achievable_sparsity = (max_possible_prune / total_weights_model) * 100.0
            print(f"Maximum achievable sparsity with current layers: {achievable_sparsity:.2f}%")
            total_prune = max_possible_prune

        weight_info = []
        for name, (start, end) in layer_weight_indices.items():
            layer_weights = all_weights_concatenated[start:end]
            for idx, weight in enumerate(layer_weights):
                weight_info.append((abs(weight), name, start + idx))

        weight_info.sort(key=lambda x: x[0])
        pruned_per_layer = {name: 0 for name in layer_sizes}
        total_pruned = 0
        prune_mask = np.ones(total_weights_in_layerparams, dtype=bool)

        for weight_abs, layer_name, global_idx in weight_info:
            if total_pruned >= total_prune:
                break
            if pruned_per_layer[layer_name] < int(round((max_layer_sparsity / 100.0) * layer_sizes[layer_name])):
                prune_mask[global_idx] = False
                pruned_per_layer[layer_name] += 1
                total_pruned += 1

        pruned_weights = all_weights_concatenated * prune_mask
        per_layer_sparsity = {}
        for name, (start, end) in layer_weight_indices.items():
            layer_weights = pruned_weights[start:end]
            layer_sparsity = (1 - np.count_nonzero(layer_weights) / layer_weights.size) * 100.0
            per_layer_sparsity[name] = layer_sparsity
            print(f"Layer {name}: Sparsity {layer_sparsity:.2f}% (Capped at {max_layer_sparsity}%)")

        overall_sparsity = (total_pruned / total_weights_model) * 100.0
        print(f"Overall sparsity achieved: {overall_sparsity:.2f}% (Target: {target_sparsity}%)")
        block_list = get_blocks(copy.deepcopy(self.model))
        self.model = self.model.to(memory_device)
        model_update = copy.deepcopy(self.model).to(memory_device)
        update_dict = model_update.state_dict()
        accuracies = []
        losses = []
        layer_times = []
        layer_wise_loss = {}
        layer_wise_W = {}
        layer_wise_size = {}
        total_blocks = len(block_list) - 1

        print("Computing initial metrics for global unstructured pruning")
        dense_accuracy, dense_loss = compute_metrics(
            self.model,
            self.test_dataloader,
            criterion=nn.CrossEntropyLoss(),
            device=memory_device,
            verbose=False,
            n_samples=10000,
            seed=42
        )
        print(f"Initial OOS Acc, OOS Loss: {dense_accuracy, dense_loss}")
        accuracies.append(dense_accuracy)
        losses.append(dense_loss)
        i_layer = 0
        i_w = 0

        for name, block in block_list:
            print(f"Processing block: {name}")
            start_data = time.time()
            block = block.to(memory_device)
            with torch.no_grad():
                prune_list = find_all_module(block, self.layerparams, name, [])
                if not prune_list:
                    if batching:
                        xdata, xdata2 = forward_pass_in_batches(block, block, xdata, xdata2, batch_size, memory_device)
                    else:
                        xdata = block(xdata)
                        xdata2 = block(xdata2)
                    continue

            block_update = copy.deepcopy(block)
            if w_warm:
                print("Warming block weights (global unstructured)")
                forward_pass_in_batches_no_return(block_update, xdata, batch_size, memory_device)

            for cur_i in range(len(prune_list)):
                cur_module = prune_list[cur_i]
                start_data = time.time()
                prune_flag, prune_module = find_module(block_update, cur_module, name)
                is_depthwise = False
                if isinstance(prune_module, nn.Conv2d):
                    is_depthwise = prune_module.groups == prune_module.in_channels == prune_module.out_channels

                if i_layer >= len(self.size_list):
                    raise IndexError(f"i_layer ({i_layer}) is out of bounds for size_list with length {len(self.size_list)}")

                param_size = self.size_list[i_layer]
                if isinstance(prune_module, nn.Conv2d):
                    d_out, d_in, k_h, k_w = param_size
                elif isinstance(prune_module, nn.Linear):
                    d_out, d_in = param_size
                else:
                    raise NotImplementedError(f"Unsupported module type: {type(prune_module)}")

                if w_warm or mask_alg != "MP":
                    hook_handle = prune_module.register_forward_hook(self.getinput_hook)
                    if isinstance(prune_module, nn.Conv2d):
                        unfold = nn.Unfold(
                            prune_module.kernel_size,
                            dilation=prune_module.dilation,
                            padding=prune_module.padding,
                            stride=prune_module.stride
                        )
                    self.input_buffer = []
                    forward_pass_in_batches_no_return(block_update, xdata, batch_size, memory_device)
                    if isinstance(prune_module, nn.Conv2d):
                        inp = np.vstack([
                            unfold(inss).permute([1, 0, 2]).flatten(1).cpu().numpy()
                            for inss in self.input_buffer
                        ])
                    elif isinstance(prune_module, nn.Linear):
                        inp = np.vstack([inss.permute([1, 0]).to("cpu").numpy() for inss in self.input_buffer])
                    else:
                        raise NotImplementedError(f"Unsupported module type: {type(prune_module)}")

                    hook_handle.remove()
                    prune_flag, prune_module = find_module(block, cur_module, name)
                    hook_handle = prune_module.register_forward_hook(self.getinput_hook)
                    self.input_buffer = []
                    forward_pass_in_batches_no_return(block, xdata2, batch_size, memory_device)
                    if isinstance(prune_module, nn.Conv2d):
                        inp2 = np.vstack([
                            unfold(inss).permute([1, 0, 2]).flatten(1).cpu().numpy()
                            for inss in self.input_buffer
                        ])
                    elif isinstance(prune_module, nn.Linear):
                        inp2 = np.vstack([inss.permute([1, 0]).to("cpu").numpy() for inss in self.input_buffer])
                    else:
                        raise NotImplementedError(f"Unsupported module type: {type(prune_module)}")

                    inss_2 = self.input_buffer[0]
                    hook_handle.remove()

                count = np.prod(param_size)
                w_var = np.copy(w_layer[i_w:i_w + count]).reshape(param_size[0], -1).T
                conv_layer = extract_conv_layer(cur_module)
                layer_name = f"{name}.{cur_module}" if not cur_module.startswith(name + ".") else cur_module
                layer_sparsity = per_layer_sparsity.get(layer_name, target_sparsity)
                print(f"Pruning layer {layer_name} with {layer_sparsity:.2f}% sparsity")

                if self.algo == "MP":
                    w_sol = MP_unstr(w_var, 100.0 - layer_sparsity)
                elif self.algo == "MP+":
                    w_sol = MP_unstr(w_var, 100.0 - layer_sparsity)
                    w_sol = weight_update_unstr_torch(w_sol, XTX, XTY)
                elif self.algo == "FS":
                    W_MP = MP_unstr(w_var, 100.0 - layer_sparsity)
                    W_MP = weight_update_unstr_torch(W_MP, XTX, XTY)
                    w_sol, _ = forward_selection(w_var, W_MP, XTX, XTY, groups=None, N=100.0 - layer_sparsity, num_cycles=1)
                    w_sol = weight_update_unstr_torch(w_sol, XTX, XTY)
                elif self.algo == "Backward":
                    if not is_depthwise:
                        w_sol = backward_selection_all_unstr(w_var, XTX, XTY, int(w_var.size * (1 - layer_sparsity / 100)))
                    else:
                        w_sol = MP_unstr(w_var, 100.0 - layer_sparsity)
                elif self.algo == "SNOWS":
                    if mask_alg == "MP":
                        w_sol = MP_unstr(w_var, 100.0 - layer_sparsity)
                        mask = (torch.abs(torch.tensor(w_sol)) > 0).float()
                    elif mask_alg == "FS":
                        W_MP = MP_unstr(w_var, 100.0 - layer_sparsity)
                        W_MP = weight_update_unstr_torch(W_MP, XTX, XTY)
                        w_sol, _ = forward_selection(w_var, W_MP, XTX, XTY, groups=None, N=100.0 - layer_sparsity, num_cycles=1)
                        w_sol = weight_update_unstr_torch(w_sol, XTX, XTY)
                    elif mask_alg == "Custom":
                        w_sol = self.custom_weights[cur_module]
                        mask = self.custom_masks[cur_module]
                        w_sol = w_sol * mask
                    elif mask_alg == "Backward":
                        if not is_depthwise:
                            w_sol = backward_selection_all_unstr(w_var, XTX, XTY, int(w_var.size * (1 - layer_sparsity / 100)))
                    if (w_warm and not is_depthwise) or 'conv_layer' == 'fc':
                        w_sol = weight_update_unstr_torch(w_sol, XTX, XTY)
                        del XTX, XTY
                    else:
                        w_sol = MP_unstr(w_var, 100.0 - layer_sparsity)
                        mask = (torch.abs(torch.tensor(w_sol)) > 0).float()
                    w_sol = torch.tensor(w_sol).to(memory_device)
                    if mask_alg != "Custom":
                        w_sol = w_sol.T.reshape(-1).reshape(param_size).to(device).to(torch.float32)
                    if mask_alg != "Custom":
                        mask = (torch.abs(w_sol) > 0).float()
                        w_sol = w_sol * mask
                    if 'fc' not in conv_layer:
                        w_sol, _ = solve_for_W_given_Z(
                            Z=mask,
                            model=self.model,
                            model_update=model_update,
                            xdata=xdata,
                            block_number=get_block_number_for_param(self.layerparams, cur_module),
                            k_step=k_step,
                            total_blocks=total_blocks,
                            block_list=block_list,
                            device=memory_device,
                            N=None,
                            M=None,
                            w_warm=w_sol,
                            conv_layer=extract_conv_layer(cur_module),
                            batch_size=batch_size,
                            batching=batching,
                            CG_iterations=max_CG_iterations,
                            max_steps=max_steps
                        )
                    else:
                        w_sol = w_var
                else:
                    w_sol = w_var

                w_sol = torch.tensor(w_sol).to(memory_device)
                w_sol_shape = w_sol.shape
                if w_sol_shape == param_size:
                    w_output = w_sol.detach().cpu().numpy()
                else:
                    w_output = np.copy((w_sol.T).reshape(-1))
                i_w += count
                i_layer += 1
                state_dict = block_update.state_dict()
                if cur_module.startswith(name + "."):
                    param_local = cur_module[len(name) + 1:]
                else:
                    param_local = cur_module
                original_shape = state_dict[param_local].shape
                state_dict[param_local] = torch.Tensor(w_output).reshape(original_shape).to("cpu")
                block_update.load_state_dict(state_dict)
                hook_handle = prune_module.register_forward_hook(self.getinput_hook)

            accuracy, loss = compute_metrics(
                model_update,
                self.test_dataloader,
                criterion=nn.CrossEntropyLoss(),
                device=memory_device,
                verbose=False,
                n_samples=10000,
                seed=42
            )
            print(f"Block {name} -> OOS Loss: {loss}, OOS Acc: {accuracy}")
            accuracies.append(accuracy)
            losses.append(loss)
            elapsed_time = time.time() - start_data
            layer_times.append(elapsed_time)
            start_data = time.time()
            with torch.no_grad():
                xdata, xdata2 = forward_pass_in_batches(block_update, block, xdata, xdata2, batch_size, memory_device)
                state_dict = block_update.state_dict()
                update_list = find_all_module(block_update, self.params, name, [])
                for upd_i in range(len(update_list)):
                    upd_module = update_list[upd_i]
                    if upd_module.startswith(name + "."):
                        param_local = upd_module[len(name) + 1:]
                    else:
                        param_local = upd_module
                    update_dict[upd_module] = state_dict[param_local].clone()
            model_update.load_state_dict(update_dict)

        model_update.load_state_dict(update_dict)
        final_acc, final_loss = compute_metrics(
            model_update,
            self.test_dataloader,
            criterion=nn.CrossEntropyLoss(),
            device=memory_device,
            verbose=False,
            n_samples=10000,
            seed=42
        )
        print(f"Final OOS Loss: {final_loss}, OOS Acc: {final_acc}")
        accuracies.append(final_acc)
        self.results['sparsity_true'] = calculate_model_sparsity(model_update)
        self.results['test_acc'] = final_acc
        self.results['test_loss'] = final_loss
        self.results['accuracies'] = accuracies
        self.results['losses'] = losses
        self.results['layer_times'] = layer_times
        return model_update, accuracies, losses, layer_times, layer_wise_loss, layer_wise_W, layer_wise_size


def calculate_model_sparsity(model):
    total_zeros = 0
    total_params = 0
    for param in model.parameters():
        if param.requires_grad:
            total_zeros += torch.sum(param == 0).item()
            total_params += param.numel()
    total_sparsity = total_zeros / total_params if total_params != 0 else 0
    return total_sparsity


def forward_pass_in_batches(block1, block2, xdata, xdata2, batch_size, device='cuda'):
    new_xdata, new_xdata2 = None, None
    with torch.no_grad():
        for i in range(0, xdata.size(0), batch_size):
            xdata_batch = xdata[i:i + batch_size].to(device)
            xdata2_batch = xdata2[i:i + batch_size].to(device)
            xdata_batch_out = block1(xdata_batch)
            xdata2_batch_out = block2(xdata2_batch)
            if new_xdata is None:
                new_xdata = torch.zeros((xdata.size(0), *xdata_batch_out.shape[1:])).to('cpu')
            if new_xdata2 is None:
                new_xdata2 = torch.zeros((xdata2.size(0), *xdata2_batch_out.shape[1:])).to('cpu')
            new_xdata[i:i + batch_size] = xdata_batch_out.to('cpu')
            new_xdata2[i:i + batch_size] = xdata2_batch_out.to('cpu')
    return new_xdata, new_xdata2


def forward_pass_in_batches_no_return(block, xdata, batch_size, device='cuda', max_examples=None):
    total_examples = xdata.size(0)
    if max_examples is not None:
        total_examples = min(total_examples, max_examples)
    with torch.no_grad():
        for i in range(0, total_examples, batch_size):
            xdata_batch = xdata[i:i + batch_size].to(device)
            block(xdata_batch)
            del xdata_batch
            torch.cuda.empty_cache()
            if max_examples is not None and i + batch_size >= max_examples:
                break
    return None


def collect_layer_params(model):
    layerparams = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            layerparams.append(f"{name}.weight")
        if hasattr(module, 'downsample') and isinstance(module.downsample, nn.Sequential):
            for downsample_name, downsample_module in module.downsample.named_children():
                if isinstance(downsample_module, nn.Conv2d):
                    layerparams.append(f"{name}.downsample.{downsample_name}.weight")
    return layerparams
