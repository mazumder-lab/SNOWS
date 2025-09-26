#!/usr/bin/env python3

import os
import sys
import time
import copy
import pickle
import argparse
from itertools import product
import torch
from torch.utils.data import DataLoader

# Adjust these imports if needed
from prune.utils import model_factory, set_seed
from prune.Layer_pruner_vit import LayerPruner

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--arch', type=str, default='resnet20', help='Model architecture')
    parser.add_argument('--dset', type=str, default='cifar10', help='Dataset name')
    parser.add_argument('--num_workers', type=int, default=40, help='Number of DataLoader workers')
    parser.add_argument('--exp_name', type=str, default='', help='Experiment name')
    parser.add_argument('--exp_id', type=str, default='', help='Experiment ID')
    parser.add_argument('--test_batch_size', type=int, default=500, help='Batch size for testing')
    parser.add_argument('--ngrads', type=int, default=128, help='Number of gradients used for pruning')
    parser.add_argument('--seed', type=int, nargs='+', default=[42], help='Random seeds (space-separated list)')
    parser.add_argument('--algo', type=str, nargs='+', default=['Newton'], help='Pruning algorithm(s)')
    parser.add_argument('--NM_N', type=str, default='2', help='N parameter for N:M pruning')
    parser.add_argument('--NM_M', type=str, default='4', help='M parameter for N:M pruning')

    # Specify layers to prune from command line
    parser.add_argument(
        '--layers_to_prune',
        type=str,
        nargs='+',
        default=['in_proj', 'out_proj', 'mlp'],
        help='Layers to prune, space-separated. E.g.: --layers_to_prune in_proj out_proj mlp'
    )

    # NEW: Command-line options for k_step and max_CG_iterations
    parser.add_argument('--k_step', type=int, default=1, help='Number of Newton steps (or horizon) for pruning')
    parser.add_argument('--max_CG_iterations', type=int, default=500, help='Maximum Conjugate Gradient iterations')

    args = parser.parse_args()

    # Example environment setup for ImageNet mini:
    os.environ['IMAGENET_PATH'] = './imagenet-mini'
    dset_paths = {'imagenet': os.environ['IMAGENET_PATH']}
    dset_path = dset_paths.get(args.dset, './data')

    ROOT_PATH = './Sparse_NN'
    FOLDER = f'{ROOT_PATH}/results/{args.arch}_{args.dset}_{args.exp_name}'
    os.makedirs(FOLDER, exist_ok=True)
    FILE = f'{FOLDER}/data{args.exp_id}_{int(time.time())}.csv'

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Using device: {device}')

    # Model and dataset creation
    model, train_dataset, test_dataset, criterion, modules_to_prune = model_factory(args.arch, dset_path)
    train_dataloader = DataLoader(train_dataset,
                                  batch_size=args.test_batch_size,
                                  shuffle=True,
                                  num_workers=args.num_workers,
                                  pin_memory=True)
    test_dataloader = DataLoader(test_dataset,
                                 batch_size=args.test_batch_size,
                                 shuffle=True,
                                 num_workers=args.num_workers,
                                 pin_memory=True)

    if torch.cuda.device_count() > 1:
        print(f'Using DataParallel with {torch.cuda.device_count()} GPUs')
        model = torch.nn.DataParallel(model)
        modules_to_prune = ["module." + x for x in modules_to_prune]

    model.to(device)
    model.eval()

    # Extract k_step and max_CG_iterations from arguments
    k_step = args.k_step
    max_CG_iterations = args.max_CG_iterations

    # Optionally track loss histories
    loss_histories = []

    # Iterate over seeds and algorithms
    for seed, algo in product(args.seed, args.algo):
        print(f'Running with seed: {seed}, algo: {algo}, layers: {args.layers_to_prune}')
        set_seed(seed)
        model_pruned = copy.deepcopy(model)

        pruner = LayerPruner(
            model=model_pruned,
            params=modules_to_prune,
            train_dataset=train_dataset,
            test_dataloader=test_dataloader,
            nsamples=args.ngrads,
            criterion=criterion,
            gradseed=seed,
            device=device,
            algo=algo,
            layers_to_prune=args.layers_to_prune
        )

        start_time = time.time()
        print("N:M = ", int(args.NM_N), int(args.NM_M))

        # Perform the pruning
        model_update, accuracies, losses = pruner.prune_NM(
            N=int(args.NM_N),
            M=int(args.NM_M),
            newton_k_step=k_step,
            batch_size=128,
            newton_steps=1,
            max_CG_iterations=max_CG_iterations,
            save_memory=False
        )

        try:
            loss_histories = pruner.loss_histories
        except AttributeError:
            loss_histories = []

        # Save results
        results = {
            'accuracies': accuracies,
            'losses': losses,
            'loss_histories': loss_histories
        }
        filename = (
            f"{args.arch}_{algo}_{int(args.NM_N)}:{int(args.NM_M)}_"
            f"layers_{'_'.join(args.layers_to_prune)}_results_k{k_step}.pickle"
        )
        with open(filename, 'wb') as f:
            pickle.dump(results, f)
        print(f"Results saved as {filename}")

        # Save pruned model
        model_filename = (
            f"{args.arch}_{algo}_{int(args.NM_N)}:{int(args.NM_M)}_"
            f"layers_{'_'.join(args.layers_to_prune)}_pruned_model_k{k_step}.pth"
        )
        torch.save(model_update.state_dict(), model_filename)
        print(f"Pruned model saved as {model_filename}")

        elapsed_time = time.time() - start_time
        print(f"Total time for this configuration: {elapsed_time:.2f} seconds")

if __name__ == "__main__":
    main()
