#!/usr/bin/env python3

import os
import sys
import time
import copy
import json
import pickle
import argparse
from itertools import product

import torch
from torch.utils.data import DataLoader

# Local imports (adjust these as necessary for your environment)
from prune.utils import model_factory, set_seed
from prune.Layer_pruner_new import LayerPruner


def parse_args():
    parser = argparse.ArgumentParser(description='Script for N:M pruning.')

    # Basic arguments
    parser.add_argument('--arch', type=str, default='resnet20', help='Model architecture.')
    parser.add_argument('--dset', type=str, default='cifar10', help='Dataset name.')
    parser.add_argument('--num_workers', type=int, default=40, help='Number of DataLoader workers.')
    parser.add_argument('--exp_name', type=str, default='', help='Experiment name for bookkeeping.')
    parser.add_argument('--exp_id', type=str, default='', help='Experiment ID for bookkeeping.')
    parser.add_argument('--shuffle_train', type=lambda x: (str(x).lower() == 'true'), default=False,
                        help='If True, shuffle training data.')
    parser.add_argument('--test_batch_size', type=int, default=500, help='Batch size for testing/dataloaders.')
    parser.add_argument('--ngrads', type=int, default=128, help='Number of gradient samples.')
    parser.add_argument('--seed', type=int, nargs='+', default=[42], help='Random seed(s).')
    parser.add_argument('--lambda_inv', type=float, nargs='+', default=[0.0001], help='Lambda inverse values.')
    parser.add_argument('--algo', type=str, nargs='+', default=['SNOWS'], help='List of algorithms.')
    parser.add_argument('--NM_N', type=str, default='1', help='N in N:M pruning.')
    parser.add_argument('--NM_M', type=str, default='4', help='M in N:M pruning.')
    parser.add_argument('--lambda_2', type=float, default=0.001, help='(Unused in snippet) Additional lambda parameter.')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Batch size for the pruning process.')
    parser.add_argument('--batching', type=lambda x: (str(x).lower() == 'true'), default=True,
                        help='If True, use batch-wise forward passes.')
    parser.add_argument('--k_step', type=int, default=40,
                        help='Number of lookahead steps used in SNOWS.')
    parser.add_argument('--w_warm', type=lambda x: (str(x).lower() == 'true'), default=False,
                        help='If True, perform a warm-start backsolve for weights (w).')

    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    # Set up paths
    os.environ['IMAGENET_PATH'] = './datasets/MiniImageNet/imagenet-mini'
    dset_paths = {
        'imagenet': os.environ['IMAGENET_PATH'],
        'cifar10': './datasets',
        'cifar100': './datasets',
        'mnist': './datasets'
    }
    dset_path = dset_paths.get(args.dset, './datasets')

    ROOT_PATH = './Sparse_NN'
    FOLDER = f'{ROOT_PATH}/results/{args.arch}_{args.dset}_{args.exp_name}'
    os.makedirs(FOLDER, exist_ok=True)

    FILE = f'{FOLDER}/data{args.exp_id}_{int(time.time())}.csv'  # If you need a CSV for logs

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Using device: {device}')

    # Model setup
    model, train_dataset, test_dataset, criterion, modules_to_prune = model_factory(args.arch, dset_path)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.test_batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.test_batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )

    # If multiple GPUs are available, use DataParallel
    if torch.cuda.device_count() > 1:
        print(f'Using DataParallel with {torch.cuda.device_count()} GPUs')
        model = torch.nn.DataParallel(model)
        modules_to_prune = ["module." + x for x in modules_to_prune]

    model.to(device)
    model.eval()

    # Use user-specified architecture/hyperparameters from command line
    batch_size = args.batch_size
    batching = args.batching
    k_step = args.k_step
    w_warm = args.w_warm

    print("=== Configuration ===")
    print(f"  Architecture : {args.arch}")
    print(f"  Dataset      : {args.dset}")
    print(f"  Batch size   : {batch_size}")
    print(f"  Batching     : {batching}")
    print(f"  k_step       : {k_step}")
    print(f"  ngrads       : {args.ngrads}")
    print(f"  w_warm       : {w_warm}")
    print("====================")

    print("Algorithms to run:", args.algo)

    # Pruning and training across all combinations of seed, lambda_inv, and algo
    for seed, lambda_inv, algo in product(args.seed, args.lambda_inv, args.algo):
        print(f'\nRunning with seed: {seed}, lambda_inv: {lambda_inv}, algo: {algo}')
        
        set_seed(seed)
        model_pruned = copy.deepcopy(model)

        pruner = LayerPruner(
            model_pruned,
            modules_to_prune,
            train_dataset,
            train_dataloader,
            test_dataloader,
            args.ngrads,
            criterion,
            lambda_inv,
            seed,
            device,
            algo
        )
        pruner.scaled = True

        start_time = time.time()

        # Perform N:M pruning
        model_update, accuracies, losses, layer_times, layer_wise_loss, layer_wise_W, layer_wise_size = pruner.prune_NM(
            N=int(args.NM_N),
            M=int(args.NM_M),
            k_step=k_step,
            w_warm=w_warm,
            save_memory=False,
            batching=batching,
            batch_size=batch_size,
            max_CG_iterations=1000,
            mask_alg="MP"
        )

        # Save results
        results = {
            'accuracies': accuracies,
            'losses': losses
        }
        os.makedirs("./Results", exist_ok=True)
        result_filename = f"./Results/{args.arch}_{algo}_{args.NM_N}:{args.NM_M}_k={k_step}_results.pickle"
        
        total_time = time.time() - start_time
        results['total_time'] = total_time
        print(f"Total time for the entire process: {total_time:.2f} seconds")

        with open(result_filename, 'wb') as f:
            pickle.dump(results, f)

        # Save the pruned model
        os.makedirs("./Pruned_models", exist_ok=True)
        model_filename = f"./Pruned_models/{args.arch}_{algo}_{args.NM_N}:{args.NM_M}_k={k_step}_model.pth"
        torch.save(model_update.state_dict(), model_filename)

        print(f"Results saved to {result_filename} and model saved to {model_filename} in {time.time() - start_time:.2f} seconds")

        # Save layer-wise data
        os.makedirs("./Layer_wise", exist_ok=True)
        ls_filename = f"./Layer_wise/{args.arch}_{algo}_{args.NM_N}:{args.NM_M}_k={k_step}_layer_wise_data.pickle"

        layer_wise_data = {
            'layer_wise_losses': layer_wise_loss,
            'layer_wise_W': layer_wise_W,
            'layer_wise_size': layer_wise_size
        }

        with open(ls_filename, 'wb') as f:
            pickle.dump(layer_wise_data, f)

        print(f"Layer-wise data saved to {ls_filename}")


if __name__ == "__main__":
    main()
