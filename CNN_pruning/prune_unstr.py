#!/usr/bin/env python3

import os
import sys
import time
import copy
import pickle
import argparse
import torch
from torch.utils.data import DataLoader

# Example imports from your local files/modules
# Make sure `prune/utils.py` and `prune/Layer_pruner_new.py` exist and are accessible
from prune.utils import model_factory, set_seed
from prune.Layer_pruner_new import LayerPruner

# In case you have MFAC in a subdirectory named MFAC; adjust as necessary
MFACPATH = 'MFAC'
sys.path.append(MFACPATH)


def parse_args_with_kwargs(**kwargs):
    """
    Parse command-line arguments if no kwargs are provided.
    If kwargs are provided, override default arguments accordingly.
    """

    parser = argparse.ArgumentParser(description='Layer Pruning Script with optional kwargs.')

    # Add all arguments
    parser.add_argument('--arch', type=str, default='resnet20')
    parser.add_argument('--dset', type=str, default='cifar10')
    parser.add_argument('--num_workers', type=int, default=40)
    parser.add_argument('--exp_name', type=str, default='')
    parser.add_argument('--exp_id', type=str, default='')
    parser.add_argument('--shuffle_train', type=lambda x: (str(x).lower() == 'true'), default=False)
    parser.add_argument('--test_batch_size', type=int, default=1000)
    parser.add_argument('--ngrads', type=int, default=3000)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--lambda_inv', type=float, default=0.0001)
    parser.add_argument('--algo', type=str, default='SNOWS')  
    parser.add_argument('--lambda_2', type=float, default=0.001)

    # Single values for target_sparsity and max_layer_sparsity
    parser.add_argument('--target_sparsity', type=float, default=0.7,
                        help='Overall target sparsity (e.g. 0.7 means 70% of weights pruned).')
    parser.add_argument('--max_layer_sparsity', type=float, default=0.9,
                        help='Maximum per-layer sparsity (e.g. 0.9 means any layer can be at most 90% pruned).')

    # Make the previously hard-coded values into command line arguments
    parser.add_argument('--batch_size', type=int, default=1024,
                        help='Batch size for training/gradient samples.')
    parser.add_argument('--batching', type=lambda x: (str(x).lower() == 'true'), default=True,
                        help='Boolean: whether to use batching.')
    parser.add_argument('--k_step', type=int, default=40,
                        help='Number of steps or blocks for the pruning algorithm.')
    parser.add_argument('--w_warm', type=lambda x: (str(x).lower() == 'true'), default=True,
                        help='Boolean: if True, warm up weights or some part of the process before pruning.')
    parser.add_argument('--max_CG_iterations', type=int, default=500,
                        help='Maximum Conjugate Gradient iterations during pruning.')

    if not kwargs:
        args = parser.parse_args()
    else:
        # Create a namespace of defaults (but do not parse sys.argv)
        defaults = vars(parser.parse_args([]))
        # Update defaults with kwargs
        for key, value in kwargs.items():
            if key in defaults:
                defaults[key] = value
            else:
                raise ValueError(f"Invalid argument: {key}")
        # Convert back to namespace
        args = argparse.Namespace(**defaults)
    return args


def main(**kwargs):
    """
    Main function that can be called:
      - from command line:  python prune_script.py --arch <ARCH> ...
      - from Python:        main(arch='resnet20_cifar10', dset='cifar10', ...)
    """
    args = parse_args_with_kwargs(**kwargs)

    # Setup environment paths (example: adjust as needed)
    os.environ['IMAGENET_PATH'] = './datasets/MiniImageNet/imagenet-mini'
    dset_paths = {
        'imagenet': os.environ['IMAGENET_PATH'],
        'cifar10': './datasets',
        'cifar100': './datasets'
    }

    # If dataset is unknown, handle accordingly; for now assume valid
    dset_path = dset_paths.get(args.dset, './datasets')

    ROOT_PATH = './Sparse_NN'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Using device: {device}')

    # We'll store final accuracies in a dictionary for convenience
    accuracies_table = {}

    # Directly use the arguments without architecture-specific logic:
    batch_size = args.batch_size
    batching = args.batching
    k_step = args.k_step
    w_warm = args.w_warm
    max_CG_iterations = args.max_CG_iterations

    # Print out the config for clarity
    print("=== Configuration ===")
    print(f"  Architecture        : {args.arch}")
    print(f"  Dataset            : {args.dset}")
    print(f"  Batch size         : {batch_size}")
    print(f"  Batching           : {batching}")
    print(f"  k_step             : {k_step}")
    print(f"  ngrads             : {args.ngrads}")
    print(f"  w_warm             : {w_warm}")
    print(f"  max_CG_iterations  : {max_CG_iterations}")
    print("=====================")

    # Model & dataset setup (make sure model_factory is properly defined in prune/utils.py)
    model, train_dataset, test_dataset, criterion, modules_to_prune = model_factory(args.arch, dset_path)

    # Create dataloaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=args.shuffle_train,
        num_workers=args.num_workers,
        pin_memory=True
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )

    model.to(device)
    model.eval()

    # Single run with one target_sparsity and one max_layer_sparsity
    target_sparsity = args.target_sparsity
    max_layer_sparsity = args.max_layer_sparsity
    print(f'Running with target_sparsity: {target_sparsity} and max_layer_sparsity: {max_layer_sparsity}')

    # Fix seed
    set_seed(args.seed)

    # Copy model for pruning
    model_pruned = copy.deepcopy(model)

    # Initialize pruner
    pruner = LayerPruner(
        model=model_pruned,
        params=modules_to_prune,
        train_dataset=train_dataset,
        train_dataloader=train_dataloader,
        test_dataloader=test_dataloader,
        nsamples=args.ngrads,
        criterion=criterion,
        lambda_inv=args.lambda_inv,
        gradseed=args.seed,
        device=device,
        algo=args.algo,
    )

    pruner.scaled = True

    # Prune (note: target_sparsity & max_layer_sparsity as percentages)
    model_update, accuracies, losses, layer_times, layer_wise_loss, layer_wise_W, layer_wise_size = pruner.prune_unstructured_global(
        target_sparsity=target_sparsity * 100,
        max_layer_sparsity=max_layer_sparsity * 100,
        device=device,
        save_memory=False,
        k_step=k_step,
        w_warm=w_warm,
        batching=batching,
        batch_size=batch_size,
        max_CG_iterations=max_CG_iterations,
        mask_alg="MP",
        stagnation_threshold=0.995
    )

    # Store the results
    accuracies_table[(target_sparsity, max_layer_sparsity)] = accuracies

    # Save pickled results
    os.makedirs("./Results", exist_ok=True)
    result_filename = f"./Results/{args.arch}_{args.algo}_{target_sparsity}_{max_layer_sparsity}_results.pickle"
    with open(result_filename, 'wb') as f:
        pickle.dump({'accuracies': accuracies, 'losses': losses}, f)

    # Save the pruned model
    os.makedirs("./Pruned_models", exist_ok=True)
    model_filename = f"./Pruned_models/{args.arch}_{args.algo}_{target_sparsity}_{max_layer_sparsity}_model.pth"
    torch.save(model_update.state_dict(), model_filename)

    print(f"Results and model saved for target_sparsity {target_sparsity} and max_layer_sparsity {max_layer_sparsity}")

    # Finally, save the aggregated accuracies
    with open("./Results/accuracies_table.pickle", 'wb') as f:
        pickle.dump(accuracies_table, f)

    print("Accuracies table saved.")


if __name__ == "__main__":
    main()
