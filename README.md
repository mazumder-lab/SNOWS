# SNOWS Official Codebase

This repository contains the official code for the ICLR 2025 paper:  
**Preserving Deep Representations in One-Shot Pruning: A Hessian-Free Second-Order Optimization Framework**

SNOWS is a one-shot, post-training pruning framework aimed at reducing the cost of deep vision model inference without retraining. Unlike existing one-shot methods that focus on layer-wise reconstruction, SNOWS optimizes a more global reconstruction objective that captures deep, nonlinear feature representations. We solve this challenging objective efficiently via second-order (Hessian-free) optimization.

---

## Repository Structure

```
.
├── CNN_pruning/
├── VIT_pruning/
└── requirements.txt
```

- **CNN_pruning/**: Contains scripts for pruning CNN architectures (e.g., ResNet-20 on CIFAR-10, ResNet-50 on CIFAR-100).  
- **VIT_pruning/**: Contains scripts for pruning Vision Transformers (e.g., ViT-B/16 on ImageNet).  
- **requirements.txt**: List of Python dependencies.

---

## Installation

1. Clone this repository.
2. Install the necessary dependencies:

   ```
   pip install -r requirements.txt
   ```

---

## Usage

### Pruning CNNs

Inside the **CNN_pruning** folder, you will find various scripts to prune CNNs.

**Example: Unstructured pruning** (e.g., ResNet-20 on CIFAR-10)
```
python unstr.py \
--arch resnet20_cifar10 \
--dset cifar10 \
--batch_size 512 \
--batching True \
--algo SNOWS \
--k_step 20 \
--ngrads 3000 \
--target_sparsity 0.7 \
--max_layer_sparsity 0.8
```

**Example: N:M structured pruning** (e.g., ResNet-50 on CIFAR-100)
```
python prune_NM.py \
--arch resnet50_cifar100 \
--dset cifar100 \
--ngrads 3000 \
--algo SNOWS \
--NM_N 1 \
--NM_M 4 \
--batch_size 128 \
--batching True \
--k_step 30 \
--w_warm False
```

---

### Pruning Vision Transformers

Inside the **VIT_pruning** folder, you will find scripts to prune Vision Transformers (e.g., ViT-B/16 on ImageNet):

```
python prune_NM.py \
--arch vit_b_16 \
--dset imagenet \
--test_batch_size 128 \
--ngrads 256 \
--seed 42 \
--algo SNOWS \
--NM_N 2 \
--NM_M 4 \
--layers_to_prune in_proj out_proj mlp \
--k_step 3 \
--max_CG_iterations 500
```

---

## Important Arguments

- **--ngrads**  
  Defines the size of the calibration dataset used for pruning (i.e., total number of samples). More samples generally improve performance but increase run-time linearly with the number of samples.

- **--k_step**  
  Corresponds to the \(K\) parameter from the paper (number of lookahead operations). Increasing \(K\) generally leads to better performance at the cost of higher run-time and memory consumption.

- **--max_CG_iterations**  
  Maximum number of conjugate gradient steps when pruning a given layer. While larger values can yield more precise solutions, most of the progress usually occurs in the first few iterations; therefore, reducing this parameter has a relatively small impact on performance but significantly reduces run-time.

- **--batching**  
  Determines whether to enable mini-batching for the Hessian-free updates. Set this to True in most cases (especially when your calibration dataset is large), as it helps manage memory usage effectively.

---

## Citation

If you find this code helpful in your research, please cite:

```
@misc{lucas2024preservingdeeprepresentationsoneshot,
      title={Preserving Deep Representations In One-Shot Pruning: A Hessian-Free Second-Order Optimization Framework}, 
      author={Ryan Lucas and Rahul Mazumder},
      year={2024},
      eprint={2411.18376},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2411.18376}, 
}
```

---

## License

This project is released under the MIT License. 

For questions or issues, feel free to contact `ryanlu@mit.edu`.