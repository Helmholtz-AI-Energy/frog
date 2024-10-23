# Beyond Backpropagation: Optimization with Multi-Tangent Forward Gradients

[![](https://img.shields.io/badge/License-MIT-8CB423)](./LICENSE)
[![](https://img.shields.io/badge/Python-3.9+-8CB423)](https://www.python.org/downloads/)
[![](https://img.shields.io/badge/Contact-katharina.fluegel%40kit.edu-8CB423)](mailto:katharina.fluegel@kit.edu)

The gradients used to train neural networks are typically computed using backpropagation. 
While an efficient way to obtain exact gradients, backpropagation is computationally expensive, hinders parallelization, and is biologically implausible. 
Forward gradients are an approach to approximate the gradients from directional derivatives along random tangents computed by forward-mode automatic differentiation. 
So far, research has focused on using a single tangent per step. 
However, we find that aggregation over multiple tangents improves both approximation quality and optimization performance across various tasks.

This repository contains the source code, instructions on how to reproduce our results, and the learning rates as CSV for our paper "Beyond Backpropagation: Optimization with Multi-Tangent Forward Gradients", which provides an in-depth analysis of multi-tangent forward gradients and introduces an improved approach to combining the forward gradients from multiple tangents based on orthogonal projections. 

## Installation
Our experiments were implemented using Python 3.9, newer versions of python might work but have not yet been tested.
It is recommended to create a new virtual environment.
Then, install the requirements from `requirements.txt`, e.g. with
```bash
pip install -r requirements.txt
```

## Usage

### Reproducing our results
Here, we give instructions on how to reproduce the experimental results presented in Section 4.
The outputs of all our experiments are automatically saved to `results/` and ordered by date and experiment.
All the following scripts come with a command-line interface. 
Use `--help` to find out more about additional parameters, e.g. to reduce the number of samples, seeds, or epochs.

#### Approximation Quality (Section 4.1)
To evaluate the cosine similarity and norm of the forward gradients compared to the true gradient $\nabla f$, call
```bash
  PYTHONPATH=. python approximation_quality.py
```
You can reduce the number of samples via the `--num_samples` to get faster results.


#### Optimization of Closed-Form Functions (Section 4.2)
To reproduce the optimization of the closed-form functions, set `<function>` to `sphere`, `rosenbrock`, or `styblinski-tang` and call
```bash
 PYTHONPATH=. python function_optimization/math_experiments.py --function <function> math_experiments
```
This runs the optimization for all gradient approaches and all dimensions $n$, automatically reading the corresponding learning rate from `lrs/math.csv`.

#### Using Custom Tangents (Section 4.3)
To reproduce the approximation quality and optimization results for tangents with specific angles to the first tangent, call
```bash
  PYTHONPATH=. python approximation_quality.py --tangents angle --angles 15 30 45 60 75 90
```
and
```bash
  PYTHONPATH=. python function_optimization/math_experiments.py --function styblinski-tang custom_tangents
```

#### Neural Network Training (Sections 4.4 and 4.5)

The `nn_training/train.py` provides the interface to train neural networks with different gradients.
It downloads the datasets automatically to `data/`.

To specify the gradient, use
- `<GRADIENT>=bp` for the true gradient $\nabla f$ obtained via backpropagation 
- `<GRADIENT>=fg` and `<K>=1` for the single-tangent forward gradient baseline $g_v$
- `<GRADIENT>=fg` and `<K>` in `2`, `4`, `16` for multi-tangent forward gradient with mean aggregation $\overline{g_V}$
- `<GRADIENT>=frog` and `<K>` in `2`, `4`, `16` for multi-tangent forward gradient with orthogonal projection $P_U$

The learning rate `<LR>` should be set according to the tables given in the appendix or `lrs/fc_nns.csv` and `lrs/sota_nns.csv`. 
The random seed is set via `<SEED>`, we used seeds `0` to `2`.
Pass `--device cuda` to use the GPU.

For the fully-connected neural networks trained in Section 4.4, use
```bash
PYTHONPATH=. python nn_training/train.py --model fc --model_hidden_size <WIDTH> --experiment_id fc_nn --output_name fc_w<WIDTH> --gradient_computation <GRADIENT> --num_directions <K> --initial_lr <LR> --seed <SEED>
```
for the hidden layer width `<WIDTH>` set to `256`, `1024`, or `4096`. 

For ResNet18 and ViT trained in Section 4.5, use
```bash
PYTHONPATH=. python nn_training/train.py --model <MODEL> --dataset <DATASET> --experiment_id sota_nn --output_name <MODEL>_<DATASET> --gradient_computation <GRADIENT> --num_directions <K> --initial_lr <LR> --seed <SEED>
```
with `<MODEL>` set to `resnet18` or `vit` and `<DATASET>` set to `mnist` or `cifar10`.


### Learning Rate Search
Code and instructions to run the learning rate search are given in `lr_search/`.
The learning rates determined by the search are given as CSV files in `lrs/`.