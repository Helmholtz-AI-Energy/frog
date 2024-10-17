# Learning Rate Search

## Installation

**Step 1:** Make sure your system has MPI installed  
The learning rate search uses `mpi4py` and thus relies on an OpenMPI installation.

**Step 2:** install the additional requirements from `lr_search/requirements.txt`:
```
pip install -r lr_search/requirements.txt
```

## Running the LR-Search
The script must be called via MPI as
```
PYTHONPATH=. mpirun -n <N> python lr_search/propulate_search.py <search_config> <mode> <mode_config>
```
where `<N>` is the number of workers (MPI ranks) to use. 
You can use the `--help` argument to get more information on the available CLI arguments.

### Reproduction

To reproduce our LR search, use the following commands with the gradient combinations
- `GRADIENT=bp` for the true gradient $\nabla f$ obtained via backpropagation 
- `GRADIENT=fg-mean` and `K=1` for the single-tangent forward gradient baseline $g_v$
- `GRADIENT=fg-mean` and `K` in `2`, `4`, `16` for multi-tangent forward gradient with mean aggregation $\overline{g_V}$
- `GRADIENT=frog` and `K` in `2`, `4`, `16` for multi-tangent forward gradient with orthogonal projection $P_U$

For the closed-form function `FUNCTION` in `sphere`, `rosenbrock`, `styblinski-tang` with input dimension `N` in `1`, `2`, `4`, `8`, `16`, `32`, `64`, `128`, `256`, `512`, `1024` use
```
PYTHONPATH=. mpirun -n 4 python lr_search/propulate_search.py --num_evaluations 100 --gradient ${GRADIENT} -k ${K} math --function ${FUNCTION} -n ${N}
```

For the fully-connected networks with `WIDTH` in `256`, `1024`, and `4096` use
```
PYTHONPATH=. mpirun -n 4 python lr_search/propulate_search.py --num_evaluations 100 --gradient ${GRADIENT} -k ${K:-1} nn --device cuda --model fc --model_width ${WIDTH}
```
For the ResNet18 and ViT use
```
PYTHONPATH=. mpirun -n 4 python lr_search/propulate_search.py --num_evaluations 100 --gradient ${GRADIENT} -k ${K:-1} nn --device cuda --model ${MODEL} --dataset ${DATASET} --epochs 50
```
with `MODEL` set to `resnet18` and `vit` correspondingly.

Note that due to the nature of the parallel search, results may vary slightly even for the same random seed and rank count. 