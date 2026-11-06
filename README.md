# LieFlow
This repository contains code to perform Flow Matching (FM) on Lie Groups, as described in [[1]](#1). The results of [[1]](#1) can be reproduced with the notebooks in the `experiments/` directory.

This work extends FM on Euclidean space by Lipman et al. [[2]](#2). 
The goal is somewhat similar to Riemannian FM by Chen & Lipman [[3]](#3), though they use geodesics of (pre)metrics. As a consequence, their method works on manifolds that aren't Lie groups, but require the design of a (pre)metric. Additionally, Riemannian FM requires integration of the learned vector field during training in all but the most simple cases, whereas FM on Lie Groups is simulation-free.

## Installation
The core functionality of this repository requires:
* `python>=3.12`
* `pytorch==2.6`
* `numpy`
* `scipy`
* `tqdm`

To reproduce the experiments, one additionally needs:
* `jupyter`
* `matplotlib`

The figures were made using LaTeX. Hence, to run the notebooks one either needs to have a LaTeX installation, or to remove the `matplotlib` configuration code
```json
    "text.usetex": true,
    "font.family": "serif",
    "font.serif": ["Computer Modern"],
    "font.size": 10.0,
    "text.latex.preamble": "\\usepackage{lmodern} \\usepackage{amssymb} \\usepackage{amsmath}",
```
from `experiments/plot.json`.

Alternatively, one can create the required conda environment from `env.yml`:
```
conda env create -f env.yml
```
This creates a conda environment called `flow-matching`.

Subsequently, one must install the code of this project as a package, by running:
```
pip install -e .
```

## Functionality
The package consists of three submodules:
### `groups`
Here the basic group operations are defined. The models and metrics only use the exponential curves of a space, through the components of the generator connecting two points (`generator_components`), and following an exponential curve (`flow`). These are provided by the following abstract types of space implementations:
* `Group`: This is meant for groups that can be parametrised with as many parameters as the dimension of the group (e.g. SE(2)). This requires one to implement left multiplication `L`, left multiplication by inverse `L_inv`, logarithm `log`, and exponential `exp`.
* `MatrixGroup`: This is meant for groups with a matrix representation (e.g. SO(3)). The basic group operations are then given by their matrix equivalents. Since PyTorch has no matrix logarithm, one has to design their own `log` implementation. Additionally, one must define a basis for the Lie algebra `lie_algebra_basis`, and implement projection onto that basis `lie_algebra_components`. With this design, we ensure that the network always returns an element of the Lie algebra, and so we remain in the group when we integrate the trained vector field.
* `HomogeneousSpace`: This is meant for homogeneous spaces of a `MatrixGroup` (e.g. position-orientation space M3 of SE(3)). This requires one to implement `generator_components`, which returns the components of a generator in the Lie algebra of the group of an exponential curve connecting two points (`get_generator` returns the generator itself); by default, the group acts on the points by matrix multiplication `act`.
* `ProductSpace`: This is the direct product of any of the above spaces, which may be of different types (e.g. SE(2) x R^2, or M3 x R^n), for which all operations act factorwise; e.g. the flow is the product of the flows of the factors. The weights of the norm are the concatenation of the weights of the factors.
* `PowerSpace`: This is the power G^m of any of the above spaces G (e.g. point clouds in R^2, or sets of rotations in SO(3)), for which all operations act factorwise. By default, the factors are unordered, so that the distance between two elements is minimised over all permutations of the factors.

Notably, the data are not wrapped in any custom object, but are assumed to be `torch.Tensor`s. For data in a `Group`, the last dimension must have length equal to the dimension of the group, for data in a `MatrixGroup` or `HomogeneousSpace`, the last two dimensions must equal the size of the matrix representation of the points, for data in a `ProductSpace`, the points of the factors are flattened and concatenated along the last dimension (see `ProductSpace.pack` and `ProductSpace.unpack`), while for data in a `PowerSpace` of G^m, the factors are stacked along an extra dimension of length m before those of a point of G.

### `models`
Here, the flow matching model is defined. One must first define a space `space`, which is an instance of (a subclass of) `groups.Group`, `groups.MatrixGroup`, `groups.HomogeneousSpace`, or `groups.PowerSpace`. Then `FlowField(space)` returns a new FM model on this space. Since the model only uses the exponential curves of the space, it should not be necessary to manually adapt the model if the operations of the space are correctly implemented.

The network predicts the components of the generator of the flow field at a point and time, and its architecture depends on the space:
* By default, it is an MLP acting on the flattened point and the time, with `L` hidden layers of width `H`, which are the same pre-activation residual block (so the hidden layers share their weights).
* If `space` is a `groups.PowerSpace`, it is a transformer with one token per (flattened) factor, consisting of a linear embedding of dimension `embed_dim`, `L` encoder blocks with `num_heads` attention heads, which are the same encoder block, and a linear output layer. Since it has no positional encoding, it is equivariant under permutations of the factors, and the same network can be applied to data in various powers of the same space.

Training pairs can be drawn with `DataGenerator`, optionally coupled by mini-batch optimal transport with `OTCoupling`, and the model can be trained with its `train_epoch` method using the `LogarithmicDistance` loss. See `experiments/training.py` for an example.

### `metrics`
Here, sample-based metrics are defined to quantify how well a trained model works. Given samples generated with the `sample` method of a model, which integrates the flow from samples at $t = 0$ to $t = 1$, `compare` compares the generated samples with samples from the target distribution, in terms of the logarithmic distance $d(g, h) = \lVert \log(g^{-1} h) \rVert$. It computes the Wasserstein-1 distance, precision (mean distance from a generated sample to the nearest target sample), and coverage (mean distance from a target sample to the nearest generated sample). The `evaluate` method of a model does both at once. Since finitely many samples of the same distribution are not at zero distance, these metrics should be compared with the metrics between two independent sets of target samples. See `experiments/flow_matching_SO3.ipynb` for an example.

## Cite
If you use this code in your own work, please cite our paper:

<a id="1">[1]</a> Sherry, F.M., Smets, B.M.N. "Flow Matching on Lie Groups."7th International Conference on Geometric Science of Information (GSI) (2025).
```
@inproceedings{Sherry2025LieFlow,
  author =       {Sherry, Finn M. and Smets, Bart M.N.},
  title =        {{Flow Matching on Lie Groups}},
  booktitle =    {7th International Conference on Geometric Science of Information},
  year =         {2025},
  doi =          {10.1007/978-3-032-03918-7_6},
}
```

We extend FM on Euclidean space by Lipman et al:

<a id="2">[2]</a> Lipman, Y., Havasi, M., Holderrieth, P., Shaul, N., Le, M., Karrer, B., Chen, R.T.Q., Lopez-Paz, D., Ben-Hamu, H., Gat, I. "Flow Matching Guide and Code." arXiv preprint (2024). https://arxiv.org/abs/2412.06264
```
@article{Lipman2024FlowCode,
  author = {Lipman, Yaron and Havasi, Marton and Holderrieth, Peter and Shaul, Neta and Le, Matt and Karrer, Brian and Chen, Ricky T. Q. and Lopez-Paz, David and Ben-Hamu, Heli and Gat, Itai},
  title = {{Flow Matching Guide and Code}},
  journal = {arXiv preprint},
  year = 2024,
  url = {https://arxiv.org/abs/2412.06264}
}
```

The goal of our work is similar to FM on Riemannian manifolds by Chen & Lipman:

<a id="3">[3]</a> Chen, R.T.Q., Lipman, Y. "Flow Matching on General Geometries." 12th International Conference on Learning Representations (ICLR) (2024). https://openreview.net/forum?id=g7ohDlTITL
```
@inproceedings{Chen2024FlowGeometries,
  author = {Chen, Ricky T. Q. and Lipman, Yaron},
  title = {{Flow Matching on General Geometries}},
  booktitle = {12th International Conference on Learning Representations},
  year = {2024},
  url = {https://openreview.net/forum?id=g7ohDlTITL}
}
```