# LieFlow
This repository contains code to perform Flow Matching (FM) on Lie Groups, as described in [[1]](#1). The results of [[1]](#1) can be reproduced with the notebooks in the `experiments` directory.

This work extends FM on Euclidean space by Lipman et al. [[2]](#2). 
The goal is somewhat similar to Riemannian FM by Chen & Lipman [[3]](#3), though they use geodesics of (pre)metrics. As a consequence, their method works on manifolds that aren't Lie groups, but require the design of a (pre)metric. Additionally, Riemannian FM requires integration of the learned vector field during training in all but the most simple cases, whereas FM on Lie Groups is simulation-free.

## Installation
The core functionality of this repository requires:
* `python>=3.12`
* `pytorch==2.6`
* `numpy`
* `tqdm`

To reproduce the experiments, one additionally needs:
* `jupyter`
* `matplotlib`

The figures were made using LaTeX. Hence, to run the notebooks one either needs to have a LaTeX installation, or to remove the `matplotlib` configuration code
```python
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern"],
        "font.size": 10.0,
    })
```
from the top of the notebooks.

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
The package consists of two submodules:
### `groups`
Here the basic group operations are defined. There are two abstract types of group implementations:
* `Group`: This is meant for groups that can be parametrised with as many parameters as the dimension of the group (e.g. SE(2)). This requires one to implement left multiplication `L`, left multiplication by inverse `L_inv`, logarithm `log`, and exponential `exp`. It is straightforward to implement direct products of such groups, see e.g. `groups.SE2byRn`.
* `MatrixGroup`: This is meant for groups with a matrix representation (e.g. SO(3)). The basic group operations are then given by their matrix equivalents. Since PyTorch has no matrix logarithm, one has to design their own `log` implementation. Additionally, one must define a basis for the Lie algebra `lie_algebra_basis`, and implement projection onto that basis `lie_algebra_components`. With this design, we ensure that the network always returns an element of the Lie algebra, and so we remain in the group when we integrate the trained vector field.

Notably, the data are not wrapped in any custom object, but are assumed to be `torch.Tensor`s. For data in a `Group`, the last dimension must have length equal to the dimension of the group, while for data in a `MatrixGroup`, the last two dimensions must equal the size of the matrix representation of the group.

### `models`
Here, the network architectures are defined. We have implementations for both FM on Lie groups and so-called shortcut models [[4]](#4) on Lie groups; we did not discuss the shortcut models in [[1]](#1) due to space constraints.

One must first define a group `g`, which is an instance of (a subclass of) either `groups.Group` or `groups.MatrixGroup`. The functions `get_model_FM` and `get_model_SCFM` will then return a new FM or shortcut model, respectively. If `g` is a `groups.Group`, then the model will be an instance of `FlowFieldGroup` or `ShortCutFieldGroup`, while if `g` is a `groups.MatrixGroup`, the model will be an instance of `FlowFieldMatrixGroup` or `ShortCutFieldMatrixGroup`. If the group operations are correctly implemented, it should not be necessary to manually adapt the network architecture.

Finally, there is an experimental model architecture `FlowFieldPowerGroup`. This is made for data that live in a direct power of some `group.Group`.
Since `FlowFieldPowerGroup` uses a transformer architecture, it is possible to apply the same network to data in various powers of the same group.

## Cite
If you use this code in your own work, please cite our paper:

<a id="1">[1]</a> Sherry, F.M., Smets, B.M.N. "Flow Matching on Lie Groups." arXiv preprint. Accepted in 7th International Conference on Geometric Science of Information (GSI) (2025).
```
@inproceedings{Sherry2025LieFlow,
  author =       {Sherry, Finn M. and Smets, Bart M.N.},
  title =        {{Diffusion-Shock Filtering on the Space of Positions and Orientations}},
  booktitle =    {arXiv preprint. Accepted in 7th International Conference on Geometric Science of Information},
  year =         {2025},
  url =          {https://doi.org/10.48550/arXiv.2504.00494},
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

One limitation of FM models is that they require integrating the learned vector field at inference time, which typically requires many network evaluations. Frans et al. showed that the architecture can be adapted to learn instead how to integrate the vector field, which means that one can perform how quality inference with only a few network evaluations:

<a id="4">[4]</a> Frans, K., Hafner, D., Levine, S., Abbeel, P. "One Step Diffusion via Shortcut Models." 13th International Conference on Learning Representations (ICLR) (2025). https://openreview.net/forum?id=OlzB6LnXcS
```
@article{Frans2025OneModels,
  author = {Frans, Kevin and Hafner, Danijar and Levine, Sergey and Abbeel, Pieter},
  title = {{One Step Diffusion via Shortcut Models}},
  journal = {ICLR},
  year = {2025},
  url = {https://openreview.net/forum?id=OlzB6LnXcS}
}
```