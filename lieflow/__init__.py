"""
    LieFlow
    =======

    The python package *lieflow* contains methods to perform flow matching on
    Lie groups over exponential curves.[1] This can be seen as a generalisation
    of (Euclidean) flow matching by Lipman et al.[2] 
    We also generalise shortcut modeling, which can be seen as learning the
    integrator of the flow field from flow matching, and in this way can
    speed up inferencing.[3]
    There are two submodules:
      1. `groups`: encapsulates basic Lie group and Lie algebra properties.
      2. `models`: implementations of Lie group flow matching and shortcut
      models.

    References:
      [1]:
      [2]: Y. Lipman, R.T.Q. Chen, H. Ben-Hami, M. Nickel, and M. Le.
      "Flow Matching for Generative Modeling." arXiv preprint (2022).
      DOI:10.48550/arXiv.2210.02747.
      [3]: K. Frans, D. Hafner, S. Levine, and P. Abbeel.
      "One Step Diffusion via Shortcut Models." arXiv preprint (2024).
      DOI:10.48550/arXiv.2410.12557
"""

import lieflow.groups
import lieflow.models
