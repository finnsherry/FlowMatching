"""
    LieFlow
    =======

    The python package *lieflow* contains methods to perform flow matching on
    Lie groups and their homogeneous spaces over exponential curves.[1] This
    can be seen as a generalisation of (Euclidean) flow matching by Lipman et
    al.[2]
    There are three submodules:
      1. `groups`: encapsulates basic Lie group and Lie algebra properties.
      2. `models`: implementations of Lie group flow matching, and the
      generation of (optionally optimal transport coupled) training pairs.
      3. `metrics`: sample-based metrics to quantify how well generated
      samples match a target distribution.

    References:
      [1]: F.M. Sherry and B.M.N. Smets. "Flow Matching on Lie Groups." 7th
      International Conference on Geometric Science of Information (2025).
      DOI:10.1007/978-3-032-03918-7_6.
      [2]: Y. Lipman, R.T.Q. Chen, H. Ben-Hami, M. Nickel, and M. Le.
      "Flow Matching for Generative Modeling." arXiv preprint (2022).
      DOI:10.48550/arXiv.2210.02747.
"""

import lieflow.groups
import lieflow.models
import lieflow.metrics