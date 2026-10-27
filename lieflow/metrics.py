"""
metrics
=======

Sample-based metrics to quantify how well a generated distribution matches a
target distribution on a Lie group or homogeneous space. All metrics are based
on the logarithmic distance `d(g, h) = ||log(g^{-1} h)||_w`, i.e. the weighted
norm of the components of the generator of the exponential curve connecting `g`
and `h`, which is also what the flow matching loss is based on. On homogeneous
spaces, the generator is given by `get_generator`.

Generated samples can be obtained with the `sample` method of the models.
Provides the following functions:
  1. `distance_matrix`: pairwise logarithmic distances between two sets of
  samples.
  2. `wasserstein`: Wasserstein-1 distance between two empirical distributions
  with the same number of samples.
  3. `precision_coverage`: mean nearest neighbour distances from generated to
  target samples (precision) and from target to generated samples (coverage).
  4. `mmd`: squared maximum mean discrepancy with Gaussian kernels.
  5. `evaluate`: compute all of the above metrics at once.

Since finite samples of the same distribution do not have zero distance, one
should compare against the metrics between two independent sets of samples of
the target distribution.
"""

import torch

from lieflow.groups import Group, HomogeneousSpace, MatrixGroup


def _generator_components(G, g, h):
    if isinstance(G, Group):
        return G.log(G.L_inv(g, h))
    elif isinstance(G, MatrixGroup):
        return G.lie_algebra_components(G.log(G.L_inv(g, h)))
    elif isinstance(G, HomogeneousSpace):
        return G.G.lie_algebra_components(G.get_generator(g, h))
    else:
        raise TypeError(
            f"{G} is neither a `Group`, a `MatrixGroup`, nor a  `HomogeneousSpace`!"
        )


@torch.no_grad()
def distance_matrix(
    G: Group | MatrixGroup | HomogeneousSpace,
    X: torch.Tensor,
    Y: torch.Tensor,
    w: torch.Tensor | None = None,
    chunk_size: int = 128,
) -> torch.Tensor:
    """
    Compute the matrix of logarithmic distances `D[i, j] = d(X[i], Y[j])`.

    Args:
        `G`: group of type `Group` or `MatrixGroup`, or a `HomogeneousSpace`.
        `X`: samples of shape [N, ...].
        `Y`: samples of shape [M, ...].
      Optional:
        `w`: weights of the components of the generator, as in
        `LogarithmicDistance`. Defaults to all ones.
        `chunk_size`: number of rows of `D` computed at once, to limit memory
        usage. Defaults to 128.
    """
    D = torch.empty(len(X), len(Y), device=X.device)
    for i in range(0, len(X), chunk_size):
        a = _generator_components(G, X[i : i + chunk_size, None], Y[None])
        if w is not None:
            a = w * a
        D[i : i + chunk_size] = a.norm(dim=-1)
    return D


def wasserstein(D: torch.Tensor) -> float:
    """
    Compute the Wasserstein-1 distance between two empirical distributions with
    the same number of samples, given the matrix `D` of pairwise distances. For
    uniform weights, the optimal transport plan is a permutation, which we find
    by solving the assignment problem.
    """
    from scipy.optimize import linear_sum_assignment

    if D.shape[0] != D.shape[1]:
        raise ValueError(f"Need the same number of samples, but D has shape {tuple(D.shape)}!")
    D = D.cpu().numpy()
    rows, cols = linear_sum_assignment(D)
    return float(D[rows, cols].mean())


def precision_coverage(D: torch.Tensor) -> tuple[float, float]:
    """
    Compute the mean distance from every generated sample to the nearest target
    sample (precision), and from every target sample to the nearest generated
    sample (coverage), given the matrix `D` of pairwise distances between
    generated samples (rows) and target samples (columns). A large precision
    means that generated samples lie off the support of the target; a large
    coverage means that parts of the target are not generated.
    """
    return float(D.min(dim=1).values.mean()), float(D.min(dim=0).values.mean())


def mmd(
    D_XX: torch.Tensor,
    D_YY: torch.Tensor,
    D_XY: torch.Tensor,
    bandwidths: torch.Tensor | list[float],
) -> float:
    """
    Compute the unbiased estimate of the squared maximum mean discrepancy with
    the sum of Gaussian kernels `exp(-d^2 / (2 σ^2))` over `σ` in `bandwidths`,
    given the matrices of pairwise distances within and between the samples
    `X` and `Y`. The estimate may be slightly negative.
    """
    N, M = len(D_XX), len(D_YY)

    def k(D):
        return sum(torch.exp(-(D**2) / (2 * σ**2)) for σ in bandwidths)

    K_XX, K_YY, K_XY = k(D_XX), k(D_YY), k(D_XY)
    return float(
        (K_XX.sum() - K_XX.diagonal().sum()) / (N * (N - 1))
        + (K_YY.sum() - K_YY.diagonal().sum()) / (M * (M - 1))
        - 2 * K_XY.mean()
    )


@torch.no_grad()
def evaluate(
    G: Group | MatrixGroup | HomogeneousSpace,
    X: torch.Tensor,
    Y: torch.Tensor,
    w: torch.Tensor | None = None,
    bandwidths: torch.Tensor | list[float] | None = None,
) -> dict[str, float]:
    """
    Compute the Wasserstein-1 distance, precision, coverage, and squared MMD
    between generated samples `X` and target samples `Y`.

    Args:
        `G`: group of type `Group` or `MatrixGroup`, or a `HomogeneousSpace`.
        `X`: generated samples of shape [N, ...].
        `Y`: target samples of shape [N, ...].
      Optional:
        `w`: weights of the components of the generator, as in
        `LogarithmicDistance`. Defaults to all ones.
        `bandwidths`: bandwidths of the Gaussian kernels in the MMD. Should be
        the same for all models that are compared. Defaults to `[m/2, m, 2m]`,
        with `m` the median distance between target samples, so it only depends
        on `Y`.
    """
    D_XY = distance_matrix(G, X, Y, w)
    D_XX = distance_matrix(G, X, X, w)
    D_YY = distance_matrix(G, Y, Y, w)
    if bandwidths is None:
        m = D_YY[~torch.eye(len(Y), dtype=torch.bool, device=D_YY.device)].median()
        bandwidths = [m / 2, m, 2 * m]
    precision, coverage = precision_coverage(D_XY)
    return {
        "W_1": wasserstein(D_XY),
        "precision": precision,
        "coverage": coverage,
        "MMD^2": mmd(D_XX, D_YY, D_XY, bandwidths),
    }
