"""
groups
======

Classes that encapsulate basic Lie group and Lie algebra properties.
There are three abstract parent classes:
  1. `Group`: Use this class when the group and algebra can be efficiently
  parametrised with the same number of parameters. Requires implementing
  hand crafted group multiplication, multiplication by inverse, exponential,
  and logarithm.
  2. `MatrixGroup`: Use this class when the group can be efficiently
  represented with matrices. Group multiplication, multiplication by
  inverse, and exponential make use of corresponding PyTorch methods. Since
  PyTorch does not implement a matrix logarithm, this must be provided.
  3. `HomogeneousSpace`: This class is for homogeneous spaces which are acted on
  by a `MatrixGroup`. Requires implementing the action of the group on the
  homogeneous space and the computation of generators in the Lie algebra
  connecting pairs of points in the homogeneous space.
Also provides example implementations of
  1. `Rn(n)` <: `Group`: n-dimensional translation group R^n.
  2. `SE2()` <: `Group`: special Euclidean group of roto-translations on
  R^2.
  3. `SE2byRn(se2, rn)` <: `Group`: direct product of SE(2) and R^n.
  4. `TSn(n)` <: `Group`: translation-scaling group on R^n.
  5. `RmbyTSn(rm, tsn)` <: `Group`: direct product of R^m and TS(n).
  6. `Aff2()` <: `Group`: group of affine transformations on R^2.
  7. `SO3()` <: `MatrixGroup`: special orthogonal group of rotations on R^3.
  8. `SE3()` <: `MatrixGroup`: special Euclidean group of roto-translations on
  R^3.
  9. `SOplus12()` <: `MatrixGroup`: proper orthochronous special orthogonal group of Lorentz transformations on R^3.
  10. `SEplus12()` <: `MatrixGroup`: proper orthochronous special Euclidean group of Poincare transformations on R^3.
  11. `M3()` <: `HomogeneousSpace`: position-orientation space R^3 x S^2, which
  is an SE(3)-homogeneous space.
  12. `P3()` <: `HomogeneousSpace`: event-velocity space R^3 x H^2, which
  is an SE^+(1, 2)-homogeneous space.
"""

import math
from abc import ABC, abstractmethod
from typing import Literal

import torch
from scipy.optimize import linear_sum_assignment


class Group(ABC):
    """
    Class encapsulating basic Lie group and Lie algebra properties for groups
    that can be efficiently parametrised with as many parameters as the group
    dimension.

    Requires implementing hand crafted group multiplication, multiplication by
    inverse, exponential, and logarithm.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    @abstractmethod
    def L(self, g_1: torch.Tensor, g_2: torch.Tensor) -> torch.Tensor:
        """
        Left multiplication of `g_2` by `g_1`, i.e. `g_1 + g_2`.
        """
        raise NotImplementedError

    @abstractmethod
    def L_inv(self, g_1: torch.Tensor, g_2: torch.Tensor) -> torch.Tensor:
        """
        Left multiplication of `g_2` by `g_1^-1`, i.e. `g_2 - g_1`.
        """
        raise NotImplementedError

    @abstractmethod
    def log(self, g: torch.Tensor) -> torch.Tensor:
        """
        Lie group logarithm of `g`, i.e. `g`.
        """
        raise NotImplementedError

    @abstractmethod
    def exp(self, A: torch.Tensor) -> torch.Tensor:
        """
        Lie group exponential of `A`, i.e. `A`.
        """
        raise NotImplementedError

    @torch.no_grad()
    def distance_matrix(
        self,
        g_1: torch.Tensor,
        g_2: torch.Tensor,
        w: torch.Tensor | None = None,
        chunk_size: int = 128,
        permutation_invariant: bool = False,
    ) -> torch.Tensor:
        """
        Compute the pairwise logarithmic distances between `g_1` and `g_2`.

        Args:
            `g_1`: samples of shape [N, ..., dim].
            `g_2`: samples of shape [M, ..., dim].
          Optional:
            `w`: metric parameters. Defaults to all ones.
            `chunk_size`: number of rows of `D` computed at once, to limit memory
            usage. Defaults to 128.
            `permutation_invariant`: only for samples in a power of the group,
            i.e. of shape [N, m, dim]. If True, the distance is minimised over
            all permutations of the m factors, so that samples are treated as
            unordered sets. Defaults to False.
        """
        if permutation_invariant:
            return self._permutation_invariant_distance_matrix(g_1, g_2, w, chunk_size)
        D = torch.empty(len(g_1), len(g_2), device=g_2.device, dtype=g_2.dtype)
        for i in range(0, len(g_1), chunk_size):
            A = self.log(self.L_inv(g_1[i:i + chunk_size, None], g_2[None]))
            if w is not None:
                A = w * A
            D[i:i + chunk_size] = A.flatten(start_dim=2).norm(dim=-1)
        return D

    @torch.no_grad()
    def cost_matrices(
        self, g_1: torch.Tensor, g_2: torch.Tensor, w: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        Compute the squared logarithmic distances between all pairs within
        blocks, `C[k, i, j] = ρ(g_1[k, i], g_2[k, j])^2`, as the cost of a
        mini-batch optimal transport coupling.

        Args:
            `g_1`: samples of shape [B, b, dim].
            `g_2`: samples of shape [B, b, dim].
          Optional:
            `w`: metric parameters. Defaults to all ones.
        """
        return torch.stack(
            [self.distance_matrix(a, b, w, chunk_size=len(a)) ** 2 for a, b in zip(g_1, g_2)]
        )

    def _permutation_invariant_distance_matrix(self, g_1, g_2, w, chunk_size):
        D_squared = []
        for i in range(0, len(g_1), chunk_size):
            A = self.log(self.L_inv(g_1[i:i + chunk_size, None, :, None], g_2[None, :, None]))
            if w is not None:
                A = w * A
            C = (A**2).sum(-1).cpu().numpy()
            for c in C.reshape(-1, *C.shape[2:]):
                rows, cols = linear_sum_assignment(c)
                D_squared.append(c[rows, cols].sum())
        return torch.tensor(D_squared, device=g_2.device, dtype=g_2.dtype).view(len(g_1), len(g_2)).sqrt()


class Rn(Group):
    """
    Translation group.

    Args:
        `n`: dimension of the translation group.
    """

    def __init__(self, n):
        super().__init__(dim=n)

    def L(self, g_1, g_2):
        """
        Left multiplication of `g_2` by `g_1`, i.e. `g_1 + g_2`.
        """
        return g_1 + g_2

    def L_inv(self, g_1, g_2):
        """
        Left multiplication of `g_2` by `g_1^-1`, i.e. `g_2 - g_1`.
        """
        return g_2 - g_1

    def log(self, g):
        """
        Lie group logarithm of `g`, i.e. `g`.
        """
        return g.clone()

    def exp(self, A):
        """
        Lie group exponential of `A`, i.e. `A`.
        """
        return A.clone()

    @torch.no_grad()
    def cost_matrices(self, g_1, g_2, w=None):
        """
        Compute the squared (weighted) Euclidean distances between all pairs
        within blocks, `C[k, i, j] = ||w * (g_2[k, j] - g_1[k, i])||^2`. Samples
        in a power of the group, of shape [B, b, m, dim], use the product
        distance.
        """
        if w is not None:
            g_1, g_2 = w * g_1, w * g_2
        return torch.cdist(g_1.flatten(2), g_2.flatten(2), compute_mode="donot_use_mm_for_euclid_dist") ** 2

    def __repr__(self):
        return f"R^{self.dim}"


class SE2(Group):
    """
    Special Euclidean group of roto-translations on R^2.
    """

    def __init__(self):
        super().__init__(dim=3)

    def L(self, g_1, g_2):
        """
        Left multiplication of `g_2` by `g_1`.
        """
        g = torch.zeros(torch.broadcast_shapes(g_1.shape, g_2.shape), device=g_2.device, dtype=g_2.dtype)
        x_1 = g_1[..., 0]
        y_1 = g_1[..., 1]
        θ_1 = g_1[..., 2]

        cos = torch.cos(θ_1)
        sin = torch.sin(θ_1)

        x_2 = g_2[..., 0]
        y_2 = g_2[..., 1]
        θ_2 = g_2[..., 2]

        g[..., 0] = x_1 + cos * x_2 - sin * y_2
        g[..., 1] = y_1 + sin * x_2 + cos * y_2
        g[..., 2] = _mod_offset(θ_1 + θ_2, 2 * torch.pi, -torch.pi)
        return g

    def L_inv(self, g_1, g_2):
        """
        Left multiplication of `g_2` by `g_1^-1`.
        """
        g = torch.zeros(torch.broadcast_shapes(g_1.shape, g_2.shape), device=g_2.device, dtype=g_2.dtype)
        x_1 = g_1[..., 0]
        y_1 = g_1[..., 1]
        θ_1 = g_1[..., 2]

        cos = torch.cos(θ_1)
        sin = torch.sin(θ_1)

        x_2 = g_2[..., 0]
        y_2 = g_2[..., 1]
        θ_2 = g_2[..., 2]

        g[..., 0] = cos * (x_2 - x_1) + sin * (y_2 - y_1)
        g[..., 1] = -sin * (x_2 - x_1) + cos * (y_2 - y_1)
        g[..., 2] = _mod_offset(θ_2 - θ_1, 2 * torch.pi, -torch.pi)
        return g

    def log(self, g):
        """
        Lie group logarithm of `g`, i.e. `A` in Lie algebra such that
        `exp(A) = g`.
        """
        A = torch.zeros_like(g)
        x = g[..., 0]
        y = g[..., 1]
        θ = _mod_offset(g[..., 2], 2 * torch.pi, -torch.pi)

        cos = torch.cos(θ / 2.0)
        sin = torch.sin(θ / 2.0)
        sinc = _sinc(θ / 2.0)

        A[..., 0] = (x * cos + y * sin) / sinc
        A[..., 1] = (-x * sin + y * cos) / sinc
        A[..., 2] = θ
        return A

    def exp(self, A):
        """
        Lie group exponential of `A`, i.e. `g` in Lie group such that
        `exp(A) = g`.
        """
        g = torch.zeros_like(A)
        c1 = A[..., 0]
        c2 = A[..., 1]
        c3 = A[..., 2]

        cos = torch.cos(c3 / 2.0)
        sin = torch.sin(c3 / 2.0)
        sinc = _sinc(c3 / 2.0)

        g[..., 0] = (c1 * cos - c2 * sin) * sinc
        g[..., 1] = (c1 * sin + c2 * cos) * sinc
        g[..., 2] = _mod_offset(c3, 2 * torch.pi, -torch.pi)
        return g

    def L_star(self, g, A):
        """
        Push-forward of `A` under left multiplication by `g`.
        """
        B = torch.zeros_like(A)
        θ = g[..., 2]

        cos = torch.cos(θ)
        sin = torch.sin(θ)

        B[..., 0] = cos * A[..., 0] - sin * A[..., 1]
        B[..., 1] = sin * A[..., 0] + cos * A[..., 1]
        B[..., 2] = A[..., 2]
        return B

    def __repr__(self):
        return "SE(2)"


class SE2byRn(Group):
    """
    Direct product of special Euclidean group of roto-translations on R^2 and
    n-dimensional translation group.

    Args:
        `se2`: instance of the special Euclidean group.
        `rn`: instance of the n-dimensional translation group.
    """

    def __init__(self, se2: SE2, rn: Rn):
        super().__init__(dim=se2.dim + rn.dim)
        self.se2 = se2
        self.rn = rn

    def L(self, g_1, g_2):
        """
        Left multiplication of `g_2 = (x_2, p_2)` by `g_1 = (x_1, p_1)`, i.e.
        `(x_1 + x_2, p_1 p_2)`.
        """
        g = torch.zeros(torch.broadcast_shapes(g_1.shape, g_2.shape), device=g_2.device, dtype=g_2.dtype)
        g[..., :3] = self.se2.L(g_1[..., :3], g_2[..., :3])
        g[..., 3:] = self.rn.L(g_1[..., 3:], g_2[..., 3:])
        return g

    def L_inv(self, g_1, g_2):
        """
        Left multiplication of `g_2 = (x_2, p_2)` by `g_1^-1 = (-x_1, p_1^-1)`,
        i.e. `(x_2 - x_1, p_1^-1 p_2)`.
        """
        g = torch.zeros(torch.broadcast_shapes(g_1.shape, g_2.shape), device=g_2.device, dtype=g_2.dtype)
        g[..., :3] = self.se2.L_inv(g_1[..., :3], g_2[..., :3])
        g[..., 3:] = self.rn.L_inv(g_1[..., 3:], g_2[..., 3:])
        return g

    def log(self, g):
        """
        Lie group logarithm of `g = (x, p)`, i.e. `(x, P)` with `P` in Lie
        algebra such that `exp(P) = p`.
        """
        A = torch.zeros_like(g)
        A[..., :3] = self.se2.log(g[..., :3])
        A[..., 3:] = self.rn.log(g[..., 3:])
        return A

    def exp(self, A):
        """
        Lie group exponential of `A = (x, P)`, i.e. `(x, p)` with `p` in Lie
        group such that `exp(P) = p`.
        """
        g = torch.zeros_like(A)
        g[..., :3] = self.se2.exp(A[..., :3])
        g[..., 3:] = self.rn.exp(A[..., 3:])
        return g

    def __repr__(self):
        return f"SE(2) x R^{self.rn.dim}"


class TSn(Group):
    """
    Translation-Scaling group.

    Args:
        `n`: dimension of the translational part of the group.
    """

    def __init__(self, n):
        super().__init__(dim=n + 1)

    def L(self, g_1, g_2):
        """
        Left multiplication of `g_2` by `g_1`.
        """
        g = torch.zeros(torch.broadcast_shapes(g_1.shape, g_2.shape), device=g_2.device, dtype=g_2.dtype)
        x_1 = g_1[..., :-1]
        s_1 = g_1[..., -1]
        x_2 = g_2[..., :-1]
        s_2 = g_2[..., -1]

        g[..., :-1] = x_1 + torch.exp(_sigmoid(s_1))[..., None] * x_2
        g[..., -1] = _sigmoid(s_1 + s_2)
        return g

    def L_inv(self, g_1, g_2):
        """
        Left multiplication of `g_2` by `g_1^-1`.
        """
        g = torch.zeros(torch.broadcast_shapes(g_1.shape, g_2.shape), device=g_2.device, dtype=g_2.dtype)
        x_1 = g_1[..., :-1]
        s_1 = g_1[..., -1]
        x_2 = g_2[..., :-1]
        s_2 = g_2[..., -1]

        g[..., :-1] = (x_2 - x_1) * torch.exp(_sigmoid(-s_1))[..., None]
        g[..., -1] = _sigmoid(s_2 - s_1)
        return g

    def log(self, g):
        """
        Lie group logarithm of `g`.
        """
        A = torch.zeros_like(g)
        x = g[..., :-1]
        s = _sigmoid(g[..., -1])

        A[..., :-1] = _expc(s)[..., None] * x
        A[..., -1] = s.clone()
        return A

    def exp(self, A):
        """
        Lie group exponential of `A`.
        """
        g = torch.zeros_like(A)
        cx = A[..., :-1]
        cs = _sigmoid(A[..., -1])

        g[..., :-1] = cx / _expc(cs)[..., None]
        g[..., -1] = cs.clone()
        return g

    def __repr__(self):
        return f"TS({self.dim})"


class RmbyTSn(Group):
    """
    Direct product of the m-dimensional translation group and the translation-
    scaling group on R^n.

    Args:
        `rm`: instance of the m-dimensional translation group.
        `tsn`: instance of the translation-scaling group on R^n.
    """

    def __init__(self, rm: Rn, tsn: TSn):
        super().__init__(dim=rm.dim + tsn.dim)
        self.rm = rm
        self.tsn = tsn

    def L(self, g_1, g_2):
        """
        Left multiplication of `g_2 = (x_2, p_2)` by `g_1 = (x_1, p_1)`, i.e.
        `(x_1 + x_2, p_1 p_2)`.
        """
        g = torch.zeros(torch.broadcast_shapes(g_1.shape, g_2.shape), device=g_2.device, dtype=g_2.dtype)
        g[..., : self.rm.dim] = self.rm.L(
            g_1[..., : self.rm.dim], g_2[..., : self.rm.dim]
        )
        g[..., self.rm.dim :] = self.tsn.L(
            g_1[..., self.rm.dim :], g_2[..., self.rm.dim :]
        )
        return g

    def L_inv(self, g_1, g_2):
        """
        Left multiplication of `g_2 = (x_2, p_2)` by `g_1^-1 = (-x_1, p_1^-1)`,
        i.e. `(x_2 - x_1, p_1^-1 p_2)`.
        """
        g = torch.zeros(torch.broadcast_shapes(g_1.shape, g_2.shape), device=g_2.device, dtype=g_2.dtype)
        g[..., : self.rm.dim] = self.rm.L_inv(
            g_1[..., : self.rm.dim], g_2[..., : self.rm.dim]
        )
        g[..., self.rm.dim :] = self.tsn.L_inv(
            g_1[..., self.rm.dim :], g_2[..., self.rm.dim :]
        )
        return g

    def log(self, g):
        """
        Lie group logarithm of `g = (x, p)`, i.e. `(x, P)` with `P` in Lie
        algebra such that `exp(P) = p`.
        """
        A = torch.zeros_like(g)
        A[..., : self.rm.dim] = self.rm.log(g[..., : self.rm.dim])
        A[..., self.rm.dim :] = self.tsn.log(g[..., self.rm.dim :])
        return A

    def exp(self, A):
        """
        Lie group exponential of `A = (x, P)`, i.e. `(x, p)` with `p` in Lie
        group such that `exp(P) = p`.
        """
        g = torch.zeros_like(A)
        g[..., : self.rm.dim] = self.rm.exp(A[..., : self.rm.dim])
        g[..., self.rm.dim :] = self.tsn.exp(A[..., self.rm.dim :])
        return g

    def __repr__(self):
        return f"R^{self.rm.dim} x TS({self.tsn.dim})"


class Aff2(Group):
    """
    Group of affine transformations on R^2.
    """

    def __init__(self):
        super().__init__(dim=6)

    def L(self, g_1, g_2):
        """
        Left multiplication of `g_2` by `g_1`.
        """
        g = torch.zeros(torch.broadcast_shapes(g_1.shape, g_2.shape), device=g_2.device, dtype=g_2.dtype)

        t_1 = g_1[..., :2]
        A_1 = g_1[..., 2:].view(*g_1.shape[:-1], 2, 2)

        t_2 = g_2[..., :2, None]
        A_2 = g_2[..., 2:].view(*g_2.shape[:-1], 2, 2)

        g[..., :2] = t_1 + (A_1 @ t_2).squeeze(-1)
        g[..., 2:] = (A_1 @ A_2).view(*g.shape[:-1], 4)
        return g

    def L_inv(self, g_1, g_2):
        """
        Left multiplication of `g_2` by `g_1^-1`.
        """
        g = torch.zeros(torch.broadcast_shapes(g_1.shape, g_2.shape), device=g_2.device, dtype=g_2.dtype)

        t_1 = g_1[..., :2, None]
        A_1 = g_1[..., 2:].view(*g_1.shape[:-1], 2, 2)
        A_1_inv = torch.linalg.inv(A_1)

        t_2 = g_2[..., :2, None]
        A_2 = g_2[..., 2:].view(*g_2.shape[:-1], 2, 2)

        g[..., :2] = (A_1_inv @ (t_2 - t_1)).squeeze(-1)
        g[..., 2:] = (A_1_inv @ A_2).flatten(start_dim=-2, end_dim=-1)
        return g

    def log(self, g):
        """
        Lie group logarithm of `g`, i.e. `A` in Lie algebra such that
        `exp(A) = g`.
        """
        A = torch.zeros_like(g)
        t = g[..., :2]
        T = g[..., 2:].view(*g.shape[:-1], 2, 2)

        TTT = T.transpose(-1, -2) @ T

        evals, evecs = torch.linalg.eigh(TTT)

        S_inv = evecs @ torch.diag_embed(1.0 / evals.sqrt()) @ evecs.transpose(-1, -2)

        R = T @ S_inv
        r = torch.atan2(R[..., 1, 0], R[..., 0, 0])

        log_S = evecs @ torch.diag_embed(evals.log() / 2.0) @ evecs.transpose(-1, -2)

        A[..., :2] = t
        A[..., 2] = r
        A[..., 3] = log_S[..., 0, 0]
        A[..., 4] = log_S[..., 1, 1]
        A[..., 5] = log_S[..., 0, 1]
        return A

    def exp(self, A):
        """
        Exponential of `A`, i.e. `g` in Lie group such that `exp(A) = g`.

        Notably not the Lie group exponential, since it is not surjective on
        Aff^+(2).
        """
        g = torch.zeros_like(A)
        t = A[..., :2]
        r = A[..., 2]
        s = A[..., 3:]
        mat_shape = (*g.shape[:-1], 2, 2)

        cos, sin = r.cos(), r.sin()
        R = torch.stack([cos, -sin, sin, cos], dim=-1).view(mat_shape)

        s_1, s_2, s_3 = s[..., 0], s[..., 1], s[..., 2]
        S = torch.matrix_exp(torch.stack([s_1, s_3, s_3, s_2], dim=-1).view(mat_shape))

        g[..., :2] = t
        g[..., 2:] = (R @ S).flatten(start_dim=-2, end_dim=-1)
        return g

    def __repr__(self):
        return "Aff^+(2)"


class MatrixGroup(ABC):
    """
    Class encapsulating basic Lie group and Lie algebra properties for groups
    that can be efficiently represented with matrices.

    Group multiplication, multiplication by inverse, and exponential make use of
    corresponding PyTorch methods. Since PyTorch does not implement a matrix
    logarithm, this must be provided.
    """

    def __init__(self, dim: int, mat_dim: int, lie_algebra_basis):
        super().__init__()
        self.dim = dim
        self.mat_dim = mat_dim
        self.lie_algebra_basis = lie_algebra_basis

    def L(self, R_1: torch.Tensor, R_2: torch.Tensor) -> torch.Tensor:
        """
        Left multiplication of `R_2` by `R_1`.
        """
        return R_1 @ R_2

    def L_inv(self, R_1: torch.Tensor, R_2: torch.Tensor) -> torch.Tensor:
        """
        Left multiplication of `R_2` by `R_1^-1`.
        """
        return torch.linalg.solve(R_1, R_2)

    @abstractmethod
    def log(self, R: torch.Tensor) -> torch.Tensor:
        """
        Lie group logarithm of `R`, i.e. `A` in Lie algebra such that
        `exp(A) = R`.

        Pytorch does not actually have a matrix log built-in, so this must be
        provided.
        """
        raise NotImplementedError

    def exp(self, A: torch.Tensor) -> torch.Tensor:
        """
        Lie group exponential of `A`, i.e. `R` in Lie group such that
        `exp(A) = R`.
        """
        return torch.matrix_exp(A)

    @abstractmethod
    def lie_algebra_components(self, A: torch.Tensor) -> torch.Tensor:
        """
        Compute the components of Lie algebra basis `A` with respect to the
        basis given by `self.lie_algebra_basis`.
        """
        raise NotImplementedError

    @torch.no_grad()
    def distance_matrix(
        self,
        R_1: torch.Tensor,
        R_2: torch.Tensor,
        w: torch.Tensor | None = None,
        chunk_size: int = 128,
    ) -> torch.Tensor:
        """
        Compute the pairwise logarithmic distances between `R_1` and `R_2`.

        Args:
            `R_1`: samples of shape [N, n, n].
            `R_2`: samples of shape [M, n, n].
          Optional:
            `w`: weights of the components of the generator, as in
            `LogarithmicDistance`. Defaults to all ones.
            `chunk_size`: number of rows of `D` computed at once, to limit memory
            usage. Defaults to 128.
        """
        D = torch.empty(len(R_1), len(R_2), device=R_2.device, dtype=R_2.dtype)
        for i in range(0, len(R_1), chunk_size):
            a = self.lie_algebra_components(self.log(self.L_inv(R_1[i:i + chunk_size, None], R_2[None])))
            if w is not None:
                a = w * a
            D[i:i + chunk_size] = a.norm(dim=-1)
        return D

    @torch.no_grad()
    def cost_matrices(
        self, R_1: torch.Tensor, R_2: torch.Tensor, w: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        Compute the squared logarithmic distances between all pairs within
        blocks, `C[k, i, j] = ρ(R_1[k, i], R_2[k, j])^2`, as the cost of a
        mini-batch optimal transport coupling.

        Args:
            `R_1`: samples of shape [B, b, n, n].
            `R_2`: samples of shape [B, b, n, n].
          Optional:
            `w`: weights of the components of the generator, as in
            `LogarithmicDistance`. Defaults to all ones.
        """
        return torch.stack(
            [self.distance_matrix(a, b, w, chunk_size=len(a)) ** 2 for a, b in zip(R_1, R_2)]
        )


class SO3(MatrixGroup):
    """
    Special orthogonal group of rotations on R^3.
    """

    def __init__(self, device: str | torch.DeviceObjType = "cpu"):
        super().__init__(
            dim=3,
            mat_dim=3 * 3,
            lie_algebra_basis=torch.tensor(
                [
                    [
                        [0.0, 0.0, 0.0],
                        [0.0, 0.0, -1.0],
                        [0.0, 1.0, 0.0],
                    ],
                    [
                        [0.0, 0.0, 1.0],
                        [0.0, 0.0, 0.0],
                        [-1.0, 0.0, 0.0],
                    ],
                    [
                        [0.0, -1.0, 0.0],
                        [1.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0],
                    ],
                ], device=device
            ),
        )

    def log(self, R, ε_π=0.1):
        """
        Lie group logarithm of `R`, i.e. `A` in Lie algebra such that
        `exp(A) = R`.

        Pytorch does not actually have a matrix log built in, but for SO(3) it
        is not too complicated. Writing `R = exp(θ [n]_×)`, the skew-symmetric
        part of `R` gives `sin(θ) n`, which vanishes as `θ -> π`. Hence, if `θ`
        is within `ε_π` of `π`, we instead extract the axis `n` from the
        symmetric part of `R`, which gives `n n^T`.
        """
        sin_n = self.lie_algebra_components(R - R.transpose(-2, -1)) / 2
        cos = (_trace(R) - 1) / 2
        θ = torch.atan2(sin_n.norm(dim=-1), cos)
        near_π = θ > torch.pi - ε_π

        # Away from π: A = θ / sin(θ) (R - R^T) / 2.
        a_regular = sin_n / _sinc(torch.where(near_π, 0.0, θ))[..., None]

        # Near π: n n^T = (sym(R) - cos(θ) I) / (1 - cos(θ)); every column is a
        # multiple of n, the one with the largest diagonal entry is the most
        # accurate. Its sign is fixed by sin(θ) n.
        nnT = (
            (R + R.transpose(-2, -1)) / 2 - cos[..., None, None] * torch.eye(3, device=R.device, dtype=R.dtype)
        ) / (1 - cos).clamp_min(1.0)[..., None, None]
        i = nnT.diagonal(dim1=-2, dim2=-1).argmax(-1)
        n = torch.gather(nnT, -1, i[..., None, None].expand(*i.shape, 3, 1))[..., 0]
        n = n / n.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        n = torch.where((n * sin_n).sum(-1, keepdim=True) < 0, -n, n)
        a_near_π = θ[..., None] * n

        a = torch.where(near_π[..., None], a_near_π, a_regular)
        return (a[..., None, None] * self.lie_algebra_basis).sum(-3)

    def exp(self, A, ε_stab=0.001):
        """
        Lie group exponential of `A`, i.e. `A` in Lie algebra such that
        `exp(A) = R`.

        While PyTorch has a built in matrix exp, it can still be more
        efficient and accurate to use a custom formula. For SO(3), it is
        well-known (Rodrigues' rotation formula) and not so difficult.
        """
        A_vec = self.lie_algebra_components(A)
        θ = (A_vec**2).sum(-1).clamp_min(1e-12).sqrt()[..., None, None]

        return (
            torch.eye(3, device=A.device, dtype=A.dtype)
            + _sinc(θ) * A
            + _sinc(θ / 2) ** 2 / 2 * A @ A
        )

    def lie_algebra_components(self, A):
        """
        Compute the components of Lie algebra basis `A` with respect to the
        basis given by `self.lie_algebra_basis`.
        """
        return torch.cat(
            (A[..., 2, 1, None], A[..., 0, 2, None], A[..., 1, 0, None]), dim=-1
        )

    @torch.no_grad()
    def cost_matrices(self, R_1, R_2, w=None):
        """
        Compute the squared logarithmic distances between all pairs within
        blocks, `C[k, i, j] = ρ(R_1[k, i], R_2[k, j])^2`. If the weights are
        uniform, `ρ(R, S) = w θ` with the rotation angle `θ` of `R^T S`, and
        `cos(θ) = (tr(R^T S) - 1) / 2`, where `tr(R^T S)` is the inner product
        of `R` and `S` as vectors in R^9. Hence no logarithms are needed.
        """
        if w is not None and not bool((w == w[0]).all()):
            return super().cost_matrices(R_1, R_2, w)
        traces = R_1.flatten(-2) @ R_2.flatten(-2).transpose(-1, -2)
        C = torch.arccos(((traces - 1) / 2).clamp(-1, 1)) ** 2
        return C if w is None else w[0] ** 2 * C

    def __repr__(self):
        return "SO(3)"


class SE3(MatrixGroup):
    """
    Special euclidean group of roto-translations on R^3.
    """

    def __init__(self, device: str | torch.DeviceObjType = "cpu"):
        super().__init__(
            dim=6,
            mat_dim=4 * 4,
            lie_algebra_basis=torch.tensor(
                [
                    [
                        [0.0, 0.0, 0.0, 1.0],
                        [0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0],
                    ],
                    [
                        [0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 1.0],
                        [0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0],
                    ],
                    [
                        [0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 1.0],
                        [0.0, 0.0, 0.0, 0.0],
                    ],
                    [
                        [0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, -1.0, 0.0],
                        [0.0, 1.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0],
                    ],
                    [
                        [0.0, 0.0, 1.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0],
                        [-1.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0],
                    ],
                    [
                        [0.0, -1.0, 0.0, 0.0],
                        [1.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0],
                    ],
                ], device=device
            ),
        )
        self.so3 = SO3(device=device)

    def log(self, g, ε_stab=0.001):
        """
        Lie group logarithm of `R`, i.e. `A` in Lie algebra such that
        `exp(A) = R`.

        Pytorch does not actually have a matrix log built in, but for SE(3) it
        is not too complicated.
        """
        A = torch.zeros_like(g)
        c, t_par, ω = self.get_screw_displacement_generator(g, ε_stab=ε_stab)
        v = -(ω @ c[..., None])[..., 0] + t_par
        A = self.pack_translation_rotation(A, v, ω)
        return A

    def get_screw_displacement_generator(self, g, ε_stab=0.001):
        """
        Decompose `g` as a screw motion: return a point `c` on the screw axis,
        the translation `t_par` along the axis, and the generator `ω` of the
        rotation.
        """
        t, R = self.get_translation_rotation(g)
        ω = self.so3.log(R)
        ω_vec = self.so3.lie_algebra_components(ω)
        θ = (ω_vec**2).sum(-1, keepdim=True).sqrt()

        parallel = θ < ε_stab

        K = (ω_vec / θ).nan_to_num()

        t_par = (t * K).sum(-1, keepdim=True) * K
        t_perp = t - t_par

        c = 0.5 * (t_perp + cross_product(_cotan(θ / 2.0) * K, t_perp)).nan_to_num()

        return c * ~parallel, t * parallel + t_par * ~parallel, ω

    def exp(self, A, ε_stab=0.0001):
        """
        Lie group exponential of `A`. 

        While PyTorch has a built in matrix exp, it can still be more
        efficient and accurate to use a custom formula. For SE(3), we
        can use Rodrigues' rotation formula.
        """
        ω, v = A[..., :3, :3], A[..., :3, 3:]
        θ_squared = (self.so3.lie_algebra_components(ω) ** 2).sum(-1)[..., None, None]
        return _exp_affine(ω, v, θ_squared, ε_stab)

    def get_translation_rotation(self, g):
        t = g[..., :3, -1]
        R = g[..., :3, :3]
        return t, R

    def pack_translation_rotation(self, g, t, R):
        g[..., :3, -1] = t
        g[..., :3, :3] = R
        return g

    def lie_algebra_components(self, A):
        """
        Compute the components of Lie algebra basis `A` with respect to the
        basis given by `self.lie_algebra_basis`.
        """
        return torch.cat(
            (
                A[..., 0, 3, None],
                A[..., 1, 3, None],
                A[..., 2, 3, None],
                A[..., 2, 1, None],
                A[..., 0, 2, None],
                A[..., 1, 0, None],
            ),
            dim=-1,
        )

    def __repr__(self):
        return "SE(3)"


class SOplus12(MatrixGroup):
    """
    Orthochronous Lorentz group on R^{1, 2}.
    """

    def __init__(self, device: str | torch.DeviceObjType = "cpu"):
        super().__init__(
            dim=3,
            mat_dim=3 * 3,
            lie_algebra_basis=torch.tensor(
                [
                    [
                        [0.0, 0.0, 0.0],
                        [0.0, 0.0, 1.0],
                        [0.0, -1.0, 0.0],
                    ],
                    [
                        [0.0, 0.0, 1.0],
                        [0.0, 0.0, 0.0],
                        [1.0, 0.0, 0.0],
                    ],
                    [
                        [0.0, -1.0, 0.0],
                        [-1.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0],
                    ],
                ], device=device
            ),
        )
        self.η = torch.tensor(
            [
                [1.0, 0.0, 0.0],
                [0.0, -1.0, 0.0],
                [0.0, 0.0, -1.0],
            ], device=device
        )

    def log(self, Λ, ε_stab=0.001, ε_π=1.5):
        """
        Lie group logarithm of `Λ`, i.e. `A` in Lie algebra such that
        `exp(A) = Λ`.

        Pytorch does not actually have a matrix log built in, but for SO^+(1, 2) it
        is not too complicated. Writing `Λ = exp(θ Â)` with `Â` a unit generator
        and `c = (tr(Λ) - 1) / 2`, we have `(Λ - Λ^{-1}) / 2 = A / f(c)`, where
        `f(c) = θ / sin(θ)` with `c = cos(θ)` if `A` is timelike, `θ / sinh(θ)`
        with `c = cosh(θ)` if `A` is spacelike, and `1` if `A` is lightlike.
        Since `f` is analytic around `c = 1`, we use its Taylor expansion if
        `|1 - c| < ε_stab`, so that we need not classify `A`.

        If `A` is timelike and `θ` is within `ε_π` of `π`, `f` is ill-conditioned
        and blows up as `sin(θ)` vanishes. We then instead extract the axis `k` of `Â`,
        normalised such that `k^T η k = 1`, from `(Λ + Λ^{-1}) / 2 = I + (1 - c) Â^2`
        using `Â^2 + I = k k^T η`.
        """
        η = self.η.to(Λ.dtype)
        Λ_inv = η @ Λ.transpose(-2, -1) @ η
        k_tilde = self.lie_algebra_components(Λ - Λ_inv) / 2
        c = (_trace(Λ) - 1) / 2
        u = 1 - c
        near_identity = u.abs() < ε_stab
        timelike = c < 1
        near_π = c < -math.cos(ε_π)

        # Away from π: A = f(c) (Λ - Λ^{-1}) / 2.
        c_timelike = torch.where(timelike & ~near_identity & ~near_π, c, 0.0)
        c_spacelike = torch.where(~timelike & ~near_identity, c, 2.0)
        f = torch.where(
            near_identity,
            1 + u / 3 + 2 * u**2 / 15 + 2 * u**3 / 35,
            torch.where(
                timelike,
                torch.arccos(c_timelike) / ((1 - c_timelike) * (1 + c_timelike)).sqrt(),
                torch.arccosh(c_spacelike) / ((c_spacelike - 1) * (c_spacelike + 1)).sqrt(),
            ),
        )
        a_regular = f[..., None] * k_tilde

        # Near π: the first column of k k^T η is k_0 k, with k_0^2 >= 1.
        kkTη = (
            (Λ + Λ_inv) / 2 - c[..., None, None] * torch.eye(3, device=Λ.device, dtype=Λ.dtype)
        ) / u.clamp_min(ε_stab)[..., None, None]
        k = kkTη[..., 0] / kkTη[..., 0, 0, None].clamp_min(1.0).sqrt()
        # (Λ - Λ^{-1}) / 2 = sin(θ) Â, so this gives θ including its sign.
        θ = torch.atan2((k_tilde @ η * k).sum(-1), c)
        a_near_π = θ[..., None] * k

        a = torch.where(near_π[..., None], a_near_π, a_regular)
        return (a[..., None, None] * self.lie_algebra_basis.to(Λ.dtype)).sum(-3)

    def exp(self, A, ε_stab=0.0001):
        """
        Lie group exponential of `A`.

        While PyTorch has a built in matrix exp, it can still be more
        efficient and accurate to use a custom formula. For SO^+(1, 2), 
        we can use a generalisation of Rodrigues' rotation formula.
        """
        A_vec = self.lie_algebra_components(A)
        s_squared = (A_vec @ self.η.to(A.dtype) * A_vec).sum(-1)[..., None, None]
        F_1, F_2, _ = _rodrigues_coefficients(s_squared, ε_stab)

        return torch.eye(3, device=A.device, dtype=A.dtype) + F_1 * A + F_2 * A @ A

    def lie_algebra_components(self, A):
        """
        Compute the components of Lie algebra basis `A` with respect to the
        basis given by `self.lie_algebra_basis`.
        """
        return torch.cat(
            (A[..., 1, 2, None], A[..., 0, 2, None], -A[..., 1, 0, None]), dim=-1
        )

    def __repr__(self):
        return "SO^+(1, 2)"


class SEplus12(MatrixGroup):
    """
    Orthochronous Poincaré group on R^{1,2}.
    """

    def __init__(self, device: str | torch.DeviceObjType = "cpu"):
        super().__init__(
            dim=6,
            mat_dim=4 * 4,
            lie_algebra_basis=torch.tensor(
                [
                    [
                        [0.0, 0.0, 0.0, 1.0],
                        [0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0],
                    ],
                    [
                        [0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 1.0],
                        [0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0],
                    ],
                    [
                        [0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 1.0],
                        [0.0, 0.0, 0.0, 0.0],
                    ],
                    [
                        [0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 1.0, 0.0],
                        [0.0, -1.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0],
                    ],
                    [
                        [0.0, 0.0, 1.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0],
                        [1.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0],
                    ],
                    [
                        [0.0, -1.0, 0.0, 0.0],
                        [-1.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0],
                    ],
                ], device=device
            ),
        )
        self.soplus12 = SOplus12(device=device)
        self.η = self.soplus12.η

    def log(self, g, ε_stab=0.001):
        """
        Lie group logarithm of `g`, i.e. `A` in Lie algebra such that
        `exp(A) = g`.

        Pytorch does not actually have a matrix log built in, but for SE^+(1, 2)
        it is not too complicated.
        """
        A = torch.zeros_like(g)
        c, t_par, ω = self.get_screw_displacement_generator(g, ε_stab=ε_stab)
        v = -(ω @ c[..., None])[..., 0] + t_par

        ω_vec = self.soplus12.lie_algebra_components(ω)
        s_squared = (ω_vec @ self.η * ω_vec).sum(-1, keepdim=True)
        lightlike = s_squared.abs() < ε_stab
        t_light = t_par[..., None]

        A = self.pack_translation_lorentz(
            A,
            (
                v * ~lightlike
                + (t_light - 1.0 / 2.0 * ω @ t_light + 1.0 / 12.0 * ω @ ω @ t_light)[
                    ..., 0
                ]
                * lightlike
            ),
            ω,
        )
        return A

    def get_screw_displacement_generator(self, g, ε_stab=0.001):
        """
        Decompose `g` as a screw motion: return a point `c` on the axis, the
        translation `t_par` along the axis, and the generator `ω` of the
        Lorentz transformation.
        """
        t, Λ = self.get_translation_lorentz(g)
        ω = self.soplus12.log(Λ, ε_stab=ε_stab)
        ω_vec = self.soplus12.lie_algebra_components(ω)

        s_squared = (ω_vec @ self.η * ω_vec).sum(-1, keepdim=True)
        s = s_squared.abs().sqrt()

        lightlike = s_squared.abs() < ε_stab
        timelike = (s_squared > 0.0) * ~lightlike
        spacelike = (s_squared < 0.0) * ~lightlike

        t_par = (
            (ω_vec @ self.η * t).sum(-1, keepdim=True) / s_squared * ω_vec
        ).nan_to_num(0)
        t_perp = t - t_par

        c_timelike = (
            0.5
            * (
                t_perp + hyperbolic_cross_product(_cotan(s / 2.0) * ω_vec / s, t_perp)
            ).nan_to_num()
        )

        c_spacelike = (
            0.5
            * (
                t_perp - hyperbolic_cross_product(_cotanh(s / 2.0) * ω_vec / s, t_perp)
            ).nan_to_num()
        )

        return (
            c_timelike * timelike + c_spacelike * spacelike,
            t * lightlike + t_par * ~lightlike,
            ω,
        )

    def exp(self, A, ε_stab=0.0001):
        """
        Lie group exponential of `A`.

        While PyTorch has a built in matrix exp, it can still be more
        efficient and accurate to use a custom formula. For SE^+(1, 2),
        we can use a generalisation of Rodrigues' rotation formula.
        """
        ω, v = A[..., :3, :3], A[..., :3, 3:]
        k = self.soplus12.lie_algebra_components(ω)
        s_squared = (k @ self.η.to(A.dtype) * k).sum(-1)[..., None, None]
        return _exp_affine(ω, v, s_squared, ε_stab)

    def get_translation_lorentz(self, g):
        t = g[..., :3, -1]
        Λ = g[..., :3, :3]
        return t, Λ

    def pack_translation_lorentz(self, g, t, Λ):
        g[..., :3, -1] = t
        g[..., :3, :3] = Λ
        return g

    def lie_algebra_components(self, A):
        """
        Compute the components of Lie algebra basis `A` with respect to the
        basis given by `self.lie_algebra_basis`.
        """
        return torch.cat(
            (
                A[..., 0, 3, None],
                A[..., 1, 3, None],
                A[..., 2, 3, None],
                A[..., 1, 2, None],
                A[..., 0, 2, None],
                -A[..., 1, 0, None],
            ),
            dim=-1,
        )

    def __repr__(self):
        return "SE^+(1, 2)"


class HomogeneousSpace(ABC):
    """
    Class encapsulating basic properties of a homogeneous space of a
    `MatrixGroup`.
    """

    def __init__(self, G: MatrixGroup, mat_dim: int):
        super().__init__()
        self.G = G
        self.mat_dim = mat_dim

    @abstractmethod
    def get_generator(self, p_1: torch.Tensor, p_2: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def act(self, g: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


    @torch.no_grad()
    def distance_matrix(
        self,
        p_1: torch.Tensor,
        p_2: torch.Tensor,
        w: torch.Tensor | None = None,
        chunk_size: int = 128,
    ) -> torch.Tensor:
        """
        Compute the matrix of logarithmic distances `D[i, j] = ρ(p_1[i], p_2[j])`
        with `ρ(p, q) = ||w * A||`, where `A` is the generator from
        `get_generator(p, q)` and the norm is taken of its components with
        respect to the Lie algebra basis of `G`.

        Args:
            `p_1`: samples of shape [N, ...].
            `p_2`: samples of shape [M, ...].
          Optional:
            `w`: weights of the components of the generator, as in
            `LogarithmicDistance`. Defaults to all ones.
            `chunk_size`: number of rows of `D` computed at once, to limit memory
            usage. Defaults to 128.
        """
        D = torch.empty(len(p_1), len(p_2), device=p_2.device, dtype=p_2.dtype)
        for i in range(0, len(p_1), chunk_size):
            a = self.G.lie_algebra_components(self.get_generator(p_1[i:i + chunk_size, None], p_2[None]))
            if w is not None:
                a = w * a
            D[i:i + chunk_size] = a.norm(dim=-1)
        return D

    @torch.no_grad()
    def cost_matrices(
        self, p_1: torch.Tensor, p_2: torch.Tensor, w: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        Compute the squared logarithmic distances between all pairs within
        blocks, `C[k, i, j] = ρ(p_1[k, i], p_2[k, j])^2`, as the cost of a
        mini-batch optimal transport coupling.

        Args:
            `p_1`: samples of shape [B, b, ...].
            `p_2`: samples of shape [B, b, ...].
          Optional:
            `w`: weights of the components of the generator, as in
            `LogarithmicDistance`. Defaults to all ones.
        """
        return torch.stack(
            [self.distance_matrix(a, b, w, chunk_size=len(a)) ** 2 for a, b in zip(p_1, p_2)]
        )


class M3(HomogeneousSpace):
    """
    Position-Orientation space on 3D Euclidean space.
    """

    def __init__(self, generator: Literal["mav", "pure_rotation"] | float = "mav", device: str | torch.DeviceObjType = "cpu"):
        self.se3 = SE3(device=device)
        super().__init__(G=self.se3, mat_dim=4 * 2)
        self.generator = generator

    def get_generator(
        self,
        p_1,
        p_2,
        ε_stab=0.001,
        generator: Literal["mav", "pure_rotation"] | float | None = None,
    ):
        """
        Compute a generator between `p_1` and `p_2` [1, Prop. 1].
        Choose either the `mav` or the `pure_rotation` generator, or supply a
        number to select the corresponding rotation axis.

        References:
          [1]: G. Bellaard and B.M.N. Smets. "Roto-Translation Invariant Metrics
          on Position-Orientation Space." 7th International Conference on
          Geometric Science of Information (2025).
          DOI:10.1007/978-3-032-03918-7_4.
        """
        if generator is None:
            generator = self.generator
        shape = torch.broadcast_shapes(p_1.shape, p_2.shape)[:-2]
        device = p_2.device
        x_1, n_1 = self.get_position_orientation(p_1)
        x_2, n_2 = self.get_position_orientation(p_2)
        x_m = (x_1 + x_2) / 2.0
        x_diff = x_2 - x_1

        cross_n = cross_product(n_1, n_2)
        sinθ = cross_n.norm(dim=-1, keepdim=True)
        cosθ = (n_1 * n_2).sum(-1, keepdim=True)
        θ = torch.atan2(sinθ, cosθ)
        parallel = θ < ε_stab

        k0 = (cross_n / sinθ).nan_to_num()
        k0 = k0 * ~parallel + torch.zeros_like(k0) * parallel
        use_mav = x_diff.norm(dim=-1, keepdim=True) < ε_stab
        x_perp = (k0 * x_diff).sum(-1, keepdim=True) * k0
        x_par = x_diff - x_perp

        c_mav = (x_m + 0.5 * _cotan(θ / 2.0) * cross_product(k0, x_par)).nan_to_num()
        v_mav = x_perp

        ω_vec_mav = θ * k0
        ω_mav = (ω_vec_mav[..., None, None] * self.se3.so3.lie_algebra_basis).sum(-3)

        A = torch.zeros(*shape, 4, 4, device=device, dtype=p_2.dtype)
        match generator:
            case "mav":
                return self.se3.pack_translation_rotation(
                    A,
                    x_diff * parallel
                    + (-cross_product(ω_vec_mav, c_mav) + v_mav) * ~parallel,
                    ω_mav * ~parallel[..., None],
                )
            case "pure_rotation":
                k = cross_product(x_diff, n_2 - n_1)
                k = (k / k.norm(dim=-1, keepdim=True)).nan_to_num()
            case float() | int():
                φ = torch.tensor([generator], device=device, dtype=p_2.dtype)
                khalfπ = (n_1 + n_2) / (n_1 + n_2).norm(
                    dim=-1, keepdim=True
                ).nan_to_num()
                k = torch.cos(φ) * k0 + torch.sin(φ) * khalfπ
            case _:
                raise ValueError(f"{generator} is not a supported type of generator!")

        n_1p = n_1 - (k * n_1).sum(-1, keepdim=True) * k
        n_2p = n_2 - (k * n_2).sum(-1, keepdim=True) * k
        θ = torch.atan2(
            (k * cross_product(n_1p, n_2p)).sum(-1, keepdim=True),
            (n_1p * n_2p).sum(-1, keepdim=True),
        )

        x_perp = (k * x_diff).sum(-1, keepdim=True) * k
        x_par = x_diff - x_perp

        c = (x_m + 0.5 * _cotan(θ / 2.0) * cross_product(k, x_par)).nan_to_num()
        v = x_perp

        ω_vec = θ * k
        ω = (ω_vec[..., None, None] * self.se3.so3.lie_algebra_basis).sum(-3)

        return self.se3.pack_translation_rotation(
            A,
            x_diff * parallel
            + (
                (-cross_product(ω_vec, c) + v) * ~use_mav
                + (-cross_product(ω_vec_mav, c_mav) + v_mav) * use_mav
            )
            * ~parallel,
            (ω * ~use_mav[..., None] + ω_mav * use_mav[..., None])
            * ~parallel[..., None],
        )

    @torch.no_grad()
    def cost_matrices(self, p_1, p_2, w=None, ε_stab=0.001):
        """
        Compute the squared logarithmic distances between all pairs within
        blocks, `C[k, i, j] = ρ(p_1[k, i], p_2[k, j])^2`.

        For the `mav` generator with weights `w_t` for all translational and
        `w_r` for all rotational components, this only needs inner products of
        the samples, i.e. matrix multiplications. With `cos(θ) = n_1 . n_2`,
        `k = n_1 x n_2 / sin(θ)`, `h = θ / 2 cot(θ / 2)`, `Δx = x_2 - x_1`, and
        `x_m = (x_1 + x_2) / 2`, the generator has rotational part `θ k` and
        translational part `(k . Δx) k + h (Δx - (k . Δx) k) - θ k x x_m`, so
          `ρ^2 = w_r^2 θ^2 + w_t^2 ((k . Δx)^2 + h^2 (|Δx|^2 - (k . Δx)^2)
                 + θ^2 (|x_m|^2 - (k . x_m)^2) - 2 h θ k . (x_1 x x_2))`,
        where `sin(θ) k . x_2 = n_1 . (n_2 x x_2)`,
        `sin(θ) k . x_1 = (x_1 x n_1) . n_2`, and
        `sin(θ) k . (x_1 x x_2) = (n_1 . x_1) (n_2 . x_2) - (n_1 . x_2) (x_1 . n_2)`.
        Near `θ = 0`, `sin(θ)` loses precision, so a few nearly parallel pairs
        have a slightly inaccurate cost; it is meant for couplings, use
        `distance_matrix` for metrics.
        """
        w_t, w_r = (1.0, 1.0) if w is None else (w[0], w[3])
        if self.generator != "mav" or (
            w is not None and not bool((w[:3] == w_t).all() and (w[3:] == w_r).all())
        ):
            return super().cost_matrices(p_1, p_2, w)
        x_1, n_1 = self.get_position_orientation(p_1)
        x_2, n_2 = self.get_position_orientation(p_2)
        inner = lambda a, b: a @ b.transpose(-1, -2)

        cosθ = inner(n_1, n_2).clamp(-1.0, 1.0)
        sinθ = (1.0 - cosθ**2).clamp_min(0.0).sqrt()
        θ = torch.atan2(sinθ, cosθ)
        parallel = θ < ε_stab
        sinθ = torch.where(parallel, 1.0, sinθ)
        k_x_2 = inner(n_1, cross_product(n_2, x_2)) / sinθ
        k_x_1 = inner(cross_product(x_1, n_1), n_2) / sinθ
        k_x_1_x_2 = (
            (n_1 * x_1).sum(-1)[..., :, None] * (n_2 * x_2).sum(-1)[..., None, :]
            - inner(n_1, x_2) * inner(x_1, n_2)
        ) / sinθ
        k_Δx = k_x_2 - k_x_1
        k_x_m = (k_x_1 + k_x_2) / 2.0

        x_1_x_2 = inner(x_1, x_2)
        x_1_squared = (x_1**2).sum(-1)[..., :, None]
        x_2_squared = (x_2**2).sum(-1)[..., None, :]
        Δx_squared = x_1_squared + x_2_squared - 2.0 * x_1_x_2
        x_m_squared = (x_1_squared + x_2_squared + 2.0 * x_1_x_2) / 4.0

        h = torch.where(parallel, 1.0, (θ / 2.0) * _cotan(torch.where(parallel, 1.0, θ / 2.0)))
        v_squared = (
            k_Δx**2
            + h**2 * (Δx_squared - k_Δx**2)
            + θ**2 * (x_m_squared - k_x_m**2)
            - 2.0 * h * θ * k_x_1_x_2
        )
        C = torch.where(parallel, w_t**2 * Δx_squared, w_t**2 * v_squared + w_r**2 * θ**2)
        return C.clamp_min(0.0)

    def act(self, g, p):
        return g @ p

    def get_position_orientation(self, p):
        x = p[..., :3, -2]
        n = p[..., :3, -1]
        return x, n

    def pack_position_orientation(self, x, n):
        p = torch.zeros(*n.shape[:-1], 4, 2, device=n.device, dtype=n.dtype)
        p[..., :3, -2] = x
        p[..., 3, -2] = 1
        p[..., :3, -1] = n
        return p

    def __repr__(self):
        return "M3"


class P3(HomogeneousSpace):
    """
    Event-Velocity space on 3D flat spacetime.

    We choose P3 := {(x, n) in R^3 x R^3 | ||n|| = 1, nt >= 1}.
    """

    def __init__(
        self,
        generator: Literal["timelike", "lightlike", "spacelike"] | float = "spacelike",
        device: str | torch.DeviceObjType = "cpu",
    ):
        self.seplus12 = SEplus12(device=device)
        self.η = self.seplus12.η
        super().__init__(G=self.seplus12, mat_dim=4 * 2)
        self.generator = generator

    def get_generator(
        self,
        p_1,
        p_2,
        generator: Literal["timelike", "lightlike", "spacelike"] | float | None = None,
    ):
        """
        Compute a generator between `p_1` and `p_2`.

        Let `p_1 = (x_1, n_1)` and `p_2 = (x_2, n_2)`. Then
        - `generator == "spacelike"` gives the generator where the axis `ks` of
          the Lorentz transformation is the spacelike `n_1 x n_2`, normalised to
          -1.
        - `generator == "timelike"` gives the generator where the axis `kt` of
          the Lorentz transformation is the timelike `n_1 + n_2`, normalised to
          1.
        - `generator == "lightlike"` gives the generator where the axis `kl` of
          the Lorentz transformation is the lightlike `ks + kt`.
        - `generator == φ` for some float `φ` gives the generator where the axis
          of the Lorentz transformation is `cosh(φ) ks - sinh(φ) kt`, which has
          norm -1.
        """
        if generator is None:
            generator = self.generator
        shape = torch.broadcast_shapes(p_1.shape, p_2.shape)[:-2]
        device = p_2.device
        x_1, n_1 = self.get_event_velocity(p_1)
        x_2, n_2 = self.get_event_velocity(p_2)
        x_m = (x_1 + x_2) / 2.0
        x_diff = x_2 - x_1

        ks = hyperbolic_cross_product(n_1, n_2)
        ks = ks / (-(ks @ self.η * ks).sum(-1, keepdim=True)).sqrt()
        kt = n_1 + n_2
        kt = kt / (kt @ self.η * kt).sum(-1, keepdim=True).sqrt()

        i = (n_1 @ self.η * n_2).sum(-1, keepdim=True)

        match generator:
            case "spacelike":
                # The axis `ks` is undefined if `n_1 = n_2`, and `arccosh(i)` is
                # ill-conditioned for `i -> 1`. Hence, we write `n_1 x n_2 =
                # sinh(w) ks`, so that `k = w ks = asinh(s) / s n_1 x n_2` with
                # `s = sinh(w)`, and write the translation part
                # `-k x (x_m - coth(w / 2) / 2 ks x x_par) + x_perp` as
                # `x_diff - k x x_m + (w / 2 coth(w / 2) - 1) x_par`, using
                # `ks x (ks x x_par) = x_par`. All terms are then well-defined
                # and accurate as `n_2 -> n_1`.
                ε_stab = 1e-2
                n_cross = hyperbolic_cross_product(n_1, n_2)
                s_squared = (-(n_cross @ self.η * n_cross).sum(-1, keepdim=True)).clamp_min(0.0)
                positive = s_squared > 0.0
                s = torch.where(positive, torch.where(positive, s_squared, 1.0).sqrt(), 0.0)
                w = torch.asinh(s)
                small = s < ε_stab
                asinhc = torch.where(
                    small, 1.0 - s**2 / 6.0 + 3.0 * s**4 / 40.0, w / torch.where(small, 1.0, s)
                )
                k = asinhc * n_cross

                w_safe = torch.where(small, 1.0, w)
                xcoth_minus_1 = torch.where(
                    small, w**2 / 12.0 - w**4 / 720.0, (w_safe / 2.0) / torch.tanh(w_safe / 2.0) - 1.0
                )
                ks_safe = n_cross / torch.where(positive, s, 1.0)
                x_par = x_diff + (ks_safe @ self.η * x_diff).sum(-1, keepdim=True) * ks_safe

                c = torch.zeros_like(x_diff)
                v = x_diff - hyperbolic_cross_product(k, x_m) + xcoth_minus_1 * x_par
            case "timelike":
                θ = torch.tensor([torch.pi], device=device, dtype=p_2.dtype)
                k = θ * kt

                x_perp = (kt @ self.η * x_diff).sum(-1, keepdim=True) * kt
                x_par = x_diff - x_perp

                c = x_m
                v = x_perp
            case "lightlike":
                s = 2 * ((i - 1.0) / (i + 1.0)).sqrt()
                k = s * (ks + kt)
                ω = (k[..., None, None] * self.seplus12.soplus12.lie_algebra_basis).sum(
                    -3
                )

                t = x_2[..., None] - self.seplus12.soplus12.exp(ω) @ x_1[..., None]
                c = torch.zeros_like(x_diff)
                v = (t - 1.0 / 2.0 * ω @ t + 1.0 / 12.0 * ω @ ω @ t)[..., 0]
            case float() | int():
                φ = torch.tensor([generator], device=device, dtype=p_2.dtype)
                kφ = torch.cosh(φ) * ks - torch.sinh(φ) * kt

                n_1p = n_1 + (kφ @ self.η * n_1).sum(-1, keepdim=True) * kφ
                n_2p = n_2 + (kφ @ self.η * n_2).sum(-1, keepdim=True) * kφ
                s = torch.atanh(
                    (kφ @ self.η * hyperbolic_cross_product(n_1p, n_2p)).sum(
                        -1, keepdim=True
                    )
                    / -(n_1p @ self.η * n_2p).sum(-1, keepdim=True)
                )
                k = s * kφ

                x_perp = -1.0 * (kφ @ self.η * x_diff).sum(-1, keepdim=True) * kφ
                x_par = x_diff - x_perp

                c = (
                    x_m - 0.5 * _cotanh(s / 2.0) * hyperbolic_cross_product(kφ, x_par)
                ).nan_to_num()
                v = x_perp
            case _:
                raise ValueError(f"{generator} is not a supported type of generator!")

        ω = (k[..., None, None] * self.seplus12.soplus12.lie_algebra_basis).sum(-3)

        A = torch.zeros(*shape, 4, 4, device=device, dtype=p_2.dtype)

        return self.seplus12.pack_translation_lorentz(
            A,
            -hyperbolic_cross_product(k, c) + v,
            ω,
        )

    @torch.no_grad()
    def cost_matrices(self, p_1, p_2, w=None, ε_stab=1e-2):
        """
        Compute the squared logarithmic distances between all pairs within
        blocks, `C[k, i, j] = ρ(p_1[k, i], p_2[k, j])^2`.

        For the `spacelike` generator with weights `w_t` for all translational
        and `w_r` for all Lorentz components, this only needs inner products of
        the samples, i.e. matrix multiplications. Write `u = n_1 x n_2` for the
        Euclidean cross product, so that the hyperbolic cross product is
        `n_1 x_η n_2 = η u`, and `s^2 = -u^T η u`, `w = asinh(s)`,
        `α = w / s`, `β = w / 2 coth(w / 2) - 1`, `γ = β / s^2`,
        `Δx = x_2 - x_1`, `x_m = (x_1 + x_2) / 2`, `p = u . Δx`. Then the
        generator has Lorentz part `α η u` and translational part
        `v = (1 + β) Δx - α Q + γ p η u` with `Q = u x η x_m`. Since
        `η = diag(1, -1, -1)` is a rotation, `η a . η b = a . b` and
        `η (a x b) = η a x η b`, so all inner products of `Δx`, `Q`, and `η u`
        reduce to inner products of the samples by
        `(a x b) . (c x d) = (a . c) (b . d) - (a . d) (b . c)`, e.g.
        `Q . η u = -2 u_t u . (e_t x η x_m)` using `u x η u = 2 u_t u x e_t`.
        Expanding the squares loses relative precision for very close pairs
        (absolute error ~1e-3 in single precision); it is meant for couplings,
        use `distance_matrix` for metrics.
        """
        w_t, w_r = (1.0, 1.0) if w is None else (w[0], w[3])
        if self.generator != "spacelike" or (
            w is not None and not bool((w[:3] == w_t).all() and (w[3:] == w_r).all())
        ):
            return super().cost_matrices(p_1, p_2, w)
        x_1, n_1 = self.get_event_velocity(p_1)
        x_2, n_2 = self.get_event_velocity(p_2)
        η = self.η.diagonal().to(x_1.dtype)
        e_t = torch.zeros_like(n_2)
        e_t[..., 0] = 1.0
        inner = lambda a, b: a @ b.transpose(-1, -2)
        inner_1 = lambda a, b: (a * b).sum(-1)[..., :, None]  # of sample 1, along rows
        inner_2 = lambda a, b: (a * b).sum(-1)[..., None, :]  # of sample 2, along columns
        # u . y_1 = n_2 . (y_1 x n_1), u . y_2 = n_1 . (n_2 x y_2)
        u_dot_1 = lambda y_1: inner(cross_product(y_1, n_1), n_2)
        u_dot_2 = lambda y_2: inner(n_1, cross_product(n_2, y_2))

        u_squared = inner_1(n_1, n_1) * inner_2(n_2, n_2) - inner(n_1, n_2) ** 2
        u_t = u_dot_2(e_t)
        s_squared = (u_squared - 2.0 * u_t**2).clamp_min(0.0)
        p = u_dot_2(x_2) - u_dot_1(x_1)
        u_ηx_m = (u_dot_1(η * x_1) + u_dot_2(η * x_2)) / 2.0

        x_1_x_2 = inner(x_1, x_2)
        x_1_squared, x_2_squared = inner_1(x_1, x_1), inner_2(x_2, x_2)
        Δx_squared = x_1_squared + x_2_squared - 2.0 * x_1_x_2
        x_m_squared = (x_1_squared + x_2_squared + 2.0 * x_1_x_2) / 4.0
        n_1_ηx_m = (inner_1(n_1, η * x_1) + inner(n_1, η * x_2)) / 2.0
        n_2_ηx_m = (inner(η * x_1, n_2) + inner_2(n_2, η * x_2)) / 2.0
        n_1_Δx = inner(n_1, x_2) - inner_1(n_1, x_1)
        n_2_Δx = inner_2(n_2, x_2) - inner(x_1, n_2)

        Q_Q = u_squared * x_m_squared - u_ηx_m**2
        Δx_Q = n_1_ηx_m * n_2_Δx - n_1_Δx * n_2_ηx_m
        Δx_ηu = u_dot_2(η * x_2) - u_dot_1(η * x_1)
        Q_ηu = -2.0 * u_t * (inner_1(n_1, e_t) * n_2_ηx_m - n_1_ηx_m * inner_2(n_2, e_t))

        # Series for small s, as in `get_generator`.
        s = s_squared.sqrt()
        small = s < ε_stab
        w_ = torch.asinh(s)
        w_safe = torch.where(small, 1.0, w_)
        α = torch.where(small, 1.0 - s_squared / 6.0 + 3.0 * s_squared**2 / 40.0, w_ / torch.where(small, 1.0, s))
        β = torch.where(small, w_**2 / 12.0 - w_**4 / 720.0, (w_safe / 2.0) / torch.tanh(w_safe / 2.0) - 1.0)
        γ = torch.where(small, (1.0 / 12.0 - w_**2 / 720.0) * α**2, β / torch.where(small, 1.0, s_squared))

        a, b, c = 1.0 + β, -α, γ * p
        v_squared = (
            a**2 * Δx_squared
            + b**2 * Q_Q
            + c**2 * u_squared
            + 2.0 * a * b * Δx_Q
            + 2.0 * a * c * Δx_ηu
            + 2.0 * b * c * Q_ηu
        )
        return (w_t**2 * v_squared + w_r**2 * α**2 * u_squared).clamp_min(0.0)

    def act(self, g, p):
        return g @ p

    def get_event_velocity(self, p):
        x = p[..., :3, -2]
        n = p[..., :3, -1]
        return x, n

    def pack_event_velocity(self, x, n):
        p = torch.zeros(*n.shape[:-1], 4, 2, device=n.device, dtype=n.dtype)
        p[..., :3, -2] = x
        p[..., 3, -2] = 1
        p[..., :3, -1] = n
        return p

    def __repr__(self):
        return "P3"


# Utils


def _mod_offset(x, period, offset):
    """Compute `x` modulo `period` with offset `offset`."""
    return x - (x - offset) // period * period


def _trace(R: torch.Tensor) -> torch.Tensor:
    return R.diagonal(offset=0, dim1=-1, dim2=-2).sum(-1)


def _expc(x: torch.Tensor) -> torch.Tensor:
    """Compute `x / (exp(x) - 1)`."""
    return torch.where(
        x.abs() < 1.0,
        1.0 - x / 2.0 + x**2 / 12.0 - x**4 / 720.0 + x**6 / 30240.0,
        x / (torch.exp(x) - 1.0),
    )


def cross_product(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    shape = torch.broadcast_shapes(x.shape, y.shape)
    return torch.linalg.cross(x.expand(shape), y.expand(shape))


def hyperbolic_cross_product(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    shape = torch.broadcast_shapes(x.shape, y.shape)
    z = torch.zeros(*shape, device=x.device, dtype=x.dtype)

    x1 = x[..., 0]
    x2 = x[..., 1]
    x3 = x[..., 2]
    y1 = y[..., 0]
    y2 = y[..., 1]
    y3 = y[..., 2]
    z[..., 0] = x2 * y3 - x3 * y2
    z[..., 1] = x1 * y3 - x3 * y1
    z[..., 2] = x2 * y1 - x1 * y2

    return z


def _cotan(x: torch.Tensor) -> torch.Tensor:
    return 1 / torch.tan(x)


def _cotanh(x: torch.Tensor) -> torch.Tensor:
    return 1 / torch.tanh(x)


def _sinc(x: torch.Tensor) -> torch.Tensor:
    return torch.sinc(x / torch.pi)


def _sinhc(x: torch.Tensor) -> torch.Tensor:
    return torch.where(
        x.abs() < 1.0,
        1 + x**2 / 6.0 + x**4 / 120.0 + x**6 / 5040 + x**8 / 362880 + x**10 / 39916800,
        torch.sinh(x) / x,
    )


def _rodrigues_coefficients(
    s_squared: torch.Tensor, ε_stab: float = 0.0001
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute `F_1(s^2) = sin(s) / s`, `F_2(s^2) = (1 - cos(s)) / s^2`, and
    `F_3(s^2) = (s - sin(s)) / s^3` if `s^2 > 0`, and their hyperbolic analogues
    if `s^2 < 0`. If `A^3 = -s^2 A`, then `exp(A) = I + F_1 A + F_2 A^2` and
    `int_0^1 exp(τ A) dτ = I + F_2 A + F_3 A^2`. Since the `F_i` are analytic in
    `s^2`, we use their Taylor expansions if `|s^2| < ε_stab`, or `|s^2| < 1`
    for `F_3`, which suffers more from cancellation.
    """
    small = s_squared.abs() < ε_stab
    timelike = s_squared > 0.0
    θ = torch.where(small, 1.0, s_squared.abs()).sqrt()

    F_1 = torch.where(
        small,
        1 - s_squared / 6 + s_squared**2 / 120 - s_squared**3 / 5040,
        torch.where(timelike, _sinc(θ), _sinhc(θ)),
    )
    F_2 = torch.where(
        small,
        1 / 2 - s_squared / 24 + s_squared**2 / 720 - s_squared**3 / 40320,
        torch.where(timelike, _sinc(θ / 2) ** 2, _sinhc(θ / 2) ** 2) / 2,
    )
    small_3 = s_squared.abs() < 1.0
    θ_3 = torch.where(small_3, 1.0, s_squared.abs()).sqrt()
    F_3 = torch.where(
        small_3,
        1 / 6
        - s_squared / 120
        + s_squared**2 / 5040
        - s_squared**3 / 362880
        + s_squared**4 / 39916800
        - s_squared**5 / 6227020800,
        torch.where(timelike, θ_3 - torch.sin(θ_3), torch.sinh(θ_3) - θ_3) / θ_3**3,
    )
    return F_1, F_2, F_3


def _exp_affine(
    ω: torch.Tensor, v: torch.Tensor, s_squared: torch.Tensor, ε_stab: float = 0.0001
) -> torch.Tensor:
    """
    Compute `exp([[ω, v], [0, 0]]) = [[exp(ω), V v], [0, 1]]` for `ω` with
    `ω^3 = -s^2 ω`, where `exp(ω) = I + F_1 ω + F_2 ω^2` and
    `V = I + F_2 ω + F_3 ω^2`, with `F_i` as in `_rodrigues_coefficients`.
    """
    F_1, F_2, F_3 = _rodrigues_coefficients(s_squared, ε_stab)
    I = torch.eye(3, device=ω.device, dtype=ω.dtype)
    ω_squared = ω @ ω
    g = torch.zeros(*ω.shape[:-2], 4, 4, device=ω.device, dtype=ω.dtype)
    g[..., :3, :3] = I + F_1 * ω + F_2 * ω_squared
    g[..., :3, 3:] = (I + F_2 * ω + F_3 * ω_squared) @ v
    g[..., 3, 3] = 1.0
    return g


def _sigmoid(x: torch.Tensor, scale=88.0) -> torch.Tensor:
    return scale * torch.tanh(x / scale)
