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
  homogeneous space and the computation of the components of generators in
  the Lie algebra connecting pairs of points in the homogeneous space.
Additionally, `PowerSpace(G, m)` is the power G^m of any of these spaces, by
default with unordered factors.
Also provides example implementations of
  1. `Rn(n)` <: `Group`: n-dimensional translation group R^n.
  2. `SE2()` <: `Group`: special Euclidean group of roto-translations on
  R^2.
  3. `TSn(n)` <: `Group`: translation-scaling group on R^n.
  4. `Aff2()` <: `Group`: group of affine transformations on R^2.
  5. `SO3()` <: `MatrixGroup`: special orthogonal group of rotations on R^3.
  6. `SE3()` <: `MatrixGroup`: special Euclidean group of roto-translations on
  R^3.
  7. `SOplus12()` <: `MatrixGroup`: proper orthochronous special orthogonal group
  of Lorentz transformations on R^3.
  8. `SEplus12()` <: `MatrixGroup`: proper orthochronous special Euclidean group
  of Poincare transformations on R^3.
  9. `M3()` <: `HomogeneousSpace`: position-orientation space R^3 x S^2, which
  is an SE(3)-homogeneous space.
  10. `P3()` <: `HomogeneousSpace`: event-velocity space R^3 x H^2, which
  is an SE^+(1, 2)-homogeneous space.
"""

import math
from abc import ABC, abstractmethod
from typing import Literal

import torch
from scipy.optimize import linear_sum_assignment


class Space(ABC):
    """
    Common base of `Group`, `MatrixGroup`, `HomogeneousSpace`, `ProductSpace`,
    and `PowerSpace`, which describes a space in terms of the exponential
    curves connecting its points, as used by the flow matching models and the
    metrics: the components of their generators, and following them. Provides the logarithmic distance
    `ρ(x_1, x_2) = ||a||_{x_1}`, with `a` the components of the generator of
    the exponential curve from `x_1` to `x_2`, and `||.||_x` the norm of
    `generator_norm_squared` at `x`.

    Subclasses provide `point_shape`, the shape of a single point, and
    `generator_dim`, the number of components of a generator. `weight_dim`, the
    number of weights of `generator_norm_squared`, defaults to `generator_dim`.
    """

    @property
    def point_size(self) -> int:
        """Number of entries of a single point."""
        return math.prod(self.point_shape)

    @property
    def weight_dim(self) -> int:
        return self.generator_dim

    @abstractmethod
    def generator_components(self, x_1: torch.Tensor, x_2: torch.Tensor) -> torch.Tensor:
        """
        Components of the generator of the exponential curve from `x_1` to
        `x_2`, broadcasting over their leading dimensions.
        """
        raise NotImplementedError

    @abstractmethod
    def flow(self, x: torch.Tensor, a: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Follow the exponential curve from `x` with generator components `a` for
        a time `t`, which broadcasts with `a`.
        """
        raise NotImplementedError

    def generator_norm_squared(
        self, x: torch.Tensor, a: torch.Tensor, w: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        Squared norm `||a||_x^2` of the generator components `a` at the point
        `x`, weighted by `w`, broadcasting over their leading dimensions. It
        must be invariant, `||Ad_g a||_{g x} = ||a||_x`. By default, this is
        `||w * a||^2`, which is invariant for groups, whose generators
        `log(g_1^-1 g_2)` are already in the frame of `g_1`.
        """
        return ((a if w is None else w * a) ** 2).sum(-1)

    @torch.no_grad()
    def distance_matrix(
        self,
        x_1: torch.Tensor,
        x_2: torch.Tensor,
        w: torch.Tensor | None = None,
        chunk_size: int = 128,
    ) -> torch.Tensor:
        """
        Compute the matrix of logarithmic distances `D[i, j] = ρ(x_1[i], x_2[j])`.

        Args:
            `x_1`: samples of shape [N, ...].
            `x_2`: samples of shape [M, ...].
          Optional:
            `w`: weights of the generator, as in `generator_norm_squared`.
            Defaults to all ones.
            `chunk_size`: number of rows of `D` computed at once, to limit memory
            usage. Defaults to 128.
        """
        D = torch.empty(len(x_1), len(x_2), device=x_2.device, dtype=x_2.dtype)
        for i in range(0, len(x_1), chunk_size):
            a = self.generator_components(x_1[i:i + chunk_size, None], x_2[None])
            D[i:i + chunk_size] = self.generator_norm_squared(x_1[i:i + chunk_size, None], a, w).clamp_min(0.0).sqrt()
        return D

    @torch.no_grad()
    def cost_matrices(
        self, x_1: torch.Tensor, x_2: torch.Tensor, w: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        Compute the squared logarithmic distances between all pairs within
        blocks, `C[k, i, j] = ρ(x_1[k, i], x_2[k, j])^2`, as the cost of a
        mini-batch optimal transport coupling.

        Args:
            `x_1`: samples of shape [B, b, ...].
            `x_2`: samples of shape [B, b, ...].
          Optional:
            `w`: weights of the generator, as in `generator_norm_squared`.
            Defaults to all ones.
        """
        return torch.stack(
            [self.distance_matrix(a, b, w, chunk_size=len(a)) ** 2 for a, b in zip(x_1, x_2)]
        )


class ProductSpace(Space):
    """
    Direct product of spaces of type `Group`, `MatrixGroup`, `HomogeneousSpace`,
    or `ProductSpace`, e.g. `ProductSpace(SE2(), Rn(2))` for SE(2) x R^2. A
    point is the concatenation of the flattened points of the factors, of shape
    [..., point_size]; `pack` and `unpack` convert from and to the points of the
    factors. All operations act on every factor: e.g. the flow is the product of
    the flows of the factors, and the squared norm is the sum of the squared
    norms of the factors. The weights are the concatenation of the weights of
    the factors.

    Args:
        `*spaces`: instances of the spaces of the factors.
    """

    def __init__(self, *spaces: Space):
        super().__init__()
        if any(isinstance(S, PowerSpace) for S in spaces):
            raise ValueError("The factors cannot be a PowerSpace!")
        self.spaces = spaces
        self.point_slices = _slices([S.point_size for S in spaces])
        self.generator_slices = _slices([S.generator_dim for S in spaces])
        self.weight_slices = _slices([S.weight_dim for S in spaces])

    @property
    def point_shape(self) -> tuple[int, ...]:
        return (sum(S.point_size for S in self.spaces),)

    @property
    def generator_dim(self) -> int:
        return sum(S.generator_dim for S in self.spaces)

    @property
    def weight_dim(self) -> int:
        return sum(S.weight_dim for S in self.spaces)

    def pack(self, *xs: torch.Tensor) -> torch.Tensor:
        """
        Point of the product with the points `xs` of the factors.
        """
        return torch.cat(
            [x.flatten(start_dim=x.ndim - len(S.point_shape)) for S, x in zip(self.spaces, xs)], dim=-1
        )

    def unpack(self, x: torch.Tensor) -> list[torch.Tensor]:
        """
        Points of the factors of the point `x` of the product.
        """
        return [x[..., s].reshape(*x.shape[:-1], *S.point_shape) for S, s in zip(self.spaces, self.point_slices)]

    def _weights(self, w: torch.Tensor | None) -> list[torch.Tensor | None]:
        """Weights of the factors."""
        return [None if w is None else w[s] for s in self.weight_slices]

    def generator_components(self, x_1, x_2):
        """
        Components of the generators of the exponential curves from `x_1` to
        `x_2`, factorwise.
        """
        return torch.cat(
            [S.generator_components(y_1, y_2) for S, y_1, y_2 in zip(self.spaces, self.unpack(x_1), self.unpack(x_2))],
            dim=-1,
        )

    def flow(self, x, a, t):
        """
        Follow the exponential curves from `x` with generator components `a` for
        a time `t`, factorwise.
        """
        return self.pack(
            *(S.flow(y, a[..., s], t) for S, y, s in zip(self.spaces, self.unpack(x), self.generator_slices))
        )

    def generator_norm_squared(self, x, a, w=None):
        """
        Squared norm of the generator components `a` at `x`, the sum of the
        squared norms of the factors.
        """
        return sum(
            S.generator_norm_squared(y, a[..., s], w_S)
            for S, y, s, w_S in zip(self.spaces, self.unpack(x), self.generator_slices, self._weights(w))
        )

    @torch.no_grad()
    def cost_matrices(self, x_1, x_2, w=None):
        """
        Compute the squared product distances between all pairs within blocks,
        the sum over the factors of their `cost_matrices`, so that their closed
        forms are used.
        """
        return sum(
            S.cost_matrices(y_1, y_2, w_S)
            for S, y_1, y_2, w_S in zip(self.spaces, self.unpack(x_1), self.unpack(x_2), self._weights(w))
        )

    def __repr__(self):
        return " x ".join(str(S) for S in self.spaces)


class PowerSpace(Space):
    """
    Power G^m of a space `G` of type `Group`, `MatrixGroup`, or
    `HomogeneousSpace`, whose elements are `m` points of `G`, of shape
    [..., m, ...], with the trailing dimensions those of a point of `G`; all
    operations act on every factor.

    Args:
        `G`: instance of the space of the factors.
        `m`: number of factors.
      Optional:
        `permutation_invariant`: whether the factors are unordered, so that the
          distance between two elements is minimised over all permutations of
          the factors. Defaults to True.
    """

    def __init__(
        self, G: Space, m: int, permutation_invariant: bool = True
    ):
        super().__init__()
        self.G = G
        self.m = m
        self.permutation_invariant = permutation_invariant

    @property
    def point_shape(self) -> tuple[int, ...]:
        return (self.m, *self.G.point_shape)

    @property
    def generator_dim(self) -> int:
        return self.m * self.G.generator_dim

    @property
    def weight_dim(self) -> int:
        return self.G.weight_dim

    def generator_components(self, x_1, x_2):
        """
        Components of the generators of the exponential curves from `x_1` to
        `x_2`, factorwise.
        """
        return self.G.generator_components(x_1, x_2)

    def flow(self, x, a, t):
        """
        Follow the exponential curves from `x` with generator components `a` for
        a time `t`, factorwise.
        """
        return self.G.flow(x, a, t)

    def generator_norm_squared(self, x, a, w=None):
        """
        Squared norm of the generator components `a` at `x`, the sum of the
        squared norms of the factors.
        """
        return self.G.generator_norm_squared(x, a, w).sum(-1)

    @torch.no_grad()
    def distance_matrix(
        self,
        x_1: torch.Tensor,
        x_2: torch.Tensor,
        w: torch.Tensor | None = None,
        chunk_size: int = 128,
    ) -> torch.Tensor:
        """
        Compute the matrix of logarithmic distances `D[i, j] = ρ(x_1[i], x_2[j])`,
        with the product distance `ρ^2 = sum_k ρ_G(x_1[i, k], x_2[j, k])^2`. If
        the space is `permutation_invariant`, it is minimised over all
        permutations of the factors of `x_2[j]`.
        """
        if not self.permutation_invariant:
            return super().distance_matrix(x_1, x_2, w, chunk_size)
        D_squared = []
        for i in range(0, len(x_1), chunk_size):
            x_1_chunk = x_1[i:i + chunk_size, None, :, None]
            A = self.generator_components(x_1_chunk, x_2[None, :, None])
            C = self.G.generator_norm_squared(x_1_chunk, A, w).cpu().numpy()
            for c in C.reshape(-1, *C.shape[2:]):
                rows, cols = linear_sum_assignment(c)
                D_squared.append(c[rows, cols].sum())
        return torch.tensor(D_squared, device=x_2.device, dtype=x_2.dtype).view(len(x_1), len(x_2)).sqrt()

    @torch.no_grad()
    def cost_matrices(self, x_1, x_2, w=None):
        """
        Compute the squared product distances between all pairs within blocks,
        `C[k, i, j] = ρ(x_1[k, i], x_2[k, j])^2`, not minimised over
        permutations, as the sum over the factors of `G.cost_matrices`.
        """
        B, b = x_1.shape[:2]
        # Treat every factor of every block as a block of `G`.
        C = self.G.cost_matrices(x_1.movedim(2, 1).flatten(0, 1), x_2.movedim(2, 1).flatten(0, 1), w)
        return C.view(B, self.m, b, b).sum(1)

    def __repr__(self):
        return f"({self.G})^{self.m}"


class Group(Space):
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

    @property
    def point_shape(self) -> tuple[int, ...]:
        return (self.dim,)

    @property
    def generator_dim(self) -> int:
        return self.dim

    @abstractmethod
    def L(self, g_1: torch.Tensor, g_2: torch.Tensor) -> torch.Tensor:
        """
        Left multiplication of `g_2` by `g_1`, i.e. `g_1 g_2`.
        """
        raise NotImplementedError

    @abstractmethod
    def L_inv(self, g_1: torch.Tensor, g_2: torch.Tensor) -> torch.Tensor:
        """
        Left multiplication of `g_2` by `g_1^-1`, i.e. `g_1^-1 g_2`.
        """
        raise NotImplementedError

    @abstractmethod
    def log(self, g: torch.Tensor) -> torch.Tensor:
        """
        Lie group logarithm of `g`, i.e. the components of `A` in the Lie
        algebra such that `exp(A) = g`.
        """
        raise NotImplementedError

    @abstractmethod
    def exp(self, A: torch.Tensor) -> torch.Tensor:
        """
        Lie group exponential of the Lie algebra element with components `A`.
        """
        raise NotImplementedError

    def generator_components(self, g_1: torch.Tensor, g_2: torch.Tensor) -> torch.Tensor:
        """
        Components of the generator of the exponential curve from `g_1` to
        `g_2`, i.e. `log(g_1^-1 g_2)`.
        """
        return self.log(self.L_inv(g_1, g_2))

    def flow(self, g: torch.Tensor, A: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Follow the exponential curve from `g` with generator `A` for a time
        `t`, i.e. `g exp(t A)`.
        """
        return self.L(g, self.exp(t * A))


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
        within blocks, `C[k, i, j] = ||w * (g_2[k, j] - g_1[k, i])||^2`.
        """
        if w is not None:
            g_1, g_2 = w * g_1, w * g_2
        return torch.cdist(g_1, g_2, compute_mode="donot_use_mm_for_euclid_dist") ** 2

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

    def __repr__(self):
        return "SE(2)"


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
        return f"TS({self.dim - 1})"


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


class MatrixGroup(Space):
    """
    Class encapsulating basic Lie group and Lie algebra properties for groups
    that can be efficiently represented with matrices.

    Group multiplication, multiplication by inverse, and exponential make use of
    corresponding PyTorch methods. Since PyTorch does not implement a matrix
    logarithm, this must be provided.
    """

    def __init__(self, dim: int, lie_algebra_basis):
        super().__init__()
        self.dim = dim
        self.lie_algebra_basis = lie_algebra_basis

    @property
    def point_shape(self) -> tuple[int, ...]:
        return tuple(self.lie_algebra_basis.shape[-2:])

    @property
    def generator_dim(self) -> int:
        return self.dim

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

    def lie_algebra_element(self, a: torch.Tensor) -> torch.Tensor:
        """
        Compute the Lie algebra element with components `a` with respect to the
        basis given by `self.lie_algebra_basis`.
        """
        return (a[..., None, None] * self.lie_algebra_basis).sum(-3)

    def generator_components(self, R_1: torch.Tensor, R_2: torch.Tensor) -> torch.Tensor:
        """
        Components of the generator of the exponential curve from `R_1` to
        `R_2`, i.e. of `log(R_1^-1 R_2)`.
        """
        return self.lie_algebra_components(self.log(self.L_inv(R_1, R_2)))

    def flow(self, R: torch.Tensor, a: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Follow the exponential curve from `R` with generator components `a` for
        a time `t`, i.e. `R exp(t A)` with `A` the Lie algebra element with
        components `a`.
        """
        return self.L(R, self.exp(self.lie_algebra_element(t * a)))


class SO3(MatrixGroup):
    """
    Special orthogonal group of rotations on R^3.
    """

    def __init__(self, device: str | torch.DeviceObjType = "cpu", dtype: torch.dtype = torch.float32):
        super().__init__(
            dim=3,
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
                ], device=device, dtype=dtype
            ),
        )

    def L_inv(self, R_1, R_2):
        """
        Left multiplication of `R_2` by `R_1^-1 = R_1^T`.
        """
        return R_1.mT @ R_2

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
        return self.lie_algebra_element(a)

    def exp(self, A):
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
        traces = _gram(R_1.flatten(-2), R_2.flatten(-2))
        C = torch.arccos(((traces - 1) / 2).clamp(-1, 1)) ** 2
        return C if w is None else w[0] ** 2 * C

    def __repr__(self):
        return "SO(3)"


class SE3(MatrixGroup):
    """
    Special euclidean group of roto-translations on R^3.
    """

    def __init__(self, device: str | torch.DeviceObjType = "cpu", dtype: torch.dtype = torch.float32):
        super().__init__(
            dim=6,
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
                ], device=device, dtype=dtype
            ),
        )
        self.so3 = SO3(device=device, dtype=dtype)

    def log(self, g):
        """
        Lie group logarithm of `g`, i.e. `A` in Lie algebra such that
        `exp(A) = g`, with the rotation part from `SO3.log`.
        """
        t, R = self.get_translation_rotation(g)
        ω = self.so3.log(R)
        θ_squared = (self.so3.lie_algebra_components(ω) ** 2).sum(-1)[..., None, None]
        return _log_affine(ω, t[..., None], θ_squared)

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

    def __init__(self, device: str | torch.DeviceObjType = "cpu", dtype: torch.dtype = torch.float32):
        super().__init__(
            dim=3,
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
                ], device=device, dtype=dtype
            ),
        )
        self.η = torch.tensor(
            [
                [1.0, 0.0, 0.0],
                [0.0, -1.0, 0.0],
                [0.0, 0.0, -1.0],
            ], device=device, dtype=dtype
        )

    def L_inv(self, Λ_1, Λ_2):
        """
        Left multiplication of `Λ_2` by `Λ_1^-1 = η Λ_1^T η`.
        """
        return self.η @ Λ_1.mT @ self.η @ Λ_2

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
        η = self.η
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
        return self.lie_algebra_element(a)

    def exp(self, A, ε_stab=0.0001):
        """
        Lie group exponential of `A`.

        While PyTorch has a built in matrix exp, it can still be more
        efficient and accurate to use a custom formula. For SO^+(1, 2), 
        we can use a generalisation of Rodrigues' rotation formula.
        """
        A_vec = self.lie_algebra_components(A)
        s_squared = (A_vec @ self.η * A_vec).sum(-1)[..., None, None]
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

    def __init__(self, device: str | torch.DeviceObjType = "cpu", dtype: torch.dtype = torch.float32):
        super().__init__(
            dim=6,
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
                ], device=device, dtype=dtype
            ),
        )
        self.soplus12 = SOplus12(device=device, dtype=dtype)
        self.η = self.soplus12.η

    def log(self, g):
        """
        Lie group logarithm of `g`, i.e. `A` in Lie algebra such that
        `exp(A) = g`, with the Lorentz part from `SOplus12.log`.
        """
        t, Λ = self.get_translation_lorentz(g)
        ω = self.soplus12.log(Λ)
        k = self.soplus12.lie_algebra_components(ω)
        s_squared = (k @ self.η * k).sum(-1)[..., None, None]
        return _log_affine(ω, t[..., None], s_squared)

    def exp(self, A, ε_stab=0.0001):
        """
        Lie group exponential of `A`.

        While PyTorch has a built in matrix exp, it can still be more
        efficient and accurate to use a custom formula. For SE^+(1, 2),
        we can use a generalisation of Rodrigues' rotation formula.
        """
        ω, v = A[..., :3, :3], A[..., :3, 3:]
        k = self.soplus12.lie_algebra_components(ω)
        s_squared = (k @ self.η * k).sum(-1)[..., None, None]
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


class HomogeneousSpace(Space):
    """
    Class encapsulating basic properties of a homogeneous space of a
    `MatrixGroup`.
    """

    def __init__(self, G: MatrixGroup, point_shape: tuple[int, ...]):
        super().__init__()
        self.G = G
        self._point_shape = point_shape

    @property
    def point_shape(self) -> tuple[int, ...]:
        return self._point_shape

    @property
    def generator_dim(self) -> int:
        return self.G.dim

    def act(self, g: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        """
        Action of `g` on `p`, i.e. `g p`.
        """
        return g @ p

    def get_generator(self, p_1: torch.Tensor, p_2: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Generator of the exponential curve from `p_1` to `p_2`, i.e. the
        element of the Lie algebra of `G` with components
        `generator_components(p_1, p_2, **kwargs)`.
        """
        return self.G.lie_algebra_element(self.generator_components(p_1, p_2, **kwargs))

    def flow(self, p: torch.Tensor, a: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Follow the exponential curve from `p` with generator components `a` for
        a time `t`, i.e. `exp(t A) p` with `A` the Lie algebra element with
        components `a`.
        """
        return self.act(self.G.exp(self.G.lie_algebra_element(t * a)), p)


class M3(HomogeneousSpace):
    """
    Position-Orientation space on 3D Euclidean space.
    """

    def __init__(self, generator: Literal["mav", "pure_rotation"] | float = "mav", device: str | torch.DeviceObjType = "cpu", dtype: torch.dtype = torch.float32):
        self.se3 = SE3(device=device, dtype=dtype)
        super().__init__(G=self.se3, point_shape=(4, 2))
        self.generator = generator

    @property
    def weight_dim(self) -> int:
        return 4

    def generator_components(
        self,
        p_1,
        p_2,
        ε_stab=0.001,
        generator: Literal["mav", "pure_rotation"] | float | None = None,
    ):
        """
        Compute the components `(v, ω)` of a generator between `p_1` and `p_2`
        [1, Prop. 1], with respect to the Lie algebra basis of SE(3).
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

        k0 = (cross_n / sinθ).nan_to_num() * ~parallel
        # Near θ = π, k0 is ill-conditioned, and at θ = π any axis orthogonal
        # to n_1 works. The part of x_diff orthogonal to n_1 gives an
        # equivariant choice; if it vanishes, no equivariant choice exists, and
        # a uniformly random axis is equivariant in distribution.
        antiparallel = θ > torch.pi - ε_stab
        if antiparallel.any():
            x_diff_perp = x_diff - (n_1 * x_diff).sum(-1, keepdim=True) * n_1
            k_random = torch.randn(*shape, 3, device=device, dtype=p_2.dtype)
            k_random = k_random - (n_1 * k_random).sum(-1, keepdim=True) * n_1
            k_anti = torch.where(x_diff_perp.norm(dim=-1, keepdim=True) < ε_stab, k_random, x_diff_perp)
            k0 = torch.where(antiparallel, k_anti / k_anti.norm(dim=-1, keepdim=True), k0)
        use_mav = x_diff.norm(dim=-1, keepdim=True) < ε_stab
        x_perp = (k0 * x_diff).sum(-1, keepdim=True) * k0
        x_par = x_diff - x_perp

        c_mav = (x_m + 0.5 * _cotan(θ / 2.0) * cross_product(k0, x_par)).nan_to_num()
        v_mav = x_perp

        ω_vec_mav = θ * k0

        match generator:
            case "mav":
                return torch.cat(
                    (
                        x_diff * parallel
                        + (-cross_product(ω_vec_mav, c_mav) + v_mav) * ~parallel,
                        ω_vec_mav * ~parallel,
                    ),
                    dim=-1,
                )
            case "pure_rotation":
                k = cross_product(x_diff, n_2 - n_1)
                k = (k / k.norm(dim=-1, keepdim=True)).nan_to_num()
            case float() | int():
                khalfπ = (n_1 + n_2) / (n_1 + n_2).norm(
                    dim=-1, keepdim=True
                ).nan_to_num()
                k = math.cos(generator) * k0 + math.sin(generator) * khalfπ
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

        return torch.cat(
            (
                x_diff * parallel
                + (
                    (-cross_product(ω_vec, c) + v) * ~use_mav
                    + (-cross_product(ω_vec_mav, c_mav) + v_mav) * use_mav
                )
                * ~parallel,
                (ω_vec * ~use_mav + ω_vec_mav * use_mav) * ~parallel,
            ),
            dim=-1,
        )

    def generator_norm_squared(self, p, a, w=None):
        """
        Squared norm `||a||_p^2` at `p = (x, n)` of the generator components
        `a = (v, ω)`, which is invariant under roto-translations. It is the
        norm of the velocity `u = v + ω x x` of the position `x`, and of the
        angular velocity `ω`, both split along and across `n`:
          `||a||_p^2 = w_main^2 (u . n)^2 + w_lateral^2 |u_⊥|^2
                       + w_spin^2 (ω . n)^2 + w_tilt^2 |ω_⊥|^2`,
        with `w = (w_main, w_lateral, w_spin, w_tilt)`, defaulting to all ones.
        """
        x, n = self.get_position_orientation(p)
        v, ω = a[..., :3], a[..., 3:]
        u = v + cross_product(ω, x)
        u_n, ω_n = (u * n).sum(-1), (ω * n).sum(-1)
        w_main, w_lateral, w_spin, w_tilt = (1.0, 1.0, 1.0, 1.0) if w is None else w
        return (
            w_main**2 * u_n**2
            + w_lateral**2 * ((u**2).sum(-1) - u_n**2)
            + w_spin**2 * ω_n**2
            + w_tilt**2 * ((ω**2).sum(-1) - ω_n**2)
        )

    @torch.no_grad()
    def cost_matrices(self, p_1, p_2, w=None, ε_stab=0.001):
        """
        Compute the squared logarithmic distances between all pairs within
        blocks, `C[k, i, j] = ρ(p_1[k, i], p_2[k, j])^2`.
        """
        if self.generator not in ("mav", "pure_rotation"):
            return super().cost_matrices(p_1, p_2, w)
        w_main, w_lateral, w_spin, w_tilt = (1.0, 1.0, 1.0, 1.0) if w is None else w
        x_1, n_1 = self.get_position_orientation(p_1)
        x_2, n_2 = self.get_position_orientation(p_2)
        # The cost is translation invariant, and centring the blocks reduces cancellation in the inner products.
        centre = (x_1.mean(-2, keepdim=True) + x_2.mean(-2, keepdim=True)) / 2.0
        x_1, x_2 = x_1 - centre, x_2 - centre

        cosα = _gram(n_1, n_2).clamp(-1.0, 1.0)
        α = cosα.arccos()
        n_1_Δx = _gram(n_1, x_2) - _dot(n_1, x_1)
        n_2_Δx = _dot(n_2, x_2).mT - _gram(x_1, n_2)
        Δx_squared = _dot(x_1, x_1) + _dot(x_2, x_2).mT - 2.0 * _gram(x_1, x_2)
        T = _gram(n_1, cross_product(n_2, x_2)) - _gram(cross_product(x_1, n_1), n_2)

        def cost(u_n, u_squared, ω_n_squared, ω_squared):
            return (
                w_main**2 * u_n**2
                + w_lateral**2 * (u_squared - u_n**2)
                + w_spin**2 * ω_n_squared
                + w_tilt**2 * (ω_squared - ω_n_squared)
            )

        # For nearly parallel pairs, `generator_components` translates.
        C = torch.where(α < ε_stab, cost(n_1_Δx, Δx_squared, 0.0, 0.0), 0.0)

        # `k` is ill-conditioned for (anti)parallel pairs, so clamp `sin(α)`.
        sinα = α.sin().clamp_min(math.sin(ε_stab))
        antiparallel = α > math.pi - ε_stab
        k_Δx_squared = torch.where(antiparallel, Δx_squared - n_1_Δx**2, (T / sinα) ** 2)
        u_n = (α / 2.0).cos() / _sinc(α / 2.0) * n_1_Δx - ~antiparallel * (α / 2.0) * (cosα * n_1_Δx - n_2_Δx) / sinα
        u_squared = k_Δx_squared + (Δx_squared - k_Δx_squared) / _sinc(α / 2.0) ** 2
        C_mav = cost(u_n, u_squared, 0.0, α**2)
        if self.generator == "mav":
            return torch.where(α < ε_stab, C, C_mav).clamp_min(0.0)

        # For `pure_rotation`, `generator_components` uses `mav` if `Δx` is small. If
        # `Δx` is parallel to `d`, the axis is not unique, and the cost is not
        # meaningful.
        d_Δx = n_2_Δx - n_1_Δx
        q = (2.0 * (1.0 - cosα) * Δx_squared - d_Δx**2).clamp_min(0.0).sqrt().clamp_min(ε_stab**2)
        k_n_1 = -T / q
        θ = torch.atan2((1.0 - cosα) * (n_1_Δx + n_2_Δx) / q, cosα - k_n_1**2)
        u_n = (θ / 2.0).cos() / _sinc(θ / 2.0) * n_1_Δx - (θ / 2.0) * ((cosα - 1.0) * Δx_squared - n_1_Δx * d_Δx) / q
        C_pure = cost(u_n, Δx_squared / _sinc(θ / 2.0) ** 2, (θ * k_n_1) ** 2, θ**2)
        C_rot = torch.where(Δx_squared < ε_stab**2, C_mav, C_pure)
        return torch.where(α < ε_stab, C, C_rot).clamp_min(0.0)

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
        dtype: torch.dtype = torch.float32,
    ):
        self.seplus12 = SEplus12(device=device, dtype=dtype)
        self.η = self.seplus12.η
        super().__init__(G=self.seplus12, point_shape=(4, 2))
        self.generator = generator

    @property
    def weight_dim(self) -> int:
        return 4

    def generator_components(
        self,
        p_1,
        p_2,
        generator: Literal["timelike", "lightlike", "spacelike"] | float | None = None,
    ):
        """
        Compute the components `(v, k)` of a generator between `p_1` and `p_2`,
        with respect to the Lie algebra basis of SE^+(1, 2).

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
        x_1, n_1 = self.get_event_velocity(p_1)
        x_2, n_2 = self.get_event_velocity(p_2)
        x_m = (x_1 + x_2) / 2.0
        x_diff = x_2 - x_1

        if generator != "spacelike":
            # The unit axes, which the spacelike generator avoids since `ks` is
            # undefined for `n_1 = n_2`.
            ks = cross_product(n_1, n_2) * self.η.diagonal()
            ks = ks / (-(ks @ self.η * ks).sum(-1, keepdim=True)).sqrt()
            kt = n_1 + n_2
            kt = kt / (kt @ self.η * kt).sum(-1, keepdim=True).sqrt()

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
                n_cross = cross_product(n_1, n_2) * self.η.diagonal()
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

                v = x_diff - cross_product(k, x_m) * self.η.diagonal() + xcoth_minus_1 * x_par
                return torch.cat((v, k), dim=-1)
            case "timelike":
                k = torch.pi * kt

                x_perp = (kt @ self.η * x_diff).sum(-1, keepdim=True) * kt
                x_par = x_diff - x_perp

                c = x_m
                v = x_perp
            case "lightlike":
                i = (n_1 @ self.η * n_2).sum(-1, keepdim=True)
                s = 2 * ((i - 1.0) / (i + 1.0)).sqrt()
                k = s * (ks + kt)
                ω = self.seplus12.soplus12.lie_algebra_element(k)

                t = x_2[..., None] - self.seplus12.soplus12.exp(ω) @ x_1[..., None]
                c = torch.zeros_like(x_diff)
                v = (t - 1.0 / 2.0 * ω @ t + 1.0 / 12.0 * ω @ ω @ t)[..., 0]
            case float() | int():
                kφ = math.cosh(generator) * ks - math.sinh(generator) * kt

                n_1p = n_1 + (kφ @ self.η * n_1).sum(-1, keepdim=True) * kφ
                n_2p = n_2 + (kφ @ self.η * n_2).sum(-1, keepdim=True) * kφ
                s = torch.atanh(
                    (kφ @ self.η * cross_product(n_1p, n_2p) * self.η.diagonal()).sum(
                        -1, keepdim=True
                    )
                    / -(n_1p @ self.η * n_2p).sum(-1, keepdim=True)
                )
                k = s * kφ

                x_perp = -1.0 * (kφ @ self.η * x_diff).sum(-1, keepdim=True) * kφ
                x_par = x_diff - x_perp

                c = (
                    x_m - 0.5 * _cotanh(s / 2.0) * cross_product(kφ, x_par) * self.η.diagonal()
                ).nan_to_num()
                v = x_perp
            case _:
                raise ValueError(f"{generator} is not a supported type of generator!")

        return torch.cat((-cross_product(k, c) * self.η.diagonal() + v, k), dim=-1)

    def generator_norm_squared(self, p, a, w=None):
        """
        Squared norm `||a||_p^2` at `p = (x, n)` of the generator components
        `a = (v, k)`, which is invariant under Poincaré transformations. With
        the Minkowski inner product `<a, b> = a^T η b`, it is the norm of the
        velocity `u = v + k x_η x` of the event `x`, and of the Lorentz part
        `k`, both split along and across `n`, using that `<n, n> = 1` and that
        `-<., .>` is positive definite on the orthogonal complement of `n`:
          `||a||_p^2 = w_time^2 <u, n>^2 + w_space^2 (<u, n>^2 - <u, u>)
                       + w_rotation^2 <k, n>^2 + w_boost^2 (<k, n>^2 - <k, k>)`,
        with `w = (w_time, w_space, w_rotation, w_boost)`, defaulting to all ones.
        """
        x, n = self.get_event_velocity(p)
        v, k = a[..., :3], a[..., 3:]
        η = self.η.diagonal()
        u = v + cross_product(k, x) * η
        u_n, k_n = (u * η * n).sum(-1), (k * η * n).sum(-1)
        w_time, w_space, w_rotation, w_boost = (1.0, 1.0, 1.0, 1.0) if w is None else w
        return (
            w_time**2 * u_n**2
            + w_space**2 * (u_n**2 - (u * η * u).sum(-1))
            + w_rotation**2 * k_n**2
            + w_boost**2 * (k_n**2 - (k * η * k).sum(-1))
        )

    @torch.no_grad()
    def cost_matrices(self, p_1, p_2, w=None, ε_stab=1e-2):
        """
        Compute the squared logarithmic distances between all pairs within
        blocks, `C[k, i, j] = ρ(p_1[k, i], p_2[k, j])^2`.

        For the `spacelike` generator, this only needs inner products of the
        samples, i.e. matrix multiplications. Write `<a, b> = a^T η b`,
        `u = n_1 x n_2` for the Euclidean cross product, so that the hyperbolic
        cross product is `n_1 x_η n_2 = η u`, `s^2 = -<u, u> = <n_1, n_2>^2 - 1`,
        `w = asinh(s)`, `α = w / s`, `β = w / 2 coth(w / 2) - 1`, `γ = β / s^2`,
        `Δx = x_2 - x_1`, and `p = u . Δx`. Then the generator has Lorentz part
        `k = α η u`, with `<k, n_1> = 0` and `<k, k> = -α^2 s^2`, and the velocity
        of `x_1` is `(1 + β) Δx - α / 2 u x η Δx + γ p η u`. Since
        `η = diag(1, -1, -1)` is a rotation, `η a . η b = a . b` and
        `η (a x b) = η a x η b`, so its inner products reduce to inner products
        of the samples by `(a x b) . (c x d) = (a . c) (b . d) - (a . d) (b . c)`:
        `<u x η Δx, u x η Δx> = -s^2 <Δx, Δx> - p^2`,
        `<u x η Δx, n_1> = <n_1, n_2> <n_1, Δx> - <n_2, Δx>`, `<Δx, η u> = p`,
        and the others vanish. Expanding the squares loses relative precision
        for very close pairs; it is meant for couplings, use `distance_matrix`
        for metrics.
        """
        if self.generator != "spacelike":
            return super().cost_matrices(p_1, p_2, w)
        w_time, w_space, w_rotation, w_boost = (1.0, 1.0, 1.0, 1.0) if w is None else w
        x_1, n_1 = self.get_event_velocity(p_1)
        x_2, n_2 = self.get_event_velocity(p_2)
        # The cost is translation invariant, and centring the blocks reduces cancellation in the inner products.
        centre = (x_1.mean(-2, keepdim=True) + x_2.mean(-2, keepdim=True)) / 2.0
        x_1, x_2 = x_1 - centre, x_2 - centre
        η = self.η.diagonal()

        n_1_n_1, n_2_n_2 = _dot(n_1, η * n_1), _dot(n_2, η * n_2).mT
        n_1_n_2 = _gram(n_1, η * n_2)
        n_1_Δx = _gram(n_1, η * x_2) - _dot(n_1, η * x_1)
        n_2_Δx = _dot(n_2, η * x_2).mT - _gram(η * x_1, n_2)
        Δx_Δx = _dot(x_1, η * x_1) + _dot(x_2, η * x_2).mT - 2.0 * _gram(x_1, η * x_2)
        # u . x_1 = (x_1 x n_1) . n_2 and u . x_2 = n_1 . (n_2 x x_2).
        p = _gram(n_1, cross_product(n_2, x_2)) - _gram(cross_product(x_1, n_1), n_2)
        s_squared = (n_1_n_2**2 - n_1_n_1 * n_2_n_2).clamp_min(0.0)

        # Series for small s, as in `generator_components`.
        s = s_squared.sqrt()
        small = s < ε_stab
        w_ = torch.asinh(s)
        w_safe = torch.where(small, 1.0, w_)
        α = torch.where(small, 1.0 - s_squared / 6.0 + 3.0 * s_squared**2 / 40.0, w_ / torch.where(small, 1.0, s))
        β = torch.where(small, w_**2 / 12.0 - w_**4 / 720.0, (w_safe / 2.0) / torch.tanh(w_safe / 2.0) - 1.0)
        γ = torch.where(small, (1.0 / 12.0 - w_**2 / 720.0) * α**2, β / torch.where(small, 1.0, s_squared))

        a, b, c = 1.0 + β, -α / 2.0, γ * p
        u_n = a * n_1_Δx + b * (n_1_n_2 * n_1_Δx - n_1_n_1 * n_2_Δx)
        u_u = a**2 * Δx_Δx - b**2 * (s_squared * Δx_Δx + p**2) - c**2 * s_squared + 2.0 * a * c * p
        C = w_time**2 * u_n**2 + w_space**2 * (u_n**2 - u_u) + w_boost**2 * α**2 * s_squared
        return C.clamp_min(0.0)

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


def _gram(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Compute the pairwise inner products `G[..., i, j] = a[..., i, :] . b[..., j, :]`."""
    return a @ b.mT


def _dot(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Compute the row-wise inner products `d[..., i, 0] = a[..., i, :] . b[..., i, :]`."""
    return (a * b).sum(-1, keepdim=True)


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


def _log_affine(ω: torch.Tensor, t: torch.Tensor, s_squared: torch.Tensor) -> torch.Tensor:
    """
    Compute `log([[exp(ω), t], [0, 1]]) = [[ω, V^-1 t], [0, 0]]` for `ω` with
    `ω^3 = -s^2 ω`, with `V` as in `_exp_affine`: `V^-1 = I - ω / 2 + G ω^2`,
    where `G(s^2) = (1 - θ / 2 cot(θ / 2)) / θ^2` if `s^2 = θ^2 > 0`, and
    `(θ / 2 coth(θ / 2) - 1) / θ^2` if `s^2 = -θ^2 < 0`. Since `G` is analytic in
    `s^2`, we use its Taylor expansion if `|s^2| < 1`.
    """
    small = s_squared.abs() < 1.0
    θ = torch.where(small, 1.0, s_squared.abs()).sqrt()
    G = torch.where(
        small,
        1 / 12 + s_squared / 720 + s_squared**2 / 30240 + s_squared**3 / 1209600,
        torch.where(s_squared > 0, 1 - (θ / 2) / torch.tan(θ / 2), (θ / 2) / torch.tanh(θ / 2) - 1) / θ**2,
    )
    I = torch.eye(3, device=ω.device, dtype=ω.dtype)
    A = torch.zeros(*ω.shape[:-2], 4, 4, device=ω.device, dtype=ω.dtype)
    A[..., :3, :3] = ω
    A[..., :3, 3:] = (I - ω / 2 + G * ω @ ω) @ t
    return A


def _sigmoid(x: torch.Tensor, scale=88.0) -> torch.Tensor:
    return scale * torch.tanh(x / scale)


def _slices(sizes: list[int]) -> list[slice]:
    """Consecutive slices of the given sizes."""
    ends = [sum(sizes[:i + 1]) for i in range(len(sizes))]
    return [slice(end - size, end) for size, end in zip(sizes, ends)]
