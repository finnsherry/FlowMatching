"""
groups
======

Classes that encapsulate basic Lie group and Lie algebra properties.
There are two abstract parent classes:
  1. `Group`: Use this class when the group and algebra can be efficiently
  parametrised with the same number of parameters. Requires implementing
  hand crafted group multiplication, multiplication by inverse, exponential,
  and logarithm.
  2. `MatrixGroup`: Use this class when the group can be efficiently
  represented with matrices. Group multiplication, multiplication by
  inverse, and exponential make use of corresponding PyTorch methods. Since
  PyTorch does not implement a matrix logarithm, this must be provided.
Also provides four example implementations of
  1. `Rn(n)` <: `Group`: n-dimensional translation group R^n.
  2. `SE2()` <: `Group`: special Euclidean group of roto-translations on
  R^2.
  3. `SE2byRn` <: `Group`: direct product of SE(2) and R^n.
  4. `SO3()` <: `MatrixGroup`: special orthogonal group of rotations on R^3.
"""

from abc import ABC, abstractmethod
from typing import Literal
import torch


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
        g = torch.zeros(torch.broadcast_shapes(g_1.shape, g_2.shape), device=g_2.device)
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
        g = torch.zeros(torch.broadcast_shapes(g_1.shape, g_2.shape), device=g_2.device)
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
        g = torch.zeros(torch.broadcast_shapes(g_1.shape, g_2.shape), device=g_2.device)
        g[..., :3] = self.se2.L(g_1[..., :3], g_2[..., :3])
        g[..., 3:] = self.rn.L(g_1[..., 3:], g_2[..., 3:])
        return g

    def L_inv(self, g_1, g_2):
        """
        Left multiplication of `g_2 = (x_2, p_2)` by `g_1^-1 = (-x_1, p_1^-1)`,
        i.e. `(x_2 - x_1, p_1^-1 p_2)`.
        """
        g = torch.zeros(torch.broadcast_shapes(g_1.shape, g_2.shape), device=g_2.device)
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
        g = torch.zeros(torch.broadcast_shapes(g_1.shape, g_2.shape), device=g_2.device)
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
        g = torch.zeros(torch.broadcast_shapes(g_1.shape, g_2.shape), device=g_2.device)
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
    Direct product of m-dimensional translation group and the translation-
    scaling group on .

    Args:
        `se2`: instance of the special Euclidean group.
        `rn`: instance of the n-dimensional translation group.
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
        g = torch.zeros(torch.broadcast_shapes(g_1.shape, g_2.shape), device=g_2.device)
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
        g = torch.zeros(torch.broadcast_shapes(g_1.shape, g_2.shape), device=g_2.device)
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
        g = torch.zeros(torch.broadcast_shapes(g_1.shape, g_2.shape), device=g_2.device)

        t_1 = g_1[..., :2]
        A_1 = g_1[..., 2:].view(*g_1.shape[:-1], 2, 2)

        t_2 = g_2[..., :2, None]
        A_2 = g_2[..., 2:].view(*g_2.shape[:-1], 2, 2)

        g[..., 0:2] = t_1 + (A_1 @ t_2).squeeze(-1)
        g[..., 2:] = (A_1 @ A_2).view(*g.shape[:-1], 4)
        return g

    def L_inv(self, g_1, g_2):
        """
        Left multiplication of `g_2` by `g_1^-1`.
        """
        g = torch.zeros(torch.broadcast_shapes(g_1.shape, g_2.shape), device=g_2.device)

        t_1 = g_1[..., :2, None]
        A_1 = g_1[..., 2:].view(*g_1.shape[:-1], 2, 2)
        A_1_inv = torch.linalg.inv(A_1)

        t_2 = g_2[..., :2, None]
        A_2 = g_2[..., 2:].view(*g_2.shape[:-1], 2, 2)

        g[..., 0:2] = (A_1_inv @ (t_2 - t_1)).squeeze(-1)
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
        r = _mod_offset(
            R[..., 1, 0] / torch.sinc(_arccos(R[..., 0, 0]) / torch.pi),
            2.0 * torch.pi,
            -torch.pi,
        )

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

        cos, sin = r.cos(), r.sin()  # [...], [...]
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


class SO3(MatrixGroup):
    """
    Special orthogonal group of rotations on R^3.
    """

    def __init__(self):
        super().__init__(
            dim=3,
            mat_dim=3 * 3,
            lie_algebra_basis=torch.Tensor(
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
                ]
            ),
        )

    def log(self, R, ε_stab=0.001):
        """
        Lie group logarithm of `R`, i.e. `A` in Lie algebra such that
        `exp(A) = R`.

        Pytorch does not actually have a matrix log built in, but for SO(3) it
        is not too complicated.
        """
        q = _arccos((_trace(R) - 1) / 2)
        return (R - R.transpose(-2, -1)) / (
            2 * _sinc(q[..., None, None], ε_stab=ε_stab)
        )

    def exp(self, A, ε_stab=0.001):
        """Rodrigues formula"""
        A_vec = self.lie_algebra_components(A)
        θ = (A_vec**2).sum(-1).sqrt()[..., None, None]
        A_norm = torch.where(θ < ε_stab, A, A / θ)

        return (
            torch.eye(3) + torch.sin(θ) * A_norm + (1 - torch.cos(θ)) * A_norm @ A_norm
        )

    def lie_algebra_components(self, A):
        """
        Compute the components of Lie algebra basis `A` with respect to the
        basis given by `self.lie_algebra_basis`.
        """
        return torch.cat(
            (A[..., 2, 1, None], A[..., 0, 2, None], A[..., 1, 0, None]), dim=-1
        )

    def __repr__(self):
        return "SO(3)"


class SE3(MatrixGroup):
    """
    Special euclidean group of roto-translations on R^3.
    """

    def __init__(self):
        super().__init__(
            dim=6,
            mat_dim=4 * 4,
            lie_algebra_basis=torch.Tensor(
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
                ]
            ),
        )
        self.so3 = SO3()

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

    def get_screw_displacement(self, g, ε_stab=0.001):
        c, t_par, _, R = self._get_screw_displacement(g, ε_stab=ε_stab)
        return c, t_par, R

    def get_screw_displacement_generator(self, g, ε_stab=0.001):
        c, t_par, ω, _ = self._get_screw_displacement(g, ε_stab=ε_stab)
        return c, t_par, ω

    def _get_screw_displacement(self, g, ε_stab=0.001):
        t, R = self.get_translation_rotation(g)
        ω = self.so3.log(R, ε_stab=ε_stab)
        ω_vec = self.so3.lie_algebra_components(ω)
        θ = (ω_vec**2).sum(-1, keepdim=True).sqrt()

        parallel = θ < ε_stab

        K = (ω_vec / θ).nan_to_num()

        t_par = (t * K).sum(-1, keepdim=True) * K
        t_perp = t - t_par

        c = 0.5 * (t_perp + cross_product(_cotan(θ / 2.0) * K, t_perp)).nan_to_num()

        return c * ~parallel, t * parallel + t_par * ~parallel, ω, R

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


class M3(HomogeneousSpace):
    """
    Position-Orientation space on 3D Euclidean space.
    """

    def __init__(self, generator: Literal["mav", "pure_rotation"] | float = "mav"):
        self.se3 = SE3()
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
        Choose either the `mav` or the `pure_rotation` generator.

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

        k0 = (cross_product(n_1, n_2) / sinθ).nan_to_num()
        k0 = k0 * ~parallel + torch.zeros_like(k0) * parallel
        x_m = (x_1 + x_2) / 2.0
        x_diff = x_2 - x_1
        use_mav = x_diff.norm(dim=-1, keepdim=True) < ε_stab
        x_perp = (k0 * x_diff).sum(-1, keepdim=True) * k0
        x_par = x_diff - x_perp

        c_mav = (x_m + 0.5 * _cotan(θ / 2.0) * cross_product(k0, x_par)).nan_to_num()
        v_mav = x_perp

        ω_vec_mav = θ * k0
        ω_mav = (ω_vec_mav[..., None, None] * self.se3.so3.lie_algebra_basis).sum(-3)

        A = torch.zeros(*shape, 4, 4, device=device)
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
                φ = torch.tensor([generator])
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

    def act(self, g, p):
        return g @ p

    def get_position_orientation(self, p):
        x = p[..., :3, -2]
        n = p[..., :3, -1]
        return x, n

    def pack_position_orientation(self, x, n):
        p = torch.zeros(*n.shape[:-1], 4, 2, device=n.device)
        p[..., :3, -2] = x
        p[..., 3, -2] = 1
        p[..., :3, -1] = n
        return p

    def __repr__(self):
        return "M3"


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


def _cotan(x: torch.Tensor) -> torch.Tensor:
    return 1 / torch.tan(x)


def _arccos(x: torch.Tensor) -> torch.Tensor:
    return torch.arccos(torch.clamp(x, -1.0, 1.0))


def _sinc(x: torch.Tensor, ε_stab=0.0001) -> torch.Tensor:
    return torch.sinc(x / ((1 + ε_stab) * torch.pi))


def _sigmoid(x: torch.Tensor, scale=88.0) -> torch.Tensor:
    return scale * torch.tanh(x / scale)
