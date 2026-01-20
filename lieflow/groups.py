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

from abc import ABC
import torch


class Group(ABC):
    """
    Class encapsulating basic Lie group and Lie algebra properties for groups
    that can be efficiently parametrised with as many parameters as the group
    dimension.

    Requires implementing hand crafted group multiplication, multiplication by
    inverse, exponential, and logarithm.
    """

    def __init__(self):
        super().__init__()
        self.dim = None

    def L(self, g_1, g_2):
        """
        Left multiplication of `g_2` by `g_1`, i.e. `g_1 + g_2`.
        """
        raise NotImplementedError

    def L_inv(self, g_1, g_2):
        """
        Left multiplication of `g_2` by `g_1^-1`, i.e. `g_2 - g_1`.
        """
        raise NotImplementedError

    def log(self, g):
        """
        Lie group logarithm of `g`, i.e. `g`.
        """
        raise NotImplementedError

    def exp(self, A):
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
        super().__init__()
        self.dim = n

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
        super().__init__()
        self.dim = 3

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
        sinc = torch.sinc(θ / (2.0 * torch.pi))  # torch.sinc(x) = sin(pi x) / (pi x)

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
        sinc = torch.sinc(c3 / (2.0 * torch.pi))  # torch.sinc(x) = sin(pi x) / (pi x)

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
        super().__init__()
        self.dim = se2.dim + rn.dim
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
        super().__init__()
        self.dim = n + 1

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
        super().__init__()
        self.dim = rm.dim + tsn.dim
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


class MatrixGroup(ABC):
    """
    Class encapsulating basic Lie group and Lie algebra properties for groups
    that can be efficiently represented with matrices.

    Group multiplication, multiplication by inverse, and exponential make use of
    corresponding PyTorch methods. Since PyTorch does not implement a matrix
    logarithm, this must be provided.
    """

    def __init__(self):
        super().__init__()
        self.dim = None
        self.mat_dim = None
        self.lie_algebra_basis = None

    def L(self, R_1, R_2):
        """
        Left multiplication of `R_2` by `R_1`.
        """
        return R_1 @ R_2

    def L_inv(self, R_1, R_2):
        """
        Left multiplication of `R_2` by `R_1^-1`.
        """
        return torch.linalg.solve(R_1, R_2)

    def log(self, R):
        """
        Lie group logarithm of `R`, i.e. `A` in Lie algebra such that
        `exp(A) = R`.

        Pytorch does not actually have a matrix log built-in, so this must be
        provided.
        """
        raise NotImplementedError

    def exp(self, A):
        """
        Lie group exponential of `A`, i.e. `R` in Lie group such that
        `exp(A) = R`.
        """
        return torch.matrix_exp(A)

    def lie_algebra_components(self, A):
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
        super().__init__()
        self.dim = 3
        self.mat_dim = 3 * 3
        self.lie_algebra_basis = torch.Tensor(
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
            2 * torch.sinc(q[..., None, None] / ((1 + ε_stab) * torch.pi))
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
        super().__init__()
        self.dim = 6
        self.mat_dim = 4 * 4
        self.lie_algebra_basis = torch.Tensor(
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

    def __init__(self):
        super().__init__()
        self.G: MatrixGroup = None
        self.mat_dim = None

    def get_generator(self, p_1, p_2):
        raise NotImplementedError

    def act(self, g, p):
        raise NotImplementedError


class M3(HomogeneousSpace):
    """
    Position-Orientation space on 3D Euclidean space.
    """

    def __init__(self, generator="mav"):
        super().__init__()
        self.se3 = SE3()
        self.G = self.se3
        self.mat_dim = 4 * 2
        self.generator = generator

    def get_generator(self, p_1, p_2, ε_stab=0.001, generator=None):
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
                n_1p = n_1 - (k * n_1).sum(-1, keepdim=True) * k
                n_2p = n_2 - (k * n_2).sum(-1, keepdim=True) * k
                θ = torch.atan2(
                    (k * cross_product(n_1p, n_2p)).sum(-1, keepdim=True),
                    (n_1p * n_2p).sum(-1, keepdim=True),
                )

                x_par = x_diff
                c = (x_m + 0.5 * _cotan(θ / 2.0) * cross_product(k, x_par)).nan_to_num()

                ω_vec = θ * k
                ω = (ω_vec[..., None, None] * self.se3.so3.lie_algebra_basis).sum(-3)
                print(use_mav.shape, parallel.shape)

                return self.se3.pack_translation_rotation(
                    A,
                    x_diff * parallel
                    + (
                        -cross_product(ω_vec, c) * ~use_mav
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


def _trace(R):
    return R.diagonal(offset=0, dim1=-1, dim2=-2).sum(-1)


def _expc(x):
    """Compute `x / (exp(x) - 1)`."""
    return torch.where(
        x.abs() < 1.0,
        1.0 - x / 2.0 + x**2 / 12.0 - x**4 / 720.0 + x**6 / 30240.0,
        x / (torch.exp(x) - 1.0),
    )


def cross_product(x, y):
    shape = torch.broadcast_shapes(x.shape, y.shape)
    return torch.linalg.cross(x.expand(shape), y.expand(shape))


def _cotan(x):
    return 1 / torch.tan(x)


def _arccos(x):
    return torch.arccos(_sigmoid(x))


def _sigmoid(x, scale=88.0):
    return scale * torch.tanh(x / scale)
