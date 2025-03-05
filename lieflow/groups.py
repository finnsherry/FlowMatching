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
    Also provides three example implementations of 
      1. `Rn(n)` <: `Group`: n-dimensional translation group R^n.
      2. `SE2()` <: `Group`: special Euclidean group of roto-translations on
      R^2.
      3. `SO3()` <: `MatrixGroup`: special orthogonal group of rotations on R^3.
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
        g = torch.zeros_like(g_2)
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
        g = torch.zeros_like(g_2)
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

        tan = torch.tan(θ/2.)
        tanc = _tanc(θ/2.)

        A[..., 0] = (y * tan + x) / tanc
        A[..., 1] = (-x * tan + y) / tanc
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
        
        cos = torch.cos(c3/2.)
        sin = torch.sin(c3/2.)
        sinc = torch.sinc(c3/(2. * torch.pi)) # torch.sinc(x) = sin(pi x) / (pi x)

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

        B[..., 0] = cos * A[..., 0] - sin * A[1]
        B[..., 1] = sin * A[..., 0] + cos * A[1]
        B[..., 2] = A[..., 2]
        return B
    
    def __repr__(self):
        return "SE(2)"


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
        self.lie_algebra_basis = torch.Tensor([
            [
                [0., 0., 0.],
                [0., 0., -1.],
                [0., 1., 0.],
            ],
            [
                [0., 0., 1.],
                [0., 0., 0.],
                [-1., 0., 0.],
            ],
            [
                [0., -1., 0.],
                [1., 0., 0.],
                [0., 0., 0.],
            ],
        ])

    def log(self, R):
        """
        Lie group logarithm of `R`, i.e. `A` in Lie algebra such that
        `exp(A) = R`.

        Pytorch does not actually have a matrix log built in, but for SO(3) it
        is not too complicated.
        """
        q = torch.arccos((R.diagonal(offset=0, dim1=-1, dim2=-2).sum(-1) - 1) / 2)
        return (R - R.transpose(-2, -1)) / (2 * torch.sinc(q[..., None, None] / torch.pi))
    
    def lie_algebra_components(self, A):
        """
        Compute the components of Lie algebra basis `A` with respect to the
        basis given by `self.lie_algebra_basis`.
        """
        return torch.cat((A[..., 2, 1], A[..., 0, 2], A[..., 1, 0]), dim=-1)
    
    def __repr__(self):
        return "SO(3)"
    

# Utils
    
def _mod_offset(x, period, offset):
    """Compute `x` modulo `period` with offset `offset`."""
    return x - (x - offset)//period * period

def _tanc(x):
    """Compute `tan(x)/x`."""
    return torch.where(x == 0, 1, torch.tan(x) / x)