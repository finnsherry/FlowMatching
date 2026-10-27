"""
models
======

Implementations of flow matching[1] on Lie groups over exponential curves,[2]
as well as on position-orientation space M3, which is a homogeneous space
of SE(3), using mav exponential curves.[3]
The implementation is slightly different for groups of type `Group` than
for those of type `MatrixGroup`; the method `get_model_FM` will return an
instance of a flow matching model, corresponding to the type of the group.

References:
  [1]: Y. Lipman, R.T.Q. Chen, H. Ben-Hami, M. Nickel, and M. Le.
  "Flow Matching for Generative Modeling." arXiv preprint (2022).
  DOI:10.48550/arXiv.2210.02747.
  [2]: F.M. Sherry and B.M.N. Smets. "Flow Matching on Lie Groups." 7th
  International Conference on Geometric Science of Information (2025).
  DOI:10.1007/978-3-032-03918-7_6.
  [3]: G. Bellaard and B.M.N. Smets. "Roto-Translation Invariant Metrics on
  Position-Orientation Space." 7th International Conference on Geometric Science
  of Information (2025).
  DOI:10.1007/978-3-032-03918-7_4.
"""

from functools import partial

import numpy as np
import torch
from torch import nn
from tqdm.auto import tqdm
from typing import Literal

from lieflow.groups import Group, HomogeneousSpace, MatrixGroup
from lieflow.metrics import compare


def get_model_FM(
    G: Group | MatrixGroup | HomogeneousSpace,
    L: int = 2,
    power_group: bool = False,
    H: int = 64,
    embed_dim: int = 256,
    num_heads: int = 8,
    expansion: int = 4,
    activation: Literal[nn.ReLU, nn.GELU, nn.ELU, nn.SiLU] = nn.ReLU,
):
    """
    Return an instance of a flow matching model corresponding to the type of
    `G`.

    Args:
        `G`: group of type `Group` or `MatrixGroup`, or a `HomogeneousSpace`.
      Optional:
        `L`: depth of the network: number of layers - 2. Defaults to 2.
        `power_group`: boolean describing whether data live in the group `G`
        (False) or in a simple power thereof (True). If `power_group == False`,
        we use an MLP, with optional arguments `H`. If `power_group == True`, we
        use a transformer, with optional arguments `embed_dim`, `num_heads`, and
        `expansion`. Defaults to False.
        `H`: width of the network: number of channels. Defaults to 64.
        `embed_dim`: dimension of the linear embedding of the tokens.
        `num_heads`: number of heads in the multihead attention.
        `expansion`: expansion factor in the pointwise feedforward network.
        `activation`: activation function. Defaults to `nn.ReLU`.
    """
    if isinstance(G, Group):
        if power_group:
            return FlowFieldPowerGroup(
                G,
                embed_dim=embed_dim,
                num_heads=num_heads,
                expansion=expansion,
                L=L,
                activation=activation,
            )
        else:
            return FlowFieldGroup(G, H=H, L=L, activation=activation)
    elif isinstance(G, MatrixGroup):
        return FlowFieldMatrixGroup(G, H=H, L=L, activation=activation)
    elif isinstance(G, HomogeneousSpace):
        return FlowFieldHomogeneousSpace(G, H=H, L=L, activation=activation)
    else:
        raise TypeError(
            f"{G} is neither a `Group`, a `MatrixGroup`, nor a  `HomogeneousSpace`!"
        )


class FlowFieldGroup(nn.Module):
    """
    Model for flow matching[1] on Lie groups of type `Group` over exponential
    curves.[2]

    Args:
        `G`: group of type `Group`.
      Optional:
        `H`: width of the network: number of channels. Defaults to 64.
        `L`: depth of the network: number of hidden layers, which are the same
          `ResidualBlock`. Defaults to 2.
        `activation`: activation function. Defaults to `nn.ReLU`.

    References:
        [1]: Y. Lipman, R.T.Q. Chen, H. Ben-Hami, M. Nickel, and M. Le.
          "Flow Matching for Generative Modeling." arXiv preprint (2022).
          DOI:10.48550/arXiv.2210.02747.
        [2]: F.M. Sherry and B.M.N. Smets. "Flow Matching on Lie Groups." 7th
          International Conference on Geometric Science of Information (2025).
          DOI:10.1007/978-3-032-03918-7_6.
    """

    def __init__(
        self,
        G: Group,
        H: int = 64,
        L: int = 2,
        activation: Literal[nn.ReLU, nn.GELU, nn.ELU, nn.SiLU] = nn.ReLU,
    ):
        super().__init__()
        self.G = G
        self.network = _mlp(G.dim + 1, G.dim, H, L, activation)

    def forward(self, g_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return self.network(torch.cat((g_t, t), dim=-1))

    def step(
        self, g_t: torch.Tensor, t: torch.Tensor, Δt: torch.Tensor
    ) -> torch.Tensor:
        t = t.view(1, 1).expand(g_t.shape[0], 1)
        return self.G.L(g_t, self.G.exp(Δt * self(g_t, t)))

    def step_back(
        self, g_t: torch.Tensor, t: torch.Tensor, Δt: torch.Tensor
    ) -> torch.Tensor:
        return self.step(g_t, t, -Δt)

    @torch.no_grad()
    def sample(self, g_0: torch.Tensor, N_steps: int = 120) -> torch.Tensor:
        """
        Integrate the flow from samples `g_0` at t = 0 to t = 1 in `N_steps`
        steps.
        """
        self.eval()
        Δt = torch.tensor([1.0 / N_steps], device=g_0.device, dtype=g_0.dtype)
        g_t = g_0
        for i in range(N_steps):
            t = torch.tensor([i / N_steps], device=g_0.device, dtype=g_0.dtype)
            g_t = self.step(g_t, t, Δt)
        return g_t

    def train_network(
        self, device, train_loader, optimizer, scheduler, ema, loss, show_progress=True
    ):
        """
        Train for one epoch. After every batch, step the learning rate
        `scheduler`, and update the exponential moving average `ema` of the
        weights, which is a `torch.optim.swa_utils.AveragedModel` of `self`.
        """
        self.train()
        N_batches = len(train_loader)
        losses = np.zeros(N_batches)
        if show_progress:
            loop = tqdm(
                enumerate(train_loader),
                total=N_batches,
                desc="Training",
                dynamic_ncols=True,
                unit="batch",
                leave=False,
            )
        else:
            loop = enumerate(train_loader)
        for i, (g_0, g_1) in loop:
            t = torch.rand(len(g_1), 1, device=device)
            g_0, g_1 = g_0.to(device), g_1.to(device)
            A_t = self.G.log(self.G.L_inv(g_0, g_1))
            g_t = self.G.L(g_0, self.G.exp(t * A_t))
            optimizer.zero_grad()
            batch_loss = loss(self(g_t, t), A_t)
            losses[i] = float(batch_loss.cpu().item())
            batch_loss.backward()
            optimizer.step()
            scheduler.step()
            ema.update_parameters(self)
        return losses.mean()

    @torch.no_grad()
    def evaluate(
        self,
        g_0: torch.Tensor,
        g_1: torch.Tensor,
        w: torch.Tensor | None = None,
        bandwidths: torch.Tensor | list[float] | None = None,
        chunk_size: int = 128,
        N_steps: int = 120,
    ) -> dict[str, float]:
        """
        Compute the Wasserstein-1 distance, precision, coverage, and squared MMD
        between samples generated by integrating the flow from `g_0`, and
        target samples `g_1`.

        Args:
            `g_0`: initial samples of shape [N, ...].
            `g_1`: target samples of shape [N, ...].
          Optional:
            `w`: weights of the components of the generator, as in
            `LogarithmicDistance`. Defaults to all ones.
            `bandwidths`: bandwidths of the Gaussian kernels in the MMD. Should be
            the same for all models that are compared. Defaults to `[m/2, m, 2m]`,
            with `m` the median distance between target samples, so it only depends
            on `g_1`.
            `chunk_size`: number of rows of the distance matrices computed at once,
            to limit memory usage. Defaults to 128.
            `N_steps`: number of integration steps. Defaults to 120.
        """
        g_1_pred = self.sample(g_0, N_steps=N_steps)
        return compare(
            self.G, g_1_pred, g_1, w, bandwidths=bandwidths, chunk_size=chunk_size
        )

    @property
    def parameter_count(self):
        return sum(p.numel() for p in self.parameters())


class FlowFieldMatrixGroup(nn.Module):
    """
    Model for flow matching[1] on Lie groups of type `MatrixGroup` over
    exponential curves.[2]

    Args:
        `G`: group of type `MatrixGroup`.
      Optional:
        `H`: width of the network: number of channels. Defaults to 64.
        `L`: depth of the network: number of hidden layers, which are the same
          `ResidualBlock`. Defaults to 2.
        `activation`: activation function. Defaults to `nn.ReLU`.

    References:
        [1]: Y. Lipman, R.T.Q. Chen, H. Ben-Hami, M. Nickel, and M. Le.
          "Flow Matching for Generative Modeling." arXiv preprint (2022).
          DOI:10.48550/arXiv.2210.02747.
        [2]: F.M. Sherry and B.M.N. Smets. "Flow Matching on Lie Groups." 7th
          International Conference on Geometric Science of Information (2025).
          DOI:10.1007/978-3-032-03918-7_6.
    """

    def __init__(
        self,
        G: MatrixGroup,
        H: int = 64,
        L: int = 2,
        activation: Literal[nn.ReLU, nn.GELU, nn.ELU, nn.SiLU] = nn.ReLU,
    ):
        super().__init__()
        self.G = G
        self.network = _mlp(G.mat_dim + 1, G.dim, H, L, activation)

    def forward(self, R_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return self.network(
            torch.cat(
                (
                    R_t.flatten(start_dim=-2, end_dim=-1),
                    t.flatten(start_dim=-2, end_dim=-1),
                ),
                dim=-1,
            )
        )

    def step(
        self, R_t: torch.Tensor, t: torch.Tensor, Δt: torch.Tensor
    ) -> torch.Tensor:
        t = t.view(1, 1).expand(R_t.shape[0], 1, 1)
        # Components w.r.t. Lie algebra basis.
        a_t = self(R_t, t)
        basis = self.G.lie_algebra_basis
        A_t = (a_t[..., None, None] * basis).sum(-3)
        return self.G.L(R_t, self.G.exp(Δt * A_t))

    def step_back(
        self, R_t: torch.Tensor, t: torch.Tensor, Δt: torch.Tensor
    ) -> torch.Tensor:
        return self.step(R_t, t, -Δt)

    @torch.no_grad()
    def sample(self, R_0: torch.Tensor, N_steps: int = 120) -> torch.Tensor:
        """
        Integrate the flow from samples `R_0` at t = 0 to t = 1 in `N_steps`
        steps.
        """
        self.eval()
        Δt = torch.tensor([1.0 / N_steps], device=R_0.device, dtype=R_0.dtype)
        R_t = R_0
        for i in range(N_steps):
            t = torch.tensor([i / N_steps], device=R_0.device, dtype=R_0.dtype)
            R_t = self.step(R_t, t, Δt)
        return R_t

    def train_network(
        self, device, train_loader, optimizer, scheduler, ema, loss, show_progress=True
    ):
        """
        Train for one epoch. After every batch, step the learning rate
        `scheduler`, and update the exponential moving average `ema` of the
        weights, which is a `torch.optim.swa_utils.AveragedModel` of `self`.
        """
        self.train()
        N_batches = len(train_loader)
        losses = np.zeros(N_batches)
        if show_progress:
            loop = tqdm(
                enumerate(train_loader),
                total=N_batches,
                desc="Training",
                dynamic_ncols=True,
                unit="batch",
                leave=False,
            )
        else:
            loop = enumerate(train_loader)
        for i, (R_0, R_1) in loop:
            t = torch.rand(len(R_1), 1, 1, device=device)
            R_0, R_1 = R_0.to(device), R_1.to(device)
            A_t = self.G.log(self.G.L_inv(R_0, R_1))
            R_t = self.G.L(R_0, self.G.exp(t * A_t))
            # Components w.r.t. Lie algebra basis.
            a_t = self.G.lie_algebra_components(A_t)
            optimizer.zero_grad()
            batch_loss = loss(self(R_t, t), a_t)
            losses[i] = float(batch_loss.cpu().item())
            batch_loss.backward()
            optimizer.step()
            scheduler.step()
            ema.update_parameters(self)
        return losses.mean()

    @torch.no_grad()
    def evaluate(
        self,
        R_0: torch.Tensor,
        R_1: torch.Tensor,
        w: torch.Tensor | None = None,
        bandwidths: torch.Tensor | list[float] | None = None,
        chunk_size: int = 128,
        N_steps: int = 120,
    ) -> dict[str, float]:
        """
        Compute the Wasserstein-1 distance, precision, coverage, and squared MMD
        between samples generated by integrating the flow from `R_0`, and
        target samples `R_1`.

        Args:
            `R_0`: initial samples of shape [N, ...].
            `R_1`: target samples of shape [N, ...].
          Optional:
            `w`: weights of the components of the generator, as in
            `LogarithmicDistance`. Defaults to all ones.
            `bandwidths`: bandwidths of the Gaussian kernels in the MMD. Should be
            the same for all models that are compared. Defaults to `[m/2, m, 2m]`,
            with `m` the median distance between target samples, so it only depends
            on `R_1`.
            `chunk_size`: number of rows of the distance matrices computed at once,
            to limit memory usage. Defaults to 128.
            `N_steps`: number of integration steps. Defaults to 120.
        """
        R_1_pred = self.sample(R_0, N_steps=N_steps)
        return compare(
            self.G, R_1_pred, R_1, w, bandwidths=bandwidths, chunk_size=chunk_size
        )

    @property
    def parameter_count(self):
        return sum(p.numel() for p in self.parameters())


class FlowFieldHomogeneousSpace(nn.Module):
    """
    Model for flow matching[1] on Lie group homogeneous spaces.
    Of particular interest is position-orientation space `M3` over SE(3), for
    which we can use mav exponential curves.[2][3]

    Args:
        `h`: instance of `HomogeneousSpace`.
      Optional:
        `H`: width of the network: number of channels. Defaults to 64.
        `L`: depth of the network: number of hidden layers, which are the same
          `ResidualBlock`. Defaults to 2.
        `activation`: activation function. Defaults to `nn.ReLU`.

    References:
        [1]: Y. Lipman, R.T.Q. Chen, H. Ben-Hami, M. Nickel, and M. Le.
          "Flow Matching for Generative Modeling." arXiv preprint (2022).
          DOI:10.48550/arXiv.2210.02747.
        [2]: F.M. Sherry and B.M.N. Smets. "Flow Matching on Lie Groups." 7th
          International Conference on Geometric Science of Information (2025).
          DOI:10.1007/978-3-032-03918-7_6.
        [3]: G. Bellaard and B.M.N. Smets. "Roto-Translation Invariant Metrics
          on Position-Orientation Space." 7th International Conference on
          Geometric Science of Information (2025).
          DOI:10.1007/978-3-032-03918-7_4.
    """

    def __init__(
        self,
        h: HomogeneousSpace,
        H: int = 64,
        L: int = 2,
        activation: Literal[nn.ReLU, nn.GELU, nn.ELU, nn.SiLU] = nn.ReLU,
    ):
        super().__init__()
        self.h = h
        self.network = _mlp(h.mat_dim + 1, h.G.dim, H, L, activation)

    def forward(self, p_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return self.network(torch.cat((p_t.flatten(-2, -1), t.flatten(-2, -1)), dim=-1))

    def step(
        self, p_t: torch.Tensor, t: torch.Tensor, Δt: torch.Tensor
    ) -> torch.Tensor:
        t = t.view(1, 1).expand(len(p_t), 1, 1)
        A_t = (self(p_t, t)[..., None, None] * self.h.G.lie_algebra_basis).sum(-3)
        return self.h.act(self.h.G.exp(Δt * A_t), p_t)

    def step_back(
        self, p_t: torch.Tensor, t: torch.Tensor, Δt: torch.Tensor
    ) -> torch.Tensor:
        return self.step(p_t, t, -Δt)

    @torch.no_grad()
    def sample(self, p_0: torch.Tensor, N_steps: int = 120) -> torch.Tensor:
        """
        Integrate the flow from samples `p_0` at t = 0 to t = 1 in `N_steps`
        steps.
        """
        self.eval()
        Δt = torch.tensor([1.0 / N_steps], device=p_0.device, dtype=p_0.dtype)
        p_t = p_0
        for i in range(N_steps):
            t = torch.tensor([i / N_steps], device=p_0.device, dtype=p_0.dtype)
            p_t = self.step(p_t, t, Δt)
        return p_t

    def train_network(
        self, device, train_loader, optimizer, scheduler, ema, loss, show_progress=True
    ):
        """
        Train for one epoch. After every batch, step the learning rate
        `scheduler`, and update the exponential moving average `ema` of the
        weights, which is a `torch.optim.swa_utils.AveragedModel` of `self`.
        """
        self.train()
        N_batches = len(train_loader)
        losses = np.zeros(N_batches)
        if show_progress:
            loop = tqdm(
                enumerate(train_loader),
                total=N_batches,
                desc="Training",
                dynamic_ncols=True,
                unit="batch",
                leave=False,
            )
        else:
            loop = enumerate(train_loader)
        for i, (p_0, p_1) in loop:
            t = torch.rand(len(p_1), 1, 1, device=device)
            p_0, p_1 = p_0.to(device), p_1.to(device)
            A_t = self.h.get_generator(p_0, p_1)
            g_t = self.h.G.exp(t * A_t)
            p_t = self.h.act(g_t, p_0)
            a_t = self.h.G.lie_algebra_components(A_t)
            optimizer.zero_grad()
            batch_loss = loss(self(p_t, t), a_t)
            losses[i] = float(batch_loss.cpu().item())
            batch_loss.backward()
            optimizer.step()
            scheduler.step()
            ema.update_parameters(self)
        return losses.mean()

    @torch.no_grad()
    def evaluate(
        self,
        p_0: torch.Tensor,
        p_1: torch.Tensor,
        w: torch.Tensor | None = None,
        bandwidths: torch.Tensor | list[float] | None = None,
        chunk_size: int = 128,
        N_steps: int = 120,
    ) -> dict[str, float]:
        """
        Compute the Wasserstein-1 distance, precision, coverage, and squared MMD
        between samples generated by integrating the flow from `p_0`, and
        target samples `p_1`.

        Args:
            `p_0`: initial samples of shape [N, ...].
            `p_1`: target samples of shape [N, ...].
          Optional:
            `w`: weights of the components of the generator, as in
            `LogarithmicDistance`. Defaults to all ones.
            `bandwidths`: bandwidths of the Gaussian kernels in the MMD. Should be
            the same for all models that are compared. Defaults to `[m/2, m, 2m]`,
            with `m` the median distance between target samples, so it only depends
            on `p_1`.
            `chunk_size`: number of rows of the distance matrices computed at once,
            to limit memory usage. Defaults to 128.
            `N_steps`: number of integration steps. Defaults to 120.
        """
        p_1_pred = self.sample(p_0, N_steps=N_steps)
        return compare(
            self.h, p_1_pred, p_1, w, bandwidths=bandwidths, chunk_size=chunk_size
        )

    @property
    def parameter_count(self):
        return sum(p.numel() for p in self.parameters())


class FlowFieldPowerGroup(nn.Module):
    """
    Model for flow matching[1] on powers of Lie groups of type `Group` over
    exponential curves.[2]

    Args:
        `G`: group of type `Group`.
      Optional:
        `embed_dim`: dimension of the linear embedding of the tokens.
        `num_heads`: number of heads in the multihead attention.
        `expansion`: expansion factor in the pointwise feedforward network.
        `L`: depth of the network: number of encoder blocks, which are the same
          `EncoderBlock`. Defaults to 2.

    References:
        [1]: Y. Lipman, R.T.Q. Chen, H. Ben-Hami, M. Nickel, and M. Le.
          "Flow Matching for Generative Modeling." arXiv preprint (2022).
          DOI:10.48550/arXiv.2210.02747.
        [2]: F.M. Sherry and B.M.N. Smets. "Flow Matching on Lie Groups." 7th
          International Conference on Geometric Science of Information (2025).
          DOI:10.1007/978-3-032-03918-7_6.
    """

    def __init__(
        self,
        G: Group,
        embed_dim: int = 256,
        num_heads: int = 8,
        expansion: int = 4,
        L: int = 2,
        activation: Literal[nn.ReLU, nn.GELU, nn.ELU, nn.SiLU] = nn.ReLU,
    ):
        super().__init__()
        self.G = G

        self.network = nn.Sequential(
            nn.Linear(G.dim + 1, embed_dim),
            # The same block L times, so the blocks share their weights.
            *(L * [EncoderBlock(embed_dim, num_heads, expansion, activation=activation)]),
            nn.Linear(embed_dim, G.dim),
        )

    def forward(self, g_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return self.network(torch.cat((g_t, t), dim=-1))

    def step(
        self, g_t: torch.Tensor, t: torch.Tensor, Δt: torch.Tensor
    ) -> torch.Tensor:
        t = t.view(1, 1, 1).expand(*g_t.shape[:-1], 1)
        return self.G.L(g_t, self.G.exp(Δt * self(g_t, t)))

    def step_back(
        self, g_t: torch.Tensor, t: torch.Tensor, Δt: torch.Tensor
    ) -> torch.Tensor:
        return self.step(g_t, t, -Δt)

    @torch.no_grad()
    def sample(self, g_0: torch.Tensor, N_steps: int = 120) -> torch.Tensor:
        """
        Integrate the flow from samples `g_0` at t = 0 to t = 1 in `N_steps`
        steps.
        """
        self.eval()
        Δt = torch.tensor([1.0 / N_steps], device=g_0.device, dtype=g_0.dtype)
        g_t = g_0
        for i in range(N_steps):
            t = torch.tensor([i / N_steps], device=g_0.device, dtype=g_0.dtype)
            g_t = self.step(g_t, t, Δt)
        return g_t

    def train_network(
        self, device, train_loader, optimizer, scheduler, ema, loss, show_progress=True
    ):
        """
        Train for one epoch. After every batch, step the learning rate
        `scheduler`, and update the exponential moving average `ema` of the
        weights, which is a `torch.optim.swa_utils.AveragedModel` of `self`.
        """
        self.train()
        N_batches = len(train_loader)
        losses = np.zeros(N_batches)
        if show_progress:
            loop = tqdm(
                enumerate(train_loader),
                total=N_batches,
                desc="Training",
                dynamic_ncols=True,
                unit="batch",
                leave=False,
            )
        else:
            loop = enumerate(train_loader)
        for i, (g_0, g_1) in loop:
            t = torch.rand(len(g_1), 1, 1, device=device).expand(*g_1.shape[:-1], 1)
            g_0, g_1 = g_0.to(device), g_1.to(device)
            A_t = self.G.log(self.G.L_inv(g_0, g_1))
            g_t = self.G.L(g_0, self.G.exp(t * A_t))
            optimizer.zero_grad()
            batch_loss = loss(self(g_t, t), A_t)
            losses[i] = float(batch_loss.cpu().item())
            batch_loss.backward()
            optimizer.step()
            scheduler.step()
            ema.update_parameters(self)
        return losses.mean()

    @torch.no_grad()
    def evaluate(
        self,
        g_0: torch.Tensor,
        g_1: torch.Tensor,
        w: torch.Tensor | None = None,
        bandwidths: torch.Tensor | list[float] | None = None,
        chunk_size: int = 128,
        N_steps: int = 120,
        permutation_invariant: bool = True,
    ) -> dict[str, float]:
        """
        Compute the Wasserstein-1 distance, precision, coverage, and squared MMD
        between samples generated by integrating the flow from `g_0`, and
        target samples `g_1`.

        Args:
            `g_0`: initial samples of shape [N, ...].
            `g_1`: target samples of shape [N, ...].
          Optional:
            `w`: weights of the components of the generator, as in
            `LogarithmicDistance`. Defaults to all ones.
            `bandwidths`: bandwidths of the Gaussian kernels in the MMD. Should be
            the same for all models that are compared. Defaults to `[m/2, m, 2m]`,
            with `m` the median distance between target samples, so it only depends
            on `g_1`.
            `chunk_size`: number of rows of the distance matrices computed at once,
            to limit memory usage. Defaults to 128.
            `N_steps`: number of integration steps. Defaults to 120.
            `permutation_invariant`: whether to treat samples as unordered sets
            of factors, by minimising the distance over all permutations of the
            factors. The transformer is equivariant under permutations of the
            factors, so it cannot reproduce a particular ordering. Defaults to
            True.
        """
        g_1_pred = self.sample(g_0, N_steps=N_steps)
        return compare(
            self.G,
            g_1_pred,
            g_1,
            w,
            bandwidths=bandwidths,
            chunk_size=chunk_size,
            permutation_invariant=permutation_invariant,
        )

    @property
    def parameter_count(self):
        return sum(p.numel() for p in self.parameters())


class EncoderBlock(nn.Module):
    """
    Simple transformer encoder block.

    Args:
        `embed_dim`: dimension of the linear embedding of the tokens.
        `num_heads`: number of heads in the multihead attention.
        `expansion`: expansion factor in the pointwise feedforward network.
    """
    
    def __init__(self, embed_dim: int, num_heads: int, expansion: int, activation: Literal[nn.ReLU, nn.GELU, nn.ELU, nn.SiLU] = nn.ReLU):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, expansion * embed_dim),
            activation(),
            nn.Linear(expansion * embed_dim, embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        A, _ = self.attn(x, x, x, need_weights=False)
        x = self.ln1(A + x)
        return self.ln2(x + self.ff(x))

    @property
    def parameter_count(self):
        return sum(p.numel() for p in self.parameters())


def _mlp(
    in_dim: int,
    out_dim: int,
    H: int,
    L: int,
    activation: type[nn.Module],
) -> nn.Sequential:
    """
    Return an MLP with `L` hidden layers of width `H`, which are the same
    `ResidualBlock`, followed by a layer normalisation and activation before the
    output layer.
    """
    return nn.Sequential(
        nn.Linear(in_dim, H),
        # The same block L times, so the blocks share their weights.
        *(L * [ResidualBlock(H, activation)]),
        nn.LayerNorm(H),
        activation(),
        nn.Linear(H, out_dim),
    )


class ResidualBlock(nn.Module):
    """
    Pre-activation residual block `h + W act(LayerNorm(h)) + b`. The weights `W`
    and bias `b` are initialised to zero, so that the block starts out as the
    identity. Since the hidden layers of the MLPs share their weights, a stack of
    these blocks is a forward Euler discretisation of the autonomous ODE
    `dh/ds = W act(LayerNorm(h)) + b`.

    Args:
        `H`: width of the block.
        `activation`: activation function.
    """

    def __init__(self, H: int, activation: type[nn.Module]):
        super().__init__()
        self.norm = nn.LayerNorm(H)
        self.activation = activation()
        self.linear = nn.Linear(H, H)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return h + self.linear(self.activation(self.norm(h)))


class LogarithmicDistance(nn.Module):
    def __init__(self, w: torch.Tensor):
        super().__init__()
        self.w = w
        self.ρ = partial(ρ_c_normalised, w=w)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.float:
        return self.ρ(pred, target)

    def __repr__(self):
        return f"LogarithmicDistance{tuple(self.w.numpy())}"


class PermutationInvariantLogarithmicDistance(nn.Module):
    def __init__(self, w: torch.Tensor):
        super().__init__()
        self.w = w
        self.ρ = partial(ρ_c_normalised, w=w)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.float:
        pred_norms = torch.norm(pred, dim=-1)
        pred_sorted_indices = torch.argsort(pred_norms, dim=-1)
        pred_sorted = torch.gather(
            pred, -2, pred_sorted_indices[..., None].expand_as(pred)
        )
        target_norms = torch.norm(target, dim=-1)
        target_sorted_indices = torch.argsort(target_norms, dim=-1)
        target_sorted = torch.gather(
            target, -2, target_sorted_indices[..., None].expand_as(target)
        )

        return self.ρ(pred_sorted, target_sorted)

    def __repr__(self):
        return f"LogarithmicDistance{tuple(self.w.numpy())}"


def ρ_c_normalised(
    A_1: torch.Tensor, A_2: torch.Tensor, w: torch.Tensor
) -> torch.Tensor:
    return (w**2 * (A_2 - A_1) ** 2).mean()


class DataGenerator:
    """Draws new pairs (g_0, g_1) for every batch."""
    def __init__(self, generate_g_0, generate_g_1, batch_size, batches_per_epoch):
        self.generate_g_0 = generate_g_0
        self.generate_g_1 = generate_g_1
        self.batch_size = batch_size
        self.batches_per_epoch = batches_per_epoch

    def __len__(self):
        return self.batches_per_epoch

    def __iter__(self):
        for _ in range(self.batches_per_epoch):
            yield self.generate_g_0(self.batch_size), self.generate_g_1(self.batch_size)