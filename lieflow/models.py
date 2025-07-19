"""
models
======

Implementations of flow matching[1] and shortcut modeling[2] on Lie groups
over exponential curves,[3] as well as on position-orientation space M3, which
is a homogeneous space of SE(3), using mav exponential curves.[4]
The implementation is slightly different for groups of type `Group` than
for those of type `MatrixGroup`; the methods `get_model_FM` and
`get_model_SCFM` will return an instance of a flow matching and shortcut
model, respectively, corresponding to the type of the group.

References:
  [1]: Y. Lipman, R.T.Q. Chen, H. Ben-Hami, M. Nickel, and M. Le.
  "Flow Matching for Generative Modeling." arXiv preprint (2022).
  DOI:10.48550/arXiv.2210.02747.
  [2]: K. Frans, D. Hafner, S. Levine, and P. Abbeel.
  "One Step Diffusion via Shortcut Models." arXiv preprint (2024).
  DOI:10.48550/arXiv.2410.12557
  [3]: F.M. Sherry and B.M.N. Smets. "Flow Matching on Lie Groups." arXiv
  preprint (2025).
  DOI:10.48550/arXiv.2504.00494.
  [4]: G. Bellaard and B.M.N. Smets. "Roto-Translation Invariant Metrics on
  Position-Orientation Space." arXiv preprint (2025).
  DOI:10.48550/arXiv.2504.03309.
"""

import torch
import torch.nn as nn
import numpy as np
from tqdm.auto import tqdm
from lieflow.groups import Group, MatrixGroup, M3


def get_model_FM(
    G: Group | MatrixGroup,
    L=2,
    power_group=False,
    H=64,
    embed_dim=256,
    num_heads=8,
    expansion=4,
):
    """
    Return an instance of a flow matching model corresponding to the type of
    `G`.

    Args:
        `G`: group of type `Group` or `MatrixGroup`.
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
    """
    if isinstance(G, Group):
        if power_group:
            return FlowFieldPowerGroup(
                G, embed_dim=embed_dim, num_heads=num_heads, expansion=expansion, L=L
            )
        else:
            return FlowFieldGroup(G, H=H, L=L)
    elif isinstance(G, MatrixGroup):
        return FlowFieldMatrixGroup(G, H=H, L=L)
    else:
        raise ValueError(f"{G} is neither a `Group` nor a `Matrix Group`!")


def get_model_SCFM(G: Group | MatrixGroup, H=64, L=2):
    """
    Return an instance of a shortcut model corresponding to the type of `G`.

    Args:
        `G`: group of type `Group` or `MatrixGroup`.
      Optional:
        `H`: width of the network: number of channels. Defaults to 64.
        `L`: depth of the network: number of layers - 2. Defaults to 2.
    """
    if isinstance(G, Group):
        return ShortCutFieldGroup(G, H=H, L=L)
    elif isinstance(G, MatrixGroup):
        return ShortCutFieldMatrixGroup(G, H=H, L=L)
    else:
        raise ValueError(f"{G} is neither a `Group` nor a `Matrix Group`!")


class FlowFieldGroup(nn.Module):
    """
    Model for flow matching[1] on Lie groups of type `Group` over exponential
    curves.[2]

    Args:
        `G`: group of type `Group`.
      Optional:
        `H`: width of the network: number of channels. Defaults to 64.
        `L`: depth of the network: number of layers - 2. Defaults to 2.

    References:
        [1]: Y. Lipman, R.T.Q. Chen, H. Ben-Hami, M. Nickel, and M. Le.
          "Flow Matching for Generative Modeling." arXiv preprint (2022).
          DOI:10.48550/arXiv.2210.02747.
        [2]: F.M. Sherry and B.M.N. Smets. "Flow Matching on Lie Groups." arXiv
          preprint (2025).
          DOI:10.48550/arXiv.2504.00494.
    """

    def __init__(self, G: Group, H=64, L=2):
        super().__init__()
        self.G = G
        self.network = nn.Sequential(
            nn.Linear(G.dim + 1, H),
            nn.ReLU(),
            *(
                L
                * (
                    nn.Linear(H, H),
                    nn.ReLU(),
                )
            ),
            nn.Linear(H, G.dim),
        )

    def forward(self, g_t, t):
        return self.network(torch.cat((g_t, t), dim=-1))

    def step(self, g_t, t, Δt):
        t = t.view(1, 1).expand(g_t.shape[0], 1)
        return self.G.L(g_t, self.G.exp(Δt * self(g_t, t)))

    def step_back(self, g_t, t, Δt):
        t = t.view(1, 1).expand(g_t.shape[0], 1)
        return self.G.L(g_t, self.G.exp(-Δt * self(g_t, t)))

    def train_network(self, device, train_loader, optimizer, loss):
        self.train()
        N_batches = len(train_loader)
        losses = np.zeros(N_batches)
        for i, (g_0, g_1) in tqdm(
            enumerate(train_loader),
            total=N_batches,
            desc="Training",
            dynamic_ncols=True,
            unit="batch",
            leave=False,
        ):
            t = torch.rand(len(g_1), 1).to(device)
            g_0, g_1 = g_0.to(device), g_1.to(device)
            A_t = self.G.log(self.G.L_inv(g_0, g_1))
            g_t = self.G.L(g_0, self.G.exp(t * A_t))
            optimizer.zero_grad()
            batch_loss = loss(self(g_t, t), A_t)
            losses[i] = float(batch_loss.cpu().item())
            batch_loss.backward()
            optimizer.step()
        return losses.mean()

    @property
    def parameter_count(self):
        return sum(p.numel() for p in self.parameters())


class ShortCutFieldGroup(nn.Module):
    """
    Model for shortcut modeling[1] on Lie groups of type `Group` over
    exponential curves.[2]

    Args:
        `G`: group of type `Group`.
      Optional:
        `H`: width of the network: number of channels. Defaults to 64.
        `L`: depth of the network: number of layers - 2. Defaults to 2.

    References:
        [1]: K. Frans, D. Hafner, S. Levine, and P. Abbeel.
          "One Step Diffusion via Shortcut Models." arXiv preprint (2024).
          DOI:10.48550/arXiv.2410.12557
        [2]: F.M. Sherry and B.M.N. Smets. "Flow Matching on Lie Groups." arXiv
          preprint (2025).
          DOI:10.48550/arXiv.2504.00494.
    """

    def __init__(self, G: Group, H=64, L=2):
        super().__init__()
        self.G = G
        self.network = nn.Sequential(
            nn.Linear(G.dim + 2, H),
            nn.ReLU(),
            *(
                L
                * (
                    nn.Linear(H, H),
                    nn.ReLU(),
                )
            ),
            nn.Linear(H, G.dim),
        )

    def forward(self, g_t, t, Δt):
        return self.network(torch.cat((g_t, t, Δt), dim=-1))

    def step(self, g_t, t, Δt):
        t = t.view(1, 1).expand(g_t.shape[0], 1)
        Δt = Δt.view(1, 1).expand(g_t.shape[0], 1)
        return self.G.L(g_t, self.G.exp(Δt * self(g_t, t, Δt)))

    def step_back(self, g_t, t, Δt):
        t = t.view(1, 1).expand(g_t.shape[0], 1)
        Δt = Δt.view(1, 1).expand(g_t.shape[0], 1)
        return self.G.L(g_t, self.G.exp(-Δt * self(g_t, t, Δt)))

    def train_network(self, device, train_loader, optimizer, loss, k=1 / 4):
        self.train()
        N_batches = len(train_loader)
        losses = np.zeros(N_batches)
        for i, (g_0, g_1) in tqdm(
            enumerate(train_loader),
            total=N_batches,
            desc="Training",
            dynamic_ncols=True,
            unit="batch",
            leave=False,
        ):
            N_total = len(g_1)  # Total number of samples in batch.
            t = torch.rand(len(g_1), 1).to(device)
            g_0, g_1 = g_0.to(device), g_1.to(device)
            g_t = self.G.L(g_0, self.G.exp(t * self.G.log(self.G.L_inv(g_0, g_1))))

            Δt = torch.rand(N_total, 1).to(device) * (1 - t)  # t + Δt <= 1.
            A_t = torch.zeros_like(g_t)
            N_SG = int(k * N_total)  # Number of samples used for self-consistency loss.

            # Flow-Matching
            g_1_FM, g_0_FM = g_1[N_SG:], g_0[N_SG:]
            A_t[N_SG:] = self.G.log(self.G.L_inv(g_0_FM, g_1_FM))
            Δt[N_SG:] = 0.0

            # Self-Consistency
            g_t_SG, t_SG, Δt_SG = g_t[:N_SG], t[:N_SG], Δt[:N_SG]
            B_t = self(g_t_SG, t_SG, Δt_SG)
            g_t_Δt = self.G.L(g_t_SG, self.G.exp(Δt_SG * B_t))
            B_t_Δt = self(g_t_Δt, t_SG + Δt_SG, Δt_SG)
            with torch.no_grad():
                A_t[:N_SG] = (B_t + B_t_Δt) / 2

            optimizer.zero_grad()
            batch_loss = loss(self(g_t, t, 2 * Δt), A_t)
            losses[i] = float(batch_loss.cpu().item())
            batch_loss.backward()
            optimizer.step()
        return losses.mean()

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
        `L`: depth of the network: number of layers - 2. Defaults to 2.

    References:
        [1]: Y. Lipman, R.T.Q. Chen, H. Ben-Hami, M. Nickel, and M. Le.
          "Flow Matching for Generative Modeling." arXiv preprint (2022).
          DOI:10.48550/arXiv.2210.02747.
        [2]: K. Frans, D. Hafner, S. Levine, and P. Abbeel.
          "One Step Diffusion via Shortcut Models." arXiv preprint (2024).
          DOI:10.48550/arXiv.2410.12557
        [3]: F.M. Sherry and B.M.N. Smets. "Flow Matching on Lie Groups." arXiv
          preprint (2025).
          DOI:10.48550/arXiv.2504.00494.
    """

    def __init__(self, G: MatrixGroup, H=64, L=2):
        super().__init__()
        self.G = G
        self.network = nn.Sequential(
            nn.Linear(G.mat_dim + 1, H),
            nn.ReLU(),
            *(
                L
                * (
                    nn.Linear(H, H),
                    nn.ReLU(),
                )
            ),
            nn.Linear(H, G.dim),
        )

    def forward(self, R_t, t):
        return self.network(
            torch.cat(
                (
                    R_t.flatten(start_dim=-2, end_dim=-1),
                    t.flatten(start_dim=-2, end_dim=-1),
                ),
                dim=-1,
            )
        )

    def step(self, R_t, t, Δt):
        t = t.view(1, 1).expand(R_t.shape[0], 1, 1)
        # Components w.r.t. Lie algebra basis.
        a_t = self(R_t, t)
        basis = self.G.lie_algebra_basis
        A_t = (a_t[..., None, None] * basis).sum(-3)
        return self.G.L(R_t, self.G.exp(Δt * A_t))

    def step_back(self, R_t, t, Δt):
        t = t.view(1, 1).expand(R_t.shape[0], 1, 1)
        # Components w.r.t. Lie algebra basis.
        a_t = -self(R_t, t)
        basis = self.G.lie_algebra_basis
        A_t = (a_t[..., None, None] * basis).sum(-3)
        return self.G.L(R_t, self.G.exp(Δt * A_t))

    def train_network(self, device, train_loader, optimizer, loss):
        self.train()
        N_batches = len(train_loader)
        losses = np.zeros(N_batches)
        for i, (R_0, R_1) in tqdm(
            enumerate(train_loader),
            total=N_batches,
            desc="Training",
            dynamic_ncols=True,
            unit="batch",
            leave=False,
        ):
            t = torch.rand(len(R_1), 1, 1).to(device)
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
        return losses.mean()

    @property
    def parameter_count(self):
        return sum(p.numel() for p in self.parameters())


class ShortCutFieldMatrixGroup(nn.Module):
    """
    Model for shortcut modeling[1] on Lie groups of type `MatrixGroup` over
    exponential curves.[2]

    Args:
        `G`: group of type `MatrixGroup`.
      Optional:
        `H`: width of the network: number of channels. Defaults to 64.
        `L`: depth of the network: number of layers - 2. Defaults to 2.

    References:
        [1]: Y. Lipman, R.T.Q. Chen, H. Ben-Hami, M. Nickel, and M. Le.
          "Flow Matching for Generative Modeling." arXiv preprint (2022).
          DOI:10.48550/arXiv.2210.02747.
        [2]: K. Frans, D. Hafner, S. Levine, and P. Abbeel.
          "One Step Diffusion via Shortcut Models." arXiv preprint (2024).
          DOI:10.48550/arXiv.2410.12557
        [3]: F.M. Sherry and B.M.N. Smets. "Flow Matching on Lie Groups." arXiv
          preprint (2025).
          DOI:10.48550/arXiv.2504.00494.
    """

    def __init__(self, G: MatrixGroup, H=64, L=2):
        super().__init__()
        self.G = G
        self.network = nn.Sequential(
            nn.Linear(G.mat_dim + 2, H),
            nn.ReLU(),
            *(
                L
                * (
                    nn.Linear(H, H),
                    nn.ReLU(),
                )
            ),
            nn.Linear(H, G.dim),
        )

    def forward(self, R_t, t, Δt):
        return self.network(
            torch.cat(
                (
                    R_t.flatten(start_dim=-2, end_dim=-1),
                    t.flatten(start_dim=-2, end_dim=-1),
                    Δt.flatten(start_dim=-2, end_dim=-1),
                ),
                dim=-1,
            )
        )

    def step(self, R_t, t, Δt):
        t = t.view(1, 1).expand(R_t.shape[0], 1, 1)
        Δt = Δt.view(1, 1).expand(R_t.shape[0], 1, 1)
        # Components w.r.t. Lie algebra basis.
        a_t = self(R_t, t, Δt)
        basis = self.G.lie_algebra_basis
        A_t = (a_t[..., None, None] * basis).sum(-3)
        return self.G.L(R_t, self.G.exp(Δt * A_t))

    def step_back(self, R_t, t, Δt):
        t = t.view(1, 1).expand(R_t.shape[0], 1, 1)
        Δt = Δt.view(1, 1).expand(R_t.shape[0], 1, 1)
        # Components w.r.t. Lie algebra basis.
        a_t = -self(R_t, t, Δt)
        basis = self.G.lie_algebra_basis
        A_t = (a_t[..., None, None] * basis).sum(-3)
        return self.G.L(R_t, self.G.exp(Δt * A_t))

    def train_network(self, device, train_loader, optimizer, loss, k=1 / 4):
        self.train()
        N_batches = len(train_loader)
        losses = np.zeros(N_batches)
        basis = self.G.lie_algebra_basis
        for i, (R_0, R_1) in tqdm(
            enumerate(train_loader),
            total=N_batches,
            desc="Training",
            dynamic_ncols=True,
            unit="batch",
            leave=False,
        ):
            N_total = len(R_1)  # Total number of samples in batch.
            t = torch.rand(len(R_1), 1, 1).to(device)
            R_0, R_1 = R_0.to(device), R_1.to(device)
            R_t = self.G.L(R_0, self.G.exp(t * self.G.log(self.G.L_inv(R_0, R_1))))

            Δt = torch.rand(N_total, 1, 1).to(device) * (1 - t)  # t + Δt <= 1.
            A_t = torch.zeros_like(R_t)
            N_SG = int(k * N_total)  # Number of samples used for self-consistency loss.

            # Flow-Matching
            R_1_FM, R_0_FM = R_1[N_SG:], R_0[N_SG:]
            A_t[N_SG:] = self.G.log(self.G.L_inv(R_0_FM, R_1_FM))
            Δt[N_SG:] = 0.0

            # Self-Consistency
            R_t_SG, t_SG, Δt_SG = R_t[:N_SG], t[:N_SG], Δt[:N_SG]
            b_t = self(R_t_SG, t_SG, Δt_SG)
            B_t = (b_t[..., None, None] * basis).sum(-1)
            R_t_Δt = self.G.L(R_t_SG, self.G.exp(Δt_SG * B_t))
            b_t_Δt = self(R_t_Δt, t_SG + Δt_SG, Δt_SG)
            B_t_Δt = (b_t_Δt[..., None, None] * basis).sum(-1)
            with torch.no_grad():
                A_t[:N_SG] = (B_t + B_t_Δt) / 2

            # Components w.r.t. Lie algebra basis.
            a_t = self.G.lie_algebra_components(A_t)

            optimizer.zero_grad()
            batch_loss = loss(self(R_t, t, 2 * Δt), a_t)
            losses[i] = float(batch_loss.cpu().item())
            batch_loss.backward()
            optimizer.step()
        return losses.mean()

    @property
    def parameter_count(self):
        return sum(p.numel() for p in self.parameters())


class FlowFieldM3(nn.Module):
    """
    Model for flow matching[1] on the position-orientation space `M3` over SE(3)
    mav exponential curves.[2][3]

    Args:
        `m3`: instance of `M3`.
      Optional:
        `H`: width of the network: number of channels. Defaults to 64.
        `L`: depth of the network: number of layers - 2. Defaults to 2.

    References:
        [1]: Y. Lipman, R.T.Q. Chen, H. Ben-Hami, M. Nickel, and M. Le.
          "Flow Matching for Generative Modeling." arXiv preprint (2022).
          DOI:10.48550/arXiv.2210.02747.
        [2]: F.M. Sherry and B.M.N. Smets. "Flow Matching on Lie Groups." arXiv
          preprint (2025).
          DOI:10.48550/arXiv.2504.00494.
        [3]: G. Bellaard and B.M.N. Smets. "Roto-Translation Invariant Metrics
          on Position-Orientation Space." arXiv preprint (2025).
          DOI:10.48550/arXiv.2504.03309.
    """

    def __init__(self, m3: M3, H=64, L=2):
        super().__init__()
        self.m3 = M3()
        self.network = nn.Sequential(
            nn.Linear(m3.mat_dim + 1, H),
            nn.ReLU(),
            *(
                L
                * (
                    nn.Linear(H, H),
                    nn.ReLU(),
                )
            ),
            nn.Linear(H, m3.se3.dim),
        )

    def forward(self, p_t, t):
        return self.network(torch.cat((p_t.flatten(-2, -1), t), dim=-1))

    def step(self, p_t, t, Δt):
        t = t.view(1, 1).expand(p_t.shape[0], 1)
        return self.m3.act(self.m3.se3.exp(Δt * self(p_t, t)), p_t)

    def step_back(self, p_t, t, Δt):
        t = t.view(1, 1).expand(p_t.shape[0], 1)
        return self.m3.act(self.m3.se3.exp(-Δt * self(p_t, t)), p_t)

    def train_network(self, device, train_loader, optimizer, loss):
        self.train()
        N_batches = len(train_loader)
        losses = np.zeros(N_batches)
        for i, (p_0, p_1) in tqdm(
            enumerate(train_loader),
            total=N_batches,
            desc="Training",
            dynamic_ncols=True,
            unit="batch",
            leave=False,
        ):
            t = torch.rand(len(p_1), 1).to(device)
            p_0, p_1 = p_0.to(device), p_1.to(device)
            A_t = self.m3.get_mav_generator(p_0, p_1)
            g_t = self.m3.se3.exp(t * A_t)
            p_t = self.m3.act(g_t, p_0)
            optimizer.zero_grad()
            batch_loss = loss(self(p_t, t), A_t)
            losses[i] = float(batch_loss.cpu().item())
            batch_loss.backward()
            optimizer.step()
        return losses.mean()

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
        `L`: depth of the network: number of layers - 2. Defaults to 2.

    References:
        [1]: Y. Lipman, R.T.Q. Chen, H. Ben-Hami, M. Nickel, and M. Le.
          "Flow Matching for Generative Modeling." arXiv preprint (2022).
          DOI:10.48550/arXiv.2210.02747.
        [2]: F.M. Sherry and B.M.N. Smets. "Flow Matching on Lie Groups." arXiv
          preprint (2025).
          DOI:10.48550/arXiv.2504.00494.
    """

    def __init__(self, G: Group, embed_dim=256, num_heads=8, expansion=4, L=2):
        super().__init__()
        self.G = G

        self.network = nn.Sequential(
            nn.Linear(G.dim + 1, embed_dim),
            *(L * (EncoderBlock(embed_dim, num_heads, expansion),)),
            nn.Linear(embed_dim, G.dim),
        )

    def forward(self, g_t, t):
        return self.network(torch.cat((g_t, t), dim=-1))

    def step(self, g_t, t, Δt):
        t = t.view(1, 1, 1).expand(*g_t.shape[:-1], 1)
        return self.G.L(g_t, self.G.exp(Δt * self(g_t, t)))

    def step_back(self, g_t, t, Δt):
        t = t.view(1, 1, 1).expand(*g_t.shape[:-1], 1)
        return self.G.L(g_t, self.G.exp(-Δt * self(g_t, t)))

    def train_network(self, device, train_loader, optimizer, loss):
        self.train()
        N_batches = len(train_loader)
        losses = np.zeros(N_batches)
        for i, (g_0, g_1) in tqdm(
            enumerate(train_loader),
            total=N_batches,
            desc="Training",
            dynamic_ncols=True,
            unit="batch",
            leave=False,
        ):
            t = torch.rand(len(g_1), 1, 1).expand(*g_1.shape[:-1], 1).to(device)
            g_0, g_1 = g_0.to(device), g_1.to(device)
            A_t = self.G.log(self.G.L_inv(g_0, g_1))
            g_t = self.G.L(g_0, self.G.exp(t * A_t))
            optimizer.zero_grad()
            batch_loss = loss(self(g_t, t), A_t)
            losses[i] = float(batch_loss.cpu().item())
            batch_loss.backward()
            optimizer.step()
        return losses.mean()

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

    def __init__(self, embed_dim, num_heads, expansion):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, expansion * embed_dim),
            nn.ReLU(),
            nn.Linear(expansion * embed_dim, embed_dim),
        )

    def forward(self, x):
        A, _ = self.attn(x, x, x, need_weights=False)
        x = self.ln1(A + x)
        return self.ln2(x + self.ff(x))

    @property
    def parameter_count(self):
        return sum(p.numel() for p in self.parameters())


class LogarithmicDistance(nn.Module):
    def __init__(self, w):
        super().__init__()
        self.w = w
        self.ρ = lambda A_1, A_2: self.ρ_c_normalised(A_1, A_2, w)

    def ρ_c_normalised(self, A_1, A_2, w):
        return (w**2 * (A_2 - A_1) ** 2).mean()

    def forward(self, pred, target):
        return self.ρ(pred, target)

    def __repr__(self):
        return f"LogarithmicDistance{tuple(self.w.numpy())}"


class PermutationInvariantLogarithmicDistance(nn.Module):
    def __init__(self, w):
        super().__init__()
        self.w = w
        self.ρ = lambda A_1, A_2: self.ρ_c_normalised(A_1, A_2, w)

    def ρ_c_normalised(self, A_1, A_2, w):
        return (w**2 * (A_2 - A_1) ** 2).mean()

    def forward(self, pred, target):
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
