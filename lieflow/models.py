"""
models
======

Implementation of flow matching[1] on Lie groups over exponential curves,[2]
on powers of Lie groups, and on homogeneous spaces of matrix groups, such as
position-orientation space M3 of SE(3) with mav exponential curves,[3] and
event-velocity space P3 of SE^+(1, 2).
The model is implemented in the `FlowField` class.
`DataGenerator` draws the training pairs, optionally coupled by mini-batch
optimal transport with `OTCoupling`.

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

import torch
from torch import nn
from tqdm.auto import tqdm

from lieflow.groups import PowerSpace, Space
from lieflow.metrics import compare


class FlowField(nn.Module):
    """
    Model for flow matching[1] over exponential curves[2] on a Lie group or a
    homogeneous space, such as position-orientation space `M3` with mav
    exponential curves.[3] The model only uses the exponential curves of the
    space, through its methods `generator_components` and `flow`. The network
    is an MLP on the flattened points, or, if `space` is a `PowerSpace`, a
    transformer with one token per flattened factor, which is equivariant under
    permutations of the factors.

    Args:
        `space`: a `Space`, i.e. a group of type `Group` or `MatrixGroup`, a
          `HomogeneousSpace`, or a `ProductSpace` or `PowerSpace` of these.
      Optional:
        `L`: depth of the network: number of hidden layers of the MLP, which
          are the same `ResidualBlock`, or number of encoder blocks of the
          transformer, which are the same `EncoderBlock`. Defaults to 2.
        `H`: width of the MLP: number of channels. Defaults to 64.
        `embed_dim`: dimension of the linear embedding of the tokens of the
          transformer. Defaults to 256.
        `num_heads`: number of heads in the multihead attention of the
          transformer. Defaults to 8.
        `expansion`: expansion factor in the pointwise feedforward network of
          the transformer. Defaults to 4.

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
        space: Space,
        L: int = 2,
        H: int = 64,
        embed_dim: int = 256,
        num_heads: int = 8,
        expansion: int = 4,
    ):
        super().__init__()
        self.space = space
        self.tokens = isinstance(space, PowerSpace)
        if self.tokens:
            self.network = _transformer(
                space.G.point_size + 1, space.G.generator_dim, embed_dim, num_heads, expansion, L
            )
        else:
            self.network = _mlp(space.point_size + 1, space.generator_dim, H, L)

    def forward(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Return the predicted generator components at the points `x_t` of shape
        [N, ...] at the times `t`, with one time per point.
        """
        if self.tokens:
            x_t = x_t.flatten(start_dim=2)
            t = t.reshape(len(t), 1, 1).expand(*x_t.shape[:-1], 1)
            return self.network(torch.cat((x_t, t), dim=-1))
        return self.network(torch.cat((x_t.flatten(start_dim=1), t.reshape(len(t), 1)), dim=-1))

    def velocity(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Return the components of the flow field at `x_t` at the time `t`.
        """
        return self(x_t, t.reshape(1).expand(len(x_t)))

    def step(
        self, x_t: torch.Tensor, t: torch.Tensor, Δt: torch.Tensor, method: str = "heun"
    ) -> torch.Tensor:
        """
        Take a step of size `Δt` from time `t`, with the explicit Euler method
        (`method="euler"`) or Heun's method (`method="heun"`): with `v_1` the
        velocity at `x_t` and `v_2` the velocity at the Euler prediction, follow
        the exponential curve with the average `(v_1 + v_2) / 2`. Since the
        exponential curves are exact for constant generators, this is
        second-order accurate. Take a negative `Δt` to integrate backwards.
        """
        v = self.velocity(x_t, t)
        if method == "heun":
            v = (v + self.velocity(self.space.flow(x_t, v, Δt), t + Δt)) / 2
        elif method != "euler":
            raise ValueError(f"Unknown integration method {method}!")
        return self.space.flow(x_t, v, Δt)

    @torch.no_grad()
    def sample(
        self, x_0: torch.Tensor, N_steps: int = 120, method: str = "heun"
    ) -> torch.Tensor:
        """
        Integrate the flow from samples `x_0` at t = 0 to t = 1 in `N_steps`
        steps, with the explicit Euler method (`method="euler"`) or Heun's
        method (`method="heun"`), which costs two evaluations of the network per
        step but is second-order accurate.
        """
        self.eval()
        Δt = torch.tensor([1.0 / N_steps], device=x_0.device, dtype=x_0.dtype)
        x_t = x_0
        for i in range(N_steps):
            t = torch.tensor([i / N_steps], device=x_0.device, dtype=x_0.dtype)
            x_t = self.step(x_t, t, Δt, method)
        return x_t

    def train_epoch(
        self, device, train_loader, optimizer, scheduler, ema, loss, show_progress=True
    ) -> float:
        """
        Train for one epoch, and return the mean batch loss. `train_loader`
        yields batches of pairs `(g_0, g_1)`, e.g. a `DataLoader` of fixed
        pairs; if it has a method `blocks`, such as `DataGenerator`, the
        training pairs are computed for blocks of several batches at once, which
        is much faster for small batches. After every batch, step the learning
        rate `scheduler`, and update the exponential moving average `ema` of the
        weights, which is a `torch.optim.swa_utils.AveragedModel` of `self`.
        """
        self.train()
        N_batches = len(train_loader)
        # Kept on the device, so that recording the losses does not synchronise every batch.
        losses = torch.zeros(N_batches, device=device)
        progress = (
            tqdm(total=N_batches, desc="Training", dynamic_ncols=True, unit="batch", leave=False)
            if show_progress
            else None
        )
        if hasattr(train_loader, "blocks"):
            blocks = train_loader.blocks()
        else:
            blocks = ((g_0, g_1, len(g_1)) for g_0, g_1 in train_loader)
        i = 0
        for g_0, g_1, batch_size in blocks:
            x_t, t, target = self.training_pairs(g_0.to(device), g_1.to(device))
            for x_t_batch, t_batch, target_batch in zip(
                x_t.split(batch_size), t.split(batch_size), target.split(batch_size)
            ):
                optimizer.zero_grad()
                batch_loss = loss(x_t_batch, self(x_t_batch, t_batch), target_batch)
                losses[i] = batch_loss.detach()
                batch_loss.backward()
                optimizer.step()
                scheduler.step()
                ema.update_parameters(self)
                i += 1
                if progress is not None:
                    progress.update()
        if progress is not None:
            progress.close()
        return float(losses.mean().item())

    def training_pairs(
        self, x_0: torch.Tensor, x_1: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Return the points `x_t` at uniformly random times `t` on the exponential
        curves from `x_0` to `x_1`, the times `t`, and the components `a` of the
        generators of the curves, which are the regression targets.
        """
        a = self.space.generator_components(x_0, x_1)
        t = torch.rand(len(x_1), *(a.ndim - 1) * [1], device=x_1.device, dtype=x_1.dtype)
        return self.space.flow(x_0, a, t), t, a

    @torch.no_grad()
    def evaluate(
        self,
        x_0: torch.Tensor,
        x_1: torch.Tensor,
        w: torch.Tensor | None = None,
        N_steps: int = 120,
        method: str = "heun",
    ) -> dict[str, float]:
        """
        Compute the Wasserstein-1 distance, precision, and coverage between
        samples generated by integrating the flow from `x_0`, and target samples
        `x_1`.

        Args:
            `x_0`: initial samples of shape [N, ...].
            `x_1`: target samples of shape [N, ...].
          Optional:
            `w`: weights of the generator, as in the space's
            `generator_norm_squared`. Defaults to all ones.
            `N_steps`: number of integration steps. Defaults to 120.
            `method`: integration method, `"euler"` or `"heun"`. Defaults to
            `"heun"`.
        """
        x_1_pred = self.sample(x_0, N_steps=N_steps, method=method)
        return compare(self.space, x_1_pred, x_1, w)

    @property
    def parameter_count(self):
        return sum(p.numel() for p in self.parameters())


def _transformer(
    in_dim: int,
    out_dim: int,
    embed_dim: int,
    num_heads: int,
    expansion: int,
    L: int,
) -> nn.Sequential:
    """
    Return a transformer acting on tokens of dimension `in_dim`: a linear
    embedding, `L` encoder blocks, which are the same `EncoderBlock`, and a
    linear output layer. Without positional encoding, it is equivariant under
    permutations of the tokens.
    """
    return nn.Sequential(
        nn.Linear(in_dim, embed_dim),
        # The same block L times, so the blocks share their weights.
        *(L * [EncoderBlock(embed_dim, num_heads, expansion)]),
        nn.Linear(embed_dim, out_dim),
    )


class EncoderBlock(nn.Module):
    """
    Simple transformer encoder block.

    Args:
        `embed_dim`: dimension of the linear embedding of the tokens.
        `num_heads`: number of heads in the multihead attention.
        `expansion`: expansion factor in the pointwise feedforward network.
    """
    
    def __init__(self, embed_dim: int, num_heads: int, expansion: int):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, expansion * embed_dim),
            nn.ReLU(),
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
) -> nn.Sequential:
    """
    Return an MLP with `L` hidden layers of width `H`, which are the same
    `ResidualBlock`, followed by a layer normalisation and ReLU activation
    before the output layer.
    """
    return nn.Sequential(
        nn.Linear(in_dim, H),
        # The same block L times, so the blocks share their weights.
        *(L * [ResidualBlock(H)]),
        nn.LayerNorm(H),
        nn.ReLU(),
        nn.Linear(H, out_dim),
    )


class ResidualBlock(nn.Module):
    """
    Pre-activation residual block `h + W ReLU(LayerNorm(h)) + b`. The weights `W`
    and bias `b` are initialised to zero, so that the block starts out as the
    identity. Since the hidden layers of the MLPs share their weights, a stack of
    these blocks is a forward Euler discretisation of the autonomous ODE
    `dh/ds = W act(LayerNorm(h)) + b`.

    Args:
        `H`: width of the block.
    """

    def __init__(self, H: int):
        super().__init__()
        self.norm = nn.LayerNorm(H)
        self.activation = nn.ReLU()
        self.linear = nn.Linear(H, H)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return h + self.linear(self.activation(self.norm(h)))

    @property
    def parameter_count(self):
        return sum(p.numel() for p in self.parameters())


class LogarithmicDistance(nn.Module):
    """
    Flow matching loss: the squared norm `space.generator_norm_squared` at the
    points `x_t` of the difference of the predicted and target generators,
    weighted by `w`, averaged over the samples and divided by the number of
    components. For groups, this is the mean of the squared differences of the
    components weighted by `w**2`.
    """

    def __init__(self, space: Space, w: torch.Tensor | None = None):
        super().__init__()
        self.space = space
        self.w = w

    def forward(self, x_t: torch.Tensor, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.space.generator_norm_squared(x_t, target - pred, self.w).mean() / self.space.generator_dim

    def __repr__(self):
        return f"LogarithmicDistance{() if self.w is None else tuple(self.w.tolist())}"


class DataGenerator:
    """
    Draws new pairs (g_0, g_1) for every batch. The samples are drawn for
    `batches_per_block` batches at once, which is much faster for small batches:
    `blocks` yields these blocks, iterating yields single batches.

    Optionally, `coupling`, e.g. an `OTCoupling`, re-pairs the targets to the
    sources within each batch; by default, sources and targets are independent.
    """
    def __init__(
        self,
        generate_g_0,
        generate_g_1,
        batch_size,
        batches_per_epoch,
        batches_per_block=16,
        coupling=None,
    ):
        self.generate_g_0 = generate_g_0
        self.generate_g_1 = generate_g_1
        self.batch_size = batch_size
        self.batches_per_epoch = batches_per_epoch
        self.batches_per_block = batches_per_block
        self.coupling = coupling
        if coupling is not None and batch_size % coupling.block_size != 0:
            raise ValueError(
                f"The batch size {batch_size} is not a multiple of the coupling's block size {coupling.block_size}!"
            )

    def __len__(self):
        return self.batches_per_epoch

    def blocks(self):
        """Yield blocks `(g_0, g_1, batch_size)` of several batches."""
        for start in range(0, self.batches_per_epoch, self.batches_per_block):
            N = min(self.batches_per_block, self.batches_per_epoch - start) * self.batch_size
            g_0, g_1 = self.generate_g_0(N), self.generate_g_1(N)
            if self.coupling is not None:
                # The coupling's blocks divide the batches, so this pairs within batches.
                g_1 = self.coupling(g_0, g_1)
            yield g_0, g_1, self.batch_size

    def __iter__(self):
        for g_0, g_1, batch_size in self.blocks():
            yield from zip(g_0.split(batch_size), g_1.split(batch_size))


class OTCoupling:
    """
    Mini-batch optimal transport coupling.[1, 2] Within consecutive blocks of
    `block_size` samples, the targets are re-paired to the sources according to
    the entropic optimal transport plan for the cost `G.cost_matrices`, the
    squared logarithmic distance, which is the kinetic energy of the
    exponential curve from source to target: the partner of each source is
    drawn from its row of the plan. This makes the exponential curves cross
    less, so that the flow field is easier to learn and to integrate.

    Args:
        `G`: group or homogeneous space with a method `cost_matrices`.
      Optional:
        `w`: weights of the generator, as in the space's
          `generator_norm_squared`. Defaults to all ones.
        `block_size`: number of samples that are coupled together. Defaults to
          128.
        `reg`: entropic regularisation, relative to the mean cost of the block.
          Defaults to 0.05.
        `iters`: number of Sinkhorn iterations. Defaults to 50.

    References:
        [1]: A. Tong, K. Fatras, N. Malkin, G. Huguet, Y. Zhang, J. Rector-Brooks,
          G. Wolf, and Y. Bengio. "Improving and Generalizing Flow-Based
          Generative Models with Minibatch Optimal Transport." Transactions on
          Machine Learning Research (2024). DOI:10.48550/arXiv.2302.00482.
        [2]: A.-A. Pooladian, H. Ben-Hamu, C. Domingo-Enrich, B. Amos, Y. Lipman,
          and R.T.Q. Chen. "Multisample Flow Matching: Straightening Flows with
          Minibatch Couplings." International Conference on Machine Learning
          (2023). DOI:10.48550/arXiv.2304.14772.
    """

    def __init__(self, G, w=None, block_size=128, reg=0.05, iters=50):
        self.G = G
        self.w = w
        self.block_size = block_size
        self.reg = reg
        self.iters = iters

    @torch.no_grad()
    def __call__(self, g_0: torch.Tensor, g_1: torch.Tensor) -> torch.Tensor:
        """Return the targets `g_1` re-paired to the sources `g_0`."""
        B, b = len(g_0) // self.block_size, self.block_size
        C = self.G.cost_matrices(g_0.view(B, b, *g_0.shape[1:]), g_1.view(B, b, *g_1.shape[1:]), self.w)
        P = _sinkhorn(C, self.reg, self.iters).flatten(0, 1)
        # Inverse transform sampling, much faster than `torch.multinomial` for many short rows: the partner of
        # source i is the first j where the cumulative sum of row i exceeds a uniform draw from [0, row sum).
        cumulative = P.cumsum(-1)
        u = torch.rand(len(P), 1, dtype=P.dtype, device=P.device) * cumulative[:, -1:]
        j = torch.searchsorted(cumulative, u, right=True).clamp_max(b - 1).view(B, b)
        return g_1[(j + b * torch.arange(B, device=j.device)[:, None]).flatten()]


def _sinkhorn(C, reg, iters):
    """
    Entropic optimal transport plans with uniform marginals for the cost
    matrices `C` [B, b, b], up to normalisation of the rows. The cost is
    divided by `reg` times its mean, and shifted by its row minima, which the
    scaling absorbs; if this still underflows, fall back to the log domain.
    """
    C = torch.nan_to_num(C, nan=0.0, posinf=0.0)
    E = (C - C.amin(-1, keepdim=True)) / (reg * C.mean((-2, -1), keepdim=True).clamp_min(1e-12))
    K = torch.exp(-E)
    v = torch.ones(*K.shape[:-2], K.shape[-1], 1, dtype=K.dtype, device=K.device)
    for _ in range(iters):
        u = 1.0 / (K @ v)
        v = 1.0 / (K.transpose(-1, -2) @ u)
    P = u * K * v.transpose(-1, -2)
    if torch.isfinite(P).all() and (P.sum(-1) > 0).all():
        return P
    f = torch.zeros(K.shape[:-1], dtype=K.dtype, device=K.device)
    g = torch.zeros_like(f)
    for _ in range(iters):
        f = -torch.logsumexp(g[..., None, :] - E, -1)
        g = -torch.logsumexp(f[..., :, None] - E, -2)
    return torch.softmax(f[..., :, None] + g[..., None, :] - E, -1)
