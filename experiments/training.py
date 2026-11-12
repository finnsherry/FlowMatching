"""
Training for the experiment notebooks: loading the configuration of a space,
training a flow matching model with the parameters of a test case, and plotting
its training curves.
"""

import json

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
from tqdm.notebook import tqdm

from lieflow.models import DataGenerator, FlowField, LogarithmicDistance, OTCoupling


def load_config(space, path="config.json"):
    """
    Return the parameters shared by all test cases of `space`, and a dict with
    the parameters of every test case. The default parameters are overridden by
    those of the space, which are in turn overridden by those of the test case.
    """
    with open(path, "r") as f:
        config_all = json.load(f)
    config_space = config_all[space]
    config_shared = config_all["defaults"] | {
        key: value for key, value in config_space.items() if not isinstance(value, dict)
    }
    config = {test: config_shared | value for test, value in config_space.items() if isinstance(value, dict)}
    return config_shared, config


def train_model(space, generate_0, generate_1, config, x_0s_eval, x_1s_eval, floor, device="cpu"):
    """
    Train a flow matching model on `space` with the parameters in `config`, on
    pairs drawn by `generate_0` and `generate_1`, optionally coupled by
    mini-batch optimal transport. After every epoch, evaluate the exponential
    moving average of the weights on the samples `x_0s_eval` and `x_1s_eval`,
    and plot the training curves against the `floor` of the metrics. Return the
    averaged model, and the loss and metrics per epoch.
    """
    w = torch.tensor(config["loss_weights"], device=device)
    coupling = None if config["ot_block_size"] is None else OTCoupling(space, w, config["ot_block_size"])
    train_loader = DataGenerator(generate_0, generate_1, config["batch_size"], config["batches_per_epoch"], coupling=coupling)

    transformer = {key: config[key] for key in ("embed_dim", "num_heads", "expansion") if key in config}
    model_FM = FlowField(space, L=config["depth"], H=config["width"], **transformer).to(device)
    print("Number of parameters: ", model_FM.parameter_count)

    optimizer_FM = torch.optim.Adam(model_FM.parameters(), config["learning_rate"], weight_decay=config["weight_decay"], fused=True)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer_FM, T_max=config["epochs"] * config["batches_per_epoch"], eta_min=config["learning_rate_final"]
    )
    ema = AveragedModel(model_FM, multi_avg_fn=get_ema_multi_avg_fn(config["ema_decay"]))
    loss = LogarithmicDistance(space, w)

    losses_FM = np.zeros(config["epochs"])
    metrics_FM = {key: np.zeros(config["epochs"]) for key in floor}
    for i in tqdm(range(config["epochs"])):
        losses_FM[i] = model_FM.train_epoch(device, train_loader, optimizer_FM, scheduler, ema, loss, show_progress=False)
        for key, value in ema.module.evaluate(x_0s_eval, x_1s_eval, N_steps=config["N_steps"]).items():
            metrics_FM[key][i] = value

    plot_training(losses_FM, metrics_FM, floor)
    return ema.module, losses_FM, metrics_FM


def plot_training(losses, metrics, floor):
    """
    Plot the loss, and the metrics with their floors, against the epoch.
    """
    epochs = np.arange(1, len(losses) + 1)
    fig, ax = plt.subplots(1, 2, figsize=(10, 4.5))
    ax[0].plot(epochs, losses)
    ax[0].set_title("Batch Loss Flow Matching")
    ax[0].set_yscale("log")
    labels = {"W_1": "$W_1$", "precision": "precision", "coverage": "coverage"}
    for (key, label), color in zip(labels.items(), ("tab:blue", "tab:orange", "tab:green")):
        ax[1].plot(epochs, metrics[key], color=color, label=label)
        ax[1].axhline(floor[key], color=color, linestyle="--", linewidth=0.8)
    ax[1].set_title("Distances (dashed: floor)")
    ax[1].set_yscale("log")
    ax[1].legend()
    for a in ax:
        a.set_xscale("log")
        a.set_xlabel("Epoch")
