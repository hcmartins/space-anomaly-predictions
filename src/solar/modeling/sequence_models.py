"""GOES sequence LSTM (Chapter 3.6, Table 3.3) -- the first deep-learning
model in this project. Consumes preprocessing/goes_sequences.py's
(T_G, d_G) tensors.

Mirrors martian/modeling/sequence_models.py's real, working PyTorch
LSTM/GRU structure (LSTM -> take last timestep -> Linear(1) head), with one
deliberate addition: `SequenceModel.encode()` exposes the pre-head hidden
state directly -- the h_S Chapter 5's cross-domain fusion Eq. (5.1) needs
-- rather than only the task prediction, which is all martian's current
version exposes..
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .metrics import classification_metrics


class SequenceModel(nn.Module):
    def __init__(self, *, input_size: int, hidden_size: int, num_layers: int, dropout: float):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_size, 1)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Returns h_S = the final LSTM hidden state (T_G's last time step),
        shape (batch, hidden_size) -- NOT the task prediction."""
        out, _ = self.lstm(x)
        return out[:, -1, :]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns P(flare) directly (sigmoid applied here, matching
        Chapter 3.7's p_i notation), shape (batch,)."""
        h = self.dropout(self.encode(x))
        return torch.sigmoid(self.head(h)).squeeze(-1)


def compute_class_weights(y_train: np.ndarray) -> tuple[float, float]:
    """Chapter 3.7's w+ = N/(2*N+), w- = N/(2*N-), computed on TRAIN data
    only -- the standard balanced-class-weight formula the thesis text
    itself suggests as an example."""
    y_train = np.asarray(y_train)
    n = len(y_train)
    n_pos = int(np.sum(y_train == 1))
    n_neg = n - n_pos
    w_pos = n / (2 * n_pos) if n_pos > 0 else 0.0
    w_neg = n / (2 * n_neg) if n_neg > 0 else 0.0
    return w_pos, w_neg


def weighted_binary_cross_entropy(y_prob: torch.Tensor, y_true: torch.Tensor, w_pos: float, w_neg: float) -> torch.Tensor:
    """Chapter 3.7's L_wBCE = -(1/N) sum[w+ y log p + w- (1-y) log(1-p)]."""
    eps = 1e-7
    p = y_prob.clamp(eps, 1 - eps)
    loss = -(w_pos * y_true * torch.log(p) + w_neg * (1 - y_true) * torch.log(1 - p))
    return loss.mean()


def train_sequence_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    *,
    hyperparameters: dict,
    training_config: dict,
    seed: int,
) -> tuple[SequenceModel, list[dict], str, float]:
    """Returns (model, training_history, convergence_status, runtime_seconds).
    Model selection uses the epoch with the BEST validation TSS (never
    train loss alone), consistent with this project's val-only
    threshold/model-selection discipline elsewhere
    (modeling/metrics.py::select_threshold)."""
    start_time = time.perf_counter()
    torch.manual_seed(seed)

    d_G = X_train.shape[2]
    model = SequenceModel(
        input_size=d_G,
        hidden_size=hyperparameters["hidden_size"],
        num_layers=hyperparameters["num_layers"],
        dropout=hyperparameters["dropout"],
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=training_config["learning_rate"])
    w_pos, w_neg = compute_class_weights(y_train)

    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32)
    X_val_t = torch.tensor(X_val, dtype=torch.float32) if len(X_val) else None

    val_has_both_classes = len(X_val) > 0 and len(np.unique(y_val)) >= 2
    batch_size = training_config.get("batch_size", 256)
    n_train = len(X_train_t)

    history: list[dict] = []
    best_val_tss = -np.inf
    best_state = None
    epochs_without_improvement = 0
    convergence_status = "completed_all_epochs"

    for epoch in range(training_config["epochs"]):
        model.train()
        perm = torch.randperm(n_train)
        epoch_loss = 0.0
        for start in range(0, n_train, batch_size):
            idx = perm[start : start + batch_size]
            xb, yb = X_train_t[idx], y_train_t[idx]
            optimizer.zero_grad()
            p = model(xb)
            loss = weighted_binary_cross_entropy(p, yb, w_pos, w_neg)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(idx)
        epoch_loss /= n_train

        val_tss = float("nan")
        if val_has_both_classes:
            model.eval()
            with torch.no_grad():
                val_prob = model(X_val_t).numpy()
            val_tss = classification_metrics(y_val, val_prob)["tss"]

        history.append({"epoch": epoch, "train_loss": epoch_loss, "val_tss": val_tss})

        if not np.isnan(val_tss) and val_tss > best_val_tss:
            best_val_tss = val_tss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= training_config["early_stopping_patience"]:
            convergence_status = f"early_stopped_epoch_{epoch}"
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    elif not val_has_both_classes:
        convergence_status += "_no_validation_selection_available"

    runtime_seconds = time.perf_counter() - start_time
    return model, history, convergence_status, runtime_seconds


def save_sequence_checkpoint(model: SequenceModel, hyperparameters: dict, path: Path) -> None:
    """`hyperparameters` need only contain hidden_size/num_layers/dropout --
    input_size is read directly off the model (nn.LSTM stores it), so the
    caller never has to remember to duplicate it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    architecture = {**hyperparameters, "input_size": model.lstm.input_size}
    torch.save({"state_dict": model.state_dict(), "hyperparameters": architecture}, path)


def load_sequence_checkpoint(path: Path) -> SequenceModel:
    checkpoint = torch.load(path, weights_only=True)
    hp = checkpoint["hyperparameters"]
    model = SequenceModel(
        input_size=hp["input_size"], hidden_size=hp["hidden_size"],
        num_layers=hp["num_layers"], dropout=hp["dropout"],
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model
