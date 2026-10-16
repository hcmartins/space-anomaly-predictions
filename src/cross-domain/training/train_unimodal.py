"""Single-domain training loop for A2 (fusion/unimodal.py::
CapacityMatchedUnimodalModel) -- no cross-domain terms at all (no
alignment loss, no balance regularizer, no other-domain input), matching
configs/ablation.yaml's A2 (`use_fusion_pathway: false,
use_alignment_loss: false`). Mirrors training/train_fusion.py's structure
(early stopping, best-checkpoint restore) but reuses training/objective.py's
existing per-domain loss functions directly rather than duplicating them.
"""
from __future__ import annotations

import time

import numpy as np
import torch

from .objective import compute_class_weights, martian_task_loss, weight_decay_penalty, weighted_binary_cross_entropy


def _sample_batch(X: np.ndarray, y: np.ndarray, batch_size: int, rng: np.random.Generator) -> tuple[torch.Tensor, torch.Tensor]:
    idx = rng.choice(len(X), size=min(batch_size, len(X)), replace=False)
    return torch.tensor(X[idx], dtype=torch.float32), torch.tensor(y[idx], dtype=torch.float32)


def train_unimodal(
    model,
    X_train: np.ndarray, y_train: np.ndarray,
    X_val: np.ndarray, y_val: np.ndarray,
    *,
    domain: str,
    training_config: dict,
    seed: int,
) -> tuple[object, list[dict], str, float]:
    """`domain`: "solar" (weighted BCE, Chapter 3.7's formula) or "martian"
    (plain BCE, matching training/objective.py::martian_task_loss's own
    documented placeholder status). Returns (model, history,
    convergence_status, runtime_seconds) -- same shape as train_fusion's
    return, so callers can treat both uniformly."""
    if domain not in ("solar", "martian"):
        raise ValueError(f"domain must be 'solar' or 'martian', got {domain!r}")

    start_time = time.perf_counter()
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    optimizer = torch.optim.Adam(model.trainable_parameters(), lr=training_config["learning_rate"])
    batch_size = training_config[f"batch_size_{domain}"]
    steps_per_epoch = max(1, len(X_train) // batch_size)
    val_available = len(X_val) > 0

    solar_class_weights = compute_class_weights(torch.tensor(y_train, dtype=torch.float32)) if domain == "solar" else None

    history: list[dict] = []
    best_val_score = -np.inf
    best_state = None
    epochs_without_improvement = 0
    convergence_status = "completed_all_epochs"

    def _task_loss(y_prob: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        if domain == "solar":
            w_pos, w_neg = solar_class_weights
            return weighted_binary_cross_entropy(y_prob, y_true, w_pos, w_neg)
        return martian_task_loss(y_prob, y_true)

    for epoch in range(training_config["epochs"]):
        model.train()
        epoch_loss = 0.0
        for _ in range(steps_per_epoch):
            x_batch, y_batch = _sample_batch(X_train, y_train, batch_size, rng)
            optimizer.zero_grad()
            y_prob = model(x_batch)
            loss = _task_loss(y_prob, y_batch) + training_config["lambda_wd"] * weight_decay_penalty(model.trainable_parameters())
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        epoch_loss /= steps_per_epoch

        val_score = float("nan")
        if val_available:
            model.eval()
            with torch.no_grad():
                y_prob_val = model(torch.tensor(X_val, dtype=torch.float32))
                val_loss = _task_loss(y_prob_val, torch.tensor(y_val, dtype=torch.float32))
                val_score = -val_loss.item()

        history.append({"epoch": epoch, "loss": epoch_loss, "val_score": val_score})

        if not np.isnan(val_score) and val_score > best_val_score:
            best_val_score = val_score
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= training_config["early_stopping_patience"]:
            convergence_status = f"early_stopped_epoch_{epoch}"
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    elif not val_available:
        convergence_status += "_no_validation_selection_available"

    runtime_seconds = time.perf_counter() - start_time
    return model, history, convergence_status, runtime_seconds
