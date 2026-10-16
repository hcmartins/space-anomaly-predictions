"""Algorithm 5.1, Stage 2: staged fusion training (Section 5.5.1).

Stage 1a/1b (independent domain pre-training) are NOT this module's job --
they are `encoders.solar_encoder.load_solar_encoder` /
`encoders.martian_encoder.load_martian_encoder` loading the two real,
already-trained, already-frozen checkpoints. This module implements only
Stage 2: independent batch sampling (Section 5.5's "asynchronous data
handling" -- no sample-level pairing between domains), the frozen-encoder
forward pass, Eq. (5.9)'s gradient update, early stopping, and
best-checkpoint restore.
"""
from __future__ import annotations

import time

import numpy as np
import torch

from fusion.model import CrossDomainFusionModel

from .objective import compute_class_weights, total_objective


def _sample_batch(X: np.ndarray, y: np.ndarray, batch_size: int, rng: np.random.Generator) -> tuple[torch.Tensor, torch.Tensor]:
    idx = rng.choice(len(X), size=min(batch_size, len(X)), replace=False)
    return torch.tensor(X[idx], dtype=torch.float32), torch.tensor(y[idx], dtype=torch.float32)


def train_fusion(
    model: CrossDomainFusionModel,
    X_solar_train: np.ndarray, y_solar_train: np.ndarray,
    X_martian_train: np.ndarray, y_martian_train: np.ndarray,
    X_solar_val: np.ndarray, y_solar_val: np.ndarray,
    X_martian_val: np.ndarray, y_martian_val: np.ndarray,
    *,
    training_config: dict,
    seed: int,
    use_fusion_pathway: bool = True,
    other_domain_embedding: str = "real",
) -> tuple[CrossDomainFusionModel, list[dict], str, float]:
    """Returns (model, training_history, convergence_status, runtime_seconds).

    Only `model.trainable_parameters()` (projections + attention scorers +
    heads) ever receives a gradient -- the frozen encoders' parameters
    have requires_grad=False by construction
    (encoders/frozen_encoder.py::FrozenSequenceEncoder.__init__), so
    optimizing only trainable_parameters() here makes that guarantee
    explicit in this module too, not just relied on implicitly. This is
    the precise mechanism behind Table 5.1's gradient-flow claims.

    `use_fusion_pathway`/`other_domain_embedding` are passed straight
    through to every `model(...)` forward call (training and validation
    steps alike) -- see fusion/model.py::CrossDomainFusionModel.forward for
    what they mean. Both default to A7's ("full_model") real behavior, so
    every existing caller is unaffected."""
    start_time = time.perf_counter()
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    optimizer = torch.optim.Adam(model.trainable_parameters(), lr=training_config["learning_rate"])
    solar_class_weights = compute_class_weights(torch.tensor(y_solar_train, dtype=torch.float32))

    n_S = training_config["batch_size_solar"]
    n_M = training_config["batch_size_martian"]
    steps_per_epoch = max(1, min(len(X_solar_train) // n_S, len(X_martian_train) // n_M))
    val_available = len(X_solar_val) > 0 and len(X_martian_val) > 0

    history: list[dict] = []
    best_val_score = -np.inf
    best_state = None
    epochs_without_improvement = 0
    convergence_status = "completed_all_epochs"

    for epoch in range(training_config["epochs"]):
        model.train()
        epoch_loss_terms = {"total": 0.0, "loss_solar": 0.0, "loss_martian": 0.0, "alignment_loss": 0.0, "balance_loss": 0.0}
        for _ in range(steps_per_epoch):
            x_solar, y_solar = _sample_batch(X_solar_train, y_solar_train, n_S, rng)
            x_martian, y_martian = _sample_batch(X_martian_train, y_martian_train, n_M, rng)

            optimizer.zero_grad()
            out = model(x_solar, x_martian, use_fusion_pathway=use_fusion_pathway, other_domain_embedding=other_domain_embedding)
            losses = total_objective(
                out, y_solar, y_martian,
                solar_class_weights=solar_class_weights,
                trainable_parameters=model.trainable_parameters(),
                lambda_align=training_config["lambda_align"],
                lambda_bal=training_config["lambda_bal"],
                lambda_wd=training_config["lambda_wd"],
            )
            losses["total"].backward()
            optimizer.step()
            for key in epoch_loss_terms:
                epoch_loss_terms[key] += losses[key].item()
        epoch_loss_terms = {k: v / steps_per_epoch for k, v in epoch_loss_terms.items()}

        val_score = float("nan")
        if val_available:
            model.eval()
            with torch.no_grad():
                out_val = model(
                    torch.tensor(X_solar_val, dtype=torch.float32), torch.tensor(X_martian_val, dtype=torch.float32),
                    use_fusion_pathway=use_fusion_pathway, other_domain_embedding=other_domain_embedding,
                )
                val_losses = total_objective(
                    out_val,
                    torch.tensor(y_solar_val, dtype=torch.float32), torch.tensor(y_martian_val, dtype=torch.float32),
                    solar_class_weights=solar_class_weights,
                    trainable_parameters=model.trainable_parameters(),
                    lambda_align=training_config["lambda_align"],
                    lambda_bal=training_config["lambda_bal"],
                    lambda_wd=training_config["lambda_wd"],
                )
                # Higher-is-better validation criterion (negative total
                # loss) -- a real deployment would swap in each domain's
                # own TSS-based skill score; this module deliberately has
                # no dependency on either domain's own metrics package
                # (see README's decoupling decision).
                val_score = -val_losses["total"].item()

        history.append({"epoch": epoch, **epoch_loss_terms, "val_score": val_score})

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

    _update_prototypes(model, X_solar_train, X_martian_train)

    runtime_seconds = time.perf_counter() - start_time
    return model, history, convergence_status, runtime_seconds


def _update_prototypes(model: CrossDomainFusionModel, X_solar_train: np.ndarray, X_martian_train: np.ndarray) -> None:
    """Records the training-set mean PROJECTED embedding per domain
    (configs/fusion.yaml's inference_missing_domain_policy=fixed_prototype)
    -- used by predict_solar_only/predict_martian_only when the other
    domain's live data is unavailable at inference."""
    model.eval()
    with torch.no_grad():
        if len(X_solar_train):
            h_solar_proj = model.projection_solar(model.solar_encoder(torch.tensor(X_solar_train, dtype=torch.float32)))
            model.solar_prototype.copy_(h_solar_proj.mean(dim=0))
        if len(X_martian_train):
            h_martian_proj = model.projection_martian(
                model.martian_encoder(torch.tensor(X_martian_train, dtype=torch.float32))
            )
            model.martian_prototype.copy_(h_martian_proj.mean(dim=0))
