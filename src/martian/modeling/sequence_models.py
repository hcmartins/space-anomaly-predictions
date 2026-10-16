"""PyTorch LSTM/GRU sequence models -- built and trained ONLY after
modeling/sequence_feasibility.py's run-length report has been produced and
reviewed (see modeling/train.py's run_sequence_task, which always runs the
feasibility report first and logs an infeasible window as a structured
negative result rather than crashing or silently skipping it).

A sequence window is only ever built from a fully unbroken run of
consecutive sols in the trailing-filled series ending at a genuinely
observed anchor sol (never fabricated) -- see `build_sequence_dataset`.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch import nn


class SequenceInfeasibleError(RuntimeError):
    """Raised (and caught by the caller as a structured negative result, not
    a crash) when zero sequence windows can be built for a given window
    length -- e.g. no cell has a long-enough consecutive run."""


def build_sequence_dataset(
    dataset_df: pd.DataFrame,
    features_df: pd.DataFrame,
    feature_columns: list[str],
    *,
    window_length_sols: int,
    target_col: str,
    cell_col: str = "cell_id",
    mars_year_col: str = "mars_year",
    sol_col: str = "sol",
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """For every anchor row in `dataset_df` (already restricted to a
    genuinely observed current sol with a valid, non-fabricated target --
    see modeling/datasets.py), build a (window_length_sols, n_features)
    sequence ending at that anchor sol, ONLY if all window_length_sols
    consecutive sols exist in `features_df` (the trailing-filled series).
    Anchors without a full unbroken window are dropped, never padded with
    fabricated steps.

    A window is dropped ONLY if one of its `window_length_sols` consecutive
    sols is genuinely ABSENT from `features_df` (a real gap, never
    bridgeable). It is NOT dropped merely because some individual feature
    column (e.g. a spatial-neighbour statistic, frequently unavailable under
    the sparse simulated sampling) happens to be NaN at one step -- those
    NaNs are imputed downstream (modeling/train.py), exactly as for the
    tabular models, rather than silently discarding an otherwise perfectly
    good, sol-contiguous window. (An earlier version of this function DID
    drop on any feature NaN, which -- given how often the sparse
    neighbour-stat features are NaN -- silently produced zero usable
    windows even when sol-adjacency was fine; fixed after being caught by
    inspecting a real end-to-end run, see planning/MODELING_DOCUMENTATION.md.)

    Returns (X of shape (n_samples, window_length_sols, n_features), y of
    shape (n_samples,), metadata dataframe with cell_id/mars_year/sol)."""
    lookup: dict[tuple[int, int, int], int] = {}
    for idx, cid, my, sol in features_df[[cell_col, mars_year_col, sol_col]].itertuples(name=None):
        lookup[(int(cid), int(my), int(sol))] = idx

    X_rows: list[np.ndarray] = []
    y_rows: list[float] = []
    meta_rows: list[dict] = []

    for cid, my, sol, target in dataset_df[[cell_col, mars_year_col, sol_col, target_col]].itertuples(index=False, name=None):
        cid, my, sol = int(cid), int(my), int(sol)
        window_sols = [sol - k for k in range(window_length_sols - 1, -1, -1)]
        window_indices = [lookup.get((cid, my, s)) for s in window_sols]
        if any(i is None for i in window_indices):
            continue
        window_values = features_df.loc[window_indices, feature_columns].to_numpy(dtype=float)
        X_rows.append(window_values)
        y_rows.append(float(target))
        meta_rows.append({"cell_id": cid, "mars_year": my, "sol": sol})

    if not X_rows:
        return (
            np.empty((0, window_length_sols, len(feature_columns))),
            np.empty((0,)),
            pd.DataFrame(columns=["cell_id", "mars_year", "sol"]),
        )

    X = np.stack(X_rows, axis=0)
    y = np.array(y_rows, dtype=float)
    meta = pd.DataFrame(meta_rows)
    return X, y, meta


class SequenceModel(nn.Module):
    def __init__(self, *, input_size: int, hidden_size: int, num_layers: int, dropout: float, rnn_type: str):
        super().__init__()
        rnn_cls = nn.LSTM if rnn_type == "lstm" else nn.GRU
        self.rnn = rnn_cls(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.rnn(x)
        last_step = out[:, -1, :]
        return self.head(last_step).squeeze(-1)


def train_sequence_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    *,
    rnn_type: str,
    task_type: str,
    hyperparameters: dict,
    training_config: dict,
    seed: int,
) -> tuple[SequenceModel, list[dict], str, float]:
    """Returns (model, training_history, convergence_status, runtime_seconds)."""
    import time

    torch.manual_seed(seed)

    n_features = X_train.shape[2]
    model = SequenceModel(
        input_size=n_features,
        hidden_size=hyperparameters.get("hidden_size", 16),
        num_layers=hyperparameters.get("num_layers", 1),
        dropout=hyperparameters.get("dropout", 0.0),
        rnn_type=rnn_type,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=training_config.get("learning_rate", 1e-3))
    loss_fn = nn.MSELoss() if task_type == "regression" else nn.BCEWithLogitsLoss()

    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32)
    has_val = len(X_val) > 0
    if has_val:
        X_val_t = torch.tensor(X_val, dtype=torch.float32)
        y_val_t = torch.tensor(y_val, dtype=torch.float32)

    epochs = training_config.get("epochs", 30)
    patience = training_config.get("early_stopping_patience", 5)
    best_val_loss = float("inf")
    epochs_without_improvement = 0
    history = []
    convergence_status = "max_epochs_reached"

    start = time.perf_counter()
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        train_pred = model(X_train_t)
        train_loss = loss_fn(train_pred, y_train_t)
        train_loss.backward()
        optimizer.step()

        entry = {"epoch": epoch, "train_loss": float(train_loss.item())}
        if has_val:
            model.eval()
            with torch.no_grad():
                val_pred = model(X_val_t)
                val_loss = float(loss_fn(val_pred, y_val_t).item())
            entry["val_loss"] = val_loss

            if val_loss < best_val_loss - 1e-6:
                best_val_loss = val_loss
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= patience:
                    history.append(entry)
                    convergence_status = "early_stopped"
                    break
        history.append(entry)
    runtime_seconds = time.perf_counter() - start

    return model, history, convergence_status, runtime_seconds
