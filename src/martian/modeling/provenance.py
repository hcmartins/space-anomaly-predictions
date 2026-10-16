"""Per-trained-model provenance record: exact parameters, random seed,
software versions, training history, checkpoint path, feature order,
train/validation sample counts, convergence status, runtime, and hardware
information. Every model trained anywhere in modeling/ (baseline, classical,
or sequence) writes one of these records -- the schema is uniform across
model kinds so a reviewer never has to learn a different provenance format
per model type.
"""
from __future__ import annotations

import json
import platform
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


def software_versions() -> dict:
    versions = {
        "python": sys.version,
        "platform": platform.platform(),
    }
    for module_name in ("numpy", "pandas", "sklearn", "scipy"):
        try:
            module = __import__(module_name)
            versions[module_name] = getattr(module, "__version__", "unknown")
        except ImportError:
            versions[module_name] = "not installed"
    try:
        import torch

        versions["torch"] = torch.__version__
        versions["torch_cuda_available"] = torch.cuda.is_available()
    except ImportError:
        versions["torch"] = "not installed"
        versions["torch_cuda_available"] = False
    return versions


def hardware_info() -> dict:
    info = {
        "processor": platform.processor(),
        "machine": platform.machine(),
        "system": platform.system(),
        "cpu_count": None,
    }
    try:
        import os

        info["cpu_count"] = os.cpu_count()
    except Exception:
        pass
    try:
        import torch

        info["gpu_available"] = torch.cuda.is_available()
        info["gpu_name"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except ImportError:
        info["gpu_available"] = False
        info["gpu_name"] = None
    return info


@dataclass
class TrainingRecord:
    task_name: str
    model_kind: str  # "baseline" | "classical" | "sequence"
    model_name: str
    horizon_sols: int
    exact_parameters: dict
    random_seed: int
    feature_order: list
    n_train: int
    n_val: int
    convergence_status: str
    runtime_seconds: float
    training_history: list = field(default_factory=list)
    checkpoint_path: str | None = None
    software_versions: dict = field(default_factory=software_versions)
    hardware_info: dict = field(default_factory=hardware_info)
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def write_training_record(path: Path, record: TrainingRecord) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record.to_dict(), indent=2, default=str), encoding="utf-8")
    return path


class Timer:
    """Context manager measuring wall-clock runtime in seconds, for the
    `runtime_seconds` provenance field."""

    def __enter__(self) -> "Timer":
        self._start = time.perf_counter()
        self.elapsed_seconds: float = 0.0
        return self

    def __exit__(self, *exc_info) -> None:
        self.elapsed_seconds = time.perf_counter() - self._start
