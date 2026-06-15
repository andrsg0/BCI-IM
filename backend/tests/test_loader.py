"""Tests del loader de datasets.

- Tests rápidos (sintéticos) de la estructura EpochedData: siempre corren.
- Un test de integración que usa la caché procesada (data/processed/*.npz) si
  existe; se omite si aún no se ha ejecutado download_data.py --save.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from bci.datasets.moabb_loader import EpochedData

PROCESSED_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"


def _fake_epoched(n_trials=10, n_ch=22, n_times=1001) -> EpochedData:
    rng = np.random.default_rng(0)
    X = rng.standard_normal((n_trials, n_ch, n_times))
    y = np.array(["left_hand", "right_hand"] * (n_trials // 2))
    meta = pd.DataFrame({
        "subject": [1] * n_trials,
        "session": ["0train"] * n_trials,
        "run": ["0"] * n_trials,
    })
    ch = [f"C{i}" for i in range(n_ch)]
    return EpochedData(X=X, y=y, metadata=meta, ch_names=ch, sfreq=250.0)


def test_epocheddata_shapes():
    d = _fake_epoched()
    assert d.n_trials == 10
    assert d.n_channels == 22
    assert d.n_times == 1001
    assert d.X.shape == (10, 22, 1001)


def test_epocheddata_class_distribution():
    d = _fake_epoched()
    dist = d.class_distribution()
    assert dist == {"left_hand": 5, "right_hand": 5}


def test_epocheddata_summary_runs():
    d = _fake_epoched()
    s = d.summary()
    assert "trials" in s and "250" in s


@pytest.mark.skipif(
    not list(PROCESSED_DIR.glob("*.npz")),
    reason="No hay caché procesada; ejecuta scripts/download_data.py --save",
)
def test_processed_cache_is_valid():
    """Si existe caché del 2a, verifica forma esperada (N, 22, 1001)."""
    npz = sorted(PROCESSED_DIR.glob("BNCI2014_001*.npz"))[0]
    data = np.load(npz, allow_pickle=True)
    X = data["X"]
    assert X.ndim == 3
    assert X.shape[1] == 22, "el 2a tiene 22 canales EEG"
    assert X.shape[2] == 1001, "ventana [2,6]s @250Hz = 1001 muestras"
    assert set(np.unique(data["y"])) <= {"left_hand", "right_hand", "feet", "tongue"}
