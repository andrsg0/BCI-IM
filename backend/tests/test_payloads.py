"""Tests de los payloads de visualización precomputados (bci.server.payloads).

Garantías que fijan:
  - cada ``build_*`` produce la estructura que el frontend espera.
  - ``save_payload``/``load_payload`` hacen un roundtrip byte-idéntico (lo precomputado
    y lo calculado al vuelo son indistinguibles — el servidor puede servir cualquiera).

Usa datos sintéticos separables por CSP (mismo enfoque que test_training), sin red ni
descargas de MOABB.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from bci.datasets.moabb_loader import EpochedData
from bci.pipeline.offline import MotorImageryPipeline
from bci.server import payloads as pl


def _synthetic(per_class=40, n_ch=8, n_t=256, fs=128.0, seed=0):
    rng = np.random.default_rng(seed)
    X_list, y_list, rows = [], [], []
    for sess in ("0train", "1test"):
        for cls, hot in [("left_hand", (0, 1)), ("right_hand", (2, 3))]:
            x = rng.standard_normal((per_class, n_ch, n_t)).astype(np.float32)
            x[:, hot, :] *= 3.0
            X_list.append(x)
            y_list.extend([cls] * per_class)
            rows.extend({"subject": 1, "session": sess, "run": "0"} for _ in range(per_class))
    return EpochedData(
        X=np.concatenate(X_list, axis=0),
        y=np.array(y_list),
        metadata=pd.DataFrame(rows),
        ch_names=[f"C{i}" for i in range(n_ch)],
        sfreq=fs,
    )


def _cfg():
    return {
        "dataset": {"name": "SYNTH", "classes": ["left_hand", "right_hand"]},
        "fir_filter": {"low_hz": 8.0, "high_hz": 30.0, "num_taps": 51,
                       "window": "hamming", "fs": 128},
        "csp": {"n_components": 4, "log_variance": True, "shrinkage": 0.1},
        "classifier": {"type": "lda", "cv_folds": 5},
    }


def _fitted():
    data = _synthetic()
    sess = data.metadata["session"].to_numpy()
    idx_train = np.where(sess == "0train")[0]
    idx_demo = np.where(sess == "1test")[0]
    pipe = MotorImageryPipeline(_cfg(), fs=data.sfreq).fit(data.X[idx_train], data.y[idx_train])
    return data, pipe, idx_train, idx_demo


def test_build_csp_payload_structure():
    data, pipe, idx_train, _ = _fitted()
    p = pl.build_csp_payload(data, pipe, idx_train)
    assert len(p["eigenvalues"]) == 4
    assert np.array(p["patterns"]).shape == (4, 8)        # (n_componentes, n_canales)
    assert len(p["features"]) == len(idx_train) == len(p["labels"])
    assert len(p["lda_disc"]) == len(idx_train)
    assert set(p["pos2d"]) == set(data.ch_names)


def test_build_lda_payload_separable_high_acc():
    data, pipe, idx_train, idx_demo = _fitted()
    p = pl.build_lda_payload(data, pipe, idx_train, idx_demo, _cfg(), card=None)
    assert p["n_eval"] == len(idx_demo)
    assert p["accuracy"] > 0.8                            # datos separables
    assert np.array(p["confusion"]["matrix"]).shape == (2, 2)
    assert len(p["weights"]) == 4


def test_build_csp_signal_all_components():
    data, pipe, _, _ = _fitted()
    p = pl.build_csp_signal_all(data, pipe)
    assert p["n_components"] == 4
    assert len(p["by_component"]) == 4
    c0 = p["by_component"][0]
    assert len(c0["raw"]) == len(c0["csp"]) == len(c0["t"])


def test_payload_roundtrip_byte_identical(tmp_path):
    data, pipe, idx_train, _ = _fitted()
    built = pl.build_csp_payload(data, pipe, idx_train)
    pl.save_payload(tmp_path, "SYNTH", 1, "csp_lda", "csp", built)
    loaded = pl.load_payload(tmp_path, "SYNTH", 1, "csp_lda", "csp")
    assert json.dumps(built) == json.dumps(loaded)


def test_load_payload_missing_returns_none(tmp_path):
    assert pl.load_payload(tmp_path, "SYNTH", 1, "csp_lda", "csp") is None
