"""Tests del régimen CROSS-SUBJECT (opción A): split por sujeto + entrenamiento.

No tocan la red: usan datos sintéticos separables por CSP y monkeypatch de
``load_from_config``, de modo que prueban la lógica (no la descarga de MOABB).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from bci.datasets.moabb_loader import EpochedData
from bci.pipeline import training


# --------------------------------------------------------------------------- #
# Datos sintéticos: dos clases con covarianza ESPACIAL distinta (lo que CSP usa).
# clase 0 -> potencia en canales {0,1}; clase 1 -> potencia en canales {2,3}.
# --------------------------------------------------------------------------- #
def _synthetic(subjects=(1, 2, 3), sessions=("s0", "s1"), per_class=30,
               n_ch=8, n_t=256, fs=128.0, seed=0):
    rng = np.random.default_rng(seed)
    X_list, y_list, rows = [], [], []
    for subj in subjects:
        for sess in sessions:
            for cls, hot in [("left_hand", (0, 1)), ("right_hand", (2, 3))]:
                x = rng.standard_normal((per_class, n_ch, n_t)).astype(np.float32)
                x[:, hot, :] *= 3.0                      # sube varianza en esos canales
                X_list.append(x)
                y_list.extend([cls] * per_class)
                rows.extend({"subject": subj, "session": sess, "run": "0"}
                            for _ in range(per_class))
    return EpochedData(
        X=np.concatenate(X_list, axis=0),
        y=np.array(y_list),
        metadata=pd.DataFrame(rows),
        ch_names=[f"C{i}" for i in range(n_ch)],
        sfreq=fs,
    )


def _cfg():
    return {
        "dataset": {"name": "SYNTH", "subjects": [1, 2, 3],
                    "classes": ["left_hand", "right_hand"]},
        "fir_filter": {"low_hz": 8.0, "high_hz": 30.0, "num_taps": 51,
                       "window": "hamming", "fs": 128},
        "csp": {"n_components": 4, "log_variance": True, "shrinkage": 0.1},
        "classifier": {"type": "lda", "cv_folds": 5},
    }


# --------------------------------------------------------------------------- #
# split_train_demo_subject (lógica pura)
# --------------------------------------------------------------------------- #
def test_split_subject_excludes_demo_from_train():
    data = _synthetic()
    idx_train, idx_demo, holdout = training.split_train_demo_subject(data, demo_subject=3)
    subj = data.metadata["subject"].to_numpy()
    # El sujeto demo NO aparece en entrenamiento, y SÍ es exactamente el held-out.
    assert set(subj[idx_train]) == {1, 2}
    assert set(subj[idx_demo]) == {3}
    assert holdout == {"by": "subject", "value": 3}
    # Partición exhaustiva y sin solape.
    assert len(idx_train) + len(idx_demo) == len(data.y)
    assert set(idx_train).isdisjoint(idx_demo)


def test_split_subject_raises_on_missing_subject():
    data = _synthetic(subjects=(1, 2))
    with pytest.raises(ValueError):
        training.split_train_demo_subject(data, demo_subject=99)


def test_split_subject_raises_when_no_train_subjects():
    data = _synthetic(subjects=(7,))            # un único sujeto: nada con que entrenar
    with pytest.raises(ValueError):
        training.split_train_demo_subject(data, demo_subject=7)


# --------------------------------------------------------------------------- #
# train_crosssubject (punta a punta, sin red)
# --------------------------------------------------------------------------- #
def test_train_crosssubject_card_and_accuracy(monkeypatch):
    data = _synthetic(seed=1)
    monkeypatch.setattr(training, "load_from_config", lambda cfg: data)

    pipe, card, out = training.train_crosssubject(_cfg(), "SYNTH", [1, 2, 3], demo_subject=3)

    # Ficha coherente con el régimen cross-subject.
    assert card.method == "csp_lda_cross"
    assert card.subject == 3                       # el held-out
    assert card.holdout == {"by": "subject", "value": 3}
    assert card.train_session is None
    assert card.extra["train_subjects"] == [1, 2]
    assert card.n_demo == int(np.sum(data.metadata["subject"].to_numpy() == 3))
    # Datos separables -> debe clasificar a un sujeto NUEVO muy por encima del azar.
    assert card.accuracy > 0.8
    # El pipeline devuelto predice sobre los datos del held-out.
    held = data.metadata["subject"].to_numpy() == 3
    assert pipe.predict(data.X[held]).shape == (held.sum(),)
