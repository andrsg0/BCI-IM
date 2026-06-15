"""Tests del Hito 5: log-varianza, LDA y pipeline offline completo."""
from __future__ import annotations

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from bci.features.log_variance import log_variance
from bci.models.lda import LDA
from bci.pipeline.offline import MotorImageryPipeline, evaluate_kfold


# --------------------------------------------------------------------------- #
# log-varianza
# --------------------------------------------------------------------------- #
def test_log_variance_shape_and_finite():
    rng = np.random.default_rng(0)
    Z = rng.standard_normal((20, 4, 300))
    F = log_variance(Z)
    assert F.shape == (20, 4)
    assert np.all(np.isfinite(F))


def test_log_variance_normalized_sums_to_one_before_log():
    rng = np.random.default_rng(1)
    Z = rng.standard_normal((5, 4, 200))
    F = log_variance(Z, normalize=True)
    # exp(F) son fracciones de varianza que deben sumar 1 por trial.
    assert np.allclose(np.exp(F).sum(axis=1), 1.0)


# --------------------------------------------------------------------------- #
# LDA a mano vs sklearn
# --------------------------------------------------------------------------- #
def _blobs(n=200, seed=0):
    rng = np.random.default_rng(seed)
    X0 = rng.standard_normal((n, 3)) + np.array([0, 0, 0])
    X1 = rng.standard_normal((n, 3)) + np.array([2.5, -1.5, 1.0])
    X = np.vstack([X0, X1])
    y = np.array([0] * n + [1] * n)
    return X, y


def test_lda_matches_sklearn():
    X, y = _blobs()
    ours = LDA().fit(X, y)
    ref = LinearDiscriminantAnalysis(solver="lsqr").fit(X, y)
    # Mismas predicciones (covarianza compartida + priors empíricos => equivalentes).
    assert np.array_equal(ours.predict(X), ref.predict(X))


def test_lda_decision_is_linear():
    """La frontera debe ser lineal: δ_k(x) = w·x + b (coef_ e intercept_ definidos)."""
    X, y = _blobs()
    lda = LDA().fit(X, y)
    assert lda.coef_.shape == (2, 3)
    assert lda.intercept_.shape == (2,)
    # decision_function coincide con w·x + b
    manual = X @ lda.coef_.T + lda.intercept_
    assert np.allclose(manual, lda.decision_function(X))


# --------------------------------------------------------------------------- #
# Pipeline completo (sintético) — sin fuga de datos
# --------------------------------------------------------------------------- #
def _synthetic_eeg(n_per_class=50, n_ch=8, n_time=500, seed=0):
    """EEG sintético: clase A con potencia extra en la fuente 0, clase B en la 1."""
    rng = np.random.default_rng(seed)
    mixing = rng.standard_normal((n_ch, n_ch))

    def gen(src):
        out = []
        for _ in range(n_per_class):
            S = rng.standard_normal((n_ch, n_time))
            S[src] *= 3.0
            out.append(mixing @ S)
        return np.array(out)

    X = np.concatenate([gen(0), gen(1)], axis=0)
    y = np.array(["A"] * n_per_class + ["B"] * n_per_class)
    return X, y


_CFG = {
    "fir_filter": {"low_hz": 8, "high_hz": 30, "fs": 250, "num_taps": 101, "window": "hamming"},
    "csp": {"n_components": 4, "log_variance": True},
}


def test_pipeline_fit_predict_separates():
    X, y = _synthetic_eeg()
    pipe = MotorImageryPipeline(_CFG, fs=250).fit(X, y)
    assert pipe.score(X, y) > 0.9


def test_pipeline_kfold_runs_and_separates():
    X, y = _synthetic_eeg()
    res = evaluate_kfold(_CFG, X, y, fs=250, n_splits=5)
    assert res.accuracy > 0.85
    assert res.confusion.sum() == len(y)     # toda muestra evaluada una vez
