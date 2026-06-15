"""Tests del pipeline FBCSP (banco de filtros → CSP por banda → MI → LDA)."""
import numpy as np

from bci.pipeline.fbcsp import DEFAULT_BANDS, FBCSPPipeline


def _synthetic(seed=0):
    """Datos con potencia dependiente de la clase en bandas/canales distintos.

    Clase 0 → oscilación de 10 Hz (banda µ) en el canal 0.
    Clase 1 → oscilación de 18 Hz (banda β) en el canal 1.
    FBCSP debería separarlas combinando banda + canal.
    """
    rng = np.random.default_rng(seed)
    fs, n, T, C = 128.0, 60, 256, 6
    t = np.arange(T) / fs
    y = np.array([0, 1] * (n // 2))
    X = rng.standard_normal((n, C, T)) * 0.5
    for i in range(n):
        f, ch = (10.0, 0) if y[i] == 0 else (18.0, 1)
        X[i, ch] += 2.0 * np.sin(2 * np.pi * f * t)
    return X, y, fs


def test_fit_predict_shapes_and_labels():
    X, y, fs = _synthetic()
    bands = [(8, 12), (12, 16), (16, 20)]
    pipe = FBCSPPipeline(fs=fs, bands=bands, num_taps=51, n_components=2, n_select=4).fit(X, y)
    # banco devuelve una banda por filtro
    assert len(pipe.filter_bank(X)) == len(bands)
    # se seleccionan exactamente n_select características (de B*n_components disponibles)
    assert pipe.selected_ is not None and len(pipe.selected_) == 4
    pred = pipe.predict(X)
    assert pred.shape == (len(y),)
    assert set(np.unique(pred)).issubset(set(np.unique(y)))


def test_separates_synthetic_classes():
    X, y, fs = _synthetic()
    pipe = FBCSPPipeline(fs=fs, bands=[(8, 12), (16, 20)], num_taps=51,
                         n_components=2, n_select=4).fit(X, y)
    # sobre los datos de entrenamiento debe separar bien la señal sintética
    assert pipe.score(X, y) > 0.8


def test_default_bands_present():
    assert (8, 12) in DEFAULT_BANDS and len(DEFAULT_BANDS) >= 5
