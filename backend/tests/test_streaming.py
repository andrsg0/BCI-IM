"""Tests del Hito 6: filtrado causal con estado y simulación de streaming."""
from __future__ import annotations

import numpy as np

from bci.pipeline.offline import MotorImageryPipeline
from bci.streaming.simulator import CausalFIR, StreamSimulator


# --------------------------------------------------------------------------- #
# FIR causal con estado
# --------------------------------------------------------------------------- #
def test_causal_fir_chunks_equal_whole():
    """Filtrar por chunks == filtrar todo de golpe (sin saltos en las fronteras)."""
    rng = np.random.default_rng(0)
    x = rng.standard_normal((3, 1000))
    h = rng.standard_normal(31)

    fir = CausalFIR(h, n_channels=3)
    out = np.concatenate([fir.process_chunk(x[:, i:i + 25]) for i in range(0, 1000, 25)], axis=1)

    # Referencia: filtrado causal de toda la señal de una vez.
    fir2 = CausalFIR(h, n_channels=3)
    ref = fir2.process_chunk(x)
    assert np.allclose(out, ref, atol=1e-10)


def test_causal_fir_matches_numpy_causal():
    """El FIR causal coincide con np.convolve recortado a la parte causal."""
    rng = np.random.default_rng(1)
    x = rng.standard_normal(500)
    h = rng.standard_normal(21)
    fir = CausalFIR(h, n_channels=1)
    out = fir.process_chunk(x[None, :])[0]
    ref = np.convolve(x, h)[:len(x)]            # y[n]=Σ h[k]x[n-k], causal
    assert np.allclose(out, ref, atol=1e-10)


def test_causal_fir_reset():
    h = np.array([0.5, 0.5])
    fir = CausalFIR(h, n_channels=1)
    fir.process_chunk(np.ones((1, 5)))
    fir.reset()
    assert np.allclose(fir.buf, 0.0)


# --------------------------------------------------------------------------- #
# Simulador de streaming (con un pipeline entrenado en datos sintéticos)
# --------------------------------------------------------------------------- #
_CFG = {
    "fir_filter": {"low_hz": 8, "high_hz": 30, "fs": 250, "num_taps": 101, "window": "hamming"},
    "csp": {"n_components": 4, "log_variance": True},
}


def _synthetic_eeg(n_per_class=40, n_ch=8, n_time=1000, seed=0):
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


def test_stream_produces_predictions():
    X, y = _synthetic_eeg()
    pipe = MotorImageryPipeline(_CFG, fs=250).fit(X, y)
    sim = StreamSimulator(pipe, window_s=2.0, step_s=0.2)
    results = sim.stream(X[0])
    assert len(results) > 0
    r = results[0]
    assert set(r) == {"t", "pred", "probs", "power", "feat", "disc"}
    assert len(r["power"]) == X.shape[1]                   # una potencia por canal
    assert abs(sum(r["probs"].values()) - 1.0) < 1e-6     # softmax suma 1
    assert len(r["feat"]) == _CFG["csp"]["n_components"]   # un rasgo por componente CSP
    assert isinstance(r["disc"], float)                    # proyección discriminante escalar


def test_stream_vote_matches_class():
    """El voto mayoritario del stream debe acertar la clase de trials claros."""
    X, y = _synthetic_eeg(seed=2)
    pipe = MotorImageryPipeline(_CFG, fs=250).fit(X, y)
    sim = StreamSimulator(pipe, window_s=2.0, step_s=0.2)
    correct = 0
    idx = list(range(0, 80, 7))               # muestreo de trials de ambas clases
    for i in idx:
        vote, _ = sim.stream_epoch_vote(X[i])
        correct += (vote == y[i])
    assert correct / len(idx) > 0.8
