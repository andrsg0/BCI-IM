"""Respuesta en frecuencia de un FIR: H(e^jω) = DTFT{h[n]}.

La clave de los sistemas LTI: convolución en el tiempo == multiplicación en
frecuencia. Por eso, para entender QUÉ hace un filtro, miramos su respuesta en
frecuencia, la Transformada de Fourier en Tiempo Discreto de su h[n]:

        H(e^jω) = Σ_n  h[n] · e^{-jωn}

|H(e^jω)| dice cuánto se amplifica/atenúa cada frecuencia; su ángulo, el desfase.
Para nuestro pasa-banda µ/β esperamos |H| ≈ 1 entre 8-30 Hz y ≈ 0 fuera.

Lo evaluamos a mano (suma directa de la DTFT) para que la teoría sea explícita;
los tests verifican que coincide con la FFT (np.fft) salvo error numérico.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class FrequencyResponse:
    """Respuesta en frecuencia muestreada en [0, fs/2]."""

    freqs_hz: np.ndarray     # eje de frecuencias (Hz)
    H: np.ndarray            # H(e^jω) complejo
    fs: float

    @property
    def magnitude(self) -> np.ndarray:
        return np.abs(self.H)

    @property
    def magnitude_db(self) -> np.ndarray:
        return 20 * np.log10(np.abs(self.H) + 1e-12)

    @property
    def phase(self) -> np.ndarray:
        return np.unwrap(np.angle(self.H))

    def gain_at(self, freq_hz: float) -> float:
        """Magnitud de |H| interpolada a una frecuencia dada (Hz)."""
        return float(np.interp(freq_hz, self.freqs_hz, self.magnitude))


def frequency_response(
    h: np.ndarray,
    fs: float,
    n_freqs: int = 512,
) -> FrequencyResponse:
    """Calcula H(e^jω) evaluando la DTFT directamente (suma explícita).

    Parameters
    ----------
    h
        Respuesta al impulso del filtro (coeficientes FIR).
    fs
        Frecuencia de muestreo (Hz).
    n_freqs
        Número de puntos de frecuencia entre 0 y Nyquist (fs/2).
    """
    h = np.asarray(h, dtype=float)
    n = np.arange(len(h))

    # Frecuencias físicas 0..fs/2 y su equivalente angular ω = 2π f / fs.
    freqs_hz = np.linspace(0, fs / 2, n_freqs)
    omega = 2 * np.pi * freqs_hz / fs

    # DTFT explícita: H[m] = Σ_n h[n] e^{-j ω_m n}.  Matriz (n_freqs x len(h)).
    exponent = np.exp(-1j * np.outer(omega, n))
    H = exponent @ h

    return FrequencyResponse(freqs_hz=freqs_hz, H=H, fs=fs)
