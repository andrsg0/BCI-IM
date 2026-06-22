"""Diseño de filtros FIR pasa-banda por el método de VENTANEO (windowed-sinc).

Objetivo: construir la respuesta al impulso h[n] de un filtro que deje pasar la
banda µ/β (8-30 Hz) y atenúe el resto. Una vez tenemos h[n], filtrar la señal es
simplemente convolucionar (ver bci.dsp.convolution) -> teoría LTI pura.

Idea del método (3 pasos):
  1. Filtro IDEAL: un pasa-banda ideal tiene respuesta al impulso = un sinc
     (infinito y no causal).  h_ideal[n] = 2*f2*sinc(2*f2*n) - 2*f1*sinc(2*f1*n)
     (diferencia de dos pasa-bajos ideales, con f1<f2 normalizadas a fs).
  2. TRUNCAR a N taps -> aparecen rizados (fenómeno de Gibbs).
  3. VENTANEAR: multiplicar por una ventana suave (Hamming) para reducir esos
     lóbulos laterales, a cambio de una transición algo más ancha.

El resultado es SIMÉTRICO -> fase lineal -> retardo constante (N-1)/2 muestras,
sin distorsión de forma. Por eso exigimos num_taps impar.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# --------------------------------------------------------------------------- #
# Ventanas (implementadas a mano para que la fórmula sea explícita).
# n = 0..N-1.  Todas suavizan los extremos de h[n] hacia cero.
# --------------------------------------------------------------------------- #
def _window(name: str, N: int) -> np.ndarray:
    n = np.arange(N)
    if name == "rectangular":
        return np.ones(N)                                   # sin suavizado (solo trunca)
    if name == "hann":
        return 0.5 - 0.5 * np.cos(2 * np.pi * n / (N - 1))
    if name == "hamming":
        return 0.54 - 0.46 * np.cos(2 * np.pi * n / (N - 1))
    if name == "blackman":
        return (0.42 - 0.5 * np.cos(2 * np.pi * n / (N - 1))
                + 0.08 * np.cos(4 * np.pi * n / (N - 1)))
    raise ValueError(f"Ventana desconocida: {name!r}")


@dataclass
class FIRFilter:
    """Un filtro FIR diseñado: sus coeficientes h[n] y sus metadatos."""

    h: np.ndarray            # respuesta al impulso (coeficientes / taps)
    fs: float                # frecuencia de muestreo (Hz)
    low_hz: float
    high_hz: float
    window: str

    @property
    def num_taps(self) -> int:
        return len(self.h)

    @property
    def group_delay(self) -> float:
        """Retardo de grupo (constante por fase lineal), en muestras."""
        return (self.num_taps - 1) / 2

    def is_linear_phase(self, atol: float = 1e-12) -> bool:
        """True si h es simétrica (condición de fase lineal)."""
        return np.allclose(self.h, self.h[::-1], atol=atol)


def design_bandpass_fir(
    low_hz: float,
    high_hz: float,
    fs: float,
    num_taps: int,
    window: str = "hamming",
) -> FIRFilter:
    """Diseña un FIR pasa-banda [low_hz, high_hz] por ventaneo.

    Parameters
    ----------
    low_hz, high_hz
        Bordes de la banda de paso en Hz (p. ej. 8 y 30 para µ/β).
    fs
        Frecuencia de muestreo en Hz.
    num_taps
        Número de coeficientes (longitud de h). DEBE ser impar para fase lineal
        de tipo I y respuesta no nula en la banda.
    window
        Ventana de suavizado: 'hamming' (defecto), 'hann', 'blackman', 'rectangular'.
    """
    if num_taps % 2 == 0:
        raise ValueError("num_taps debe ser impar para un pasa-banda de fase lineal.")
    if not (0 < low_hz < high_hz < fs / 2):
        raise ValueError("Debe cumplirse 0 < low_hz < high_hz < fs/2 (Nyquist).")

    # Frecuencias de corte normalizadas a la de muestreo (ciclos/muestra).
    f1 = low_hz / fs
    f2 = high_hz / fs

    # Eje de muestras centrado en cero: m = n - (N-1)/2  -> filtro simétrico.
    N = num_taps
    m = np.arange(N) - (N - 1) / 2

    # Paso 1: respuesta al impulso IDEAL del pasa-banda = LP(f2) - LP(f1).
    # np.sinc(x) = sin(pi x)/(pi x); el pasa-bajos ideal a fc es 2*fc*sinc(2*fc*m).
    h_ideal = 2 * f2 * np.sinc(2 * f2 * m) - 2 * f1 * np.sinc(2 * f1 * m)

    # Paso 2 + 3: truncado implícito a N taps y ventaneo (suaviza Gibbs).
    w = _window(window, N)
    h = h_ideal * w

    # Normalización: ganancia ~1 en el centro de la banda de paso.
    f_center = 0.5 * (f1 + f2)
    gain = np.sum(h * np.cos(2 * np.pi * f_center * m))  # |H| en f_center
    if abs(gain) > 1e-12:
        h = h / gain

    return FIRFilter(h=h, fs=fs, low_hz=low_hz, high_hz=high_hz, window=window)


def design_lowpass_fir(
    cutoff_hz: float,
    fs: float,
    num_taps: int,
    window: str = "hamming",
) -> FIRFilter:
    """Diseña un FIR pasa-BAJO con corte ``cutoff_hz`` por ventaneo (windowed-sinc).

    Es el filtro ANTI-ALIASING del remuestreo (ver ``bci.dsp.resampling``): deja
    pasar lo que está por debajo de ``cutoff_hz`` y atenúa lo de arriba, para que al
    diezmar no se solape espectro (aliasing). Misma receta que el pasa-banda pero con
    un solo pasa-bajos ideal:

        h_ideal[m] = 2*fc*sinc(2*fc*m)      (fc = cutoff_hz/fs, m centrado en 0)

    Normalizado a ganancia 1 en DC (suma de coeficientes = 1).
    """
    if num_taps % 2 == 0:
        raise ValueError("num_taps debe ser impar para fase lineal tipo I.")
    if not (0 < cutoff_hz < fs / 2):
        raise ValueError("Debe cumplirse 0 < cutoff_hz < fs/2 (Nyquist).")

    fc = cutoff_hz / fs
    N = num_taps
    m = np.arange(N) - (N - 1) / 2
    h_ideal = 2 * fc * np.sinc(2 * fc * m)        # pasa-bajos ideal (sinc)
    h = h_ideal * _window(window, N)              # truncar + ventanear (Gibbs)
    s = np.sum(h)
    if abs(s) > 1e-12:
        h = h / s                                 # ganancia unidad en DC
    return FIRFilter(h=h, fs=fs, low_hz=0.0, high_hz=cutoff_hz, window=window)


def design_from_config(cfg: dict, fs: float | None = None) -> FIRFilter:
    """Diseña el FIR usando la sección 'fir_filter' del YAML.

    Si se pasa ``fs`` (la frecuencia de muestreo REAL del dataset cargado), se usa
    ese en vez del valor del config. Así el mismo config sirve para datasets con
    distinta fs: el filtro se adapta a la señal (no al revés).
    """
    fc = cfg["fir_filter"]
    return design_bandpass_fir(
        low_hz=fc["low_hz"],
        high_hz=fc["high_hz"],
        fs=fs if fs is not None else fc["fs"],
        num_taps=fc["num_taps"],
        window=fc.get("window", "hamming"),
    )
