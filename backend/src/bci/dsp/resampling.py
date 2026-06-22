"""Remuestreo (cambio de frecuencia de muestreo) como cadena LTI explícita.

POR QUÉ ESTÁ AQUÍ
-----------------
Para juntar varios datasets en un solo "pool" de entrenamiento (cross-dataset) hay
que llevarlos a una **frecuencia de muestreo común**, porque vienen a distinta fs
(PhysioNet 160 Hz, Dreyer/Cho 512 Hz, BCI IV 2a 250 Hz). Cambiar la fs NO es tirar
muestras sin más: hay que respetar el teorema de muestreo o aparece *aliasing*
(solapamiento de espectro). Este módulo lo hace con la teoría de la asignatura:

  remuestreo racional fs_in -> fs_out  (factor L/M):
    1) SOBREMUESTREO por L: insertar L-1 ceros entre muestras  (fs pasa a L·fs_in).
       -> en frecuencia aparecen "imágenes" del espectro.
    2) FILTRO PASA-BAJO FIR (anti-imágenes / anti-aliasing) con corte en la menor
       de las dos Nyquist: fc = min(fs_in, fs_out)/2. Ganancia L para compensar la
       energía perdida al meter ceros. (Es una convolución -> LTI puro.)
    3) DIEZMADO por M: quedarse con 1 de cada M muestras  (fs pasa a L·fs_in/M = fs_out).

El filtro se diseña con ``design_lowpass_fir`` (windowed-sinc) y se aplica con la
misma convolución por superposición que usa el resto del pipeline (``convolve``),
aquí en su forma BATCH (todas las filas a la vez) para que sea rápida sobre
(n_trials, n_canales, n_muestras).

Caso particular L=1 (solo diezmar, p. ej. 512 -> 128 Hz): pasa-bajo + tomar 1 de cada M.
"""
from __future__ import annotations

from math import gcd

import numpy as np

from bci.dsp.fir_filters import design_lowpass_fir


def resample_ratio(fs_in: float, fs_out: float) -> tuple[int, int]:
    """Devuelve (L, M) = (up, down) en su forma irreducible para fs_in -> fs_out."""
    a, b = int(round(fs_in)), int(round(fs_out))
    g = gcd(a, b)
    return b // g, a // g     # up = fs_out/g, down = fs_in/g


def _convolve_same_batch(X2d: np.ndarray, h: np.ndarray) -> np.ndarray:
    """Convolución 'same' por superposición sobre cada fila de X2d (forma batch de
    ``bci.dsp.convolution.convolve``): y = Σ_k h[k]·(x desplazado k)."""
    n_rows, T = X2d.shape
    n_h = len(h)
    full = np.zeros((n_rows, T + n_h - 1))
    for k in range(n_h):                  # un término por tap (vectorizado en filas)
        full[:, k:k + T] += h[k] * X2d
    start = (n_h - 1) // 2                 # recorte 'same' (compensa retardo de grupo)
    return full[:, start:start + T]


def resample_lti(
    X: np.ndarray,
    fs_in: float,
    fs_out: float,
    taps_per_phase: int = 20,
    window: str = "hamming",
) -> np.ndarray:
    """Remuestrea ``X`` de ``fs_in`` a ``fs_out`` sobre el ÚLTIMO eje (tiempo).

    ``X`` puede ser (..., T): se procesan todas las filas a la vez. Devuelve el mismo
    nº de filas con la longitud temporal reescalada ≈ T·fs_out/fs_in.

    ``taps_per_phase`` controla la longitud del FIR anti-aliasing: num_taps ≈
    taps_per_phase·max(L,M) (impar). Más taps ⇒ transición más nítida (mejor rechazo
    de aliasing) a más coste.
    """
    up, down = resample_ratio(fs_in, fs_out)
    if up == down:                         # misma fs: nada que hacer
        return np.asarray(X, dtype=float)

    X = np.asarray(X, dtype=float)
    lead, T = X.shape[:-1], X.shape[-1]
    flat = X.reshape(-1, T)

    # 1) SOBREMUESTREO por L: zero-stuffing (insertar L-1 ceros entre muestras).
    if up > 1:
        ups = np.zeros((flat.shape[0], T * up))
        ups[:, ::up] = flat
    else:
        ups = flat
    inter_fs = up * fs_in                  # fs intermedia tras el sobremuestreo

    # 2) FILTRO PASA-BAJO anti-aliasing/anti-imágenes en la fs intermedia.
    cutoff = 0.5 * min(fs_in, fs_out)      # la menor de las dos Nyquist
    num_taps = taps_per_phase * max(up, down)
    if num_taps % 2 == 0:
        num_taps += 1                      # impar -> fase lineal
    lp = design_lowpass_fir(cutoff, inter_fs, num_taps, window=window)
    h = lp.h * up                          # ganancia L (compensa el zero-stuffing)
    filt = _convolve_same_batch(ups, h)

    # 3) DIEZMADO por M: una de cada M muestras.
    out = filt[:, ::down]
    return out.reshape(*lead, out.shape[-1])
