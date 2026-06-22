"""Convolución discreta — la operación madre de todo sistema LTI.

Un sistema LTI queda totalmente descrito por su respuesta al impulso h[n], y su
salida es SIEMPRE la convolución de la entrada con h[n]:

        y[n] = (x * h)[n] = Σ_k  h[k] · x[n - k]

Este módulo la implementa a mano de tres formas, de la más didáctica a la más
eficiente, todas equivalentes (los tests lo verifican contra numpy):

  1. convolve_mac      -> doble bucle literal, "como en el libro" (lenta, clara).
  2. mac_terms         -> los términos individuales del MAC en una muestra n
                          (para animar la operación Multiplicar-ACumular en el front).
  3. convolve          -> versión vectorizada por SUPERPOSICIÓN (rápida, producción).

NO usamos scipy.signal.lfilter: ese resuelve una ecuación en diferencias (puede ser
IIR). Aquí la convolución es explícita, que es justo lo que el proyecto debe mostrar.
"""
from __future__ import annotations

import numpy as np

_MODES = ("full", "same", "valid")


def _trim(y_full: np.ndarray, len_x: int, len_h: int, mode: str) -> np.ndarray:
    """Recorta la convolución 'full' al modo pedido.

    - 'full' : longitud len_x + len_h - 1 (todas las superposiciones parciales).
    - 'same' : longitud len_x, centrada -> compensa el retardo de grupo del FIR.
    - 'valid': solo donde h se solapa por completo con x (len_x - len_h + 1).
    """
    if mode == "full":
        return y_full
    if mode == "same":
        start = (len_h - 1) // 2
        return y_full[start:start + len_x]
    if mode == "valid":
        return y_full[len_h - 1: len_x]  # asume len_x >= len_h
    raise ValueError(f"mode debe ser uno de {_MODES}, no {mode!r}")


def convolve_mac(x: np.ndarray, h: np.ndarray, mode: str = "full") -> np.ndarray:
    """Convolución LITERAL con doble bucle (didáctica).

    Implementa directamente y[n] = Σ_k h[k]·x[n-k]. Cada iteración interna es una
    operación MAC (multiplicar y acumular). Es O(N·M) en Python puro -> lenta;
    úsala para enseñar/verificar, no para procesar datasets enteros.
    """
    x = np.asarray(x, dtype=float)
    h = np.asarray(h, dtype=float)
    n_x, n_h = len(x), len(h)
    n_y = n_x + n_h - 1
    y = np.zeros(n_y)
    for n in range(n_y):                 # cada muestra de salida...
        acc = 0.0
        for k in range(n_h):             # ...es una suma de productos (MACs)
            j = n - k
            if 0 <= j < n_x:
                acc += h[k] * x[j]       # <-- la operación MAC
        y[n] = acc
    return _trim(y, n_x, n_h, mode)


def mac_terms(x: np.ndarray, h: np.ndarray, n: int) -> dict:
    """Devuelve los términos del MAC que producen la muestra de salida y[n].

    Pensado para la visualización del frontend (ConvolutionViz): muestra qué
    par (h[k], x[n-k]) se multiplica en cada paso y cómo se acumulan hasta y[n].
    """
    x = np.asarray(x, dtype=float)
    h = np.asarray(h, dtype=float)
    n_x, n_h = len(x), len(h)
    ks, xs, hs, prods = [], [], [], []
    for k in range(n_h):
        j = n - k
        if 0 <= j < n_x:
            ks.append(k)
            hs.append(float(h[k]))
            xs.append(float(x[j]))
            prods.append(float(h[k] * x[j]))
    return {
        "n": n,
        "k": ks,                 # índice del tap
        "h_k": hs,               # coeficiente del filtro
        "x_n_minus_k": xs,       # muestra de entrada alineada
        "products": prods,       # h[k]·x[n-k]
        "y_n": float(sum(prods)),  # acumulado final
    }


def convolve(x: np.ndarray, h: np.ndarray, mode: str = "full") -> np.ndarray:
    """Convolución vectorizada por SUPERPOSICIÓN (rápida, para producción).

    Usa la interpretación LTI: la salida es la superposición de copias de la
    respuesta al impulso h, cada una escalada por una muestra de entrada y
    desplazada en el tiempo. Equivale a:

        y = Σ_k  h[k] · (x desplazado k muestras)

    Cada término es una operación vectorial sobre todo x a la vez -> O(M) bucles
    de longitud N en numpy, mucho más rápido que el doble bucle de convolve_mac,
    pero matemáticamente idéntico (verificado en los tests contra np.convolve).
    """
    x = np.asarray(x, dtype=float)
    h = np.asarray(h, dtype=float)
    n_x, n_h = len(x), len(h)
    y = np.zeros(n_x + n_h - 1)
    for k in range(n_h):                 # un término por tap del filtro
        y[k:k + n_x] += h[k] * x         # copia de x escalada por h[k] y desplazada k
    return _trim(y, n_x, n_h, mode)


def apply_filter(X: np.ndarray, h: np.ndarray, mode: str = "same") -> np.ndarray:
    """Aplica el filtro h por convolución sobre el ÚLTIMO eje (el tiempo).

    Soporta señales de cualquier dimensión cuyo último eje sea el tiempo, p. ej.
    (n_trials, n_canales, n_muestras). Con mode='same' la salida conserva la
    longitud temporal y, como el FIR es de fase lineal, queda alineada con la
    entrada (se compensa el retardo de grupo (N-1)/2).
    """
    # Preservamos el dtype de punto flotante de entrada: con float64 (lo habitual)
    # el comportamiento es idéntico, pero si entra float32 (datasets grandes
    # pooled, p. ej. Dreyer2023 87 sujetos) NO duplicamos el array a float64 — eso
    # evita un pico de memoria que disparaba el OOM. Entradas enteras -> float64.
    X = np.asarray(X)
    fdt = X.dtype if np.issubdtype(X.dtype, np.floating) else np.dtype(float)
    X = np.asarray(X, dtype=fdt)
    h = np.asarray(h, dtype=float)
    n_time = X.shape[-1]
    out_len = {"full": n_time + len(h) - 1,
               "same": n_time,
               "valid": n_time - len(h) + 1}[mode]

    flat = X.reshape(-1, n_time)
    out = np.empty((flat.shape[0], out_len), dtype=fdt)
    for i in range(flat.shape[0]):
        out[i] = convolve(flat[i], h, mode=mode)
    return out.reshape(*X.shape[:-1], out_len)
