"""Característica log-varianza de las componentes CSP — el puente DSP → ML.

Tras el CSP tenemos las "señales virtuales" Z (n_trials, n_componentes, tiempo).
La información de clase está en su POTENCIA, que para una señal de media cero es su
VARIANZA (recordar la ERD: la imaginación motora cambia la potencia µ/β). Resumimos
cada componente en un único número por trial:

        f_i = log( var(z_i) / Σ_j var(z_j) )

Dos detalles y su porqué:
  - La VARIANZA convierte una serie temporal en una medida de potencia (1 número/comp).
  - El LOG normaliza la distribución (la varianza es muy asimétrica) y la acerca a una
    gaussiana, que es justo lo que asume el clasificador lineal (LDA) del siguiente paso.
  - La normalización por la suma hace la característica robusta a la amplitud global del
    trial (un trial "más fuerte" no debe cambiar la clase).
"""
from __future__ import annotations

import numpy as np


def log_variance(Z: np.ndarray, normalize: bool = True) -> np.ndarray:
    """Extrae la log-varianza de cada componente CSP.

    Parameters
    ----------
    Z
        Señales proyectadas por el CSP: (n_trials, n_componentes, n_tiempo).
    normalize
        Si True (defecto), normaliza cada varianza por la suma de varianzas del
        trial antes del log (fórmula clásica de características CSP).

    Returns
    -------
    np.ndarray
        Matriz de características (n_trials, n_componentes), lista para el clasificador.
    """
    Z = np.asarray(Z, dtype=float)
    var = np.var(Z, axis=-1)                       # potencia por componente: (trials, comp)
    if normalize:
        var = var / var.sum(axis=1, keepdims=True)
    return np.log(var)
