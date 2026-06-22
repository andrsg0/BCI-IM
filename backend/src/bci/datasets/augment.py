"""Aumentación de datos en el espacio de la señal (para entrenar EEGNet).

POR QUÉ
-------
El cuello de botella de las BCI de imaginación motora no es la potencia del modelo,
sino la ESCASEZ de datos etiquetados: cada trial cuesta varios segundos de grabación
y los sujetos se cansan. La aumentación crea variantes plausibles de los trials que ya
tenemos, de modo que la red ve más ejemplos sin grabar a nadie más. Es lo MÁS BARATO
que sube el rendimiento (sobre todo el within-subject; el cross-subject sube algo).

Cada transformación está pensada como una invariancia REAL de la señal EEG:

  1. Desplazamiento temporal  -> la latencia del ERD/ERS varía trial a trial; la
     decisión no debería depender del instante exacto dentro de la ventana.
  2. Ruido gaussiano leve     -> robustez al ruido de sensor/electrodo.
  3. Escalado de amplitud     -> la impedancia y la ganancia cambian entre sesiones
     y montajes; la clase no depende de la amplitud absoluta.

IMPORTANTE (disciplina de no fuga de datos): la aumentación se aplica SOLO a la
partición de ENTRENAMIENTO, nunca a la de test/held-out (si no, inflaríamos la
precisión de forma tramposa). Por eso vive como función pura que el entrenador
llama sobre ``X[train]``.
"""
from __future__ import annotations

import numpy as np


def augment_trials(
    X: np.ndarray,
    y: np.ndarray,
    *,
    copies: int = 2,
    noise_std: float = 0.1,
    max_shift: int = 8,
    scale_range: float = 0.1,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Devuelve (X_aug, y_aug) = originales + ``copies`` variantes aumentadas.

    Parámetros
    ----------
    X : (n_trials, n_canales, n_muestras) ya recortado a la ventana de clasificación.
    y : (n_trials,) etiquetas; se replican tal cual (las transformaciones conservan la clase).
    copies : nº de copias aumentadas por trial (2 ⇒ el set se triplica: 1 original + 2).
    noise_std : desviación del ruido gaussiano, RELATIVA a la std de cada canal del trial.
    max_shift : desplazamiento temporal máximo (en muestras), aleatorio en [-max_shift, max_shift].
    scale_range : escalado de amplitud aleatorio en [1-scale_range, 1+scale_range] por canal.
    seed : semilla para reproducibilidad.
    """
    X = np.asarray(X, dtype=np.float32)
    rng = np.random.default_rng(seed)
    n, c, _ = X.shape
    per_chan_std = X.std(axis=2, keepdims=True)  # (n, c, 1) escala del ruido por canal

    out_X = [X]
    out_y = [np.asarray(y)]
    for _ in range(copies):
        Xa = X.copy()
        # 1) desplazamiento temporal (mismo shift para todo el trial; roll circular)
        if max_shift > 0:
            shifts = rng.integers(-max_shift, max_shift + 1, size=n)
            for i, sh in enumerate(shifts):
                if sh:
                    Xa[i] = np.roll(Xa[i], int(sh), axis=1)
        # 2) ruido gaussiano leve, proporcional a la amplitud de cada canal
        if noise_std > 0:
            Xa = Xa + rng.normal(0.0, noise_std, Xa.shape).astype(np.float32) * per_chan_std
        # 3) escalado de amplitud por canal
        if scale_range > 0:
            s = (1.0 + rng.uniform(-scale_range, scale_range, size=(n, c, 1))).astype(np.float32)
            Xa = Xa * s
        out_X.append(Xa.astype(np.float32))
        out_y.append(np.asarray(y))

    return np.concatenate(out_X, axis=0), np.concatenate(out_y, axis=0)
