"""Pipeline offline completo: FIR → CSP → log-varianza → LDA.

Reúne las cuatro etapas en una clase con fit/predict, reutilizable tal cual en la
simulación en vivo (Hito 6). Incluye dos evaluaciones:

  - evaluate_kfold     : validación cruzada estratificada (mezcla sesiones).
  - evaluate_by_session: entrenar en la sesión '0train' y evaluar en '1test'.
    Es la estimación HONESTA de cómo generalizaría a una sesión nueva (lo que
    pasaría en vivo), porque train y test no comparten día de grabación.

IMPORTANTE (evitar fuga de datos): el CSP y el LDA se ajustan SOLO con los datos de
entrenamiento dentro de cada partición. El FIR no se "entrena" (sus coeficientes son
fijos), así que aplicarlo no introduce fuga.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold

from bci.dsp.convolution import apply_filter
from bci.dsp.fir_filters import FIRFilter, design_from_config
from bci.features.log_variance import log_variance
from bci.models.lda import LDA
from bci.spatial.csp import CSP


class MotorImageryPipeline:
    """Cadena entrenable FIR(fijo) → CSP → log-varianza → LDA.

    Recibe ``fs`` (frecuencia de muestreo REAL del dataset) para diseñar el FIR y
    convertir la ventana de clasificación de segundos a muestras. Así el mismo
    pipeline funciona con datasets de distinta fs (2a=250, PhysioNet=160, …).
    """

    def __init__(self, cfg: dict, fs: float):
        self.cfg = cfg
        self.fs = fs
        self.filt: FIRFilter = design_from_config(cfg, fs=fs)
        self.csp = CSP(
            n_components=cfg["csp"]["n_components"],
            shrinkage=cfg["csp"].get("shrinkage", 0.0),
        )
        self.lda = LDA()
        self._normalize = cfg["csp"].get("log_variance", True)
        # Ventana de clasificación opcional (recorte tras filtrar): (i0, i1) en muestras.
        self._crop = None
        win = cfg.get("classification_window")
        if win:
            self._crop = (int(win["tmin_rel"] * fs), int(win["tmax_rel"] * fs))

    def _features(self, X: np.ndarray, fit_csp: bool = False, y=None) -> np.ndarray:
        """FIR → (recorte ventana activa) → CSP → log-varianza. Ajusta CSP si fit_csp."""
        Xf = apply_filter(X, self.filt.h, mode="same")          # filtrado temporal LTI
        if self._crop is not None:                              # recorte a imaginación activa
            Xf = Xf[..., self._crop[0]:self._crop[1]]
        if fit_csp:
            self.csp.fit(Xf, y)                                  # CSP se aprende en TRAIN
        Z = self.csp.transform(Xf)                              # filtrado espacial
        return log_variance(Z, normalize=self._normalize)       # característica

    def fit(self, X: np.ndarray, y: np.ndarray) -> "MotorImageryPipeline":
        F = self._features(X, fit_csp=True, y=y)
        self.lda.fit(F, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.lda.predict(self._features(X, fit_csp=False))

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        return float(np.mean(self.predict(X) == np.asarray(y)))

    def classify_window(self, Xf_window: np.ndarray):
        """Clasifica UNA ventana YA FILTRADA (n_canales, n_tiempo).

        Pensado para streaming: el filtrado FIR causal se hace fuera (con estado),
        y aquí solo aplicamos CSP → log-varianza → LDA. Devuelve
        ``(clase, prob_dict, info)``, donde ``info`` expone las dos etapas para
        visualizarlas EN VIVO:
          - ``feat`` : vector de log-varianza por componente CSP (lo que produce el
            filtrado espacial; cada número es la potencia de una "señal virtual").
          - ``disc``: proyección sobre la recta discriminante del LDA
            δ(clase₁) − δ(clase₀). Es un único escalar con signo: la FRONTERA está
            en 0; el signo da la clase y la magnitud, la confianza. Sirve para situar
            la ventana respecto a la frontera lineal aprendida.
        """
        Z = self.csp.transform(Xf_window[None, :, :])           # (1, comp, tiempo)
        F = log_variance(Z, normalize=self._normalize)
        scores = self.lda.decision_function(F)[0]
        pred = self.lda.classes_[int(np.argmax(scores))]
        # softmax de las puntuaciones -> pseudo-probabilidades para mostrar confianza
        e = np.exp(scores - scores.max())
        probs = {str(c): float(p) for c, p in zip(self.lda.classes_, e / e.sum())}
        # disc = δ_1 − δ_0: distancia con signo a la frontera lineal (caso binario).
        disc = float(scores[1] - scores[0]) if len(scores) == 2 else float(
            scores.max() - np.sort(scores)[-2])
        info = {"feat": F[0].tolist(), "disc": disc}
        return pred, probs, info


@dataclass
class EvalResult:
    """Métricas de una evaluación."""

    accuracy: float
    kappa: float
    confusion: np.ndarray
    labels: list[str]
    detail: str = ""

    def __str__(self) -> str:
        head = f"  accuracy = {self.accuracy:.3f} | kappa = {self.kappa:.3f}"
        cm = "  matriz de confusión (filas=real, col=pred):\n"
        cm += "    " + "  ".join(f"{l[:6]:>6}" for l in self.labels) + "\n"
        for i, l in enumerate(self.labels):
            cm += f"    {l[:6]:>6} " + " ".join(f"{v:6d}" for v in self.confusion[i]) + "\n"
        extra = f"  ({self.detail})\n" if self.detail else ""
        return f"{head}\n{extra}{cm}"


def _metrics(y_true, y_pred, detail="") -> EvalResult:
    labels = sorted(np.unique(np.concatenate([y_true, y_pred])).tolist())
    return EvalResult(
        accuracy=accuracy_score(y_true, y_pred),
        kappa=cohen_kappa_score(y_true, y_pred),
        confusion=confusion_matrix(y_true, y_pred, labels=labels),
        labels=labels,
        detail=detail,
    )


def evaluate_kfold(cfg: dict, X, y, fs: float, n_splits: int = 5, seed: int = 42) -> EvalResult:
    """Validación cruzada estratificada. CSP+LDA se reajustan en cada fold."""
    y = np.asarray(y)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    y_true, y_pred = [], []
    for tr, te in skf.split(X, y):
        pipe = MotorImageryPipeline(cfg, fs).fit(X[tr], y[tr])
        y_pred.extend(pipe.predict(X[te]))
        y_true.extend(y[te])
    return _metrics(np.array(y_true), np.array(y_pred), f"{n_splits}-fold CV")


def evaluate_by_session(cfg: dict, X, y, metadata, fs: float,
                        train_session="0train", test_session="1test") -> EvalResult:
    """Entrena en una sesión y evalúa en otra (estimación honesta inter-sesión)."""
    y = np.asarray(y)
    sess = metadata["session"].to_numpy()
    tr = sess == train_session
    te = sess == test_session
    pipe = MotorImageryPipeline(cfg, fs).fit(X[tr], y[tr])
    y_pred = pipe.predict(X[te])
    return _metrics(y[te], y_pred, f"train={train_session} → test={test_session}")
