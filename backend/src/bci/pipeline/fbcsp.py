"""FBCSP — Filter Bank Common Spatial Patterns (Ang et al., 2008).

EXPERIMENTO (resultado negativo). Esta es la extensión "de libro" del CSP de banda
única, y es 100 % teoría LTI: en vez de un solo FIR pasa-banda µ/β, se usa un BANCO
de filtros FIR (varias sub-bandas) y se aplica CSP en cada una. La idea es que cada
ritmo (p. ej. µ 8–12 vs β 18–26) puede discriminar mejor en su propia banda.

Pipeline:
    señal → [banco de FIR: B sub-bandas] → CSP por banda → log-varianza
          → selección de características (información mutua) → LDA

POR QUÉ ESTÁ AQUÍ Y NO EN PRODUCCIÓN
------------------------------------
Medido con rigor (ver scripts/eval_fbcsp.py), FBCSP **no mejora** al CSP de banda
única en la evaluación honesta **inter-sesión** (generalizar a otro día): de hecho
empeora, porque las muchas sub-bandas + la selección de características **sobreajustan
la sesión de entrenamiento** y no transfieren. Curiosamente, en k-fold (que mezcla
sesiones) a veces sube — justo la señal del sobreajuste. La banda única 8–30, más
simple, generaliza mejor. Por eso el sistema en vivo sigue usando CSP+LDA y FBCSP
queda como experimento documentado (igual que el resultado negativo del shrinkage).
"""
from __future__ import annotations

import numpy as np
from sklearn.feature_selection import mutual_info_classif

from bci.dsp.convolution import apply_filter
from bci.dsp.fir_filters import FIRFilter, design_bandpass_fir
from bci.features.log_variance import log_variance
from bci.models.lda import LDA
from bci.spatial.csp import CSP

# Sub-bandas por defecto: cubren µ y β en pasos de 4 Hz (esquema clásico FBCSP).
DEFAULT_BANDS: list[tuple[float, float]] = [
    (4, 8), (8, 12), (12, 16), (16, 20), (20, 24), (24, 28), (28, 32),
]


class FBCSPPipeline:
    """Banco de filtros FIR → CSP por banda → log-varianza → selección MI → LDA."""

    def __init__(self, fs: float, bands: list[tuple[float, float]] | None = None,
                 num_taps: int = 199, n_components: int = 4, n_select: int = 8,
                 csp_shrinkage: float = 0.0, window: str = "hamming",
                 classification_window: dict | None = None):
        self.fs = fs
        self.bands = bands or DEFAULT_BANDS
        self.n_select = n_select
        # Un FIR (respuesta al impulso h[n]) por sub-banda: el "banco de filtros".
        self.filters: list[FIRFilter] = [
            design_bandpass_fir(lo, hi, fs, num_taps, window) for lo, hi in self.bands
        ]
        # Un CSP por sub-banda (se aprende en fit).
        self.csps: list[CSP] = [CSP(n_components, shrinkage=csp_shrinkage) for _ in self.bands]
        self.lda = LDA()
        self.selected_: np.ndarray | None = None   # índices de características elegidas
        self.mi_: np.ndarray | None = None          # información mutua por característica
        self._crop = None
        if classification_window:
            self._crop = (int(classification_window["tmin_rel"] * fs),
                          int(classification_window["tmax_rel"] * fs))

    @classmethod
    def from_config(cls, cfg: dict, fs: float, **kwargs) -> "FBCSPPipeline":
        return cls(fs=fs, classification_window=cfg.get("classification_window"),
                   n_components=cfg["csp"]["n_components"], **kwargs)

    # --- banco de FIR (filtrado fijo, no se entrena) ---
    def filter_bank(self, X: np.ndarray) -> list[np.ndarray]:
        """Aplica el banco de FIR y recorta la ventana activa. Lista de B arrays.

        El filtrado FIR (convolución) es por trial e independiente del fold, así que
        puede precomputarse una vez y reutilizarse (lo aprovecha el script de eval).
        """
        out = []
        for filt in self.filters:
            Xf = apply_filter(X, filt.h, mode="same")
            if self._crop is not None:
                Xf = Xf[..., self._crop[0]:self._crop[1]]
            out.append(Xf)
        return out

    def _features(self, banded: list[np.ndarray], fit: bool = False, y=None) -> np.ndarray:
        feats = []
        for csp, Xf in zip(self.csps, banded):
            if fit:
                csp.fit(Xf, y)
            feats.append(log_variance(csp.transform(Xf)))
        return np.concatenate(feats, axis=1)     # (n_trials, B * n_components)

    # --- API sobre datos ya pasados por el banco (eficiente para CV) ---
    def fit_banded(self, banded: list[np.ndarray], y: np.ndarray) -> "FBCSPPipeline":
        F = self._features(banded, fit=True, y=y)
        self.mi_ = mutual_info_classif(F, y, random_state=0)   # relevancia por característica
        k = min(self.n_select, F.shape[1])
        self.selected_ = np.argsort(self.mi_)[::-1][:k]        # las K más informativas
        self.lda.fit(F[:, self.selected_], y)
        return self

    def predict_banded(self, banded: list[np.ndarray]) -> np.ndarray:
        F = self._features(banded)
        return self.lda.predict(F[:, self.selected_])

    # --- API sobre la señal cruda ---
    def fit(self, X: np.ndarray, y: np.ndarray) -> "FBCSPPipeline":
        return self.fit_banded(self.filter_bank(X), np.asarray(y))

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.predict_banded(self.filter_bank(X))

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        return float(np.mean(self.predict(X) == np.asarray(y)))
