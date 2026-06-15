"""LDA (Análisis Discriminante Lineal) — el clasificador LINEAL que cierra la cadena.

Toda la cadena ha sido lineal (FIR en el tiempo, CSP en el espacio); el LDA la
remata con una **frontera de decisión lineal** (un hiperplano).

Modelo: cada clase es una gaussiana con su propia media μ_k pero una MISMA matriz
de covarianza Σ (compartida). Bajo ese supuesto, la regla de Bayes se reduce a una
función discriminante LINEAL en x:

        δ_k(x) = xᵀ Σ⁻¹ μ_k − ½ μ_kᵀ Σ⁻¹ μ_k + log π_k

Se asigna la clase con mayor δ_k. Como los términos cuadráticos en x se cancelan
(la Σ es común), la frontera entre dos clases es un hiperplano -> "lineal".

Lo implementamos a mano para que la frontera lineal sea explícita; los tests
verifican que coincide con sklearn.LinearDiscriminantAnalysis.
"""
from __future__ import annotations

import numpy as np


class LDA:
    """LDA gaussiano con covarianza compartida (estilo sklearn: fit/predict)."""

    def __init__(self):
        self.classes_: np.ndarray | None = None
        self.means_: np.ndarray | None = None        # (n_clases, n_features)
        self.priors_: np.ndarray | None = None        # (n_clases,)
        self.cov_inv_: np.ndarray | None = None        # Σ⁻¹ (n_features, n_features)
        self.coef_: np.ndarray | None = None           # pesos lineales por clase
        self.intercept_: np.ndarray | None = None      # término independiente por clase

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LDA":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        n_samples, n_features = X.shape

        means, priors = [], []
        # Covarianza COMPARTIDA = promedio ponderado de las covarianzas por clase
        # (matriz "within-class" o de dispersión intra-clase).
        Sw = np.zeros((n_features, n_features))
        for c in self.classes_:
            Xc = X[y == c]
            mu = Xc.mean(axis=0)
            means.append(mu)
            priors.append(len(Xc) / n_samples)
            d = Xc - mu
            Sw += d.T @ d
        Sw /= (n_samples - len(self.classes_))         # estimador insesgado

        self.means_ = np.array(means)
        self.priors_ = np.array(priors)
        self.cov_inv_ = np.linalg.pinv(Sw)             # pinv por estabilidad numérica

        # Forma lineal explícita: δ_k(x) = w_kᵀ x + b_k.
        self.coef_ = self.means_ @ self.cov_inv_       # (n_clases, n_features)
        self.intercept_ = (
            -0.5 * np.einsum("ki,ij,kj->k", self.means_, self.cov_inv_, self.means_)
            + np.log(self.priors_)
        )
        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """Devuelve δ_k(x) para cada muestra y clase: (n_muestras, n_clases)."""
        X = np.asarray(X, dtype=float)
        return X @ self.coef_.T + self.intercept_

    def predict(self, X: np.ndarray) -> np.ndarray:
        scores = self.decision_function(X)
        return self.classes_[np.argmax(scores, axis=1)]

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        return float(np.mean(self.predict(X) == np.asarray(y)))
