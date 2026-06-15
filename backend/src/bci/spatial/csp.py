"""CSP (Common Spatial Patterns) — un filtro espacial LINEAL.

ANALOGÍA CON LA TEORÍA LTI
--------------------------
El FIR (Hito 3) es un sistema lineal en el TIEMPO: mezcla muestras vecinas,
y[n] = Σ_k h[k]·x[n-k]. El CSP es un sistema lineal en el ESPACIO: en cada
instante mezcla los canales con pesos fijos,

        z[c, n] = Σ_j  W[c, j] · x[j, n]      (en matricial: Z = W · X)

Donde x es (n_canales, n_tiempo) y W (n_componentes, n_canales) son los filtros
espaciales. No hay convolución temporal: es una combinación lineal de canales.

QUÉ OPTIMIZA
------------
Busca las combinaciones de canales cuya VARIANZA (= potencia, recordar la ERD)
sea máxima para una clase y mínima para la otra. Eso lleva a un problema de
AUTOVALORES GENERALIZADOS:  C1 · w = λ · (C1 + C2) · w,  con C1, C2 las
covarianzas medias de cada clase. Los autovalores λ ∈ [0,1] reparten la varianza:
λ≈1 → componente que "se enciende" en la clase 1; λ≈0 → en la clase 2.

MÉTODO (whitening + diagonalización conjunta de Koles, 1990), paso a paso:
  1. Covarianza media por clase, C1 y C2 (normalizadas por traza).
  2. Cc = C1 + C2.  Whitening: Cc = U Λ Uᵀ  →  P = Λ^{-1/2} Uᵀ  (deja Cc = I).
  3. S1 = P C1 Pᵀ.  Diagonalizar: S1 = B Ψ Bᵀ  (Ψ son los autovalores λ).
  4. Filtros espaciales: W = Bᵀ P. Los autovectores extremos (λ máx y mín) son
     los más discriminativos.
  5. Patrones espaciales (para topomapas): A = pinv(W). Sus columnas dicen cómo
     se proyecta cada fuente sobre el cuero cabelludo.

Resolvemos las descomposiciones simétricas con np.linalg.eigh (LAPACK); eso es
álgebra lineal estándar, no "ocultar" la teoría: los pasos del CSP son explícitos.
"""
from __future__ import annotations

import numpy as np


def _class_covariance(X_class: np.ndarray) -> np.ndarray:
    """Covarianza espacial media de una clase, normalizada por la traza.

    X_class: (n_trials, n_canales, n_tiempo). Para cada trial calculamos
    C = E·Eᵀ / traza(E·Eᵀ) y promediamos. La normalización por traza evita que
    trials con más amplitud global dominen el promedio.
    """
    n_trials, n_ch, _ = X_class.shape
    C = np.zeros((n_ch, n_ch))
    for trial in X_class:
        cov = trial @ trial.T
        tr = np.trace(cov)
        if tr > 0:
            cov = cov / tr
        C += cov
    return C / n_trials


def _shrink(C: np.ndarray, gamma: float) -> np.ndarray:
    """Regulariza una covarianza por SHRINKAGE hacia una identidad escalada.

        C_reg = (1 - γ)·C + γ·(traza(C)/n)·I

    Intuición: con pocos trials, la covarianza empírica C es ruidosa y casi
    singular (autovalores extremos poco fiables) -> el CSP sobreajusta. El shrinkage
    "encoge" C hacia una esfera (la identidad escalada), atenuando esos autovalores
    extremos y volviendo C bien condicionada. γ=0 no cambia nada; γ=1 da una esfera.
    Es la regularización clásica del CSP (Lotte & Guan, 2011).
    """
    if gamma <= 0:
        return C
    n = C.shape[0]
    mu = np.trace(C) / n
    return (1 - gamma) * C + gamma * mu * np.eye(n)


class CSP:
    """Common Spatial Patterns binario (estilo sklearn: fit / transform).

    Parameters
    ----------
    n_components
        Nº de filtros espaciales a conservar. Se toman a pares de los extremos
        del espectro de autovalores (los más discriminativos). Debe ser par.
    shrinkage
        Regularización de las covarianzas de clase (γ ∈ [0, 1]). 0 = sin regularizar;
        valores pequeños (0.05–0.2) estabilizan el CSP cuando hay pocos trials por
        clase respecto al nº de canales. Ver ``_shrink``.

    Attributes
    ----------
    filters_ : (n_components, n_canales)
        Filtros espaciales W (las filas proyectan la señal: Z = W·X).
    patterns_ : (n_canales, n_components)
        Patrones espaciales A = pinv(W); sus columnas se dibujan como topomapa.
    eigenvalues_ : (n_components,)
        Autovalor λ de cada filtro conservado (cercano a 1 o a 0 = discriminativo).
    classes_ : (2,)
        Las dos etiquetas, en el orden usado (clase 0 ↔ λ alto, clase 1 ↔ λ bajo).
    """

    def __init__(self, n_components: int = 4, shrinkage: float = 0.0):
        if n_components % 2 != 0:
            raise ValueError("n_components debe ser par (se toman pares de extremos).")
        if not 0.0 <= shrinkage <= 1.0:
            raise ValueError("shrinkage debe estar en [0, 1].")
        self.n_components = n_components
        self.shrinkage = shrinkage
        self.filters_: np.ndarray | None = None
        self.patterns_: np.ndarray | None = None
        self.eigenvalues_: np.ndarray | None = None
        self.classes_: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "CSP":
        """Aprende los filtros espaciales a partir de epochs etiquetados.

        X: (n_trials, n_canales, n_tiempo).  y: (n_trials,) con DOS clases.
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        classes = np.unique(y)
        if len(classes) != 2:
            raise ValueError(f"CSP binario: se requieren 2 clases, hay {len(classes)}.")
        self.classes_ = classes

        # 1) Covarianzas medias por clase (+ regularización por shrinkage).
        C1 = _shrink(_class_covariance(X[y == classes[0]]), self.shrinkage)
        C2 = _shrink(_class_covariance(X[y == classes[1]]), self.shrinkage)
        Cc = C1 + C2

        # 2) Whitening de la covarianza compuesta: Cc = U Λ Uᵀ -> P = Λ^{-1/2} Uᵀ.
        eigvals, U = np.linalg.eigh(Cc)
        # Descartamos direcciones casi nulas (rango deficiente) por estabilidad.
        tol = 1e-10 * eigvals.max()
        keep = eigvals > tol
        eigvals, U = eigvals[keep], U[:, keep]
        P = np.diag(1.0 / np.sqrt(eigvals)) @ U.T          # (rango, n_canales)

        # 3) Llevamos C1 al espacio blanqueado y lo diagonalizamos.
        S1 = P @ C1 @ P.T
        lam, B = np.linalg.eigh(S1)                         # lam ascendente
        # Ordenamos de mayor a menor autovalor (mayor varianza de clase 0 primero).
        order = np.argsort(lam)[::-1]
        lam, B = lam[order], B[:, order]

        # 4) Filtros espaciales en el espacio original: W = Bᵀ P.
        W_full = B.T @ P                                    # (rango, n_canales)

        # 5) Selección de los componentes extremos: m/2 de arriba + m/2 de abajo.
        m = self.n_components
        idx = list(range(m // 2)) + list(range(len(lam) - m // 2, len(lam)))
        self.filters_ = W_full[idx]                         # (n_components, n_canales)
        self.eigenvalues_ = lam[idx]
        # Patrones espaciales (pseudo-inversa de los filtros) para visualización.
        self.patterns_ = np.linalg.pinv(self.filters_)     # (n_canales, n_components)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Proyecta los epochs por los filtros espaciales: Z = W·X.

        Devuelve (n_trials, n_components, n_tiempo): las "señales virtuales" CSP.
        La extracción de características (log-varianza) se hace aparte (Hito 5).
        """
        if self.filters_ is None:
            raise RuntimeError("Llama a fit() antes de transform().")
        X = np.asarray(X, dtype=float)
        # Producto matricial por trial: (comp, canales) @ (canales, tiempo).
        return np.einsum("cj,tjn->tcn", self.filters_, X)

    def fit_transform(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        return self.fit(X, y).transform(X)
