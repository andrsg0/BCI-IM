"""Tests del CSP (Hito 4).

Validan tres cosas:
  1. Matemática: nuestros autovalores coinciden con el problema de autovalores
     GENERALIZADOS de referencia (scipy.linalg.eigh(C1, C1+C2)).
  2. Diagonalización conjunta: los filtros blanquean Cc (W Cc Wᵀ ≈ I) y
     diagonalizan C1 (W C1 Wᵀ ≈ diag(λ)).
  3. Funcional: en datos sintéticos con varianza separada por clase, el CSP
     produce componentes cuya log-varianza distingue las clases.
"""
from __future__ import annotations

import numpy as np
import pytest
import scipy.linalg as sla

from bci.spatial.csp import CSP, _class_covariance


def _make_synthetic(n_per_class=60, n_ch=6, n_time=400, seed=0):
    """Dos clases: la clase A concentra potencia en la fuente 0, la B en la 1.

    Mezclamos fuentes independientes con una matriz fija (modelo lineal del EEG:
    los electrodos ven una mezcla lineal de fuentes corticales).
    """
    rng = np.random.default_rng(seed)
    mixing = rng.standard_normal((n_ch, n_ch))

    def gen(strong_source):
        Xs = []
        for _ in range(n_per_class):
            S = rng.standard_normal((n_ch, n_time))
            S[strong_source] *= 3.0       # fuente dominante para esta clase
            Xs.append(mixing @ S)
        return np.array(Xs)

    XA, XB = gen(0), gen(1)
    X = np.concatenate([XA, XB], axis=0)
    y = np.array(["A"] * n_per_class + ["B"] * n_per_class)
    return X, y


def test_eigenvalues_match_generalized_problem():
    X, y = _make_synthetic()
    csp = CSP(n_components=4).fit(X, y)

    cls = np.unique(y)
    C1 = _class_covariance(X[y == cls[0]])
    C2 = _class_covariance(X[y == cls[1]])
    # Referencia: GEVD C1 w = λ (C1+C2) w  -> autovalores en [0,1].
    ref = np.sort(sla.eigh(C1, C1 + C2, eigvals_only=True))
    ours = np.sort(csp.eigenvalues_)
    # Nuestros 4 componentes son los 2 extremos por cada lado del espectro de ref.
    ref_extremes = np.sort(np.concatenate([ref[:2], ref[-2:]]))
    assert np.allclose(ours, ref_extremes, atol=1e-8)


def test_filters_diagonalize():
    X, y = _make_synthetic()
    csp = CSP(n_components=4).fit(X, y)
    cls = np.unique(y)
    C1 = _class_covariance(X[y == cls[0]])
    Cc = C1 + _class_covariance(X[y == cls[1]])
    W = csp.filters_
    # W blanquea la covarianza compuesta: W Cc Wᵀ ≈ I.
    assert np.allclose(W @ Cc @ W.T, np.eye(len(W)), atol=1e-6)
    # W diagonaliza C1: fuera de la diagonal ≈ 0.
    D = W @ C1 @ W.T
    off = D - np.diag(np.diag(D))
    assert np.allclose(off, 0.0, atol=1e-6)


def test_csp_separates_classes():
    """La log-varianza de las componentes extremas debe separar A de B."""
    X, y = _make_synthetic()
    csp = CSP(n_components=2).fit(X, y)
    Z = csp.transform(X)                      # (trials, 2, tiempo)
    logvar = np.log(np.var(Z, axis=-1))       # (trials, 2)
    # Componente 0 (λ alto) debe tener mayor potencia en la clase A que en la B.
    comp0_A = logvar[y == "A", 0].mean()
    comp0_B = logvar[y == "B", 0].mean()
    assert comp0_A > comp0_B + 0.5


def test_transform_shapes_and_patterns():
    X, y = _make_synthetic(n_ch=6)
    csp = CSP(n_components=4).fit(X, y)
    Z = csp.transform(X)
    assert Z.shape == (X.shape[0], 4, X.shape[2])
    assert csp.filters_.shape == (4, 6)
    assert csp.patterns_.shape == (6, 4)


def test_csp_validation_errors():
    with pytest.raises(ValueError):
        CSP(n_components=3)                    # impar
    with pytest.raises(ValueError):
        CSP(n_components=2, shrinkage=1.5)     # shrinkage fuera de [0,1]
    X, y = _make_synthetic()
    y3 = y.copy(); y3[:5] = "C"
    with pytest.raises(ValueError):
        CSP(n_components=2).fit(X, y3)         # 3 clases


# --------------------------------------------------------------------------- #
# Regularización (shrinkage)
# --------------------------------------------------------------------------- #
def test_shrink_formula():
    """C_reg = (1-γ)C + γ(tr(C)/n)I: con γ=1 da una identidad escalada."""
    from bci.spatial.csp import _shrink
    C = np.array([[4.0, 1.0], [1.0, 2.0]])
    assert np.allclose(_shrink(C, 0.0), C)                 # γ=0 no cambia nada
    mu = np.trace(C) / 2
    assert np.allclose(_shrink(C, 1.0), mu * np.eye(2))    # γ=1 = esfera


def test_shrinkage_zero_equals_unregularized():
    X, y = _make_synthetic()
    a = CSP(4, shrinkage=0.0).fit(X, y).eigenvalues_
    b = CSP(4).fit(X, y).eigenvalues_
    assert np.allclose(a, b)


def test_shrinkage_stabilizes_rank_deficient():
    """Con muchos más canales que muestras la covarianza es singular;
    el shrinkage la vuelve invertible y el CSP entrena sin problemas."""
    rng = np.random.default_rng(0)
    # 8 canales pero solo 3 muestras de tiempo por trial -> covarianza rank<=3.
    n_per = 10
    X = np.concatenate([rng.standard_normal((n_per, 8, 3)),
                        rng.standard_normal((n_per, 8, 3)) * 2.0], axis=0)
    y = np.array([0] * n_per + [1] * n_per)
    csp = CSP(4, shrinkage=0.2).fit(X, y)
    assert csp.filters_.shape == (4, 8)
    assert np.all(np.isfinite(csp.eigenvalues_))
