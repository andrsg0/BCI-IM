"""Ensamblado de resultados para el frontend (sección Resultados).

Lee los artefactos que ya producen los scripts de evaluación —CSV por sujeto y
fichas ``ModelCard`` (.json)— y los consolida en una estructura estándar que la
API expone. **No** recalcula nada: solo recolecta lo que hay en disco, así que la
página refleja siempre la última evaluación sin re-entrenar.

Grados de completitud (campo ``status``):
  - ``measured`` : existe la comparación 2×2 completa (CSP+LDA vs EEGNet,
                   within vs cross) — hoy via ``compare_methods_<id>.csv``.
  - ``partial``  : solo hay within-subject de CSP+LDA (k-fold) — ``results_<tag>.csv``.
  - ``pending``  : no hay artefactos; solo la metadata declarada del dataset.

Diseño: este módulo es PURO (no importa ``server.app``) para evitar ciclos; quien
lo llama le pasa la metadata del REGISTRY, el directorio ``processed`` y las clases.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from bci.pipeline.offline import itr as _itr_raw

# Algunos CSV legados usan un alias en vez del nombre de clase MOABB
# (p. ej. ``results_2a.csv`` para BNCI2014_001). Probamos el id y sus alias.
_CSV_ALIASES: dict[str, list[str]] = {
    "BNCI2014_001": ["BNCI2014_001", "2a"],
}


def _csv_tags(dataset: str) -> list[str]:
    return _CSV_ALIASES.get(dataset, [dataset])


def _find_csv(processed: Path, prefix: str, dataset: str) -> Path | None:
    """Primer CSV ``<prefix>_<tag>.csv`` que exista, probando id y alias."""
    for tag in _csv_tags(dataset):
        p = processed / f"{prefix}_{tag}.csv"
        if p.exists():
            return p
    return None


def _stats(values: list[float]) -> dict | None:
    """Resumen honesto de una métrica: media, desviación y RANGO (no solo media)."""
    arr = np.asarray([v for v in values if v is not None and not np.isnan(v)], dtype=float)
    if arr.size == 0:
        return None
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=0)),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "n": int(arr.size),
    }


def _gini(values: list[float]) -> float | None:
    """Coeficiente de Gini de una lista de accuracies (dispersión inter-sujeto).

    Cuantifica la desigualdad en rendimiento entre sujetos: 0 = todos iguales,
    1 = toda la "riqueza" (accuracy) concentrada en un sujeto. En BCI captura
    el fenómeno de «BCI illiteracy»: si la mitad de los sujetos rinden cerca
    del azar y unos pocos llegan a 0.90, el Gini es alto.

    Usa la fórmula clásica basada en la curva de Lorenz.
    """
    arr = np.asarray([v for v in values if v is not None and not np.isnan(v)], dtype=float)
    if arr.size < 2:
        return None
    arr = np.sort(arr)
    n = arr.size
    index = np.arange(1, n + 1)
    return float((2 * np.sum(index * arr) - (n + 1) * np.sum(arr)) / (n * np.sum(arr)))


def _itr_for_acc(acc: float | None, n_classes: int, trial_time_s: float) -> float | None:
    """ITR (bits/min) para un solo valor de accuracy. None si acc no disponible."""
    if acc is None or np.isnan(acc):
        return None
    return _itr_raw(acc, n_classes, trial_time_s)


def _read_pooled_card(processed: Path, dataset: str) -> dict | None:
    """Ficha del modelo EEGNet pooled (cross-subject), si se entrenó."""
    import json

    p = processed / f"model_{dataset}_s0_eegnet_pooled.json"
    if not p.exists():
        return None
    try:
        card = json.loads(p.read_text())
    except Exception:  # noqa: BLE001
        return None
    extra = card.get("extra", {}) or {}
    return {
        "loso_mean": extra.get("loso_mean"),
        "loso_per_subject": extra.get("loso_per_subject", {}),
        "n_subjects": extra.get("n_subjects"),
        "n_train": card.get("n_train"),
        "epochs": extra.get("epochs"),
        "augment": extra.get("augment"),
        "augment_copies": extra.get("augment_copies"),
        "device": extra.get("device"),
        "trained_on": card.get("trained_on"),
        "n_channels": len(card.get("channels", []) or []),
        "fir": card.get("fir"),
    }


def _wilcoxon(a: list[float], b: list[float]) -> dict | None:
    """Test de Wilcoxon pareado (¿la diferencia entre dos métodos es significativa?).

    Empareja sujeto a sujeto donde ambos valores existen. Devuelve estadístico,
    p-valor y nº de pares; ``None`` si no hay datos suficientes o no hay scipy.
    """
    pairs = [(x, y) for x, y in zip(a, b)
             if x is not None and y is not None
             and not np.isnan(x) and not np.isnan(y)]
    if len(pairs) < 6:  # Wilcoxon no es fiable con muy pocos pares
        return None
    xa = np.array([p[0] for p in pairs])
    yb = np.array([p[1] for p in pairs])
    if np.allclose(xa, yb):
        return {"stat": 0.0, "p": 1.0, "n": len(pairs)}
    try:
        from scipy.stats import wilcoxon
    except Exception:  # noqa: BLE001
        return None
    try:
        stat, p = wilcoxon(xa, yb)
    except Exception:  # noqa: BLE001
        return None
    return {"stat": float(stat), "p": float(p), "n": len(pairs)}


def _per_subject(processed: Path, dataset: str) -> tuple[list[dict], bool, bool]:
    """Fusiona los CSV por sujeto en filas únicas.

    Devuelve ``(filas, tiene_intersession, tiene_compare)``. Cada fila trae las
    métricas disponibles (algunas pueden faltar según el dataset).
    """
    rows: dict[int, dict] = {}

    # 1) results_<tag>.csv: within-subject (k-fold) de CSP+LDA, + inter-sesión si la hay.
    res = _find_csv(processed, "results", dataset)
    has_inter = False
    if res is not None:
        df = pd.read_csv(res)
        has_inter = "intersession_acc" in df.columns
        for _, r in df.iterrows():
            s = int(r["subject"])
            row = rows.setdefault(s, {"subject": s})
            row["n_trials"] = int(r["n_trials"]) if "n_trials" in r else None
            row["csp_within_acc"] = _f(r.get("kfold_acc"))
            row["csp_within_kappa"] = _f(r.get("kfold_kappa"))
            # Sensitivity/specificity (disponibles si el CSV se regeneró).
            row["csp_within_sens"] = _f(r.get("kfold_sens"))
            row["csp_within_spec"] = _f(r.get("kfold_spec"))
            if has_inter:
                row["csp_inter_acc"] = _f(r.get("intersession_acc"))
                row["csp_inter_kappa"] = _f(r.get("intersession_kappa"))
                row["csp_inter_sens"] = _f(r.get("intersession_sens"))
                row["csp_inter_spec"] = _f(r.get("intersession_spec"))

    # 2) compare_methods_<id>.csv: matriz 2×2 por sujeto.
    cmp = _find_csv(processed, "compare_methods", dataset)
    has_compare = cmp is not None
    if has_compare:
        df = pd.read_csv(cmp)
        for _, r in df.iterrows():
            s = int(r["subject"])
            row = rows.setdefault(s, {"subject": s})
            # csp_within ya viene de results; si no, lo tomamos de aquí.
            row.setdefault("csp_within_acc", _f(r.get("csp_within")))
            row["csp_cross_acc"] = _f(r.get("csp_cross"))
            row["eegnet_within_acc"] = _f(r.get("eegnet_within"))
            row["eegnet_cross_acc"] = _f(r.get("eegnet_cross"))
            # Kappa para los 4 regímenes (disponible si CSV regenerado).
            row.setdefault("csp_within_kappa", _f(r.get("csp_within_kappa")))
            row["csp_cross_kappa"] = _f(r.get("csp_cross_kappa"))
            row["eegnet_within_kappa"] = _f(r.get("eegnet_within_kappa"))
            row["eegnet_cross_kappa"] = _f(r.get("eegnet_cross_kappa"))

    return [rows[k] for k in sorted(rows)], has_inter, has_compare


def _f(v) -> float | None:
    """Castea a float devolviendo None para NaN/ausente (JSON-friendly)."""
    if v is None:
        return None
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return None if np.isnan(x) else x


_METRICS = [
    "csp_within_acc", "csp_within_kappa",
    "csp_within_sens", "csp_within_spec",
    "csp_inter_acc", "csp_inter_kappa",
    "csp_inter_sens", "csp_inter_spec",
    "csp_cross_acc", "csp_cross_kappa",
    "eegnet_within_acc", "eegnet_within_kappa",
    "eegnet_cross_acc", "eegnet_cross_kappa",
]

# Métricas de accuracy (para ITR y Gini, que solo aplican a acc).
_ACC_METRICS = [
    "csp_within_acc", "csp_inter_acc", "csp_cross_acc",
    "eegnet_within_acc", "eegnet_cross_acc",
]


def dataset_results(dataset: str, meta: dict, processed: Path,
                    classes: list[str] | None) -> dict:
    """Estructura completa de resultados de un dataset (para /api/results/{dataset})."""
    subjects, has_inter, has_compare = _per_subject(processed, dataset)
    n_classes = len(classes) if classes else 2
    chance = 1.0 / n_classes if n_classes else 0.5

    summary = {m: _stats([s.get(m) for s in subjects]) for m in _METRICS}
    summary = {k: v for k, v in summary.items() if v is not None}

    # Matriz 2×2 (medias) — celdas que existan.
    def cell(metric: str) -> float | None:
        st = summary.get(metric)
        return st["mean"] if st else None

    matrix = {
        "csp": {"within": cell("csp_within_acc"), "cross": cell("csp_cross_acc")},
        "eegnet": {"within": cell("eegnet_within_acc"), "cross": cell("eegnet_cross_acc")},
    }

    # Significancia entre métodos (Wilcoxon pareado sobre sujetos).
    significance = {}
    if has_compare:
        w = _wilcoxon([s.get("csp_within_acc") for s in subjects],
                      [s.get("eegnet_within_acc") for s in subjects])
        c = _wilcoxon([s.get("csp_cross_acc") for s in subjects],
                      [s.get("eegnet_cross_acc") for s in subjects])
        significance = {k: v for k, v in {"within": w, "cross": c}.items() if v}

    if has_compare:
        status = "measured"
    elif subjects:
        status = "partial"
    else:
        status = "pending"

    # --- ITR (bits/min) por método/escenario, usando streaming.window_s del YAML ---
    # Lee el YAML del dataset para obtener el tiempo por decisión.
    trial_time_s = 2.0  # default conservador
    try:
        from bci.config import load_config, BACKEND_ROOT
        cfg_path = meta.get('config')
        if cfg_path:
            from pathlib import Path as _P
            cp = _P(cfg_path)
            if not cp.is_absolute():
                cp = BACKEND_ROOT.parent / cp
            cfg = load_config(cp)
            trial_time_s = float(cfg.get('streaming', {}).get('window_s', 2.0))
    except Exception:  # noqa: BLE001
        pass

    itr_matrix = {
        "csp": {
            "within": _itr_for_acc(cell("csp_within_acc"), n_classes, trial_time_s),
            "cross": _itr_for_acc(cell("csp_cross_acc"), n_classes, trial_time_s),
        },
        "eegnet": {
            "within": _itr_for_acc(cell("eegnet_within_acc"), n_classes, trial_time_s),
            "cross": _itr_for_acc(cell("eegnet_cross_acc"), n_classes, trial_time_s),
        },
    }

    # --- Kappa matrix (espejo de la accuracy matrix) ---
    kappa_matrix = {
        "csp": {"within": cell("csp_within_kappa"), "cross": cell("csp_cross_kappa")},
        "eegnet": {"within": cell("eegnet_within_kappa"), "cross": cell("eegnet_cross_kappa")},
    }

    # --- Gini coefficient (dispersión inter-sujeto por método) ---
    gini = {m: _gini([s.get(m) for s in subjects]) for m in _ACC_METRICS}
    gini = {k: v for k, v in gini.items() if v is not None}

    return {
        "id": dataset,
        "label": meta.get("label", dataset),
        "sessions": int(meta.get("sessions", 1)),
        "live": int(meta.get("sessions", 1)) >= 2,
        "fs": meta.get("fs"),
        "n_subjects_declared": meta.get("subjects"),
        "n_subjects_evaluated": len(subjects),
        "classes": classes or [],
        "chance": chance,
        "status": status,
        "has_intersession": has_inter,
        "has_compare": has_compare,
        "subjects": subjects,
        "summary": summary,
        "matrix": matrix,
        "itr": itr_matrix,
        "kappa_matrix": kappa_matrix,
        "gini": gini,
        "trial_time_s": trial_time_s,
        "significance": significance,
        "pooled": _read_pooled_card(processed, dataset),
    }


def dataset_summary(dataset: str, meta: dict, processed: Path,
                    classes: list[str] | None) -> dict:
    """Versión ligera para el índice (/api/results), sin la tabla por sujeto."""
    full = dataset_results(dataset, meta, processed, classes)
    full.pop("subjects", None)
    return full


def aggregate_methods(datasets: list[tuple[str, dict, Path, list[str] | None]]) -> dict:
    """Vista GENERAL: compara los métodos agregando sobre TODOS los datasets.

    En vez de "¿cómo rinde el método en el dataset X?", responde "¿cómo rinde
    CSP+LDA vs EEGNet, within vs cross, sobre toda la población?". Agrupa los
    valores **por sujeto** de cada dataset en un solo conjunto (un sujeto = un
    punto), así que pondera naturalmente por nº de sujetos. Expone también el
    desglose por dataset para que la agregación sea honesta (no esconde la
    heterogeneidad entre poblaciones).
    """
    pooled: dict[str, list[float]] = {m: [] for m in _METRICS}
    per_dataset: list[dict] = []

    for dataset, meta, processed, classes in datasets:
        r = dataset_results(dataset, meta, processed, classes)
        rows = r.get("subjects", []) or []
        cells: dict[str, float] = {}
        for m in _METRICS:
            vals = [s.get(m) for s in rows]
            pooled[m].extend(vals)
            st = _stats(vals)
            if st:
                cells[m] = st["mean"]
        per_dataset.append({
            "id": dataset,
            "label": r["label"],
            "live": r.get("live"),
            "n": r["n_subjects_evaluated"],
            "cells": cells,
            "gini": r.get("gini"),
            "itr": r.get("itr"),
        })

    summary = {m: _stats(pooled[m]) for m in _METRICS}
    summary = {k: v for k, v in summary.items() if v is not None}

    def cell(metric: str) -> float | None:
        st = summary.get(metric)
        return st["mean"] if st else None

    matrix = {
        "csp": {"within": cell("csp_within_acc"), "cross": cell("csp_cross_acc")},
        "eegnet": {"within": cell("eegnet_within_acc"), "cross": cell("eegnet_cross_acc")},
    }

    # Gini global (pooled de todos los sujetos por método).
    gini = {m: _gini(pooled[m]) for m in _ACC_METRICS}
    gini = {k: v for k, v in gini.items() if v is not None}

    significance = {}
    w = _wilcoxon(pooled["csp_within_acc"], pooled["eegnet_within_acc"])
    c = _wilcoxon(pooled["csp_cross_acc"], pooled["eegnet_cross_acc"])
    significance = {k: v for k, v in {"within": w, "cross": c}.items() if v}

    return {
        "matrix": matrix,
        "summary": summary,
        "gini": gini,
        "significance": significance,
        "per_dataset": per_dataset,
        "n_datasets": sum(1 for d in per_dataset if d["n"] > 0),
    }
