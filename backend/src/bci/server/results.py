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
            if has_inter:
                row["csp_inter_acc"] = _f(r.get("intersession_acc"))
                row["csp_inter_kappa"] = _f(r.get("intersession_kappa"))

    # 2) compare_methods_<id>.csv: matriz 2×2 por sujeto (acc; sin kappa).
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
    "csp_inter_acc", "csp_inter_kappa",
    "csp_cross_acc",
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

    return {
        "id": dataset,
        "label": meta.get("label", dataset),
        "role": meta.get("role", "training"),
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
            "role": r.get("role"),
            "n": r["n_subjects_evaluated"],
            "cells": cells,
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

    significance = {}
    w = _wilcoxon(pooled["csp_within_acc"], pooled["eegnet_within_acc"])
    c = _wilcoxon(pooled["csp_cross_acc"], pooled["eegnet_cross_acc"])
    significance = {k: v for k, v in {"within": w, "cross": c}.items() if v}

    return {
        "matrix": matrix,
        "summary": summary,
        "significance": significance,
        "per_dataset": per_dataset,
        "n_datasets": sum(1 for d in per_dataset if d["n"] > 0),
    }
