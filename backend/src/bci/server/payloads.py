"""Constructores de los payloads de visualización del MUNDO OFFLINE + su persistencia.

Idea (portabilidad): la web debe poder mostrar las páginas offline (El Modelo/CSP,
EEGNet, Entrenamiento) SIN datos crudos — solo con artefactos. Para ello, cada payload
de visualización se PRECOMPUTA una vez (``scripts/precompute_payloads.py``) y se guarda
como JSON junto al modelo; el servidor lo sirve tal cual. Los datos crudos solo se
necesitan para el **streaming en vivo** del dataset de demo.

Para no duplicar lógica, las funciones ``build_*`` de aquí son la ÚNICA fuente de verdad:
las usan tanto el script de precompute (con datos cargados) como el servidor (como
respaldo, si el JSON precomputado no existe todavía).

Tipos de payload (``kind``) y a qué endpoint corresponden:
  - ``csp``        -> /api/csp        (patrones/filtros + nube de entrenamiento + LDA disc)
  - ``csp_signal`` -> /api/csp_signal (señal cruda vs componente CSP en el tiempo; TODOS
                       los componentes en un mismo archivo)
  - ``lda``        -> /api/lda        (frontera lineal + confusión sobre el held-out)
  - ``eegnet``     -> /api/eegnet     (respuesta en frecuencia de los filtros aprendidos)
  - ``info``       -> /api/info       (canales, fs, nº de trials, clases)
  - ``positions``  -> /api/positions  (posiciones 2D/3D de los electrodos)
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from bci.dsp.convolution import apply_filter
from bci.dsp.frequency_response import frequency_response


# --- Posiciones de electrodos (montaje 10-20) ------------------------------
def electrode_positions(ch_names):
    """Posiciones de los electrodos: 2D (topomapa) y 3D (cerebro), montaje estándar MNE."""
    import mne

    info = mne.create_info(list(ch_names), 250.0, "eeg")
    info.set_montage("standard_1005", match_case=False, on_missing="ignore")
    ch_pos = info.get_montage().get_positions()["ch_pos"]
    coords = [ch_pos.get(ch) for ch in ch_names]
    scale = max((abs(float(c[i])) for c in coords if c is not None for i in (0, 1)), default=1.0) or 1.0
    pos2d, pos3d = {}, {}
    for ch, c in zip(ch_names, coords):
        if c is None:
            pos2d[ch] = None; pos3d[ch] = None
        else:
            pos3d[ch] = [float(c[0]), float(c[1]), float(c[2])]
            pos2d[ch] = [float(c[0] / scale * 0.85), float(c[1] / scale * 0.85)]
    return pos2d, pos3d


# --- Constructores de payload (fuente única de verdad) ---------------------
def build_info_payload(dataset: str, subject: int, data) -> dict:
    return {
        "dataset": dataset, "subject": subject, "fs": data.sfreq,
        "channels": data.ch_names, "n_trials": data.n_trials,
        "classes": sorted(set(map(str, data.y))),
        "class_distribution": data.class_distribution(),
    }


def build_positions_payload(ch_names) -> dict:
    pos2d, pos3d = electrode_positions(ch_names)
    return {"channels": list(ch_names), "pos2d": pos2d, "pos3d": pos3d}


def build_csp_payload(data, pipe, idx_train: np.ndarray) -> dict:
    """Patrones/filtros CSP + nube de rasgos de ENTRENAMIENTO + proyección LDA.

    La nube (features + lda_disc) es la partición de entrenamiento; el held-out NO se
    incluye, para no mezclar el mundo offline con la demo en vivo."""
    feats = pipe._features(data.X[idx_train])          # (n_train, n_componentes)
    pos2d, pos3d = electrode_positions(data.ch_names)
    scores = pipe.lda.decision_function(feats)
    lda_disc = (scores[:, 1] - scores[:, 0]).tolist() if scores.shape[1] == 2 else \
        (scores.max(axis=1) - np.sort(scores, axis=1)[:, -2]).tolist()
    return {
        "channels": data.ch_names,
        "eigenvalues": pipe.csp.eigenvalues_.tolist(),
        "patterns": pipe.csp.patterns_.T.tolist(),     # (n_componentes, n_canales)
        "filters": pipe.csp.filters_.tolist(),         # W: Z = W·X
        "classes": sorted(set(map(str, data.y))),
        "features": feats.tolist(),
        "labels": [str(c) for c in np.asarray(data.y)[idx_train]],
        "lda_disc": lda_disc,
        "pos2d": pos2d, "pos3d": pos3d,
    }


def build_csp_signal_payload(data, pipe, component: int) -> dict:
    """Señal de UN canal crudo (z-score) vs la salida del componente CSP en el tiempo."""
    n_comp = int(pipe.csp.filters_.shape[0])
    component = max(0, min(component, n_comp - 1))

    patt = np.abs(pipe.csp.patterns_[:, component])
    ch_idx = int(np.argmax(patt))
    ch_name = data.ch_names[ch_idx]

    lam = float(pipe.csp.eigenvalues_[component])
    classes = sorted(set(map(str, data.y)))
    favored = classes[0] if lam >= 0.5 else classes[1]
    y = np.asarray(list(map(str, data.y)))
    cand = np.where(y == favored)[0]
    trial = int(cand[0]) if len(cand) else 0

    X = data.X[trial]
    fs = float(data.sfreq)
    Xf = apply_filter(X[None], pipe.filt.h, mode="same")[0]
    crop = pipe._crop
    if crop is not None:
        Xf = Xf[:, crop[0]:crop[1]]
        Xraw = X[:, crop[0]:crop[1]]
    else:
        Xraw = X
    z = pipe.csp.transform(Xf[None])[0, component, :]
    raw = Xraw[ch_idx]

    def zscore(a):
        a = np.asarray(a, dtype=float)
        s = float(a.std()) or 1.0
        return ((a - a.mean()) / s).tolist()

    return {
        "fs": fs, "component": component, "n_components": n_comp,
        "favored_class": favored, "eigenvalue": lam,
        "channel": ch_name, "trial": trial,
        "t": (np.arange(raw.shape[0]) / fs).tolist(),
        "raw": zscore(raw), "csp": zscore(z),
    }


def build_csp_signal_all(data, pipe) -> dict:
    """Todos los componentes CSP en un mismo archivo (el servidor elige cuál servir)."""
    n_comp = int(pipe.csp.filters_.shape[0])
    return {
        "n_components": n_comp,
        "by_component": [build_csp_signal_payload(data, pipe, c) for c in range(n_comp)],
    }


def build_lda_payload(data, pipe, idx_train: np.ndarray, idx_demo: np.ndarray,
                      cfg: dict, card: dict | None) -> dict:
    """Frontera lineal LDA (global + plano 2D dibujado) + confusión sobre el held-out."""
    from sklearn.metrics import cohen_kappa_score, confusion_matrix

    from bci.models.lda import LDA

    classes = sorted(set(map(str, data.y)))
    y = np.asarray(list(map(str, data.y)))

    lda = pipe.lda
    if lda.coef_.shape[0] == 2:
        w = (lda.coef_[0] - lda.coef_[1])
        b = float(lda.intercept_[0] - lda.intercept_[1])
        positive_class = classes[0]
    else:
        w = lda.coef_[0]
        b = float(lda.intercept_[0])
        positive_class = classes[0]

    feats_tr = pipe._features(data.X[idx_train])
    last = feats_tr.shape[1] - 1
    lda2 = LDA().fit(feats_tr[:, [0, last]], y[idx_train])
    if lda2.coef_.shape[0] == 2:
        w2 = (lda2.coef_[0] - lda2.coef_[1]).tolist()
        b2 = float(lda2.intercept_[0] - lda2.intercept_[1])
    else:
        w2 = lda2.coef_[0].tolist(); b2 = float(lda2.intercept_[0])

    y_true = y[idx_demo]
    y_pred = np.asarray(list(map(str, pipe.predict(data.X[idx_demo]))))
    cm = confusion_matrix(y_true, y_pred, labels=classes).tolist()
    acc = float(np.mean(y_pred == y_true)) if len(y_true) else 0.0
    kappa = float(cohen_kappa_score(y_true, y_pred, labels=classes)) if len(y_true) else 0.0

    clf_cfg = cfg.get("classifier", {})
    holdout = card.get("holdout") if card else None
    if holdout and holdout.get("by") == "session":
        holdout_kind = "inter-sesión"
    elif holdout and holdout.get("by") == "subject":
        holdout_kind = "cross-subject"
    else:
        holdout_kind = "hold-out 30% estratificado"

    return {
        "classes": classes,
        "positive_class": positive_class,
        "weights": w.tolist(),
        "bias": b,
        "n_features": int(len(w)),
        "boundary2d": {"comp_x": 0, "comp_y": int(last), "w": w2, "b": b2},
        "confusion": {"labels": classes, "matrix": cm},
        "accuracy": acc,
        "kappa": kappa,
        "n_eval": int(len(y_true)),
        "holdout_kind": holdout_kind,
        "cv_folds": clf_cfg.get("cv_folds"),
        "has_model": card is not None,
    }


def build_eegnet_payload(clf, card: dict, csp_card: dict | None) -> dict:
    """Respuesta en frecuencia de los filtros temporales aprendidos + pesos espaciales.

    No necesita datos crudos: solo el modelo EEGNet (.pkl) y su ficha. Aun así se
    precomputa para que la web no requiera torch ni cargar el .pkl."""
    mdl = clf.model
    fs = float(card["fs"])

    Wt = mdl.temporal[0].weight.detach().cpu().numpy()[:, 0, 0, :]   # (F1, kern)
    freqs = frequency_response(Wt[0], fs, n_freqs=256).freqs_hz.tolist()
    temporal = []
    for h in Wt:
        mag = frequency_response(h, fs, n_freqs=256).magnitude
        temporal.append((mag / (float(mag.max()) + 1e-12)).tolist())

    Ws = mdl.spatial[0].weight.detach().cpu().numpy()[:, 0, :, 0]    # (F1*D, n_canales)
    pos2d, pos3d = electrode_positions(card["channels"])
    extra = card.get("extra") or {}

    csp_cmp = None
    if csp_card is not None:
        csp_cmp = {
            "accuracy_intersession": float(csp_card["accuracy"]),
            "kappa": float(csp_card["kappa"]),
        }

    return {
        "channels": card["channels"],
        "fs": fs,
        "freqs": freqs,
        "temporal": temporal,
        "spatial": Ws.tolist(),
        "classes": card["classes"],
        "accuracy_intersession": extra.get("accuracy_intersession", card["accuracy"]),
        "accuracy_kfold": extra.get("accuracy_kfold"),
        "folds": extra.get("folds"),
        "kern_length": int(Wt.shape[1]),
        "pos2d": pos2d, "pos3d": pos3d,
        "n_train": int(card["n_train"]),
        "n_demo": int(card["n_demo"]),
        "kappa": float(card["kappa"]),
        "trained_on": card["trained_on"],
        "epochs": extra.get("epochs"),
        "fir": card["fir"],
        "n_temporal": int(Wt.shape[0]),
        "csp_lda": csp_cmp,
    }


# --- Persistencia de payloads ----------------------------------------------
def payload_path(out_dir: Path, dataset: str, subject: int, method: str, kind: str) -> Path:
    """Ruta del JSON precomputado de un payload (junto a los modelos, en paths.processed)."""
    return Path(out_dir) / f"viz_{dataset}_s{subject}_{method}_{kind}.json"


def save_payload(out_dir: Path, dataset: str, subject: int, method: str, kind: str,
                 payload: dict) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = payload_path(out_dir, dataset, subject, method, kind)
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return path


def load_payload(out_dir: Path, dataset: str, subject: int, method: str,
                 kind: str) -> dict | None:
    """Lee el JSON precomputado si existe; si no, ``None`` (el servidor calcula al vuelo)."""
    path = payload_path(out_dir, dataset, subject, method, kind)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001 - un payload corrupto no debe tumbar el endpoint
        return None
