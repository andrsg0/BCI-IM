"""Entrenamiento offline + persistencia del modelo (separación entrenar / transmitir).

Este módulo materializa una idea central del proyecto: el modelo se entrena ANTES
del streaming, con una partición de ENTRENAMIENTO, y se reserva una partición
HELD-OUT que NUNCA se usa para entrenar. Esa partición held-out es justamente la
que se "transmite" en la demo en vivo, de modo que la clasificación en tiempo real
evalúa señales que el modelo no vio jamás (sin fuga de datos).

Reglas del split:
  - Si el dataset tiene >= 2 sesiones (p. ej. el BCI IV 2a: '0train' y '1test'),
    se entrena con la primera sesión y se reserva la segunda para la demo. Es la
    estimación más honesta de cómo generalizaría a una grabación de otro día.
  - Si tiene una sola sesión (PhysioNet, Liu2024), se reserva una fracción
    estratificada (30 %) como demo.

El artefacto se guarda en dos ficheros junto a los datos procesados:
  - ``model_{dataset}_s{subject}.pkl``  -> el pipeline entrenado (CSP + LDA).
  - ``model_{dataset}_s{subject}.json`` -> la "ficha del modelo" (ModelCard).
"""
from __future__ import annotations

import json
import pickle
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold, train_test_split

from bci.datasets.moabb_loader import EpochedData, load_from_config
from bci.pipeline.offline import MotorImageryPipeline, _metrics


@dataclass
class ModelCard:
    """Metadatos legibles de un modelo entrenado (lo que la web muestra etiquetado).

    Todo aquí describe el mundo OFFLINE: con qué se entrenó, qué se reservó para la
    demo y qué precisión honesta dio sobre esa partición reservada.
    """

    dataset: str
    subject: int
    method: str            # 'csp_lda' | 'eegnet'
    fs: float
    classes: list[str]
    channels: list[str]
    holdout: dict          # {"by":"session","value":"1test"} | {"by":"index","indices":[...]}
    train_session: str | None
    n_train: int           # nº de trials usados para ENTRENAR
    n_demo: int            # nº de trials reservados para la demo en vivo
    accuracy: float        # accuracy honesta sobre la partición held-out (inter-sesión)
    kappa: float
    trained_on: str        # fecha de entrenamiento (ISO)
    n_components: int       # componentes CSP
    fir: dict              # banda y nº de coeficientes del FIR
    extra: dict | None = None   # métricas adicionales (p. ej. accuracy k-fold de EEGNet)


# --- Split entrenar / demo -------------------------------------------------
def split_train_demo(data: EpochedData):
    """Devuelve (idx_train, idx_demo, holdout_spec, train_session).

    ``idx_demo`` son los trials que se reservan para el streaming en vivo y que el
    modelo NO verá durante el entrenamiento.
    """
    sessions = sorted(data.metadata["session"].unique().tolist())
    if len(sessions) >= 2:
        train_session, demo_session = sessions[0], sessions[1]
        sess = data.metadata["session"].to_numpy()
        idx_train = np.where(sess == train_session)[0]
        idx_demo = np.where(sess == demo_session)[0]
        return idx_train, idx_demo, {"by": "session", "value": demo_session}, train_session
    # Una sola sesión: reservamos una fracción estratificada como demo.
    idx = np.arange(len(data.y))
    idx_train, idx_demo = train_test_split(
        idx, test_size=0.3, stratify=data.y, random_state=42
    )
    idx_demo = np.sort(idx_demo)
    return idx_train, idx_demo, {"by": "index", "indices": idx_demo.tolist()}, None


# --- Persistencia ----------------------------------------------------------
def model_paths(out_dir: Path, dataset: str, subject: int, method: str = "csp_lda") -> tuple[Path, Path]:
    """Rutas del .pkl (modelo) y .json (ficha) para un (dataset, sujeto, método)."""
    out_dir = Path(out_dir)
    stem = f"model_{dataset}_s{subject}_{method}"
    return out_dir / f"{stem}.pkl", out_dir / f"{stem}.json"


def save_model(model, card: ModelCard, out_dir: Path) -> tuple[Path, Path]:
    """Persiste el modelo entrenado (CSP+LDA o EEGNet) y su ficha."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pkl_path, json_path = model_paths(out_dir, card.dataset, card.subject, card.method)
    with pkl_path.open("wb") as fh:
        pickle.dump(model, fh)
    json_path.write_text(json.dumps(asdict(card), indent=2, ensure_ascii=False), encoding="utf-8")
    return pkl_path, json_path


def load_model(pkl_path: Path) -> MotorImageryPipeline:
    with Path(pkl_path).open("rb") as fh:
        return pickle.load(fh)


def load_card(json_path: Path) -> dict:
    return json.loads(Path(json_path).read_text(encoding="utf-8"))


# --- Entrenamiento ---------------------------------------------------------
def train_subject(cfg: dict, dataset: str, subject: int) -> tuple[MotorImageryPipeline, ModelCard, EpochedData]:
    """Entrena un modelo para un sujeto con el split honesto (train / demo held-out).

    Devuelve el pipeline entrenado, su ficha y los datos cargados (para reutilizar).
    El pipeline se ajusta SOLO con ``idx_train``; la accuracy se mide sobre el
    held-out (``idx_demo``), que es lo que luego se transmite en vivo.
    """
    cfg = {**cfg, "dataset": {**cfg["dataset"], "subjects": [subject]}}
    data = load_from_config(cfg)
    fs = data.sfreq
    y = np.asarray(data.y)

    idx_train, idx_demo, holdout, train_session = split_train_demo(data)
    pipe = MotorImageryPipeline(cfg, fs=fs).fit(data.X[idx_train], y[idx_train])

    res = _metrics(y[idx_demo], pipe.predict(data.X[idx_demo]))
    fir = cfg["fir_filter"]
    card = ModelCard(
        dataset=dataset,
        subject=subject,
        method="csp_lda",
        fs=float(fs),
        classes=sorted(set(map(str, y))),
        channels=list(data.ch_names),
        holdout=holdout,
        train_session=train_session,
        n_train=int(len(idx_train)),
        n_demo=int(len(idx_demo)),
        accuracy=float(res.accuracy),
        kappa=float(res.kappa),
        trained_on=date.today().isoformat(),
        n_components=int(cfg["csp"]["n_components"]),
        fir={"low_hz": fir["low_hz"], "high_hz": fir["high_hz"], "num_taps": fir["num_taps"]},
    )
    return pipe, card, data


def train_eegnet_subject(cfg: dict, dataset: str, subject: int, epochs: int = 250,
                         weight_decay: float = 1e-3, folds: int = 4):
    """Entrena un EEGNet **para visualizar sus filtros** (EEGNet como espejo).

    A diferencia de CSP+LDA, a EEGNet se le da la señal sin el filtro µ/β ni el CSP:
    solo un filtrado de banda amplia (4–40 Hz) y el recorte a la ventana activa. La red
    aprende ella misma los filtros temporales (≈ FIR) y espaciales (≈ CSP).

    Como EEGNet ya **no se usa en vivo**, el modelo que se guarda (y cuyos filtros se
    visualizan) se entrena con **todos los trials** del sujeto: más datos ⇒ filtros más
    limpios e interpretables. Para no mentir con la precisión, se reportan DOS números
    honestos, medidos con modelos entrenados aparte:
      - inter-sesión (train `0train` → test held-out): generalización a otro día (~baja).
      - within-subject k-fold: el mejor caso justo con más datos (~media).
    """
    from bci.dsp.convolution import apply_filter
    from bci.dsp.fir_filters import design_bandpass_fir
    from bci.models.eegnet import EEGNetClassifier

    cfg = {**cfg, "dataset": {**cfg["dataset"], "subjects": [subject]}}
    data = load_from_config(cfg)
    fs = data.sfreq
    y = np.asarray(data.y)

    win = cfg.get("classification_window")
    high = float(min(40.0, fs / 2 - 1))
    wide = design_bandpass_fir(4.0, high, fs, 101)            # banda amplia (no µ/β estricta)
    Xf = apply_filter(data.X, wide.h, mode="same")
    Xc = Xf[:, :, int(win["tmin_rel"] * fs):int(win["tmax_rel"] * fs)] if win else Xf

    idx_train, idx_demo, holdout, train_session = split_train_demo(data)
    n_classes = len(np.unique(y))
    kern = int(fs // 2)                                        # filtro temporal ~0.5 s

    def _new():
        return EEGNetClassifier(n_classes=n_classes, epochs=epochs, kern_length=kern,
                                weight_decay=weight_decay)

    # (1) Honestidad: inter-sesión (entrena solo con train, evalúa en el held-out).
    is_clf = _new().fit(Xc[idx_train], y[idx_train])
    res_is = _metrics(y[idx_demo], is_clf.predict(Xc[idx_demo]))

    # (2) Honestidad: within-subject k-fold (el mejor caso justo, con más datos variados).
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
    accs = [accuracy_score(y[te], _new().fit(Xc[tr], y[tr]).predict(Xc[te]))
            for tr, te in skf.split(Xc, y)]
    acc_kfold = float(np.mean(accs))

    # (3) Modelo de VISUALIZACIÓN: entrenado con TODOS los datos (filtros más limpios).
    clf = _new().fit(Xc, y)
    n_temporal = int(clf.model.temporal[0].weight.shape[0])   # nº de filtros temporales (F1)

    card = ModelCard(
        dataset=dataset,
        subject=subject,
        method="eegnet",
        fs=float(fs),
        classes=sorted(set(map(str, y))),
        channels=list(data.ch_names),
        holdout=holdout,
        train_session=train_session,
        n_train=int(data.n_trials),          # el modelo visualizado usa todos los trials
        n_demo=int(len(idx_demo)),
        accuracy=float(res_is.accuracy),     # principal = inter-sesión (coherente con CSP)
        kappa=float(res_is.kappa),
        trained_on=date.today().isoformat(),
        n_components=n_temporal,
        fir={"low_hz": 4.0, "high_hz": high, "num_taps": 101},
        extra={
            "accuracy_intersession": float(res_is.accuracy),
            "accuracy_kfold": acc_kfold,
            "folds": int(folds),
            "viz_trained_on": "all_trials",
        },
    )
    return clf, card, data
