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

    Con ≥2 sesiones, la demo es la ÚLTIMA sesión y el entrenamiento usa TODAS las
    demás (no solo la primera): así se aprovechan todos los datos en datasets de
    varias sesiones (2b=5, Kumar=6) sin dejar de ser una estimación honesta
    inter-sesión (train y demo no comparten día). En el 2a (2 sesiones) esto equivale
    al split de siempre: entrena en '0train', demo en '1test'.
    """
    sessions = sorted(data.metadata["session"].unique().tolist())
    if len(sessions) >= 2:
        demo_session = sessions[-1]                       # la última sesión = demo
        sess = data.metadata["session"].to_numpy()
        idx_train = np.where(sess != demo_session)[0]     # TODAS las demás = entrenamiento
        idx_demo = np.where(sess == demo_session)[0]
        train_session = "+".join(str(s) for s in sessions if s != demo_session)
        return idx_train, idx_demo, {"by": "session", "value": demo_session}, train_session
    # Una sola sesión: reservamos una fracción estratificada como demo.
    idx = np.arange(len(data.y))
    idx_train, idx_demo = train_test_split(
        idx, test_size=0.3, stratify=data.y, random_state=42
    )
    idx_demo = np.sort(idx_demo)
    return idx_train, idx_demo, {"by": "index", "indices": idx_demo.tolist()}, None


def split_train_demo_subject(data: EpochedData, demo_subject: int):
    """Split CROSS-SUBJECT (opción A): entrena con TODOS los sujetos menos uno.

    A diferencia de ``split_train_demo`` (que reserva una SESIÓN del mismo sujeto), aquí
    se reserva un SUJETO entero: el modelo se entrena con los demás y la demo se transmite
    sobre ``demo_subject``, a quien el modelo NUNCA vio. Es la estimación honesta de
    "ponérselo a una persona nueva sin calibrar". Un solo entrenamiento (un fold), no el
    LOSO completo de N folds (ese es opcional, solo para la accuracy media de Resultados).

    Devuelve (idx_train, idx_demo, holdout_spec).
    """
    subj = data.metadata["subject"].to_numpy()
    idx_demo = np.where(subj == demo_subject)[0]
    idx_train = np.where(subj != demo_subject)[0]
    if len(idx_demo) == 0:
        raise ValueError(f"El sujeto demo {demo_subject} no está en los datos cargados.")
    if len(idx_train) == 0:
        raise ValueError("No quedan sujetos para entrenar (se necesita >= 2 sujetos).")
    return idx_train, idx_demo, {"by": "subject", "value": int(demo_subject)}


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
            "epochs": int(epochs),
            "viz_trained_on": "all_trials",
        },
    )
    return clf, card, data


# --- Régimen cross-subject (opción A: persona nueva, 1 fold) ---------------
def train_crosssubject(cfg: dict, dataset: str, subjects: list[int], demo_subject: int):
    """CSP+LDA CROSS-SUBJECT: entrena con ``subjects`` menos ``demo_subject``.

    Mismo pipeline que ``train_subject`` (FIR→CSP→log-var→LDA) pero el split es por
    SUJETO: el modelo no ve a ``demo_subject`` y se evalúa sobre él. Como CSP es
    sujeto-específico (sus filtros se ajustan a una anatomía), aquí suele rendir bajo
    — y ESE contraste con el within-subject es justamente lo didáctico.

    Devuelve (pipeline, ficha, datos) — misma firma que ``train_subject``.
    """
    cfg = {**cfg, "dataset": {**cfg["dataset"], "subjects": list(subjects)}}
    data = load_from_config(cfg)
    fs = data.sfreq
    y = np.asarray(data.y)

    idx_train, idx_demo, holdout = split_train_demo_subject(data, demo_subject)
    pipe = MotorImageryPipeline(cfg, fs=fs).fit(data.X[idx_train], y[idx_train])

    res = _metrics(y[idx_demo], pipe.predict(data.X[idx_demo]))
    fir = cfg["fir_filter"]
    train_subjects = sorted(int(s) for s in np.unique(data.metadata["subject"]) if s != demo_subject)
    card = ModelCard(
        dataset=dataset,
        subject=int(demo_subject),           # el sujeto HELD-OUT (cuyos datos se transmiten)
        method="csp_lda_cross",
        fs=float(fs),
        classes=sorted(set(map(str, y))),
        channels=list(data.ch_names),
        holdout=holdout,
        train_session=None,
        n_train=int(len(idx_train)),
        n_demo=int(len(idx_demo)),
        accuracy=float(res.accuracy),        # honesta cross-subject (persona nueva)
        kappa=float(res.kappa),
        trained_on=date.today().isoformat(),
        n_components=int(cfg["csp"]["n_components"]),
        fir={"low_hz": fir["low_hz"], "high_hz": fir["high_hz"], "num_taps": fir["num_taps"]},
        extra={"train_subjects": train_subjects, "n_train_subjects": len(train_subjects)},
    )
    return pipe, card, data


def train_eegnet_crosssubject(cfg: dict, dataset: str, subjects: list[int], demo_subject: int,
                              epochs: int = 250, weight_decay: float = 1e-3,
                              device: str | None = None, augment: bool = False,
                              augment_copies: int = 2):
    """EEGNet CROSS-SUBJECT: entrena con ``subjects`` menos ``demo_subject``.

    Es el régimen donde el deep learning brilla frente a CSP: una red entrenada con
    muchas personas puede generalizar (algo) a una nueva sin calibrar. Un solo
    entrenamiento (no LOSO completo): el más caro de los cuatro regímenes.

    Devuelve (clf, ficha, datos) — misma firma que ``train_eegnet_subject``.
    """
    from bci.datasets.augment import augment_trials
    from bci.models.eegnet import EEGNetClassifier, pick_device

    cfg = {**cfg, "dataset": {**cfg["dataset"], "subjects": list(subjects)}}
    data = load_from_config(cfg)
    y = np.asarray(data.y)

    Xc, fs, kern, high = _eegnet_features(cfg, data)
    idx_train, idx_demo, holdout = split_train_demo_subject(data, demo_subject)
    n_classes = len(np.unique(y))
    dev = pick_device(device)

    Xtr, ytr = Xc[idx_train], y[idx_train]
    if augment:
        Xtr, ytr = augment_trials(Xtr, ytr, copies=augment_copies)
    clf = EEGNetClassifier(n_classes=n_classes, epochs=epochs, kern_length=kern,
                           weight_decay=weight_decay, device=dev).fit(Xtr, ytr)

    res = _metrics(y[idx_demo], clf.predict(Xc[idx_demo]))
    n_temporal = int(clf.model.temporal[0].weight.shape[0])
    train_subjects = sorted(int(s) for s in np.unique(data.metadata["subject"]) if s != demo_subject)
    card = ModelCard(
        dataset=dataset,
        subject=int(demo_subject),
        method="eegnet_cross",
        fs=float(fs),
        classes=sorted(set(map(str, y))),
        channels=list(data.ch_names),
        holdout=holdout,
        train_session=None,
        n_train=int(len(idx_train)),
        n_demo=int(len(idx_demo)),
        accuracy=float(res.accuracy),        # honesta cross-subject (persona nueva)
        kappa=float(res.kappa),
        trained_on=date.today().isoformat(),
        n_components=n_temporal,
        fir={"low_hz": 4.0, "high_hz": high, "num_taps": 101},
        extra={
            "train_subjects": train_subjects,
            "n_train_subjects": len(train_subjects),
            "device": dev,
            "epochs": int(epochs),
            "augment": bool(augment),
        },
    )
    return clf, card, data


def _eegnet_features(cfg: dict, data: EpochedData):
    """Prepara la señal para EEGNet: filtro pasa-banda + recorte a la ventana activa.

    Devuelve ``(Xc, fs, kern_length, high)``. Por defecto usa una banda amplia
    4–40 Hz (la red aprende sus propios filtros µ/β dentro de ella). Pero la banda
    es **configurable por dataset** vía la sección ``eegnet`` del YAML, porque en
    datasets con artefactos fuertes fuera de la banda MI (p. ej. Dreyer2023: deriva
    lenta/ocular en 4–13 Hz) la banda amplia hace que EEGNet —libre de aprender lo
    que sea— se sobreajuste al artefacto y colapse, mientras CSP (restringido a
    potencia de banda) es inmune. Restringir la entrada a beta (13–30 Hz) lo rescata
    (verificado: Dreyer S3 0.44 → 0.85, sin dañar 2a). Es un caso donde el sesgo
    inductivo LTI (elegir la banda) salva al modelo de deep learning.
    """
    from bci.dsp.convolution import apply_filter
    from bci.dsp.fir_filters import design_bandpass_fir

    fs = data.sfreq
    win = cfg.get("classification_window")
    eeg_cfg = cfg.get("eegnet") or {}
    low = float(eeg_cfg.get("band_low", 4.0))
    high = min(float(eeg_cfg.get("band_high", 40.0)), fs / 2 - 1)
    band = design_bandpass_fir(low, high, fs, 101)
    Xf = apply_filter(data.X, band.h, mode="same")
    Xc = Xf[:, :, int(win["tmin_rel"] * fs):int(win["tmax_rel"] * fs)] if win else Xf
    return Xc, fs, int(fs // 2), high


def train_eegnet_pooled(cfg: dict, dataset: str, subjects: list[int], epochs: int = 250,
                        weight_decay: float = 1e-3, do_loso: bool = True, device: str | None = None,
                        augment: bool = False, augment_copies: int = 2,
                        loso_subset: int | None = None, verbose: bool = False):
    """Entrena un EEGNet **pooled** (varios sujetos juntos) y mide generalización cross-subject.

    A diferencia de CSP+LDA, que es **sujeto-específico**, aquí se prueba si una red puede
    GENERALIZAR a un sujeto que NO vio. Se reportan dos cosas honestas:

      - **LOSO** (leave-one-subject-out): para cada sujeto se entrena con TODOS los demás y
        se evalúa en él. Es la estimación honesta de "ponérselo a un usuario nuevo" (sin
        calibración). Suele ser bastante más baja que la within-subject.
      - El **modelo final** que se guarda se entrena con TODOS los sujetos: es el modelo
        BASE del que partiría un *fine-tuning* con una calibración corta del usuario nuevo.

    Parámetros para escalar datos / cómputo:
      - ``augment`` : aplica aumentación (ruido/desplazamiento/escala) SOLO al entrenamiento
        de cada partición (nunca al test) — sube el rendimiento sin grabar más sujetos.
      - ``loso_subset`` : limita el LOSO a los primeros N sujetos (cada uno se sigue
        entrenando con TODOS los demás). Con muchos sujetos (p. ej. PhysioNet 109), evaluar
        los 109 lleva horas; un subconjunto da una estimación honesta en una fracción del tiempo.
        El modelo base final SIEMPRE se entrena con todos los sujetos.
      - ``verbose`` : imprime el progreso del LOSO sujeto a sujeto (útil al correr en segundo plano).
    """
    from bci.datasets.augment import augment_trials
    from bci.models.eegnet import EEGNetClassifier, pick_device

    cfg = {**cfg, "dataset": {**cfg["dataset"], "subjects": list(subjects)}}
    data = load_from_config(cfg)
    y = np.asarray(data.y)
    subj = data.metadata["subject"].to_numpy()

    Xc, fs, kern, high = _eegnet_features(cfg, data)
    n_classes = len(np.unique(y))
    dev = pick_device(device)

    def _new():
        return EEGNetClassifier(n_classes=n_classes, epochs=epochs, kern_length=kern,
                                weight_decay=weight_decay, device=dev)

    def _fit_on(Xtr, ytr):
        if augment:
            Xtr, ytr = augment_trials(Xtr, ytr, copies=augment_copies)
        return _new().fit(Xtr, ytr)

    # (1) LOSO: la estimación honesta de generalización a un sujeto nuevo (sin calibrar).
    present = sorted(int(s) for s in np.unique(subj))
    loso_targets = present[:loso_subset] if loso_subset else present
    loso: dict[int, float] = {}
    if do_loso and len(present) >= 2:
        for k, s in enumerate(loso_targets, 1):
            te = subj == s
            tr = ~te
            if len(np.unique(y[tr])) < 2 or te.sum() == 0:
                continue
            clf_s = _fit_on(Xc[tr], y[tr])
            loso[s] = float(accuracy_score(y[te], clf_s.predict(Xc[te])))
            if verbose:
                print(f"  [LOSO {k}/{len(loso_targets)}] sujeto {s:>3}: {loso[s]:.3f}", flush=True)
    loso_mean = float(np.mean(list(loso.values()))) if loso else 0.0

    # (2) Modelo final pooled: entrenado con TODOS los sujetos (base para fine-tuning).
    if verbose:
        print(f"  Entrenando modelo base con TODOS los sujetos ({len(present)})…", flush=True)
    clf = _fit_on(Xc, y)
    n_temporal = int(clf.model.temporal[0].weight.shape[0])

    card = ModelCard(
        dataset=dataset,
        subject=0,                       # 0 = pooled (no pertenece a un sujeto concreto)
        method="eegnet_pooled",
        fs=float(fs),
        classes=sorted(set(map(str, y))),
        channels=list(data.ch_names),
        holdout={"by": "loso", "subjects": present},
        train_session=None,
        n_train=int(data.n_trials),
        n_demo=0,
        accuracy=loso_mean,              # principal = media LOSO (honesto cross-subject)
        kappa=0.0,
        trained_on=date.today().isoformat(),
        n_components=n_temporal,
        fir={"low_hz": 4.0, "high_hz": high, "num_taps": 101},
        extra={
            "subjects": present,
            "n_subjects": len(present),
            "loso_per_subject": loso,
            "loso_subjects": loso_targets if (do_loso and len(present) >= 2) else [],
            "loso_mean": loso_mean,
            "device": dev,
            "epochs": int(epochs),
            "augment": bool(augment),
            "augment_copies": int(augment_copies) if augment else 0,
            "viz_trained_on": "all_subjects",
        },
    )
    return clf, card, data
