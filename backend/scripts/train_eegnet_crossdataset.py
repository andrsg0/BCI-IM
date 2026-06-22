"""Entrena un EEGNet POOLED entre VARIOS datasets (Paso 3: cross-dataset).

Junta PhysioNet + Dreyer2023 + Cho2017 (todos imaginación motora izq./der.) en un
solo pool. Como vienen a distinta frecuencia de muestreo y con distinto montaje, antes
de juntarlos hay que ACONDICIONAR la señal (el tema de Sistemas Lineales y Señales):

  1. Armonizar el MONTAJE: quedarse con los canales motores COMUNES a todos (19).
  2. Remuestrear a una fs COMÚN (128 Hz) con filtro anti-aliasing (bci.dsp.resampling).
  3. Extraer las features de EEGNet (banda 4-40 Hz + recorte de la ventana activa),
     que a 128 Hz da exactamente 256 muestras (2 s) en TODOS los datasets -> alineados.

Evaluación honesta = LEAVE-ONE-DATASET-OUT: entrena con 2 datasets y prueba en el 3º
(generalización a un dataset NUEVO, lo más exigente). El modelo base final se entrena
con TODOS y se guarda como base para fine-tuning.

Uso (ver docs/remuestreo.md y docs/entrenamiento.md):
    python scripts/train_eegnet_crossdataset.py --max-subjects 20 --epochs 200 --augment
    python scripts/train_eegnet_crossdataset.py --datasets physionet dreyer2023 cho2017 \
        --max-subjects 0 --epochs 300 --augment        # 0 = TODOS los sujetos (muy largo)
"""
from __future__ import annotations

import argparse

import numpy as np
from sklearn.metrics import accuracy_score

from bci.config import load_config, resolve_path
from bci.datasets.augment import augment_trials
from bci.datasets.moabb_loader import EpochedData, load_from_config
from bci.dsp.resampling import resample_lti
from bci.models.eegnet import EEGNetClassifier, pick_device
from bci.pipeline.training import ModelCard, _eegnet_features, save_model

# Canales motores COMUNES a BCI IV 2a + PhysioNet + Dreyer2023 + Cho2017 (intersección
# de nombres, derivada cargando cada dataset). Centrales/motores, ideales para MI.
COMMON_CHANNELS = [
    "FC3", "FC1", "FCz", "FC2", "FC4",
    "C5", "C3", "C1", "Cz", "C2", "C4", "C6",
    "CP3", "CP1", "CPz", "CP2", "CP4",
    "Fz", "Pz",
]

CONFIG_OF = {
    "physionet": "../configs/physionet.yaml",
    "dreyer2023": "../configs/dreyer2023.yaml",
    "cho2017": "../configs/cho2017.yaml",
    "default": "../configs/default.yaml",   # BCI IV 2a (normalmente reservado para en vivo)
}


def load_harmonized(cfg_name: str, subjects: list[int] | None, fs_common: float):
    """Carga un dataset, lo lleva a los canales comunes y a la fs común, y extrae las
    features de EEGNet. Devuelve (Xc, y, fs, kern)."""
    cfg = load_config(CONFIG_OF[cfg_name])
    if subjects:
        cfg = {**cfg, "dataset": {**cfg["dataset"], "subjects": subjects}}
    data = load_from_config(cfg)

    # 1) armonizar montaje: seleccionar y reordenar a los canales comunes
    idx = [data.ch_names.index(ch) for ch in COMMON_CHANNELS]
    Xh = data.X[:, idx, :]

    # 2) remuestrear a la fs común (anti-aliasing FIR + diezmado)
    Xr = resample_lti(Xh, data.sfreq, fs_common)

    # 3) features EEGNet (banda 4-40 + recorte ventana activa) ya a la fs común
    data_r = EpochedData(X=Xr, y=np.asarray(data.y), metadata=data.metadata,
                         ch_names=list(COMMON_CHANNELS), sfreq=fs_common)
    Xc, fs, kern, _ = _eegnet_features(cfg, data_r)
    return Xc, np.asarray(data.y), fs, kern


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["physionet", "dreyer2023", "cho2017"],
                    choices=list(CONFIG_OF))
    ap.add_argument("--max-subjects", type=int, default=20,
                    help="nº de sujetos por dataset (0 = TODOS; cuidado con el tiempo)")
    ap.add_argument("--fs-common", type=float, default=128.0, help="fs común de remuestreo")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--augment", action="store_true")
    ap.add_argument("--augment-copies", type=int, default=2)
    ap.add_argument("--no-eval", action="store_true", help="omitir leave-one-dataset-out")
    ap.add_argument("--device", default=None)
    ap.add_argument("--no-save", action="store_true")
    args = ap.parse_args()

    dev = pick_device(args.device)
    fs_c = args.fs_common
    print(f"=== EEGNet CROSS-DATASET | {args.datasets} | fs_común={fs_c:g} Hz | "
          f"{len(COMMON_CHANNELS)} canales | max_subj={args.max_subjects or 'TODOS'} | "
          f"épocas={args.epochs} | augment={args.augment} | device={dev} ===\n", flush=True)

    # Cargar, armonizar y remuestrear cada dataset
    parts: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    kern = None
    for name in args.datasets:
        subs = list(range(1, args.max_subjects + 1)) if args.max_subjects else None
        Xc, y, _, kern = load_harmonized(name, subs, fs_c)
        parts[name] = (Xc, y)
        print(f"  {name:11}: {Xc.shape[0]:>5} trials  x {Xc.shape[1]} ch x {Xc.shape[2]} muestras", flush=True)

    n_classes = len(np.unique(np.concatenate([y for _, y in parts.values()])))

    def new_clf():
        return EEGNetClassifier(n_classes=n_classes, epochs=args.epochs, kern_length=kern,
                                weight_decay=1e-3, device=dev)

    def fit_on(X, y):
        if args.augment:
            X, y = augment_trials(X, y, copies=args.augment_copies)
        return new_clf().fit(X, y)

    # Evaluación LEAVE-ONE-DATASET-OUT (generalización a un dataset nuevo)
    lodo: dict[str, float] = {}
    if not args.no_eval and len(parts) >= 2:
        names = list(parts)
        for test_name in names:
            Xtr = np.concatenate([parts[n][0] for n in names if n != test_name])
            ytr = np.concatenate([parts[n][1] for n in names if n != test_name])
            Xte, yte = parts[test_name]
            clf = fit_on(Xtr, ytr)
            lodo[test_name] = float(accuracy_score(yte, clf.predict(Xte)))
            print(f"  [LODO] entrena sin {test_name:11} -> test en {test_name:11}: {lodo[test_name]:.3f}", flush=True)
        print(f"  -> media leave-one-dataset-out = {np.mean(list(lodo.values())):.3f}\n", flush=True)

    # Modelo base final: TODOS los datasets juntos
    print("  Entrenando modelo base con TODOS los datasets…", flush=True)
    X_all = np.concatenate([X for X, _ in parts.values()])
    y_all = np.concatenate([y for _, y in parts.values()])
    clf = fit_on(X_all, y_all)

    card = ModelCard(
        dataset="+".join(args.datasets), subject=0, method="eegnet_crossdataset",
        fs=float(fs_c), classes=sorted(set(map(str, y_all))), channels=list(COMMON_CHANNELS),
        holdout={"by": "leave_one_dataset_out", "datasets": list(parts)}, train_session=None,
        n_train=int(len(y_all)), n_demo=0,
        accuracy=float(np.mean(list(lodo.values()))) if lodo else 0.0, kappa=0.0,
        trained_on=__import__("datetime").date.today().isoformat(),
        n_components=int(clf.model.temporal[0].weight.shape[0]),
        fir={"low_hz": 4.0, "high_hz": min(40.0, fs_c / 2 - 1), "num_taps": 101},
        extra={"datasets": list(parts), "fs_common": fs_c, "common_channels": COMMON_CHANNELS,
               "lodo_per_dataset": lodo, "max_subjects": args.max_subjects, "epochs": args.epochs,
               "augment": args.augment, "device": dev, "viz_trained_on": "all_datasets"},
    )

    if args.no_save:
        print("(--no-save: modelo NO guardado)")
        return
    out_dir = resolve_path(load_config(CONFIG_OF[args.datasets[0]])["paths"]["processed"])
    tag = "+".join(args.datasets)
    pkl, js = save_model(clf, card, out_dir)
    print(f"\nModelo cross-dataset guardado ({len(y_all)} trials de {tag}):\n  {pkl.name}\n  {js.name}")


if __name__ == "__main__":
    main()
