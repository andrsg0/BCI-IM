"""Entrena EEGNet y lo compara con CSP+LDA (within-subject, k-fold).

EEGNet recibe la señal CRUDA (recortada a la ventana de imaginación), sin el
filtro µ/β ni el CSP: aprende esos filtros por sí misma. CSP+LDA usa el pipeline
clásico. Ambos se evalúan con los mismos folds para una comparación justa.

Uso:
    python scripts/train_eegnet.py --subjects 1
    python scripts/train_eegnet.py --config configs/bci2b.yaml --subjects 1 2 3 --epochs 200
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold

from bci.config import load_config, resolve_path
from bci.datasets.moabb_loader import load_dataset
from bci.dsp.convolution import apply_filter
from bci.dsp.fir_filters import design_bandpass_fir
from bci.models.eegnet import EEGNetClassifier
from bci.pipeline.offline import MotorImageryPipeline


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--subjects", type=int, nargs="+", default=[1])
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=250)
    args = ap.parse_args()

    cfg = load_config(args.config)
    ds, ep = cfg["dataset"], cfg["epoching"]
    win = cfg.get("classification_window")
    tag = ds["name"]
    print(f"=== EEGNet vs CSP+LDA | {tag} | sujetos={args.subjects} | {args.folds}-fold | {args.epochs} épocas ===\n")

    rows = []
    for s in args.subjects:
        data = load_dataset(ds["name"], [s], ds.get("classes"), ep["tmin"], ep["tmax"],
                            ep.get("picks", "eeg"), ep.get("baseline"))
        X, y, fs = data.X, np.asarray(data.y), data.sfreq
        # banda amplia 4–40 Hz (quita deriva y EMG; EEGNet aprende los ritmos dentro)
        wide = design_bandpass_fir(4.0, min(40.0, fs / 2 - 1), fs, 101)
        Xf = apply_filter(X, wide.h, mode="same")
        # ventana de imaginación activa
        Xc = Xf[:, :, int(win["tmin_rel"] * fs):int(win["tmax_rel"] * fs)] if win else Xf
        n_classes = len(np.unique(y))
        kern = int(fs // 2)  # filtro temporal ~0.5 s, acorde a fs

        skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=42)
        csp_accs, eeg_accs = [], []
        for k, (tr, te) in enumerate(skf.split(X, y), 1):
            pipe = MotorImageryPipeline(cfg, fs=fs).fit(X[tr], y[tr])
            csp_accs.append(accuracy_score(y[te], pipe.predict(X[te])))
            clf = EEGNetClassifier(n_classes=n_classes, epochs=args.epochs, kern_length=kern).fit(Xc[tr], y[tr])
            eeg_accs.append(accuracy_score(y[te], clf.predict(Xc[te])))
            print(f"  sujeto {s} fold {k}/{args.folds}: CSP+LDA={csp_accs[-1]:.3f}  EEGNet={eeg_accs[-1]:.3f}")

        row = {"subject": s, "csp_lda": float(np.mean(csp_accs)), "eegnet": float(np.mean(eeg_accs))}
        rows.append(row)
        print(f"  -> sujeto {s}: CSP+LDA={row['csp_lda']:.3f}  EEGNet={row['eegnet']:.3f}\n")

    df = pd.DataFrame(rows)
    print("=== Resumen (media ± std) ===")
    print(f"  CSP+LDA : {df['csp_lda'].mean():.3f} ± {df['csp_lda'].std():.3f}")
    print(f"  EEGNet  : {df['eegnet'].mean():.3f} ± {df['eegnet'].std():.3f}")

    out = resolve_path(cfg["paths"]["processed"]) / f"results_eegnet_{tag}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"\nCSV guardado: {out}")


if __name__ == "__main__":
    main()
