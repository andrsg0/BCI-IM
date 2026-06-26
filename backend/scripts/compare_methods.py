"""Comparación 2×2: CSP+LDA vs EEGNet, within-subject vs cross-subject.

Produce los cuatro números clave del proyecto en una sola tabla, todos medidos de
forma CONSISTENTE (mismos folds, mismo dataset, mismo pre-proceso):

  - within-subject (k-fold): se entrena y evalúa en el MISMO sujeto (calibrado).
  - cross-subject (LOSO):   se entrena con los DEMÁS sujetos y se evalúa en el
    excluido (generalización a un usuario NUEVO, sin calibrar).

CSP+LDA es sujeto-específico por diseño (sus filtros espaciales se ajustan a la
anatomía de cada persona), así que su versión cross-subject suele caer cerca del
azar: ese contraste con EEGNet es justo el punto didáctico de la comparación.

Uso:
    python scripts/compare_methods.py --config ../configs/default.yaml --subjects 1 2 3 4 5 6 7 8 9
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, cohen_kappa_score
from sklearn.model_selection import StratifiedKFold

from bci.config import load_config, resolve_path
from bci.datasets.moabb_loader import load_from_config
from bci.models.eegnet import EEGNetClassifier, pick_device
from bci.pipeline.offline import MotorImageryPipeline
from bci.pipeline.training import _eegnet_features


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None, help="YAML del dataset (p. ej. ../configs/default.yaml)")
    ap.add_argument("--subjects", type=int, nargs="+", required=True)
    ap.add_argument("--folds", type=int, default=5, help="folds del within-subject")
    ap.add_argument("--epochs", type=int, default=250, help="épocas de EEGNet")
    ap.add_argument("--device", default=None, help="cuda | cpu (default: auto)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    cfg = {**cfg, "dataset": {**cfg["dataset"], "subjects": list(args.subjects)}}
    dataset = cfg["dataset"]["name"]

    data = load_from_config(cfg)
    y = np.asarray(data.y)
    subj = data.metadata["subject"].to_numpy()
    X = data.X                                   # crudo: el pipeline CSP filtra internamente
    Xc, fs, kern, _ = _eegnet_features(cfg, data)  # banda amplia + recorte para EEGNet
    n_classes = len(np.unique(y))
    dev = pick_device(args.device)
    present = sorted(int(s) for s in np.unique(subj))

    print(f"=== CSP+LDA vs EEGNet | {dataset} | sujetos={present} | "
          f"within={args.folds}-fold · cross=LOSO | device={dev} ===\n")

    def new_eeg() -> EEGNetClassifier:
        return EEGNetClassifier(n_classes=n_classes, epochs=args.epochs, kern_length=kern,
                                weight_decay=1e-3, device=dev)

    rows = []
    for s in present:
        m = subj == s
        Xs, Xcs, ys = X[m], Xc[m], y[m]

        # --- within-subject (k-fold) ---
        skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=42)
        csp_w, eeg_w = [], []
        csp_wk, eeg_wk = [], []
        for tr, te in skf.split(Xs, ys):
            pipe = MotorImageryPipeline(cfg, fs=fs).fit(Xs[tr], ys[tr])
            csp_pred = pipe.predict(Xs[te])
            csp_w.append(accuracy_score(ys[te], csp_pred))
            csp_wk.append(cohen_kappa_score(ys[te], csp_pred))
            clf = new_eeg().fit(Xcs[tr], ys[tr])
            eeg_pred = clf.predict(Xcs[te])
            eeg_w.append(accuracy_score(ys[te], eeg_pred))
            eeg_wk.append(cohen_kappa_score(ys[te], eeg_pred))

        # --- cross-subject (LOSO: entrenar con los demás, evaluar en s) ---
        other = subj != s
        pipe = MotorImageryPipeline(cfg, fs=fs).fit(X[other], y[other])
        csp_c_pred = pipe.predict(Xs)
        csp_c = accuracy_score(ys, csp_c_pred)
        csp_ck = cohen_kappa_score(ys, csp_c_pred)
        clf = new_eeg().fit(Xc[other], y[other])
        eeg_c_pred = clf.predict(Xcs)
        eeg_c = accuracy_score(ys, eeg_c_pred)
        eeg_ck = cohen_kappa_score(ys, eeg_c_pred)

        row = {"subject": s,
               "csp_within": float(np.mean(csp_w)),
               "csp_within_kappa": float(np.mean(csp_wk)),
               "eegnet_within": float(np.mean(eeg_w)),
               "eegnet_within_kappa": float(np.mean(eeg_wk)),
               "csp_cross": float(csp_c),
               "csp_cross_kappa": float(csp_ck),
               "eegnet_cross": float(eeg_c),
               "eegnet_cross_kappa": float(eeg_ck)}
        rows.append(row)
        print(f"  sujeto {s:>2}: CSP within={row['csp_within']:.3f}(κ={row['csp_within_kappa']:.3f}) "
              f"cross={row['csp_cross']:.3f}(κ={row['csp_cross_kappa']:.3f})  |  "
              f"EEGNet within={row['eegnet_within']:.3f}(κ={row['eegnet_within_kappa']:.3f}) "
              f"cross={row['eegnet_cross']:.3f}(κ={row['eegnet_cross_kappa']:.3f})")

    df = pd.DataFrame(rows)
    print("\n=== Matriz 2×2 (media sobre sujetos) ===")
    print("              within-subject (acc/κ)    cross-subject (acc/κ)")
    print(f"  CSP+LDA :     {df['csp_within'].mean():.3f}/{df['csp_within_kappa'].mean():.3f}"
          f"             {df['csp_cross'].mean():.3f}/{df['csp_cross_kappa'].mean():.3f}")
    print(f"  EEGNet  :     {df['eegnet_within'].mean():.3f}/{df['eegnet_within_kappa'].mean():.3f}"
          f"             {df['eegnet_cross'].mean():.3f}/{df['eegnet_cross_kappa'].mean():.3f}")

    out = resolve_path(cfg["paths"]["processed"]) / f"compare_methods_{dataset}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"\nCSV guardado: {out}")


if __name__ == "__main__":
    main()
