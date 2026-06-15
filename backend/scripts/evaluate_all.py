"""Evaluación WITHIN-SUBJECT de todos los sujetos del BCI IV 2a.

Entrena y evalúa un modelo INDEPENDIENTE por sujeto (no se mezclan sujetos: los
patrones EEG varían mucho entre personas y el CSP no se alinea entre ellos). Reporta
una tabla por sujeto y la media ± desviación estándar.

Uso:
    python scripts/evaluate_all.py                  # sujetos 1..9
    python scripts/evaluate_all.py --subjects 1 2 3
    python scripts/evaluate_all.py --no-plot --no-csv

Salidas:
  - tabla en consola + media±std
  - docs/figures/subjects_2a.png  (barras por sujeto)
  - backend/data/processed/results_2a.csv
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from bci.config import BACKEND_ROOT, load_config, resolve_path
from bci.datasets.moabb_loader import load_dataset
from bci.pipeline.offline import evaluate_by_session, evaluate_kfold

FIG_DIR = BACKEND_ROOT.parent / "docs" / "figures"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--subjects", type=int, nargs="+", default=list(range(1, 10)))
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--no-plot", action="store_true")
    ap.add_argument("--no-csv", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    ds = cfg["dataset"]
    ep = cfg["epoching"]
    print(f"=== Evaluación within-subject | {ds['name']} | clases={ds.get('classes')} ===")
    print(f"Sujetos: {args.subjects}\n")

    rows = []
    for s in args.subjects:
        # Carga independiente por sujeto (descarga a la caché de MNE si falta).
        try:
            data = load_dataset(
                name=ds["name"], subjects=[s], classes=ds.get("classes"),
                tmin=ep["tmin"], tmax=ep["tmax"], picks=ep.get("picks", "eeg"),
                baseline=ep.get("baseline"),
            )
        except Exception as exc:  # algunos sujetos de PhysioNet tienen datos corruptos
            print(f"  sujeto {s}: OMITIDO ({type(exc).__name__}: {str(exc)[:60]})")
            continue
        kf = evaluate_kfold(cfg, data.X, data.y, fs=data.sfreq, n_splits=args.folds)
        row = {"subject": s, "n_trials": data.n_trials,
               "kfold_acc": kf.accuracy, "kfold_kappa": kf.kappa}
        if {"0train", "1test"} <= set(data.metadata["session"].unique()):
            iss = evaluate_by_session(cfg, data.X, data.y, data.metadata, fs=data.sfreq)
            row["intersession_acc"] = iss.accuracy
            row["intersession_kappa"] = iss.kappa
        rows.append(row)
        msg = f"  sujeto {s}: k-fold acc={kf.accuracy:.3f} (κ={kf.kappa:.3f})"
        if "intersession_acc" in row:
            msg += f" | inter-sesión acc={row['intersession_acc']:.3f} (κ={row['intersession_kappa']:.3f})"
        print(msg)

    df = pd.DataFrame(rows)

    # ---- Resumen media ± std ----
    print("\n=== Resumen (media ± std sobre sujetos) ===")
    for col in ["kfold_acc", "kfold_kappa", "intersession_acc", "intersession_kappa"]:
        if col in df:
            print(f"  {col:20s}: {df[col].mean():.3f} ± {df[col].std():.3f}")

    tag = ds["name"]
    # ---- Guardar CSV ----
    if not args.no_csv:
        out = resolve_path(cfg["paths"]["processed"]) / f"results_{tag}.csv"
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out, index=False)
        print(f"\nCSV guardado: {out}")

    # ---- Figura ----
    if not args.no_plot and len(df):
        import matplotlib.pyplot as plt

        from bci.viz.plots import plot_subject_results
        FIG_DIR.mkdir(parents=True, exist_ok=True)
        # Datasets de 1 sesión no tienen inter-sesión -> solo k-fold.
        second = df["intersession_acc"] if "intersession_acc" in df else None
        fig = plot_subject_results(df["subject"], df["kfold_acc"], second,
                                   title=f"Evaluación por sujeto — {tag}")
        fig.tight_layout()
        fig.savefig(FIG_DIR / f"subjects_{tag}.png", dpi=120)
        plt.close(fig)
        print(f"Figura guardada: {FIG_DIR / f'subjects_{tag}.png'}")


if __name__ == "__main__":
    main()
