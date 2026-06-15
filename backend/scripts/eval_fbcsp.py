"""Compara FBCSP (banco de filtros) vs CSP de banda única — experimento honesto.

Mide ambos con el MISMO esquema, por sujeto:
  - inter-sesión: train '0train' → test '1test' (generalización a otro día = en vivo).
  - k-fold estratificado (mezcla sesiones; estimación más optimista).

Guarda un CSV y una figura comparativa. Conclusión esperada (resultado negativo):
FBCSP NO mejora al CSP de banda única en inter-sesión; sobreajusta la sesión.

Uso:
    python scripts/eval_fbcsp.py                 # sujetos 1..9 del 2a
    python scripts/eval_fbcsp.py --subjects 1 3 8
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from bci.config import load_config, resolve_path
from bci.datasets.moabb_loader import load_dataset
from bci.pipeline.fbcsp import FBCSPPipeline
from bci.pipeline.offline import MotorImageryPipeline


def _single(cfg, fs, Xtr, ytr, Xte):
    return MotorImageryPipeline(cfg, fs=fs).fit(Xtr, ytr).predict(Xte)


def _fbcsp(cfg, fs, banded_tr, ytr, banded_te):
    pipe = FBCSPPipeline.from_config(cfg, fs)
    pipe.fit_banded(banded_tr, ytr)
    return pipe.predict_banded(banded_te)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--subjects", type=int, nargs="+", default=list(range(1, 10)))
    ap.add_argument("--folds", type=int, default=5)
    args = ap.parse_args()

    cfg = load_config(args.config)
    ds, ep = cfg["dataset"], cfg["epoching"]
    tag = ds["name"]
    print(f"=== FBCSP vs CSP banda única | {tag} | sujetos={args.subjects} ===\n")

    rows = []
    for s in args.subjects:
        d = load_dataset(ds["name"], [s], ds.get("classes"), ep["tmin"], ep["tmax"],
                         ep.get("picks", "eeg"), ep.get("baseline"))
        X, y, fs = d.X, np.asarray(d.y), d.sfreq
        sess = d.metadata["session"].to_numpy()
        tr, te = sess == "0train", sess == "1test"

        # El banco de FIR es fijo: se precomputa una vez y se reusa en todos los folds.
        banded = FBCSPPipeline.from_config(cfg, fs).filter_bank(X)

        # inter-sesión
        s_is = float(np.mean(_single(cfg, fs, X[tr], y[tr], X[te]) == y[te]))
        f_is = float(np.mean(_fbcsp(cfg, fs, [b[tr] for b in banded], y[tr],
                                    [b[te] for b in banded]) == y[te]))
        # k-fold
        skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=42)
        sk, fk = [], []
        for a, b in skf.split(X, y):
            sk.append(np.mean(_single(cfg, fs, X[a], y[a], X[b]) == y[b]))
            fk.append(np.mean(_fbcsp(cfg, fs, [bd[a] for bd in banded], y[a],
                                     [bd[b] for bd in banded]) == y[b]))
        row = {"subject": s, "csp_intersession": s_is, "fbcsp_intersession": f_is,
               "csp_kfold": float(np.mean(sk)), "fbcsp_kfold": float(np.mean(fk))}
        rows.append(row)
        print(f"sujeto {s}: inter-ses CSP={s_is:.3f} FBCSP={f_is:.3f} | "
              f"kfold CSP={row['csp_kfold']:.3f} FBCSP={row['fbcsp_kfold']:.3f}")

    df = pd.DataFrame(rows)
    print("\n=== Media sobre sujetos ===")
    for c in ["csp_intersession", "fbcsp_intersession", "csp_kfold", "fbcsp_kfold"]:
        print(f"  {c:22s}: {df[c].mean():.3f} ± {df[c].std():.3f}")

    out = resolve_path(cfg["paths"]["processed"]) / f"results_fbcsp_{tag}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"\nCSV guardado: {out}")

    _plot(df, tag)


def _plot(df: pd.DataFrame, tag: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    subs = df["subject"].to_numpy()
    x = np.arange(len(subs))
    w = 0.38
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(x - w / 2, df["csp_intersession"], w, label="CSP banda única", color="#2563eb")
    ax.bar(x + w / 2, df["fbcsp_intersession"], w, label="FBCSP (banco)", color="#e11d48")
    ax.axhline(0.5, ls="--", lw=1, color="#94a3b8", label="azar")
    ax.set_xticks(x); ax.set_xticklabels([f"S{s}" for s in subs])
    ax.set_ylabel("accuracy inter-sesión"); ax.set_ylim(0.4, 1.0)
    ax.set_title(f"FBCSP vs CSP de banda única (inter-sesión, {tag})")
    ax.legend(); fig.tight_layout()
    path = resolve_path("..") / "docs" / "figures" / f"fbcsp_vs_csp_{tag}.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    print(f"Figura guardada: {path}")


if __name__ == "__main__":
    main()
