"""Barrido de regularización (shrinkage) del CSP — antes/después.

Evalúa el pipeline con varios valores de shrinkage γ y reporta la accuracy media
(k-fold, within-subject) sobre los sujetos, para ver cuánto ayuda la regularización.
Carga cada sujeto UNA vez y reutiliza los datos para todos los γ.

Uso:
    python scripts/sweep_shrinkage.py --config configs/bci2b.yaml --subjects $(seq 1 20)
    python scripts/sweep_shrinkage.py --config configs/kumar2024.yaml --subjects $(seq 1 20)
    python scripts/sweep_shrinkage.py --gammas 0 0.05 0.1 0.2 0.3 0.5

Salidas: tabla en consola + docs/figures/shrinkage_sweep_<dataset>.png
"""
from __future__ import annotations

import argparse
import copy

import numpy as np

from bci.config import BACKEND_ROOT, load_config
from bci.datasets.moabb_loader import load_dataset
from bci.pipeline.offline import evaluate_kfold

FIG_DIR = BACKEND_ROOT.parent / "docs" / "figures"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--subjects", type=int, nargs="+", default=list(range(1, 10)))
    ap.add_argument("--gammas", type=float, nargs="+",
                    default=[0.0, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5])
    ap.add_argument("--folds", type=int, default=5)
    args = ap.parse_args()

    cfg = load_config(args.config)
    ds = cfg["dataset"]
    ep = cfg["epoching"]
    tag = ds["name"]
    print(f"=== Barrido de shrinkage | {tag} | sujetos={args.subjects} ===\n")

    # Cargar cada sujeto UNA vez (reutilizado para todos los γ).
    loaded = []
    for s in args.subjects:
        try:
            d = load_dataset(name=ds["name"], subjects=[s], classes=ds.get("classes"),
                             tmin=ep["tmin"], tmax=ep["tmax"], picks=ep.get("picks", "eeg"),
                             baseline=ep.get("baseline"))
            loaded.append((s, d.X, d.y, d.sfreq))
        except Exception as exc:
            print(f"  (sujeto {s} omitido: {type(exc).__name__})")
    print(f"Sujetos cargados: {len(loaded)}\n")

    means, stds = [], []
    print(f"{'γ':>6} | {'acc media ± std':>18}")
    print("-" * 30)
    for g in args.gammas:
        cfg_g = copy.deepcopy(cfg)
        cfg_g["csp"]["shrinkage"] = g
        accs = [evaluate_kfold(cfg_g, X, y, fs=fs, n_splits=args.folds).accuracy
                for _, X, y, fs in loaded]
        m, sd = float(np.mean(accs)), float(np.std(accs))
        means.append(m); stds.append(sd)
        print(f"{g:6.2f} | {m:.3f} ± {sd:.3f}")

    best_i = int(np.argmax(means))
    print(f"\nMejor γ = {args.gammas[best_i]:.2f}  ->  {means[best_i]:.3f} "
          f"(γ=0 daba {means[0]:.3f}, mejora {means[best_i]-means[0]:+.3f})")

    # ---- Figura ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4))
    g = np.array(args.gammas)
    means = np.array(means); stds = np.array(stds)
    ax.plot(g, means, "o-", color="C1")
    ax.fill_between(g, means - stds, means + stds, alpha=0.15, color="C1")
    ax.axhline(means[0], color="gray", ls="--", lw=1, label=f"sin reg. (γ=0) = {means[0]:.3f}")
    ax.scatter([g[best_i]], [means[best_i]], color="C3", zorder=5,
               label=f"mejor γ={g[best_i]:.2f} = {means[best_i]:.3f}")
    ax.set(title=f"Efecto de la regularización CSP — {tag}",
           xlabel="shrinkage γ", ylabel="accuracy k-fold (media sobre sujetos)")
    ax.legend(); ax.grid(alpha=0.3)
    out = FIG_DIR / f"shrinkage_sweep_{tag}.png"
    fig.tight_layout(); fig.savefig(out, dpi=120); plt.close(fig)
    print(f"Figura: {out}")


if __name__ == "__main__":
    main()
