"""Genera las figuras del INFORME (docs/informe/figures/) a partir del código real.

Usa la caché local del dataset BCI IV 2a (sujeto 1) — `data/processed/BNCI2014_001_s1.npz`,
creada por `scripts/download_data.py --save` — así que NO descarga nada de MOABB. Cada figura
se construye con las MISMAS funciones del pipeline que usa la app (FIR, CSP, log-varianza, LDA),
de modo que lo del informe coincide con lo que se programó.

Uso:
    cd backend
    python scripts/make_report_figures.py [--config ../configs/default.yaml] [--subject 1]

Genera (numeradas por sección del informe):
  00-pipeline.png          esquema FIR→CSP→log-var→LDA (sección 0)
  02-fir-impulse.png       respuesta al impulso h[n] (sección 2)
  02-fir-frequency.png     |H(e^jω)| con la banda µ/β (sección 2)
  02-filter-effect.png     un trial: crudo vs filtrado, tiempo y frecuencia (sección 2)
  02-mac.png               la operación MAC en una muestra (sección 2)
  03-csp-patterns.png      patrones espaciales CSP como topomapas (sección 3)
  03-logvar-separability.png  nube de log-varianza por clase (sección 3)
  04-lda-boundary.png      la misma nube + la frontera lineal del LDA (sección 4)
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np

from bci.config import BACKEND_ROOT, load_config, resolve_path
from bci.dsp.convolution import apply_filter
from bci.dsp.fir_filters import design_from_config
from bci.features.log_variance import log_variance
from bci.models.lda import LDA
from bci.spatial.csp import CSP
from bci.viz.plots import (
    plot_csp_patterns, plot_filter_effect, plot_frequency_response,
    plot_impulse_response, plot_mac_operation,
)

FIG_DIR = BACKEND_ROOT.parent / "docs" / "informe" / "figures"
CLASS_COLORS = ["#2563eb", "#e11d48"]   # mismos colores que el frontend


def _fig_of(obj):
    """Resuelve la Figure de un objeto que puede ser Figure, Axes o array de Axes."""
    if hasattr(obj, "savefig"):
        return obj
    if isinstance(obj, np.ndarray):
        return obj.flat[0].figure
    if isinstance(obj, (list, tuple)):
        return obj[0].figure
    return obj.figure


def _save(obj, name: str) -> None:
    fig = _fig_of(obj)
    fig.savefig(FIG_DIR / name, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {name}")


def fig_pipeline() -> None:
    """Esquema del pipeline central (espejo del diagrama de la página El Modelo)."""
    stages = [
        ("Entrada EEG", "22 canales × tiempo", "#64748b"),
        ("FIR (fijo)", "banda µ/β 8–30 Hz\n(convolución)", "#0891b2"),
        ("CSP", "Z = W·X\n(filtro espacial)", "#7c3aed"),
        ("log-varianza", "energía por\ncomponente", "#d97706"),
        ("LDA", "y = w·F + b\n(frontera)", "#059669"),
    ]
    fig, ax = plt.subplots(figsize=(12, 2.6))
    ax.set_xlim(0, len(stages) * 2.4)
    ax.set_ylim(0, 2)
    ax.axis("off")
    for i, (name, sub, color) in enumerate(stages):
        x = i * 2.4 + 0.15
        box = FancyBboxPatch((x, 0.35), 1.9, 1.3, boxstyle="round,pad=0.05",
                             linewidth=2, edgecolor=color, facecolor="white")
        ax.add_patch(box)
        ax.text(x + 0.95, 1.30, name, ha="center", va="center", fontsize=11, fontweight="bold", color="#1e293b")
        ax.text(x + 0.95, 0.78, sub, ha="center", va="center", fontsize=8.5, color="#475569")
        if i < len(stages) - 1:
            ax.add_patch(FancyArrowPatch((x + 1.95, 1.0), (x + 2.25, 1.0),
                                         arrowstyle="-|>", mutation_scale=16, color="#94a3b8"))
    fig.suptitle("Pipeline LTI: de la señal cruda a la decisión", fontsize=12, y=1.02)
    _save(fig, "00-pipeline.png")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(BACKEND_ROOT.parent / "configs" / "default.yaml"))
    ap.add_argument("--subject", type=int, default=1)
    args = ap.parse_args()

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    cfg = load_config(args.config)

    npz = resolve_path(cfg["paths"]["processed"]) / f"{cfg['dataset']['name']}_s{args.subject}.npz"
    if not npz.exists():
        raise SystemExit(f"Falta la caché {npz}. Ejecuta: python scripts/download_data.py --save")
    d = np.load(npz, allow_pickle=True)
    X, y, ch = d["X"], d["y"].astype(str), list(d["ch_names"])
    fs = float(d["sfreq"])
    classes = sorted(set(y))
    print(f"Dataset {cfg['dataset']['name']} s{args.subject}: {X.shape[0]} trials, {len(ch)} canales, {fs:.0f} Hz")

    # --- FIR (sección 2) -----------------------------------------------------
    filt = design_from_config(cfg, fs=fs)
    Xf = apply_filter(X, filt.h, mode="same")
    print("Figuras:")
    _save(plot_impulse_response(filt), "02-fir-impulse.png")
    _save(plot_frequency_response(filt), "02-fir-frequency.png")

    ci = ch.index("C3") if "C3" in ch else 0
    _save(plot_filter_effect(X[0, ci], Xf[0, ci], fs), "02-filter-effect.png")

    # MAC sobre una señal de juguete + el propio h (didáctico)
    _save(plot_mac_operation(X[0, ci, :60], filt.h, len(filt.h)), "02-mac.png")

    # --- CSP (sección 3) -----------------------------------------------------
    csp = CSP(n_components=cfg["csp"]["n_components"], shrinkage=cfg["csp"].get("shrinkage", 0.0)).fit(Xf, y)
    _save(plot_csp_patterns(csp.patterns_, ch, csp.eigenvalues_), "03-csp-patterns.png")

    # log-varianza: nube de separabilidad (comp 0 vs comp extremo)
    F = log_variance(csp.transform(Xf))
    last = F.shape[1] - 1
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    for k, cls in enumerate(classes):
        sel = y == cls
        ax.scatter(F[sel, 0], F[sel, last], s=22, alpha=0.6, color=CLASS_COLORS[k % 2], label=cls)
    ax.set_xlabel(f"log-var comp 0  →  {classes[0]}")
    ax.set_ylabel(f"log-var comp {last}  →  {classes[1]}")
    ax.set_title("Separabilidad de clases (espacio de características)")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.25)
    _save(fig, "03-logvar-separability.png")

    # --- LDA (sección 4): misma nube + frontera lineal -----------------------
    F2 = F[:, [0, last]]
    lda = LDA().fit(F2, y)
    w = lda.coef_[0] - lda.coef_[1]
    b = float(lda.intercept_[0] - lda.intercept_[1])
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    for k, cls in enumerate(classes):
        sel = y == cls
        ax.scatter(F2[sel, 0], F2[sel, 1], s=22, alpha=0.6, color=CLASS_COLORS[k % 2], label=cls)
    xs = np.array([F2[:, 0].min(), F2[:, 0].max()])
    if abs(w[1]) > 1e-9:
        ax.plot(xs, -(w[0] * xs + b) / w[1], "k--", lw=2, label="frontera LDA")
    ax.set_xlabel(f"log-var comp 0  →  {classes[0]}")
    ax.set_ylabel(f"log-var comp {last}  →  {classes[1]}")
    ax.set_title("Frontera de decisión del LDA")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.25)
    _save(fig, "04-lda-boundary.png")

    fig_pipeline()
    print(f"\nListo. Figuras en {FIG_DIR}")


if __name__ == "__main__":
    main()
