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
from bci.dsp.frequency_response import frequency_response
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


def fig_epoching(cfg: dict, fs: float) -> None:
    """Línea de tiempo del trial: epoch (FIR completo) + recorte de clasificación + transitorio FIR."""
    epo, cw, fir = cfg["epoching"], cfg["classification_window"], cfg["fir_filter"]
    t0, t1 = float(epo["tmin"]), float(epo["tmax"])
    cs, ce = t0 + float(cw["tmin_rel"]), t0 + float(cw["tmax_rel"])
    gd = (int(fir["num_taps"]) - 1) / 2 / fs   # retardo de grupo (s)

    fig, ax = plt.subplots(figsize=(10, 2.8))
    ax.set_xlim(-0.3, t1 + 0.5); ax.set_ylim(0, 3); ax.axis("off")
    # eje de tiempo
    ax.annotate("", xy=(t1 + 0.4, 0.5), xytext=(-0.2, 0.5), arrowprops=dict(arrowstyle="->", color="#94a3b8"))
    for tt in range(0, int(t1) + 1):
        ax.plot([tt, tt], [0.44, 0.56], color="#cbd5e1")
        ax.text(tt, 0.18, f"{tt}s", ha="center", fontsize=8, color="#64748b")
    ax.text(t1 + 0.45, 0.18, "t", fontsize=9, color="#64748b")
    ax.plot([0, 0], [0.5, 2.5], ":", color="#94a3b8", lw=1)
    ax.text(0, 2.65, "onset del trial", ha="center", fontsize=8.5, color="#475569")
    # epoch (se filtra entero)
    ax.add_patch(FancyBboxPatch((t0, 1.15), t1 - t0, 0.7, boxstyle="round,pad=0.01",
                                facecolor="#e0f2fe", edgecolor="#0891b2", lw=1.5))
    ax.text((t0 + t1) / 2, 2.05, f"epoch [{t0:g}, {t1:g}] s — se filtra (FIR) entero",
            ha="center", fontsize=9, color="#075985")
    # transitorio FIR en los bordes
    for xr in (t0, t1 - gd):
        ax.add_patch(plt.Rectangle((xr, 1.15), gd, 0.7, facecolor="#fecaca", alpha=0.7, edgecolor="none"))
    ax.text(t0 + gd / 2, 0.95, f"~{gd*1000:.0f} ms\ntransitorio", ha="center", fontsize=6.5, color="#b91c1c")
    # ventana de clasificación (recorte)
    ax.add_patch(FancyBboxPatch((cs, 1.22), ce - cs, 0.56, boxstyle="round,pad=0.01",
                                facecolor="#ddd6fe", edgecolor="#7c3aed", lw=1.5))
    ax.text((cs + ce) / 2, 1.5, f"clasificación\n[{cs:g}, {ce:g}] s", ha="center", fontsize=8.5, color="#5b21b6")
    fig.suptitle("Epoching: del trial al recorte de imaginación (BCI IV 2a)", fontsize=11, y=1.0)
    _save(fig, "01-epoching.png")


def fig_eegnet(cfg: dict, dataset: str, subject: int, filt, ch: list[str]) -> None:
    """EEGNet como espejo: filtros temporales/espaciales APRENDIDOS vs FIR/CSP a mano.

    Carga el modelo EEGNet ya persistido (no reentrena). Si no existe (o no hay
    torch), se omite con un aviso, para no romper la generación del resto.
    """
    from bci.pipeline.training import load_model, model_paths

    pkl, _ = model_paths(resolve_path(cfg["paths"]["processed"]), dataset, subject, "eegnet")
    if not pkl.exists():
        print(f"  · (omito EEGNet: no existe {pkl.name}; entrena con train_eegnet.py)")
        return
    try:
        clf = load_model(pkl)
        mdl = clf.model
    except Exception as e:  # noqa: BLE001
        print(f"  · (omito EEGNet: no se pudo cargar el modelo: {e})")
        return

    Wt = mdl.temporal[0].weight.detach().cpu().numpy()[:, 0, 0, :]   # (F1, kern)
    Ws = mdl.spatial[0].weight.detach().cpu().numpy()[:, 0, :, 0]    # (F1*D, n_canales)
    fs = filt.fs   # misma frecuencia de muestreo que el FIR a mano

    # --- (1) Respuesta en frecuencia: filtros temporales aprendidos vs FIR ---
    fr0 = frequency_response(Wt[0], fs, n_freqs=256)
    freqs = fr0.freqs_hz
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    ax.axvspan(8, 30, color="#059669", alpha=0.12, label="banda µ/β (FIR a mano)")
    for h in Wt:
        mag = frequency_response(h, fs, n_freqs=256).magnitude
        ax.plot(freqs, mag / (mag.max() + 1e-12), color="#7c3aed", alpha=0.5, lw=1.2)
    # el FIR a mano, normalizado, para comparar
    fr_fir = frequency_response(filt.h, fs, n_freqs=256)
    mfir = fr_fir.magnitude
    ax.plot(fr_fir.freqs_hz, mfir / (mfir.max() + 1e-12), color="#0891b2", lw=2.4,
            label="FIR diseñado a mano")
    ax.plot([], [], color="#7c3aed", alpha=0.6, lw=1.2, label="filtros temporales aprendidos (EEGNet)")
    ax.set_xlim(0, 45); ax.set_ylim(0, 1.05)
    ax.set_xlabel("Frecuencia (Hz)"); ax.set_ylabel("|H| (normalizado)")
    ax.set_title("Filtros temporales aprendidos (EEGNet) vs FIR a mano")
    ax.legend(fontsize=8.5, loc="upper right")
    ax.grid(alpha=0.25)
    _save(fig, "05-eegnet-temporal.png")

    # --- (2) Filtros espaciales aprendidos como topomapas (≈ CSP) ------------
    # Tomamos los 4 primeros para no saturar; Ws es (F1*D, n_canales) -> columnas.
    k = min(4, Ws.shape[0])
    _save(plot_csp_patterns(Ws[:k].T, ch, title="Filtros espaciales aprendidos (EEGNet ≈ CSP)"),
          "05-eegnet-spatial.png")


def fig_results(cfg: dict) -> None:
    """Comparación de los 4 regímenes (CSP/EEGNet × within/cross) por dataset.

    Lee los ``compare_methods_<id>.csv` ya producidos por la evaluación (no recalcula).
    Barras = accuracy media por régimen; la línea punteada es el azar (0.5).
    """
    processed = resolve_path(cfg["paths"]["processed"])
    datasets = [("BNCI2014_001", "BCI IV 2a"), ("BNCI2014_004", "BCI IV 2b"), ("Kumar2024", "Kumar2024")]
    regimes = [("csp_within", "CSP within"), ("eegnet_within", "EEGNet within"),
               ("csp_cross", "CSP cross"), ("eegnet_cross", "EEGNet cross")]
    colors = ["#7c3aed", "#d97706", "#a78bfa", "#fbbf24"]

    import csv as _csv
    means: dict[str, list[float]] = {}
    for ds_id, _ in datasets:
        p = processed / f"compare_methods_{ds_id}.csv"
        if not p.exists():
            print(f"  · (omito Resultados: falta {p.name})")
            return
        with p.open() as fh:
            rows = list(_csv.DictReader(fh))
        means[ds_id] = [float(np.mean([float(r[col]) for r in rows if r.get(col)]))
                        for col, _ in regimes]

    x = np.arange(len(datasets))
    w = 0.2
    fig, ax = plt.subplots(figsize=(8.5, 4.4))
    for j, (col, lab) in enumerate(regimes):
        vals = [means[ds_id][j] for ds_id, _ in datasets]
        ax.bar(x + (j - 1.5) * w, vals, w, label=lab, color=colors[j])
    ax.axhline(0.5, ls="--", color="#94a3b8", lw=1, label="azar (0.5)")
    ax.set_xticks(x); ax.set_xticklabels([lbl for _, lbl in datasets])
    ax.set_ylabel("Accuracy media (held-out honesto)")
    ax.set_ylim(0, 0.8)
    ax.set_title("Los 4 regímenes por dataset (within vs cross · CSP vs EEGNet)")
    ax.legend(fontsize=8.5, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    ax.grid(axis="y", alpha=0.25)
    _save(fig, "06-regimenes.png")


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

    # MAC didáctico: señal de juguete (seno) + kernel corto de suavizado, escalas
    # comparables y solape completo en n=20 (la señal cruda es minúscula frente a h).
    t_toy = np.arange(40)
    x_toy = np.sin(2 * np.pi * t_toy / 12)
    w9 = np.hamming(9)
    h_toy = w9 / w9.sum()
    _save(plot_mac_operation(x_toy, h_toy, 20), "02-mac.png")

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

    # --- EEGNet (sección 5): filtros aprendidos vs FIR/CSP a mano -----------
    fig_eegnet(cfg, cfg["dataset"]["name"], args.subject, filt, ch)

    # --- Resultados (sección 6): comparación de los 4 regímenes -------------
    fig_results(cfg)

    fig_pipeline()
    fig_epoching(cfg, fs)
    print(f"\nListo. Figuras en {FIG_DIR}")


if __name__ == "__main__":
    main()
