"""Demo del Hito 3: genera las figuras didácticas del pipeline LTI.

Uso:
    python scripts/demo_dsp.py            # usa el trial 0 de la caché del 2a
    python scripts/demo_dsp.py --trial 5 --channel Cz

Genera en docs/figures/:
  - fir_impulse_response.png   : h[n] (coeficientes del FIR)
  - fir_frequency_response.png : |H(e^jω)| con la banda µ/β resaltada
  - filter_effect.png          : un trial antes/después (tiempo y frecuencia)
  - mac_operation.png          : la operación MAC en una muestra (señal de juguete)
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from bci.config import BACKEND_ROOT, load_config, resolve_path
from bci.dsp.convolution import apply_filter
from bci.dsp.fir_filters import design_from_config
from bci.viz.plots import (
    plot_filter_effect,
    plot_frequency_response,
    plot_impulse_response,
    plot_mac_operation,
)

FIG_DIR = BACKEND_ROOT.parent / "docs" / "figures"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trial", type=int, default=0)
    ap.add_argument("--channel", default="C3")
    args = ap.parse_args()

    cfg = load_config()
    filt = design_from_config(cfg)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    # 1) Respuesta al impulso
    ax = plot_impulse_response(filt)
    ax.figure.tight_layout(); ax.figure.savefig(FIG_DIR / "fir_impulse_response.png", dpi=120)
    plt.close(ax.figure)

    # 2) Respuesta en frecuencia
    ax = plot_frequency_response(filt)
    ax.figure.tight_layout(); ax.figure.savefig(FIG_DIR / "fir_frequency_response.png", dpi=120)
    plt.close(ax.figure)

    # 3) Efecto del filtro sobre un trial real
    npz = resolve_path(cfg["paths"]["processed"]) / f"{cfg['dataset']['name']}_s1.npz"
    if npz.exists():
        d = np.load(npz, allow_pickle=True)
        X, ch = d["X"], list(d["ch_names"])
        fs = float(d["sfreq"])
        c = ch.index(args.channel) if args.channel in ch else 0
        x_raw = X[args.trial, c]
        x_filt = apply_filter(x_raw, filt.h, mode="same")
        axes = plot_filter_effect(x_raw, x_filt, fs)
        fig = axes[0].figure
        fig.suptitle(f"Efecto del FIR µ/β — trial {args.trial}, canal {ch[c]}")
        fig.tight_layout(); fig.savefig(FIG_DIR / "filter_effect.png", dpi=120)
        plt.close(fig)
    else:
        print(f"(aviso) no hay caché {npz.name}; omito filter_effect.png. "
              f"Ejecuta scripts/download_data.py --save")

    # 4) Operación MAC con una señal de juguete (clara de leer)
    x_toy = np.array([0, 1, 2, 3, 2, 1, 0, -1, -2, -1, 0], float)
    h_toy = np.array([0.2, 0.5, 0.2], float)
    ax = plot_mac_operation(x_toy, h_toy, n=4)
    ax.figure.tight_layout(); ax.figure.savefig(FIG_DIR / "mac_operation.png", dpi=120)
    plt.close(ax.figure)

    print(f"Figuras guardadas en {FIG_DIR}")
    for p in sorted(FIG_DIR.glob("*.png")):
        print(f"  - {p.name}")


if __name__ == "__main__":
    main()
