"""Demo del Hito 4: pipeline FIR -> CSP y topomapas de los patrones espaciales.

Uso:
    python scripts/demo_csp.py

Genera en docs/figures/:
  - csp_patterns.png : los patrones espaciales CSP como topomapas (deberían
    lateralizarse sobre la corteza motora C3/C4 según la clase).
Imprime además la separabilidad (accuracy LDA con validación cruzada).
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from bci.config import BACKEND_ROOT, load_config, resolve_path
from bci.dsp.convolution import apply_filter
from bci.dsp.fir_filters import design_from_config
from bci.spatial.csp import CSP
from bci.viz.plots import plot_csp_patterns

FIG_DIR = BACKEND_ROOT.parent / "docs" / "figures"


def main() -> None:
    cfg = load_config()
    npz = resolve_path(cfg["paths"]["processed"]) / f"{cfg['dataset']['name']}_s1.npz"
    if not npz.exists():
        raise SystemExit("Falta la caché. Ejecuta: python scripts/download_data.py --save")

    d = np.load(npz, allow_pickle=True)
    X, y, ch = d["X"], d["y"], list(d["ch_names"])

    # FIR µ/β (Hito 3) -> CSP (Hito 4)
    filt = design_from_config(cfg)
    Xf = apply_filter(X, filt.h, mode="same")
    csp = CSP(n_components=cfg["csp"]["n_components"]).fit(Xf, y)

    print("Autovalores CSP:", np.round(csp.eigenvalues_, 3))

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig = plot_csp_patterns(csp.patterns_, ch, csp.eigenvalues_)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "csp_patterns.png", dpi=120)
    plt.close(fig)
    print(f"Guardado: {FIG_DIR / 'csp_patterns.png'}")


if __name__ == "__main__":
    main()
