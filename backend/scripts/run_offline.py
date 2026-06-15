"""Pipeline offline end-to-end (Hito 5): carga → FIR → CSP → log-var → LDA + métricas.

Uso:
    python scripts/run_offline.py                 # config por defecto
    python scripts/run_offline.py --subjects 1 2  # varios sujetos
    python scripts/run_offline.py --folds 10

Reporta dos evaluaciones:
  1. Validación cruzada estratificada (k-fold).
  2. Inter-sesión: entrenar en '0train', evaluar en '1test' (estimación honesta).
"""
from __future__ import annotations

import argparse

from bci.config import load_config
from bci.datasets.moabb_loader import load_from_config
from bci.pipeline.offline import evaluate_by_session, evaluate_kfold


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--subjects", type=int, nargs="+", default=None)
    ap.add_argument("--folds", type=int, default=5)
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.subjects is not None:
        cfg["dataset"]["subjects"] = args.subjects

    print(f"=== Pipeline offline | {cfg['dataset']['name']} "
          f"| sujetos={cfg['dataset']['subjects']} "
          f"| clases={cfg['dataset'].get('classes')} ===")
    data = load_from_config(cfg)
    print(data.summary())

    fc = cfg["fir_filter"]
    print(f"\nFIR µ/β: {fc['num_taps']} taps [{fc['low_hz']}-{fc['high_hz']}] Hz "
          f"@ fs={data.sfreq:g} Hz | CSP: {cfg['csp']['n_components']} componentes | LDA\n")

    print("[1] Validación cruzada estratificada")
    print(evaluate_kfold(cfg, data.X, data.y, fs=data.sfreq, n_splits=args.folds))

    if {"0train", "1test"} <= set(data.metadata["session"].unique()):
        print("[2] Evaluación inter-sesión (honesta)")
        print(evaluate_by_session(cfg, data.X, data.y, data.metadata, fs=data.sfreq))


if __name__ == "__main__":
    main()
