"""Entrena y GUARDA el modelo de cada sujeto (mundo OFFLINE: "antes del streaming").

Este script es la frontera explícita entre las dos fases del proyecto:

  1) ENTRENAMIENTO (aquí): con una partición de entrenamiento se ajustan el CSP y el
     LDA, se mide la precisión honesta sobre una partición HELD-OUT y se persiste el
     modelo a disco. Esto ocurre UNA vez, antes de cualquier demo.
  2) STREAMING (en el servidor): se CARGA ese modelo ya entrenado y se le transmiten
     en tiempo real los trials held-out, que el modelo nunca vio. Sin fuga de datos.

Uso:
    python scripts/train_model.py                       # 2a, sujeto 1
    python scripts/train_model.py --subjects 1 2 3
    python scripts/train_model.py --config configs/physionet.yaml --subjects 1
"""
from __future__ import annotations

import argparse

from bci.config import load_config, resolve_path
from bci.pipeline.training import save_model, train_eegnet_subject, train_subject


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None, help="YAML del dataset (default: configs/default.yaml)")
    ap.add_argument("--subjects", type=int, nargs="+", default=[1])
    ap.add_argument("--eegnet", action="store_true", help="entrenar también EEGNet (lento, CPU)")
    ap.add_argument("--epochs", type=int, default=250, help="épocas de EEGNet")
    args = ap.parse_args()

    cfg = load_config(args.config)
    dataset = cfg["dataset"]["name"]
    out_dir = resolve_path(cfg["paths"]["processed"])
    print(f"=== Entrenamiento OFFLINE | {dataset} | sujetos={args.subjects} ===\n")

    for s in args.subjects:
        pipe, card, _ = train_subject(cfg, dataset, s)
        pkl_path, json_path = save_model(pipe, card, out_dir)
        split = (
            f"sesión '{card.train_session}'" if card.train_session
            else "fracción estratificada (70 %)"
        )
        print(
            f"sujeto {s} · CSP+LDA: entrenado con {split} "
            f"({card.n_train} trials) | demo held-out = {card.n_demo} trials\n"
            f"  accuracy honesta (held-out) = {card.accuracy:.3f} | kappa = {card.kappa:.3f}\n"
            f"  guardado: {pkl_path.name}\n"
        )
        if args.eegnet:
            print(f"sujeto {s} · EEGNet: entrenando {args.epochs} épocas (puede tardar)…")
            clf, ecard, _ = train_eegnet_subject(cfg, dataset, s, epochs=args.epochs)
            epkl, _ = save_model(clf, ecard, out_dir)
            print(
                f"  accuracy honesta (held-out) = {ecard.accuracy:.3f} | kappa = {ecard.kappa:.3f}\n"
                f"  guardado: {epkl.name}\n"
            )


if __name__ == "__main__":
    main()
