"""Entrena un EEGNet POOLED (cross-subject) y lo evalúa con LOSO.

A diferencia de ``train_model.py`` (CSP+LDA y EEGNet **sujeto-específicos**), este
script prueba la GENERALIZACIÓN entre sujetos: entrena una sola red con varios
sujetos juntos y mide, con *leave-one-subject-out* (LOSO), cómo se comportaría con
un usuario NUEVO sin calibración. El modelo final (entrenado con todos los sujetos)
se guarda como modelo BASE del que partiría un futuro fine-tuning con calibración corta.

Uso:
    python scripts/train_eegnet_pooled.py --config ../configs/default.yaml --subjects 1 2 3 4 5 6 7 8 9
    python scripts/train_eegnet_pooled.py --config ../configs/default.yaml --subjects 1 2 3 --epochs 200 --no-loso
"""
from __future__ import annotations

import argparse

from bci.config import load_config, resolve_path
from bci.pipeline.training import save_model, train_eegnet_pooled


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None, help="YAML del dataset (p. ej. ../configs/default.yaml)")
    ap.add_argument("--subjects", type=int, nargs="+", required=True, help="sujetos a agrupar (>= 2 para LOSO)")
    ap.add_argument("--epochs", type=int, default=250)
    ap.add_argument("--no-loso", action="store_true", help="omitir LOSO (solo entrenar el modelo base)")
    ap.add_argument("--device", default=None, help="cuda | cpu (default: auto)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    dataset = cfg["dataset"]["name"]
    out_dir = resolve_path(cfg["paths"]["processed"])
    print(f"=== EEGNet POOLED (cross-subject) | {dataset} | sujetos={args.subjects} | {args.epochs} épocas ===\n")

    clf, card, _ = train_eegnet_pooled(
        cfg, dataset, args.subjects, epochs=args.epochs,
        do_loso=not args.no_loso, device=args.device,
    )

    loso = card.extra["loso_per_subject"]
    if loso:
        print("LOSO (entrena con los demás, evalúa en el sujeto excluido):")
        for s, acc in sorted(loso.items()):
            print(f"  sujeto {s:>2}: {acc:.3f}")
        print(f"  -> media LOSO = {card.extra['loso_mean']:.3f}\n")
    else:
        print("(LOSO omitido)\n")

    pkl_path, json_path = save_model(clf, card, out_dir)
    print(f"Modelo base pooled guardado (entrenado con TODOS los sujetos · {card.n_train} trials · device={card.extra['device']}):")
    print(f"  {pkl_path.name}\n  {json_path.name}")


if __name__ == "__main__":
    main()
