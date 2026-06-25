"""Entrena un EEGNet POOLED (cross-subject) y lo evalúa con LOSO.

A diferencia de ``train_model.py`` (CSP+LDA y EEGNet **sujeto-específicos**), este
script prueba la GENERALIZACIÓN entre sujetos: entrena una sola red con varios
sujetos juntos y mide, con *leave-one-subject-out* (LOSO), cómo se comportaría con
un usuario NUEVO sin calibración. El modelo final (entrenado con todos los sujetos)
se guarda como modelo BASE del que partiría un futuro fine-tuning con calibración corta.

Uso:
    python scripts/train_eegnet_pooled.py --config ../configs/default.yaml --subjects 1 2 3 4 5 6 7 8 9
    python scripts/train_eegnet_pooled.py --config ../configs/default.yaml --subjects 1 2 3 --epochs 200 --no-loso

Run "máximo" (muchos datos, ver docs/entrenamiento.md):
    python scripts/train_eegnet_pooled.py --config ../configs/kumar2024.yaml \
        --subjects {1..18} --epochs 300 --augment --loso-subset 25   # llaves: zsh y bash las separan
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
    ap.add_argument("--augment", action="store_true",
                    help="aumentación de datos en entrenamiento (ruido/desplazamiento/escala)")
    ap.add_argument("--augment-copies", type=int, default=2, help="copias aumentadas por trial (default 2)")
    ap.add_argument("--loso-subset", type=int, default=None,
                    help="evaluar LOSO solo en los primeros N sujetos (el modelo base usa TODOS)")
    ap.add_argument("--device", default=None, help="cuda | cpu (default: auto)")
    ap.add_argument("--no-save", action="store_true",
                    help="no guardar el modelo (útil para pruebas: no pisa un modelo bueno)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    dataset = cfg["dataset"]["name"]
    out_dir = resolve_path(cfg["paths"]["processed"])
    print(f"=== EEGNet POOLED (cross-subject) | {dataset} | {len(args.subjects)} sujetos | "
          f"{args.epochs} épocas | augment={args.augment} | loso_subset={args.loso_subset} ===\n", flush=True)

    clf, card, _ = train_eegnet_pooled(
        cfg, dataset, args.subjects, epochs=args.epochs,
        do_loso=not args.no_loso, device=args.device,
        augment=args.augment, augment_copies=args.augment_copies,
        loso_subset=args.loso_subset, verbose=True,
    )

    loso = card.extra["loso_per_subject"]
    if loso:
        print("LOSO (entrena con los demás, evalúa en el sujeto excluido):")
        for s, acc in sorted(loso.items()):
            print(f"  sujeto {s:>2}: {acc:.3f}")
        print(f"  -> media LOSO = {card.extra['loso_mean']:.3f}\n")
    else:
        print("(LOSO omitido)\n")

    if args.no_save:
        print("(--no-save: modelo NO guardado)")
        return
    pkl_path, json_path = save_model(clf, card, out_dir)
    print(f"Modelo base pooled guardado (entrenado con TODOS los sujetos · {card.n_train} trials · device={card.extra['device']}):")
    print(f"  {pkl_path.name}\n  {json_path.name}")


if __name__ == "__main__":
    main()
