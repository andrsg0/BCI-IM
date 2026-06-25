"""Entrena y PERSISTE los 4 regímenes de un dataset (para el selector de la web).

Los 4 regímenes del proyecto, por sujeto:

  | Régimen          | Método                       | Archivo (.pkl/.json)            |
  |------------------|------------------------------|---------------------------------|
  | CSP+LDA within   | train_subject                | model_<ds>_s<N>_csp_lda         |
  | CSP+LDA cross    | train_crosssubject           | model_<ds>_s<N>_csp_lda_cross   |
  | EEGNet within    | train_eegnet_subject         | model_<ds>_s<N>_eegnet          |
  | EEGNet cross     | train_eegnet_crosssubject    | model_<ds>_s<N>_eegnet_cross    |

Cada modelo se guarda como `.pkl` (pesos) + `.json` (ficha) APENAS termina. La web los
CARGA por nombre de método (no entrena en vivo).

CHECKPOINT / REANUDABLE: por defecto se OMITE todo (sujeto, régimen) cuyo .pkl+.json ya
existan. Así, si el proceso se corta (cierre de terminal, corte de luz), basta volver a
lanzar el MISMO comando y retoma donde quedó — se pierde, como mucho, el modelo que se
estaba entrenando (40 s–7 min), nunca las horas ya hechas. Usa ``--force`` para reentrenar.
Para que sobreviva al cierre del terminal, lánzalo desligado, p. ej.:
    nohup python scripts/train_all_regimes.py --config ../configs/kumar2024.yaml \
        --subjects $(seq 1 18) --cross-subjects 18 7 9 > train_kumar.log 2>&1 &

COSTE: el **EEGNet cross** es el más caro (entrena una red con N-1 sujetos por cada
sujeto demo). Por eso se separa qué sujetos reciben modelos *cross* (``--cross-subjects``)
de los que reciben modelos *within* (``--subjects``): en datasets grandes conviene generar
los cross solo para el/los sujeto(s) de la demo (+ un subconjunto para la media de
Resultados). El pool de sujetos se carga UNA sola vez y se reutiliza para todos los cross.

Uso:
    # 2a completo (9 sujetos), los 4 regímenes:
    python scripts/train_all_regimes.py --config ../configs/default.yaml --subjects 1 2 3 4 5 6 7 8 9

    # within para todos, pero cross (caro) solo para algunos sujetos:
    python scripts/train_all_regimes.py --config ../configs/kumar2024.yaml \
        --subjects $(seq 1 18) --cross-subjects 18 7 9

    # sin EEGNet (rápido: solo CSP within + cross):
    python scripts/train_all_regimes.py --config ../configs/default.yaml --subjects 1 2 3 --no-eegnet
"""
from __future__ import annotations

import argparse
import time

from bci.config import load_config, resolve_path
from bci.datasets.moabb_loader import load_from_config
from bci.pipeline.training import (
    model_paths, save_model, train_crosssubject, train_eegnet_crosssubject,
    train_eegnet_subject, train_subject,
)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None, help="YAML del dataset (p. ej. ../configs/default.yaml)")
    ap.add_argument("--subjects", type=int, nargs="+", required=True,
                    help="sujetos para los modelos WITHIN (y pool de los cross)")
    ap.add_argument("--cross-subjects", type=int, nargs="+", default=None,
                    help="sujetos para los modelos CROSS (default: los mismos que --subjects)")
    ap.add_argument("--no-eegnet", action="store_true", help="omitir ambos regímenes EEGNet")
    ap.add_argument("--epochs", type=int, default=250, help="épocas de EEGNet")
    ap.add_argument("--device", default=None, help="cuda | cpu (default: auto)")
    ap.add_argument("--force", action="store_true",
                    help="reentrenar aunque el modelo ya exista (por defecto se omite = reanudable)")
    ap.add_argument("--no-save", action="store_true", help="no escribir a disco (prueba)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    dataset = cfg["dataset"]["name"]
    out_dir = resolve_path(cfg["paths"]["processed"])
    pool = list(args.subjects)                       # universo para los modelos cross
    cross_subjects = args.cross_subjects if args.cross_subjects is not None else pool
    do_eegnet = not args.no_eegnet

    print(f"=== 4 regímenes | {dataset} | within={pool} | cross={cross_subjects} | "
          f"eegnet={'sí' if do_eegnet else 'no'} | resume={'no (force)' if args.force else 'sí'} ===\n",
          flush=True)

    def _exists(s: int, method: str) -> bool:
        pkl, js = model_paths(out_dir, dataset, s, method)
        return pkl.exists() and js.exists()

    def _persist(model, card):
        if not args.no_save:
            save_model(model, card, out_dir)

    def _skip(s: int, method: str) -> bool:
        """True si ya está hecho (checkpoint): omite y deja constancia."""
        if not args.force and _exists(s, method):
            print(f"sujeto {s:>2} · {method}: ya existe, omitido (checkpoint)", flush=True)
            return True
        return False

    results: dict[int, dict] = {}
    t0 = time.time()

    # --- WITHIN-subject (un modelo por sujeto, calibrado) ---
    for s in pool:
        row = results.setdefault(s, {"subject": s})
        if not _skip(s, "csp_lda"):
            pipe, card, _ = train_subject(cfg, dataset, s)
            _persist(pipe, card)
            row["csp_within"] = card.accuracy
            print(f"sujeto {s:>2} · CSP within  = {card.accuracy:.3f}", flush=True)
        if do_eegnet and not _skip(s, "eegnet"):
            clf, ecard, _ = train_eegnet_subject(cfg, dataset, s, epochs=args.epochs)
            _persist(clf, ecard)
            row["eegnet_within"] = ecard.accuracy
            print(f"           · EEGNet within = {ecard.accuracy:.3f}", flush=True)

    # --- CROSS-subject (entrena con el pool MENOS el sujeto demo) ---
    if len(set(pool)) < 2:
        print("\n⚠ El pool tiene <2 sujetos: no se pueden entrenar modelos cross "
              "(amplía --subjects con más sujetos).", flush=True)
        cross_subjects = []

    # ¿Hace falta cargar el pool? Solo si hay algún cross pendiente (o --force).
    needed = [s for s in cross_subjects if s in pool]
    pending = [s for s in needed
               if args.force or not _exists(s, "csp_lda_cross")
               or (do_eegnet and not _exists(s, "eegnet_cross"))]
    data_pool = None
    if pending:
        print(f"\nCargando el pool de {len(pool)} sujetos UNA vez para los cross…", flush=True)
        cfg_pool = {**cfg, "dataset": {**cfg["dataset"], "subjects": pool}}
        data_pool = load_from_config(cfg_pool)

    for s in cross_subjects:
        if s not in pool:
            print(f"⚠ sujeto {s} no está en el pool {pool}: omitido para cross.", flush=True)
            continue
        row = results.setdefault(s, {"subject": s})
        if not _skip(s, "csp_lda_cross"):
            pipe, card, _ = train_crosssubject(cfg, dataset, pool, demo_subject=s, data=data_pool)
            _persist(pipe, card)
            row["csp_cross"] = card.accuracy
            print(f"sujeto {s:>2} · CSP cross   = {card.accuracy:.3f}  "
                  f"(train={card.extra['n_train_subjects']} suj.)", flush=True)
        if do_eegnet and not _skip(s, "eegnet_cross"):
            clf, ecard, _ = train_eegnet_crosssubject(cfg, dataset, pool, demo_subject=s,
                                                      epochs=args.epochs, device=args.device,
                                                      data=data_pool)
            _persist(clf, ecard)
            row["eegnet_cross"] = ecard.accuracy
            print(f"           · EEGNet cross  = {ecard.accuracy:.3f}", flush=True)

    # --- Resumen 2×2 (media sobre los sujetos entrenados en ESTA corrida) ---
    def _avg(key):
        vals = [r[key] for r in results.values() if key in r]
        return sum(vals) / len(vals) if vals else float("nan")

    print(f"\n=== Matriz 2×2 (media de lo entrenado ahora) | tiempo {time.time()-t0:.0f}s ===")
    print("              within-subject    cross-subject")
    print(f"  CSP+LDA :     {_avg('csp_within'):.3f}            {_avg('csp_cross'):.3f}")
    if do_eegnet:
        print(f"  EEGNet  :     {_avg('eegnet_within'):.3f}            {_avg('eegnet_cross'):.3f}")
    if not args.no_save:
        print(f"\nModelos guardados en: {out_dir}")


if __name__ == "__main__":
    main()
