"""Precomputa y persiste los payloads de visualización del MUNDO OFFLINE.

Objetivo: PORTABILIDAD. Tras correr esto, la web puede mostrar las páginas offline
(El Modelo/CSP, EEGNet, Entrenamiento, Resultados) **sin datos crudos** — el servidor
sirve los JSON precomputados que quedan junto a los modelos (en ``paths.processed``).
Los datos crudos solo hacen falta para el streaming en vivo del dataset de demo.

Qué genera, por (dataset, sujeto) con modelo en disco:
  - ``viz_*_any_info.json`` / ``viz_*_any_positions.json``  (canales, fs, posiciones)
  - ``viz_*_csp_lda_csp.json``         (patrones CSP + nube de entrenamiento)
  - ``viz_*_csp_lda_csp_signal.json``  (señal cruda vs componente CSP, TODOS los comp.)
  - ``viz_*_csp_lda_lda.json``         (frontera + confusión sobre el held-out)
  - ``viz_*_eegnet_eegnet.json``       (filtros aprendidos; solo si hay modelo EEGNet)

Reusa los MISMOS constructores que el servidor (``bci.server.payloads``): el JSON
precomputado y el calculado al vuelo son idénticos byte a byte.

Uso:
    python scripts/precompute_payloads.py --config ../configs/default.yaml
    python scripts/precompute_payloads.py --config ../configs/default.yaml --subjects 1 2 3
    python scripts/precompute_payloads.py --config ../configs/default.yaml --no-eegnet
"""
from __future__ import annotations

import argparse
import time

import numpy as np

from bci.config import load_config, resolve_path
from bci.datasets.moabb_loader import load_from_config
from bci.pipeline.training import load_card, load_model, model_paths
from bci.server import payloads as pl


def _split_from_card(data, card: dict) -> tuple[np.ndarray, np.ndarray]:
    """(idx_train, idx_demo) según la ficha — mismo criterio que el servidor (_split_idx)."""
    n = data.n_trials
    spec = card["holdout"]
    if spec["by"] == "subject":
        idx_demo = np.arange(n)                               # todo el sujeto es held-out
    elif spec["by"] == "session":
        sess = data.metadata["session"].to_numpy()
        idx_demo = np.where(sess == spec["value"])[0]
    else:
        idx_demo = np.array(spec["indices"], dtype=int)
    demo_set = set(idx_demo.tolist())
    idx_train = np.array([i for i in range(n) if i not in demo_set], dtype=int)
    return idx_train, idx_demo


def _discover_subjects(out_dir, dataset: str) -> list[int]:
    """Sujetos con un modelo CSP+LDA within en disco (los que se pueden visualizar)."""
    subs = []
    for p in sorted(out_dir.glob(f"model_{dataset}_s*_csp_lda.json")):
        stem = p.stem  # model_<ds>_s<NN>_csp_lda
        try:
            subs.append(int(stem.split("_s")[1].split("_")[0]))
        except (IndexError, ValueError):
            continue
    return sorted(set(subs))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None, help="YAML del dataset (p. ej. ../configs/default.yaml)")
    ap.add_argument("--subjects", type=int, nargs="+", default=None,
                    help="sujetos (default: todos los que tengan modelo csp_lda en disco)")
    ap.add_argument("--no-eegnet", action="store_true", help="omitir el payload de EEGNet")
    args = ap.parse_args()

    cfg = load_config(args.config)
    dataset = cfg["dataset"]["name"]
    out_dir = resolve_path(cfg["paths"]["processed"])

    subjects = args.subjects or _discover_subjects(out_dir, dataset)
    if not subjects:
        print(f"No hay modelos csp_lda en {out_dir} para {dataset}. Entrena primero "
              f"(scripts/train_all_regimes.py).")
        return

    print(f"=== precompute_payloads | {dataset} | {len(subjects)} sujetos → {out_dir} ===\n",
          flush=True)
    t0 = time.time()
    n_files = 0
    for s in subjects:
        pkl_path, json_path = model_paths(out_dir, dataset, s, "csp_lda")
        if not (pkl_path.exists() and json_path.exists()):
            print(f"  sujeto {s:>3}: ✗ sin modelo csp_lda, omitido", flush=True)
            continue

        # Datos crudos del sujeto (solo aquí, en el precompute; la web ya no los necesita).
        cfg_s = {**cfg, "dataset": {**cfg["dataset"], "subjects": [s]}}
        data = load_from_config(cfg_s)
        pipe = load_model(pkl_path)
        card = load_card(json_path)
        idx_train, idx_demo = _split_from_card(data, card)

        # Payloads independientes del método (canales/posiciones/info).
        pl.save_payload(out_dir, dataset, s, "any", "info",
                        pl.build_info_payload(dataset, s, data)); n_files += 1
        pl.save_payload(out_dir, dataset, s, "any", "positions",
                        pl.build_positions_payload(data.ch_names)); n_files += 1
        # Payloads del régimen CSP+LDA (within).
        pl.save_payload(out_dir, dataset, s, "csp_lda", "csp",
                        pl.build_csp_payload(data, pipe, idx_train)); n_files += 1
        pl.save_payload(out_dir, dataset, s, "csp_lda", "csp_signal",
                        pl.build_csp_signal_all(data, pipe)); n_files += 1
        pl.save_payload(out_dir, dataset, s, "csp_lda", "lda",
                        pl.build_lda_payload(data, pipe, idx_train, idx_demo, cfg, card)); n_files += 1
        msg = "csp,csp_signal,lda,info,positions"

        # Payload de EEGNet (si hay modelo): no necesita datos crudos, solo el .pkl.
        if not args.no_eegnet:
            e_pkl, e_json = model_paths(out_dir, dataset, s, "eegnet")
            if e_pkl.exists() and e_json.exists():
                clf = load_model(e_pkl)
                e_card = load_card(e_json)
                pl.save_payload(out_dir, dataset, s, "eegnet", "eegnet",
                                pl.build_eegnet_payload(clf, e_card, card)); n_files += 1
                msg += ",eegnet"

        print(f"  sujeto {s:>3}: ✓ {msg}", flush=True)

    print(f"\n=== Listo en {time.time()-t0:.0f}s | {n_files} payloads escritos ===")


if __name__ == "__main__":
    main()
