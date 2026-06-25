"""Ensambla la matriz 2×2 de Resultados (compare_methods_<id>.csv) desde las FICHAS
de los modelos ya entrenados — sin recalcular nada.

A diferencia de ``compare_methods.py`` (que reentrena k-fold + LOSO desde cero, caro),
aquí solo se LEEN las fichas ``.json`` que ``train_all_regimes.py`` ya dejó en disco y
se vuelcan a un CSV con las cuatro celdas por sujeto:

  subject, csp_within, csp_cross, eegnet_within, eegnet_cross

Ventaja: los números del CSV coinciden EXACTAMENTE con los modelos que la web sirve en
vivo (misma red, mismo split), y el coste es leer JSON. Celdas vacías = ese (sujeto,
régimen) aún no se entrenó (p. ej. cross solo de un subconjunto en datasets grandes).

Convención de métricas (coherente con la página):
  - *_within: número CALIBRADO. CSP = k-fold (de results_<tag>.csv si está; si no, la
    inter-sesión de la ficha). EEGNet = k-fold de su ficha (extra.accuracy_kfold).
  - *_cross : accuracy cross-subject (persona nueva) de la ficha del modelo cross.

Uso:
    python scripts/assemble_compare.py --config ../configs/kumar2024.yaml
"""
from __future__ import annotations

import argparse

import pandas as pd

from bci.config import load_config, resolve_path
from bci.pipeline.training import load_card, model_paths


def _card(out_dir, dataset, subject, method):
    _, js = model_paths(out_dir, dataset, subject, method)
    return load_card(js) if js.exists() else None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    dataset = cfg["dataset"]["name"]
    out_dir = resolve_path(cfg["paths"]["processed"])

    # k-fold within de CSP desde results_<tag>.csv (si existe), para el número calibrado.
    kfold_csp: dict[int, float] = {}
    res = out_dir / f"results_{dataset}.csv"
    if res.exists():
        df = pd.read_csv(res)
        if "kfold_acc" in df.columns:
            kfold_csp = {int(r["subject"]): float(r["kfold_acc"]) for _, r in df.iterrows()}

    # Descubre los sujetos con CUALQUIER modelo en disco.
    subjects = sorted({
        int(p.stem.split("_s")[1].split("_")[0])
        for p in out_dir.glob(f"model_{dataset}_s*.json")
    })
    rows = []
    for s in subjects:
        cw = _card(out_dir, dataset, s, "csp_lda")
        cc = _card(out_dir, dataset, s, "csp_lda_cross")
        ew = _card(out_dir, dataset, s, "eegnet")
        ec = _card(out_dir, dataset, s, "eegnet_cross")
        if not any([cw, cc, ew, ec]):
            continue
        rows.append({
            "subject": s,
            # within calibrado: CSP k-fold (de results CSV) o, en su defecto, inter-sesión.
            "csp_within": kfold_csp.get(s, cw["accuracy"] if cw else None),
            "csp_cross": cc["accuracy"] if cc else None,
            # EEGNet within calibrado = k-fold de su ficha (coherente con CSP k-fold).
            "eegnet_within": (ew.get("extra", {}) or {}).get("accuracy_kfold") if ew else None,
            "eegnet_cross": ec["accuracy"] if ec else None,
        })

    if not rows:
        print(f"No hay fichas de modelos para {dataset} en {out_dir}.")
        return

    df_out = pd.DataFrame(rows)
    out = out_dir / f"compare_methods_{dataset}.csv"
    df_out.to_csv(out, index=False)
    n = lambda c: df_out[c].notna().sum()  # noqa: E731
    print(f"=== compare_methods ensamblado | {dataset} | {len(rows)} sujetos ===")
    print(f"  celdas con dato: csp_within={n('csp_within')} csp_cross={n('csp_cross')} "
          f"eegnet_within={n('eegnet_within')} eegnet_cross={n('eegnet_cross')}")
    print(f"  guardado: {out}")


if __name__ == "__main__":
    main()
