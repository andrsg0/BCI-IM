"""Pre-descarga TODOS los sujetos de un dataset de una sola vez (setup de máquina nueva).

Resuelve el dolor de "en una computadora nueva los datos no se bajan solos y hay que
ir sujeto por sujeto en la web para que se descarguen". Aquí, con un comando, se calienta
la caché de MNE (`~/mne_data`) con todos los sujetos del dataset, sujeto a sujeto (RAM
acotada) y tolerando fallos: si un sujeto no se puede bajar (timeout de OSF/Zenodo), se
omite y se sigue, reportando al final cuáles faltaron para reintentar.

Uso:
    # todos los sujetos del dataset del YAML (lo normal para el dataset de demo en vivo):
    python scripts/setup_data.py --config ../configs/default.yaml

    # un subconjunto:
    python scripts/setup_data.py --config ../configs/default.yaml --subjects 1 2 3

OJO: algunos datasets son ZIP monolíticos (p. ej. Kumar2024 = 4.47 GB con TODOS los
sujetos en un archivo): el primer sujeto baja el dataset entero y el resto va instantáneo.
"""
from __future__ import annotations

import argparse
import time

import moabb.datasets as mds

from bci.config import load_config
from bci.datasets.moabb_loader import load_dataset


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None, help="YAML del dataset (p. ej. ../configs/default.yaml)")
    ap.add_argument("--subjects", type=int, nargs="+", default=None,
                    help="sujetos a bajar (default: TODOS los del dataset)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    ds = cfg["dataset"]
    name = ds["name"]
    classes = ds.get("classes")
    tmin, tmax = cfg["epoching"]["tmin"], cfg["epoching"]["tmax"]

    subjects = args.subjects
    if subjects is None:                       # por defecto, todos los del catálogo MOABB
        subjects = list(getattr(mds, name)().subject_list)

    print(f"=== setup_data | {name} | {len(subjects)} sujetos → caché de MNE (~/mne_data) ===\n",
          flush=True)
    ok, fail = [], []
    t0 = time.time()
    for i, s in enumerate(subjects, 1):
        try:
            # Sujeto a sujeto: calienta la caché sin acumular todo en RAM. El loader ya
            # reintenta/omite por dentro; el ndarray se descarta al salir del bucle.
            d = load_dataset(name, [s], classes, tmin, tmax)
            ok.append(s)
            print(f"  [{i}/{len(subjects)}] sujeto {s:>3}: ✓ {d.X.shape[0]} trials", flush=True)
        except Exception as exc:               # noqa: BLE001 - un sujeto roto no tumba el setup
            fail.append(s)
            print(f"  [{i}/{len(subjects)}] sujeto {s:>3}: ✗ {type(exc).__name__}: {exc}", flush=True)

    print(f"\n=== Listo en {time.time()-t0:.0f}s | OK: {len(ok)} | fallidos: {len(fail)} ===")
    if fail:
        print(f"Reintenta los fallidos con: --subjects {' '.join(map(str, fail))}")


if __name__ == "__main__":
    main()
