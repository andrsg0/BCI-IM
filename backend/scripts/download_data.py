"""Descarga y verifica un dataset de imaginación motora (Hito 2).

Uso:
    python scripts/download_data.py                 # usa configs/default.yaml
    python scripts/download_data.py --subjects 1 2  # sobreescribe sujetos
    python scripts/download_data.py --no-validate   # omite la validación con MOABB
    python scripts/download_data.py --save          # cachea a data/processed/*.npz

Qué hace:
  1. Carga la señal CRUDA y la epoca con nuestro loader (bci.datasets.moabb_loader).
  2. (Opcional) Valida que nuestro epoching coincide con el Paradigm de MOABB.
  3. Imprime un resumen y, opcionalmente, guarda X/y en data/processed.
"""
from __future__ import annotations

import argparse

import numpy as np

from bci.config import load_config, resolve_path
from bci.datasets.moabb_loader import load_from_config


def validate_against_moabb(cfg: dict) -> None:
    """Comprueba que nuestro epoching manual coincide con el Paradigm de MOABB.

    Es una verificación de cordura: MOABB ya sabe la ventana correcta del dataset,
    así que si nuestras formas y conteos por clase coinciden, nuestro loader es fiel.
    """
    from moabb.paradigms import MotorImagery

    import moabb.datasets as mds

    ds = getattr(mds, cfg["dataset"]["name"])()
    classes = cfg["dataset"].get("classes")

    # MotorImagery con banda muy amplia ~ sin filtrar; misma ventana que el dataset.
    paradigm = MotorImagery(
        events=classes,
        n_classes=len(classes) if classes else None,
        fmin=1,
        fmax=40,
        resample=None,
    )
    Xm, ym, _ = paradigm.get_data(ds, subjects=cfg["dataset"]["subjects"])

    print("\n--- Validación contra el Paradigm de MOABB ---")
    print(f"  MOABB  -> X.shape = {Xm.shape}")
    return Xm.shape


def report_validation(ours: tuple, moabb_shape: tuple) -> None:
    """Compara nuestra forma con la de MOABB y explica diferencias esperadas.

    La verificación REAL de que tmin/tmax/picks son correctos son los ejes de
    canales y muestras. El nº de trials puede ser menor que en MOABB porque
    NO concatenamos runs (decisión de diseño): descartamos los epochs truncados
    en los bordes de cada run, que no tienen la ventana temporal completa.
    """
    chans_ok = ours[1] == moabb_shape[1]
    times_ok = ours[2] == moabb_shape[2]
    dropped = moabb_shape[0] - ours[0]
    print(f"  nuestro -> X.shape = {ours}")
    print(f"  canales coinciden : {'✓' if chans_ok else '✗'} ({ours[1]} vs {moabb_shape[1]})")
    print(f"  muestras coinciden: {'✓' if times_ok else '✗'} ({ours[2]} vs {moabb_shape[2]})")
    if chans_ok and times_ok:
        if dropped == 0:
            print("  RESULTADO: ✓ COINCIDE EXACTAMENTE con MOABB")
        else:
            print(f"  RESULTADO: ✓ CORRECTO. {dropped} epochs menos: truncados en bordes")
            print("             de run (TOO_SHORT), descartados a propósito (no concatenamos runs).")
    else:
        print("  RESULTADO: ✗ DIFIERE en canales/muestras -> revisar tmin/tmax/picks")


def main() -> None:
    parser = argparse.ArgumentParser(description="Descarga/verifica dataset BCI.")
    parser.add_argument("--config", default=None, help="Ruta a un YAML alternativo.")
    parser.add_argument("--subjects", type=int, nargs="+", default=None,
                        help="Sobreescribe la lista de sujetos del config.")
    parser.add_argument("--no-validate", action="store_true",
                        help="Omite la validación cruzada con el Paradigm de MOABB.")
    parser.add_argument("--save", action="store_true",
                        help="Guarda X/y/metadata en data/processed/<dataset>_sN.npz.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.subjects is not None:
        cfg["dataset"]["subjects"] = args.subjects

    ds_name = cfg["dataset"]["name"]
    subjects = cfg["dataset"]["subjects"]
    print(f"=== Cargando {ds_name} | sujetos={subjects} | clases={cfg['dataset'].get('classes')} ===")
    print("(la primera vez descarga ~varias decenas de MB a la caché de MNE)\n")

    data = load_from_config(cfg)
    print(data.summary())
    print(f"  canales ({data.n_channels}): {data.ch_names}")

    if not args.no_validate:
        moabb_shape = validate_against_moabb(cfg)
        report_validation(data.X.shape, moabb_shape)

    if args.save:
        out_dir = resolve_path(cfg["paths"]["processed"])
        out_dir.mkdir(parents=True, exist_ok=True)
        tag = f"{ds_name}_s{'-'.join(map(str, subjects))}"
        out_path = out_dir / f"{tag}.npz"
        np.savez_compressed(
            out_path,
            X=data.X,
            y=data.y,
            ch_names=np.array(data.ch_names),
            sfreq=data.sfreq,
            metadata=data.metadata.to_records(index=False),
        )
        print(f"\nGuardado: {out_path}  ({out_path.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
