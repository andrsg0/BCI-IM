"""Sonda de viabilidad de un dataset MOABB para ESTE proyecto (MI izq./der.).

Responde, para un dataset cualquiera del catálogo de MOABB, la pregunta:
**¿sirve para mi proyecto?** Lo comprueba de forma EMPÍRICA (no fiándose de la tabla
de MOABB), porque los metadatos a veces mienten: Lee2019_MI declara 2 sesiones pero
MOABB 1.5.0 solo expone 1 (ver docs/datasets.md). Aquí se descarga 1 sujeto y se
verifica de verdad.

Qué verifica, en orden:
  1. METADATOS (sin descargar): n_sesiones declaradas, clases, fs, ventana del cue.
  2. CLASES izq./der.: que existan left_hand y right_hand (o sea subseteable).
  3. CARGA REAL (descarga 1 sujeto): epochs crudos vía el loader del proyecto.
  4. SESIONES REALES: cuántas sesiones cargan de verdad y si cada una tiene ambas
     clases (la lección de Lee2019_MI).
  5. CANALES MOTORES: que estén C3/Cz/C4 (mínimo para CSP de imaginación motora).
  6. CSP+LDA: k-fold within-subject (cordura) + inter-sesión si hay >=2 sesiones
     (la estimación honesta "otro día").
  7. EEGNet: entrenamiento corto para confirmar que la red entrena (no para medir
     su techo). Se remuestrea a fs baja para que sea rápido.

Uso:
    python scripts/probe_dataset.py --dataset Kumar2024
    python scripts/probe_dataset.py --dataset Yang2025 --subject 1 --no-eegnet
    python scripts/probe_dataset.py --dataset Stieger2021 --epochs 30 --device cpu

NOTA: descarga datos a la caché de MNE (puede ser de cientos de MB a varios GB por
sujeto). Se prueba 1 sujeto por defecto para acotar el coste.
"""
from __future__ import annotations

import argparse
import time

import numpy as np
from sklearn.model_selection import train_test_split

import moabb.datasets as mds

from bci.datasets.moabb_loader import load_dataset
from bci.dsp.resampling import resample_lti
from bci.pipeline.offline import MotorImageryPipeline, evaluate_kfold, evaluate_by_session

LR = ["left_hand", "right_hand"]
MOTOR_MIN = ["C3", "Cz", "C4"]   # canales mínimos para CSP de MI


def make_odd(n: int) -> int:
    return n if n % 2 == 1 else n + 1


def build_cfg(fs: float, interval_len: float, n_channels: int = 22) -> dict:
    """Construye un cfg equivalente al default.yaml pero adaptado a este dataset.

    - num_taps del FIR escala con fs para mantener ~0.4 s de longitud (a 250 Hz da
      101 taps, igual que el default del proyecto).
    - La ventana de clasificación recorta 0.5 s del inicio (transitorio del FIR) y
      toma hasta 2 s (o lo que quepa en el intervalo del cue).
    - n_components del CSP se capa a los canales disponibles (par y ≤ n_canales): así
      funciona en datasets con pocos canales como BCI IV 2b (3 canales → 2 componentes).
    """
    num_taps = make_odd(int(0.4 * fs))
    n_comp = max(2, min(4, n_channels - (n_channels % 2)))
    tmin_rel = 0.5
    tmax_rel = min(interval_len - 0.1, tmin_rel + 2.0)
    if tmax_rel <= tmin_rel:                      # intervalo muy corto: sin recorte
        tmin_rel, tmax_rel = 0.0, interval_len
    return {
        "fir_filter": {"low_hz": 8.0, "high_hz": 30.0, "num_taps": num_taps,
                       "window": "hamming", "fs": fs},
        "csp": {"n_components": n_comp, "log_variance": True, "shrinkage": 0.1},
        "classifier": {"type": "lda", "cv_folds": 5},
        "classification_window": {"tmin_rel": tmin_rel, "tmax_rel": tmax_rel},
    }


def section(title: str) -> None:
    print(f"\n{'─' * 4} {title} {'─' * (60 - len(title))}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", required=True, help="clase MOABB, p. ej. Kumar2024")
    ap.add_argument("--subject", type=int, default=1, help="sujeto a probar (default 1)")
    ap.add_argument("--classes", nargs="+", default=LR, help="clases a conservar")
    ap.add_argument("--epochs", type=int, default=40, help="épocas EEGNet (corto = cordura)")
    ap.add_argument("--eegnet-fs", type=float, default=128.0,
                    help="remuestreo para EEGNet (rapidez). 0 = no remuestrear")
    ap.add_argument("--no-eegnet", action="store_true", help="omitir la prueba EEGNet")
    ap.add_argument("--device", default=None, help="cuda | cpu (default: auto)")
    args = ap.parse_args()

    notas: list[str] = []   # advertencias que entran en el veredicto
    t0 = time.time()

    # === 1. METADATOS (sin descargar) =====================================
    section("1. METADATOS (sin descarga)")
    if not hasattr(mds, args.dataset):
        print(f"✗ '{args.dataset}' no existe en moabb {__import__('moabb').__version__}.")
        return
    ds = getattr(mds, args.dataset)()
    interval = getattr(ds, "interval", None) or [0.0, 4.0]
    interval_len = float(interval[1] - interval[0])
    fs_decl = getattr(ds, "code", None)
    print(f"  dataset           : {args.dataset}")
    print(f"  sujetos           : {len(ds.subject_list)}")
    print(f"  sesiones (declar.): {ds.n_sessions}")
    print(f"  clases (event_id) : {ds.event_id}")
    print(f"  ventana del cue   : {interval} s  (len={interval_len:.1f}s)")

    # === 2. CLASES izq./der. =============================================
    section("2. CLASES izquierda/derecha")
    presentes = [c for c in args.classes if c in ds.event_id]
    if set(LR).issubset(ds.event_id):
        nat = len(ds.event_id) == 2
        print(f"  ✓ left_hand y right_hand presentes ({'nativo 2-clases' if nat else 'subset de multiclase'}).")
        if not nat:
            notas.append("multiclase: se subsetea a izq./der.")
    else:
        print(f"  ✗ NO mapeable a izq./der. (clases: {list(ds.event_id)}). Descartar.")
        return

    # === 3. CARGA REAL (descarga 1 sujeto) ===============================
    section(f"3. CARGA REAL · sujeto {args.subject} (descarga si falta)")
    print("  descargando/cargando epochs crudos…", flush=True)
    data = load_dataset(
        name=args.dataset, subjects=[args.subject], classes=args.classes,
        tmin=float(interval[0]), tmax=float(interval[1]), picks="eeg",
    )
    fs = data.sfreq
    n_trials, n_ch, n_samp = data.X.shape
    print(f"  ✓ X = {data.X.shape}  (trials × canales × muestras)")
    print(f"  fs real           : {fs:.0f} Hz")
    print(f"  canales           : {n_ch}")
    clases, cuenta = np.unique(data.y, return_counts=True)
    print(f"  balance de clases : {dict(zip(clases, cuenta.tolist()))}")

    # === 4. SESIONES REALES (la lección de Lee2019) ======================
    section("4. SESIONES REALES (¿coinciden con lo declarado?)")
    sesiones = sorted(data.metadata["session"].unique().tolist())
    print(f"  sesiones cargadas : {len(sesiones)}  → {sesiones}")
    if len(sesiones) < ds.n_sessions:
        notas.append(f"OJO: declara {ds.n_sessions} sesiones pero MOABB expone {len(sesiones)}")
        print(f"  ⚠ declara {ds.n_sessions} pero solo cargan {len(sesiones)} (como Lee2019_MI).")
    sesiones_ok = []
    for s in sesiones:
        ys = data.y[data.metadata["session"].to_numpy() == s]
        tiene_ambas = set(LR).issubset(set(ys.tolist()))
        print(f"    · sesión {s!r}: {len(ys)} trials, ambas clases={'sí' if tiene_ambas else 'NO'}")
        if tiene_ambas:
            sesiones_ok.append(s)
    multi_sesion = len(sesiones_ok) >= 2

    # === 5. CANALES MOTORES ==============================================
    section("5. CANALES MOTORES")
    up = {c.upper(): c for c in data.ch_names}
    motores = [c for c in MOTOR_MIN if c.upper() in up]
    print(f"  motores presentes : {motores or 'NINGUNO'}  (de {MOTOR_MIN})")
    if not motores:
        notas.append("sin C3/Cz/C4: CSP de MI poco fiable")
        print("  ⚠ sin canales motores centrales: CSP puede rendir mal.")

    # === 6. CSP+LDA ======================================================
    section("6. CSP+LDA (tu pipeline real)")
    cfg = build_cfg(fs, interval_len, n_channels=n_ch)
    print(f"  FIR num_taps={cfg['fir_filter']['num_taps']} · "
          f"ventana={cfg['classification_window']['tmin_rel']:.2f}–"
          f"{cfg['classification_window']['tmax_rel']:.2f}s", flush=True)
    kfold = evaluate_kfold(cfg, data.X, data.y, fs, n_splits=5)
    print(f"  within-subject 5-fold : acc = {kfold.accuracy:.3f}")
    acc_inter = None
    if multi_sesion:
        s_tr, s_te = sesiones_ok[0], sesiones_ok[1]
        inter = evaluate_by_session(cfg, data.X, data.y, data.metadata, fs,
                                    train_session=s_tr, test_session=s_te)
        acc_inter = inter.accuracy
        print(f"  inter-sesión ({s_tr}→{s_te}) : acc = {acc_inter:.3f}  ← estimación 'otro día'")
    else:
        print("  inter-sesión          : N/A (no hay 2 sesiones con ambas clases)")

    # === 7. EEGNet (cordura) =============================================
    acc_eeg = None
    if not args.no_eegnet:
        section("7. EEGNet (entrenamiento corto de cordura)")
        from bci.models.eegnet import EEGNetClassifier   # import perezoso (torch)
        # Recorte a la ventana activa (mismo crop que el CSP) sobre la señal CRUDA.
        i0 = int(cfg["classification_window"]["tmin_rel"] * fs)
        i1 = int(cfg["classification_window"]["tmax_rel"] * fs)
        Xe = data.X[..., i0:i1]
        # Remuestreo a fs baja para rapidez (EEGNet aprende su propio filtro).
        if args.eegnet_fs and fs > args.eegnet_fs:
            Xe = resample_lti(Xe, fs_in=fs, fs_out=args.eegnet_fs).astype(np.float32)
            print(f"  remuestreado {fs:.0f}→{args.eegnet_fs:.0f} Hz para EEGNet: {Xe.shape}")
        Xtr, Xte, ytr, yte = train_test_split(
            Xe, data.y, test_size=0.3, stratify=data.y, random_state=42)
        print(f"  entrenando {args.epochs} épocas…", flush=True)
        clf = EEGNetClassifier(epochs=args.epochs, device=args.device).fit(Xtr, ytr)
        acc_eeg = clf.score(Xte, yte)
        print(f"  EEGNet (split 70/30)  : acc = {acc_eeg:.3f}")

    # === VEREDICTO =======================================================
    section("VEREDICTO")
    usable = bool(motores) and kfold.accuracy >= 0.55
    print(f"  ¿utilizable en el proyecto? : {'SÍ' if usable else 'DUDOSO'}")
    print(f"  ¿>=2 sesiones reales (vivo) : {'SÍ' if multi_sesion else 'NO'}")
    for n in notas:
        print(f"  ⚠ {n}")
    print(f"  (probado en sujeto {args.subject}; tiempo {time.time()-t0:.0f}s)")

    # Fila lista para pegar en docs/datasets.md
    acc_is = f"{acc_inter:.3f}" if acc_inter is not None else "—"
    acc_en = f"{acc_eeg:.3f}" if acc_eeg is not None else "—"
    print("\n  Fila para docs/datasets.md (suj. {}):".format(args.subject))
    print(f"  | {args.dataset} | {len(ds.subject_list)} | {len(sesiones_ok)} | {n_ch} | "
          f"{fs:.0f} | {kfold.accuracy:.3f} | {acc_is} | {acc_en} |")


if __name__ == "__main__":
    main()
