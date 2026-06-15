"""Simulación de transmisión en vivo (Hito 6) — demo end-to-end.

Entrena el pipeline con la sesión '0train' y luego REPRODUCE los trials de la
sesión '1test' como si llegaran de un casco en tiempo real: filtrado FIR causal +
ventana deslizante + clasificación sobre la marcha.

Uso:
    python scripts/run_live_sim.py                  # rápido (sin esperas)
    python scripts/run_live_sim.py --realtime       # a velocidad real (presentación)
    python scripts/run_live_sim.py --trial 5        # muestra la línea temporal de 1 trial
"""
from __future__ import annotations

import argparse

import numpy as np

from bci.config import load_config
from bci.datasets.moabb_loader import load_from_config
from bci.pipeline.offline import MotorImageryPipeline
from bci.streaming.simulator import StreamSimulator


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--realtime", action="store_true", help="Espera en tiempo (≈) real.")
    ap.add_argument("--trial", type=int, default=0, help="Trial de test cuya evolución mostrar.")
    args = ap.parse_args()

    cfg = load_config(args.config)
    s = cfg["streaming"]
    data = load_from_config(cfg)

    sess = data.metadata["session"].to_numpy()
    tr, te = sess == "0train", sess == "1test"
    print(f"=== Simulación en vivo | train={tr.sum()} trials, test={te.sum()} trials ===")
    print(f"Ventana {s['window_s']}s, paso {s['step_s']}s, "
          f"FIR causal {cfg['fir_filter']['num_taps']} taps "
          f"(retardo {(cfg['fir_filter']['num_taps']-1)//2} muestras "
          f"= {(cfg['fir_filter']['num_taps']-1)/2/cfg['fir_filter']['fs']:.2f} s)\n")

    # Entrenar offline con la sesión de entrenamiento.
    pipe = MotorImageryPipeline(cfg, fs=data.sfreq).fit(data.X[tr], data.y[tr])
    sim = StreamSimulator(pipe, window_s=s["window_s"], step_s=s["step_s"],
                          realtime_factor=s.get("realtime_factor", 1.0))

    X_test, y_test = data.X[te], np.asarray(data.y)[te]

    # 1) Línea temporal de un trial concreto (cómo evoluciona la predicción en vivo).
    ti = min(args.trial, len(X_test) - 1)
    print(f"--- Evolución de la predicción en el trial de test #{ti} (real: {y_test[ti]}) ---")
    res = sim.stream(X_test[ti], sleep=args.realtime)
    for r in res[::max(len(res) // 8, 1)]:
        bar = " ".join(f"{k}={v:.2f}" for k, v in r["probs"].items())
        print(f"  t={r['t']:.2f}s  ->  {r['pred']:11s}  [{bar}]")
    print()

    # 2) Precisión de la simulación en vivo (voto mayoritario por trial).
    print("--- Precisión streaming (voto mayoritario por trial) ---")
    votes = []
    for i in range(len(X_test)):
        vote, _ = sim.stream_epoch_vote(X_test[i])
        votes.append(vote)
    votes = np.array(votes)
    acc = float(np.mean(votes == y_test))
    print(f"  accuracy streaming = {acc:.3f}  (sobre {len(y_test)} trials de test)")


if __name__ == "__main__":
    main()
