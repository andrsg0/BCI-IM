"""Simulación de transmisión en vivo (Hito 6).

Aunque tenemos toda la señal grabada, la reproducimos "como si llegara" de un casco
en tiempo real: por trozos (chunks), procesándola con una ventana deslizante y
clasificando sobre la marcha. Esto prepara la futura entrada por LSL del Ultracortex.

LA CLAVE LTI: CAUSALIDAD
------------------------
Offline filtramos con mode='same', que para calcular y[n] usa muestras FUTURAS
(x[n+1], ...). En vivo eso es imposible: solo existe el pasado. Un filtro CAUSAL
calcula y[n] = Σ_{k≥0} h[k]·x[n−k] usando solo muestras ya recibidas. Para ello el
filtro mantiene un ESTADO (las últimas M−1 muestras) entre chunk y chunk.

Precio de la causalidad: el FIR de fase lineal introduce un retardo de grupo de
(M−1)/2 muestras (aquí 50 = 0.2 s a 250 Hz). En vivo ese retardo es real e
inevitable; offline lo compensábamos con mode='same'. Es un punto didáctico clave.
"""
from __future__ import annotations

import time
from collections import Counter

import numpy as np

from bci.dsp.convolution import convolve


class CausalFIR:
    """Filtro FIR causal con estado, para filtrar un stream por chunks.

    Mantiene un buffer con las últimas M−1 muestras de cada canal, de modo que el
    filtrado de un chunk continúa exactamente donde quedó el anterior (sin saltos
    en las fronteras). Filtrar todo de golpe o por trozos da idéntico resultado.
    """

    def __init__(self, h: np.ndarray, n_channels: int):
        self.h = np.asarray(h, dtype=float)
        self.M = len(self.h)
        self.n_channels = n_channels
        self.reset()

    def reset(self) -> None:
        """Reinicia el estado (p. ej. al empezar un nuevo trial)."""
        self.buf = np.zeros((self.n_channels, self.M - 1))

    def process_chunk(self, chunk: np.ndarray) -> np.ndarray:
        """Filtra causalmente un chunk (n_canales, L). Devuelve (n_canales, L)."""
        chunk = np.asarray(chunk, dtype=float)
        n_ch, L = chunk.shape
        ext = np.concatenate([self.buf, chunk], axis=1)         # pasado + nuevo
        out = np.empty((n_ch, L))
        for c in range(n_ch):
            # 'valid' usa solo solapamientos completos -> salida causal de longitud L.
            out[c] = convolve(ext[c], self.h, mode="valid")
        self.buf = ext[:, -(self.M - 1):]                       # guardamos el nuevo pasado
        return out


class StreamSimulator:
    """Reproduce una señal continua y clasifica con ventana deslizante."""

    def __init__(self, pipeline, window_s=2.0, step_s=0.1, realtime_factor=1.0, ref_idx=None):
        self.pipe = pipeline
        self.fs = pipeline.filt.fs
        self.window = int(window_s * self.fs)
        self.step = int(step_s * self.fs)
        self.realtime_factor = realtime_factor
        self.ref_idx = ref_idx   # canal de referencia para enviar señal cruda/filtrada (opcional)

    def stream(self, X_cont: np.ndarray, sleep: bool = False, on_predict=None,
               include_all: bool = False):
        """Emite predicciones a lo largo de una señal continua (n_canales, n_muestras).

        Parameters
        ----------
        X_cont
            Señal continua a reproducir.
        sleep
            Si True, espera en tiempo (≈) real entre chunks (para demos en vivo).
        on_predict
            Callback opcional llamado con cada predicción (para UI/streaming).

        Returns
        -------
        list[dict]
            Por cada ventana: {t, pred, probs}.
        """
        n_ch, N = X_cont.shape
        fir = CausalFIR(self.pipe.filt.h, n_ch)
        filtered = np.zeros((n_ch, 0))
        results = []

        for start in range(0, N - self.step + 1, self.step):
            chunk = X_cont[:, start:start + self.step]
            f = fir.process_chunk(chunk)                         # filtrado causal en vivo
            filtered = np.concatenate([filtered, f], axis=1)
            if filtered.shape[1] > self.window:                 # ventana deslizante
                filtered = filtered[:, -self.window:]

            if filtered.shape[1] == self.window:
                pred, probs, info = self.pipe.classify_window(filtered)
                # Potencia µ/β por canal (log-varianza de la ventana filtrada): sirve para
                # "iluminar" en vivo qué zonas del cuero cabelludo están activas.
                power = np.log(np.var(filtered, axis=1) + 1e-12)
                rec = {"t": (start + self.step) / self.fs, "pred": pred, "probs": probs,
                       "power": power.tolist(),
                       # Etapas CSP/LDA de ESTA ventana (para visualizarlas en vivo).
                       "feat": info["feat"], "disc": info["disc"]}
                # Señal cruda y filtrada del canal de referencia (en µV) para el dashboard.
                if self.ref_idx is not None:
                    rec["raw"] = (chunk[self.ref_idx] * 1e6).tolist()
                    rec["filt"] = (f[self.ref_idx] * 1e6).tolist()
                # Todos los canales crudos del chunk (para el Laboratorio multicanal,
                # que filtra en cliente). Solo bajo petición, para no engordar el frame.
                if include_all:
                    rec["raw_all"] = (chunk * 1e6).tolist()
                results.append(rec)
                if on_predict is not None:
                    on_predict(rec)

            if sleep:
                time.sleep(self.step / self.fs / max(self.realtime_factor, 1e-9))

        return results

    def stream_epoch_vote(self, X_cont: np.ndarray):
        """Streamea un trial y devuelve (voto_mayoritario, lista_de_resultados)."""
        results = self.stream(X_cont)
        preds = [r["pred"] for r in results]
        vote = Counter(preds).most_common(1)[0][0] if preds else None
        return vote, results


class EEGNetStreamSimulator:
    """Versión del simulador para EEGNet (clasificación en vivo de los 4 regímenes).

    Misma idea causal que ``StreamSimulator`` (FIR con estado + ventana deslizante),
    pero clasifica con una red EEGNet en vez de CSP→LDA. Diferencias:
      - El FIR es la banda AMPLIA con la que se entrenó EEGNet (4–40 Hz por defecto),
        no la banda µ/β estricta del CSP. Se pasa ``h`` ya diseñado desde fuera.
      - La ventana debe medir EXACTAMENTE las muestras con que se entrenó la red
        (la arquitectura depende de ``n_samples``); se pasa ``window`` en muestras.
      - EEGNet no tiene etapas CSP/LDA, así que ``feat``/``disc`` van a ``None`` (la
        UI simplemente no dibuja esos paneles para este método).
    """

    def __init__(self, clf, h, fs, window: int, step: int, ref_idx=None):
        self.clf = clf
        self.h = np.asarray(h, dtype=float)
        self.fs = fs
        self.window = int(window)
        self.step = int(step)
        self.ref_idx = ref_idx

    def stream(self, X_cont: np.ndarray, on_predict=None, include_all: bool = False):
        n_ch, N = X_cont.shape
        fir = CausalFIR(self.h, n_ch)
        filtered = np.zeros((n_ch, 0))
        results = []

        for start in range(0, N - self.step + 1, self.step):
            chunk = X_cont[:, start:start + self.step]
            f = fir.process_chunk(chunk)                         # filtrado causal en vivo
            filtered = np.concatenate([filtered, f], axis=1)
            if filtered.shape[1] > self.window:
                filtered = filtered[:, -self.window:]

            if filtered.shape[1] == self.window:
                proba = self.clf.predict_proba(filtered[None, :, :])[0]
                classes = [str(c) for c in self.clf.classes_]
                pred = classes[int(np.argmax(proba))]
                probs = {c: float(p) for c, p in zip(classes, proba)}
                power = np.log(np.var(filtered, axis=1) + 1e-12)
                rec = {"t": (start + self.step) / self.fs, "pred": pred, "probs": probs,
                       "power": power.tolist(), "feat": None, "disc": None}
                if self.ref_idx is not None:
                    rec["raw"] = (chunk[self.ref_idx] * 1e6).tolist()
                    rec["filt"] = (f[self.ref_idx] * 1e6).tolist()
                # Todos los canales crudos del chunk (para el Laboratorio multicanal,
                # que filtra en cliente). Solo bajo petición, para no engordar el frame.
                if include_all:
                    rec["raw_all"] = (chunk * 1e6).tolist()
                results.append(rec)
                if on_predict is not None:
                    on_predict(rec)

        return results
