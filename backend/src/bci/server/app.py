"""API del backend (Etapa 2): expone el pipeline LTI al frontend React.

REST:
  GET  /api/health                         estado del servidor
  GET  /api/datasets                       metadatos de los datasets soportados
  GET  /api/info?dataset=&subject=         canales, fs, nº de trials, clases
  GET  /api/filter?fs=&low=&high=&taps=    coeficientes h[n] + respuesta en frecuencia
  GET  /api/trial?dataset=&subject=&trial=&channel=   señal x[n] cruda y filtrada y[n]
WebSocket:
  WS   /ws/stream?dataset=&subject=        simulación en vivo (predicción + confianza)

Reusa todo el código de la Etapa 1 (loader, FIR, CSP, pipeline, simulador). Cachea
en memoria los datos y los modelos por (dataset, subject) para no recargar en cada
petición.
"""
from __future__ import annotations

import asyncio
from pathlib import Path

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

import numpy as np

from bci.config import BACKEND_ROOT, load_config, resolve_path
from bci.datasets.moabb_loader import EpochedData, load_from_config
from bci.dsp.convolution import apply_filter
from bci.dsp.fir_filters import design_bandpass_fir
from bci.dsp.frequency_response import frequency_response
from bci.pipeline.offline import MotorImageryPipeline
from bci.pipeline.training import (
    load_card, load_model, model_paths, save_model, train_eegnet_subject, train_subject,
)
from bci.server import results as results_mod
from bci.streaming.simulator import StreamSimulator

# Registro de datasets: id -> (etiqueta, config, metadatos conocidos de la Etapa 1).
# 'role' (ver docs/datasets.md) es la fuente única de la división por uso:
#   'live'     = demo en vivo (≥2 sesiones, calibrar día1→probar día2). Hoy solo 2a.
#   'training' = benchmark / pool cross-subject (muchos sujetos). PhysioNet/Dreyer/Cho/Liu.
# La sección "Resultados" muestra los 'training'; "Demo en vivo" muestra los 'live'.
REGISTRY = {
    'BNCI2014_001': {'label': 'BCI IV 2a', 'config': 'configs/default.yaml', 'subjects': 9, 'fs': 250, 'accuracy': 0.688, 'role': 'live'},
    'PhysionetMI': {'label': 'PhysioNet MMI', 'config': 'configs/physionet.yaml', 'subjects': 109, 'fs': 160, 'accuracy': 0.608, 'role': 'training'},
    'Liu2024': {'label': 'Liu2024', 'config': 'configs/liu2024.yaml', 'subjects': 50, 'fs': 500, 'accuracy': 0.536, 'role': 'training'},
    # Datasets nuevos (ver docs/datasets.md). accuracy = provisional (medida en pocos
    # sujetos); reemplazar por la media real con scripts/evaluate_all.py.
    'Dreyer2023': {'label': 'Dreyer2023', 'config': 'configs/dreyer2023.yaml', 'subjects': 87, 'fs': 512, 'accuracy': 0.696, 'role': 'training'},
    'Cho2017': {'label': 'Cho2017', 'config': 'configs/cho2017.yaml', 'subjects': 52, 'fs': 512, 'accuracy': 0.755, 'role': 'training'},
}

app = FastAPI(title='BCI · Imaginación Motora', version='0.2.0')
app.add_middleware(
    CORSMiddleware, allow_origins=['*'], allow_methods=['*'], allow_headers=['*'],
)

# --- Cachés en memoria -----------------------------------------------------
_data_cache: dict[tuple[str, int], EpochedData] = {}
# Modelos cacheados por (dataset, subject, method). El valor es (modelo, ficha).
_model_cache: dict[tuple[str, int, str], tuple[object, dict]] = {}


def _config_for(dataset: str) -> dict:
    if dataset not in REGISTRY:
        raise KeyError(dataset)
    # Los configs/ viven en la RAÍZ del repo (BACKEND_ROOT.parent), no en backend/;
    # resolvemos a absoluta para que el servidor funcione sea cual sea el directorio
    # desde el que se arranque (las rutas de datos del YAML sí cuelgan de backend/).
    cfg_path = Path(REGISTRY[dataset]['config'])
    if not cfg_path.is_absolute():
        cfg_path = BACKEND_ROOT.parent / cfg_path
    cfg = load_config(cfg_path)
    return cfg


def _get_data(dataset: str, subject: int) -> EpochedData:
    key = (dataset, subject)
    if key not in _data_cache:
        cfg = _config_for(dataset)
        cfg['dataset']['subjects'] = [subject]
        _data_cache[key] = load_from_config(cfg)
    return _data_cache[key]


def _ensure_model(dataset: str, subject: int, method: str = 'csp_lda') -> tuple[object, dict]:
    """Carga el modelo entrenado de disco; si no existe, lo entrena y lo guarda.

    SEPARACIÓN entrenar/transmitir: el modelo se entrena ANTES (scripts/train_model.py)
    con una partición de entrenamiento y se reserva un held-out para la demo. Aquí solo
    se CARGA ese artefacto. El entrenamiento al vuelo es solo un respaldo para que la
    demo nunca falle, y usa el mismo split honesto (sin fuga de datos). ``method`` elige
    el clasificador: 'csp_lda' (clásico) o 'eegnet' (red neuronal).
    """
    key = (dataset, subject, method)
    if key in _model_cache:
        return _model_cache[key]

    out_dir = resolve_path(_config_for(dataset)['paths']['processed'])
    pkl_path, json_path = model_paths(out_dir, dataset, subject, method)
    if pkl_path.exists() and json_path.exists():
        model, card = load_model(pkl_path), load_card(json_path)
    else:  # respaldo: entrenar y persistir con el split honesto
        from dataclasses import asdict
        cfg = _config_for(dataset)
        if method == 'eegnet':
            model, card_obj, _ = train_eegnet_subject(cfg, dataset, subject)
        else:
            model, card_obj, _ = train_subject(cfg, dataset, subject)
        save_model(model, card_obj, out_dir)
        card = asdict(card_obj)

    _model_cache[key] = (model, card)
    return model, card


def _get_pipeline(dataset: str, subject: int) -> MotorImageryPipeline:
    return _ensure_model(dataset, subject, 'csp_lda')[0]  # type: ignore[return-value]


def _split_idx(dataset: str, subject: int) -> tuple[np.ndarray, np.ndarray]:
    """(idx_train, idx_demo) según la ficha del modelo entrenado.

    ``idx_demo`` son los trials reservados (held-out) que se transmiten en vivo;
    ``idx_train`` el resto, lo que el modelo realmente usó para aprender. El split es
    el mismo para ambos métodos, así que basta consultar la ficha de CSP+LDA.
    """
    data = _get_data(dataset, subject)
    card = _ensure_model(dataset, subject, 'csp_lda')[1]
    n = data.n_trials
    spec = card['holdout']
    if spec['by'] == 'session':
        sess = data.metadata['session'].to_numpy()
        idx_demo = np.where(sess == spec['value'])[0]
    else:
        idx_demo = np.array(spec['indices'], dtype=int)
    idx_train = np.array([i for i in range(n) if i not in set(idx_demo.tolist())], dtype=int)
    return idx_train, idx_demo


# --- REST ------------------------------------------------------------------
@app.get('/api/health')
def health():
    return {'status': 'ok', 'datasets': list(REGISTRY)}


_glossary_cache: list | None = None


@app.get('/api/glossary')
def glossary():
    """Parsea docs/glosario.md en una lista de {category, term, body(markdown)}.

    Fuente única: el mismo glosario del proyecto. Categorías = '## N. Título',
    términos = '### Título', y el cuerpo es todo lo que va hasta el siguiente
    encabezado.
    """
    global _glossary_cache
    if _glossary_cache is not None:
        return _glossary_cache

    path = BACKEND_ROOT.parent / 'docs' / 'glosario.md'
    entries: list[dict] = []
    category = 'General'
    term: str | None = None
    body: list[str] = []

    def flush():
        if term is not None:
            entries.append({'category': category, 'term': term, 'body': '\n'.join(body).strip()})

    for line in path.read_text(encoding='utf-8').splitlines():
        if line.startswith('### '):
            flush(); term = line[4:].strip(); body = []
        elif line.startswith('## '):
            flush(); term = None; body = []
            category = line[3:].strip().lstrip('0123456789. ').strip()
        elif line.startswith('# ') or line.strip() == '---':
            continue
        elif term is not None:
            body.append(line)
    flush()
    _glossary_cache = entries
    return entries


@app.get('/api/datasets')
def datasets():
    return [{'id': k, **{kk: vv for kk, vv in v.items() if kk != 'config'}} for k, v in REGISTRY.items()]


# --- Resultados (sección Resultados) ---------------------------------------
# Ensamblan los CSV/fichas que ya hay en disco (ver server/results.py). Caché de
# proceso: se invalida reiniciando el servidor (igual que el resto de cachés).
_results_cache: dict[str, dict] = {}


def _processed_dir(dataset: str) -> Path:
    return resolve_path(_config_for(dataset)['paths']['processed'])


def _classes_of(dataset: str) -> list[str] | None:
    return _config_for(dataset)['dataset'].get('classes')


@app.get('/api/results')
def results_index():
    """Índice ligero: una tarjeta-resumen por dataset (sin la tabla por sujeto)."""
    out = []
    for did, meta in REGISTRY.items():
        try:
            out.append(results_mod.dataset_summary(
                did, meta, _processed_dir(did), _classes_of(did)))
        except Exception as exc:  # noqa: BLE001 - un dataset roto no tumba el índice
            out.append({'id': did, 'label': meta.get('label', did),
                        'status': 'pending', 'error': str(exc)})
    return out


@app.get('/api/results_aggregate')
def results_aggregate():
    """Vista general agregada: comparación de métodos sobre toda la población."""
    datasets = [(did, meta, _processed_dir(did), _classes_of(did))
                for did, meta in REGISTRY.items()]
    return results_mod.aggregate_methods(datasets)


@app.get('/api/results/{dataset}')
def results_detail(dataset: str):
    if dataset not in REGISTRY:
        raise HTTPException(status_code=404, detail=f"dataset '{dataset}' no existe")
    if dataset not in _results_cache:
        _results_cache[dataset] = results_mod.dataset_results(
            dataset, REGISTRY[dataset], _processed_dir(dataset), _classes_of(dataset))
    return _results_cache[dataset]


@app.get('/api/info')
def info(dataset: str, subject: int = 1):
    data = _get_data(dataset, subject)
    return {
        'dataset': dataset, 'subject': subject, 'fs': data.sfreq,
        'channels': data.ch_names, 'n_trials': data.n_trials,
        'classes': sorted(set(map(str, data.y))),
        'class_distribution': data.class_distribution(),
    }


@app.get('/api/filter')
def filter_response(fs: float = 250, low: float = 8, high: float = 30,
                    taps: int = 101, window: str = 'hamming', n_freqs: int = 256):
    """Diseña el FIR y devuelve h[n] + su respuesta en frecuencia. No carga datos."""
    filt = design_bandpass_fir(low, high, fs, taps, window)
    fr = frequency_response(filt.h, fs, n_freqs=n_freqs)
    return {
        'fs': fs, 'low': low, 'high': high, 'taps': taps, 'window': window,
        'group_delay': filt.group_delay,
        'h': filt.h.tolist(),
        'freqs': fr.freqs_hz.tolist(),
        'magnitude': fr.magnitude.tolist(),
        'magnitude_db': fr.magnitude_db.tolist(),
    }


@app.get('/api/trial')
def trial(dataset: str, subject: int = 1, trial: int = 0, channel: str = 'C3'):
    """Señal cruda x[n] y filtrada y[n] de un canal de un trial."""
    data = _get_data(dataset, subject)
    if channel not in data.ch_names:
        channel = data.ch_names[0]
    ch = data.ch_names.index(channel)
    trial = max(0, min(trial, data.n_trials - 1))

    x = data.X[trial, ch] * 1e6  # a µV (consistente con /api/continuous)
    filt = design_bandpass_fir(8, 30, data.sfreq, 101, 'hamming')
    y = apply_filter(x, filt.h, mode='same')
    return {
        'fs': data.sfreq, 'channel': channel, 'trial': trial,
        'n_trials': data.n_trials, 'channels': data.ch_names,
        'label': str(data.y[trial]),
        'x': x.tolist(), 'y': y.tolist(),
    }


def _positions(ch_names):
    """Posiciones de los electrodos (sistema 10-20): 2D para el topomapa y 3D
    para el cerebro. Usa el montaje estándar de MNE."""
    import mne

    info = mne.create_info(list(ch_names), 250.0, 'eeg')
    info.set_montage('standard_1005', match_case=False, on_missing='ignore')
    ch_pos = info.get_montage().get_positions()['ch_pos']
    coords = [ch_pos.get(ch) for ch in ch_names]
    scale = max((abs(float(c[i])) for c in coords if c is not None for i in (0, 1)), default=1.0) or 1.0
    pos2d, pos3d = {}, {}
    for ch, c in zip(ch_names, coords):
        if c is None:
            pos2d[ch] = None; pos3d[ch] = None
        else:
            pos3d[ch] = [float(c[0]), float(c[1]), float(c[2])]
            pos2d[ch] = [float(c[0] / scale * 0.85), float(c[1] / scale * 0.85)]  # x=derecha, y=anterior
    return pos2d, pos3d


@app.get('/api/positions')
def positions(dataset: str, subject: int = 1):
    """Canales y sus posiciones 2D/3D (montaje 10-20). Ligero: no entrena modelo.

    Lo usa el Cerebro 3D en vivo para situar los electrodos sin cargar el CSP."""
    data = _get_data(dataset, subject)
    pos2d, pos3d = _positions(data.ch_names)
    return {'channels': data.ch_names, 'pos2d': pos2d, 'pos3d': pos3d}


@app.get('/api/model')
def model(dataset: str, subject: int = 1, method: str = 'csp_lda'):
    """Ficha del modelo ENTRENADO (mundo offline): con qué se entrenó, qué se
    reservó para la demo y la precisión honesta sobre ese held-out."""
    return _ensure_model(dataset, subject, method)[1]


@app.get('/api/eegnet')
def eegnet_info(dataset: str, subject: int = 1):
    """Filtros que EEGNet APRENDE (mundo offline) — el puente teoría ↔ IA.

    Devuelve, del modelo ya entrenado:
      - temporal: forma de |H(e^jω)| de cada filtro de la conv temporal (≈ banco FIR
        aprendido). Si la red 'redescubrió' la banda µ/β, los picos caen en 8–30 Hz.
      - spatial : pesos por canal de la conv depthwise (≈ filtros espaciales tipo CSP),
        para dibujar topomapas y compararlos con los del CSP.
    """
    clf, card = _ensure_model(dataset, subject, 'eegnet')
    mdl = clf.model  # type: ignore[attr-defined]
    fs = float(card['fs'])

    # Conv temporal: pesos (F1, 1, 1, kern) -> respuesta en frecuencia de cada filtro.
    Wt = mdl.temporal[0].weight.detach().cpu().numpy()[:, 0, 0, :]   # (F1, kern)
    freqs = frequency_response(Wt[0], fs, n_freqs=256).freqs_hz.tolist()
    temporal = []
    for h in Wt:
        mag = frequency_response(h, fs, n_freqs=256).magnitude
        temporal.append((mag / (float(mag.max()) + 1e-12)).tolist())  # normalizado a pico 1

    # Conv depthwise espacial: pesos (F1*D, 1, n_canales, 1) -> un vector por filtro.
    Ws = mdl.spatial[0].weight.detach().cpu().numpy()[:, 0, :, 0]    # (F1*D, n_canales)
    pos2d, pos3d = _positions(card['channels'])
    extra = card.get('extra') or {}
    return {
        'channels': card['channels'],
        'fs': fs,
        'freqs': freqs,
        'temporal': temporal,          # (F1, n_freqs) formas de |H|
        'spatial': Ws.tolist(),        # (F1*D, n_canales) pesos espaciales
        'classes': card['classes'],
        'accuracy_intersession': extra.get('accuracy_intersession', card['accuracy']),
        'accuracy_kfold': extra.get('accuracy_kfold'),
        'folds': extra.get('folds'),
        'kern_length': int(Wt.shape[1]),
        'pos2d': pos2d, 'pos3d': pos3d,
    }


_csp_cache: dict[tuple[str, int], dict] = {}


@app.get('/api/csp')
def csp(dataset: str, subject: int = 1):
    """Filtros/patrones espaciales CSP + separación log-varianza + posiciones.

    Pertenece al mundo OFFLINE: muestra el modelo ya entrenado. Los rasgos
    log-varianza son los de la partición de ENTRENAMIENTO (no se incluye el
    held-out reservado para la demo, para no mezclar los dos mundos).

    Es un artefacto FIJO del entrenamiento: el modelo, el split y los rasgos son
    deterministas, así que cacheamos la respuesta por (dataset, sujeto). La sección
    En vivo la reutiliza como fondo (nube de entrenamiento) sobre el que dibuja, en
    tiempo real, el punto de cada ventana en el espacio CSP y en la recta del LDA."""
    key = (dataset, subject)
    if key in _csp_cache:
        return _csp_cache[key]

    data = _get_data(dataset, subject)
    pipe = _get_pipeline(dataset, subject)
    idx_train, _ = _split_idx(dataset, subject)
    feats = pipe._features(data.X[idx_train])     # (n_train, n_componentes)
    pos2d, pos3d = _positions(data.ch_names)
    # Proyección de cada trial de entrenamiento sobre la recta discriminante del LDA
    # (δ_1 − δ_0). Da el HISTOGRAMA de fondo del eje de decisión que la sección En
    # vivo usa para situar la ventana actual respecto a la frontera (en 0).
    scores = pipe.lda.decision_function(feats)    # (n_train, n_clases)
    lda_disc = (scores[:, 1] - scores[:, 0]).tolist() if scores.shape[1] == 2 else \
        (scores.max(axis=1) - np.sort(scores, axis=1)[:, -2]).tolist()
    resp = {
        'channels': data.ch_names,
        'eigenvalues': pipe.csp.eigenvalues_.tolist(),
        'patterns': pipe.csp.patterns_.T.tolist(),   # (n_componentes, n_canales)
        'classes': sorted(set(map(str, data.y))),
        'features': feats.tolist(),
        'labels': [str(c) for c in np.asarray(data.y)[idx_train]],
        'lda_disc': lda_disc,                        # proyección discriminante (n_train,)
        'pos2d': pos2d, 'pos3d': pos3d,
    }
    _csp_cache[key] = resp
    return resp


_raw_cache: dict[tuple[str, int], object] = {}


def _get_raw(dataset: str, subject: int):
    """Carga (y cachea) la señal continua cruda del primer run, solo canales EEG."""
    import moabb.datasets as mds
    key = (dataset, subject)
    if key not in _raw_cache:
        ds = getattr(mds, dataset)()
        data = ds.get_data(subjects=[subject])
        sess = list(data[subject])[0]
        run = list(data[subject][sess])[0]
        _raw_cache[key] = data[subject][sess][run].copy().pick('eeg')
    return _raw_cache[key]


@app.get('/api/continuous')
def continuous(dataset: str, subject: int = 1, channel: str = 'C3', seconds: float = 60):
    """Señal CONTINUA cruda de UN canal (para ver el filtrado en el tiempo)."""
    raw = _get_raw(dataset, subject)
    fs = float(raw.info['sfreq'])
    if channel not in raw.ch_names:
        channel = raw.ch_names[0]
    ci = raw.ch_names.index(channel)
    n = int(seconds * fs)
    x = raw.get_data(picks=[ci])[0][:n] * 1e6  # a µV
    return {'fs': fs, 'channel': channel, 'channels': raw.ch_names,
            'seconds': len(x) / fs, 'x': x.tolist()}


@app.get('/api/epoch')
def epoch(dataset: str, subject: int = 1, trial: int = 0):
    """Un trial con TODOS los canales (n_canales × n_muestras), en µV."""
    data = _get_data(dataset, subject)
    trial = max(0, min(trial, data.n_trials - 1))
    X = (data.X[trial] * 1e6)
    return {'fs': data.sfreq, 'channels': data.ch_names, 'n_trials': data.n_trials,
            'label': str(data.y[trial]), 'X': X.tolist()}


@app.get('/api/continuous_all')
def continuous_all(dataset: str, subject: int = 1, seconds: float = 30):
    """Señal continua con TODOS los canales (para el modo en vivo multicanal), en µV."""
    raw = _get_raw(dataset, subject)
    fs = float(raw.info['sfreq'])
    n = int(seconds * fs)
    X = raw.get_data()[:, :n] * 1e6
    return {'fs': fs, 'channels': raw.ch_names, 'seconds': X.shape[1] / fs, 'X': X.tolist()}


# --- WebSocket: simulación en vivo -----------------------------------------
@app.websocket('/ws/stream')
async def ws_stream(websocket: WebSocket, dataset: str = 'BNCI2014_001', subject: int = 1,
                    channel: str = 'C3'):
    await websocket.accept()
    loop = asyncio.get_event_loop()
    try:
        # Cargar datos y el modelo YA ENTRENADO fuera del event loop (operación pesada).
        # Clave de la separación entrenar/transmitir: solo transmitimos los trials
        # HELD-OUT, que el modelo nunca vio al entrenar (sin fuga de datos).
        data = await loop.run_in_executor(None, _get_data, dataset, subject)
        pipe = await loop.run_in_executor(None, _get_pipeline, dataset, subject)
        _, idx_demo = await loop.run_in_executor(None, _split_idx, dataset, subject)
        cfg = _config_for(dataset)
        s = cfg['streaming']
        ws_len = s['window_s']
        ref_ch = channel if channel in data.ch_names else ('C3' if 'C3' in data.ch_names else data.ch_names[0])
        ref_idx = data.ch_names.index(ref_ch)
        sim = StreamSimulator(pipe, window_s=ws_len, step_s=s['step_s'], ref_idx=ref_idx)
        step = s['step_s']

        # Ventana de imaginación ACTIVA en el eje 't' (fin de ventana): el centro de la
        # ventana (t - window_s/2) debe caer en [tmin_rel, tmax_rel]. El frontend usa
        # estos límites para que la decisión por trial solo cuente las ventanas útiles.
        win = cfg.get('classification_window')
        alo = (win['tmin_rel'] + ws_len / 2) if win else 0.0
        ahi = (win['tmax_rel'] + ws_len / 2) if win else 1e9

        # Reproduce SOLO los trials reservados (held-out) en bucle.
        j = 0
        while True:
            idx = int(idx_demo[j])
            results = await loop.run_in_executor(None, sim.stream, data.X[idx])
            for r in results:
                await websocket.send_json({
                    'trial': idx, 'true': str(data.y[idx]),
                    't': r['t'], 'pred': r['pred'], 'probs': r['probs'],
                    'power': r['power'], 'raw': r.get('raw'), 'filt': r.get('filt'),
                    # Etapas CSP (vector log-varianza) y LDA (proyección discriminante)
                    # de la ventana, para diferenciarlas en la sección En vivo.
                    'feat': r.get('feat'), 'disc': r.get('disc'),
                    'ref_ch': ref_ch, 'alo': alo, 'ahi': ahi,
                })
                await asyncio.sleep(step)
            j = (j + 1) % len(idx_demo)
    except WebSocketDisconnect:
        pass
