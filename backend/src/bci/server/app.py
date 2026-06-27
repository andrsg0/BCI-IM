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
from bci.server import payloads as pl
from bci.server import results as results_mod
from bci.streaming.simulator import EEGNetStreamSimulator, StreamSimulator

# Métodos válidos para el selector de 4 regímenes (within/cross × CSP/EEGNet).
WITHIN_METHODS = ('csp_lda', 'eegnet')
CROSS_METHODS = ('csp_lda_cross', 'eegnet_cross')
VALID_METHODS = WITHIN_METHODS + CROSS_METHODS

# Registro de datasets: id -> (etiqueta, config, metadatos conocidos de la Etapa 1).
# Se describen por PROPIEDADES (no por un rol manual confuso): ``sessions`` = nº de
# sesiones reales del dataset. De ahí se DERIVA el uso (ver ``_is_live``): con ≥2
# sesiones (días distintos) hay una estimación honesta inter-sesión, así que el dataset
# sirve para la demo en vivo (calibrar día1 → probar día2). TODOS los datasets aparecen
# en "Resultados" (su benchmark de población); los de ≥2 sesiones aparecen ADEMÁS en la
# "Demo en vivo". Antes esto era un campo 'role' ('live'/'training') que había que poner
# a mano y excluía el dataset live de Resultados — eliminado en jun 2026.
# Lista final (jun 2026): 3 datasets de imaginación motora izq./der. con ≥2 sesiones
# reales (todos aptos para la demo en vivo inter-sesión). accuracy = provisional;
# la media real por dataset la calcula /api/results desde los artefactos entrenados.
REGISTRY = {
    'BNCI2014_001': {'label': 'BCI IV 2a', 'config': 'configs/default.yaml', 'subjects': 9, 'fs': 250, 'accuracy': 0.688, 'sessions': 2},
    'BNCI2014_004': {'label': 'BCI IV 2b', 'config': 'configs/bci2b.yaml', 'subjects': 9, 'fs': 250, 'accuracy': 0.604, 'sessions': 5},
    'Kumar2024': {'label': 'Kumar2024', 'config': 'configs/kumar2024.yaml', 'subjects': 18, 'fs': 512, 'accuracy': 0.644, 'sessions': 6},
}


def _is_live(meta: dict) -> bool:
    """¿El dataset sirve para la demo en vivo? Sí si tiene ≥2 sesiones (estimación
    honesta inter-sesión). Derivado de propiedades, sin campo manual."""
    return int(meta.get('sessions', 1)) >= 2

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
    elif method in WITHIN_METHODS:
        # Respaldo barato (un solo sujeto): entrena y persiste con el split honesto.
        from dataclasses import asdict
        cfg = _config_for(dataset)
        if method == 'eegnet':
            model, card_obj, _ = train_eegnet_subject(cfg, dataset, subject)
        else:
            model, card_obj, _ = train_subject(cfg, dataset, subject)
        save_model(model, card_obj, out_dir)
        card = asdict(card_obj)
    else:
        # Cross-subject: NO se auto-entrena (cargaría TODO el pool de sujetos, caro).
        # Debe pre-entrenarse con scripts/train_all_regimes.py.
        raise HTTPException(
            status_code=409,
            detail=(f"Modelo '{method}' no entrenado para {dataset} s{subject}. "
                    f"Córrelo con: python scripts/train_all_regimes.py "
                    f"--config ../configs/<ds>.yaml --cross-subjects {subject}"),
        )

    _model_cache[key] = (model, card)
    return model, card


def _load_card_if_exists(dataset: str, subject: int, method: str) -> dict | None:
    """Lee SOLO de disco la ficha de un modelo ya entrenado; nunca entrena.

    Para usos donde una ausencia debe degradar con gracia (p. ej. la comparación
    CSP+LDA dentro de /api/eegnet) en vez de disparar un entrenamiento al vuelo."""
    out_dir = resolve_path(_config_for(dataset)['paths']['processed'])
    _, json_path = model_paths(out_dir, dataset, subject, method)
    if not json_path.exists():
        return None
    try:
        return load_card(json_path)
    except Exception:  # noqa: BLE001
        return None


def _get_pipeline(dataset: str, subject: int) -> MotorImageryPipeline:
    return _ensure_model(dataset, subject, 'csp_lda')[0]  # type: ignore[return-value]


def _split_idx(dataset: str, subject: int, method: str = 'csp_lda') -> tuple[np.ndarray, np.ndarray]:
    """(idx_train, idx_demo) según la ficha del modelo entrenado.

    ``idx_demo`` son los trials reservados (held-out) que se transmiten en vivo;
    ``idx_train`` el resto. El split depende del régimen del modelo (su ficha):
      - within (holdout 'session'/'index'): demo = la sesión o fracción reservada.
      - cross (holdout 'subject'): el modelo NUNCA vio a este sujeto, así que TODOS
        sus trials son demo válida (idx_train local vacío).
    """
    data = _get_data(dataset, subject)
    card = _ensure_model(dataset, subject, method)[1]
    n = data.n_trials
    spec = card['holdout']
    if spec['by'] == 'subject':
        idx_demo = np.arange(n)                          # todo el sujeto es held-out
    elif spec['by'] == 'session':
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
    return [{'id': k, 'live': _is_live(v),
             **{kk: vv for kk, vv in v.items() if kk != 'config'}}
            for k, v in REGISTRY.items()]


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
    """Vista general agregada: comparación de métodos sobre toda la población.

    Incluye TODOS los datasets (cada uno es autosuficiente con sus 4 regímenes). Los
    de ≥2 sesiones aparecen además en «Demo en vivo», pero también cuentan aquí como
    benchmark de población.
    """
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
    cached = pl.load_payload(_processed_dir(dataset), dataset, subject, 'any', 'info')
    if cached is not None:
        return cached
    data = _get_data(dataset, subject)
    return pl.build_info_payload(dataset, subject, data)


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


# Posiciones de electrodos (montaje 10-20): fuente única en server/payloads.py.
_positions = pl.electrode_positions


@app.get('/api/positions')
def positions(dataset: str, subject: int = 1):
    """Canales y sus posiciones 2D/3D (montaje 10-20). Ligero: no entrena modelo.

    Lo usa el Cerebro 3D en vivo para situar los electrodos sin cargar el CSP.
    Sirve primero el payload precomputado (portabilidad sin datos crudos)."""
    cached = pl.load_payload(_processed_dir(dataset), dataset, subject, 'any', 'positions')
    if cached is not None:
        return cached
    data = _get_data(dataset, subject)
    return pl.build_positions_payload(data.ch_names)


@app.get('/api/model')
def model(dataset: str, subject: int = 1, method: str = 'csp_lda'):
    """Ficha del modelo ENTRENADO (mundo offline): con qué se entrenó, qué se
    reservó para la demo y la precisión honesta sobre ese held-out."""
    return _ensure_model(dataset, subject, method)[1]


@app.get('/api/train_config')
def train_config(dataset: str, subject: int = 1):
    """Ficha de CONFIGURACIÓN del entrenamiento (mundo offline) para la sección
    Entrenamiento: combina el YAML del dataset (parámetros de preprocesamiento,
    CSP y validación) con la ficha del modelo CSP+LDA ya entrenado (si existe en
    disco — NO entrena al vuelo). Centraliza lo que la página muestra como
    «cómo se preparó la señal / cómo se validó / qué dataset es».
    """
    if dataset not in REGISTRY:
        raise HTTPException(status_code=404, detail=f"dataset '{dataset}' no existe")
    meta = REGISTRY[dataset]
    cfg = _config_for(dataset)
    card = _load_card_if_exists(dataset, subject, 'csp_lda')

    fs = float(card['fs']) if card else float(meta.get('fs') or cfg['fir_filter'].get('fs', 0))
    epo = cfg.get('epoching', {})
    cw = cfg.get('classification_window', {})
    fir = cfg.get('fir_filter', {})
    csp_cfg = cfg.get('csp', {})
    clf = cfg.get('classifier', {})

    # Ventana de clasificación en segundos absolutos del trial (epoch + offset relativo).
    tmin_rel = cw.get('tmin_rel'); tmax_rel = cw.get('tmax_rel')
    win_abs = None
    if tmin_rel is not None and tmax_rel is not None and epo.get('tmin') is not None:
        win_abs = [float(epo['tmin']) + float(tmin_rel), float(epo['tmin']) + float(tmax_rel)]

    num_taps = int(fir.get('num_taps', 0))
    group_delay = (num_taps - 1) / 2 if num_taps else None  # muestras (fase lineal)

    holdout = card.get('holdout') if card else None
    if holdout and holdout.get('by') == 'session':
        holdout_kind = 'inter-sesión'
        holdout_desc = (f"entrena en «{card.get('train_session') or '0train'}», "
                        f"evalúa en «{holdout.get('value')}»")
    elif holdout:
        holdout_kind = 'hold-out 30% estratificado'
        holdout_desc = 'dataset de 1 sola sesión; se reserva un 30% estratificado'
    else:
        holdout_kind = None
        holdout_desc = None

    return {
        'dataset': {
            'id': dataset,
            'label': meta.get('label', dataset),
            'fs': fs,
            'sessions': meta.get('sessions'),
            'live': _is_live(meta),
            'n_subjects': meta.get('subjects'),
            'subject': subject,
            'classes': cfg['dataset'].get('classes') or (card['classes'] if card else []),
            'n_channels': len(card['channels']) if card else None,
            'channels': card['channels'] if card else None,
            'n_trials': (int(card['n_train']) + int(card['n_demo'])) if card else None,
        },
        'preprocessing': {
            'epoching': {'tmin': epo.get('tmin'), 'tmax': epo.get('tmax')},
            'classification_window': {
                'tmin_rel': tmin_rel, 'tmax_rel': tmax_rel,
                'abs_s': win_abs,
                'len_s': (float(tmax_rel) - float(tmin_rel)) if (tmin_rel is not None and tmax_rel is not None) else None,
            },
            'fir': {
                'low_hz': fir.get('low_hz'), 'high_hz': fir.get('high_hz'),
                'num_taps': num_taps or None, 'window': fir.get('window'),
                'group_delay_samples': group_delay,
                'group_delay_ms': (group_delay / fs * 1000.0) if (group_delay and fs) else None,
            },
            'csp': {
                'n_components': csp_cfg.get('n_components'),
                'log_variance': csp_cfg.get('log_variance'),
                'shrinkage': csp_cfg.get('shrinkage'),
            },
        },
        'validation': {
            'classifier': clf.get('type'),
            'cv_folds': clf.get('cv_folds'),
            'holdout_kind': holdout_kind,
            'holdout_desc': holdout_desc,
            'n_train': int(card['n_train']) if card else None,
            'n_demo': int(card['n_demo']) if card else None,
            'accuracy_intersession': float(card['accuracy']) if card else None,
            'kappa': float(card['kappa']) if card else None,
            'trained_on': card.get('trained_on') if card else None,
        },
        'has_model': card is not None,
    }


@app.get('/api/eegnet')
def eegnet_info(dataset: str, subject: int = 1):
    """Filtros que EEGNet APRENDE (mundo offline) — el puente teoría ↔ IA.

    Devuelve, del modelo ya entrenado:
      - temporal: forma de |H(e^jω)| de cada filtro de la conv temporal (≈ banco FIR
        aprendido). Si la red 'redescubrió' la banda µ/β, los picos caen en 8–30 Hz.
      - spatial : pesos por canal de la conv depthwise (≈ filtros espaciales tipo CSP),
        para dibujar topomapas y compararlos con los del CSP.
    """
    cached = pl.load_payload(_processed_dir(dataset), dataset, subject, 'eegnet', 'eegnet')
    if cached is not None:
        return cached
    clf, card = _ensure_model(dataset, subject, 'eegnet')
    csp_card = _load_card_if_exists(dataset, subject, 'csp_lda')
    return pl.build_eegnet_payload(clf, card, csp_card)


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

    cached = pl.load_payload(_processed_dir(dataset), dataset, subject, 'csp_lda', 'csp')
    if cached is None:
        data = _get_data(dataset, subject)
        pipe = _get_pipeline(dataset, subject)
        idx_train, _ = _split_idx(dataset, subject)
        cached = pl.build_csp_payload(data, pipe, idx_train)
    _csp_cache[key] = cached
    return cached


@app.get('/api/csp_signal')
def csp_signal(dataset: str, subject: int = 1, component: int = 0):
    """Señal de UN canal CRUDO vs la salida del componente CSP (Z = W·X) en el tiempo.

    Didáctico (sección Entrenamiento): el electrodo más relevante del componente capta
    una señal ruidosa; el filtro espacial CSP la combina con el resto de canales y
    produce una «señal virtual» mucho más limpia. Se replica el pipeline real
    (FIR → recorte a la ventana activa → CSP) sobre un trial de la clase que el
    componente favorece. Ambas series se normalizan (z-score) para comparar formas,
    ya que sus escalas (µV vs unidades de proyección) son muy distintas.

    El payload precomputado guarda TODOS los componentes; aquí se sirve el pedido."""
    cached = pl.load_payload(_processed_dir(dataset), dataset, subject, 'csp_lda', 'csp_signal')
    if cached is not None:
        by_comp = cached['by_component']
        return by_comp[max(0, min(component, len(by_comp) - 1))]
    data = _get_data(dataset, subject)
    pipe = _get_pipeline(dataset, subject)
    return pl.build_csp_signal_payload(data, pipe, component)


@app.get('/api/lda')
def lda_info(dataset: str, subject: int = 1):
    """Clasificador LINEAL (LDA) que cierra la cadena — sección Entrenamiento.

    Devuelve lo necesario para visualizar la frontera de decisión y su rendimiento:
      - weights/bias : la forma lineal del discriminante binario  y = w·F + b, con
        ``y > 0`` ⇒ ``positive_class``. Es la diferencia de los dos discriminantes
        gaussianos δ₀−δ₁ (los términos cuadráticos se cancelan ⇒ frontera = hiperplano).
      - boundary2d   : un LDA reajustado SOLO sobre los dos componentes que se dibujan
        en el scatter (comp 0 vs comp extremo), para trazar la recta en ESE plano.
      - confusion    : matriz de confusión sobre la partición HELD-OUT (la que el modelo
        nunca vio), junto a accuracy y κ honestos (coinciden con la ficha del modelo).
    """
    cached = pl.load_payload(_processed_dir(dataset), dataset, subject, 'csp_lda', 'lda')
    if cached is not None:
        return cached
    data = _get_data(dataset, subject)
    pipe = _get_pipeline(dataset, subject)
    idx_train, idx_demo = _split_idx(dataset, subject)
    card = _load_card_if_exists(dataset, subject, 'csp_lda')
    return pl.build_lda_payload(data, pipe, idx_train, idx_demo, _config_for(dataset), card)


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
                    channel: str = 'C3', method: str = 'csp_lda', allch: bool = False):
    await websocket.accept()
    loop = asyncio.get_event_loop()
    if method not in VALID_METHODS:
        await websocket.send_json({'error': f"método '{method}' inválido; usa {VALID_METHODS}"})
        await websocket.close()
        return
    try:
        # Cargar datos y el modelo YA ENTRENADO fuera del event loop (operación pesada).
        # Clave de la separación entrenar/transmitir: solo transmitimos los trials
        # HELD-OUT, que el modelo nunca vio al entrenar (sin fuga de datos).
        data = await loop.run_in_executor(None, _get_data, dataset, subject)
        try:
            model, _card = await loop.run_in_executor(None, _ensure_model, dataset, subject, method)
            _, idx_demo = await loop.run_in_executor(None, _split_idx, dataset, subject, method)
        except HTTPException as exc:   # modelo cross no entrenado todavía
            await websocket.send_json({'error': exc.detail})
            await websocket.close()
            return
        cfg = _config_for(dataset)
        s = cfg['streaming']
        ws_len = s['window_s']
        step = s['step_s']
        fs = data.sfreq
        ref_ch = channel if channel in data.ch_names else ('C3' if 'C3' in data.ch_names else data.ch_names[0])
        ref_idx = data.ch_names.index(ref_ch)
        win = cfg.get('classification_window')

        # Selección del simulador según el método (CSP→LDA vs EEGNet).
        if method in ('csp_lda', 'csp_lda_cross'):
            sim = StreamSimulator(model, window_s=ws_len, step_s=step, ref_idx=ref_idx)
            win_s = ws_len
        else:  # eegnet / eegnet_cross: banda amplia + ventana = longitud de entrenamiento
            eeg_cfg = cfg.get('eegnet') or {}
            low = float(eeg_cfg.get('band_low', 4.0))
            high = min(float(eeg_cfg.get('band_high', 40.0)), fs / 2 - 1)
            band = design_bandpass_fir(low, high, fs, 101)
            w0 = int(win['tmin_rel'] * fs) if win else 0
            w1 = int(win['tmax_rel'] * fs) if win else data.X.shape[2]
            sim = EEGNetStreamSimulator(model, band.h, fs, w1 - w0, int(step * fs), ref_idx=ref_idx)
            win_s = (w1 - w0) / fs

        # Ventana de imaginación ACTIVA en el eje 't' (fin de ventana): el centro de la
        # ventana (t - win_s/2) debe caer en [tmin_rel, tmax_rel]. El frontend usa
        # estos límites para que la decisión por trial solo cuente las ventanas útiles.
        alo = (win['tmin_rel'] + win_s / 2) if win else 0.0
        ahi = (win['tmax_rel'] + win_s / 2) if win else 1e9

        # La señal de la demo es FINITA: cada pasada recorre los trials reservados una
        # vez (n_demo trials × trial_s segundos) y luego repite. Mandamos la posición en
        # la tanda (demo_i/demo_n) y la duración de trial para una barra de progreso.
        trial_s = data.X.shape[2] / fs
        n_demo = len(idx_demo)

        # Reproduce SOLO los trials reservados (held-out) en bucle.
        j = 0
        while True:
            idx = int(idx_demo[j])
            results = await loop.run_in_executor(
                None, lambda: sim.stream(data.X[idx], include_all=allch))
            for r in results:
                await websocket.send_json({
                    'trial': idx, 'true': str(data.y[idx]),
                    't': r['t'], 'pred': r['pred'], 'probs': r['probs'],
                    'power': r['power'], 'raw': r.get('raw'), 'filt': r.get('filt'),
                    # Todos los canales crudos del chunk (Laboratorio multicanal): solo si
                    # se pidió ``allch`` (si no, None y los demás consumidores lo ignoran).
                    'raw_all': r.get('raw_all'), 'channels': data.ch_names if allch else None,
                    # Etapas CSP (vector log-varianza) y LDA (proyección discriminante)
                    # de la ventana, para diferenciarlas en la sección En vivo.
                    'feat': r.get('feat'), 'disc': r.get('disc'),
                    'ref_ch': ref_ch, 'alo': alo, 'ahi': ahi,
                    # Progreso de la señal finita (pasada actual sobre los held-out).
                    'demo_i': j, 'demo_n': n_demo, 'trial_s': trial_s,
                })
                await asyncio.sleep(step)
            j = (j + 1) % len(idx_demo)
    except WebSocketDisconnect:
        pass
