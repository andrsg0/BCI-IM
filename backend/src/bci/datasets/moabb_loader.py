"""Cargador de datasets de imaginación motora vía MOABB.

DECISIÓN DE DISEÑO CLAVE
------------------------
Usamos MOABB solo en su capa ``Dataset`` (señal cruda + eventos) y hacemos el
*epoching* nosotros con MNE. **No** aplicamos aquí ningún filtro frecuencial:
la señal sale CRUDA, porque el filtrado pasa-banda µ/β lo implementaremos a mano
como una convolución FIR (ver ``bci.dsp.fir_filters``). Así la teoría LTI queda
explícita y no oculta dentro del ``Paradigm`` de MOABB.

Estructura que devuelve MOABB:  data[sujeto][sesión][run] -> mne.io.Raw
Nosotros la aplanamos a la tripleta estándar de ML: (X, y, metadata).
"""
from __future__ import annotations

import gc
from dataclasses import dataclass, field

import mne
import numpy as np
import pandas as pd

# MOABB y MNE son muy verbosos; silenciamos su log salvo errores.
mne.set_log_level("ERROR")


@dataclass
class EpochedData:
    """Resultado del epoching: datos listos para el pipeline LTI.

    Attributes
    ----------
    X : np.ndarray
        Señal CRUDA (sin filtrar) con forma ``(n_trials, n_canales, n_muestras)``.
    y : np.ndarray
        Etiquetas de clase (strings), forma ``(n_trials,)``.
    metadata : pd.DataFrame
        Columnas ``subject``, ``session``, ``run`` por trial. Imprescindible para
        una validación cruzada honesta (separar por sesión).
    ch_names : list[str]
        Nombres de los canales EEG, en el mismo orden que el eje 1 de ``X``.
    sfreq : float
        Frecuencia de muestreo en Hz.
    """

    X: np.ndarray
    y: np.ndarray
    metadata: pd.DataFrame
    ch_names: list[str]
    sfreq: float

    @property
    def n_trials(self) -> int:
        return self.X.shape[0]

    @property
    def n_channels(self) -> int:
        return self.X.shape[1]

    @property
    def n_times(self) -> int:
        return self.X.shape[2]

    def class_distribution(self) -> dict[str, int]:
        labels, counts = np.unique(self.y, return_counts=True)
        return {str(k): int(v) for k, v in zip(labels, counts)}

    def summary(self) -> str:
        return (
            f"EpochedData: {self.n_trials} trials x {self.n_channels} canales "
            f"x {self.n_times} muestras @ {self.sfreq:g} Hz\n"
            f"  clases: {self.class_distribution()}\n"
            f"  sesiones: {sorted(self.metadata['session'].unique().tolist())}"
        )


def _get_dataset(name: str):
    """Instancia un dataset de MOABB por su nombre de clase (p. ej. 'BNCI2014_001').

    Si el constructor admite ``accept`` (licencia que hay que aceptar para descargar,
    p. ej. Shin2017A/Ofner2017), se pasa ``accept=True`` automáticamente: en un proyecto
    académico que ya eligió usar esos datasets, aceptar es el comportamiento esperado.
    """
    import inspect

    import moabb.datasets as mds

    try:
        dataset_cls = getattr(mds, name)
    except AttributeError as exc:
        raise ValueError(
            f"Dataset '{name}' no existe en moabb.datasets. "
            f"Ejemplos válidos: BNCI2014_001, BNCI2014_004, Kumar2024."
        ) from exc
    params = inspect.signature(dataset_cls.__init__).parameters
    accepts = "accept" in params or any(
        p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values())
    return dataset_cls(accept=True) if accepts else dataset_cls()


def _download_with_retries(
    dataset,
    subjects: list[int],
    *,
    retries: int = 4,
    timeout: int = 180,
    verbose: bool = True,
) -> list[int]:
    """Descarga la caché de MNE **sujeto a sujeto** con reintentos y backoff.

    MOABB baja todos los sujetos de golpe dentro de ``get_data``: si el servidor
    (OSF en el caso de Dreyer2023) da un *read timeout*, revienta toda la corrida
    y se pierde el progreso de los sujetos restantes. Aquí descargamos uno a uno
    —lo ya cacheado se salta sin red— y reintentamos cada sujeto con backoff.

    Además subimos el timeout de lectura de ``pooch`` (su default de 30 s es corto
    para los .zip grandes de OSF, que a ratos se quedan colgados a mitad de
    descarga). Es solo robustez de red; no toca la teoría LTI.

    Devuelve la lista de sujetos que quedaron disponibles en caché (puede ser un
    subconjunto si alguno es irrecuperable tras agotar los reintentos).
    """
    import time

    try:  # subir el timeout de pooch de forma best-effort
        import pooch.downloaders as _pd

        _pd.DEFAULT_TIMEOUT = max(getattr(_pd, "DEFAULT_TIMEOUT", 30) or 30, timeout)
    except Exception:  # noqa: BLE001 - si pooch cambia, seguimos con el default
        pass

    disponibles: list[int] = []
    for s in subjects:
        for intento in range(1, retries + 1):
            try:
                dataset.get_data(subjects=[s])  # cachea; si ya está, no baja nada
                disponibles.append(s)
                break
            except Exception as exc:  # noqa: BLE001 - red flaky: reintentar
                if verbose:
                    print(
                        f"[descarga] sujeto {s}: intento {intento}/{retries} falló "
                        f"({type(exc).__name__}: {exc}).",
                        flush=True,
                    )
                if intento == retries:
                    if verbose:
                        print(
                            f"[descarga] sujeto {s}: OMITIDO tras {retries} intentos.",
                            flush=True,
                        )
                else:
                    time.sleep(min(5 * intento, 30))  # backoff lineal, tope 30 s
    return disponibles


def load_dataset(
    name: str,
    subjects: list[int],
    classes: list[str] | None,
    tmin: float,
    tmax: float,
    picks: str = "eeg",
    baseline: tuple | None = None,
) -> EpochedData:
    """Descarga (si hace falta) y carga un dataset como epochs CRUDOS.

    Parameters
    ----------
    name
        Nombre de la clase de dataset en MOABB (ej. ``"BNCI2014_001"``).
    subjects
        Lista de sujetos a cargar (ej. ``[1]``).
    classes
        Clases a conservar (ej. ``["left_hand", "right_hand"]``). ``None`` = todas.
    tmin, tmax
        Ventana del epoch en segundos, **relativa al onset de la anotación**
        (inicio del trial). Para el 2a, la imaginación va de 2 a 6 s.
    picks
        Selección de canales para MNE (``"eeg"`` descarta EOG y STIM).
    baseline
        Corrección de línea base de MNE. ``None`` = sin corrección (señal cruda).
    """
    dataset = _get_dataset(name)
    # Descarga robusta sujeto a sujeto (reintentos + timeout largo) antes del
    # get_data masivo: evita que un read-timeout de OSF tire toda la corrida.
    disponibles = _download_with_retries(dataset, subjects)
    if disponibles != subjects:
        faltan = [s for s in subjects if s not in disponibles]
        print(
            f"[descarga] AVISO: sujetos no descargables omitidos: {faltan}. "
            f"Se continúa con {len(disponibles)} sujetos.",
            flush=True,
        )
        subjects = disponibles

    X_list: list[np.ndarray] = []
    y_list: list[str] = []
    meta_rows: list[dict] = []
    ch_names: list[str] | None = None
    sfreq: float | None = None

    # Cargamos y epochamos SUJETO A SUJETO en lugar de traer los Raw de todos los
    # sujetos a la vez. Las señales continuas de MOABB pesan mucho más que los
    # epochs recortados; con muchos sujetos (Dreyer2023 = 87 a 512 Hz) tenerlas
    # todas en RAM agota la memoria (OOM → el sistema se congela y matan el proceso).
    # Aquí solo acumulamos los arrays X compactos: el Raw de cada sujeto se libera
    # (`del data` + gc) antes de pasar al siguiente, acotando el pico de memoria.
    for subject in subjects:
        data = dataset.get_data(subjects=[subject])  # ya en caché: no baja red
        for session, runs in data[subject].items():
            for run, raw in runs.items():
                # 1) Quedarnos solo con los canales pedidos (EEG: 22 canales).
                raw = raw.copy().pick(picks)

                # 2) Eventos a partir de las anotaciones (left_hand, right_hand, ...).
                events, event_id = mne.events_from_annotations(raw)

                # 3) Filtrar a las clases deseadas, si se especificaron.
                if classes is not None:
                    event_id = {k: v for k, v in event_id.items() if k in classes}
                    if not event_id:
                        continue  # este run no tiene las clases pedidas

                # 4) Epoching SIN filtrado frecuencial ni baseline (señal cruda).
                #    preload=True materializa los datos en memoria como ndarray.
                epochs = mne.Epochs(
                    raw,
                    events,
                    event_id=event_id,
                    tmin=tmin,
                    tmax=tmax,
                    baseline=baseline,
                    picks=picks,
                    preload=True,
                    verbose=False,
                )

                if ch_names is None:
                    ch_names = epochs.ch_names
                    sfreq = float(epochs.info["sfreq"])

                # copy=True para soltar el buffer de `epochs`/`raw` al liberar el sujeto.
                # float32: MNE entrega float64, pero para EEG la precisión simple
                # sobra y parte a la mitad la RAM de X_all (clave con 87 sujetos).
                X = epochs.get_data(copy=True).astype(np.float32, copy=False)
                # Etiqueta por trial: invertimos event_id (código -> nombre).
                code_to_name = {v: k for k, v in event_id.items()}
                labels = [code_to_name[c] for c in epochs.events[:, -1]]

                X_list.append(X)
                y_list.extend(labels)
                meta_rows.extend(
                    {"subject": subject, "session": session, "run": run}
                    for _ in range(len(labels))
                )
        del data  # liberar los Raw de este sujeto antes del siguiente
        gc.collect()

    if not X_list:
        raise RuntimeError(
            "No se cargó ningún trial. Revisa el nombre de las clases y los sujetos."
        )

    X_all = np.concatenate(X_list, axis=0)
    y_all = np.array(y_list)
    metadata = pd.DataFrame(meta_rows)

    return EpochedData(
        X=X_all,
        y=y_all,
        metadata=metadata,
        ch_names=ch_names or [],
        sfreq=float(sfreq or 0.0),
    )


def load_from_config(cfg: dict) -> EpochedData:
    """Atajo: carga el dataset usando la sección del YAML de configuración."""
    ds = cfg["dataset"]
    ep = cfg["epoching"]
    return load_dataset(
        name=ds["name"],
        subjects=ds["subjects"],
        classes=ds.get("classes"),
        tmin=ep["tmin"],
        tmax=ep["tmax"],
        picks=ep.get("picks", "eeg"),
        baseline=ep.get("baseline"),
    )
