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
    """Instancia un dataset de MOABB por su nombre de clase (p. ej. 'BNCI2014_001')."""
    import moabb.datasets as mds

    try:
        dataset_cls = getattr(mds, name)
    except AttributeError as exc:
        raise ValueError(
            f"Dataset '{name}' no existe en moabb.datasets. "
            f"Ejemplos válidos: BNCI2014_001, PhysionetMI, Cho2017."
        ) from exc
    return dataset_cls()


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
    data = dataset.get_data(subjects=subjects)  # descarga a la caché de MNE si falta

    X_list: list[np.ndarray] = []
    y_list: list[str] = []
    meta_rows: list[dict] = []
    ch_names: list[str] | None = None
    sfreq: float | None = None

    for subject in subjects:
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

                X = epochs.get_data(copy=False)  # (n_trials, n_canales, n_muestras)
                # Etiqueta por trial: invertimos event_id (código -> nombre).
                code_to_name = {v: k for k, v in event_id.items()}
                labels = [code_to_name[c] for c in epochs.events[:, -1]]

                X_list.append(X)
                y_list.extend(labels)
                meta_rows.extend(
                    {"subject": subject, "session": session, "run": run}
                    for _ in range(len(labels))
                )

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
