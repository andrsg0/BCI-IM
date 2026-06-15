"""Visualizaciones didácticas del pipeline LTI (figuras estáticas, matplotlib).

Pensadas para el informe y para validar visualmente. La versión INTERACTIVA de
estas ideas vivirá en el frontend React (Etapa 2); aquí generamos PNGs.

Usamos el backend 'Agg' (sin ventana) para poder guardar figuras en entornos sin
display.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from bci.dsp.convolution import mac_terms  # noqa: E402
from bci.dsp.fir_filters import FIRFilter  # noqa: E402
from bci.dsp.frequency_response import frequency_response  # noqa: E402


def plot_impulse_response(filt: FIRFilter, ax=None):
    """Dibuja la respuesta al impulso h[n] (los coeficientes del FIR)."""
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 3))
    n = np.arange(filt.num_taps)
    ax.stem(n, filt.h, basefmt=" ")
    ax.axvline(filt.group_delay, color="r", ls="--", lw=1,
               label=f"centro (retardo={filt.group_delay:g})")
    ax.set(title=f"Respuesta al impulso h[n] — FIR {filt.num_taps} taps "
                 f"({filt.low_hz}-{filt.high_hz} Hz, {filt.window})",
           xlabel="n (muestra)", ylabel="h[n]")
    ax.legend(); ax.grid(alpha=0.3)
    return ax


def plot_frequency_response(filt: FIRFilter, ax=None):
    """Dibuja |H(e^jω)| en dB con la banda de paso µ/β resaltada."""
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 3))
    fr = frequency_response(filt.h, filt.fs, n_freqs=1024)
    ax.plot(fr.freqs_hz, fr.magnitude_db)
    ax.axvspan(filt.low_hz, filt.high_hz, color="green", alpha=0.12,
               label=f"banda µ/β [{filt.low_hz}-{filt.high_hz}] Hz")
    ax.set(title="Respuesta en frecuencia |H(e^jω)|",
           xlabel="Frecuencia (Hz)", ylabel="Magnitud (dB)",
           ylim=(-80, 5), xlim=(0, filt.fs / 2))
    ax.legend(); ax.grid(alpha=0.3)
    return ax


def plot_filter_effect(x_raw, x_filt, fs, ax=None):
    """Compara un trial antes/después del filtro, en tiempo y frecuencia."""
    if ax is None:
        _, ax = plt.subplots(1, 2, figsize=(11, 3))
    t = np.arange(len(x_raw)) / fs
    ax[0].plot(t, x_raw, lw=0.8, alpha=0.6, label="cruda")
    ax[0].plot(t, x_filt, lw=0.9, label="filtrada µ/β")
    ax[0].set(title="Señal (dominio del tiempo)", xlabel="t (s)", ylabel="µV")
    ax[0].legend(); ax[0].grid(alpha=0.3)

    f = np.fft.rfftfreq(len(x_raw), 1 / fs)
    ax[1].semilogy(f, np.abs(np.fft.rfft(x_raw)) + 1e-12, lw=0.8, alpha=0.6, label="cruda")
    ax[1].semilogy(f, np.abs(np.fft.rfft(x_filt)) + 1e-12, lw=0.9, label="filtrada µ/β")
    ax[1].set(title="Espectro (dominio de la frecuencia)", xlabel="Hz",
              ylabel="|X(f)|", xlim=(0, fs / 2))
    ax[1].legend(); ax[1].grid(alpha=0.3)
    return ax


def plot_subject_results(subjects, kfold_acc, intersession_acc=None, title="Evaluación por sujeto — BCI IV 2a"):
    """Barras de accuracy por sujeto + medias.

    Si `intersession_acc` es None (datasets de 1 sesión), dibuja solo k-fold.
    """
    import numpy as np

    subjects = list(subjects)
    x = np.arange(len(subjects))
    fig, ax = plt.subplots(figsize=(1.1 * len(subjects) + 2, 4))

    if intersession_acc is None:
        ax.bar(x, kfold_acc, 0.6, label="k-fold CV", color="C0")
    else:
        w = 0.38
        ax.bar(x - w / 2, kfold_acc, w, label="k-fold CV", color="C0")
        ax.bar(x + w / 2, intersession_acc, w, label="inter-sesión", color="C1")
        m_is = np.mean(intersession_acc)
        ax.axhline(m_is, color="C1", ls="--", lw=1, label=f"media inter-sesión = {m_is:.3f}")

    ax.axhline(np.mean(kfold_acc), color="C0", ls="--", lw=1,
               label=f"media k-fold = {np.mean(kfold_acc):.3f}")
    ax.axhline(0.5, color="gray", ls=":", lw=1, label="azar (0.5)")

    ax.set(title=title, xlabel="Sujeto", ylabel="Accuracy", ylim=(0, 1))
    ax.set_xticks(x)
    ax.set_xticklabels([str(s) for s in subjects])
    ax.legend(fontsize=8, ncol=2)
    ax.grid(axis="y", alpha=0.3)
    return fig


def plot_csp_patterns(patterns, ch_names, eigenvalues=None, title="Patrones espaciales CSP"):
    """Dibuja los patrones espaciales CSP como topomapas sobre el cuero cabelludo.

    Cada columna de `patterns` (n_canales x n_componentes) es un patrón: indica
    cómo se proyecta una fuente cortical sobre los electrodos. Los componentes
    extremos (λ alto/bajo) deberían concentrarse sobre la corteza motora (C3/C4),
    lateralizados según la clase.
    """
    import mne

    n_comp = patterns.shape[1]
    info = mne.create_info(ch_names=list(ch_names), sfreq=250.0, ch_types="eeg")
    info.set_montage("standard_1005", match_case=False, on_missing="ignore")

    fig, axes = plt.subplots(1, n_comp, figsize=(2.4 * n_comp, 2.8))
    if n_comp == 1:
        axes = [axes]
    for i, ax in enumerate(axes):
        mne.viz.plot_topomap(patterns[:, i], info, axes=ax, show=False, contours=4)
        lab = f"comp {i}"
        if eigenvalues is not None:
            lab += f"\nλ={eigenvalues[i]:.2f}"
        ax.set_title(lab, fontsize=9)
    fig.suptitle(title)
    return fig


def plot_mac_operation(x, h, n, ax=None):
    """Visualiza la operación MAC que produce y[n]: kernel volteado sobre la señal.

    Muestra x[m], el kernel h[n-m] alineado, y los productos que se acumulan.
    Es la base de la animación ConvolutionViz del frontend.
    """
    x = np.asarray(x, float); h = np.asarray(h, float)
    if ax is None:
        _, ax = plt.subplots(figsize=(9, 3))
    m = np.arange(len(x))
    ax.stem(m, x, linefmt="C0-", markerfmt="C0o", basefmt=" ", label="x[m]")
    # kernel volteado y desplazado: h[n-m]
    h_pos = [(n - k, h[k]) for k in range(len(h)) if 0 <= n - k < len(x)]
    if h_pos:
        mm, hh = zip(*h_pos)
        ax.stem(mm, hh, linefmt="C1-", markerfmt="C1s", basefmt=" ", label="h[n-m] (volteado)")
    terms = mac_terms(x, h, n)
    ax.set(title=f"Operación MAC en n={n}:  y[{n}] = Σ h[k]·x[{n}-k] = {terms['y_n']:.3f}",
           xlabel="m (muestra)")
    ax.legend(); ax.grid(alpha=0.3)
    return ax
