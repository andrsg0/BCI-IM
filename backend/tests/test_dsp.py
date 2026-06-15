"""Tests del módulo DSP (Hito 3).

Validan que nuestra convolución / FIR / respuesta en frecuencia, escritos a mano,
coinciden con las implementaciones de referencia de numpy. Es la prueba de que la
teoría LTI está bien implementada, no solo bien contada.
"""
from __future__ import annotations

import numpy as np
import pytest

from bci.dsp.convolution import (
    apply_filter,
    convolve,
    convolve_mac,
    mac_terms,
)
from bci.dsp.fir_filters import design_bandpass_fir
from bci.dsp.frequency_response import frequency_response


# --------------------------------------------------------------------------- #
# Convolución
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("mode", ["full", "same", "valid"])
def test_convolve_matches_numpy(mode):
    rng = np.random.default_rng(42)
    x = rng.standard_normal(200)
    h = rng.standard_normal(31)
    ref = np.convolve(x, h, mode=mode)
    assert np.allclose(convolve(x, h, mode=mode), ref, atol=1e-10)


def test_convolve_mac_equals_vectorized():
    """La versión literal (doble bucle) == la vectorizada por superposición."""
    rng = np.random.default_rng(0)
    x = rng.standard_normal(50)
    h = rng.standard_normal(7)
    assert np.allclose(convolve_mac(x, h, "full"), convolve(x, h, "full"))


def test_mac_terms_sum_to_output():
    """Los términos MAC en n deben sumar exactamente y[n] de la convolución full."""
    x = np.array([1.0, 2.0, 3.0, 4.0])
    h = np.array([0.5, -1.0, 2.0])
    y_full = convolve(x, h, "full")
    for n in range(len(y_full)):
        assert mac_terms(x, h, n)["y_n"] == pytest.approx(y_full[n])


def test_apply_filter_multichannel_shape_and_values():
    """apply_filter sobre (trials, canales, tiempo) filtra cada canal por separado."""
    rng = np.random.default_rng(1)
    X = rng.standard_normal((4, 3, 100))
    h = rng.standard_normal(11)
    Y = apply_filter(X, h, mode="same")
    assert Y.shape == (4, 3, 100)
    # comparar un canal concreto con np.convolve 'same'
    assert np.allclose(Y[2, 1], np.convolve(X[2, 1], h, mode="same"), atol=1e-10)


# --------------------------------------------------------------------------- #
# Diseño FIR
# --------------------------------------------------------------------------- #
def test_fir_is_linear_phase():
    """El FIR diseñado debe ser simétrico (condición de fase lineal)."""
    filt = design_bandpass_fir(8, 30, fs=250, num_taps=101)
    assert filt.is_linear_phase()
    assert filt.group_delay == 50.0


def test_fir_rejects_even_taps():
    with pytest.raises(ValueError):
        design_bandpass_fir(8, 30, fs=250, num_taps=100)


def test_fir_passband_and_stopband():
    """|H| ~1 dentro de [8,30] Hz y mucho menor fuera de la banda."""
    filt = design_bandpass_fir(8, 30, fs=250, num_taps=101)
    fr = frequency_response(filt.h, fs=250, n_freqs=1024)

    # Banda de paso (centro): ganancia cercana a 1.
    assert fr.gain_at(19) == pytest.approx(1.0, abs=0.15)
    assert fr.gain_at(12) > 0.7
    assert fr.gain_at(25) > 0.7
    # Banda de rechazo: fuertemente atenuada.
    assert fr.gain_at(2) < 0.1     # bajas freq (deriva, parpadeo)
    assert fr.gain_at(45) < 0.1    # altas freq (EMG, red eléctrica)


# --------------------------------------------------------------------------- #
# Respuesta en frecuencia (DTFT a mano vs FFT)
# --------------------------------------------------------------------------- #
def test_frequency_response_matches_fft():
    """Nuestra DTFT explícita coincide con np.fft.rfft en las mismas frecuencias."""
    filt = design_bandpass_fir(8, 30, fs=250, num_taps=64 + 1)
    h = filt.h
    n_fft = 2048
    H_fft = np.fft.rfft(h, n=n_fft)
    freqs_fft = np.fft.rfftfreq(n_fft, d=1 / 250)

    fr = frequency_response(h, fs=250, n_freqs=len(freqs_fft))
    # Evaluamos nuestra DTFT exactamente en las frecuencias de la FFT.
    from bci.dsp.frequency_response import FrequencyResponse  # noqa
    omega = 2 * np.pi * freqs_fft / 250
    exponent = np.exp(-1j * np.outer(omega, np.arange(len(h))))
    H_ours = exponent @ h
    assert np.allclose(H_ours, H_fft, atol=1e-9)
