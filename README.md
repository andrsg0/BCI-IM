# BCI — Clasificación de Imaginación Motora (Proyecto Sistemas Lineales y Señales)

Interfaz Cerebro-Computadora (BCI) para clasificar **imaginación motora** a partir de
señales EEG, construida como demostración práctica de la teoría de **sistemas LTI**:
convolución discreta, filtros FIR, filtrado espacial lineal (CSP) y respuesta en frecuencia.

> Sin hardware por ahora: se trabaja sobre datasets públicos de máxima calidad.
> La arquitectura deja preparada la futura integración con el casco **Ultracortex Mark IV** (vía LSL).

## Teoría LTI en el código (dónde mirar)

| Concepto LTI                          | Archivo                                  |
|---------------------------------------|------------------------------------------|
| Convolución discreta `y[n]=Σ h[k]x[n-k]` y operación MAC | `backend/src/bci/dsp/convolution.py` |
| Filtro FIR pasa-banda (µ/β) por convolución | `backend/src/bci/dsp/fir_filters.py` |
| Respuesta en frecuencia `H(e^jω)`     | `backend/src/bci/dsp/frequency_response.py` |
| CSP = filtro espacial lineal (maximiza varianza) | `backend/src/bci/spatial/csp.py` |
| Clasificador lineal (LDA)             | `backend/src/bci/models/lda.py`          |

## Estructura

```
backend/   Pipeline DSP/ML en Python (Etapa 1)
frontend/  Interfaz React didáctica: convolución, pesos CSP, cerebro 3D (Etapa 2)
docs/      Documentación y notas teóricas
```

## Etapas del proyecto

1. **Pipeline LTI + clasificación offline + simulación de transmisión en vivo.** ← *completa*
2. **Interfaz React** con módulos didácticos (convolución, pesos espaciales, cerebro 3D reactivo).
3. **Interoperabilidad**: control de videojuegos (LSL), Arduino (Serial), etc.

## Datasets soportados

Cada dataset tiene su propio archivo de configuración en `backend/configs/`. El pipeline
detecta la frecuencia de muestreo real de los datos (**fs dinámico**), así que el mismo código
sirve para datasets con distinto `fs`.

| Config | Dataset | Sujetos | `fs` |
|---|---|---|---|
| `configs/default.yaml` | BCI Competition IV 2a (`BNCI2014_001`) | 9 | 250 Hz |
| `configs/physionet.yaml` | PhysioNet EEG MMI (`PhysionetMI`) | 109 | 160 Hz |
| `configs/liu2024.yaml` | Liu2024 (Nature Sci. Data, 2024) | 50 | 500 Hz |

## Puesta en marcha (backend)

```bash
cd backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Descargar/verificar dataset (por defecto BCI IV 2a)
python scripts/download_data.py --save

# Pipeline offline end-to-end (acepta --config para otro dataset)
python scripts/run_offline.py
python scripts/run_offline.py --config configs/physionet.yaml

# Evaluación within-subject de todos los sujetos (media ± std + figura)
python scripts/evaluate_all.py
python scripts/evaluate_all.py --config configs/physionet.yaml --subjects 1 2 3

# Simulación de transmisión en vivo (causal); --realtime para velocidad real
python scripts/run_live_sim.py

# Servidor API (FastAPI) para el frontend — REST + WebSocket en :8000
python scripts/run_server.py

# Figuras didácticas (respuesta en frecuencia, MAC, topomapas CSP)
python scripts/demo_dsp.py && python scripts/demo_csp.py
```
