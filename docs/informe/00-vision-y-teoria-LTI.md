# 0 · Visión del proyecto y teoría LTI

> Documentación de referencia para el informe y la defensa. Cada sección cubre cuatro ejes:
> **teoría**, **decisiones técnicas**, **scripts/código** y **cómo se representa en la página**.
> Todo está verificado contra el código real del repositorio (no contra planes antiguos).

---

## 0.1 Qué es y para qué

Construimos una **Interfaz Cerebro-Computadora (BCI)** que clasifica **imaginación motora**
a partir de señales **EEG**: a partir de la actividad eléctrica del cuero cabelludo, decide
qué mano está *imaginando* mover la persona (**mano izquierda vs. mano derecha** — problema
binario de 2 clases).

El valor del proyecto **no es la BCI en sí**, sino que **cada etapa del procesamiento es una
aplicación directa y explícita de la teoría de sistemas LTI** (lineales e invariantes en el
tiempo): convolución discreta, filtros FIR, respuesta en frecuencia y filtrado espacial lineal.
Por eso el código implementa estas operaciones **a mano** en lugar de esconderlas tras llamadas
de librería (decisión central, ver 0.6).

No hay hardware todavía: todo corre sobre **datasets EEG públicos** (vía MOABB). La arquitectura
deja preparada una futura integración con el casco **Ultracortex Mark IV** (vía LSL).

---

## 0.2 Las tres etapas del proyecto

| Etapa | Qué es | Estado |
|---|---|---|
| **1 · Pipeline LTI + offline + simulación en vivo** | El backend Python: DSP (FIR/convolución), CSP, log-varianza, LDA, validación honesta y un simulador de streaming causal. | **Hecha** |
| **2 · Frontend didáctico** | SPA en React que *expone explícitamente* la convolución, los pesos espaciales (CSP), la frontera del LDA y un cerebro 3D reactivo. | **En curso** |
| **3 · Interoperabilidad** | Capa para controlar juegos (LSL), Arduino (Serial), etc. | **No iniciada** |

Esta documentación cubre las etapas 1 y 2, que son las que están programadas.

---

## 0.3 El hilo conductor: teoría LTI explícita en el código

El mapa entre cada concepto de la asignatura y el archivo que lo implementa:

| Concepto LTI | Dónde está en el código | Sección |
|---|---|---|
| **Convolución discreta** `y[n] = Σ h[k]·x[n−k]` y la operación **MAC** | `backend/src/bci/dsp/convolution.py` (`convolve_mac`, `mac_terms`, `convolve`, `apply_filter`) | [2](02-fir-convolucion.md) |
| **Filtro FIR pasa-banda** (banda µ/β, 8–30 Hz) por *windowed-sinc* | `backend/src/bci/dsp/fir_filters.py` (`design_bandpass_fir`) | [2](02-fir-convolucion.md) |
| **Respuesta en frecuencia** `H(e^jω)` | `backend/src/bci/dsp/frequency_response.py` | [2](02-fir-convolucion.md) |
| **Causalidad y retardo de grupo** (filtrado por bloques en vivo) | `backend/src/bci/streaming/simulator.py` (`CausalFIR`) | [7](07-streaming-en-vivo.md) |
| **Filtrado espacial lineal (CSP)** `Z = W·X`, autovalores generalizados (Koles 1990) | `backend/src/bci/spatial/csp.py` | [3](03-csp.md) |
| **Extracción de características**: log-varianza por componente | `backend/src/bci/features/log_variance.py` | [3](03-csp.md) |
| **Clasificador lineal (LDA)**: frontera = hiperplano | `backend/src/bci/models/lda.py` | [4](04-lda.md) |
| **EEGNet** como *espejo aprendido* de las etapas LTI | `backend/src/bci/models/eegnet.py` | [5](05-eegnet.md) |

> **Idea para la defensa:** EEGNet no es "otro clasificador" sino una comprobación: si dejamos
> que una red **aprenda sola** los filtros, ¿redescubre lo que diseñamos a mano? Su convolución
> temporal ≈ el FIR, la *depthwise* ≈ el CSP, el *pooling* ≈ la log-varianza, y la capa densa ≈
> el LDA. (Detalle en la sección 5.)

---

## 0.4 El pipeline central

La abstracción núcleo es **`MotorImageryPipeline`** (`backend/src/bci/pipeline/offline.py`):
una cadena `fit`/`predict` de cuatro etapas **lineales** encadenadas:

```
   EEG crudo                FIR (fijo)        CSP            log-varianza        LDA
(canales × tiempo) ──► banda µ/β 8–30 Hz ─► Z = W·X ─► energía por componente ─► y = w·F + b
                         (convolución)      (espacial)      (característica)     (decisión)
```

- El FIR tiene **coeficientes fijos** (diseñados, no aprendidos): aplicarlo **no introduce fuga
  de datos**.
- El CSP y el LDA **se ajustan solo con la partición de entrenamiento** de cada split (disciplina
  anti-fuga; ver sección 6).
- El pipeline se construye a partir de la **frecuencia de muestreo real** del dataset (`fs`
  dinámico), así que el mismo código sirve para 250 Hz (BCI IV 2a/2b) y 512 Hz (Kumar2024).

---

## 0.5 Los dos mundos: offline vs online

Todo el proyecto se organiza en dos "mundos" con una distinción que se mantiene visible en la UI:

| Mundo | Cuándo ocurre | Qué incluye | Color en la UI |
|---|---|---|---|
| **Offline** | *antes* del streaming, una sola vez | entrenar y validar el modelo (CSP/LDA, métricas) | ámbar |
| **Online** | *ahora*, en tiempo real | la señal llega y se clasifica ventana a ventana, de forma **causal** | verde esmeralda |

La diferencia es teóricamente importante: offline se puede filtrar usando muestras futuras
(`mode='same'`), pero **en vivo no** — hay que mantener estado entre bloques y asumir un retardo
de grupo real (sección 7).

En el frontend, cada página está etiquetada con su mundo (`frontend/src/lib/nav.ts`,
`WORLD_STYLE`), y la navegación las agrupa:

- **General:** Inicio, Dashboard, Glosario.
- **Modelo (offline):** Entrenamiento (`/csp`, "El Modelo"), Resultados (`/results`).
- **En vivo (online):** Laboratorio (`/lab`), Clasificación (`/live`), Benchmark (`/demo`), Cerebro 3D (`/brain`).

---

## 0.6 Decisiones técnicas globales

Decisiones transversales que conviene justificar en la defensa:

1. **Convolución a mano, no `scipy.signal.lfilter`.** `lfilter` resuelve una ecuación en
   diferencias (puede ser IIR); nosotros queremos la **convolución explícita** de un FIR. Por eso
   `dsp/convolution.py` la implementa en tres formas: `convolve_mac` (doble bucle literal,
   didáctico), `mac_terms` (expone cada producto MAC para la visualización del frontend) y
   `convolve` (vectorizada, la de producción). *Coherente con el objetivo académico.*
2. **`fs` dinámico.** Se detecta la frecuencia de muestreo real de cada dataset desde los epochs
   en vez de hardcodearla; el mismo pipeline/config sirve para datasets con distinto `fs`.
3. **Disciplina anti-fuga de datos.** CSP y LDA se ajustan **solo** en el train de cada split;
   el FIR (fijo) no filtra información de las etiquetas. La validación usa k-fold estratificado e
   inter-sesión (sección 6).
4. **Problema binario (2 clases).** `left_hand` vs `right_hand`. El CSP de Koles que usamos es
   binario por construcción; las consecuencias (no hay clase "reposo") se tratan en la sección 7.
5. **EEGNet es secundario.** Se incluye como *espejo* de la teoría LTI; **sí** se usa además para
   inferencia en vivo (decisión revertida en jun-2026), pero la pieza central sigue siendo el
   pipeline LTI clásico.
6. **Sin base de datos.** Todo se carga fresco de MOABB/MNE, se computa al vuelo o se persiste como
   artefactos planos `.pkl`/`.json` por dataset.

---

## 0.7 Cómo se representa en la página

- **Inicio** (`frontend/src/pages/Home.tsx`): hero que decodifica el acrónimo (BCI·MI = Brain-
  Computer Interface · Motor Imagery) y ofrece los dos accesos que representan los dos mundos
  ("Explorar el laboratorio" / "Ver demo en vivo").
- El resto de páginas materializan cada etapa del pipeline:
  - **Laboratorio** → FIR y convolución en vivo (sección 2).
  - **Entrenamiento / "El Modelo"** → CSP + log-varianza + LDA, y EEGNet como pestaña (secciones 3–5).
  - **Resultados** y **Benchmark** → validación honesta (sección 6).
  - **Clasificación** → el pipeline completo en tiempo real (sección 7).
  - **Cerebro 3D** → potencia µ/β por electrodo en vivo (sección 7).
- Todas las páginas comparten un botón de ayuda `?` con la explicación teórica concisa
  (auto-enlazada al [Glosario](../glosario.md)).

---

## 0.8 Stack (resumen)

- **Backend:** Python · NumPy · MNE/MOABB (carga de datos) · PyTorch (solo EEGNet) ·
  FastAPI (sirve el pipeline al frontend por REST + WebSocket). Sin linter/formatter
  configurado; se sigue el estilo del entorno.
- **Frontend:** React 19 + TypeScript + Vite · Tailwind v4 · Zustand (estado global) ·
  uPlot (señal en vivo) · Recharts (gráficos analíticos) · three.js/@react-three/fiber
  (cerebro 3D) · Framer Motion · lucide-react.

El detalle del frontend está en la [sección 8](08-frontend.md); el de cada script, en la
[sección 9](09-scripts-y-uso.md).
