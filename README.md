# BCI — Clasificación de Imaginación Motora

Interfaz Cerebro-Computadora (BCI) para clasificar **imaginación motora** a partir de
señales EEG, construida como demostración práctica de la teoría de **sistemas LTI**:
convolución discreta, filtros FIR, filtrado espacial lineal (CSP) y respuesta en frecuencia.

> **Proyecto universitario — Sistemas Lineales y Señales.**
> Sin hardware por ahora: se trabaja sobre datasets públicos de máxima calidad.
> La arquitectura deja preparada la futura integración con el casco **Ultracortex Mark IV** (vía LSL).

---

## Teoría LTI en el código (dónde mirar)

| Concepto LTI                          | Archivo                                  |
|---------------------------------------|------------------------------------------|
| Convolución discreta `y[n]=Σ h[k]x[n-k]` y operación MAC | `backend/src/bci/dsp/convolution.py` |
| Filtro FIR pasa-banda (µ/β) por convolución | `backend/src/bci/dsp/fir_filters.py` |
| Respuesta en frecuencia `H(e^jω)`     | `backend/src/bci/dsp/frequency_response.py` |
| CSP = filtro espacial lineal (maximiza varianza) | `backend/src/bci/spatial/csp.py` |
| Clasificador lineal (LDA)             | `backend/src/bci/models/lda.py`          |

## Estructura del proyecto

```
backend/   Pipeline DSP/ML en Python (Etapa 1) + API FastAPI (Etapa 2)
frontend/  Interfaz React didáctica: convolución, pesos CSP, cerebro 3D (Etapa 2)
configs/   Archivos YAML de configuración por dataset (en la RAÍZ, no en backend/)
docs/      Documentación, glosario, diseño del frontend y notas teóricas
```

## Etapas del proyecto

1. **Pipeline LTI + clasificación offline + simulación de transmisión en vivo.** ← *completa*
2. **Interfaz React** con módulos didácticos (convolución, pesos espaciales, cerebro 3D reactivo). ← *en progreso*
3. **Interoperabilidad**: control de videojuegos (LSL), Arduino (Serial), etc.

## Datasets soportados

Cada dataset tiene su propio archivo de configuración en `configs/`. El pipeline
detecta la frecuencia de muestreo real de los datos (**fs dinámico**), así que el mismo código
sirve para datasets con distinto `fs`.

| Config | Dataset | Sujetos | `fs` | Sesiones |
|---|---|---|---|---|
| `configs/default.yaml` | BCI Competition IV 2a (`BNCI2014_001`) | 9 | 250 Hz | 2 |
| `configs/bci2b.yaml` | BCI Competition IV 2b (`BNCI2014_004`) | 9 | 250 Hz | 5 |
| `configs/kumar2024.yaml` | Kumar2024 (Nature Sci. Data) | 18 | 512 Hz | 6 |

> Los datasets con ≥ 2 sesiones sirven para la **demo en vivo** (estimación honesta
> inter-sesión: entrenar en sesión 1, evaluar en sesión 2).

---

## Requisitos previos (máquina nueva)

| Herramienta | Versión mínima | Cómo verificar |
|---|---|---|
| **Python** | ≥ 3.11 | `python3 --version` |
| **pip** | (incluido con Python) | `pip --version` |
| **Node.js** | ≥ 18 | `node --version` |
| **npm** | ≥ 9 | `npm --version` |
| **Git** | cualquiera | `git --version` |

> **Sistema operativo:** Linux, macOS o Windows (WSL recomendado en Windows).
> **Disco:** ~6 GB libres para los datasets (la caché de MNE se guarda en `~/mne_data`).
> **RAM:** ≥ 8 GB recomendados (el entrenamiento cross-subject carga múltiples sujetos).

---

## Puesta en marcha completa (máquina nueva)

Sigue estos pasos **en orden** la primera vez que clones el proyecto en una máquina nueva.

### Paso 0 — Clonar el repositorio

```bash
git clone <URL_DEL_REPO>
cd BCI_Claude
```

### Paso 1 — Instalar dependencias del backend

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate        # En Windows: .venv\Scripts\activate

# Instalar todas las dependencias (incluye torch CPU para EEGNet)
pip install -r requirements.txt

# Instalar el paquete en modo editable (para que `import bci` funcione)
pip install -e .
```

> **Nota sobre PyTorch:** `requirements.txt` instala `torch` con CPU por defecto.
> Si tienes GPU NVIDIA y quieres acelerar el entrenamiento de EEGNet, instala
> la versión CUDA manualmente:
> ```bash
> pip install torch --index-url https://download.pytorch.org/whl/cu121
> ```

### Paso 2 — Descargar los datasets (EEG)

Los datos NO están en el repositorio (son pesados). Se descargan de servidores
públicos (MOABB/MNE) y se cachean en `~/mne_data`. **Esto solo se hace una vez**
por máquina; las siguientes ejecuciones usan la caché.

```bash
# Desde backend/, con el venv activado:

# ── Opción A: Solo el dataset principal (BCI IV 2a, ~300 MB, rápido) ──
python scripts/setup_data.py --config ../configs/default.yaml

# ── Opción B: TODOS los datasets (recomendado, ~6 GB total) ──
python scripts/setup_data.py --config ../configs/default.yaml
python scripts/setup_data.py --config ../configs/bci2b.yaml
python scripts/setup_data.py --config ../configs/kumar2024.yaml
```

> **`setup_data.py`** descarga TODOS los sujetos de un dataset de una sola vez,
> es tolerante a fallos (si un sujeto falla, continúa y reporta cuáles reintentar),
> y es idempotente (si ya están descargados, no vuelve a bajar nada).
>
> **⚠ Kumar2024** es un ZIP monolítico de ~4.47 GB: el primer sujeto descarga
> el archivo completo; los demás van instantáneos.

### Paso 3 — Entrenar los modelos

Los modelos entrenados (`.pkl` + ficha `.json`) tampoco están en el repositorio
(están gitignoreados). Hay que generarlos:

```bash
# Desde backend/, con el venv activado:

# ── Entrenamiento completo del dataset principal (BCI IV 2a, 9 sujetos) ──
# Los 4 regímenes: CSP+LDA within/cross, EEGNet within/cross
python scripts/train_all_regimes.py \
    --config ../configs/default.yaml \
    --subjects 1 2 3 4 5 6 7 8 9

# ── BCI IV 2b (9 sujetos) ──
python scripts/train_all_regimes.py \
    --config ../configs/bci2b.yaml \
    --subjects 1 2 3 4 5 6 7 8 9

# ── Kumar2024 (18 sujetos; cross solo para algunos → más rápido) ──
python scripts/train_all_regimes.py \
    --config ../configs/kumar2024.yaml \
    --subjects $(seq 1 18) \
    --cross-subjects 1 2 5 7 9 18
```

> **⏱ Tiempos estimados:**
> - BCI IV 2a completo: ~20-40 min (CPU)
> - BCI IV 2b completo: ~15-30 min
> - Kumar2024 completo: ~1-3 horas
>
> **Reanudable:** si el proceso se interrumpe, volver a ejecutar el mismo comando
> omite lo ya entrenado (detección por checkpoint `.pkl` + `.json`). Usa `--force`
> para reentrenar todo.
>
> **Sin EEGNet** (más rápido): añade `--no-eegnet` para entrenar solo CSP+LDA.
>
> **En segundo plano** (para no perder progreso si cierras la terminal):
> ```bash
> nohup python scripts/train_all_regimes.py \
>     --config ../configs/default.yaml \
>     --subjects 1 2 3 4 5 6 7 8 9 > train.log 2>&1 &
> ```

### Paso 4 — Precomputar payloads de visualización (portabilidad)

Las páginas offline del frontend (El Modelo, Resultados, Entrenamiento) pueden
funcionar **sin datos crudos** si se precomputan los JSON de visualización.
Esto es necesario para portabilidad (p. ej. llevar la laptop a la presentación
sin necesitar descargar datos ahí).

```bash
# Desde backend/, con el venv activado:

python scripts/precompute_payloads.py --config ../configs/default.yaml
python scripts/precompute_payloads.py --config ../configs/bci2b.yaml
python scripts/precompute_payloads.py --config ../configs/kumar2024.yaml
```

> **Nota:** si no precomputas, el servidor calcula los payloads al vuelo desde
> los datos crudos (más lento la primera vez, pero funcional).

### Paso 5 — Instalar dependencias del frontend

```bash
cd ../frontend       # (o cd frontend desde la raíz del proyecto)
npm install
```

### Paso 6 — Levantar la aplicación

Se necesitan **dos terminales** (o una con `&`):

**Terminal 1 — Backend (FastAPI en :8000):**
```bash
cd backend
source .venv/bin/activate
python scripts/run_server.py
```

**Terminal 2 — Frontend (Vite en :5173):**
```bash
cd frontend
npm run dev
```

Abre **http://localhost:5173** en tu navegador. El frontend hace proxy automático
de `/api` y `/ws` al backend en `:8000` (configurado en `vite.config.ts`).

---

## Resumen rápido (copiar y pegar)

Para quien quiera levantar todo con los comandos mínimos:

```bash
# ── Backend ──
cd backend
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt && pip install -e .

# Descargar datos (dataset principal)
python scripts/setup_data.py --config ../configs/default.yaml

# Entrenar modelos (dataset principal, sin EEGNet para ir rápido)
python scripts/train_all_regimes.py --config ../configs/default.yaml \
    --subjects 1 2 3 4 5 6 7 8 9 --no-eegnet

# Precomputar visualizaciones
python scripts/precompute_payloads.py --config ../configs/default.yaml --no-eegnet

# Levantar servidor API
python scripts/run_server.py &

# ── Frontend ──
cd ../frontend
npm install
npm run dev
```

---

## Solo backend (sin interfaz web)

Si solo quieres usar el pipeline desde la terminal:

```bash
cd backend
source .venv/bin/activate

# Descargar y verificar un dataset
python scripts/download_data.py --save --config ../configs/default.yaml

# Pipeline offline end-to-end
python scripts/run_offline.py --config ../configs/default.yaml

# Evaluación within-subject de todos los sujetos (media ± std + figura)
python scripts/evaluate_all.py --config ../configs/default.yaml
python scripts/evaluate_all.py --config ../configs/default.yaml --subjects 1 2 3

# Simulación de transmisión en vivo (causal); --realtime para velocidad real
python scripts/run_live_sim.py --config ../configs/default.yaml

# Figuras didácticas (respuesta en frecuencia, MAC, topomapas CSP)
python scripts/demo_dsp.py && python scripts/demo_csp.py
```

> **⚠ Nota sobre `--config`:** los archivos YAML de configuración viven en `configs/`
> en la **raíz del repositorio** (no en `backend/configs/`). Cuando ejecutas scripts
> desde `backend/`, la ruta relativa es `../configs/default.yaml`. Si no pasas
> `--config`, algunos scripts intentarán buscar en `backend/configs/default.yaml`
> que **no existe** — pasa siempre el `--config` para evitar errores.

---

## Tests

```bash
cd backend
source .venv/bin/activate
python -m pytest -q                       # todos los tests
python -m pytest tests/test_dsp.py -v     # un archivo específico
python -m pytest tests/test_csp.py::test_name -v   # un test específico
```

---

## Troubleshooting

| Problema | Causa | Solución |
|---|---|---|
| `FileNotFoundError: ...configs/default.yaml` | Los YAML están en `configs/` (raíz), no en `backend/configs/` | Pasa `--config ../configs/default.yaml` explícitamente |
| `ModuleNotFoundError: bci` | Falta instalar el paquete en modo editable | `pip install -e .` desde `backend/` |
| La descarga de datos se queda colgada | Los servidores de MOABB/OSF pueden ser lentos | Reintentar; `setup_data.py` retoma donde se quedó |
| El frontend muestra errores de conexión | El backend no está corriendo | Levanta el backend primero (`python scripts/run_server.py`) |
| `CUDA out of memory` al entrenar EEGNet | GPU sin suficiente VRAM | Usa `--device cpu` o instala PyTorch CPU |
| Los gráficos offline no cargan | Faltan payloads precomputados y/o datos crudos | Corre `precompute_payloads.py` o asegúrate de que los datos estén descargados |
| Kumar2024 tarda mucho en descargar | Es un ZIP de ~4.47 GB | Paciencia; solo la primera vez. El progreso se cachea |
| `train_all_regimes.py` se interrumpió | El entrenamiento es largo | Volver a ejecutar el mismo comando (es reanudable por checkpoint) |

---

## Documentación adicional

| Documento | Contenido |
|---|---|
| `docs/pipeline.md` | Arquitectura completa del pipeline DSP/ML |
| `docs/frontend-design.md` | Decisiones de diseño del frontend |
| `docs/glosario.md` | Glosario de términos (servido vía `/api/glossary`) |
| `docs/entrenamiento.md` | Detalles del proceso de entrenamiento |
| `docs/datasets.md` | Documentación de los datasets soportados |
| `docs/presentacion.md` | Notas para la presentación del proyecto |
| `CLAUDE.md` | Contexto técnico detallado para desarrollo con IA |
