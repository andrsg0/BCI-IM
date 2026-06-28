# 9 · Scripts y uso — de cero a la demo

> Cómo correr todo de principio a fin: instalar, descargar datos, entrenar, validar, generar
> figuras, levantar el servidor y el frontend. Referencia de **todos** los scripts de
> `backend/scripts/`. Es la guía operativa para reproducir el proyecto (y la demo de la defensa).

---

## 9.1 Instalación

```bash
cd backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt        # torch se instala aparte (ver comentario en el archivo)
```

Dependencias clave y su rol (`requirements.txt`): **numpy/scipy** (scipy solo para *validar* nuestra
convolución/FIR, no para reemplazarla), **scikit-learn** (LDA, validación), **mne/moabb** (carga de
datasets), **matplotlib** (figuras), **fastapi/uvicorn/websockets** (API), **torch** (solo EEGNet).

> **Aviso de config (importante).** Los YAML viven en la **raíz** del repo (`configs/`), no en
> `backend/`. El servidor los resuelve bien, pero `config.py`'s `DEFAULT_CONFIG_PATH` apunta a
> `backend/configs/default.yaml` (que ya no existe). Por eso, al correr scripts a mano, pasa siempre
> **`--config ../configs/default.yaml`** (o el YAML del dataset).

---

## 9.2 Flujo de principio a fin

El orden lógico (mundo offline → online):

```bash
cd backend && source .venv/bin/activate

# 1) DATOS — descargar y cachear (una vez por máquina)
python scripts/setup_data.py   --config ../configs/default.yaml      # todos los sujetos
#   o, para un dataset/sujeto puntual y cachear el .npz:
python scripts/download_data.py --config ../configs/default.yaml --save

# 2) ENTRENAR y PERSISTIR los 4 regímenes (lo que usa el selector de la web)
python scripts/train_all_regimes.py --config ../configs/default.yaml

# 3) VALIDAR / consolidar la comparación de población (CSV de Resultados)
python scripts/compare_methods.py  --config ../configs/default.yaml
python scripts/assemble_compare.py --config ../configs/default.yaml   # arma compare_methods_<id>.csv

# 4) PRECOMPUTAR los payloads de visualización (portabilidad: web sin datos crudos)
python scripts/precompute_payloads.py --config ../configs/default.yaml

# 5) FIGURAS del informe (desde la caché local, sin descargar)
python scripts/make_report_figures.py --config ../configs/default.yaml

# 6) SERVIDOR (FastAPI en :8000)
python scripts/run_server.py [--reload]
```

Y el frontend en otra terminal:

```bash
cd frontend
npm install
npm run dev      # http://localhost:5173 (proxya /api y /ws a :8000)
```

---

## 9.3 Referencia de scripts

**Datos**

| Script | Qué hace |
|---|---|
| `download_data.py` | Carga la señal cruda y la epoca; `--save` cachea `X/y` en `data/processed/<id>_s<n>.npz`. |
| `setup_data.py` | Pre-descarga **todos** los sujetos de un dataset (máquina nueva), tolerando fallos. |
| `probe_dataset.py` | **Sonda de viabilidad** de cualquier dataset de MOABB (≥2 sesiones, izq/der, carga OK). |

**Entrenar / persistir** (mundo offline: "antes del streaming")

| Script | Qué hace |
|---|---|
| `train_model.py` | Entrena y guarda el modelo **CSP+LDA** de cada sujeto (`.pkl` + ficha `.json`). |
| `train_eegnet.py` | Entrena **EEGNet** y lo compara con CSP+LDA (within-subject, k-fold). |
| `train_eegnet_pooled.py` | EEGNet **pooled** cross-subject + **LOSO** (el más caro; solo se corrió en 2a). |
| `train_all_regimes.py` | Entrena y **persiste los 4 regímenes** (CSP/EEGNet × within/cross) — lo que alimenta el selector de la web. |

**Validar / comparar**

| Script | Qué hace |
|---|---|
| `run_offline.py` | Pipeline offline end-to-end: carga → FIR → CSP → log-var → LDA + métricas. |
| `evaluate_all.py` | Evaluación **within-subject** (k-fold) de todos los sujetos. |
| `compare_methods.py` | Comparación **2×2** (CSP+LDA vs EEGNet, within vs cross) por sujeto. |
| `assemble_compare.py` | Ensambla la matriz 2×2 de Resultados (`compare_methods_<id>.csv`) desde las fichas. |
| `sweep_shrinkage.py` | Barrido del **shrinkage** del CSP (antes/después). |
| `eval_fbcsp.py` | Experimento honesto **FBCSP** (banco de filtros) vs CSP de banda única. |

**Servir / demo / figuras**

| Script | Qué hace |
|---|---|
| `run_server.py` | Levanta la **API FastAPI** en `:8000` (`--reload` para desarrollo). |
| `run_live_sim.py` | Simulación de streaming en consola (`--realtime` para ritmo real). |
| `precompute_payloads.py` | Precomputa los **payloads de viz** (`viz_*.json`) → web sin datos crudos. |
| `make_report_figures.py` | Genera **las figuras de este informe** desde la caché (`*.npz`), sin descargar. |
| `demo_dsp.py` / `demo_csp.py` | Figuras didácticas sueltas (DSP / CSP). |

---

## 9.4 Tests

```bash
cd backend
python -m pytest -q                       # todos
python -m pytest tests/test_dsp.py -v     # un archivo
python -m pytest tests/test_csp.py::test_name -v
```

Los tests cumplen un papel **conceptual**, no solo de regresión: verifican que nuestras
implementaciones **a mano** coinciden con las de referencia — la convolución/FIR contra `scipy`, la
respuesta en frecuencia contra la FFT, el LDA contra `sklearn`, y los payloads precomputados
**byte-a-byte** contra los calculados al vuelo (`test_payloads.py`). Es decir: prueban que la teoría
LTI explícita está bien implementada. No hay linter/formatter configurado; se sigue el estilo del
entorno.

---

## 9.5 Las figuras del informe

`make_report_figures.py` es el que genera **todas** las imágenes de `docs/informe/figures/` a partir
del **código real** del pipeline y de la caché local (`data/processed/BNCI2014_001_s1.npz` +
modelos persistidos), sin volver a descargar nada de MOABB. Así lo del informe **coincide** con lo
que se programó. Mapa figura → sección:

| Figura | Sección | Cómo se genera |
|---|---|---|
| `00-pipeline.png` | 0 | esquema FIR→CSP→log-var→LDA |
| `01-epoching.png` | 1 | línea de tiempo del trial (epoch + ventana + transitorio) |
| `02-fir-impulse/frequency/filter-effect/mac.png` | 2 | `h[n]`, `\|H\|`, crudo-vs-filtrado, operación MAC |
| `03-csp-patterns/logvar-separability.png` | 3 | topomapas de patrones + nube de log-varianza |
| `04-lda-boundary.png` | 4 | nube + frontera del LDA |
| `05-eegnet-temporal/spatial.png` | 5 | filtros aprendidos vs FIR/CSP (carga el modelo persistido) |
| `06-regimenes.png` | 6 | barras de los 4 regímenes (lee `compare_methods_*.csv`) |
| `07-streaming.png` | 7 | esquema de la ventana deslizante causal con reposo |

> **Tras reentrenar:** vuelve a correr `precompute_payloads.py` (payloads de la web) y, si quieres
> figuras actualizadas, `make_report_figures.py`.

---

## 9.6 Recapitulación

Con esto, el informe cubre la cadena completa: la **teoría LTI** (secciones 0–4), su **espejo
aprendido** (5), la **validación honesta** (6), el **tiempo real** (7) y la **interfaz** (8), todo
anclado al código real y reproducible con los scripts de esta sección. La Etapa 3
(interoperabilidad: LSL/Arduino) queda como trabajo futuro, con la arquitectura ya preparada para
recibir una entrada en vivo por LSL del casco Ultracortex.

---

**Volver al** [índice del informe](00-vision-y-teoria-LTI.md).
