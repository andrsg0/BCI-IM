# Remuestreo a fs común y pooling cross-dataset

Documento para la **presentación** (tema de Sistemas Lineales y Señales): cómo juntamos
varios datasets EEG con distinta frecuencia de muestreo en un solo conjunto de entrenamiento,
respetando la teoría de muestreo. Incluye el proceso, los parámetros exactos y las fórmulas.

Código: `backend/src/bci/dsp/resampling.py`, `backend/src/bci/dsp/fir_filters.py`
(`design_lowpass_fir`), `backend/scripts/train_eegnet_crossdataset.py`.

---

## 1. El problema

Para que EEGNet generalice entre personas necesita **muchos sujetos diversos**. Un solo
dataset no basta, así que juntamos varios (PhysioNet + Dreyer2023 + Cho2017 ≈ 248 sujetos).
Pero vienen distintos:

| Dataset | fs (Hz) | nº canales | sujetos |
|---|---|---|---|
| PhysioNet MMI | 160 | 64 | 109 |
| Dreyer2023 | 512 | 27 | 87 |
| Cho2017 | 512 | 64 | 52 |
| *(BCI IV 2a)* | *250* | *22* | *9 (reservado para en vivo)* |

No se pueden apilar tal cual: las matrices tienen distinta longitud temporal y distintos
canales. Hay que **acondicionar la señal** antes de juntarla. Dos pasos: (A) llevar todo a
una **fs común** sin introducir aliasing, y (B) **armonizar el montaje** de canales.

## 2. Recordatorio: muestreo y aliasing

Una señal muestreada a `fs` solo puede representar frecuencias hasta `fs/2` (frecuencia de
**Nyquist**). Si al bajar la fs hay energía por encima de la nueva Nyquist, esa energía se
**solapa** (aliasing) y contamina la banda útil de forma irreversible. Por eso **diezmar no
es tirar muestras**: antes hay que filtrar pasa-bajo para quitar lo que no cabrá.

## 3. Remuestreo racional L/M (lo que implementamos)

Cambiar de `fs_in` a `fs_out` se expresa como un factor racional `L/M` (en forma irreducible):

```
L/M = fs_out / fs_in           (L = sobremuestreo, M = diezmado)
```

La cadena es **LTI pura**, tres bloques:

```
 x[n] ──► ↑L (insertar L-1 ceros) ──► h[n] (FIR pasa-bajo) ──► ↓M (1 de cada M) ──► y[m]
          sobremuestreo             anti-aliasing/anti-imágenes   diezmado
```

1. **Sobremuestreo ↑L:** insertar `L-1` ceros entre muestras. La fs intermedia es `L·fs_in`.
   En frecuencia esto crea "imágenes" del espectro que hay que limpiar.
2. **Filtro FIR pasa-bajo h[n]:** diseñado por *windowed-sinc* (igual que el FIR µ/β del
   proyecto, pero pasa-bajo). Hace de **anti-imágenes** (tras ↑L) y de **anti-aliasing**
   (antes de ↓M) a la vez. Aplicarlo es una **convolución** `y = Σ_k h[k]·x[n−k]`.
3. **Diezmado ↓M:** quedarse con 1 de cada M muestras. La fs final es `L·fs_in / M = fs_out`.

Caso `L = 1` (solo bajar, p. ej. 512→128 Hz): basta pasa-bajo + tomar 1 de cada M.

### Parámetros del filtro anti-aliasing

| Parámetro | Valor | Razón |
|---|---|---|
| **Frecuencia de corte** `fc` | `min(fs_in, fs_out) / 2` | la menor de las dos Nyquist (lo que sobrevive al diezmado) |
| **fs de diseño** | `L·fs_in` (la intermedia) | el filtro actúa tras el sobremuestreo |
| **nº de taps** | `≈ 20·max(L, M)` (impar) | más taps ⇒ transición más nítida ⇒ mejor rechazo de aliasing |
| **ventana** | Hamming | compromiso lóbulo lateral / ancho de transición |
| **ganancia** | `×L` | compensa la energía perdida al meter `L-1` ceros |

Fórmula del FIR pasa-bajo (windowed-sinc, `m` centrado en 0, `fc` normalizada a la fs de diseño):

```
h_ideal[m] = 2·fc·sinc(2·fc·m)          (sinc = pasa-bajos ideal)
h[m]       = h_ideal[m] · w_Hamming[m]   (truncar + ventanear: reduce Gibbs)
h[m]      /= Σ h[m]                       (ganancia unidad en DC, luego ×L)
```

Es **fase lineal** (h simétrico) ⇒ retardo de grupo constante `(N−1)/2`, sin distorsión de forma.

## 4. La fs común elegida: 128 Hz

**Por qué 128 Hz:**
- La banda útil de imaginación motora (µ/β) es 8–30 Hz. La nueva Nyquist (64 Hz) deja un
  margen amplio sobre 30 Hz: no se pierde nada relevante.
- Potencia de 2 y ratios limpios con los datasets del pool:

| Dataset | fs_in → fs_out | L/M (irreducible) | nº taps (≈20·max) |
|---|---|---|---|
| PhysioNet | 160 → 128 | **4/5** (↑4, ↓5) | 101 |
| Dreyer2023 | 512 → 128 | **1/4** (solo ↓4) | 81 |
| Cho2017 | 512 → 128 | **1/4** (solo ↓4) | 81 |

> **Nota:** BCI IV 2a (250 Hz) daría un ratio feo hacia 128 (64/125 → ~2500 taps), pero está
> **reservado para la demo en vivo**, no para el pool de entrenamiento, así que no entra aquí.
> Si se quisiera incluir, convendría otra fs común o aceptar el filtro largo.

## 5. Armonización del montaje (canales)

Cada dataset tiene distinto nº y nombres de canales. Tomamos la **intersección** de nombres:
los **19 canales motores/centrales comunes** a los cuatro datasets, reordenados a un orden
anatómico fijo e idéntico para todos:

```
FC3 FC1 FCz FC2 FC4 · C5 C3 C1 Cz C2 C4 C6 · CP3 CP1 CPz CP2 CP4 · Fz Pz
```

Son justamente los electrodos sobre la corteza motora (zona del ERD/ERS de la mano), así que
no se pierde información discriminativa al recortar a ellos.

## 6. Resultado: ventanas alineadas

Tras remuestrear a 128 Hz, la **ventana de clasificación** `[0.5, 2.5] s` (2 s de imaginación
activa) son exactamente `2 s × 128 Hz = 256 muestras` en **todos** los datasets. Junto con los
19 canales comunes, cada trial queda como un tensor idéntico:

```
(n_trials, 19 canales, 256 muestras)   ← mismo formato para PhysioNet, Dreyer y Cho
```

…y ya se pueden apilar en un solo pool. (Verificado en el smoke test: los 3 datasets, viniendo
de 160/512/512 Hz, salen los tres como `… × 19 × 256`.)

## 7. Validación del remuestreo

- **Correctitud:** comparado contra `scipy.signal.resample_poly` → **correlación = 1.0000**
  en los tres ratios (512→128, 160→128, 250→128).
- **Anti-aliasing:** un seno de 100 Hz (por encima de la nueva Nyquist de 64 Hz) tras
  remuestrear a 128 Hz queda con energía residual ≈ 0.01 (prácticamente eliminado, como debe).

## 8. El entrenamiento cross-dataset

Pipeline por dataset: cargar → **armonizar canales (19)** → **remuestrear a 128 Hz** →
banda 4–40 Hz + recorte ventana activa → apilar todo.

**Evaluación honesta = LEAVE-ONE-DATASET-OUT:** entrenar con 2 datasets y probar en el 3º.
Mide la generalización a un dataset **nuevo** (lo más exigente). El modelo base final se
entrena con los tres y se guarda como base para *fine-tuning*.

> Expectativa honesta: el leave-one-dataset-out suele ser **más bajo** que el cross-subject
> dentro de un mismo dataset (cada dataset tiene su hardware, montaje e instrucciones). El
> valor didáctico está en mostrar el acondicionamiento de señal y cuánto cuesta cruzar dominios.

## 9. Comandos (ejecutar en tu terminal — no consume tokens)

```bash
cd backend
source .venv/bin/activate

# Prueba rápida (pocos sujetos/épocas, no guarda):
python scripts/train_eegnet_crossdataset.py --max-subjects 5 --epochs 50 --augment --no-save

# Run real cross-dataset (subconjunto razonable):
nohup python scripts/train_eegnet_crossdataset.py \
    --datasets physionet dreyer2023 cho2017 \
    --max-subjects 20 --epochs 300 --augment \
    > /tmp/train_crossdataset.log 2>&1 &
tail -f /tmp/train_crossdataset.log

# Run máximo (TODOS los sujetos de cada dataset; MUY largo):
#   --max-subjects 0
```

`--max-subjects N` = primeros N sujetos por dataset (0 = todos). `--fs-common` cambia la fs
común (default 128). `--no-eval` omite el leave-one-dataset-out (solo entrena el base).

---

**Refs:** `bci/dsp/resampling.py` (`resample_lti`, `resample_ratio`),
`bci/dsp/fir_filters.py` (`design_lowpass_fir`), `scripts/train_eegnet_crossdataset.py`,
`docs/datasets.md` (qué datasets y por qué), `docs/entrenamiento.md` (entrenamiento por dataset).
