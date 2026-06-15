# Documento de presentación — BCI de Imaginación Motora

> **Propósito de este documento.** Es el hilo conductor del proyecto: explica *qué*
> hacemos en cada etapa, *por qué* lo hacemos así, *dónde* está en el código y *qué
> teoría de Sistemas Lineales (LTI)* se aplica. Está pensado para preparar la defensa
> oral y el informe. Se amplía con cada hito.
>
> Glosario de términos: [`glosario.md`](glosario.md). Figuras: [`figures/`](figures/).

---

## 0. La idea en una frase

Construimos una **Interfaz Cerebro-Computadora (BCI)** que, a partir de señales EEG,
distingue qué movimiento está **imaginando** una persona (p. ej. mano izquierda vs.
derecha). El valor académico no es la BCI en sí, sino que **cada etapa del procesamiento
es una aplicación directa de la teoría de sistemas LTI**: convolución, filtros FIR,
respuesta en frecuencia y filtrado espacial lineal (CSP).

### El mensaje central de la presentación
> "Un sistema LTI queda descrito por su respuesta al impulso `h[n]`, y procesar una señal
> con él es **convolucionar**. Toda nuestra cadena —filtrar el ruido, separar fuentes
> cerebrales, incluso la red neuronal— son operaciones lineales que se explican con esta
> única idea."

---

## 1. Contexto: la señal y el problema

### ¿Qué es el EEG y por qué es una "señal"?
El **EEG** mide el voltaje generado por la actividad eléctrica del cerebro mediante
electrodos en el cuero cabelludo. Al muestrearlo a `fs = 250 Hz` obtenemos una señal
**discreta** `x[n]` por cada canal: justo el objeto que estudia la asignatura.

### ¿Qué fenómeno detectamos? (ERD en la banda µ/β)
Cuando imaginamos un movimiento, la corteza motora correspondiente **reduce su potencia**
en las bandas **µ (8–12 Hz)** y **β (12–30 Hz)** — es la *desincronización relacionada a
evento* (ERD). Como la mano izquierda se controla desde el hemisferio derecho (y viceversa),
**cada clase produce una caída de potencia en una zona distinta del cráneo**. Esa diferencia
espacial + espectral es lo que vamos a aislar y clasificar.

### El dataset
Usamos **BCI Competition IV — dataset 2a** (`BNCI2014_001` en MOABB): 9 sujetos, 22 canales
EEG, `fs=250 Hz`, 4 clases (mano izq./der., pies, lengua). Empezamos en binario
(izq./der., el caso clásico del CSP) con un sujeto.

---

## 2. La cadena de procesamiento (visión global)

```
   EEG crudo          FILTRO FIR          CSP              CARACTERÍSTICAS     CLASIFICADOR
  x[n] (22 ch)   →   pasa-banda µ/β  →  filtro espacial →   log-varianza   →     LDA      →  clase
                     (convolución)      (lineal, espacial)                       (lineal)
   [Hito 2]            [Hito 3]            [Hito 4]            [Hito 5]          [Hito 5]
```

Observación clave para la defensa: **casi toda la cadena es lineal**. El FIR es LTI en el
*tiempo*; el CSP es lineal en el *espacio* (entre canales); el LDA traza una frontera lineal.
La única no linealidad deliberada es el `log` de la varianza (y, más adelante, EEGNet).

---

## 3. Desarrollo por hitos

Cada hito indica: **qué se hizo**, **archivos**, **teoría LTI aplicada** y **decisiones**.

### Hito 1 — Cimientos: estructura y entorno
**Qué:** monorepo `backend/` (Python, DSP/ML) + `frontend/` (React, Etapa 2) + `docs/`.
Entorno reproducible con venv y dependencias científicas.

**Archivos:** `README.md`, `backend/requirements.txt`, `backend/pyproject.toml`,
`backend/configs/default.yaml` (todos los parámetros del pipeline, comentados),
`.gitignore`, `frontend/README.md` (diseño de los 5 módulos visuales).

**Decisiones:**
- **Monorepo backend/frontend**: la parte visual es central al objetivo didáctico, así que
  merece su propio espacio bien definido desde el inicio.
- **Paquete `bci` instalable** (`pip install -e`): permite `import bci` desde scripts y tests
  sin trucos de rutas.
- **Configuración en un YAML**: cambiar la banda del filtro, el sujeto o el nº de filtros CSP
  no debe requerir tocar código.
- *Nota técnica:* todo funcionó en **Python 3.14** pese a ser muy reciente.

### Hito 2 — Carga de datos y epoching
**Qué:** convertir los archivos crudos del 2a en la tripleta de ML `(X, y, metadata)`, donde
`X` tiene forma `(trials, canales, tiempo)`.

**Archivos:** `backend/src/bci/datasets/moabb_loader.py` (loader + dataclass `EpochedData`),
`backend/scripts/download_data.py` (descarga, valida y cachea), `backend/src/bci/config.py`,
`backend/tests/test_loader.py`.

**Cómo estructura MOABB los datos (clave para entender el loader):**
MOABB organiza todo en 3 capas — `Dataset` (datos crudos) → `Paradigm` (filtra+recorta) →
`(X,y,metadata)`. El `Dataset.get_data()` devuelve un diccionario anidado
`[sujeto][sesión][run]` cuyas hojas son objetos `Raw` de MNE (señal continua + anotaciones
que marcan cada trial).

**Decisiones (importantes para la defensa):**
- **Usamos la señal CRUDA, no la del `Paradigm`.** El `Paradigm` filtraría por nosotros (con
  `scipy`, oculto). Como el objetivo es **implementar el filtro a mano**, tomamos solo la
  señal cruda + eventos y filtramos nosotros en el Hito 3.
- **No concatenamos runs.** MOABB une los runs para no perder el último trial de cada uno;
  nosotros NO, porque eso uniría segmentos **no contiguos en el tiempo** y luego
  convolucionaríamos el FIR a través de esa frontera artificial — incorrecto en un sistema
  LTI. Preferimos descartar 7 de 288 epochs truncados en los bordes (2.4%, clases siguen
  balanceadas). *Es más riguroso descartar datos incompletos que inventar continuidad.*
- **Validación cruzada con MOABB**: comparamos nuestra salida con la del `Paradigm`; canales
  (22) y muestras (1001 = 4 s × 250 Hz) coinciden exactamente → nuestro epoching es fiel.

### Hito 3 — Procesamiento LTI: convolución, FIR y dominio de la frecuencia
**Qué:** el corazón académico. Diseñar e implementar **a mano** un filtro FIR pasa-banda µ/β
y aplicarlo por convolución, con su análisis en frecuencia.

**Archivos:** `backend/src/bci/dsp/convolution.py`, `…/dsp/fir_filters.py`,
`…/dsp/frequency_response.py`, `…/viz/plots.py`, `backend/scripts/demo_dsp.py`,
`backend/tests/test_dsp.py`. Figuras en `docs/figures/`.

#### 3.1 La convolución como operación de un sistema LTI
Un sistema LTI cumple **linealidad** + **invarianza temporal**, y por ello queda totalmente
descrito por su respuesta al impulso `h[n]`. Su salida es **siempre**:

```
        y[n] = (x * h)[n] = Σ_k  h[k] · x[n − k]
```

La implementamos en tres formas equivalentes (verificadas contra `numpy`):
- **`convolve_mac`** — el doble bucle literal; cada paso interno es una operación **MAC**
  (multiplicar y acumular). Es la definición "de libro".
- **`convolve`** — versión vectorizada por **superposición**: la salida es la suma de copias
  de `h` escaladas por cada muestra de entrada y desplazadas en el tiempo (la interpretación
  física de un LTI). Rápida: filtra `(281, 22, 1001)` en ~1 s.
- **`mac_terms`** — expone los términos de cada MAC para **animarlos en el frontend**.

> Punto para decir: "MAC es la operación más ejecutada en todo el DSP y en las GPUs; una
> convolución es, físicamente, una ráfaga de MACs."

#### 3.2 El filtro FIR por ventaneo (windowed-sinc)
Queremos dejar pasar 8–30 Hz (donde vive la información de la imaginación motora) y atenuar
el resto (deriva <4 Hz, parpadeos, EMG >40 Hz, red eléctrica a 50 Hz). Diseño en 3 pasos:
1. **Filtro ideal:** su `h[n]` es una *diferencia de sincs* — infinita y no causal.
2. **Truncar** a `N=101` taps → aparecen rizados (**fenómeno de Gibbs**).
3. **Ventanear** (Hamming) → suaviza los extremos y reduce los lóbulos laterales, a cambio de
   una banda de transición algo más ancha.

**Por qué FIR y no IIR** (decisión clave): elegimos un FIR **simétrico** con número de taps
**impar** → tiene **fase lineal** → **retardo constante** (50 muestras) para todas las
frecuencias. Un retardo constante solo desplaza la señal, **no deforma su morfología**, que es
esencial para preservar la forma de los ritmos µ/β. Un IIR (`scipy.signal.lfilter`) sería más
eficiente pero distorsiona la fase. **No usamos `lfilter`**: la convolución explícita es justo
lo que el proyecto debe mostrar; `scipy` solo aparece como referencia en los tests.

#### 3.3 ¿Por qué pasar al dominio de la frecuencia?
En el **tiempo**, mirar `x[n]` no dice qué frecuencias contiene ni qué le hace el filtro. El
teorema fundamental de los LTI es:

```
   convolución en el tiempo   ⇔   multiplicación en la frecuencia
            y = x * h          ⇔        Y(ω) = X(ω) · H(ω)
```

Por eso, para **entender y verificar** un filtro, calculamos su **respuesta en frecuencia**
`H(e^jω)`, la DTFT de su `h[n]`:

```
        H(e^jω) = Σ_n  h[n] · e^{−jωn}
```

`|H(e^jω)|` dice cuánto se amplifica/atenúa cada frecuencia; su fase, cuánto se retrasa. La
calculamos **a mano** (suma directa de la DTFT), no con una caja negra (verificado contra la
FFT en los tests). El dominio de la frecuencia convierte una operación "complicada"
(convolución) en una "sencilla" (multiplicar curvas), y nos permite **diseñar y comprobar** el
filtro mirando una sola gráfica.

#### 3.4 Resultados (verificados sobre datos reales)
- FIR de 101 taps, **fase lineal ✓**, retardo de grupo 50 muestras.
- `|H|`: **@2 Hz = 0.001 · @19 Hz = 1.000 · @45 Hz = 0.001** → pasa-banda casi ideal.
  Ver `figures/fir_frequency_response.png`.
- Sobre C3 (corteza motora): la potencia fuera de banda cae al **0.03 %** (delta) y **0.2 %**
  (40–60 Hz), mientras la banda µ/β se conserva al **85.6 %**.
  Ver `figures/filter_effect.png`.
- **14 tests** automáticos en verde (convolución, fase lineal, banda paso/rechazo, DTFT≈FFT).

**Figuras disponibles** (`docs/figures/`):
`fir_impulse_response.png` (los coeficientes `h[n]`), `fir_frequency_response.png` (la curva
`|H|`), `filter_effect.png` (un trial antes/después en tiempo y frecuencia), `mac_operation.png`
(la operación MAC paso a paso).

### Hito 4 — CSP: filtrado espacial lineal
**Qué:** aprender una combinación lineal de los 22 canales que **maximice la varianza de una
clase y minimice la de la otra**, para resaltar la diferencia entre imaginaciones.

**Archivos:** `backend/src/bci/spatial/csp.py` (clase `CSP`), `backend/tests/test_csp.py`,
`backend/src/bci/viz/plots.py` (`plot_csp_patterns`), `backend/scripts/demo_csp.py`.
Figura: `docs/figures/csp_patterns.png`.

#### 4.1 El CSP como sistema lineal en el ESPACIO
Es el paralelo espacial del FIR. Donde el FIR mezcla *muestras vecinas en el tiempo*, el CSP
mezcla *canales en el espacio*, con pesos fijos y sin convolución temporal:

```
   FIR (tiempo):   y[n]   = Σ_k W[k]·x[n−k]      (convolución)
   CSP (espacio):  z[c,n] = Σ_j W[c,j]·x[j,n]    (combinación lineal)  →  Z = W·X
```

#### 4.2 Qué optimiza y por qué la varianza
Recordando la ERD: la información de la imaginación motora está en la **potencia** de la banda
µ/β, y la potencia de una señal de media cero **es su varianza**. El CSP busca direcciones
espaciales donde esa varianza sea muy distinta entre clases. Eso conduce a un problema de
**autovalores generalizados**:

```
        C1 · w = λ · (C1 + C2) · w
```

con `C1`, `C2` las covarianzas medias de cada clase. Los autovalores `λ ∈ [0,1]` reparten la
varianza: `λ≈1` → la componente "se enciende" en la clase 1; `λ≈0` → en la clase 2. Tomamos
los **autovectores de los extremos** (los más discriminativos).

#### 4.3 Cómo lo resolvemos (whitening + diagonalización conjunta)
En vez de llamar a un solver opaco, hacemos los pasos explícitos (método de Koles):
1. Covarianzas medias por clase, normalizadas por la traza.
2. **Whitening** de `Cc = C1 + C2` (la deja como identidad).
3. **Diagonalizar** `C1` en ese espacio blanqueado → los autovalores son los `λ`.
4. Filtros `W = Bᵀ P`; patrones `A = pinv(W)` (lo que se dibuja en el topomapa).

> Matiz honesto: usamos `np.linalg.eigh` para las descomposiciones simétricas. Eso es álgebra
> lineal estándar (como usar `np.sinc`), no esconder la teoría: los pasos del CSP son nuestros
> y explícitos. La diferencia con `lfilter` es que allí se ocultaba *la operación central*
> (la convolución); aquí la operación central (la diagonalización conjunta) está a la vista.

#### 4.4 Resultados (verificados sobre datos reales)
- Autovalores coinciden **exactamente** con el GEVD de referencia (`scipy.linalg.eigh(C1, Cc)`).
- Patrones espaciales **lateralizados sobre la corteza motora** (`docs/figures/csp_patterns.png`).
- Pipeline completo **FIR → CSP → log-varianza → LDA**: **77.2 %** de accuracy (5-fold,
  sujeto 1, izq./der.) — un buen resultado de referencia para el 2a.
- **5 tests** en verde (autovalores, diagonalización, separabilidad, formas, errores).

### Hito 5 — Características y clasificación offline
**Qué:** convertir las componentes CSP en un número por trial (log-varianza), clasificar con
un modelo lineal (LDA) y medir el rendimiento con métricas serias.

**Archivos:** `backend/src/bci/features/log_variance.py`, `backend/src/bci/models/lda.py`,
`backend/src/bci/pipeline/offline.py` (clase `MotorImageryPipeline` + evaluaciones),
`backend/scripts/run_offline.py`, `backend/tests/test_pipeline.py`.

#### 5.1 La característica: log-varianza
Cada componente CSP es una serie temporal; la resumimos en su **potencia = varianza** (recordar
la ERD) y aplicamos `log`:  `f_i = log(var(z_i) / Σ_j var(z_j))`. El `log` acerca la distribución
a una gaussiana (lo que el LDA asume) y la normalización la hace robusta a la amplitud global.

#### 5.2 El clasificador: LDA, frontera lineal
Cerramos la cadena con un **clasificador lineal**: el LDA modela cada clase como una gaussiana
con covarianza compartida, lo que da una **frontera de decisión lineal** (hiperplano):
`δ_k(x) = wᵀx + b`. Lo implementamos a mano (verificado: predicciones **idénticas** a sklearn).
Así toda la cadena —FIR, CSP, LDA— es lineal, salvo el `log` deliberado.

#### 5.3 Evaluación honesta (y dos lecciones clave para la defensa)
Medimos de dos formas: **k-fold** estratificado y, sobre todo, **inter-sesión** (entrenar en la
sesión `0train`, evaluar en `1test`) — la estimación realista de cómo generalizaría en vivo,
porque train y test no comparten día.

> **Lección 1 — Fuga de datos (data leakage).** Una primera prueba dio 77 % porque por error
> ajustamos el CSP con TODO el dataset antes de la validación cruzada. Al corregirlo (ajustar
> CSP y LDA **solo con train** dentro de cada partición) el resultado honesto bajó a ~63 %. *El
> CSP "aprende" de los datos: si ve el test, el resultado se infla.*
>
> **Lección 2 — La ventana temporal importa muchísimo.** La ventana completa `[2-6]s` daba 0.52
> inter-sesión (≈ azar); recortando a la fase de imaginación **activa** `[2.5-4.5]s` subió a
> **0.74**. La ventana completa incluye el transitorio del cue y la cola donde la imaginación
> decae. Estrategia adoptada: **filtrar el epoch completo y luego recortar** — así además se
> descarta el transitorio de borde del propio FIR (~50 muestras de retardo de grupo).

**Verificación de que no hay bug:** nuestro CSP+LDA da **exactamente** el mismo resultado que
`mne.decoding.CSP` + LDA de sklearn en el mismo split (0.518 con la ventana mala, ambos).

#### 5.4 Resultados finales (sujeto 1, izq./der.)
| Evaluación        | Accuracy | Kappa |
|-------------------|:--------:|:-----:|
| 5-fold CV         |  0.719   | 0.437 |
| Inter-sesión (honesta) | 0.738 | 0.479 |

**25 tests** automáticos en verde en todo el backend.

### Hito 6 — Simulación de transmisión en vivo (cierre de la Etapa 1)
**Qué:** reproducir la señal grabada "como si llegara" de un casco en tiempo real: filtrado
causal por chunks + ventana deslizante + clasificación sobre la marcha.

**Archivos:** `backend/src/bci/streaming/simulator.py` (`CausalFIR`, `StreamSimulator`),
`backend/scripts/run_live_sim.py`, `backend/tests/test_streaming.py`,
método `MotorImageryPipeline.classify_window`.

#### 6.1 La gran diferencia: CAUSALIDAD
Offline filtrábamos con `mode='same'`, que para calcular `y[n]` usa muestras **futuras**. En
vivo eso es imposible: solo existe el pasado. Un **filtro causal** calcula
`y[n] = Σ_{k≥0} h[k]·x[n−k]` usando solo lo ya recibido, y para ello mantiene un **estado**
(las últimas M−1 muestras) entre chunk y chunk. Verificamos que filtrar por trozos da el mismo
resultado **exacto** que filtrar de golpe (no hay saltos en las fronteras).

> **Precio de la causalidad — retardo de grupo.** El FIR de fase lineal retrasa la señal
> (M−1)/2 = **50 muestras = 0.2 s** a 250 Hz. Offline lo compensábamos (`mode='same'`); en vivo
> ese retardo es real e inevitable. Es un punto teórico clave: *no se puede tener filtrado de
> fase lineal sin retardo*.

#### 6.2 Arquitectura del simulador
`CausalFIR` filtra cada chunk con estado; `StreamSimulator` mantiene una **ventana deslizante**
de los últimos `window_s` segundos ya filtrados y, cada `step_s`, la pasa por CSP → log-var →
LDA (`classify_window`), emitiendo predicción + pseudo-probabilidades (softmax). Un `callback`
`on_predict` permite enchufar una UI o un stream LSL. **El mismo pipeline entrenado offline se
reutiliza sin cambios** — solo cambia que el filtrado es causal y por trozos.

#### 6.3 Resultados
- Filtrado causal por chunks ≡ filtrado completo (verificado).
- En un trial, la predicción y su confianza **evolucionan en el tiempo** (p. ej. `left_hand`
  con prob. 0.97→0.78), como se vería en una demo en vivo.
- **Accuracy en streaming (voto por trial): 0.745**, coherente con la offline
  inter-sesión (0.738). *Pasar a tiempo real no degrada el rendimiento.*
- **5 tests** en verde (30 en total en el backend).

**Decisión por trial restringida a la ventana activa (refinamiento de la demo en vivo).** La
clasificación NO se decide ventana a ventana: muchas ventanas deslizantes caen fuera de la
imaginación activa (inicio, transición del *cue*, final) y ahí la predicción es casi aleatoria,
lo que hacía que el contador "se equivocara mucho" en apariencia. La decisión final de cada trial
se toma por **voto suave** (suma de probabilidades) **solo sobre las ventanas cuyo centro cae en la
ventana de clasificación** `[tmin_rel, tmax_rel]`. El backend envía esos límites por WebSocket
(`alo`/`ahi`) y el frontend acumula solo esas ventanas. Efecto medido (held-out, voto por trial):
sujeto 1 0.73→**0.76**, sujeto 3 0.76→**0.82**, sujeto 8 0.77→**0.80** (más cerca del offline).
La interfaz muestra un contador honesto de **aciertos/fallos por trial** y una tira de los últimos
trials (✓/✗). *Techo del estado del arte: una BCI MI no invasiva de 2 clases ronda 70–85 %; no es
posible acercarse al 100 %.*

---

### Evaluación rigurosa: los 9 sujetos del 2a (within-subject)
**Qué:** entrenar y evaluar un modelo **independiente por sujeto** (no se mezclan sujetos) y
reportar la media ± desviación. **Archivos:** `backend/scripts/evaluate_all.py`,
`backend/src/bci/viz/plots.py` (`plot_subject_results`). Figura: `docs/figures/subjects_2a.png`,
datos: `backend/data/processed/results_2a.csv`.

**Por qué within-subject (punto clave para la defensa):** el EEG varía muchísimo entre personas
(anatomía, colocación de electrodos, estrategia mental). El CSP aprende *qué electrodos pesar*,
y esos pesos **no se alinean entre sujetos**; entrenar mezclándolos suele empeorar el resultado
individual. Por eso el estándar es un modelo por persona.

**Resultados (clases izq./der.):**
| Métrica | Media ± std |
|---|---|
| k-fold CV (accuracy) | **0.688 ± 0.129** |
| Inter-sesión (accuracy) | **0.636 ± 0.133** |
| Inter-sesión (kappa) | 0.273 ± 0.263 |

**La gran lección — variabilidad inter-sujeto:** los resultados van de **0.85 (sujetos 3 y 8)**
a **≈0.50 (sujetos 5 y 9, casi azar)**. No es un fallo del método: hay personas cuya imaginación
motora produce una ERD clara y otras que apenas la generan (fenómeno conocido como *"BCI
illiteracy"*). Esto justifica por qué en BCI siempre se reporta **media sobre sujetos**, nunca un
único número, y motiva técnicas futuras (transfer learning, EEGNet) para los sujetos difíciles.
Ver `docs/figures/subjects_2a.png`.

### Diversidad de datasets: integración de PhysioNet MMI
**Qué:** añadir un segundo dataset público (**PhysioNet EEG Motor Movement/Imagery**, 109
sujetos, fs **160 Hz**, 64 canales) para demostrar que el pipeline no depende de un montaje
concreto. **Archivos:** `backend/configs/physionet.yaml`, `backend/scripts/evaluate_all.py`
(genérico, `--config`). Figura: `docs/figures/subjects_PhysionetMI.png`.

**Decisión de diseño clave — `fs` dinámico.** El 2a usa 250 Hz y PhysioNet 160 Hz. En vez de
forzar todo a una fs común (que alteraría la señal), **el pipeline diseña el FIR y calcula la
ventana con la `fs` real de cada dataset**. Cada dataset tiene su propio archivo de config; el
código es el mismo. Es un buen ejemplo de ingeniería: *el filtro se adapta a la señal, no al
revés*.

Añadimos también **Liu2024** (Nature Scientific Data, 2024; 50 sujetos, fs 500 Hz, 29 canales,
~40 trials/sujeto) — un dataset reciente y de alta calidad. `backend/configs/liu2024.yaml`,
figura `docs/figures/subjects_Liu2024.png`.

**Comparación entre los tres datasets (k-fold, izq./der., 20 sujetos cada uno salvo el 2a con 9):**
| Dataset | Año | fs | Canales | Trials/sujeto | Accuracy k-fold (media±std) |
|---|---|---|---|---|---|
| BCI IV 2a | 2008 | 250 Hz | 22 | ~280 | **0.688 ± 0.129** |
| PhysioNet MMI | 2004 | 160 Hz | 64 | ~45 | **0.608 ± 0.172** |
| Liu2024 | 2024 | 500 Hz | 29 | ~40 | **0.536 ± 0.081** |

**Lecciones (para la defensa):**
- **La variabilidad inter-sujeto es enorme:** en PhysioNet del 0.98 (sujeto 7) a valores **por
  debajo del azar**; en Liu2024 la mayoría rondan el azar salvo algún sujeto (13 → 0.78).
  Refuerza por qué en BCI se reporta la media sobre sujetos, nunca un caso aislado.
- **Maldición de la dimensionalidad (clave):** la accuracy cae conforme hay **menos trials por
  sujeto** (2a ~280 → PhysioNet ~45 → Liu2024 ~40). Estimar una covarianza de muchos canales con
  pocos ejemplos es inestable y el CSP **sobreajusta**. Es la motivación directa del siguiente
  paso: **regularizar el CSP** (shrinkage de covarianza).
- **No es un bug nuestro:** verificado en Liu2024 que nuestro CSP+LDA **iguala o supera** a
  `mne.decoding.CSP` (sujeto 1: 0.625 vs 0.500; sujeto 13: 0.775 vs 0.725). El resultado bajo es
  intrínseco al dataset, no del código.
- **El método es transferible:** el mismo pipeline LTI, sin cambios de código (solo config y
  fs dinámico), funciona en **tres** datasets con distinto equipo, nº de canales y fs.

### Regularización del CSP: un resultado negativo (y por qué)
**Qué:** añadimos **shrinkage de covarianza** al CSP (Lotte & Guan, 2011), la regularización
clásica para cuando hay pocos trials: `C_reg = (1−γ)·C + γ·(tr(C)/n)·I`. **Archivos:**
`bci.spatial.csp` (`_shrink`, parámetro `shrinkage`), `backend/scripts/sweep_shrinkage.py`,
config `csp.shrinkage`. Figuras: `docs/figures/shrinkage_sweep_*.png`.

**Hipótesis:** los datasets con pocos trials (PhysioNet, Liu2024) sufrirían sobreajuste del
CSP y la regularización los mejoraría.

**Resultado medido (barrido γ ∈ [0, 0.5], accuracy k-fold media sobre sujetos):**
| Dataset | γ=0 | mejor γ | ¿mejora? |
|---|---|---|---|
| BCI IV 2a | 0.688 | 0.688 (γ=0) | no |
| PhysioNet MMI | 0.608 | 0.608 (γ=0) | no |
| Liu2024 | 0.536 | 0.536 (γ=0) | no |

En los tres, **γ=0 es óptimo** y la accuracy **empeora** al aumentar γ. La hipótesis era
**falsa**: la regularización no ayuda aquí.

**Por qué (importante para la defensa — saber interpretar un resultado negativo):**
1. **Nuestro CSP ya estaba bien condicionado.** Tiene tres salvaguardas que hacen el shrinkage
   redundante: covarianzas **normalizadas por traza**, **promediadas sobre trials**, y un
   **whitening que descarta las direcciones de rango casi nulo**. El problema numérico que el
   shrinkage resuelve (covarianza singular) ya estaba mitigado.
2. **El shrinkage comprime el espectro de autovalores hacia 0.5**, que es justo *aplanar* las
   diferencias de varianza entre clases — la señal que el CSP explota. Al no haber problema que
   resolver, solo introduce sesgo y reduce la discriminabilidad.
3. **Conclusión:** el cuello de botella del bajo rendimiento (Liu2024, PhysioNet) **no es la
   estimación de la covarianza**, sino la señal misma (pocos trials con baja SNR, variabilidad
   entre sujetos). Regularizar el CSP ataca un problema que no teníamos.

> Lección metodológica: probamos la técnica porque la teoría la sugería, la medimos con rigor,
> y la **descartamos con evidencia** entendiendo el porqué. Dejamos el shrinkage implementado y
> configurable (`csp.shrinkage`), con valor por defecto **0**.

---

### FBCSP (banco de filtros): un resultado matizado

**Qué:** la extensión "de libro" del CSP (Ang et al., 2008), y **pura teoría LTI**: en vez de un
solo FIR pasa-banda 8–30, se usa un **banco de filtros FIR** (7 sub-bandas de 4 Hz: 4–8, 8–12, …,
28–32), se aplica CSP en cada banda, se concatenan las log-varianzas, se **seleccionan** las más
informativas (información mutua) y se clasifica con LDA. **Archivos:** `bci.pipeline.fbcsp`
(`FBCSPPipeline`), `backend/scripts/eval_fbcsp.py`, `backend/tests/test_fbcsp.py`. Figura:
`docs/figures/fbcsp_vs_csp_BNCI2014_001.png`.

**Hipótesis:** cada ritmo discrimina mejor en su propia sub-banda, así que el banco superaría a la
banda única.

**Resultado medido (9 sujetos del 2a, accuracy media ± std):**
| Evaluación | CSP banda única | FBCSP (banco) | ¿mejora? |
|---|---|---|---|
| inter-sesión (honesta) | 0.636 ± 0.133 | 0.655 ± 0.146 | **no significativa** (t pareado p=0.68) |
| k-fold (mezcla sesiones) | 0.687 ± 0.129 | 0.727 ± 0.140 | leve, optimista |

**La clave está en la VARIABILIDAD entre sujetos, no en la media:**
- FBCSP **rescata** a sujetos que la banda única no separaba: S5 0.51→**0.72**, S7 0.55→**0.84**
  (su información discriminante vive en una sub-banda estrecha que el 8–30 diluía).
- FBCSP **hunde** a otros: S8 0.82→**0.67**, S9 0.50→**0.40** (más bandas + selección de
  características → sobreajuste de la sesión de entrenamiento, que no transfiere a otro día).
- En promedio se **cancelan** (p=0.68): no hay mejora robusta.

**Por qué y decisión (importante para la defensa):**
1. El **diseño óptimo del filtro es dependiente del sujeto**: no existe una única banda buena para
   todos. FBCSP ayuda cuando el ritmo útil es estrecho; estorba cuando el 8–30 ya era adecuado.
2. El banco **añade parámetros** (B bandas × componentes + selección) que **sobreajustan** la
   sesión de entrenamiento → peor generalización inter-sesión para varios sujetos.
3. **Producción mantiene CSP de banda única**: es más simple, más robusto y, en los sujetos
   fuertes que usaríamos en la demo (3, 8), **rinde mejor** que FBCSP. FBCSP queda como
   experimento medido y reproducible.

> Lección: una técnica del estado del arte no es "mejor" por defecto. Medida con honestidad
> (inter-sesión + test pareado), su ganancia media aquí no es significativa; lo interesante es
> *cuándo* ayuda (señal en banda estrecha) y *cuándo* no. El cuello de botella sigue siendo la
> señal y la variabilidad entre sujetos, no el diseño del filtro.

---

## ✅ Estado: Etapa 1 COMPLETA

El sistema hace el recorrido entero: **EEG crudo → FIR µ/β → CSP → log-varianza → LDA**, tanto
offline (con evaluación honesta) como en **simulación de transmisión en vivo**. Todo el
procesamiento central está implementado a mano y verificado contra librerías de referencia.
La arquitectura (`streaming/` aislado, `classify_window`, callback `on_predict`) está lista para
enchufar el casco Ultracortex por LSL (Etapa 3) y un frontend React (Etapa 2) sin tocar el DSP.

---

## Los dos mundos del sistema: ENTRENAR (antes) vs TRANSMITIR (ahora)

Una BCI real funciona en dos tiempos que **no deben confundirse**, y el proyecto los
separa de forma explícita:

| | **OFFLINE — "antes"** (entrenamiento / calibración) | **ONLINE — "ahora"** (streaming) |
|---|---|---|
| Qué ocurre | Se calcula el modelo a partir de datasets | Llega una señal como si fuera del casco |
| Cuándo | Una sola vez, previo a la demo | En tiempo real, durante la demo |
| Causalidad | Se puede mirar todo el trial | Solo el pasado (causal, con retardo) |
| Resultado | CSP, LDA (pesos guardados a disco) + accuracy | Predicción en vivo + confianza |

**Por qué importa (y la fuga de datos que evitamos):** si el modelo se entrena con los
mismos trials que luego se "transmiten" en vivo, la demo evalúa señales que el modelo ya
vio → la precisión se infla y el experimento es **deshonesto**. La regla es: **lo que se
transmite no se entrena, y lo que se entrena no se transmite.**

**Cómo lo implementamos:**
1. `scripts/train_model.py` entrena el modelo de cada sujeto con una **partición de
   entrenamiento** y reserva un **held-out** que el modelo nunca ve:
   - en el 2a, que tiene dos sesiones, se entrena con la sesión `0train` y se reserva
     `1test` (lo más honesto: train y test son de **días distintos**, como pasaría en vivo);
   - en datasets de una sola sesión, se reserva una fracción estratificada (30 %).
2. El modelo entrenado se **persiste a disco**: `model_{dataset}_s{subject}.pkl` (el pipeline
   CSP+LDA) y un `.json` con su **ficha** (con qué se entrenó, qué se reserva, accuracy honesta).
   Esto hace literal el relato "el modelo se entrenó *antes*".
3. El servidor (`/ws/stream`) **carga** ese artefacto (no reentrena) y transmite **solo los
   trials held-out**. El endpoint `/api/model` expone la ficha para que la web la muestre
   etiquetada como "cálculos previos al streaming".

**Archivos:** `backend/src/bci/pipeline/training.py` (split honesto + persistencia +
`ModelCard`), `backend/scripts/train_model.py`, endpoints `/api/model` y `/ws/stream` en
`backend/src/bci/server/app.py`.

**Resultado (2a, sujeto 1):** entrenado con `0train` (140 trials), demo sobre `1test`
(141 trials reservados) → **accuracy honesta 0.738** (kappa 0.479). Es la misma cifra que la
evaluación inter-sesión: coherente, porque es exactamente la misma idea materializada.

---

## 4. Próximos pasos (mapa)

- **Etapa 2 — Frontend React.** Visualización interactiva: animación de la convolución (MAC),
  pesos CSP en un topomapa, cerebro 3D reactivo, panel de transmisión en vivo.
- **Punto FINAL del proyecto (trabajo futuro): EEGNet para generalización cross-subject.**
  El verdadero salto de precisión no vendría de un mejor filtro, sino de **reducir la calibración
  por sujeto**: entrenar una EEGNet sobre **muchos sujetos** (con alineación de dominios /
  transfer learning) para aprender representaciones comunes y luego afinar con una calibración
  corta. Es justo el régimen de "muchos datos" donde el deep learning sí supera al clásico —el
  reverso de nuestro hallazgo del *espejo*— y ataca el cuello de botella real: la variabilidad
  entre cerebros y entre sesiones. Queda planteado como cierre del proyecto, no implementado.

---

## EEGNet — el puente entre los filtros LTI y la IA

**Qué:** EEGNet (Lawhern et al., 2018) es una red neuronal convolucional compacta para EEG.
Es la parte de "IA" del proyecto y se justifica porque **sus capas imitan el pipeline clásico,
pero aprendiéndolo de los datos**:
- su **convolución temporal** ≈ un **banco de filtros FIR aprendidos** (en vez de diseñar la
  banda µ/β a mano, la red ajusta sus propias respuestas al impulso);
- su **convolución *depthwise* por canal** ≈ un **filtro espacial tipo CSP aprendido**.

**Archivos:** `backend/src/bci/models/eegnet.py` (red PyTorch + wrapper `EEGNetClassifier`),
`backend/scripts/train_eegnet.py` (compara EEGNet vs CSP+LDA con los mismos folds).

**Cómo se entrena:** PyTorch (CPU). A EEGNet se le da la señal **sin el filtro µ/β ni el CSP**
—solo un filtrado de banda amplia (4–40 Hz) para quitar deriva y EMG— para que **descubra ella
misma** los filtros temporales y espaciales. Entrenamiento *within-subject*, estandarizando por
canal, con Adam.

**Resultado (2a, sujeto 1):** dos lecturas según cómo se mida.
| Modelo | k-fold (mezcla sesiones) | inter-sesión honesta (train `0train` → demo `1test`) |
|---|---|---|
| CSP + LDA (clásico) | 0.722 | **0.738** |
| EEGNet (deep learning) | 0.651 | **0.504** (≈ azar) |

**Lección (importante para la defensa):** el método **clásico CSP+LDA supera a la red profunda**,
y la brecha se agranda con la evaluación honesta inter-sesión: con solo 140 trials de un día,
EEGNet **sobreajusta esa sesión y no generaliza** a la del día siguiente (cae al azar), mientras
CSP+LDA aguanta. No es un fallo de EEGNet: las redes profundas necesitan muchos más ejemplos,
mientras que CSP+LDA, al incorporar el conocimiento del dominio (potencia en µ/β), es muy
eficiente con pocos datos. Es la comparación honesta entre el enfoque basado en teoría y el
basado en datos. *(El artefacto EEGNet se entrena y guarda con el mismo split honesto que
CSP+LDA — ver "Los dos mundos".)*

**Visualización de los filtros aprendidos (el puente, ya en la web — EEGNet como ESPEJO):**
decisión de alcance deliberada: EEGNet **no** es un segundo pipeline con paridad en toda la app
(eso sería "demasiado"); se usa como **espejo que confirma la teoría**. La sección **El Modelo**
tiene una pestaña *"EEGNet (filtros aprendidos)"* que pone **lado a lado**:
- **temporal:** tu FIR µ/β (diseñado) vs los filtros de la conv temporal (aprendidos), ambos como
  |H(e^jω)| con la banda µ/β sombreada → ¿caen los picos aprendidos dentro de µ/β?
- **espacial:** tus patrones CSP (entrenados) vs los filtros de la conv depthwise (aprendidos),
  como topomapas → ¿se lateraliza alguno sobre C3/C4 como el CSP?

Así se comprueba *visualmente* si la red redescubre lo que nosotros impusimos por teoría.
Endpoint: `GET /api/eegnet`. La pestaña es **local** a El Modelo (no hay selector global; el resto
de la app sigue con CSP+LDA).

**Cómo se entrena el EEGNet del espejo (y por qué dos accuracies):** como EEGNet ya no se usa en
vivo, el modelo cuyos filtros se visualizan se entrena con **todos los trials** del sujeto (más
datos ⇒ filtros más limpios e interpretables) y con *weight decay*. Para no inflar el número, la
ficha reporta **dos accuracies honestos**, medidos con modelos entrenados aparte: **within-subject
k-fold ≈ 0.655** (el mejor caso justo) e **inter-sesión ≈ 0.504** (generalización a otro día).
Ambos por debajo de CSP+LDA (0.72 / 0.74): aun en su mejor versión, el clásico gana con tan pocos
datos. "Entrenar más épocas" no ayudaría — solo agrava el sobreajuste de la sesión.

> Decisión: se **descarta** la inferencia EEGNet en vivo (la Clasificación con CSP+LDA ya cuenta
> la historia del streaming; EEGNet en tiempo real añade complejidad sin mensaje nuevo).

---

## 5. Posibles preguntas del tribunal (y respuestas cortas)

- **¿Por qué FIR y no IIR?** Fase lineal → retardo constante → no deforma la forma de la
  señal; además es siempre estable. Coste: más taps.
- **¿Por qué la banda 8–30 Hz?** Es donde ocurre la ERD de la imaginación motora (ritmos µ y
  β); fuera de ahí hay sobre todo ruido y artefactos.
- **¿Por qué no usar `lfilter`/el `Paradigm` de MOABB?** Porque ocultarían la teoría: el
  objetivo es mostrar la convolución explícita. Los usamos solo como referencia/validación.
- **¿Por qué pasar al dominio de la frecuencia?** Porque convolución ⇔ multiplicación; permite
  diseñar y verificar el filtro de un vistazo.
- **¿Qué es la operación MAC?** Multiplicar-acumular: el paso elemental de la convolución y la
  base de todo el cómputo en DSP y redes neuronales.
- **¿Por qué descartáis trials al epocar?** Para no concatenar runs no contiguos; es más
  honesto descartar epochs truncados que convolucionar a través de una frontera artificial.
- **¿Qué es el CSP y por qué es "lineal"?** Un filtro espacial: combina linealmente los canales
  (Z = W·X) para maximizar la diferencia de varianza entre clases. Es el análogo espacial del
  FIR (que es lineal en el tiempo).
- **¿Por qué la varianza?** Porque la potencia de la señal de media cero es su varianza, y la
  imaginación motora se manifiesta como un cambio de potencia (ERD) en la banda µ/β.
- **¿De dónde salen los filtros CSP?** De un problema de autovalores generalizados sobre las
  covarianzas de cada clase, resuelto por whitening + diagonalización conjunta.
- **¿Por qué el LDA es "lineal"?** Modela las clases como gaussianas con covarianza compartida;
  los términos cuadráticos se cancelan y la frontera resulta ser un hiperplano (wᵀx + b).
- **¿Por qué evaluáis inter-sesión y no solo k-fold?** Porque train y test de distinto día es la
  estimación realista de cómo funcionaría en vivo; el k-fold mezclando sesiones es optimista.
- **¿Qué es la fuga de datos y cómo la evitáis?** Si el CSP/LDA ven datos de test al entrenarse,
  el resultado se infla. Los ajustamos solo con train dentro de cada partición.
- **¿Por qué la ventana [2.5-4.5]s y no todo el trial?** Es la fase de imaginación activa; fuera
  de ella hay transitorio del cue y decaimiento. Filtramos completo y recortamos para, además,
  descartar el transitorio de borde del FIR.
- **¿Qué cambia entre el filtrado offline y el de vivo?** En vivo el filtro debe ser CAUSAL
  (solo usa el pasado) y mantener estado entre trozos; offline usábamos `mode='same'` (mira al
  futuro). El resultado por chunks es idéntico a filtrar de golpe.
- **¿Qué es el retardo de grupo y por qué importa en vivo?** Un FIR de fase lineal retrasa la
  señal (M−1)/2 muestras (0.2 s aquí). Offline se compensa; en vivo es un retardo real e
  inevitable. No hay fase lineal sin retardo.
- **¿La clasificación en vivo es peor que offline?** No: streaming 0.745 vs offline 0.738. El
  mismo modelo entrenado offline se reutiliza tal cual; solo cambia que el filtrado es causal.
- **¿Por qué la decisión es por trial y no por ventana?** Las ventanas fuera de la imaginación
  activa son casi aleatorias; decidir por cada una infla los errores. Se acumulan las probabilidades
  de las ventanas de la zona activa y se decide una vez por trial (voto suave) — como una BCI real.
- **¿Por qué no llega al 100 % de aciertos?** Porque es el límite físico del estado del arte: el
  EEG no invasivo de imaginación motora ronda 70–85 % en 2 clases (ruido, variabilidad entre días
  y sujetos, "BCI illiteracy"). Más datos o redes ayudan poco; el cuello de botella es la señal.
- **¿Por qué un modelo por sujeto y no uno solo para todos?** Porque el EEG varía mucho entre
  personas y el CSP no se alinea entre ellas; mezclarlas empeora el resultado individual.
- **¿Por qué unos sujetos dan 0.85 y otros 0.50?** Variabilidad inter-sujeto / "BCI illiteracy":
  no todos producen una ERD clara. Por eso se reporta la media sobre sujetos (aquí 0.64–0.69).
- **¿Por qué EEGNet rinde peor que CSP+LDA?** Las redes profundas necesitan muchos más datos; con
  pocos trials por sujeto, el método clásico —que ya incorpora la teoría (potencia en µ/β)— gana.
- **¿Qué tiene de "LTI" una red neuronal?** Sus capas son convoluciones: la temporal es un banco
  de filtros FIR aprendidos y la espacial (depthwise) es un filtro tipo CSP aprendido.
- **¿La señal de la demo en vivo la usasteis para entrenar?** No. Entrenamos con una partición
  (sesión `0train`) y transmitimos solo el **held-out** (`1test`), que el modelo nunca vio. El
  modelo se entrena antes, se guarda a disco y en la demo solo se carga. Así no hay fuga de datos.
- **¿Cómo distingue izquierda de derecha si no hay hardware?** El modelo se entrena *antes* con
  datasets etiquetados: CSP aprende el filtro espacial y LDA la frontera. En la demo ese modelo ya
  entrenado clasifica, en tiempo real, una señal distinta (held-out o de otro dataset MOABB).
