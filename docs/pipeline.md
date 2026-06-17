# El pipeline LTI — de la señal cruda a la decisión

> Documento de referencia técnica. Explica, paso a paso y con las fórmulas exactas que usa
> el código, cómo viaja la señal desde el dataset hasta la predicción, tanto en el mundo
> **offline** (entrenamiento) como en el mundo **online** (transmisión en vivo). Para la
> narrativa de defensa y los resultados ya redactados, ver [`presentacion.md`](presentacion.md);
> para definiciones sueltas, [`glosario.md`](glosario.md). Aquí el foco es *cómo funciona el
> programa*: qué archivo hace qué, con qué datos, y qué fórmula calcula cada coeficiente.

---

## 0. Mapa general

```
            MUNDO OFFLINE (antes, una sola vez)                         MUNDO ONLINE (ahora, en vivo)
            ────────────────────────────────────                       ──────────────────────────────
MOABB  ──►  EpochedData (X,y,metadata)  ──►  split train/demo
                                                  │
                                          ┌───────┴────────┐
                                          │ idx_train       │ idx_demo (held-out)
                                          ▼                 │
                                  FIR (mode='same')          │
                                       │                     │
                                       ▼                     │
                                  CSP.fit(Xf,y) → W           │
                                       │                     │
                                       ▼                     │
                            log-varianza → F                 │
                                       │                     │
                                       ▼                     │
                                  LDA.fit(F,y)                │
                                       │                     │
                                       ▼                     ▼
                          modelo .pkl + ficha .json   StreamSimulator.stream(X_demo)
                          (CSP+LDA congelados)              │
                                                       CausalFIR (con estado)
                                                              │
                                                      ventana deslizante (window_s, step_s)
                                                              │
                                                      classify_window():
                                                        Z = W·X  (CSP ya entrenado)
                                                        log-varianza
                                                        LDA.decision_function (ya entrenado)
                                                              │
                                                              ▼
                                                 pred + probs + feat + disc  (por ventana)
                                                              │
                                                  voto suave SOLO en ventana activa → decisión por trial
```

La idea que atraviesa todo el documento: **el CSP y el LDA se calculan una sola vez, offline,
con datos de entrenamiento, y en vivo solo se *aplican*** (multiplicar por `W`, evaluar
`δ_k(x)`). Lo único que cambia entre offline y online es *cómo se filtra* (con `mode='same'`
vs. causal con estado) y *qué datos ve* el modelo (entrenamiento vs. held-out nunca visto).

---

## 1. Mundo OFFLINE — entrenamiento

### 1.1 Carga de datos: MOABB → `EpochedData`

**Archivo:** `backend/src/bci/datasets/moabb_loader.py` (función `load_dataset` / `load_from_config`).

> Nota sobre el nombre: el paquete `backend/src/bci/datasets/downloaders/` existe como
> carpeta reservada para descargadores **propios** de datasets que no estén en MOABB (el
> objetivo declarado en `Instrucciones.txt` es no limitarse a MOABB). Hoy está vacía: los
> tres datasets soportados (`BNCI2014_001`, `PhysionetMI`, `Liu2024`) se cargan **todos** vía
> MOABB, así que el loader real es `moabb_loader.py`, no un "downloader" separado.

**Cómo organiza MOABB los datos.** `Dataset.get_data(subjects=[...])` devuelve un diccionario
anidado:

```
data[sujeto][sesión][run]  →  mne.io.Raw   (señal continua + anotaciones de eventos)
```

El loader recorre esa estructura y, **por cada run**, hace:

1. **`raw.pick(picks)`** — se queda solo con los canales EEG (`picks="eeg"` descarta EOG/STIM).
   Para BCI IV 2a quedan 22 canales.
2. **`mne.events_from_annotations(raw)`** — convierte las anotaciones del archivo crudo
   (p. ej. `"left_hand"`, `"right_hand"`) en una matriz de eventos `(n_eventos, 3)` y un
   diccionario `event_id` (nombre → código entero).
3. **Filtra por clase** si `cfg["dataset"]["classes"]` las restringe (en este proyecto, binario:
   `["left_hand", "right_hand"]`).
4. **`mne.Epochs(raw, events, event_id, tmin, tmax, baseline, preload=True)`** — recorta, para
   cada evento, la ventana `[tmin, tmax]` segundos **relativa al onset** de la anotación (el
   inicio del trial, no del cue de imaginación). Para el 2a, `tmin=2.0, tmax=6.0`
   (`configs/default.yaml`): el cue de imaginación aparece a los 2s y dura hasta los 6s del
   trial. `preload=True` materializa el resultado como un array `numpy` en memoria.
   `baseline=None` → **no se resta ninguna línea base**: la señal queda tal cual la mide el
   casco (cruda, sin filtrar, sin centrar).

**No se concatenan runs.** Cada run se epoca por separado; si se uniesen las señales continuas
de dos runs distintos antes de filtrar, el FIR convolucionaría a través de una frontera de
tiempo artificial (dos grabaciones no contiguas). El precio es perder los ~7 de 288 trials
truncados en el borde de cada run (2.4 %), una decisión deliberada de rigor LTI.

**Resultado: la dataclass `EpochedData`** (mismo archivo):

| Campo | Forma / tipo | Significado |
|---|---|---|
| `X` | `(n_trials, n_canales, n_muestras)` `float` | Señal **cruda** (sin filtrar), en **voltios** (la unidad nativa de MNE). |
| `y` | `(n_trials,)` `str` | Etiqueta de clase de cada trial (`"left_hand"`, `"right_hand"`, …). |
| `metadata` | `DataFrame` con columnas `subject`, `session`, `run` | Una fila por trial; **`session` es la clave de la validación honesta** (ver §1.7). |
| `ch_names` | `list[str]` | Nombres de canal, en el mismo orden que el eje 1 de `X` (p. ej. `C3` es el índice que usa el resto del pipeline como canal de referencia). |
| `sfreq` | `float` | Frecuencia de muestreo **real** del dataset (250 Hz en 2a, 160 Hz en PhysioNet, 500 Hz en Liu2024). Se propaga a todo el pipeline como `fs` — es la pieza que permite que el mismo código sirva para datasets distintos ("fs dinámico"). |

`n_trials`, `n_channels`, `n_times` y `class_distribution()` son propiedades derivadas de
`X`/`y` para inspección rápida (usadas por `scripts/download_data.py` y los endpoints
`/api/info`).

### 1.2 ¿Se "estandariza" la señal? — no hay un paso de z-score

A diferencia de un pipeline de ML genérico, **aquí no se normaliza la señal cruda** (no hay
`(x - media) / std` en ningún punto antes del FIR). La señal `X` que sale del loader son los
voltios crudos del casco/dataset. Las únicas dos normalizaciones que existen en toda la cadena
están **dentro** de otras etapas, con un propósito matemático preciso, no de "limpieza" de
escala:

- **Normalización por traza** dentro de la covarianza del CSP (§1.4) — evita que un trial con
  más amplitud global domine el promedio de covarianzas.
- **Normalización por suma** dentro de la log-varianza (§1.5) — hace la característica robusta
  a la amplitud global del trial.

(Los endpoints del servidor multiplican por `1e6` para mostrar **µV** en el frontend —
`x * 1e6` en `server/app.py` — pero es solo una conversión de unidad para visualización, no
toca el pipeline de clasificación.)

### 1.3 Filtro FIR pasa-banda — la convolución explícita

**Archivos:** `backend/src/bci/dsp/fir_filters.py` (diseño de `h[n]`),
`backend/src/bci/dsp/convolution.py` (aplicar `h[n]` por convolución),
`backend/src/bci/dsp/frequency_response.py` (verificación en frecuencia).

#### 1.3.1 Diseño de `h[n]` por ventaneo (windowed-sinc)

Función: `design_bandpass_fir(low_hz, high_hz, fs, num_taps, window)`.

Se normalizan los cortes a frecuencia digital (ciclos/muestra):

$$ f_1 = \frac{\text{low\_hz}}{f_s}, \qquad f_2 = \frac{\text{high\_hz}}{f_s} $$

Eje de muestras centrado (para que el filtro resultante sea simétrico, $N$ impar):

$$ m = n - \frac{N-1}{2}, \qquad n = 0, \dots, N-1 $$

**Paso 1 — filtro ideal.** La respuesta al impulso de un pasa-banda ideal es la diferencia de
dos pasa-bajos ideales (uno en $f_2$, restado el de $f_1$):

$$ h_{\text{ideal}}[m] = 2f_2 \cdot \operatorname{sinc}(2f_2\, m) \;-\; 2f_1 \cdot \operatorname{sinc}(2f_1\, m), \qquad \operatorname{sinc}(x) = \frac{\sin(\pi x)}{\pi x} $$

Esta $h_{\text{ideal}}$ es infinita y no causal (el filtro ideal no existe en la práctica).

**Paso 2 — truncar** a $N$ taps (`num_taps`, 101 por defecto): nos quedamos solo con
$m \in \left[-\frac{N-1}{2}, \frac{N-1}{2}\right]$. Eso introduce rizado (fenómeno de Gibbs).

**Paso 3 — ventanear**, multiplicando por una ventana $w[n]$ que suaviza los extremos hacia
cero (reduce los lóbulos laterales a cambio de una transición algo más ancha):

$$
w[n] =
\begin{cases}
1 & \text{rectangular (sin suavizado)} \\[4pt]
0.5 - 0.5\cos\!\left(\dfrac{2\pi n}{N-1}\right) & \text{Hann} \\[8pt]
0.54 - 0.46\cos\!\left(\dfrac{2\pi n}{N-1}\right) & \text{Hamming (por defecto)} \\[8pt]
0.42 - 0.5\cos\!\left(\dfrac{2\pi n}{N-1}\right) + 0.08\cos\!\left(\dfrac{4\pi n}{N-1}\right) & \text{Blackman}
\end{cases}
$$

$$ h[n] = h_{\text{ideal}}[n] \cdot w[n] $$

**Normalización de ganancia.** Para que el filtro tenga ganancia ≈1 en el centro de la banda de
paso $f_c = \tfrac12(f_1+f_2)$, se evalúa $|H(e^{j\omega_c})|$ directamente como suma (la DTFT
en un solo punto) y se reescala:

$$ G = \sum_n h[n] \cos(2\pi f_c\, m), \qquad h[n] \leftarrow \frac{h[n]}{G} $$

El resultado se empaqueta en la dataclass `FIRFilter(h, fs, low_hz, high_hz, window)`, con
propiedades `num_taps = len(h)` y **retardo de grupo** $= \frac{N-1}{2}$ muestras (constante,
porque $h$ es simétrico ⇒ fase lineal). Con $N=101$ a $f_s=250\,$Hz: retardo $=50$ muestras
$=0.2\,$s.

`design_from_config(cfg, fs)` simplemente lee `cfg["fir_filter"]` (`low_hz=8`, `high_hz=30`,
`num_taps=101`, `window="hamming"`) y usa el `fs` **real** del dataset cargado (no el del YAML)
para que el mismo config sirva con cualquier frecuencia de muestreo.

#### 1.3.2 Aplicar el filtro: convolución discreta

Una vez se tiene $h[n]$, filtrar es **convolucionar**:

$$ y[n] = (x * h)[n] = \sum_{k} h[k]\, x[n-k] $$

`backend/src/bci/dsp/convolution.py` la implementa de tres formas matemáticamente idénticas:

- **`convolve_mac`** — doble bucle literal: para cada $n$, acumula $h[k]\cdot x[n-k]$ sobre
  $k$. Cada iteración interna es una operación **MAC** (Multiplicar-ACumular); lenta
  ($O(N\!\cdot\!M)$ en Python puro), pensada para enseñar/animar la operación, no para producción.
- **`mac_terms(x, h, n)`** — devuelve los términos individuales $\{k,\, h[k],\, x[n-k],\,
  h[k]\!\cdot\!x[n-k]\}$ que producen una muestra $y[n]$ concreta; es lo que el frontend usa
  para animar el MAC paso a paso.
- **`convolve`** — versión vectorizada por **superposición**: en vez de recorrer $n$, recorre
  los $M$ taps y en cada paso suma una copia de $x$ entera, escalada por $h[k]$ y desplazada
  $k$ muestras:
  $$ y = \sum_{k=0}^{M-1} h[k]\cdot \operatorname{shift}(x,\,k) $$
  Es la versión que se usa en producción (`apply_filter`), mucho más rápida en numpy.

`_trim(y_full, len_x, len_h, mode)` recorta la convolución `"full"` (longitud
$N_x+N_h-1$) al modo pedido: `"same"` (longitud $N_x$, centrada — compensa el retardo de grupo,
**usado offline**) o `"valid"` (solo donde $h$ se solapa por completo — **usado en vivo**, ver
§2.2).

`apply_filter(X, h, mode='same')` aplica esto sobre el último eje (tiempo) de un array de
cualquier dimensión (p. ej. `(n_trials, n_canales, n_muestras)`), aplanando y recorriendo cada
fila.

#### 1.3.3 Verificación: respuesta en frecuencia

`backend/src/bci/dsp/frequency_response.py` calcula la DTFT de $h[n]$ por suma directa (no FFT,
para que la teoría sea explícita):

$$ H(e^{j\omega}) = \sum_{n} h[n]\, e^{-j\omega n}, \qquad \omega = \frac{2\pi f}{f_s} $$

$|H(e^{j\omega})|$ confirma que el filtro deja pasar ≈1 entre 8–30 Hz y atenúa el resto (esto
es lo que se grafica en `/api/filter` y en el Laboratorio del frontend).

### 1.4 CSP — filtrado espacial lineal

**Archivo:** `backend/src/bci/spatial/csp.py` (clase `CSP`).

El FIR mezcla **muestras vecinas en el tiempo**; el CSP mezcla **canales en el espacio**, en
cada instante, con pesos fijos (sin convolución):

$$ Z = W\!\cdot\!X, \qquad z[c,n] = \sum_j W[c,j]\, x[j,n] $$

con $X \in \mathbb{R}^{n_{\text{canales}}\times n_{\text{tiempo}}}$ (un trial) y
$W \in \mathbb{R}^{n_{\text{comp}}\times n_{\text{canales}}}$ los filtros espaciales aprendidos.

#### 1.4.1 Qué optimiza

La imaginación motora cambia la **potencia** µ/β (ERD), y la potencia de una señal de media
cero es su **varianza**. El CSP busca direcciones espaciales $w$ donde la varianza de la clase
1 sea muy distinta de la varianza de la clase 2 — un problema de **autovalores generalizados**:

$$ C_1\, w = \lambda\, (C_1 + C_2)\, w, \qquad \lambda \in [0,1] $$

$\lambda \approx 1$ → la componente "se enciende" en la clase 1; $\lambda \approx 0$ → en la
clase 2. Se conservan los autovectores de los **extremos** del espectro (los más
discriminativos).

#### 1.4.2 Cómo se calcula `fit(X, y)` — paso a paso, con las fórmulas exactas

**Paso 1 — covarianza por trial, normalizada por traza** (función `_class_covariance`). Para
cada trial $E$ (un epoch ya filtrado, $n_{\text{canales}}\times n_{\text{tiempo}}$):

$$ C^{(i)} = \frac{E_i E_i^\top}{\operatorname{tr}(E_i E_i^\top)} $$

y se promedia sobre los $N_k$ trials de la clase $k$:

$$ C_k = \frac{1}{N_k}\sum_{i \in k} C^{(i)} $$

La normalización por traza evita que un trial de mayor amplitud global domine el promedio.

**Paso 1bis — shrinkage opcional** (función `_shrink`, parámetro `csp.shrinkage` del config,
$\gamma \in [0,1]$, por defecto 0):

$$ C_k^{\text{reg}} = (1-\gamma)\, C_k + \gamma\, \frac{\operatorname{tr}(C_k)}{n}\, I_n $$

Encoge la covarianza hacia una esfera (identidad escalada), atenuando autovalores extremos poco
fiables cuando hay pocos trials. (Medido con rigor en este proyecto: con $\gamma=0$ ya
óptimo en los tres datasets — ver §4.4, **resultado negativo**: aquí no ayuda.)

**Paso 2 — whitening de la covarianza compuesta** $C_c = C_1 + C_2$. Se diagonaliza con
`np.linalg.eigh` (matriz simétrica):

$$ C_c = U \Lambda U^\top $$

se descartan direcciones casi nulas ($\lambda \le 10^{-10}\cdot\lambda_{\max}$, rango
deficiente) y se construye la matriz de blanqueo:

$$ P = \Lambda^{-1/2} U^\top \qquad (\text{de forma que } P\,C_c\,P^\top = I) $$

**Paso 3 — llevar $C_1$ al espacio blanqueado y diagonalizarlo:**

$$ S_1 = P\, C_1\, P^\top = B\, \Psi\, B^\top $$

los autovalores de $\Psi$ (ordenados de mayor a menor) son exactamente los $\lambda$ del
problema generalizado del paso 1.4.1.

**Paso 4 — filtros espaciales en el espacio original:**

$$ W_{\text{full}} = B^\top P \qquad \in \mathbb{R}^{\text{rango}\times n_{\text{canales}}} $$

**Paso 5 — selección de componentes.** Con `n_components = m` (par, por defecto 4), se toman
los $m/2$ autovectores de mayor $\lambda$ y los $m/2$ de menor $\lambda$ (los extremos del
espectro):

$$ W = W_{\text{full}}[\text{idx}], \qquad \text{idx} = \{0,\dots,\tfrac{m}{2}-1\} \cup \{\text{rango}-\tfrac{m}{2},\dots,\text{rango}-1\} $$

**Patrones espaciales** (para los topomapas — *cómo* se proyecta cada componente sobre el
cuero cabelludo, lo inverso de "qué pesos aplicar"):

$$ A = W^{+} \qquad (\text{pseudo-inversa de Moore-Penrose}) $$

Todo esto se calcula **una sola vez**, con los trials de **entrenamiento** (`fit_csp=True`
dentro de `MotorImageryPipeline._features`, ver §1.6), y se congela: `W` (`csp.filters_`),
`A` (`csp.patterns_`) y los `λ` (`csp.eigenvalues_`) quedan fijos para todo lo que sigue,
incluida la transmisión en vivo.

`transform(X)` aplica simplemente $Z = W\!\cdot\!X$ por trial (`np.einsum("cj,tjn->tcn", W, X)`).

### 1.5 Log-varianza — el puente DSP → ML

**Archivo:** `backend/src/bci/features/log_variance.py`.

Cada componente CSP $z_i$ es todavía una serie temporal; se resume en **un número por trial**:

$$ f_i = \log\!\left(\frac{\operatorname{var}(z_i)}{\sum_j \operatorname{var}(z_j)}\right) $$

- $\operatorname{var}(z_i)$ convierte la serie temporal en una medida de **potencia** (1
  número/componente) — recordar que la ERD es un cambio de potencia.
- El $\log$ acerca la distribución (muy asimétrica) a una **gaussiana**, justo lo que asume el
  LDA del siguiente paso.
- La normalización por la suma de varianzas hace la característica robusta a la amplitud
  global del trial.

`log_variance(Z, normalize=True)` calcula esto sobre `Z` con forma
`(n_trials, n_componentes, n_tiempo)` y devuelve `(n_trials, n_componentes)`, la matriz de
características $F$ que entra al clasificador.

### 1.6 LDA — el clasificador lineal que cierra la cadena

**Archivo:** `backend/src/bci/models/lda.py`.

Modelo: cada clase es una gaussiana con media propia $\mu_k$ pero **covarianza compartida**
$\Sigma$. Bajo ese supuesto, la regla de Bayes da una función discriminante lineal:

$$ \delta_k(x) = x^\top \Sigma^{-1}\mu_k \;-\; \tfrac12\,\mu_k^\top \Sigma^{-1}\mu_k \;+\; \log \pi_k $$

(los términos cuadráticos en $x$ se cancelan al ser $\Sigma$ común a todas las clases ⇒ la
frontera entre dos clases es un **hiperplano**). Se reescribe en forma explícita
$\delta_k(x) = w_k^\top x + b_k$ con:

$$ w_k = \Sigma^{-1}\mu_k \qquad (\texttt{coef\_}) $$
$$ b_k = -\tfrac12\,\mu_k^\top \Sigma^{-1}\mu_k + \log \pi_k \qquad (\texttt{intercept\_}) $$

**`fit(F, y)`** calcula, sobre la matriz de características $F$ (salida de §1.5):

- $\mu_k$ = media de las filas de $F$ de la clase $k$ (`means_`).
- $\pi_k = N_k / N$ = proporción de muestras de la clase $k$ (`priors_`).
- $\Sigma$ = covarianza **dentro de clase** (*within-class scatter*), agregando la dispersión de
  cada clase respecto a su propia media y normalizando por grados de libertad:
  $$ \Sigma = \frac{1}{N - K} \sum_{k} \sum_{i \in k} (f_i - \mu_k)(f_i - \mu_k)^\top $$
  ($N$ = nº de trials, $K$ = nº de clases). Se invierte con `np.linalg.pinv` (pseudo-inversa,
  más estable numéricamente que la inversa exacta).

**`decision_function(F)`** devuelve $\delta_k(f)$ para cada muestra y clase:
$F\!\cdot\!\texttt{coef\_}^\top + \texttt{intercept\_}$.

**`predict(F)`** = $\hat y = \arg\max_k \delta_k(f)$.

(Verificado en los tests: predicciones idénticas a `sklearn.LinearDiscriminantAnalysis`.)

### 1.7 El pipeline completo offline y su evaluación honesta

**Archivo:** `backend/src/bci/pipeline/offline.py`, clase `MotorImageryPipeline`.

Encadena las cuatro etapas con un único objeto, parametrizado por el config y el `fs` real:

```
__init__(cfg, fs):
    self.filt = design_from_config(cfg, fs)   # FIR fijo (no se "entrena")
    self.csp  = CSP(n_components, shrinkage)
    self.lda  = LDA()
```

`_features(X, fit_csp, y)` es el corazón:

1. `Xf = apply_filter(X, filt.h, mode='same')` — filtra **todo el epoch** con el FIR (§1.3.2).
2. **Recorte a la ventana de imaginación activa**: `Xf = Xf[..., i0:i1]` con
   $i_0 = \lfloor t_{\min,\text{rel}}\cdot f_s \rfloor$, $i_1 = \lfloor t_{\max,\text{rel}}\cdot
   f_s\rfloor$ (sección `classification_window` del YAML; para 2a, `[0.5, 2.5]`s relativos al
   inicio del epoch, es decir `[2.5, 4.5]`s del trial real). Se filtra **primero** y se recorta
   **después** para descartar a la vez el transitorio del *cue* y el del propio FIR (su retardo
   de grupo, ~50 muestras).
3. Si `fit_csp=True`: `self.csp.fit(Xf, y)` — el CSP **se aprende aquí**, solo con los trials
   que se le pasen (en `fit()`, eso es `idx_train`, nunca el held-out).
4. `Z = self.csp.transform(Xf)` — proyección espacial $Z=W\!\cdot\!X$ (§1.4).
5. `return log_variance(Z, normalize=...)` — características $F$ (§1.5).

`fit(X, y)` llama a `_features(X, fit_csp=True, y=y)` y luego `self.lda.fit(F, y)`.
`predict(X)` llama a `_features(X, fit_csp=False)` (CSP **ya fijo**) y `self.lda.predict(F)`.

**Disciplina contra la fuga de datos:** el CSP y el LDA se ajustan **solo** con los datos de
entrenamiento de cada partición; el FIR no se "entrena" (coeficientes fijos), así que aplicarlo
no introduce fuga. Esto se verifica con dos evaluaciones:

- **`evaluate_kfold`** — validación cruzada estratificada de 5 folds; CSP+LDA se reajustan en
  cada fold con solo el train de ese fold.
- **`evaluate_by_session`** — entrena en la sesión `'0train'`, evalúa en `'1test'`: la
  estimación **honesta** de cómo generalizaría a una grabación de **otro día** (train y test no
  comparten sesión). Es la que importa para la demo en vivo, porque ahí ocurre exactamente eso.

`EvalResult` guarda `accuracy`, `kappa` (Cohen's $\kappa$, corrige por azar) y la matriz de
confusión.

### 1.8 Entrenar y persistir — separación entrenar/transmitir

**Archivo:** `backend/src/bci/pipeline/training.py`.

**`split_train_demo(data)`** decide la partición honesta:
- Si el dataset tiene ≥2 sesiones (2a: `'0train'`, `'1test'`) → entrena con la primera, reserva
  la segunda íntegra como demo.
- Si tiene una sola sesión (PhysioNet, Liu2024) → reserva una fracción estratificada del 30 %
  (`train_test_split(..., stratify=y, random_state=42)`).

**`train_subject(cfg, dataset, subject)`**: carga los datos del sujeto, aplica el split,
**`MotorImageryPipeline(cfg, fs).fit(X[idx_train], y[idx_train])`**, mide accuracy/kappa sobre
`X[idx_demo]` (el held-out, **nunca visto en el fit**) y construye una **`ModelCard`**
(dataclass) con todos los metadatos: dataset, sujeto, método, `fs`, clases, canales, qué
partición se usó (`holdout`, `train_session`), `n_train`, `n_demo`, `accuracy`, `kappa`, fecha
de entrenamiento, nº de componentes CSP, banda/orden del FIR.

**`save_model(model, card, out_dir)`** persiste dos ficheros junto a los datos procesados:

- `model_{dataset}_s{subject}_{method}.pkl` — el objeto `MotorImageryPipeline` completo (FIR +
  CSP entrenado + LDA entrenado), serializado con `pickle`.
- `model_{dataset}_s{subject}_{method}.json` — la `ModelCard` como JSON (la "ficha" que el
  frontend muestra en `/api/model`).

`train_eegnet_subject` sigue una lógica distinta (EEGNet es un "espejo" de visualización, no se
usa en vivo): se entrena con **todos** los trials para tener filtros más limpios, y se reportan
por separado dos accuracies honestas (inter-sesión y k-fold) medidas con modelos auxiliares
entrenados aparte — ver detalle en §4.6.

---

## 2. Mundo ONLINE — transmisión en vivo

### 2.1 De dónde sale la señal "en vivo"

No hay casco real: el servidor **reproduce** los mismos datos de MOABB, pero **solo los trials
reservados como demo** (`idx_demo` de §1.8) — los que el modelo nunca vio al entrenar. Esto pasa
en `backend/src/bci/server/app.py`, función `_split_idx`, que relee el `holdout` guardado en la
`ModelCard` para reconstruir exactamente la misma partición.

El endpoint WebSocket `/ws/stream?dataset=&subject=` (`app.py`, `ws_stream`):
1. Carga `EpochedData` (`_get_data`) y el pipeline ya entrenado (`_get_pipeline`,
   des-pickleado desde el `.pkl` de §1.8 — **no se reentrena nada aquí**).
2. Recorre los trials de `idx_demo` **en bucle** (`j = (j+1) % len(idx_demo)`).
3. Para cada uno, llama a `StreamSimulator.stream(data.X[idx])` y emite cada resultado por el
   socket con un `await asyncio.sleep(step_s)` entre mensajes (ritmo de reproducción).

### 2.2 Filtro FIR **causal**, con estado

**Archivo:** `backend/src/bci/streaming/simulator.py`, clase `CausalFIR`.

Offline filtrábamos con `mode='same'`, que para calcular $y[n]$ usa también muestras
**futuras** de $x$ (mira "hacia adelante" para centrar la salida). En vivo eso es imposible:
solo existe el pasado. Un filtro **causal** calcula

$$ y[n] = \sum_{k=0}^{M-1} h[k]\, x[n-k], \qquad n \ge 0 $$

usando **solo** muestras ya recibidas. Para que esto funcione **por trozos** (chunks que van
llegando), `CausalFIR` mantiene un **buffer de estado** con las últimas $M-1$ muestras de cada
canal:

```
process_chunk(chunk):
    ext = concat(buffer, chunk)              # pasado guardado + nuevo trozo
    out[c] = convolve(ext[c], h, mode='valid')   # solo solapamientos completos → causal
    buffer = ext[:, -(M-1):]                  # se guarda el nuevo "pasado" para el próximo chunk
```

Está **verificado por test** (`test_streaming.py`) que filtrar por chunks da exactamente el
mismo resultado que filtrar la señal entera de una vez — no hay saltos en las fronteras entre
chunks.

**Precio de la causalidad: el retardo de grupo es real.** El mismo FIR de fase lineal de §1.3
retrasa la señal $(M-1)/2 = 50$ muestras $= 0.2\,$s a 250 Hz. Offline ese retardo se compensaba
centrando la salida (`mode='same'`); en vivo es un retraso real e inevitable entre lo que ocurre
en el cerebro y lo que el sistema puede clasificar.

### 2.3 La ventana deslizante — qué es exactamente "una ventana"

**Archivo:** `backend/src/bci/streaming/simulator.py`, clase `StreamSimulator`.

Parámetros del config (`streaming.window_s=2.0`, `streaming.step_s=0.1`):

- **`window_s`** = duración (en segundos) del trozo de señal **ya filtrada** que se le pasa al
  CSP+LDA de una vez. En muestras: `self.window = int(window_s * fs)` (500 muestras a 250 Hz).
- **`step_s`** = cada cuánto **avanza** la ventana, es decir, cada cuánto se emite una nueva
  predicción (en muestras: `self.step = int(step_s * fs)`, 25 muestras).

El bucle de `stream(X_cont)`:

```
filtered = []                                            # buffer de señal filtrada (causal)
for start in 0, step, 2·step, ...:
    chunk = X_cont[:, start : start+step]                  # siguiente trozo crudo
    f = CausalFIR.process_chunk(chunk)                      # filtrado causal (§2.2)
    filtered = concat(filtered, f)
    if len(filtered) > window:  filtered = filtered[-window:]   # nos quedamos con las últimas window_s
    if len(filtered) == window:
        pred, probs, info = pipeline.classify_window(filtered)  # §2.4
```

Es decir: **"una ventana"** es, en cada instante $t$ del stream, el tramo de **2 segundos** de
señal *ya filtrada* (causalmente) que termina justo en $t$ — una ventana deslizante con
solapamiento del 95 % entre ventanas consecutivas (avanza 0.1s, dura 2s). Cada `step_s`
segundos se emite una predicción nueva sobre la ventana vigente en ese instante; por eso, dentro
de un mismo trial, la predicción y su confianza **evolucionan** mensaje a mensaje (igual que lo
haría un sistema en vivo real).

**Coherencia de diseño importante:** `window_s = 2.0`s coincide exactamente con la duración de
la ventana de clasificación usada para entrenar (`classification_window`: `tmax_rel - tmin_rel
= 2.5 - 0.5 = 2.0`s). El CSP y el LDA se entrenaron viendo segmentos de 2s de la fase de
imaginación activa; en vivo se les sigue dando segmentos de exactamente 2s, aunque su posición
dentro del trial vaya deslizando — así la escala temporal con la que "habla" el modelo es
siempre la misma.

`StreamSimulator.stream` también calcula, por ventana, la **potencia µ/β por canal** (para el
Cerebro 3D en vivo):

$$ \text{power}_c = \log\!\left(\operatorname{var}(\text{filtered}_c) + 10^{-12}\right) $$

### 2.4 Clasificar la ventana: el CSP y el LDA *ya entrenados*, aplicados

**Archivo:** `backend/src/bci/pipeline/offline.py`, método `MotorImageryPipeline.classify_window`.

A diferencia del offline (que **aprende** $W$ y $\delta_k$), aquí solo se **aplican**:

1. $$ Z = W\cdot X_{\text{ventana}} \qquad (\texttt{self.csp.transform}) $$
   con $W$ = `csp.filters_`, los filtros espaciales fijos aprendidos en §1.4, y $X_{\text{ventana}}$
   el tramo de 2s ya filtrado causalmente (§2.2–2.3), de forma
   $(n_{\text{canales}}\times 500\text{ muestras})$. $Z$ tiene forma
   $(n_{\text{comp}}\times 500)$: una "señal virtual" por componente CSP.
2. $$ F = \log\!\left(\frac{\operatorname{var}(Z_i)}{\sum_j \operatorname{var}(Z_j)}\right) \qquad (\texttt{log\_variance}) $$
   exactamente la misma fórmula de §1.5, evaluada sobre esta única ventana → un vector
   `feat` de $n_{\text{comp}}$ números.
3. $$ \delta_k(F) = F\cdot w_k + b_k \qquad (\texttt{lda.decision\_function}) $$
   con $w_k, b_k$ = `lda.coef_`, `lda.intercept_` ya aprendidos en §1.6. `pred` = la clase con
   mayor $\delta_k$.
4. **Pseudo-probabilidades** por softmax sobre los $\delta_k$ (para mostrar "confianza"; el LDA
   no es probabilístico per se, esto es una transformación monótona de las puntuaciones):
   $$ p_k = \frac{e^{\delta_k - \max_j \delta_j}}{\sum_j e^{\delta_j - \max_j \delta_j}} $$
5. **`disc`** — la proyección con signo sobre el eje discriminante (caso binario), que sitúa la
   ventana respecto a la **frontera** (en 0):
   $$ \text{disc} = \delta_{1}(F) - \delta_{0}(F) $$
   Es justo lo que el frontend dibuja en la vista "LDA · frontera de decisión": un punto que se
   mueve sobre una recta, a un lado u otro del cero.

`classify_window` devuelve `(pred, probs, info)` con `info = {"feat": F, "disc": disc}` — las
dos cantidades que separan visualmente la etapa CSP de la etapa LDA en el frontend
(`components/charts/PipelineStages.tsx`).

### 2.5 De ventanas a decisión por trial — el voto suave

Cada trial genera **muchas** predicciones (una por ventana cada 0.1s), pero las ventanas cuyo
centro cae fuera de la fase de imaginación activa (inicio del trial, transición del cue, final)
son casi aleatorias — el sujeto todavía no está imaginando nada coherente, o ya paró. Por eso la
decisión final **no se toma por ventana**, sino acumulando.

El servidor calcula, por trial, los límites temporales de la zona "activa" en el eje $t$ de fin
de ventana (`app.py`, dentro de `ws_stream`):

$$ t_{\text{lo}} = t_{\min,\text{rel}} + \frac{\text{window\_s}}{2}, \qquad t_{\text{hi}} = t_{\max,\text{rel}} + \frac{\text{window\_s}}{2} $$

(se suma medio `window_s` porque `t` marca el **final** de la ventana, no su centro; así el
*centro* de la ventana cae dentro de `[tmin_rel, tmax_rel]`). Estos límites (`alo`, `ahi`) se
envían en cada mensaje del WebSocket.

El frontend (`pages/LiveStream.tsx`) acumula, **solo** para las ventanas con
$t \in [\texttt{alo}, \texttt{ahi}]$, las probabilidades `probs` de cada clase, y al cambiar de
trial:

$$ \bar p_k = \frac{1}{n}\sum_{\text{ventanas activas}} p_k, \qquad \hat y = \arg\max_k \bar p_k $$

Si $\max_k \bar p_k$ no alcanza un **umbral** configurable (control deslizante en la UI, 0.65
por defecto), el sistema **se abstiene** ("sin decisión") en vez de forzar una clase — así se
modela el reposo o la falta de intención clara sin inventar una tercera clase que el LDA nunca
vio (el clasificador sigue siendo estrictamente binario).

---

## 3. Clases y módulos — resumen de responsabilidades

| Clase / función | Archivo | Responsabilidad | Se calcula… |
|---|---|---|---|
| `EpochedData` | `datasets/moabb_loader.py` | Contenedor de datos crudos epocados: `X, y, metadata, ch_names, sfreq`. | Una vez, al cargar el dataset. |
| `FIRFilter` | `dsp/fir_filters.py` | Coeficientes `h[n]` del pasa-banda + metadatos (`fs`, banda, ventana, retardo de grupo). | Una vez por config/`fs` (no depende de los datos, solo de la banda deseada). |
| `convolve` / `convolve_mac` / `apply_filter` | `dsp/convolution.py` | Aplicar `h[n]` a una señal por convolución (offline, `mode='same'`). | Cada vez que se filtra un epoch o trial. |
| `FrequencyResponse` | `dsp/frequency_response.py` | $H(e^{j\omega})$ de un filtro, para verificación/visualización. | A demanda (no es parte del flujo de clasificación). |
| `CSP` | `spatial/csp.py` | Aprende `W` (filtros), `A` (patrones) y `λ` (autovalores) — `fit()`; proyecta `Z=W·X` — `transform()`. | **Offline, una vez**, con `idx_train`. Se congela y se reutiliza tal cual en vivo. |
| `log_variance` | `features/log_variance.py` | Reduce `Z` a un vector de características `F` por trial/ventana. | Cada vez que hay un `Z` nuevo (offline por trial, en vivo por ventana). |
| `LDA` | `models/lda.py` | Aprende `μ_k, Σ, π_k → coef_, intercept_` — `fit()`; evalúa `δ_k(x)` — `decision_function()`/`predict()`. | **Offline, una vez**, con las `F` de entrenamiento. Igual que el CSP, se congela. |
| `MotorImageryPipeline` | `pipeline/offline.py` | Orquesta FIR→CSP→log-var→LDA; `fit/predict/score`; `classify_window` para una sola ventana ya filtrada; `evaluate_kfold`/`evaluate_by_session`. | El objeto que se entrena offline y se serializa a `.pkl`. |
| `ModelCard` | `pipeline/training.py` | Metadatos legibles del modelo: con qué se entrenó, qué se reservó, accuracy honesta. | Al entrenar (`train_subject`/`train_eegnet_subject`), se guarda como `.json`. |
| `split_train_demo` | `pipeline/training.py` | Decide qué trials son `idx_train` vs `idx_demo` (por sesión o por fracción estratificada). | Una vez por sujeto, antes de entrenar. |
| `CausalFIR` | `streaming/simulator.py` | Filtra **causalmente**, con estado entre chunks (`mode='valid'`). | En vivo, continuamente, chunk a chunk. |
| `StreamSimulator` | `streaming/simulator.py` | Mantiene la ventana deslizante; llama a `classify_window` cada `step_s`; calcula la potencia por canal. | En vivo, una instancia por conexión de stream. |
| `EEGNet` / `EEGNetClassifier` | `models/eegnet.py` | Red convolucional "espejo": sus capas imitan FIR (temporal) y CSP (espacial), pero aprendidas. Solo para comparar/visualizar (§4.6), nunca en la ruta de clasificación en vivo. | Offline, por separado del CSP+LDA. |
| `FBCSPPipeline` | `pipeline/fbcsp.py` | Variante experimental: banco de FIR (varias sub-bandas) + CSP por banda + selección por información mutua + LDA. Descartada en producción (§4.5). | Solo en scripts de evaluación (`eval_fbcsp.py`). |

---

## 4. Resultados, comparaciones y por qué

Todos los números siguientes están medidos y documentados con su script de origen en
[`presentacion.md`](presentacion.md); aquí se resumen para conectar cada resultado con la
fórmula/etapa que lo explica.

### 4.1 Accuracy por dataset (k-fold vs. inter-sesión, media sobre sujetos)

| Dataset | $f_s$ | Canales | Trials/sujeto | k-fold (mezcla sesiones) | Inter-sesión (honesta) |
|---|---|---|---|:---:|:---:|
| BCI IV 2a (`BNCI2014_001`) | 250 Hz | 22 | ~280 | **0.688 ± 0.129** | **0.636 ± 0.133** |
| PhysioNet MMI (`PhysionetMI`) | 160 Hz | 64 | ~45 | **0.608 ± 0.172** | — |
| Liu2024 | 500 Hz | 29 | ~40 | **0.536 ± 0.081** | — |

**Por qué inter-sesión < k-fold (siempre).** El k-fold mezcla trials de todas las sesiones, así
que el CSP/LDA pueden "memorizar" rasgos específicos de un día concreto (drift de impedancia de
electrodos, estado del sujeto) que no estarán en el día de test real. Inter-sesión entrena en
`'0train'` y evalúa en `'1test'` —días distintos, igual que pasaría con un casco real— y por eso
es **más baja pero más honesta**. Es la cifra que predice el rendimiento de la demo en vivo.

**Por qué cae con menos trials por sujeto (maldición de la dimensionalidad).** Estimar las
covarianzas $C_k$ (matrices $n_{\text{canales}}\times n_{\text{canales}}$) del §1.4 con pocos
trials es estadísticamente inestable → el CSP sobreajusta. 2a (~280 trials) > PhysioNet (~45) >
Liu2024 (~40), en el mismo orden que la accuracy.

### 4.2 Variabilidad inter-sujeto ("BCI illiteracy")

Dentro de un mismo dataset, la accuracy por sujeto va de ~0.85 (sujetos "fuertes", 3 y 8 en 2a)
a ~0.50 — prácticamente azar (sujetos 5 y 9). No es un fallo del método: hay personas cuya
imaginación motora produce una ERD µ/β clara y separable espacialmente (lo que el CSP necesita
para encontrar un buen $W$) y otras que apenas la generan. Por eso siempre se entrena **un
modelo por sujeto** (mezclar sujetos en un único CSP empeora, porque los pesos espaciales no se
alinean entre cabezas distintas) y se reporta la **media ± desviación**, nunca un único número.

### 4.3 Shrinkage del CSP — resultado negativo

Se barrió $\gamma \in [0, 0.5]$ en la fórmula de §1.4.2 (paso 1bis) esperando que ayudase en los
datasets con pocos trials. Resultado medido: $\gamma=0$ es **óptimo** en los tres datasets; la
accuracy **empeora** al subir $\gamma$. Motivo: las covarianzas del CSP ya estaban bien
condicionadas (normalización por traza + promedio sobre trials + descarte de direcciones de
rango casi nulo en el whitening), así que el shrinkage no resuelve ningún problema numérico
real — solo aplana el espectro de $\lambda$ hacia 0.5, que es justo la señal que el CSP necesita
explotar. El parámetro queda implementado y configurable (`csp.shrinkage`), con valor por
defecto 0.

### 4.4 FBCSP (banco de filtros) — resultado matizado

Sustituir el FIR único 8–30 Hz por un banco de 7 sub-bandas (§3, `FBCSPPipeline`) no mejora de
forma significativa en promedio (inter-sesión: 0.636 vs. 0.655, $p=0.68$ en test pareado), pero
**rescata** a sujetos cuya información discriminante vive en una sub-banda estrecha (S5:
0.51→0.72) y **hunde** a otros por sobreajuste de la sesión de entrenamiento (S8: 0.82→0.67). El
diseño óptimo del filtro resulta ser **dependiente del sujeto**; en producción se mantiene el
CSP de banda única por ser más simple y más robusto en los sujetos usados en la demo.

### 4.5 EEGNet vs. CSP+LDA

| Modelo | k-fold | Inter-sesión (honesta) |
|---|:---:|:---:|
| CSP + LDA (clásico) | 0.722 | **0.738** |
| EEGNet (deep learning) | 0.651 | **0.504** (≈ azar) |

EEGNet aprende sus propios filtros temporales (≈ FIR, Conv2D) y espaciales (≈ CSP,
DepthwiseConv2D) **a partir de los datos**, sin que nadie le diga la banda 8–30 Hz ni calcule
covarianzas. Con solo ~140 trials de entrenamiento (un día), la red **sobreajusta esa sesión**
y no generaliza al día siguiente, mientras que CSP+LDA —que ya incorpora el conocimiento del
dominio (la banda µ/β, la estructura espacial vía covarianzas)— es mucho más eficiente con pocos
datos. Por esto EEGNet se usa solo como "espejo" comparativo (`/api/eegnet`, pestaña en "El
Modelo"), nunca para clasificar en vivo.

### 4.6 El techo: por qué nunca se llega al 100 %

Una BCI de imaginación motora **no invasiva** y binaria tiene, en el estado del arte, un techo
de precisión de ~70–85 %. El EEG es una señal ruidosa (atenuada por cráneo y cuero cabelludo),
varía entre sesiones del mismo sujeto y muchísimo entre sujetos distintos; no es una limitación
de este pipeline en particular, sino del canal de medición. Las cifras de este proyecto
(inter-sesión 0.636±0.133 en 2a, streaming por voto ~0.74–0.82 en los sujetos fuertes) están
dentro de ese rango esperado para el método clásico CSP+LDA.

### 4.7 Streaming vs. offline: ¿se pierde precisión al pasar a tiempo real?

No: streaming (voto por trial, sujeto 1) da **0.745**, prácticamente igual a la inter-sesión
offline (**0.738**) del mismo sujeto. Tiene sentido por construcción: es **el mismo objeto**
`MotorImageryPipeline` (mismo $W$, mismo $\delta_k$) aplicado a los mismos trials held-out; lo
único que cambia es que el filtrado es causal y por trozos (§2.2) en vez de `mode='same'` de una
sola vez — y eso, verificado, da resultados numéricamente equivalentes.

---

## Apéndice — todas las fórmulas juntas

**FIR (diseño):**
$$ f_1=\tfrac{\text{low}}{f_s},\ f_2=\tfrac{\text{high}}{f_s} \qquad h_{\text{ideal}}[m] = 2f_2\operatorname{sinc}(2f_2 m) - 2f_1\operatorname{sinc}(2f_1 m) \qquad h[n] = h_{\text{ideal}}[n]\cdot w[n] $$

**Convolución (aplicar el FIR):**
$$ y[n] = \sum_k h[k]\,x[n-k] \qquad (\text{offline: } \texttt{mode='same'};\ \ \text{en vivo: causal, } \texttt{mode='valid'} ) $$

**Respuesta en frecuencia:**
$$ H(e^{j\omega}) = \sum_n h[n] e^{-j\omega n} $$

**CSP — covarianza por clase:**
$$ C_k = \frac{1}{N_k}\sum_{i\in k} \frac{E_i E_i^\top}{\operatorname{tr}(E_i E_i^\top)}, \qquad C_k^{\text{reg}} = (1-\gamma)C_k + \gamma\tfrac{\operatorname{tr}(C_k)}{n}I $$

**CSP — autovalores generalizados / whitening:**
$$ C_1 w = \lambda(C_1+C_2) w \qquad P=\Lambda^{-1/2}U^\top \ (C_c=U\Lambda U^\top) \qquad S_1=PC_1P^\top=B\Psi B^\top \qquad W=B^\top P $$

**CSP — proyección y patrones:**
$$ Z = W\cdot X \qquad A = W^{+} $$

**Log-varianza:**
$$ f_i = \log\!\left(\frac{\operatorname{var}(z_i)}{\sum_j \operatorname{var}(z_j)}\right) $$

**LDA — entrenamiento:**
$$ \Sigma = \frac{1}{N-K}\sum_k\sum_{i\in k}(f_i-\mu_k)(f_i-\mu_k)^\top \qquad w_k=\Sigma^{-1}\mu_k \qquad b_k=-\tfrac12\mu_k^\top\Sigma^{-1}\mu_k+\log\pi_k $$

**LDA — decisión:**
$$ \delta_k(x)=w_k^\top x+b_k \qquad \hat y=\arg\max_k \delta_k(x) \qquad \text{disc}=\delta_1-\delta_0 \qquad p_k=\frac{e^{\delta_k-\max_j\delta_j}}{\sum_j e^{\delta_j-\max_j\delta_j}} $$

**Voto suave por trial (en vivo):**
$$ \bar p_k=\frac1n\sum_{t\in[\text{alo,ahi}]} p_k(t) \qquad \hat y_{\text{trial}}=\arg\max_k \bar p_k \ \text{ si } \max_k\bar p_k\ge\text{umbral, si no: abstención} $$
