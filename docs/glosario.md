# Glosario del proyecto BCI

Diccionario de términos del proyecto. Cada entrada tiene: **qué es** (definición),
**el concepto detrás** (intuición / por qué importa) y, cuando aplica, **dónde vive en
este proyecto**. Organizado en cuatro bloques: Neurofisiología/EEG, Teoría LTI y DSP,
Machine Learning, y Herramientas.

> Notación: `x[n]` = señal de entrada discreta, `h[n]` = respuesta al impulso del filtro,
> `y[n]` = salida, `fs` = frecuencia de muestreo, `ω` = frecuencia angular (rad/muestra).

---

## 1. Neurofisiología y EEG

### EEG (Electroencefalografía)
**Qué es:** registro de la actividad eléctrica del cerebro mediante electrodos colocados
sobre el cuero cabelludo. Se mide en microvoltios (µV).
**El concepto detrás:** millones de neuronas que se activan a la vez generan pequeñas
corrientes; su suma produce un voltaje medible en la superficie de la cabeza. Es una señal
temporal continua que, al muestrearla, se vuelve una señal discreta `x[n]` — la entrada de
todo nuestro pipeline LTI.

### Imaginación motora (Motor Imagery, MI)
**Qué es:** imaginar un movimiento (mover la mano izquierda, los pies…) **sin ejecutarlo**.
**El concepto detrás:** imaginar un movimiento activa las mismas zonas de la corteza motora
que ejecutarlo, produciendo patrones EEG detectables. Es la base de nuestra BCI: el usuario
"piensa" un movimiento y el sistema lo clasifica. Las clases de nuestro dataset son
`left_hand`, `right_hand`, `feet`, `tongue`.

### Corteza sensoriomotora
**Qué es:** región del cerebro que planifica y ejecuta movimientos, organizada
somatotópicamente (cada parte del cuerpo tiene su zona — el "homúnculo motor").
**El concepto detrás:** como la mano izquierda se controla desde el hemisferio derecho (y
viceversa), la imaginación de cada extremidad activa zonas distintas. Esa **separación
espacial** es justo lo que el filtro espacial CSP explota.

### Ritmos µ (mu) y β (beta) — banda µ/β
**Qué es:** oscilaciones de la actividad cerebral en rangos de frecuencia concretos:
**mu = 8–12 Hz** y **beta = 12–30 Hz**, sobre la corteza sensoriomotora.
**El concepto detrás:** en reposo, estas regiones "idlean" produciendo ondas mu/beta
fuertes. Al imaginar un movimiento, esas ondas **se atenúan** (ver ERD). Por eso nuestro
filtro FIR pasa-banda se diseña justo en **8–30 Hz**: concentra la información útil y
descarta el resto (parpadeos <4 Hz, ruido muscular >40 Hz, red eléctrica a 50/60 Hz).

### ERD / ERS (Desincronización / Sincronización relacionada a evento)
**Qué es:** **ERD** = caída de potencia en la banda µ/β durante (la imaginación de) un
movimiento; **ERS** = el rebote/aumento al terminar.
**El concepto detrás:** "desincronización" porque las neuronas, al activarse para la tarea,
dejan de oscilar al unísono → la amplitud rítmica baja. Medir *dónde* y *cuánto* baja la
potencia (la varianza de la señal filtrada) es la señal que distingue una clase de otra.

### Sistema 10-20 / Canal / Electrodo
**Qué es:** estándar internacional para nombrar y posicionar electrodos (C3, C4, Cz, Fz…).
Cada electrodo es un **canal** de la señal.
**El concepto detrás:** las letras indican región (C=central/motora) y los números
hemisferio (impar=izquierdo, par=derecho). **C3 y C4** (sobre las cortezas motoras
izquierda y derecha) son los canales más informativos para mano derecha vs izquierda.

### Trial / Época (Epoch), Run y Sesión (la jerarquía de una grabación)
**Qué son:** tres niveles de organización de un experimento EEG, de menor a mayor:
- **Trial (o época/epoch):** *un intento*. El sujeto recibe un *cue* ("imagina la mano
  izquierda") y realiza la tarea durante unos segundos. Es el ejemplo que clasificamos:
  una matriz `(canales × tiempo)` con una etiqueta. En el 2a cada trial dura 4 s y usamos
  la sub-ventana de imaginación activa (2.5–4.5 s del trial).
- **Run:** *una tanda de trials seguidos* sin pausa larga, grabada como un archivo continuo.
  En el 2a, **un run = 48 trials** (12 de cada una de las 4 clases, en orden aleatorio) y
  dura ~387 s.
- **Sesión:** *un día/bloque de grabación*, formado por varios runs. En el 2a, **una sesión
  = 6 runs** y hay **2 sesiones** por sujeto (`0train` para entrenar, `1test` para evaluar).

**Ejemplo (sujeto 1 del 2a):** 2 sesiones × 6 runs × 48 trials = 576 trials totales; de
ellos, 12+12 = 24 por run son `left/right` → 288 trials binarios.

### "Unir los runs" (¿qué significa y por qué MOABB lo hace?)
**Qué es:** pegar varios runs en una sola señal continua *antes* de recortar los trials.
**El concepto detrás (con el caso real verificado):** el último trial de cada run empieza a
los 381.0 s, pero el archivo del run dura 386.94 s — faltan **0.09 s (23 muestras)** para
completar la ventana de 4 s. Si epocas **cada run por separado**, MNE descarta ese trial por
"demasiado corto" (`TOO_SHORT`). Si **unes los runs**, ese trial se completa con las primeras
muestras del run siguiente y *no se pierde*. Por eso **MOABB** (que une los runs) obtiene
**288** trials, mientras que **nosotros** (que epocamos cada run aislado, a propósito)
obtenemos **281**. *Verificado:* al concatenar los runs sin rechazar las fronteras se
recuperan ~287, confirmando el mecanismo.
**Nuestra decisión:** NO unimos runs. Completar un trial con muestras del *siguiente* run
mezcla dos segmentos no contiguos en el tiempo, y luego el FIR convolucionaría a través de esa
frontera artificial — algo sucio para un sistema LTI. Preferimos descartar 7 de 288 trials
(2.4 %, clases balanceadas) antes que inventar continuidad. (Detalle en `docs/presentacion.md`.)

### Frecuencia de muestreo (fs)
**Qué es:** número de muestras por segundo (Hz). En el 2a, **fs = 250 Hz**.
**El concepto detrás:** convierte la señal analógica continua en `x[n]` discreta.
El **teorema de Nyquist** dice que solo podemos representar frecuencias hasta `fs/2`
(=125 Hz aquí), más que suficiente para la banda µ/β. Todas las frecuencias del DSP se
miden relativas a fs.

### Topomapa
**Qué es:** mapa de calor de un valor (potencia, peso CSP…) proyectado sobre un dibujo de
la cabeza vista desde arriba.
**El concepto detrás:** traduce números por canal en una imagen espacial intuitiva. Lo
usaremos para visualizar **los pesos del CSP** y ver qué zonas del cuero cabelludo pondera
cada filtro espacial.

---

## 2. Teoría de sistemas LTI y DSP (el núcleo académico)

### Sistema LTI (Lineal e Invariante en el Tiempo)
**Qué es:** un sistema que cumple **linealidad** (la respuesta a una suma de entradas es la
suma de las respuestas; escalar la entrada escala la salida) e **invarianza temporal** (si
retrasas la entrada, la salida se retrasa igual, sin cambiar de forma).
**El concepto detrás:** es la piedra angular del curso. Un sistema LTI queda **totalmente
descrito por su respuesta al impulso `h[n]`**, y su salida es siempre la **convolución**
de la entrada con `h[n]`. Nuestros filtros FIR y el CSP son operaciones lineales → encajan
en esta teoría, y por eso podemos analizarlos con convolución y respuesta en frecuencia.

### Respuesta al impulso `h[n]`
**Qué es:** la salida del sistema cuando la entrada es un impulso unitario `δ[n]` (un 1 en
n=0, ceros en el resto).
**El concepto detrás:** es la "huella dactilar" de un sistema LTI: conociéndola, conoces el
sistema entero. En un filtro FIR, **`h[n]` son directamente los coeficientes del filtro**
(los *taps*).

### Convolución discreta
**Qué es:** la operación `y[n] = Σ_k h[k] · x[n-k]`.
**El concepto detrás:** es el corazón del proyecto. Para cada instante `n`, se **voltea** el
kernel `h`, se **desliza** sobre la señal, se **multiplica** punto a punto y se **suma**
todo. Así un sistema LTI produce su salida. Intuición: cada muestra de salida es una mezcla
ponderada de muestras de entrada vecinas; los pesos son `h[n]`.
**Dónde vive:** `backend/src/bci/dsp/convolution.py` (implementada a mano).

### MAC (Multiply-Accumulate / Multiplicar-Acumular)
**Qué es:** la operación elemental "multiplica dos números y súmalos a un acumulador",
repetida dentro de la convolución.
**El concepto detrás:** una convolución es, físicamente, **una ráfaga de MACs**. Es la
operación más ejecutada en DSP y en hardware (DSPs, GPUs, redes neuronales). Visualizar el
MAC paso a paso es la mejor forma de "ver" la convolución funcionando.
**Dónde vive:** se animará en el módulo `ConvolutionViz/` del frontend.

### Filtro FIR (Respuesta Impulsiva Finita)
**Qué es:** filtro cuya respuesta al impulso `h[n]` tiene **longitud finita** (N *taps* =
N coeficientes). Su salida es **pura convolución**: `y[n] = Σ_{k=0}^{N-1} h[k]·x[n-k]`. No
tiene realimentación: la salida solo depende de las entradas, nunca de salidas anteriores.
**Ejemplo sencillo (media móvil de 3):** con `h = [1/3, 1/3, 1/3]`, cada salida es el promedio
de la muestra actual y las dos anteriores: `y[n] = (x[n] + x[n-1] + x[n-2]) / 3`. Es un FIR de
3 taps que suaviza la señal (pasa-bajos). Nuestro pasa-banda µ/β es la misma idea con 101 taps
bien elegidos.
**El concepto detrás:** es el filtro LTI más sencillo de entender y **siempre estable** (no
puede "explotar" porque no se realimenta). Lo elegimos por su **fase lineal**.
**Dónde vive:** `backend/src/bci/dsp/fir_filters.py`.

### Filtro IIR (Respuesta Impulsiva Infinita)
**Qué es:** filtro **con realimentación**: la salida depende también de salidas anteriores, así
que su `h[n]` es infinita (un solo impulso de entrada deja "ecos" que nunca se apagan del todo).
**Ejemplo sencillo:** `y[n] = 0.9·y[n-1] + x[n]`. Un impulso en la entrada produce
`1, 0.9, 0.81, 0.729, …` — una respuesta infinita que decae. Con muy pocos coeficientes logra
filtros muy selectivos.
**El concepto detrás:** más eficiente que el FIR (menos coeficientes para la misma selectividad)
pero puede ser **inestable** (si el eco crece en vez de decaer) y **distorsiona la fase**
(retrasa cada frecuencia de forma distinta). `scipy.signal.lfilter` típicamente implementa IIR.
Lo **evitamos a propósito** para conservar fase lineal y mantener la convolución explícita.

### Fase lineal
**Qué es:** propiedad de un FIR **simétrico** (con nº de taps impar): todas las frecuencias
se retrasan el **mismo tiempo**.
**El concepto detrás:** un retardo constante no deforma la *forma* de la señal, solo la
desplaza — crucial para EEG, donde la morfología y el timing de los ritmos importan. Por eso
usamos `num_taps` impar (101).

### Respuesta en frecuencia `H(e^jω)`
**Qué es:** la Transformada de Fourier de `h[n]`. Su módulo `|H(e^jω)|` dice cuánto se
amplifica/atenúa cada frecuencia; su fase, cuánto se retrasa.
**El concepto detrás:** **convolución en el tiempo = multiplicación en frecuencia**. Por eso
un filtro se entiende mejor mirando su `|H|`: para nuestro pasa-banda µ/β debe valer ~1 entre
8–30 Hz y ~0 fuera. Es la herramienta para *verificar* que el filtro hace lo que queremos.
**Dónde vive:** `backend/src/bci/dsp/frequency_response.py` y el módulo `FrequencyResponse/`.

### Ventana (window) — p. ej. Hamming
**Qué es:** función suave por la que se multiplican los coeficientes ideales del filtro al
diseñarlo.
**El concepto detrás:** el filtro pasa-banda ideal tiene `h[n]` infinita; al truncarla
aparecen "rizados" (fenómeno de Gibbs). Una ventana (Hamming) suaviza los extremos y reduce
esos lóbulos laterales, a cambio de una transición un poco más ancha. Es el método de diseño
FIR "por ventaneo" que implementamos.

### Fenómeno de Gibbs
**Qué es:** las **oscilaciones / rizados** que aparecen cerca de un cambio brusco cuando se
representa una señal con un número *finito* de componentes de frecuencia (o, en filtros, al
**truncar** una respuesta al impulso infinita).
**El concepto detrás:** el pasa-banda ideal tiene flancos perfectamente verticales en 8 y 30 Hz,
lo que exige una `h[n]` infinita. Al quedarnos con solo 101 taps, esos flancos no pueden ser
perfectos: aparecen ondulaciones (lóbulos laterales) en `|H(e^jω)|` que dejan pasar un poco de
las frecuencias que queríamos bloquear. Aumentar los taps reduce su anchura pero **no su
altura**; por eso se usa una **ventana** (Hamming): a cambio de una transición algo más ancha,
aplasta esos rizados. Se ven en `docs/figures/fir_frequency_response.png` (los "dientes" por
debajo de −60 dB).

### Causalidad
**Qué es:** un sistema es **causal** si su salida `y[n]` depende solo del **presente y el
pasado** (`x[n], x[n-1], …`), nunca del futuro.
**El concepto detrás:** es obligatorio en tiempo real: un casco no puede usar muestras que aún
no han llegado. Offline filtramos con `mode='same'`, que para centrar la señal usa muestras
*futuras* (no causal, pero válido porque ya tenemos todo grabado). En la simulación en vivo
(Hito 6) el filtro es causal y mantiene un **estado** entre trozos. Ver [retardo de grupo].
**Dónde vive:** `backend/src/bci/streaming/simulator.py` (`CausalFIR`).

### Retardo de grupo (group delay)
**Qué es:** el tiempo que un filtro retrasa la señal. En un FIR de fase lineal es **constante**
e igual a `(N-1)/2` muestras (con N=101 → 50 muestras = **0.2 s** a 250 Hz).
**El concepto detrás:** un retardo constante solo *desplaza* la señal sin deformarla — bueno.
Pero en tiempo real ese retraso es **real e inevitable**: la predicción llega 0.2 s "tarde".
Offline lo compensamos recentrando (`mode='same'`); en vivo se asume. Idea clave: *no existe
filtrado de fase lineal sin retardo*.

### DTFT (Transformada de Fourier en Tiempo Discreto)
**Qué es:** la transformada que lleva una señal discreta del dominio del tiempo al de la
frecuencia: `X(e^jω) = Σ_n x[n]·e^{-jωn}`. Aplicada a `h[n]` da la respuesta en frecuencia.
**El concepto detrás:** es la herramienta que hace visible *qué* frecuencias contiene una señal
o *qué* deja pasar un filtro. La calculamos a mano (suma directa) y la verificamos contra la
**FFT** (un algoritmo rápido para calcular esta misma transformada en frecuencias equiespaciadas).

---

## 3. Machine Learning

### CSP (Common Spatial Patterns / Patrones Espaciales Comunes)
**Qué es:** un **filtro espacial lineal**: combina los canales (`z = Wᵀ·x`) para crear nuevas
"señales virtuales" que **maximizan la varianza de una clase mientras minimizan la de la
otra**.
**El concepto detrás:** distintas imaginaciones activan zonas distintas (mano izq → hemisferio
derecho). El CSP aprende, por **diagonalización conjunta** de las matrices de covarianza de
cada clase (un problema de **autovalores generalizados**), las combinaciones de electrodos que
mejor separan las clases por su **potencia**. Es lineal → es un sistema LTI en el dominio
espacial (en vez de temporal): donde el FIR mezcla *muestras en el tiempo*, el CSP mezcla
*canales en el espacio*.
**Dónde vive:** `backend/src/bci/spatial/csp.py`; sus pesos se visualizan en `SpatialWeights/`.

### FBCSP (Filter Bank CSP / CSP con banco de filtros)
**Qué es:** la extensión del CSP que, en vez de una sola banda 8–30 Hz, usa un **banco de filtros
FIR** (varias sub-bandas de ~4 Hz), aplica CSP en cada una, junta las características y **selecciona
las más informativas** (información mutua) antes del LDA.
**El concepto detrás:** es un **banco de filtros**, un concepto central de DSP, y por tanto puro
LTI (varios FIR en paralelo). La idea: cada ritmo discrimina mejor en su propia sub-banda.
**El resultado (medido en el proyecto):** *no* mejora de forma robusta. En los 9 sujetos del 2a,
la ganancia media inter-sesión no es significativa (p≈0.68); ayuda mucho a algunos sujetos (su
señal está en una banda estrecha) y perjudica a otros (sobreajusta la sesión). Por eso producción
mantiene el CSP de banda única, más simple y robusto. Es un **resultado matizado** documentado,
como el del [shrinkage]. **Dónde vive:** `backend/src/bci/pipeline/fbcsp.py`,
`backend/scripts/eval_fbcsp.py`.

### Filtro espacial
**Qué es:** combinación lineal ponderada de los canales en un mismo instante.
**El concepto detrás:** generaliza la idea de filtrar: en lugar de mezclar muestras vecinas en
el *tiempo* (FIR), mezcla canales en el *espacio*. CSP es un filtro espacial *aprendido*.

### Varianza y log-varianza
**Qué es:** la varianza mide la "potencia"/dispersión de una señal; tomamos su logaritmo como
característica final.
**El concepto detrás:** tras el CSP, la información de clase está en **cuánta potencia** tiene
cada componente (recordar ERD: la potencia µ/β baja al imaginar). El `log` normaliza la
distribución para que el clasificador lineal funcione mejor. Es el puente entre el DSP y el ML.
**Dónde vive:** `backend/src/bci/features/log_variance.py`.

### Problema de autovalores generalizados
**Qué es:** resolver `C₁·w = λ·(C₁+C₂)·w`, con `C₁,C₂` las covarianzas de cada clase.
**El concepto detrás:** es la maquinaria matemática del CSP. Los autovectores `w` son los
filtros espaciales; los autovalores `λ` (entre 0 y 1) indican cómo se reparte la varianza
entre clases. Tomamos los más extremos (λ≈1 y λ≈0): los más discriminativos.

### Whitening (blanqueo) y diagonalización conjunta
**Qué es:** el método de dos pasos con que resolvemos el CSP. **Whitening:** transformar los
datos para que la covarianza combinada `C₁+C₂` se vuelva la identidad (todas las direcciones
con igual varianza, "blancas"). **Diagonalización conjunta:** en ese espacio blanqueado,
diagonalizar `C₁` revela las direcciones que más varianza tienen en una clase y menos en la otra.
**El concepto detrás:** es preferible hacer estos dos pasos explícitos que llamar a un solver
opaco: se ve que el CSP es pura álgebra lineal (rotar + reescalar + rotar). Es el corazón de
`CSP.fit`. Ver [problema de autovalores generalizados].

### Fuga de datos (data leakage)
**Qué es:** cuando información del conjunto de **test** se cuela en el **entrenamiento**,
inflando artificialmente el resultado.
**El concepto detrás:** nos pasó en el Hito 5: ajustamos el CSP con *todo* el dataset y luego
hicimos validación cruzada → 77 % engañoso. Al corregirlo (ajustar CSP y LDA **solo con train**
dentro de cada partición) bajó a ~63 %. *Cualquier paso que "aprenda" de los datos (CSP, LDA,
normalizaciones) debe ver únicamente el train.* Es el error más común y peligroso en ML aplicado.

### Validación inter-sesión
**Qué es:** entrenar con una sesión (día) y evaluar con otra distinta (`0train` → `1test`).
**El concepto detrás:** es la estimación **honesta** de cómo funcionaría en vivo, porque train y
test no comparten día (ni colocación exacta de electrodos ni estado del sujeto). El k-fold que
mezcla sesiones es más optimista. En el sujeto 1 dio 0.738 (vs 0.719 en k-fold).

### Entrenar vs transmitir (offline / online)
**Qué es:** las **dos fases** de una BCI. *Offline (antes):* se calcula el modelo a partir de
datasets etiquetados. *Online (ahora):* llega una señal en tiempo real y el modelo ya entrenado
la clasifica.
**El concepto detrás:** no deben mezclarse. La regla de oro: **lo que se transmite no se entrena,
y lo que se entrena no se transmite** (si no, hay [fuga de datos] y la demo miente). En el
proyecto se entrena con una partición y se reserva un *held-out* que solo se usa para la demo en
vivo. El modelo se guarda a disco (un **artefacto**) y en la demo solo se **carga**.
**Dónde vive:** `backend/src/bci/pipeline/training.py`, `backend/scripts/train_model.py`.

### Held-out (partición reservada)
**Qué es:** un subconjunto de datos **apartado** que el modelo nunca ve durante el entrenamiento.
**El concepto detrás:** es lo que da una medida honesta de generalización. En el proyecto el
held-out cumple doble función: estima la accuracy real **y** es justo la señal que se "transmite"
en la demo en vivo (en el 2a = la sesión `1test`; en datasets de una sesión = un 30 % estratificado).

### Artefacto del modelo (ModelCard)
**Qué es:** el modelo entrenado **guardado a disco** (`.pkl` con CSP+LDA) junto con su **ficha**
(`.json`): con qué se entrenó, qué se reservó para la demo y la accuracy honesta.
**El concepto detrás:** hace literal el relato "el modelo se entrenó *antes*". El servidor no
reentrena: carga el artefacto. La web muestra la ficha etiquetada como "cálculo previo al
streaming", separando visualmente los dos mundos.

### Softmax y voto mayoritario
**Qué es:** **softmax** convierte las puntuaciones del clasificador en pseudo-probabilidades que
suman 1 (para mostrar "confianza"); **voto mayoritario** = la clase más predicha entre varias
ventanas de un mismo trial.
**El concepto detrás:** en streaming clasificamos muchas ventanas deslizantes por trial; el voto
mayoritario combina esas predicciones en una sola decisión robusta por trial, y el softmax sirve
para visualizar cómo evoluciona la confianza en vivo.

### LDA (Análisis Discriminante Lineal)
**Qué es:** clasificador que separa clases con una **frontera lineal** (un hiperplano).
**El concepto detrás:** asume que cada clase es una gaussiana y busca la dirección que más
separa las medias relativas a la dispersión. Lo elegimos por coherencia con el tema: tras un
pipeline LTI (lineal), un clasificador lineal cierra la historia de forma elegante e
interpretable.
**Dónde vive:** `backend/src/bci/models/lda.py`.

### EEGNet
**Qué es:** red neuronal convolucional compacta diseñada para EEG.
**El concepto detrás:** es **secundaria** en el proyecto y se justifica porque sus capas
**imitan el pipeline clásico**: una convolución temporal ≈ banco de filtros FIR, y una
convolución *depthwise* por canal ≈ filtro espacial tipo CSP. Es el puente conceptual entre
"filtros LTI hechos a mano" y "filtros LTI aprendidos".
**Dónde vive:** `backend/src/bci/models/eegnet.py` (etapa posterior).

### CSP+LDA vs EEGNet (teoría vs datos)
**Qué es:** los dos enfoques del proyecto para el mismo problema (decidir izquierda/derecha).
**El concepto detrás:** se diferencian en **quién pone la teoría**.
- *CSP+LDA (clásico):* **tú** impones el conocimiento. El FIR (banda µ/β) lo diseñas a mano y no
  se entrena; "entrenar" significa solo ajustar el **CSP** (pesos del filtro espacial, vía
  autovalores) y el **LDA** (la frontera lineal). Son pocos coeficientes → funciona bien con
  **pocos datos**.
- *EEGNet (deep learning):* la red **no recibe la teoría, la reinventa**. Su 1.ª capa aprende los
  filtros temporales (≈ FIR) y la 2.ª los espaciales (≈ CSP) desde cero, ajustando **miles** de
  coeficientes por descenso de gradiente → necesita **muchos datos**.
- *Resultado con nuestros datos:* gana el clásico (0.72 vs 0.65), porque hay pocos trials. Ambos
  son, en el fondo, **sistemas LTI encadenados**: la diferencia es de dónde salen los coeficientes.

### Validación cruzada (cross-validation)
**Qué es:** evaluar el modelo entrenando y probando en particiones distintas de los datos.
**El concepto detrás:** evita el autoengaño de medir sobre los mismos datos de entrenamiento.
En EEG conviene separar por **sesión** (entrenar en `0train`, evaluar en `1test`) para una
estimación honesta de cómo generalizaría en vivo.

### Accuracy, Matriz de confusión y Kappa de Cohen
**Qué es:** **accuracy** = % de aciertos; **matriz de confusión** = tabla real vs predicho;
**kappa** = acuerdo corregido por azar.
**El concepto detrás:** la accuracy sola engaña con clases desbalanceadas. La matriz muestra
*qué* clases se confunden. La **kappa** (0 = azar, 1 = perfecto) es el estándar en BCI porque
penaliza aciertos que podrían deberse a la suerte.

---

## 4. Herramientas

### MOABB (Mother of All BCI Benchmarks)
**Qué es:** librería Python que estandariza la descarga y carga de datasets BCI públicos.
**El concepto detrás:** unifica decenas de datasets bajo una misma API (`Dataset` → `Paradigm`
→ `X, y, metadata`). Lo usamos sobre todo para **obtener la señal cruda**; el filtrado lo
hacemos nosotros para no ocultar la teoría LTI.

### MNE-Python
**Qué es:** librería estándar para análisis de señales neurofisiológicas (EEG/MEG).
**El concepto detrás:** MOABB devuelve objetos `Raw`/`Epochs` de MNE, que encapsulan señal +
metadatos (canales, fs, eventos). Nos da la estructura de datos; el procesamiento numérico lo
escribimos a mano sobre los arrays NumPy que MNE expone.

### BCI Competition IV — dataset 2a (`BNCI2014_001`)
**Qué es:** dataset de referencia en imaginación motora. 9 sujetos · 2 sesiones · 4 clases
(left_hand, right_hand, feet, tongue) · 22 canales EEG · fs 250 Hz.
**El concepto detrás:** es el "benchmark de oro" del campo: comparable con cientos de papers.
Ideal para CSP y para empezar binario (mano izq vs der).

### LSL (Lab Streaming Layer)
**Qué es:** protocolo para transmitir señales en tiempo real entre programas/dispositivos.
**El concepto detrás:** es el estándar para BCI en vivo. Hoy lo "simulamos" reproduciendo la
señal grabada; mañana, el casco **Ultracortex Mark IV** publicará por LSL y el mismo pipeline
lo consumirá sin cambios. Por eso aislamos el módulo `streaming/`.

### Ultracortex Mark IV
**Qué es:** casco EEG de hardware abierto (OpenBCI) que el proyecto contempla integrar a futuro.
**El concepto detrás:** justifica que diseñemos el sistema "stream-first": que funcione igual
con datos grabados que con un casco real, sin reescribir el procesamiento.
