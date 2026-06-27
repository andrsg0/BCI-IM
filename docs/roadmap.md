# Roadmap / TODO

Registro vivo de mejoras pendientes del proyecto BCI. Junta los **planes previos**
(generalización EEGNet, señal "en vivo" realista, cerebro 3D) con las **ideas nuevas**
de mejora de UX/contenido. No es un plan de implementación inmediata: es la lista
acordada de qué se quiere hacer y con qué decisiones ya tomadas.

Convención de estado:
- `[ ]` pendiente · `[~]` en progreso · `[x]` hecho
- **Decidido:** decisión ya tomada (no volver a discutir).
- **Abierto:** falta decidir antes de implementar.

---

## A. Generalización entre sujetos (EEGNet)

Hoy CSP+LDA es **sujeto-específico** y EEGNet solo se usa para *visualizar* filtros
(no clasifica en vivo). Esto explora si EEGNet puede generalizar a sujetos no vistos.

- [x] **EEGNet pooled + LOSO — HECHO (backend).** `train_eegnet_pooled()` en
      `pipeline/training.py` entrena una red con varios sujetos juntos, evalúa con
      leave-one-subject-out (honesto cross-subject) y guarda el modelo base
      (`model_{dataset}_s0_eegnet_pooled.pkl/.json`, con LOSO por sujeto en `extra`).
      CLI: `scripts/train_eegnet_pooled.py`. Además EEGNet ahora usa **GPU** si hay
      (`pick_device` en `models/eegnet.py`) y tiene `predict_proba`.
      **Resultado (2a, 9 sujetos, 250 épocas):** media LOSO = **0.599** (rango 0.498–0.684;
      modelo base entrenado con 2538 trials). Es la generalización honesta a un sujeto nuevo
      *sin calibrar*: por encima del azar (0.50) pero bastante por debajo del within-subject
      (~0.69 k-fold), lo que motiva el fine-tuning con calibración corta (siguiente paso).
- [ ] *Fine-tuning* con capas congeladas + calibración corta del sujeto nuevo
      (transfer learning), para acercarse al rendimiento sujeto-específico. (siguiente paso)
- [x] Exponer los resultados pooled/LOSO en el frontend — HECHO. La página Resultados
      (`pages/Results.tsx`) muestra `PooledCard` (media LOSO, nº sujetos del pool, trials,
      épocas, banda, dispositivo, fecha) y `AggregateMatrix` (matriz 2×2 agregada por sujeto
      sobre todos los datasets + Wilcoxon). Endpoints `/api/results*` y `/api/results_aggregate`.
- Refs: `backend/src/bci/models/eegnet.py`, `pipeline/training.py`
  (`train_eegnet_subject`, `train_eegnet_pooled`), `docs/presentacion.md` ("Próximos pasos").
- **Resuelto (2026-06):** EEGNet **SÍ** se usa en vivo. Es uno de los 4 regímenes de la página
  de Clasificación (CSP+LDA / EEGNet × within / cross), entrenado cross-subject DENTRO de cada
  dataset (N-1 sujetos → sujeto held-out). No hay pooled cross-dataset.

## A+. Escalado de datos y comparación de métodos

Plan para responder "¿cuántos/qué datos necesita EEGNet para rendir bien?" y para
comparar de forma honesta el método clásico (CSP+LDA) con el de IA (EEGNet).

### Comparación 2×2 (entregable principal)

Cuatro números medidos de forma **consistente** sobre el mismo dataset:

Medido en 2a (9 sujetos, mismos folds, `compare_methods_BNCI2014_001.csv`):

|            | within-subject (k-fold) | cross-subject (LOSO) |
|------------|-------------------------|----------------------|
| **CSP+LDA**| **0.688**               | 0.574                |
| **EEGNet** | 0.626                   | **0.599**            |

> **Nota:** el "~0.72" que figuraba antes era una **estimación redondeada**, no un valor
> medido. El k-fold real de CSP+LDA en 2a siempre fue **0.688** (coincide exacto entre
> `results_2a.csv` y `compare_methods`). No hubo regresión. La dispersión entre sujetos es
> grande (0.51–0.91): reportar siempre el **rango**, no solo la media.

- within-subject = entrenas y evalúas en el MISMO sujeto (calibrado).
- cross-subject (LOSO) = entrenas con los DEMÁS y evalúas en el excluido (usuario nuevo
  sin calibrar). En 2a el CSP+LDA cross (0.574) no cae a azar porque los 9 sujetos comparten
  el mismo montaje de 22 canales; aun así queda por debajo de EEGNet cross (0.599).
- [x] Script `scripts/compare_methods.py` — HECHO. Produce la matriz 2×2 + tabla por sujeto +
      CSV, todo con los mismos folds. Pendiente: mostrarla en el frontend (idea "comparación
      estadística", con κ además de accuracy).

### Plan de datos (para subir el rendimiento de EEGNet)

> **Realidad:** el techo del estado del arte de imaginación motora 2 clases no invasiva es
> ~70–85 %. Objetivos honestos: cross-subject **sin calibrar** ≈ 0.65–0.70 (no 0.80);
> por usuario **con calibración/fine-tuning** ≈ 0.75–0.80 (0.80–0.85 en buenos sujetos);
> within-subject ya ~0.69 de media (CSP+LDA en 2a). Siempre reportar el **rango entre
> sujetos** (0.51–0.91), no solo la media.

- [x] **Paso 1 — Aumentación de datos — HECHO (backend).** `bci/datasets/augment.py`:
      `augment_trials()` aplica ruido gaussiano leve, desplazamiento temporal y escalado de
      amplitud, **solo en la partición de entrenamiento** (no fuga de datos). Integrado en
      `train_eegnet_pooled(augment=True, augment_copies=...)` y en el CLI (`--augment`).
- [x] **Paso 2 — Escalar con PhysionetMI — CANCELADO (cambio de plan 2026-06-26).** Se abandona
      PhysioNet como vía de escalado. El nuevo plan usa **datasets multi-sesión** (≥2 sesiones)
      que sirven a la vez para **demo en vivo y entrenamiento**: **2a (BNCI2014_001, 2 sesiones),
      2b (BNCI2014_004, 5 sesiones) y Kumar2024 (6 sesiones)** — ver memoria
      `arquitectura-datasets-regimenes`. El cross-subject se entrena DENTRO de cada dataset (N-1
      sujetos → sujeto held-out), no con un pool de PhysioNet. Configs PhysioNet/REGISTRY ya
      eliminados.
- [x] **RESUELTO: EEGNet en Dreyer2023 era un artefacto de BANDA.** Diagnóstico 2026-06-21:
      EEGNet colapsaba a 0.500 porque, con banda amplia 4–40 Hz, se sobreajustaba a un
      artefacto fuera de la banda MI (4–13 Hz: deriva/ocular/alfa); CSP (restringido a
      potencia µ/β) era inmune. **Fix:** banda de EEGNet **configurable por dataset**
      (`_eegnet_features` lee `cfg['eegnet']['band_low/high']`); Dreyer usa **beta 13–30 Hz**
      → S3 0.44 → **0.854** (CSP 0.89), sin dañar 2a. Es un resultado pedagógico: el sesgo
      inductivo LTI rescata al DL. Ver memoria `eegnet-falla-en-dreyer`. **Pendiente:**
      re-lanzar EEGNet pooled de Dreyer con la banda nueva; verificar Cho2017 igual (también
      512 Hz, probablemente mismo artefacto → ya tiene sentido darle su `eegnet:` band).
- [ ] **Paso 4 — Fine-tuning con calibración corta.** Congelar capas tempranas del modelo
      base pooled y adaptar las últimas con 20–40 trials del usuario nuevo (1–2 runs). Es el
      paso que de verdad lleva a un usuario concreto hacia 0.75–0.85.
- [x] **Paso 3 — Pooling entre datasets (LODO) — CANCELADO (cambio de plan 2026-06-26).** Ya
      NO se hace EEGNet cross-dataset (leave-one-dataset-out); **solo cross-subject dentro de
      cada dataset**. Con ello desaparece la necesidad del remuestreo a fs común y la
      armonización de montajes para entrenar. El script `train_eegnet_crossdataset.py` ya se
      eliminó. (El trabajo de remuestreo LTI de `docs/remuestreo.md` queda como material
      didáctico de Sistemas Lineales, no como pieza del entrenamiento.)

- Refs: `pipeline/training.py` (`train_eegnet_pooled`, `_eegnet_features`),
  `scripts/train_eegnet_pooled.py`, `scripts/compare_methods.py`.

## B. Señal "en vivo" realista

Simular una señal de la que **no se conoce el inicio ni el fin** (como un casco real),
en lugar de reproducir trials recortados con fronteras conocidas.

- [ ] **B.1 — Loader continuo.** Reutilizar la señal `Raw` continua de MOABB (ya existe
      antes del epoching; ver `moabb_loader.py:128`). Endpoints ya disponibles:
      `/api/continuous`, `/api/continuous_all`, `_get_raw()` en `server/app.py` — pero hoy
      toman la **primera** sesión/run, no la held-out. Decidir qué tramo continuo replicar
      en la demo (idealmente el held-out, coherente con el split honesto de `training.py`).
- [ ] **B.2 — Quitar dependencia de `alo`/`ahi`.** Hoy `ws_stream` calcula bordes de
      ventana activa por trial (`app.py:427-428`) y el front filtra por ellos
      (`LiveStream.tsx:156`). En señal continua no hay fronteras: clasificar de forma
      continua con **umbral de confianza / abstención** en vez de votar dentro del trial.
- [x] **B.3 — Widget "muñeco de brazos" — HECHO.** `components/HandPuppet.tsx`: muñeco SVG
      (vista de espaldas, sin espejo: mano izq. del muñeco = izq. del espectador) que levanta
      y saluda con el brazo correspondiente. **Decidido (matiz del usuario):** se mueve según
      la **ETIQUETA REAL** del trial (`true`), NO la predicción del modelo, y solo durante la
      franja de imaginación activa `[alo, ahi]` (cuando la persona realmente empieza a mover
      la mano). El color es el de la clase; funciona en los 4 regímenes (el servidor ya manda
      `true`/`alo`/`ahi`). Integrado como widget `puppet` en `pages/LiveStream.tsx`
      (helper `handSideFromLabel`). Sirve para contrastar visualmente «lo que la persona hacía»
      contra «lo que el modelo predijo».
- **Resuelto:** el muñeco vive en **Clasificación (`/live`)** como un widget más del GridBoard
  (queda reubicable y, cuando exista el catálogo del Dashboard configurable, insertable allí).

## C. Cerebro 3D anatómico (GLB) — HECHO

Implementado: el cerebro 3D ya carga una **malla anatómica** real (`.glb`) dentro de la
esfera/scalp, con heatmap cortical en GPU.

- [x] `.glb` de cerebro servido desde `frontend/public/models/brain.glb` (~204 KB, ligero).
- [x] Cargado con `useGLTF` de `@react-three/drei` en el componente `components/BrainMesh.tsx`
      (auto-centra/auto-escala por bounding box; `useGLTF.preload`).
- [x] Electrodos reconciliados sobre una **cáscara elipsoidal** (no clavados en la corteza):
      cada electrodo se sitúa donde su dirección de montaje corta el elipsoide del cerebro,
      con margen para que floten justo por encima (independiente de la quiralidad del modelo).
- [x] **Heatmap cortical** µ/β inyectado en el material vía `onBeforeCompile` (interpolación
      gaussiana por píxel sobre distancia angular en GLSL; solo se actualiza el uniform
      `uValues[]` por frame, sin recrear geometría).
- **Decidido (resuelto):** se **superpone** — la esfera translúcida queda como "scalp" y la
      malla real va dentro. Integrado en `components/Brain3D.tsx` (`<BrainMesh … />`).
- Pendiente opcional (no bloqueante): comprimir con Draco si en el futuro se usa un `.glb`
      más pesado (el actual no lo necesita).

---

## Inicio / Dashboard

- [x] **Página de Inicio — HECHO (primera mejora).** Añadidas secciones "Sobre el proyecto"
      (qué es, foco LTI, datos públicos, techo ~70–85 %) y "Los dos mundos" (offline/online),
      sobre el diagrama del pipeline y las métricas. Ref: `frontend/src/pages/Home.tsx`.
      **Ampliación HECHA:** añadidas la sección «El proyecto en tres etapas» (estado real de
      cada etapa: 1 pipeline LTI ✓, 2 frontend en progreso, 3 interoperabilidad pendiente) y
      una fila «Explora todas las secciones» con accesos a Dashboard, Cerebro 3D y Glosario
      (las que no tenían tarjeta propia en Inicio).
- [x] **Dashboard configurable — HECHO (2026-06-26).** Menú **«Añadir panel»** (componente
      `WidgetPicker`) que inserta cualquier panel del catálogo y botón **×** en la cabecera de
      cada panel para quitarlo; la selección se persiste (`localStorage` `dashboardWidgets-v1`)
      aparte de la disposición (`dashboardLayout-v5`). Se **extendió `GridBoard`** vía su
      reconciliación ya existente (no `react-grid-layout`). Refs: `pages/Dashboard.tsx`.
  - **Decidido (aplicado):** comportamiento **mixto según widget** — el catálogo etiqueta cada
    panel `live` (transmite con Play: señal cruda/filtrada, confianza, decisión, predicción,
    cerebro 3D) o estático (ficha del dataset, resumen sin streaming). El chip «en vivo/estático»
    se muestra en el menú de inserción.
  - **Resuelto (era abierto):** se eligió **catálogo fijo** (declarado en `Dashboard.tsx` como
    `CATALOG`), no registro automático desde cada página — más simple y sin acoplar las páginas a
    un registro global. Añadir un panel nuevo = una entrada en `CATALOG`.

## Modelo (antes "Entrenamiento")

> **[x] Renombrado (idea 6) — HECHO:** la **sección** lateral "Entrenamiento" pasa a
> llamarse **"Modelo"**, y la **página** "El Modelo" (`/csp`, `SpatialCSP`) pasa a llamarse
> **"Entrenamiento"**. Hecho en `lib/nav.ts`, título en `SpatialCSP.tsx`, y referencias
> de texto en `LiveStream.tsx` y `Brain3DPage.tsx`.

- [x] **Precisión por sujeto — HECHO.** La nueva sección Resultados (`pages/Results.tsx`,
      data-driven desde `/api/results`) muestra tabla ordenable + gráfico por sujeto con
      las 4 celdas (CSP/EEGNet × within/cross), kappa, nº trials, rango min–max y línea de
      azar, más una ficha de detalle por sujeto al hacer clic. Backend: `server/results.py`.
- [x] **Vista GENERAL (matriz agregada) — HECHO.** Matriz 2×2 (EEGNet vs CSP+LDA × within vs
      cross) agregando **por sujeto** sobre todos los datasets (un sujeto = un punto → pondera
      por nº de sujetos). Endpoint `/api/results_aggregate` (`server/results.py:aggregate_methods`),
      componente `AggregateMatrix` arriba de `pages/Results.tsx`. Incluye desglose por dataset
      (con badge de rol) para no esconder la heterogeneidad, y Wilcoxon pooled. Resultado actual:
      CSP within 0.624 vs EEGNet 0.547 (p=0.006, CSP mejor); cross 0.581 vs 0.561 (n.s.).
- [x] **Separar Resultados por ROL del dataset — DECIDIDO.** Resultados muestra los datasets
      de **entrenamiento** del modelo general **+** cualquier dataset `live` que también se haya
      usado para entrenar (tiene artefacto pooled/LOSO `d.pooled`). Regla en `Results.tsx`:
      `d.role === 'training' || d.pooled != null`. Así BCI IV 2a (role `live`) aparece porque
      tiene `model_BNCI2014_001_s0_eegnet_pooled` (LOSO 0.599); si se quitara ese pooled,
      desaparecería de Resultados y quedaría solo en «Demo en vivo» / Datasets. Badge de rol
      («demo en vivo · usado en pooled») en el `Overview` para no esconder la heterogeneidad.
- [x] **Leyenda de interpretación — HECHO.** Componente `components/ResultInterpretation.tsx`:
      leyenda colapsable en la cabecera de Resultados (`pages/Results.tsx`, tras `DatasetRolesNote`)
      que explica honestamente por qué varía la precisión (BCI illiteracy, calidad de señal/montaje,
      pocos trials, within vs cross, ausencia de clase «reposo») **sin inventar etiquetas** que el
      dataset no tiene. Enlaza términos al glosario vía `GlossaryText`.
  - **Decidido:** ver sección "Simulador de señal" — esto es la mitad "datos reales" de
    la decisión "ambas cosas" (la otra mitad, el simulador sintético, sigue pendiente).
- [x] **EEGNet: ficha de entrenamiento — HECHO.** `components/EEGNetModel.tsx` muestra ahora
      una ficha (nº de trials de entrenamiento, reservados para demo, banda de entrada, épocas,
      nº de filtros temporales F1, fecha) + tabla de precisión honesta EEGNet vs CSP+LDA del
      **mismo sujeto** (within-subject k-fold, inter-sesión y κ). `/api/eegnet` se amplió para
      devolver esos campos (`n_train`, `n_demo`, `kappa`, `trained_on`, `epochs`, `fir`,
      `n_temporal`, `csp_lda`); `train_eegnet_subject` guarda `epochs` en `extra`. Se eliminó el
      "CSP+LDA: 0.72 / 0.74" **hardcodeado** (no era un valor medido) por la cifra real del modelo.
- [x] **Comparación estadística accuracy vs κ — HECHO (2026-06-26).** El estadístico que mencionó
      el ingeniero ("k² o algo así") es **κ — kappa de Cohen** (ya se calcula y guarda en
      `ModelCard.kappa`). Implementado como panel `AccuracyKappaScatter` en la página Resultados
      (`pages/Results.tsx`): scatter accuracy (x) vs κ (y) con un punto por sujeto y los 4 regímenes
      (CSP+LDA / EEGNet × within / cross), recta de referencia **κ = 2·acc − 1** (relación esperada
      con clases balanceadas; puntos por debajo = accuracy inflada por desbalance/azar) y línea κ=0.
      Se integró como panel (no página aparte) para reusar los datos por-sujeto que ya servía
      `/api/results`. κ además ya se mostraba en la matriz 2×2, la tabla por sujeto y la matriz
      agregada.

## En vivo

- [x] **Reinicio de señal claro — HECHO.** Nuevo estado `ended` en el store; al terminar
      la reproducción (Laboratorio, sin loop) el botón pasa a **"Volver a iniciar"** con
      icono `RotateCcw`, y al pulsarlo reinicia desde el principio. Refs: `store/useStore.ts`,
      `components/layout/Sidebar.tsx`, `pages/SignalLab.tsx`.
      Nota: en `LiveStream` la transmisión corre en bucle en el servidor (no "termina" en
      cliente), así que ahí no aplica hasta el loader continuo (Plan B).
- [x] **Indicador de duración — HECHO (global).** Barra de progreso + `transcurrido / total s`
      en el **panel lateral** (sección Reproducción), visible desde cualquier sección; la página
      activa publica su progreso al store (`elapsedSec`/`totalSec`) y lo limpia al salir.
      **Solo informativo** (el programa no lo usa). Hoy lo alimenta el Laboratorio; en
      Clasificación se hará con el loader continuo (Plan B), donde la duración es desconocida.
- [ ] **Etiquetado de sujetos/datasets** en el panel lateral sin exponer el nombre crudo
      del dataset. Refs: `lib/datasets.ts`, `components/layout/Sidebar.tsx`, `REGISTRY` en
      `server/app.py`.
  - **Decidido:** **ambas** cosas — (a) etiquetas descriptivas/neutras en el selector
    (ej. "Señal A — buena calidad") y (b) una **ficha/minipágina por sujeto** con sus
    características (nº canales, fs, calidad, accuracy) para elegir bien en la demo.
- [x] **Hover en nodos del cerebro 3D — HECHO.** Al pasar el cursor sobre un electrodo se
      muestra un tooltip (`Html` de drei) con nombre + valor exacto de la desviación de
      potencia; el nodo se agranda al señalarlo. Refs: `components/Brain3D.tsx`,
      pista añadida en la ayuda de `pages/Brain3DPage.tsx`.
      Pendiente (mejora): cómo indicar mejor las *zonas* activas (no solo nodos puntuales).
- [ ] **Laboratorio: el filtro afecta una predicción.** Hoy el filtro ya cambia la señal
      filtrada dibujada (`SignalLab.tsx:111`) pero no hay clasificador en la página.
  - **Decidido:** añadir **clasificación en vivo** al Laboratorio que use la banda elegida,
    para ver cómo cambiar el filtro mejora/empeora la decisión.
  - **Abierto (técnico):** reentrenar CSP+LDA al vuelo con la banda elegida es caro. Decidir:
    ¿reentrenar bajo demanda en backend, precalcular un conjunto de bandas, o solo re-aplicar
    el clasificador existente sobre la señal re-filtrada? Definir antes de implementar.

## Glosario

- [ ] **Mejorar el glosario:** términos hoy difíciles de entender, redactar más claro.
      Fuente única: `docs/glosario.md` (servida por `/api/glossary`).
- [x] **Ligar el texto explicativo de cada sección al glosario — HECHO.** Nuevo componente
      reutilizable `components/GlossaryText.tsx`: detecta términos del glosario dentro de
      texto plano (alias derivados de los títulos `### …` vía `buildGlossaryMatcher` en
      `lib/glossary.ts`, matching tolerante sin acentos, longest-first, solo la 1ª aparición)
      y los enlaza in-situ con preview al hover. Aplicado al `intro` y a las descripciones de
      cada punto en `components/HelpButton.tsx` (antes solo se enlazaban los chips de
      «Términos clave»). El componente es genérico: sirve para la ampliación «detectar
      términos en CUALQUIER texto/sección» — basta envolver el texto en `<GlossaryText>`.
- [x] **Preview al hover — HECHO (términos clave).** Cache cliente del glosario
      (`lib/glossary.ts`: `useGlossary`, `findEntry`, `previewText`) + chips de "Términos
      clave" en los paneles de ayuda (`HelpButton.tsx`) que muestran la definición al pasar
      el cursor y enlazan al glosario al hacer clic. Matching tolerante (sin acentos/símbolos).
      Pendiente (ampliación): detectar términos en CUALQUIER texto/sección, no solo en los
      chips de ayuda.

## Simulador de señal ("creador de señal") — CANCELADO (2026-06-26)

> **CANCELADO (cambio de plan 2026-06-26):** se descarta construir el simulador sintético que
> degrada una señal real (sujeto "iletrado", cuero cabelludo grueso, mala colocación). Se
> conserva solo la *leyenda honesta sobre datos reales* ya hecha (ver sección Modelo,
> `components/ResultInterpretation.tsx`).

---

## Notas transversales

- El renombrado de la sección (Modelo/Entrenamiento) toca `lib/nav.ts` y títulos de página;
  revisar que los `WorldBadge`/`world` sigan correctos.
- Varias ideas comparten el patrón "ficha de detalle" (sujeto, modelo EEGNet): considerar un
  componente reutilizable.
- Mantener la disciplina de no fuga de datos en cualquier evaluación nueva (LOSO, por banda).
