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
- [ ] Exponer los resultados pooled/LOSO en el frontend (página de comparación estadística).
- Refs: `backend/src/bci/models/eegnet.py`, `pipeline/training.py`
  (`train_eegnet_subject`, `train_eegnet_pooled`), `docs/presentacion.md` ("Próximos pasos").
- **Abierto:** ¿EEGNet pooled llega a usarse en vivo, o sigue siendo solo comparación
  con el CSP+LDA sujeto-específico? (Hoy CLAUDE.md dice explícitamente que EEGNet NO
  se usa en inferencia en vivo — revisar esa decisión si se quiere demo cross-subject.)

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
- [~] **Paso 2 — Escalar sujetos con PhysionetMI (gran salto) — LISTO PARA EJECUTAR.** El
      script ya soporta el run grande: `--config ../configs/physionet.yaml`, los ~105 sujetos
      (excluyendo 88/89/92/100), `--augment` y `--loso-subset N` (LOSO sobre un subconjunto;
      modelo base con todos). **Proceso documentado en `docs/entrenamiento.md`** (comandos
      exactos para correr en terminal propia, sin gastar tokens). Expectativa: LOSO ~0.60 →
      ~0.65–0.70. Falta solo lanzarlo y registrar el resultado.
      **PhysioNet ES la vía recomendada para EEGNet pooled** (verificado 2026-06-21: S2 0.622
      con 11 suj, escala con datos).
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
- [~] **Paso 3 — Pooling entre datasets — IMPLEMENTADO (falta lanzar el run grande).** Cubre
      temas de **Sistemas Lineales y Señales**: remuestreo racional L/M a un fs común (sobremuestreo
      + filtro FIR anti-aliasing + diezmado) y armonización de montajes (19 canales motores comunes).
      Hecho: `bci/dsp/resampling.py` (`resample_lti`, verificado corr=1.0 vs scipy y anti-aliasing OK),
      `design_lowpass_fir` en `fir_filters.py`, y `scripts/train_eegnet_crossdataset.py` (LODO =
      leave-one-dataset-out). **Proceso y parámetros documentados para la presentación en
      `docs/remuestreo.md`.** Falta lanzar el entrenamiento grande y registrar resultados.
      **Pool propuesto y criterios en `docs/datasets.md`**: PhysioNet (109) + **Dreyer2023 (87)**
      + **Cho2017 (52)**, ambos ya integrados ≈ 248 sujetos. Lee2019_MI se **descartó** (pesaba
      ~1.2 GB/sujeto y MOABB 1.5.0 solo exponía 1 de sus 2 sesiones — datasets.md §3.1).
      Para EN VIVO (calibrar≠probar): 2a + Stieger2021 (11 sesiones, pendiente).

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
- [ ] **B.3 — Widget "muñeco de brazos".** Persona con dos brazos que reacciona a la
      clasificación en vivo (mueve brazo izq./der. según la predicción). Es la salida
      intuitiva de la demo continua.
- **Abierto:** ¿El widget de brazos vive en Clasificación (`/live`), en su propia página,
  o como widget insertable en el Dashboard (ver sección Dashboard)?

## C. Cerebro 3D anatómico (GLB)

El "cerebro 3D" actual (`components/Brain3D.tsx`) es una **esfera procedural** translúcida
con electrodos; no es una malla anatómica y no carga ningún modelo.

- [ ] Conseguir un `.glb` de cerebro/cabeza de baja-media densidad con licencia clara
      (CC0/CC-BY). Revisar atribución antes de meterlo al repo.
- [ ] Servirlo desde `frontend/public/models/` y cargarlo con `useGLTF` de
      `@react-three/drei` (ya es dependencia) en un componente nuevo `BrainMesh.tsx`.
- [ ] Reconciliar posición de electrodos (hoy `s = 0.96 / maxNorm` asume esfera unitaria).
      Opción simple: mantener la esfera translúcida como "scalp" y meter el cerebro real
      dentro; opción fiel: reescalar a la bounding box de la malla.
- [ ] Comprimir el `.glb` (Draco) si pesa, para no inflar el bundle.
- **Abierto:** ¿reemplaza la esfera actual o se superpone? (por defecto: superpuesto).

---

## Inicio / Dashboard

- [x] **Página de Inicio — HECHO (primera mejora).** Añadidas secciones "Sobre el proyecto"
      (qué es, foco LTI, datos públicos, techo ~70–85 %) y "Los dos mundos" (offline/online),
      sobre el diagrama del pipeline y las métricas. Ref: `frontend/src/pages/Home.tsx`.
      Pendiente (ampliación): estado de las 3 etapas y accesos a las secciones restantes.
- [ ] **Dashboard configurable:** menú para **insertar/quitar** gráficos de cualquier
      sección (cerebro 3D, clasificación, pipeline, etc.). Extender `GridBoard` (no usar
      `react-grid-layout`). Refs: `pages/Dashboard.tsx`, `components/GridBoard.tsx`.
  - **Decidido:** comportamiento **mixto según widget** — los que ya transmiten
    (cerebro 3D, predicción, pipeline) van en vivo cuando se pulsa Play; el resto como
    resumen/snapshot estático.
  - **Abierto:** ¿catálogo fijo de widgets disponibles, o registro automático desde cada
    página?

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
- [ ] **Leyenda de interpretación** del resultado (en cabecera o en el botón de info):
      explicar honestamente por qué varía la precisión (BCI illiteracy, calidad de señal,
      pocos trials), **sin inventar etiquetas** que el dataset no tiene.
  - **Decidido:** ver sección "Simulador de señal" — esto es la mitad "datos reales" de
    la decisión "ambas cosas".
- [x] **EEGNet: ficha de entrenamiento — HECHO.** `components/EEGNetModel.tsx` muestra ahora
      una ficha (nº de trials de entrenamiento, reservados para demo, banda de entrada, épocas,
      nº de filtros temporales F1, fecha) + tabla de precisión honesta EEGNet vs CSP+LDA del
      **mismo sujeto** (within-subject k-fold, inter-sesión y κ). `/api/eegnet` se amplió para
      devolver esos campos (`n_train`, `n_demo`, `kappa`, `trained_on`, `epochs`, `fir`,
      `n_temporal`, `csp_lda`); `train_eegnet_subject` guarda `epochs` en `extra`. Se eliminó el
      "CSP+LDA: 0.72 / 0.74" **hardcodeado** (no era un valor medido) por la cifra real del modelo.
- [ ] **Página de comparación estadística.** El estadístico que mencionó el ingeniero
      ("k² o algo así") es **κ — kappa de Cohen** (ya se calcula y guarda en `ModelCard.kappa`).
      Mostrar comparación accuracy vs kappa entre sujetos/datasets/métodos (CSP+LDA vs EEGNet).

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

## Simulador de señal ("creador de señal")

> **Decidido (decisión "ambas cosas"):** además de la *leyenda honesta sobre datos reales*
> (ver sección Modelo), construir un **simulador sintético** explícitamente didáctico/ficticio.

- [ ] Herramienta que **degrada una señal real** para simular condiciones difíciles:
      sujeto "iletrado" (ERD débil → atenuar la banda µ/β), cuero cabelludo grueso
      (atenuación + más ruido), mala colocación de electrodos, etc.
- [ ] Marcarlo **claramente como simulación**, no como datos reales del dataset.
- [ ] Mostrar el efecto en cadena: cómo la degradación cambia la señal filtrada, las
      features CSP y la predicción (conecta con "Laboratorio: el filtro afecta una predicción").
- **Abierto:** ¿página propia o modo dentro del Laboratorio? ¿Qué parámetros de degradación
  se exponen al usuario?

---

## Notas transversales

- El renombrado de la sección (Modelo/Entrenamiento) toca `lib/nav.ts` y títulos de página;
  revisar que los `WorldBadge`/`world` sigan correctos.
- Varias ideas comparten el patrón "ficha de detalle" (sujeto, modelo EEGNet): considerar un
  componente reutilizable.
- Mantener la disciplina de no fuga de datos en cualquier evaluación nueva (LOSO, por banda).
