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
      (~0.72 k-fold), lo que motiva el fine-tuning con calibración corta (siguiente paso).
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

|            | within-subject (k-fold) | cross-subject (LOSO) |
|------------|-------------------------|----------------------|
| **CSP+LDA**| ~0.72 (ya medido)       | por medir (se espera ≈ azar: CSP es sujeto-específico) |
| **EEGNet** | ~0.72 (ya medido)       | **0.599** (medido, 2a 9 sujetos) |

- within-subject = entrenas y evalúas en el MISMO sujeto (calibrado).
- cross-subject (LOSO) = entrenas con los DEMÁS y evalúas en el excluido (usuario nuevo
  sin calibrar). El contraste CSP+LDA cross (rígido, ≈ azar) vs EEGNet cross (generaliza
  algo) es el punto didáctico clave.
- [ ] Script `scripts/compare_methods.py` que produce la matriz 2×2 + tabla por sujeto +
      CSV, todo con los mismos folds. Mostrarla luego en el frontend (idea "comparación
      estadística", con κ además de accuracy).

### Plan de datos (para subir el rendimiento de EEGNet)

> **Realidad:** el techo del estado del arte de imaginación motora 2 clases no invasiva es
> ~70–85 %. Objetivos honestos: cross-subject **sin calibrar** ≈ 0.65–0.70 (no 0.80);
> por usuario **con calibración/fine-tuning** ≈ 0.75–0.80 (0.80–0.85 en buenos sujetos);
> within-subject ya ~0.72 de media. Siempre reportar el **rango entre sujetos**, no solo la media.

- [ ] **Paso 1 — Aumentación de datos (gratis, primero).** Sin datos nuevos: ventanas
      deslizantes desplazadas (3–5 recortes por trial dentro del epoch), ruido gaussiano
      leve, *mixup*, dropout de canales. Sube algo el cross y bastante el within.
- [ ] **Paso 2 — Escalar sujetos con PhysionetMI (gran salto).** Reentrenar el pooled con
      los ~109 sujetos de PhysioNet (vs 9 de 2a): lo que más mueve la aguja en cross-subject
      es MÁS SUJETOS, no más trials. Expectativa: LOSO ~0.60 → ~0.65–0.70.
      **Cómputo:** LOSO sobre un subconjunto (p. ej. 30–40 sujetos) y modelo base con los 109,
      para no esperar horas. (decisión pendiente de confirmar)
- [ ] **Paso 4 — Fine-tuning con calibración corta.** Congelar capas tempranas del modelo
      base pooled y adaptar las últimas con 20–40 trials del usuario nuevo (1–2 runs). Es el
      paso que de verdad lleva a un usuario concreto hacia 0.75–0.85.
- [ ] **Paso 3 — Pooling entre datasets (2a + PhysioNet + Liu) — AL FINAL.** Incluido a
      petición del usuario porque cubre temas de su clase de **Sistemas Lineales y Señales**:
      remuestreo a un fs común (interpolación/diezmado + filtro anti-aliasing), armonización
      de montajes (subconjunto de canales motores común), y el efecto del filtrado FIR µ/β
      sobre señales de distinta fs. Más variedad de datos, pero requiere esa ingeniería de
      acondicionamiento de señal.

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

- [ ] **Precisión por sujeto** (no solo media k-fold por dataset). Mostrar la puntuación
      de cada individuo en Resultados o en la página de Entrenamiento.
      Ref actual: `pages/Results.tsx` solo muestra accuracy por dataset.
- [ ] **Leyenda de interpretación** del resultado (en cabecera o en el botón de info):
      explicar honestamente por qué varía la precisión (BCI illiteracy, calidad de señal,
      pocos trials), **sin inventar etiquetas** que el dataset no tiene.
  - **Decidido:** ver sección "Simulador de señal" — esto es la mitad "datos reales" de
    la decisión "ambas cosas".
- [ ] **EEGNet: ficha de entrenamiento.** Mostrar cómo se entrenó el modelo (nº de trials,
      banda, épocas, accuracy inter-sesión vs k-fold). Los datos ya están en la `ModelCard`
      (`training.py:ModelCard.extra` ya guarda `accuracy_intersession`, `accuracy_kfold`,
      `folds`, `viz_trained_on`). Falta exponerlos/mostrarlos en `components/EEGNetModel.tsx`.
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
- [ ] **Ligar el texto explicativo de cada sección al glosario.** Hoy los paneles de ayuda
      (`HelpContent.terms`) ya enlazan términos al glosario; extender esos enlaces al cuerpo
      del texto explicativo, no solo a la lista de términos. Refs: `components/HelpButton.tsx`,
      `pages/Glossary.tsx` (ya soporta `?q=` para abrir filtrado).
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
