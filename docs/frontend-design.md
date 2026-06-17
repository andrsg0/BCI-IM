# Diseño del Frontend — Interfaz BCI (Etapa 2)

Documento de referencia de la interfaz web. Consolida la arquitectura, las páginas,
el sistema de widgets, el stack de librerías y los detalles de interacción acordados.

## 1. Visión

SPA **React** didáctica e interactiva, con doble propósito: **exponer** el proyecto en la
defensa (narrativa guiada) y **experimentar** (el usuario juega con datos y parámetros).
Conecta con el pipeline Python vía **REST + WebSocket** (reusa todo el backend de la Etapa 1).

**Estética:** clara científica (tipo paper / dashboard clínico). Fondo claro, azules sobrios,
mucho espacio en blanco, tipografía limpia, tarjetas con sombra suave. Máxima legibilidad para
proyectar.

## 2. Stack de librerías (decidido)

| Necesidad | Librería | Por qué |
|---|---|---|
| Build / SPA | **Vite + React + TypeScript** | rápido; TS tipa los datos del backend y evita errores |
| Routing | **react-router-dom** | navegación entre secciones |
| Estado global | **Zustand** | ligero; ideal para el "controlador maestro" del sidebar |
| Estilos | **Tailwind CSS** | tema claro consistente sin escribir mucho CSS |
| Dashboard de widgets | **react-grid-layout** | rejilla con *snapping* tipo rompecabezas, arrastrar/añadir/quitar |
| Drag & drop fino | **dnd-kit** (si hace falta) | catálogo de widgets, accesible y moderno |
| Series de tiempo en vivo | **uPlot** | ultrarrápido con miles de puntos y streaming (señal EEG) |
| Gráficas analíticas | **Recharts** | declarativo y bonito (\|H(e^jω)\|, barras de métricas, scatter log-var) |
| 3D | **three.js + @react-three/fiber + @react-three/drei** | cerebro 3D y nodos de electrodos |
| Animaciones | **Framer Motion** | transiciones suaves, "fantasma" de la convolución, widgets |
| Iconos | **lucide-react** | iconografía limpia y coherente |

> Sobre "gráficas bonitas": no hay que dibujarlas a mano. **uPlot** (rendimiento, señales en
> vivo) + **Recharts** (estética declarativa, gráficas analíticas), ambas estilizadas con los
> tokens de color del tema, dan un resultado pulido y coherente. Alternativa de máxima
> personalización: **visx** (primitivas D3 para React), a cambio de más trabajo manual.

## 3. Layout base

```
┌──────────────────────────────────────────────────────────────────────┐
│ BCI·MI   Inicio · Dashboard · Laboratorio · CSP · Cerebro 3D · Result. · Glosario │ ← nav superior
├──────────────┬───────────────────────────────────────────────────────┤
│  SIDEBAR     │                                                         │
│ (persistente │            ÁREA PRINCIPAL                               │
│  ocultable)  │       (página / lienzo de widgets)                      │
│              │                                                         │
│  controlador │                                                         │
│  maestro     │                                                         │
└──────────────┴───────────────────────────────────────────────────────┘
```

- **Nav superior:** secciones. El item activo se resalta.
- **Sidebar persistente y ocultable** (botón colapsar): el **controlador maestro** (sección 4).

## 4. Sidebar — controlador maestro (estado global)

Controla el estado que comparten todas las páginas (en Zustand):
- **Selector de Dataset** (BCI IV 2a / PhysioNet / Liu2024).
- **Selector de Sujeto** (según dataset).
- **Selector de canal EEG** (p. ej. C3, C4, Cz…).
- **Controles de reproducción:** Play · Pause · Loop — simulan el streaming de datos.
- **Botón Limpiar vistas** (resetea las gráficas; si no se pulsa, los resultados **permanecen**
  en pantalla para examinarlos).
- **Área de Logs del sistema** (parte inferior): mensajes tipo consola.
- **Monitor de latencia (simulada)** e **indicador de Estado del Sistema**
  (conectado al dataset, buffer lleno, streaming activo…).
- Enlace **"Acerca de"**.

**Regla de reproducción:** Play/Pause/Loop actúan **solo sobre la página actual y sus widgets
actuales**, no sobre toda la app.

## 5. Sistema de Widgets ("pizarrón / rompecabezas")

- Las vistas se componen de **widgets** rectangulares con **bordes redondeados** y una **franja
  de color de acento** (en el título o lateral) que los hace distintivos.
- Se comportan como **piezas de rompecabezas**: se arrastran y **encajan** en una rejilla
  (snapping), sin solapamientos. **No** necesitan ser redimensionables.
- El usuario **reordena, añade y elimina** widgets eligiendo de un **catálogo predefinido**.
- Implementación: **react-grid-layout** (rejilla + drag + compactación tipo rompecabezas).

**Catálogo de widgets (inventario inicial):**
`SeñalCruda x[n]` · `SeñalFiltrada y[n]` · `ConvoluciónMAC` · `RespuestaFrecuencia |H|` ·
`PatronesCSP (topomapa/pesos)` · `Varianza CSP` · `Cerebro3D` · `Predicción+Confianza` ·
`Métricas (acc/kappa)` · `ComparativaDatasets`.

## 6. Páginas (organizadas en dos mundos: entrenar vs transmitir)

La navegación superior separa visualmente los **dos mundos** del sistema (ver
`presentacion.md` → "Los dos mundos"), con un encabezado y un color por grupo, y cada
página muestra un **distintivo** (`WorldBadge`) que recuerda en cuál está:

- 🟠 **Entrenamiento** (`offline`, ámbar — *"cálculo previo al streaming"*): **El Modelo**
  (CSP), **Cerebro 3D** (pesos del CSP entrenado), **Resultados**.
- 🟢 **En vivo** (`online`, verde — *"en tiempo real, con el modelo ya entrenado"*):
  **Laboratorio**, **Clasificación**.
- Transversales: **Inicio**, **Glosario**.

La sección **Clasificación** muestra además la **ficha del modelo** (`GET /api/model`):
con qué partición se entrenó, qué held-out se reserva para la demo y la accuracy honesta.
Implementación: `lib/nav.ts` (`NAV_GROUPS`, `WORLD_STYLE`), `components/WorldBadge.tsx`,
`components/PageShell.tsx` (prop `world`), `components/layout/TopNav.tsx`.

### Dashboard libre — cockpit del pipeline EN VIVO (implementado)

`pages/Dashboard.tsx` es un **cuadrícula propia** (drag + resize con snap, sin dependencias) cuyos
paneles **siguen el pipeline en tiempo real sobre la misma señal**: señal cruda → señal filtrada
(µ/β causal) → confianza del clasificador en el tiempo → decisión por trial (voto suave). Mundo
`online`. Todos los paneles se alimentan de **un único WebSocket** (`/ws/stream`) mediante un
**bus de suscripción** (contexto `LiveCtx`): el WS publica cada mensaje y cada widget se suscribe y
actualiza su gráfico de forma **imperativa** (`uPlot.setData`) para rendir a ~10 Hz sin re-render
global. Para ello el stream se enriqueció: además de `pred`/`probs` envía `power` (potencia por
canal) y la señal `raw`/`filt` de un canal de referencia (`?channel=`).

Cuadrícula: drag desde la cabecera, resize desde la esquina inferior derecha, **snap a celdas**
(12 col), **fondo de puntos**, disposición **persistida en localStorage**, botón "Restablecer".

> Nota técnica: se descartó `react-grid-layout` (su v2.x no enganchaba drag/resize bajo React 19).
> La cuadrícula propia mide el ancho con `ResizeObserver` y actualiza x/y/w/h en unidades de celda
> durante el arrastre, con `pointermove`/`pointerup` en `window`.

### Cuadrícula reutilizable en TODAS las secciones — `components/GridBoard.tsx` (implementado)

La mecánica del Dashboard se **extrajo a un componente genérico** `GridBoard` para que **cada
sección** tenga el mismo modo "pizarrón": arrastrar paneles desde la cabecera, redimensionar desde
la esquina, **snap** a una rejilla de 12 columnas sobre **fondo de puntos**, disposición **persistida
en localStorage** (una clave por página) y botón "Restablecer disposición".

- API: `<GridBoard widgets={GridWidget[]} storageKey="..." toolbar={…} />`. Cada `GridWidget` define
  `{ i, title, accent, w, h, minW, minH, actions?, el }` (tamaño/mínimos en celdas).
- **Soporta widgets dinámicos**: al añadir/quitar paneles (p. ej. "Añadir gráfica" en el Laboratorio)
  el tablero **reconcilia** la disposición — conserva los existentes y coloca los nuevos en el primer
  hueco libre (`firstFit`).
- Gráficos **rellenan el panel**: `components/charts/FillChart.tsx` mide la altura disponible con
  `ResizeObserver` y la pasa a `UPlotChart`, que ahora **aplica el alto con `setSize`** (sin recrear
  la instancia) para que el redimensionado sea suave y no rompa los gráficos en vivo.
- Páginas migradas: **Dashboard, Laboratorio (SignalLab), Clasificación (LiveStream), El Modelo
  (CSP) y Resultados**. Cada una mantiene su lógica; solo cambia la disposición a la cuadrícula.

### El recorrido CSP → LDA EN VIVO — `components/charts/PipelineStages.tsx` (implementado)

La sección **Clasificación en vivo** dejaba la impresión de que la señal filtrada "se clasifica
sin más". Ahora muestra el **recorrido de cada ventana** como tres etapas LTI **bien diferenciadas**,
para que se vea que entre el filtrado y la decisión hay dos sistemas lineales distintos:

1. **Señal filtrada que entra** (cian/FIR): traza deslizante del canal de referencia ya filtrado
   (FIR causal). Es la **entrada** a la siguiente etapa.
2. **CSP · filtrado espacial** (morado): cada ventana se proyecta `Z = W·X` y se resume en su
   **log-varianza** → un **punto** en el espacio de características (comp 0 vs comp último). De fondo,
   **quieta**, la nube de ENTRENAMIENTO; encima, **moviéndose**, el punto de la ventana actual.
3. **LDA · frontera de decisión** (verde): ese vector se **proyecta sobre la recta discriminante**
   `δ₁−δ₀`; la **frontera** está en 0. De qué lado cae el punto es la clase; la distancia, la confianza.

Puntos clave de implementación:
- Las dos vistas (`CSPSpaceLive`, `LDAAxisLive`) son **SVG imperativos**: el fondo (nube de
  entrenamiento) se renderiza una vez y el **punto en vivo se mueve por refs** (`useImperativeHandle`),
  sin re-render de React a 10 Hz (mismo criterio que `uPlot.setData`). Van envueltas en `memo`.
- El backend añade por ventana, en `classify_window` → simulador → WebSocket, el vector `feat`
  (log-varianza CSP) y `disc` (proyección discriminante con signo). `GET /api/csp` se **cachea** y
  ahora devuelve también `lda_disc` (proyección de los trials de entrenamiento) como **fondo** del eje.
- Refuerza la conexión con **«El Modelo»** (mundo offline): mismo scatter y mismos ejes, ya
  renombrados con significado (`comp 0 · favorece <clase> (λ=…)`), donde se ve el modelo FIJO; aquí se
  ve su **aplicación en vivo** sobre esa misma nube.

### Cerebro 3D EN VIVO (implementado)

`pages/Brain3DPage.tsx` pasó del mundo offline (pesos fijos del CSP) al mundo **online**: ahora la
cabeza/electrodos se **iluminan con la señal en vivo**. El stream envía la **potencia µ/β por
canal** (`power`) y la página la centra (resta la media) y la suaviza (EMA) para colorear los
electrodos; el brillo crece con la desviación, de modo que la **lateralización C3/C4** (la ERD de
la imaginación motora) “se enciende” durante la demo. Las posiciones vienen de `/api/positions`
(ligero, sin entrenar modelo). Muestra además la predicción/confianza actual.

### EEGNet como ESPEJO (no como segundo pipeline)

Decisión de alcance: EEGNet no tiene un selector global ni paridad de funciones en toda la app
(sería sobre-ingeniería para el objetivo). Vive como **una pestaña local dentro de "El Modelo"**
(`tab: 'csp' | 'eegnet'` en `pages/SpatialCSP.tsx`) que **compara lado a lado** lo diseñado a mano
con lo aprendido:

- **temporal**: tu FIR µ/β vs los filtros de la conv temporal de EEGNet (|H| con la banda µ/β
  sombreada).
- **espacial**: tus patrones CSP vs los filtros de la conv depthwise (topomapas).

Implementación: `components/EEGNetModel.tsx` (recibe el CSP ya cargado por la página y pide
`GET /api/eegnet` + `GET /api/filter` para el FIR de referencia). El resto de la app (Inicio,
Cerebro 3D, Laboratorio, Clasificación) sigue siempre con CSP+LDA. **Descartada** la inferencia
EEGNet en vivo.

### Detalle por página (orden = viaje teoría → práctica)

1. **Inicio (El Mapa).** Título + **diagrama de bloques interactivo e iluminado** del pipeline:
   `Adquisición → Filtro FIR → CSP → EEGNet (futuro)`. Cada bloque es **clicable** y lleva a su
   vista (p. ej. clic en "Filtro FIR" → Laboratorio de Señales). Explicación matemática resumida
   y **métricas clave** de precisión.
2. **Dashboard (Modo Libre).** Lienzo donde el usuario **arrastra y ancla** widgets a su gusto
   para un panorama global de **solo lectura**.
3. **Laboratorio de Señales (LTI & CSP — El Motor).** La señal temporal **x[n]** y su
   convolución **y[n]** lado a lado (demostración del sistema LTI: antes/después). La **respuesta
   en frecuencia (FFT/|H|)** como **pestaña o widget lateral**, para mostrar que se aisló el ritmo
   µ. Permite ver cómo el **CSP** procesa las señales.
4. **Filtrado Espacial (CSP).** Puente entre la señal temporal y la ubicación física: matrices /
   pesos espaciales del CSP, enlazado visualmente con el Cerebro 3D.
5. **Cerebro 3D (La Salida).** Modelo cerebral **transparente** que **ilumina nodos** en las
   posiciones de los electrodos del **sistema 10-20**, según los **pesos del CSP** / la predicción.
6. **Resultados.** Comparativa académica de cómo rinde el pipeline LTI entre datasets
   (PhysioNet vs BCI IV 2a vs Liu2024): accuracy por sujeto, media±std, el hallazgo de la
   regularización.
7. **Glosario / Teoría.** Diccionario interactivo de términos (desde `docs/glosario.md`).

## 7. Detalles de interacción

- **El "fantasma" de la convolución:** en la vista del filtro FIR, un **rectángulo translúcido**
  se desliza sobre la gráfica de x[n], ilustrando la **ventana de convolución** moviéndose en el
  tiempo (la operación MAC en cada posición).
- **Sincronización milimétrica:** el **cabezal de reproducción** (línea vertical) en la gráfica
  de la señal está **perfectamente sincronizado** con la iluminación de los electrodos en el
  Cerebro 3D y con la predicción.
- **Flujo guiado→libre:** al entrar, el **diagrama del pipeline** es el ancla (narrativa). Tras
  entender cada paso, el usuario va al **Dashboard** a montar su propia consola.

## 8. Comunicación con el backend (API)

Servidor **FastAPI** (módulo `backend/src/bci/server/`):
- **REST:** metadatos de datasets/sujetos, coeficientes del FIR + respuesta en frecuencia,
  patrones CSP, métricas/resultados, posiciones de electrodos 10-20.
- **WebSocket:** stream de la simulación en vivo (ventana deslizante → predicción + confianza),
  reusando `StreamSimulator`. Listo para sustituir la fuente por LSL (Ultracortex) en la Etapa 3.

## 9. Hitos de implementación (Etapa 2)

1. **Scaffolding** — Vite+React+TS, Tailwind, routing, Zustand, layout (nav + sidebar), tema.
2. **Backend API** — FastAPI con endpoints REST + WebSocket sobre el pipeline existente.
3. **Sistema de widgets** — react-grid-layout, catálogo, añadir/quitar/reordenar.
4. **Laboratorio de Señales** — x[n]/y[n], fantasma de convolución, |H|.
5. **CSP + Cerebro 3D** — pesos espaciales y cerebro reactivo sincronizado.
6. **En vivo (WebSocket)** + **Resultados** + **Inicio** (diagrama) + **Glosario**.
