# 8 · Frontend — la SPA didáctica

> La Etapa 2: una *single-page app* en React que **expone explícitamente** cada etapa del pipeline
> (convolución, pesos CSP, frontera LDA, cerebro 3D) en vez de esconderla. Doble propósito: defender
> el proyecto (narrativa guiada) y experimentar (jugar con datos y parámetros). Código: `frontend/`.

---

## 8.1 Stack y por qué cada pieza

| Necesidad | Librería | Por qué |
|---|---|---|
| Build / SPA | **Vite + React 19 + TypeScript** | rápido; TS tipa los datos del backend |
| Routing | **react-router-dom** | navegación entre secciones |
| Estado global | **Zustand** | ligero; el "controlador maestro" del sidebar |
| Estilos | **Tailwind v4** | tema claro consistente sin CSS a mano |
| Señal en vivo | **uPlot** | rapidísimo con streaming (`setData` imperativo, sin re-render) |
| Gráficos analíticos | **Recharts** | declarativo (\|H(e^jω)\|, barras, scatter) |
| 3D | **three.js + @react-three/fiber + drei** | cerebro 3D y electrodos |
| Animación / iconos | **Framer Motion** · **lucide-react** | transiciones y iconografía |

Sin runner de tests en el frontend: la verificación es `npm run build` (typecheck `tsc -b`) + uso
manual del dev server (`npm run dev`, que **proxya** `/api` y `/ws` a FastAPI en `:8000`).

> **Decisión técnica: GridBoard casero, NO `react-grid-layout`.** Se evaluó `react-grid-layout` pero
> su v2.x no engancha bien drag/resize bajo React 19. Se construyó un grid propio
> (`components/GridBoard.tsx`): 12 columnas, *snapping*, fondo punteado, layout por página
> **persistido a `localStorage`**. Para añadir un panel se extiende `GridBoard`, no se reintroduce la
> librería. (El `frontend-design.md` antiguo aún la mencionaba; este documento lo sustituye.)

---

## 8.2 Arquitectura

- **Estado global** (`store/useStore.ts`, Zustand): dataset/sujeto/canal seleccionados, play/pausa/
  loop, sidebar, estado del sistema, latencia, buffer de log, color de acento. El **sidebar**
  (`components/layout/Sidebar.tsx`) es el "controlador maestro" que escribe en el store; las páginas
  leen de él, pero Play/Pausa/Loop solo afectan a los widgets de la página actual.
- **Cliente API** (`api/client.ts`): REST fino (`getJSON`) + WebSocket (`openStream`). En dev, Vite
  proxya; no hace falta configurar URL base.
- **Routing** (`App.tsx`): todo cuelga de `AppLayout` (nav superior + sidebar plegable).

---

## 8.3 Los dos mundos en la UI

La distinción **offline/online** (sección 0) se mantiene **visible** (`lib/nav.ts`, `WorldBadge`):

| Grupo | Mundo | Color | Páginas |
|---|---|---|---|
| General | — | gris | Inicio, Dashboard, Glosario |
| **Modelo** | offline | ámbar | Entrenamiento (`/csp`), Resultados (`/results`) |
| **En vivo** | online | verde | Laboratorio (`/lab`), Clasificación (`/live`), Benchmark (`/demo`), Cerebro 3D (`/brain`) |

Cada página nueva debe etiquetarse correctamente en `NAV_GROUPS` para no romper esta narrativa.

---

## 8.4 Las páginas (qué materializa cada una)

| Página | Sección | Qué muestra |
|---|---|---|
| **Inicio** | 0 | hero que decodifica BCI·MI, accesos a los dos mundos |
| **Dashboard** | varias | grid libre: el usuario compone sus widgets (señal, muñeco, cerebro, confianza, trials recientes…) |
| **Laboratorio** (`/lab`) | 2 | FIR en vivo: `h[n]`, `\|H\|`, crudo-vs-filtrado (convolución **en cliente**) |
| **Entrenamiento** (`/csp`) | 3–5 | CSP (topomapas, `Z=W·X`), log-varianza, LDA (frontera), pestaña EEGNet |
| **Resultados** (`/results`) | 6 | benchmark de población: 4 regímenes, métricas, Wilcoxon, Gini |
| **Benchmark** (`/demo`) | 6 | matriz de confusión por modelo/régimen (`/api/eval`) |
| **Clasificación** (`/live`) | 7 | el pipeline completo en vivo, 3 etapas explícitas + decisión continua |
| **Cerebro 3D** (`/brain`) | 7 | potencia µ/β por electrodo iluminando el cuero cabelludo |
| **Glosario** | — | términos LTI/BCI, servido por `/api/glossary` desde `docs/glosario.md` |

---

## 8.5 Patrones de rendimiento (lo que hace fluida la app)

La señal en vivo llega a ~10 Hz; re-renderizar React a esa frecuencia sería lento. Dos patrones
clave:

- **uPlot imperativo** (`FillChart` + `UPlotChart`): `FillChart` mide el alto disponible con
  `ResizeObserver` y se lo pasa a uPlot, que redimensiona con `setSize` (sin recrear la instancia);
  los datos entran por `setData`. Cero re-render de React por frame.
- **SVG imperativo en `PipelineStages.tsx`**: las nubes de entrenamiento (CSP/LDA) se dibujan **una
  vez**; el punto en vivo se mueve por *refs*/`useImperativeHandle`. Misma filosofía que `setData`:
  evitar el ciclo de render a frecuencia de streaming.
- **Canvas hooks de uPlot** (`lib/chartBands.ts`, `drawActiveBands`): la **banda verde** que marca la
  franja de imaginación activa se pinta en el hook `drawClear`, sobre el canvas, sin componentes.

---

## 8.6 El sistema de widgets (GridBoard)

`GridBoard` se usa uniforme en Dashboard, Laboratorio, Clasificación, Entrenamiento y Resultados.
Cada panel es un `GridWidget` (`{ i, title, accent, w, h, minW, minH, actions?, el }`). Al
añadir/quitar widgets, el layout se reconcilia con colocación *first-fit* para los nuevos,
preservando las posiciones existentes; todo se persiste por página en `localStorage` (con una
`storageKey` versionada — al cambiar el conjunto de widgets se sube la versión para no heredar
layouts obsoletos).

El **Dashboard** es el grid "libre": el usuario elige del catálogo (`CATALOG`) qué widgets montar
(señal cruda/filtrada, muñeco `HandPuppet`, cerebro, traza de confianza, trials recientes,
lateralización…), reutilizando los mismos componentes que las páginas guiadas.

---

## 8.7 Detalles transversales

- **Ayuda contextual**: cada página tiene un botón `?` (`HelpButton`) con la explicación teórica
  concisa, **auto-enlazada** al Glosario (`GlossaryText` detecta los términos).
- **Glosario como única fuente**: el contenido vive en `docs/glosario.md` y lo sirve `/api/glossary`
  (no se duplica en el frontend).
- **Portabilidad sin datos crudos**: gracias a los payloads precomputados (sección 9), todas las
  páginas *offline* renderizan **sin** datos crudos; estos solo hacen falta para el `/ws/stream` de
  la demo.

---

**Siguiente:** [9 · Scripts y uso](09-scripts-y-uso.md) — cómo correr todo (backend, datos,
entrenamiento, figuras, servidor, frontend) de principio a fin.
