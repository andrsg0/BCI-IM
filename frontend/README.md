# Frontend — Interfaz didáctica BCI (Etapa 2)

SPA **React + TypeScript + Vite** con estética clara científica. Diseño completo y
decisiones en [`../docs/informe/08-frontend.md`](../docs/informe/08-frontend.md).

## Stack
React 19 · Vite · TypeScript · Tailwind v4 · react-router · **Zustand** (estado global) ·
**react-grid-layout** (widgets) · **uPlot** (señal en vivo) · **Recharts** (gráficas) ·
**three.js + react-three-fiber** (cerebro 3D) · Framer Motion · lucide-react.

## Arranque
```bash
npm install
npm run dev      # http://localhost:5173 (proxy a backend FastAPI en :8000)
npm run build    # typecheck + build de producción
```

## Estructura
```
src/
├── components/       Widget (tarjeta con franja de acento), PageShell
│   └── layout/       TopNav, Sidebar (controlador maestro), AppLayout
├── pages/            Home, Dashboard, SignalLab, SpatialCSP, Brain3DPage, Results, Glossary
├── store/            useStore.ts (Zustand: dataset, sujeto, canal, play/pause, logs…)
├── api/              client.ts (REST + WebSocket al backend)
└── lib/              datasets.ts, nav.ts
```

> Etapa 2 en curso. Hito 2.1 (scaffolding + layout + navegación) ✅.
