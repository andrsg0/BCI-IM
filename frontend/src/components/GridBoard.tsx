import {
  useCallback, useEffect, useLayoutEffect, useRef, useState,
  type ReactNode, type PointerEvent as ReactPointerEvent,
} from 'react'
import { GripVertical, RotateCcw } from 'lucide-react'
import type { Accent } from './Widget'

// ---- cuadrícula propia (drag + resize con snap, sin dependencias externas) ----
// react-grid-layout v2 no funciona en React 19, así que implementamos el grid
// con eventos de puntero. Es la misma mecánica del Dashboard, ahora reutilizable
// por todas las secciones.
const COLS = 12
const ROW_H = 52
const GAP = 16

export interface GridWidget {
  /** identificador estable (clave de la disposición guardada). */
  i: string
  title: string
  accent?: Accent
  /** controles opcionales en la cabecera (no inician el arrastre). */
  actions?: ReactNode
  el: ReactNode
  /** tamaño y mínimos por defecto (en celdas) si no hay disposición guardada. */
  w?: number
  h?: number
  minW?: number
  minH?: number
}

interface Item { i: string; x: number; y: number; w: number; h: number; minW: number; minH: number }

const clamp = (v: number, lo: number, hi: number) => Math.max(lo, Math.min(hi, v))

/** ¿se solapan dos rectángulos de la cuadrícula? */
function overlaps(a: Item, x: number, y: number, w: number, h: number) {
  return a.x < x + w && x < a.x + a.w && a.y < y + h && y < a.y + a.h
}

/** Primer hueco libre (de arriba a abajo, izquierda a derecha) para un w×h. */
function firstFit(items: Item[], w: number, h: number): { x: number; y: number } {
  for (let y = 0; y < 1000; y++) {
    for (let x = 0; x <= COLS - w; x++) {
      if (!items.some((it) => overlaps(it, x, y, w, h))) return { x, y }
    }
  }
  return { x: 0, y: 0 }
}

/** Disposición por defecto: coloca los widgets en flujo según su tamaño. */
function buildDefault(widgets: GridWidget[]): Item[] {
  const out: Item[] = []
  for (const wdg of widgets) {
    const w = wdg.w ?? 6
    const h = wdg.h ?? 4
    const { x, y } = firstFit(out, w, h)
    out.push({ i: wdg.i, x, y, w, h, minW: wdg.minW ?? 3, minH: wdg.minH ?? 3 })
  }
  return out
}

/**
 * Reconcilia la disposición guardada con la lista actual de widgets:
 * conserva los que existen, descarta los que ya no están y coloca los nuevos
 * en el primer hueco libre. Así soporta widgets dinámicos (añadir/quitar).
 */
function reconcile(saved: Item[], widgets: GridWidget[]): Item[] {
  const byId = new Map(saved.map((it) => [it.i, it]))
  const out: Item[] = []
  for (const wdg of widgets) {
    const prev = byId.get(wdg.i)
    if (prev) { out.push(prev); continue }
    const w = wdg.w ?? 6
    const h = wdg.h ?? 4
    const { x, y } = firstFit(out, w, h)
    out.push({ i: wdg.i, x, y, w, h, minW: wdg.minW ?? 3, minH: wdg.minH ?? 3 })
  }
  return out
}

function loadLayout(storageKey: string, widgets: GridWidget[]): Item[] {
  try {
    const raw = localStorage.getItem(storageKey)
    if (raw) return reconcile(JSON.parse(raw) as Item[], widgets)
  } catch { /* ignora JSON inválido */ }
  return buildDefault(widgets)
}

/**
 * Tablero de widgets reordenables y redimensionables sobre una cuadrícula con
 * fondo de puntos. La disposición se guarda en localStorage bajo `storageKey`.
 */
export function GridBoard({ widgets, storageKey, toolbar, minHeight = 320 }: {
  widgets: GridWidget[]
  storageKey: string
  /** controles extra a la izquierda del botón de restablecer. */
  toolbar?: ReactNode
  minHeight?: number
}) {
  const [items, setItems] = useState<Item[]>(() => loadLayout(storageKey, widgets))
  const [active, setActive] = useState<string | null>(null)
  const [width, setWidth] = useState(0)
  const containerRef = useRef<HTMLDivElement>(null)

  // Reaccionar a cambios en la lista de widgets (añadir/quitar dinámicos).
  const ids = widgets.map((w) => w.i).join('|')
  useEffect(() => {
    setItems((prev) => reconcile(prev, widgets))
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [ids])

  // Medir el ancho disponible para calcular el ancho de columna.
  useLayoutEffect(() => {
    const el = containerRef.current
    if (!el) return
    const ro = new ResizeObserver(() => setWidth(el.clientWidth))
    ro.observe(el)
    setWidth(el.clientWidth)
    return () => ro.disconnect()
  }, [])
  const colW = width > 0 ? (width - GAP * (COLS + 1)) / COLS : 0
  const cell = useRef({ colW, rowH: ROW_H })
  cell.current = { colW, rowH: ROW_H }

  // Persistir la disposición.
  useEffect(() => {
    try { localStorage.setItem(storageKey, JSON.stringify(items)) } catch { /* cuota llena */ }
  }, [items, storageKey])

  const drag = useRef<{ id: string; mode: 'move' | 'resize'; sx: number; sy: number; o: Item } | null>(null)
  const onMove = useCallback((e: PointerEvent) => {
    const d = drag.current
    if (!d) return
    const { colW: cw, rowH } = cell.current
    const dcx = Math.round((e.clientX - d.sx) / (cw + GAP))
    const dcy = Math.round((e.clientY - d.sy) / (rowH + GAP))
    setItems((prev) => prev.map((it) => {
      if (it.i !== d.id) return it
      if (d.mode === 'move') return { ...it, x: clamp(d.o.x + dcx, 0, COLS - it.w), y: Math.max(0, d.o.y + dcy) }
      return { ...it, w: clamp(d.o.w + dcx, it.minW, COLS - it.x), h: Math.max(it.minH, d.o.h + dcy) }
    }))
  }, [])
  const onUp = useCallback(() => {
    drag.current = null
    setActive(null)
    window.removeEventListener('pointermove', onMove)
    window.removeEventListener('pointerup', onUp)
  }, [onMove])
  const start = (e: ReactPointerEvent, id: string, mode: 'move' | 'resize') => {
    e.preventDefault()
    const o = items.find((it) => it.i === id)
    if (!o) return
    drag.current = { id, mode, sx: e.clientX, sy: e.clientY, o }
    setActive(id)
    window.addEventListener('pointermove', onMove)
    window.addEventListener('pointerup', onUp)
  }

  const rows = items.reduce((m, it) => Math.max(m, it.y + it.h), 0)
  const height = rows * (ROW_H + GAP) + GAP
  const px = (it: Item) => ({
    left: GAP + it.x * (colW + GAP),
    top: GAP + it.y * (ROW_H + GAP),
    width: it.w * colW + (it.w - 1) * GAP,
    height: it.h * ROW_H + (it.h - 1) * GAP,
  })

  return (
    <>
      <div className="mb-3 flex items-center gap-3">
        {toolbar}
        <button
          onClick={() => setItems(buildDefault(widgets))}
          className="ml-auto flex items-center gap-1.5 rounded-md border border-slate-300 px-2.5 py-1 text-xs text-slate-500 hover:bg-slate-100"
        >
          <RotateCcw size={13} /> Restablecer disposición
        </button>
      </div>

      <div
        ref={containerRef}
        className="relative rounded-xl border border-slate-200"
        style={{
          height: Math.max(height, minHeight),
          backgroundImage: 'radial-gradient(circle, #cbd5e1 1px, transparent 1px)',
          backgroundSize: '24px 24px',
        }}
      >
        {colW > 0 && widgets.map((w) => {
          const it = items.find((x) => x.i === w.i)
          if (!it) return null
          return (
            <div
              key={w.i}
              className={`absolute ${active === w.i ? 'z-10' : 'z-0'}`}
              style={{ ...px(it), transition: active === w.i ? 'none' : 'left .15s, top .15s, width .15s, height .15s' }}
            >
              <Card
                title={w.title}
                accent={w.accent ?? 'neutral'}
                actions={w.actions}
                dragging={active === w.i}
                onDragStart={(e) => start(e, w.i, 'move')}
                onResizeStart={(e) => start(e, w.i, 'resize')}
              >
                {w.el}
              </Card>
            </div>
          )
        })}
      </div>
    </>
  )
}

function Card({ title, accent, actions, children, onDragStart, onResizeStart, dragging }: {
  title: string
  accent: Accent
  actions?: ReactNode
  children: ReactNode
  onDragStart: (e: ReactPointerEvent) => void
  onResizeStart: (e: ReactPointerEvent) => void
  dragging: boolean
}) {
  return (
    <div
      className={`shadow-card relative flex h-full flex-col overflow-hidden rounded-xl border bg-white ${dragging ? 'border-primary ring-primary/20 ring-2' : 'border-slate-200'}`}
      style={{ borderTop: `3px solid var(--accent-${accent})` }}
    >
      <div
        onPointerDown={onDragStart}
        className="flex cursor-move touch-none select-none items-center gap-2 border-b border-slate-100 px-3 py-2"
      >
        <GripVertical size={14} className="text-slate-300" />
        <span className="text-sm font-semibold text-slate-700">{title}</span>
        {actions && (
          <div className="ml-auto flex items-center gap-1" onPointerDown={(e) => e.stopPropagation()}>
            {actions}
          </div>
        )}
      </div>
      <div className="min-h-0 flex-1 overflow-auto p-3">{children}</div>
      <div
        onPointerDown={onResizeStart}
        className="absolute bottom-0 right-0 h-4 w-4 cursor-se-resize touch-none"
        style={{ background: 'linear-gradient(135deg, transparent 50%, #94a3b8 50%, #94a3b8 60%, transparent 60%, transparent 72%, #94a3b8 72%, #94a3b8 82%, transparent 82%)' }}
      />
    </div>
  )
}
