import {
  createContext, useCallback, useContext, useEffect, useLayoutEffect, useMemo, useRef, useState,
  type ReactNode, type PointerEvent as ReactPointerEvent,
} from 'react'
import type uPlot from 'uplot'
import { GripVertical, RotateCcw, Radio, Check, X } from 'lucide-react'
import { PageShell } from '../components/PageShell'
import type { HelpContent } from '../components/HelpButton'
import { UPlotChart } from '../components/charts/UPlotChart'
import { useStore } from '../store/useStore'
import { DATASETS } from '../lib/datasets'
import { openStream } from '../api/client'

// ---- cuadrícula propia (drag + resize con snap, sin dependencias) ----
const COLS = 12
const ROW_H = 52
const GAP = 16
const LS_KEY = 'dashboardLayout-v3'

type Accent = 'signal' | 'fir' | 'csp' | 'brain' | 'metric' | 'neutral'
interface Item { i: string; x: number; y: number; w: number; h: number; minW: number; minH: number }

const DEFAULT_LAYOUT: Item[] = [
  { i: 'raw', x: 0, y: 0, w: 6, h: 4, minW: 4, minH: 3 },
  { i: 'filt', x: 6, y: 0, w: 6, h: 4, minW: 4, minH: 3 },
  { i: 'conf', x: 0, y: 4, w: 8, h: 4, minW: 4, minH: 3 },
  { i: 'decision', x: 8, y: 4, w: 4, h: 4, minW: 3, minH: 3 },
]

const HELP: HelpContent = {
  pipeline: 'El pipeline completo en vivo, paso a paso',
  intro: 'Un panel que sigue, en tiempo real y sobre la misma señal, todas las etapas del sistema: la señal cruda que “llega del casco”, su versión filtrada en la banda µ/β, la confianza del clasificador a lo largo del tiempo y la decisión final. Los paneles se arrastran (desde la cabecera) y se redimensionan (esquina inferior derecha), ajustándose a la cuadrícula; tu disposición se guarda en el navegador.',
  points: [
    { label: 'El recorrido de la señal', desc: 'De izquierda a derecha y de arriba abajo se ve el pipeline: señal cruda → filtro FIR causal (µ/β) → confianza del clasificador (CSP + LDA) → decisión por trial. Todo proviene del mismo flujo en vivo, así que es la misma señal en cada etapa.' },
    { label: 'Cómo iniciarlo', desc: 'Pulsa Play en el panel lateral. El canal de la señal cruda/filtrada y el sujeto se eligen también en el panel lateral.' },
  ],
}

// ---- bus de datos en vivo (un solo WebSocket para todos los widgets) ----
interface LiveMsg {
  trial: number; 'true': string; pred: string; probs: Record<string, number>
  raw?: number[]; filt?: number[]; alo?: number; ahi?: number; t: number
}
const LiveCtx = createContext<{ subscribe: (fn: (m: LiveMsg) => void) => () => void; fs: number; resetKey: string } | null>(null)
const useLive = () => {
  const c = useContext(LiveCtx)
  if (!c) throw new Error('useLive fuera de Dashboard')
  return c
}

const clamp = (v: number, lo: number, hi: number) => Math.max(lo, Math.min(hi, v))
function loadLayout(): Item[] {
  try { const raw = localStorage.getItem(LS_KEY); if (raw) return JSON.parse(raw) as Item[] } catch { /* */ }
  return DEFAULT_LAYOUT
}

const WIDGETS: { i: string; title: string; accent: Accent; el: ReactNode }[] = [
  { i: 'raw', title: 'Señal cruda', accent: 'signal', el: <SignalTrace kind="raw" /> },
  { i: 'filt', title: 'Señal filtrada (µ/β, causal)', accent: 'fir', el: <SignalTrace kind="filt" /> },
  { i: 'conf', title: 'Confianza del clasificador en el tiempo', accent: 'metric', el: <ConfidenceTrace /> },
  { i: 'decision', title: 'Decisión (voto por trial)', accent: 'metric', el: <DecisionSummary /> },
]

export default function Dashboard() {
  const { dataset, subject, channel, playing, clearToken } = useStore()
  const fs = DATASETS[dataset].fs
  const [items, setItems] = useState<Item[]>(loadLayout)
  const [active, setActive] = useState<string | null>(null)
  const [width, setWidth] = useState(0)
  const containerRef = useRef<HTMLDivElement>(null)

  // ---- WebSocket único → bus de suscriptores ----
  const subs = useRef(new Set<(m: LiveMsg) => void>())
  const subscribe = useCallback((fn: (m: LiveMsg) => void) => {
    subs.current.add(fn); return () => { subs.current.delete(fn) }
  }, [])
  useEffect(() => {
    if (!playing) return
    const ws = openStream(`/stream?dataset=${dataset}&subject=${subject}&channel=${channel}`, (d) => {
      subs.current.forEach((fn) => fn(d as LiveMsg))
    })
    return () => ws.close()
  }, [playing, dataset, subject, channel])
  const resetKey = `${clearToken}-${dataset}-${subject}-${channel}`

  // ---- medición de ancho ----
  useLayoutEffect(() => {
    const el = containerRef.current
    if (!el) return
    const ro = new ResizeObserver(() => setWidth(el.clientWidth))
    ro.observe(el); setWidth(el.clientWidth)
    return () => ro.disconnect()
  }, [])
  const colW = width > 0 ? (width - GAP * (COLS + 1)) / COLS : 0
  const cell = useRef({ colW, rowH: ROW_H }); cell.current = { colW, rowH: ROW_H }

  useEffect(() => { try { localStorage.setItem(LS_KEY, JSON.stringify(items)) } catch { /* */ } }, [items])

  const drag = useRef<{ id: string; mode: 'move' | 'resize'; sx: number; sy: number; o: Item } | null>(null)
  const onMove = useCallback((e: PointerEvent) => {
    const d = drag.current; if (!d) return
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
    drag.current = null; setActive(null)
    window.removeEventListener('pointermove', onMove); window.removeEventListener('pointerup', onUp)
  }, [onMove])
  const start = (e: ReactPointerEvent, id: string, mode: 'move' | 'resize') => {
    e.preventDefault()
    const o = items.find((it) => it.i === id); if (!o) return
    drag.current = { id, mode, sx: e.clientX, sy: e.clientY, o }; setActive(id)
    window.addEventListener('pointermove', onMove); window.addEventListener('pointerup', onUp)
  }

  const rows = items.reduce((m, it) => Math.max(m, it.y + it.h), 0)
  const height = rows * (ROW_H + GAP) + GAP
  const px = (it: Item) => ({
    left: GAP + it.x * (colW + GAP), top: GAP + it.y * (ROW_H + GAP),
    width: it.w * colW + (it.w - 1) * GAP, height: it.h * ROW_H + (it.h - 1) * GAP,
  })

  return (
    <PageShell
      title="Dashboard libre"
      subtitle="El pipeline completo en vivo: señal → filtro → confianza → decisión."
      help={HELP}
      world="online"
    >
      <div className="mb-3 flex items-center justify-end gap-3">
        <span className={`flex items-center gap-1.5 rounded-full px-3 py-1 text-xs ${playing ? 'bg-red-50 text-red-600' : 'bg-slate-100 text-slate-500'}`}>
          <Radio size={13} className={playing ? 'animate-pulse' : ''} /> {playing ? 'EN VIVO' : 'detenido — pulsa Play'}
        </span>
        <button onClick={() => setItems(DEFAULT_LAYOUT)} className="flex items-center gap-1.5 rounded-md border border-slate-300 px-2.5 py-1 text-xs text-slate-500 hover:bg-slate-100">
          <RotateCcw size={13} /> Restablecer disposición
        </button>
      </div>

      <LiveCtx.Provider value={{ subscribe, fs, resetKey }}>
        <div ref={containerRef} className="relative rounded-xl border border-slate-200"
          style={{ height: Math.max(height, 320), backgroundImage: 'radial-gradient(circle, #cbd5e1 1px, transparent 1px)', backgroundSize: '24px 24px' }}>
          {colW > 0 && WIDGETS.map((w) => {
            const it = items.find((x) => x.i === w.i); if (!it) return null
            return (
              <div key={w.i} className={`absolute ${active === w.i ? 'z-10' : 'z-0'}`}
                style={{ ...px(it), transition: active === w.i ? 'none' : 'left .15s, top .15s, width .15s, height .15s' }}>
                <Card title={w.title} accent={w.accent} dragging={active === w.i}
                  onDragStart={(e) => start(e, w.i, 'move')} onResizeStart={(e) => start(e, w.i, 'resize')}>
                  {w.el}
                </Card>
              </div>
            )
          })}
        </div>
      </LiveCtx.Provider>
    </PageShell>
  )
}

function Card({ title, accent, children, onDragStart, onResizeStart, dragging }: {
  title: string; accent: Accent; children: ReactNode
  onDragStart: (e: ReactPointerEvent) => void; onResizeStart: (e: ReactPointerEvent) => void; dragging: boolean
}) {
  return (
    <div className={`relative flex h-full flex-col overflow-hidden rounded-xl border bg-white shadow-card ${dragging ? 'border-primary ring-2 ring-primary/20' : 'border-slate-200'}`}
      style={{ borderTop: `3px solid var(--accent-${accent})` }}>
      <div onPointerDown={onDragStart} className="flex cursor-move touch-none select-none items-center gap-2 border-b border-slate-100 px-3 py-2">
        <GripVertical size={14} className="text-slate-300" />
        <span className="text-sm font-semibold text-slate-700">{title}</span>
      </div>
      <div className="min-h-0 flex-1 overflow-hidden p-3">{children}</div>
      <div onPointerDown={onResizeStart} className="absolute bottom-0 right-0 h-4 w-4 cursor-se-resize touch-none"
        style={{ background: 'linear-gradient(135deg, transparent 50%, #94a3b8 50%, #94a3b8 60%, transparent 60%, transparent 72%, #94a3b8 72%, #94a3b8 82%, transparent 82%)' }} />
    </div>
  )
}

// ---- Widgets en vivo ----
const axis = (label: string) => ({ label, labelSize: 30, stroke: '#94a3b8', grid: { stroke: '#eef2f7', width: 1 }, font: '11px Geist Variable' })

/** Gráfico uPlot que rellena la altura disponible del panel (mide su contenedor). */
function FillChart({ data, options, onCreate }: {
  data: uPlot.AlignedData; options: Omit<uPlot.Options, 'width' | 'height'>; onCreate: (u: uPlot) => void
}) {
  const ref = useRef<HTMLDivElement>(null)
  const [h, setH] = useState(0)
  useLayoutEffect(() => {
    const el = ref.current; if (!el) return
    const ro = new ResizeObserver(() => setH(el.clientHeight)); ro.observe(el); setH(el.clientHeight)
    return () => ro.disconnect()
  }, [])
  return <div ref={ref} className="h-full w-full">{h > 20 && <UPlotChart data={data} options={options} height={h} onCreate={onCreate} />}</div>
}

function SignalTrace({ kind }: { kind: 'raw' | 'filt' }) {
  const { subscribe, fs, resetKey } = useLive()
  const u = useRef<uPlot | null>(null)
  const buf = useRef<{ t: number[]; y: number[] }>({ t: [], y: [] })
  const k = useRef(0)
  const WINDOW = 8 // s visibles

  useEffect(() => { buf.current = { t: [], y: [] }; k.current = 0; u.current?.setData([[], []]) }, [resetKey])
  useEffect(() => subscribe((m) => {
    const chunk = kind === 'raw' ? m.raw : m.filt
    if (!chunk) return
    for (const v of chunk) { buf.current.t.push(k.current / fs); buf.current.y.push(v); k.current++ }
    const max = fs * WINDOW
    if (buf.current.t.length > max) { buf.current.t.splice(0, buf.current.t.length - max); buf.current.y.splice(0, buf.current.y.length - max) }
    u.current?.setData([buf.current.t, buf.current.y])
  }), [subscribe, fs, kind])

  const opts = useMemo<Omit<uPlot.Options, 'width' | 'height'>>(() => ({
    legend: { show: false }, cursor: { show: false }, scales: { x: { time: false } },
    axes: [axis('Tiempo (s)'), axis('µV')],
    series: [{}, { stroke: kind === 'raw' ? '#2563eb' : '#0891b2', width: 1.2 }],
  }), [kind])
  return <FillChart data={[[], []]} options={opts} onCreate={(x) => (u.current = x)} />
}

function ConfidenceTrace() {
  const { subscribe, resetKey } = useLive()
  const u = useRef<uPlot | null>(null)
  const buf = useRef<{ t: number[]; a: number[]; b: number[] }>({ t: [], a: [], b: [] })
  const k = useRef(0)
  useEffect(() => { buf.current = { t: [], a: [], b: [] }; k.current = 0; u.current?.setData([[], [], []]) }, [resetKey])
  useEffect(() => subscribe((m) => {
    const cls = Object.keys(m.probs)
    const h = buf.current
    h.t.push(k.current * 0.1); h.a.push(m.probs[cls[0]] ?? 0); h.b.push(m.probs[cls[1]] ?? 0); k.current++
    if (h.t.length > 250) { h.t.shift(); h.a.shift(); h.b.shift() }
    u.current?.setData([h.t, h.a, h.b])
  }), [subscribe])
  const opts = useMemo<Omit<uPlot.Options, 'width' | 'height'>>(() => ({
    legend: { show: false }, cursor: { show: false }, scales: { x: { time: false }, y: { range: [0, 1] } },
    axes: [axis('Tiempo (s)'), axis('probabilidad')],
    series: [{}, { stroke: '#2563eb', width: 1.6 }, { stroke: '#e11d48', width: 1.6 }],
  }), [])
  return <FillChart data={[[], [], []]} options={opts} onCreate={(x) => (u.current = x)} />
}

function DecisionSummary() {
  const { subscribe, resetKey } = useLive()
  const [cur, setCur] = useState<{ pred: string; t: string; conf: number } | null>(null)
  const [acc, setAcc] = useState({ correct: 0, total: 0 })
  const bufRef = useRef<{ trial: number | null; t: string; sum: Record<string, number>; n: number }>({ trial: null, t: '', sum: {}, n: 0 })

  useEffect(() => { bufRef.current = { trial: null, t: '', sum: {}, n: 0 }; setCur(null); setAcc({ correct: 0, total: 0 }) }, [resetKey])
  useEffect(() => subscribe((m) => {
    const b = bufRef.current
    // finalizar trial anterior (voto suave de las ventanas activas)
    if (b.trial !== null && m.trial !== b.trial && b.n > 0) {
      const [pred] = Object.entries(b.sum).reduce((best, x) => (x[1] > best[1] ? x : best))
      setAcc((a) => ({ correct: a.correct + (pred === b.t ? 1 : 0), total: a.total + 1 }))
      bufRef.current = { trial: null, t: '', sum: {}, n: 0 }
    }
    if (bufRef.current.trial !== m.trial) bufRef.current = { trial: m.trial, t: m['true'], sum: {}, n: 0 }
    const active = m.alo == null || m.ahi == null || (m.t >= m.alo && m.t <= m.ahi)
    if (active) {
      const cb = bufRef.current
      for (const c of Object.keys(m.probs)) cb.sum[c] = (cb.sum[c] ?? 0) + m.probs[c]
      cb.n++
      const avg = Object.entries(cb.sum).map(([c, s]) => [c, s / cb.n] as const)
      const [pred, conf] = avg.reduce((best, x) => (x[1] > best[1] ? x : best))
      setCur({ pred, t: m['true'], conf })
    }
  }), [subscribe])

  if (!cur) return <div className="flex h-full items-center justify-center text-center text-sm text-slate-300">Pulsa <strong className="mx-1">Play</strong> para iniciar</div>
  const ok = cur.pred === cur.t
  return (
    <div className="flex h-full flex-col justify-center gap-3 text-center">
      <div className="text-3xl font-bold" style={{ color: ok ? '#059669' : '#2563eb' }}>{cur.pred}</div>
      <div className={`flex items-center justify-center gap-1 text-sm ${ok ? 'text-emerald-600' : 'text-red-500'}`}>
        {ok ? <Check size={15} /> : <X size={15} />} real: {cur.t}
      </div>
      <div className="text-xs text-slate-400">confianza {(cur.conf * 100).toFixed(0)}%</div>
      {acc.total > 0 && (
        <div className="mt-1 border-t border-slate-100 pt-2 text-xs text-slate-600">
          aciertos por trial: <strong>{((acc.correct / acc.total) * 100).toFixed(0)}%</strong> ({acc.correct}/{acc.total})
        </div>
      )}
    </div>
  )
}
