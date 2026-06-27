import {
  createContext, useCallback, useContext, useEffect, useMemo, useRef, useState,
} from 'react'
import type uPlot from 'uplot'
import { Radio, Check, X, Plus } from 'lucide-react'
import { PageShell } from '../components/PageShell'
import type { HelpContent } from '../components/HelpButton'
import { GridBoard, type GridWidget } from '../components/GridBoard'
import { FillChart } from '../components/charts/FillChart'
import { Brain3D, type Pos3D } from '../components/Brain3D'
import { useStore } from '../store/useStore'
import { DATASETS } from '../lib/datasets'
import { openStream, getJSON } from '../api/client'

const HELP: HelpContent = {
  pipeline: 'Un panel libre que armas tú mismo',
  intro: 'Un tablero configurable: añade y quita los paneles que quieras (señal, filtro, clasificador, cerebro 3D…) desde el botón «Añadir panel», reordénalos arrastrándolos por la cabecera y redimensiónalos desde la esquina. Tu selección y tu disposición se guardan en el navegador. Los paneles «en vivo» se animan al pulsar Play en el panel lateral; los «estáticos» muestran un resumen.',
  points: [
    { label: 'Compón tu vista', desc: 'Pulsa «Añadir panel» para insertar cualquier panel del catálogo (de cualquier sección del sistema). Quítalos con la × de su cabecera. La selección se recuerda.' },
    { label: 'En vivo vs estático', desc: 'Los paneles en vivo (señal, filtro, confianza, predicción, cerebro 3D) transmiten en tiempo real cuando pulsas Play; los estáticos (ficha del dataset) son un resumen que no depende del streaming.' },
    { label: 'Cómo iniciarlo', desc: 'Pulsa Play en el panel lateral. El canal de la señal y el sujeto se eligen también ahí.' },
  ],
}

const CLASS_COLORS = ['#2563eb', '#e11d48', '#059669', '#d97706']

// ---- bus de datos en vivo (un solo WebSocket para todos los widgets) ----
interface LiveMsg {
  trial: number; 'true': string; pred: string; probs: Record<string, number>
  raw?: number[]; filt?: number[]; power?: number[]; alo?: number; ahi?: number; t: number
}
const LiveCtx = createContext<{ subscribe: (fn: (m: LiveMsg) => void) => () => void; fs: number; resetKey: string } | null>(null)
const useLive = () => {
  const c = useContext(LiveCtx)
  if (!c) throw new Error('useLive fuera de Dashboard')
  return c
}

// ---------------------------------------------------------------------------
// Catálogo de paneles disponibles (catálogo FIJO: cada página no se registra
// sola, se declara aquí). Cada entrada lleva metadatos extra: `live` (transmite
// con Play) o estático, y una descripción para el menú de inserción.
// ---------------------------------------------------------------------------
type CatalogEntry = GridWidget & { live: boolean; desc: string }

const CATALOG: CatalogEntry[] = [
  { i: 'raw', title: 'Señal cruda', accent: 'signal', w: 6, h: 4, minW: 4, minH: 3, live: true, desc: 'La señal que “llega del casco”, sin filtrar.', el: <SignalTrace kind="raw" /> },
  { i: 'filt', title: 'Señal filtrada (µ/β, causal)', accent: 'fir', w: 6, h: 4, minW: 4, minH: 3, live: true, desc: 'La misma señal tras el filtro FIR causal en la banda µ/β.', el: <SignalTrace kind="filt" /> },
  { i: 'conf', title: 'Confianza del clasificador en el tiempo', accent: 'metric', w: 8, h: 4, minW: 4, minH: 3, live: true, desc: 'Probabilidad de cada clase a lo largo del tiempo.', el: <ConfidenceTrace /> },
  { i: 'decision', title: 'Decisión (voto por trial)', accent: 'metric', w: 4, h: 4, minW: 3, minH: 3, live: true, desc: 'Voto suave por trial y aciertos acumulados.', el: <DecisionSummary /> },
  { i: 'prediction', title: 'Predicción en vivo', accent: 'metric', w: 4, h: 4, minW: 3, minH: 3, live: true, desc: 'Clase predicha ahora mismo y barras de probabilidad.', el: <PredictionLive /> },
  { i: 'brain3d', title: 'Cerebro 3D (actividad µ/β)', accent: 'brain', w: 6, h: 6, minW: 4, minH: 5, live: true, desc: 'Cabeza 3D coloreada con la potencia µ/β (tendencia ERD).', el: <BrainWidget /> },
  { i: 'info', title: 'Ficha del dataset', accent: 'neutral', w: 4, h: 4, minW: 3, minH: 3, live: false, desc: 'Resumen estático del dataset y sujeto seleccionados.', el: <DatasetInfoCard /> },
]

const DEFAULT_WIDGETS = ['raw', 'filt', 'conf', 'decision']
const WIDGETS_KEY = 'dashboardWidgets-v1'

function loadActive(): string[] {
  try {
    const raw = localStorage.getItem(WIDGETS_KEY)
    if (raw) {
      const ids = JSON.parse(raw) as string[]
      const valid = ids.filter((id) => CATALOG.some((c) => c.i === id))
      if (valid.length) return valid
    }
  } catch { /* JSON inválido */ }
  return DEFAULT_WIDGETS
}

export default function Dashboard() {
  const { dataset, subject, channel, playing, clearToken } = useStore()
  const fs = DATASETS[dataset].fs

  // ---- selección de paneles activos (persistida) ----
  const [activeIds, setActiveIds] = useState<string[]>(loadActive)
  useEffect(() => {
    try { localStorage.setItem(WIDGETS_KEY, JSON.stringify(activeIds)) } catch { /* cuota llena */ }
  }, [activeIds])
  const addWidget = useCallback((id: string) => setActiveIds((a) => (a.includes(id) ? a : [...a, id])), [])
  const removeWidget = useCallback((id: string) => setActiveIds((a) => a.filter((x) => x !== id)), [])

  const widgets = useMemo<GridWidget[]>(() => (
    activeIds
      .map((id) => CATALOG.find((c) => c.i === id))
      .filter((c): c is CatalogEntry => Boolean(c))
      .map((c) => ({
        ...c,
        actions: (
          <button
            onClick={() => removeWidget(c.i)}
            title="Quitar este panel"
            className="rounded p-1 text-slate-300 hover:bg-slate-100 hover:text-slate-600"
          >
            <X size={14} />
          </button>
        ),
      }))
  ), [activeIds, removeWidget])

  const available = useMemo(() => CATALOG.filter((c) => !activeIds.includes(c.i)), [activeIds])

  // ---- WebSocket único → bus de suscriptores ----
  const subs = useRef(new Set<(m: LiveMsg) => void>())
  const subscribe = useCallback((fn: (m: LiveMsg) => void) => {
    subs.current.add(fn)
    return () => { subs.current.delete(fn) }
  }, [])
  useEffect(() => {
    if (!playing) return
    const ws = openStream(`/stream?dataset=${dataset}&subject=${subject}&channel=${channel}`, (d) => {
      subs.current.forEach((fn) => fn(d as LiveMsg))
    })
    return () => ws.close()
  }, [playing, dataset, subject, channel])
  const resetKey = `${clearToken}-${dataset}-${subject}-${channel}`

  return (
    <PageShell
      title="Dashboard libre"
      subtitle="Arma tu propio panel: añade/quita paneles de cualquier sección y reordénalos."
      help={HELP}
      world="online"
    >
      <LiveCtx.Provider value={{ subscribe, fs, resetKey }}>
        <GridBoard
          widgets={widgets}
          storageKey="dashboardLayout-v5"
          toolbar={
            <div className="flex items-center gap-2">
              <WidgetPicker available={available} onAdd={addWidget} />
              <span className={`flex items-center gap-1.5 rounded-full px-3 py-1 text-xs ${playing ? 'bg-red-50 text-red-600' : 'bg-slate-100 text-slate-500'}`}>
                <Radio size={13} className={playing ? 'animate-pulse' : ''} /> {playing ? 'EN VIVO' : 'detenido — pulsa Play'}
              </span>
            </div>
          }
        />
        {widgets.length === 0 && (
          <p className="mt-3 text-center text-sm text-slate-400">
            No hay paneles. Pulsa <strong>«Añadir panel»</strong> para empezar.
          </p>
        )}
      </LiveCtx.Provider>
    </PageShell>
  )
}

// ---- menú de inserción de paneles ----
function WidgetPicker({ available, onAdd }: { available: CatalogEntry[]; onAdd: (id: string) => void }) {
  const [open, setOpen] = useState(false)
  const ref = useRef<HTMLDivElement>(null)
  useEffect(() => {
    if (!open) return
    const onDoc = (e: MouseEvent) => { if (ref.current && !ref.current.contains(e.target as Node)) setOpen(false) }
    document.addEventListener('mousedown', onDoc)
    return () => document.removeEventListener('mousedown', onDoc)
  }, [open])

  return (
    <div ref={ref} className="relative">
      <button
        onClick={() => setOpen((o) => !o)}
        className="flex items-center gap-1.5 rounded-md border border-slate-300 px-2.5 py-1 text-xs text-slate-600 hover:bg-slate-100"
      >
        <Plus size={13} /> Añadir panel
      </button>
      {open && (
        <div className="absolute left-0 top-full z-20 mt-1 w-72 rounded-lg border border-slate-200 bg-white p-1 shadow-lg">
          {available.length === 0 ? (
            <p className="px-3 py-2 text-xs text-slate-400">Ya están todos los paneles en el tablero.</p>
          ) : (
            available.map((c) => (
              <button
                key={c.i}
                onClick={() => { onAdd(c.i); setOpen(false) }}
                className="flex w-full flex-col items-start gap-0.5 rounded-md px-3 py-2 text-left hover:bg-slate-50"
              >
                <span className="flex items-center gap-2 text-sm font-medium text-slate-700">
                  {c.title}
                  <span className={`rounded-full px-1.5 py-0.5 text-[10px] ${c.live ? 'bg-red-50 text-red-500' : 'bg-slate-100 text-slate-500'}`}>
                    {c.live ? 'en vivo' : 'estático'}
                  </span>
                </span>
                <span className="text-xs text-slate-400">{c.desc}</span>
              </button>
            ))
          )}
        </div>
      )}
    </div>
  )
}

// ---- Widgets en vivo ----
const axis = (label: string) => ({ label, labelSize: 30, stroke: '#94a3b8', grid: { stroke: '#eef2f7', width: 1 }, font: '11px Geist Variable' })

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

// ---- Predicción instantánea (clase actual + barras de probabilidad) ----
function PredictionLive() {
  const { subscribe, resetKey } = useLive()
  const [m, setM] = useState<LiveMsg | null>(null)
  useEffect(() => { setM(null) }, [resetKey])
  useEffect(() => subscribe(setM), [subscribe])
  if (!m) return <div className="flex h-full items-center justify-center text-center text-sm text-slate-300">Pulsa <strong className="mx-1">Play</strong> para iniciar</div>
  const cls = Object.keys(m.probs)
  const colorOf = (c: string) => CLASS_COLORS[Math.max(0, cls.indexOf(c)) % CLASS_COLORS.length]
  return (
    <div className="flex h-full flex-col justify-center gap-3">
      <div className="text-center text-3xl font-bold" style={{ color: colorOf(m.pred) }}>{m.pred}</div>
      {cls.map((c) => (
        <div key={c}>
          <div className="mb-1 flex justify-between text-sm text-slate-600">
            <span>{c}</span><span className="font-mono">{(m.probs[c] * 100).toFixed(0)}%</span>
          </div>
          <div className="h-2.5 overflow-hidden rounded-full bg-slate-100">
            <div className="h-full rounded-full transition-all" style={{ width: `${m.probs[c] * 100}%`, background: colorOf(c) }} />
          </div>
        </div>
      ))}
    </div>
  )
}

// ---- Cerebro 3D alimentado por el bus (potencia µ/β → tendencia ERD) ----
interface PosResp { channels: string[]; pos3d: Pos3D }
function BrainWidget() {
  const { subscribe, resetKey } = useLive()
  const { dataset, subject } = useStore()
  const [pos, setPos] = useState<PosResp | null>(null)
  const [values, setValues] = useState<number[]>([])
  const base = useRef<number[]>([])   // línea base lenta por canal (EMA)
  const hist = useRef<number[][]>([]) // ventana reciente (agregación temporal)

  useEffect(() => {
    setPos(null); base.current = []; hist.current = []
    getJSON<PosResp>(`/positions?dataset=${dataset}&subject=${subject}`)
      .then((p) => { setPos(p); setValues(new Array(p.channels.length).fill(0)) })
      .catch(() => setPos(null))
  }, [dataset, subject])
  useEffect(() => { base.current = []; hist.current = []; setValues((v) => v.map(() => 0)) }, [resetKey])
  useEffect(() => subscribe((m) => {
    if (!m.power) return
    if (base.current.length !== m.power.length) base.current = m.power.slice()
    base.current = m.power.map((p, i) => 0.98 * base.current[i] + 0.02 * p)
    const target = m.power.map((p, i) => p - base.current[i])
    const h = hist.current
    h.push(target)
    while (h.length > 25) h.shift()   // ~2.5 s de tendencia (ERD)
    setValues(target.map((_, i) => { let s = 0; for (const f of h) s += f[i] ?? 0; return s / h.length }))
  }), [subscribe])

  if (!pos) return <div className="flex h-full items-center justify-center text-sm text-slate-300">Cargando modelo…</div>
  return (
    <div className="h-full min-h-[240px] overflow-hidden rounded-lg">
      <Brain3D channels={pos.channels} pos3d={pos.pos3d} values={values} />
    </div>
  )
}

// ---- Ficha estática del dataset/sujeto seleccionados ----
function DatasetInfoCard() {
  const { dataset, subject, channel } = useStore()
  const d = DATASETS[dataset]
  const rows: { label: string; v: string }[] = [
    { label: 'Dataset', v: d.label },
    { label: 'Sujeto', v: `S${subject}` },
    { label: 'Canal', v: channel },
    { label: 'Frecuencia de muestreo', v: `${d.fs} Hz` },
    { label: 'Sesiones', v: `${d.sessions}${d.sessions >= 2 ? ' · apto demo en vivo' : ''}` },
    { label: 'Accuracy k-fold (CSP+LDA)', v: `${(d.accuracy * 100).toFixed(1)}%` },
  ]
  return (
    <dl className="grid h-full content-center gap-y-2 text-sm">
      {rows.map((r) => (
        <div key={r.label} className="flex items-baseline justify-between gap-3">
          <dt className="text-xs text-slate-400">{r.label}</dt>
          <dd className="font-medium tabular-nums text-slate-700">{r.v}</dd>
        </div>
      ))}
    </dl>
  )
}
