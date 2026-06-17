import {
  createContext, useCallback, useContext, useEffect, useMemo, useRef, useState,
} from 'react'
import type uPlot from 'uplot'
import { Radio, Check, X } from 'lucide-react'
import { PageShell } from '../components/PageShell'
import type { HelpContent } from '../components/HelpButton'
import { GridBoard, type GridWidget } from '../components/GridBoard'
import { FillChart } from '../components/charts/FillChart'
import { useStore } from '../store/useStore'
import { DATASETS } from '../lib/datasets'
import { openStream } from '../api/client'

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

const WIDGETS: GridWidget[] = [
  { i: 'raw', title: 'Señal cruda', accent: 'signal', w: 6, h: 4, minW: 4, minH: 3, el: <SignalTrace kind="raw" /> },
  { i: 'filt', title: 'Señal filtrada (µ/β, causal)', accent: 'fir', w: 6, h: 4, minW: 4, minH: 3, el: <SignalTrace kind="filt" /> },
  { i: 'conf', title: 'Confianza del clasificador en el tiempo', accent: 'metric', w: 8, h: 4, minW: 4, minH: 3, el: <ConfidenceTrace /> },
  { i: 'decision', title: 'Decisión (voto por trial)', accent: 'metric', w: 4, h: 4, minW: 3, minH: 3, el: <DecisionSummary /> },
]

export default function Dashboard() {
  const { dataset, subject, channel, playing, clearToken } = useStore()
  const fs = DATASETS[dataset].fs

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
      subtitle="El pipeline completo en vivo: señal → filtro → confianza → decisión."
      help={HELP}
      world="online"
    >
      <LiveCtx.Provider value={{ subscribe, fs, resetKey }}>
        <GridBoard
          widgets={WIDGETS}
          storageKey="dashboardLayout-v4"
          toolbar={
            <span className={`flex items-center gap-1.5 rounded-full px-3 py-1 text-xs ${playing ? 'bg-red-50 text-red-600' : 'bg-slate-100 text-slate-500'}`}>
              <Radio size={13} className={playing ? 'animate-pulse' : ''} /> {playing ? 'EN VIVO' : 'detenido — pulsa Play'}
            </span>
          }
        />
      </LiveCtx.Provider>
    </PageShell>
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
