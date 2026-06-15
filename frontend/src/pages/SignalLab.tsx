import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import type uPlot from 'uplot'
import { Radio, Plus, X, RotateCcw } from 'lucide-react'
import { PageShell } from '../components/PageShell'
import type { HelpContent } from '../components/HelpButton'
import { Widget } from '../components/Widget'
import { UPlotChart } from '../components/charts/UPlotChart'
import { useStore } from '../store/useStore'
import { DATASETS } from '../lib/datasets'
import { getJSON } from '../api/client'
import { convolveCausal, timeAxis } from '../lib/dsp'

interface ContResp { fs: number; channels: string[]; seconds: number; X: number[][] }
interface FilterResp { h: number[]; freqs: number[]; magnitude_db: number[]; group_delay: number }
interface Matrix { fs: number; channels: string[]; X: number[][]; label?: string; nTrials?: number }
interface View { id: number; channel: string; mode: 'raw' | 'filtered' }

const WINDOW_SEC = 8
const LIVE_SECONDS = 30
const DEFAULTS = { low: 8, high: 30, taps: 101 }

function axisOpts(label: string) {
  return { label, labelSize: 32, labelFont: '12px Geist Variable', stroke: '#94a3b8', grid: { stroke: '#eef2f7', width: 1 }, ticks: { stroke: '#cbd5e1' }, font: '11px Geist Variable' }
}
const HELP: HelpContent = {
  pipeline: 'Etapas 1–2 del pipeline · Adquisición y filtrado FIR (dominio del tiempo)',
  intro: 'En esta sección se observa la entrada del sistema —la señal EEG— y el efecto del primer bloque de procesamiento: un filtro FIR que aísla los ritmos cerebrales asociados a la imaginación motora. Es donde la teoría de sistemas LTI se aplica de forma más directa, a través de la convolución y[n] = Σ h[k]·x[n−k].',
  points: [
    { label: 'Qué son los canales', desc: 'Cada canal corresponde a un electrodo del sistema internacional 10-20 colocado sobre el cuero cabelludo (Fz, C3, C4, Cz…). Cada uno registra la actividad eléctrica de la región cerebral situada debajo. Los electrodos C3 y C4, sobre la corteza motora izquierda y derecha, son los más informativos para distinguir qué mano se imagina mover.' },
    { label: 'Por qué la banda 8–30 Hz', desc: 'Imaginar un movimiento produce una caída de potencia (desincronización, ERD) en dos ritmos sensorimotores: el ritmo µ (8–12 Hz) y el ritmo β (13–30 Hz). El filtro se diseña para dejar pasar precisamente la banda 8–30 Hz porque ahí se concentra la información que distingue las clases, descartando lo que estorba: la deriva de muy baja frecuencia, los parpadeos oculares, el ruido muscular de alta frecuencia y la interferencia de la red eléctrica (50/60 Hz).' },
    { label: 'Qué implica cambiar el filtro', desc: 'Los controles permiten experimentar. Estrechar la banda (p. ej. solo 8–12 Hz) aísla un único ritmo; ampliarla o desplazarla fuera de µ/β haría que la señal filtrada perdiera la información útil para clasificar. El número de taps regula la nitidez del corte: más taps producen una transición más abrupta entre lo que pasa y lo que se bloquea, a costa de mayor retardo. Estos controles solo afectan a la visualización; el clasificador emplea siempre la banda µ/β (8–30 Hz).' },
    { label: 'Filtrado en tiempo real (causal)', desc: 'Esta vista simula la transmisión del casco: la señal se escribe progresivamente y el filtrado es causal, es decir, solo puede usar muestras pasadas (no conoce el futuro de la señal). Esto introduce un retardo inevitable y un transitorio al inicio —el precio de trabajar en vivo—, a diferencia del procesamiento offline, donde se puede usar toda la época y compensar el retardo.' },
    { label: 'Para qué sirve esta etapa', desc: 'El filtrado es el paso previo imprescindible: limpia y concentra la señal en la banda relevante antes de que el filtrado espacial (CSP) y el clasificador actúen. La elección de la banda condiciona en gran medida la precisión final del sistema.' },
  ],
  terms: ['Convolución', 'FIR', 'Banda µ/β', 'ERD/ERS', 'Causalidad', 'Retardo de grupo', 'Sistema 10-20'],
}

function reveal(arr: number[], idx: number): (number | null)[] {
  const out = new Array<number | null>(arr.length)
  for (let i = 0; i < arr.length; i++) out[i] = i <= idx ? arr[i] : null
  return out
}

export default function SignalLab() {
  const { dataset, subject, channel, clearToken, playing } = useStore()
  const fs = DATASETS[dataset].fs

  // El Laboratorio simula siempre la recepción en vivo del casco (mundo online).
  // El antiguo modo "Análisis (trial)" se ocultó por resultar confuso.
  const mode = 'live' as const
  const [mat, setMat] = useState<Matrix | null>(null)
  const [low, setLow] = useState(DEFAULTS.low)
  const [high, setHigh] = useState(DEFAULTS.high)
  const [taps, setTaps] = useState(DEFAULTS.taps)
  const [filter, setFilter] = useState<FilterResp | null>(null)
  const [headSec, setHeadSec] = useState(0)
  const [views, setViews] = useState<View[]>([
    { id: 0, channel, mode: 'raw' }, { id: 1, channel, mode: 'filtered' },
  ])
  const nextId = useRef(2)

  // --- carga de la señal continua (todos los canales), como llegaría del casco ---
  useEffect(() => {
    setMat(null)
    getJSON<ContResp>(`/continuous_all?dataset=${dataset}&subject=${subject}&seconds=${LIVE_SECONDS}`)
      .then((r) => setMat({ fs: r.fs, channels: r.channels, X: r.X }))
      .catch(() => setMat(null))
  }, [dataset, subject])
  useEffect(() => {
    getJSON<FilterResp>(`/filter?fs=${fs}&low=${low}&high=${high}&taps=${taps}`).then(setFilter).catch(() => setFilter(null))
  }, [fs, low, high, taps])

  // saneamiento de canales al cambiar de dataset
  useEffect(() => {
    if (mat) setViews((vs) => vs.map((v) => (mat.channels.includes(v.channel) ? v : { ...v, channel: mat.channels[0] })))
  }, [mat])

  const t = useMemo(() => (mat ? timeAxis(mat.X[0].length, mat.fs) : []), [mat])
  const emptyTrace = useMemo<(number | null)[]>(() => t.map(() => null), [t])
  const rawInit = useMemo<uPlot.AlignedData>(() => [t, emptyTrace], [t, emptyTrace])
  const freqData = useMemo<uPlot.AlignedData>(() => [filter?.freqs ?? [], filter?.magnitude_db ?? []], [filter])

  // --- refs de animación ---
  const overlay = useRef({ t: 0, half: 0, causal: false })
  const totalRef = useRef(0); const fsRef = useRef(fs); const modeRef = useRef(mode)
  const viewsRef = useRef(views); viewsRef.current = views
  const viewSeries = useRef<Map<number, number[]>>(new Map())
  const uplots = useRef<Map<number, uPlot>>(new Map())
  const verRef = useRef(0); const applied = useRef({ idx: -2, ver: -1 }); const deciRef = useRef(-1)
  const bandRef = useRef({ low, high })

  useEffect(() => { overlay.current.half = taps / fs / 2 }, [taps, fs])
  useEffect(() => { bandRef.current = { low, high } }, [low, high])
  useEffect(() => { modeRef.current = mode; overlay.current.causal = mode === 'live' }, [mode])
  useEffect(() => {
    if (mat) { fsRef.current = mat.fs; totalRef.current = (mat.X[0].length - 1) / mat.fs; overlay.current.t = 0; setHeadSec(0) }
  }, [mat])
  useEffect(() => { overlay.current.t = 0; setHeadSec(0); verRef.current++ }, [clearToken])

  // recalcular la serie de cada vista (cruda o filtrada) al cambiar vistas/filtro/datos
  useEffect(() => {
    if (!mat || !filter) { viewSeries.current = new Map(); return }
    const fcache = new Map<string, number[]>()
    const vs = new Map<number, number[]>()
    for (const v of views) {
      const ci = mat.channels.indexOf(v.channel)
      if (ci < 0) continue
      const raw = mat.X[ci]
      if (v.mode === 'raw') vs.set(v.id, raw)
      else {
        if (!fcache.has(v.channel)) fcache.set(v.channel, convolveCausal(raw, filter.h))
        vs.set(v.id, fcache.get(v.channel)!)
      }
    }
    viewSeries.current = vs
    verRef.current++
  }, [views, filter, mat])

  // bucle de animación
  useEffect(() => {
    let raf = 0, last = performance.now()
    const tick = (now: number) => {
      const dt = (now - last) / 1000; last = now
      const { playing, loop, setPlaying } = useStore.getState()
      const total = totalRef.current
      if (playing && total > 0) {
        overlay.current.t += dt
        if (overlay.current.t >= total) { if (loop) overlay.current.t = 0; else { overlay.current.t = total; setPlaying(false) } }
      }
      const idx = overlay.current.t <= 0 ? -1 : Math.min((tFullLen() - 1), Math.floor(overlay.current.t * fsRef.current))
      if (idx !== applied.current.idx || verRef.current !== applied.current.ver) {
        const slide = modeRef.current === 'live' && total > WINDOW_SEC
        for (const v of viewsRef.current) {
          const u = uplots.current.get(v.id); const s = viewSeries.current.get(v.id)
          if (u && s) {
            u.setData([tRef.current, reveal(s, idx)], !slide)
            if (slide) { const max = Math.max(WINDOW_SEC, overlay.current.t); u.setScale('x', { min: max - WINDOW_SEC, max }) }
          }
        }
        applied.current = { idx, ver: verRef.current }
      }
      const deci = Math.floor(overlay.current.t * 10)
      if (deci !== deciRef.current) { deciRef.current = deci; setHeadSec(overlay.current.t) }
      raf = requestAnimationFrame(tick)
    }
    raf = requestAnimationFrame(tick)
    return () => cancelAnimationFrame(raf)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])
  const tRef = useRef<number[]>([]); tRef.current = t
  const tFullLen = () => (tRef.current.length || 1)

  const register = useCallback((id: number, u: uPlot | null) => { if (u) uplots.current.set(id, u); else uplots.current.delete(id) }, [])
  const addView = () => setViews((vs) => [...vs, { id: nextId.current++, channel: mat?.channels[0] ?? channel, mode: 'raw' }])
  const removeView = (id: number) => setViews((vs) => (vs.length > 1 ? vs.filter((v) => v.id !== id) : vs))
  const updateView = (id: number, patch: Partial<View>) => setViews((vs) => vs.map((v) => (v.id === id ? { ...v, ...patch } : v)))
  const resetFilter = () => { setLow(DEFAULTS.low); setHigh(DEFAULTS.high); setTaps(DEFAULTS.taps) }

  const freqOptions = useMemo<Omit<uPlot.Options, 'width' | 'height'>>(() => ({
    legend: { show: false }, cursor: { y: false }, scales: { x: { time: false }, y: { range: [-80, 5] } },
    axes: [axisOpts('Frecuencia (Hz)'), axisOpts('Magnitud (dB)')], series: [{}, { stroke: '#0891b2', width: 1.6 }],
    plugins: [{ hooks: { drawClear: (u: uPlot) => drawBand(u, bandRef.current) } }],
  }), [])

  return (
    <PageShell title="Laboratorio de Señales (LTI & CSP)"
      subtitle="Señal cruda x[n] y su convolución y[n]."
      help={HELP} world="online">
      <div className="mb-4 flex flex-wrap items-center gap-3 text-sm text-slate-600">
        <span className={`flex items-center gap-1.5 rounded-full px-2.5 py-0.5 text-xs ${playing ? 'bg-red-50 text-red-600' : 'bg-slate-100 text-slate-500'}`}>
          <Radio size={13} className={playing ? 'animate-pulse' : ''} /> {playing ? 'EN VIVO' : 'detenido'} · {headSec.toFixed(1)} s
        </span>
        <button onClick={addView} className="ml-auto flex items-center gap-1 rounded-md border border-slate-300 px-2.5 py-1 hover:bg-slate-100"><Plus size={15} /> Añadir gráfica</button>
      </div>

      <div className="grid items-start gap-4 lg:grid-cols-3">
        <div className="space-y-4 lg:col-span-2">
          {!mat ? <div className="flex h-48 items-center justify-center text-slate-300">Cargando…</div> : views.map((v) => (
            <Widget key={v.id} accent={v.mode === 'raw' ? 'signal' : 'fir'}
              title={v.mode === 'raw' ? 'Cruda' : 'Filtrada (causal)'}
              actions={
                <div className="flex items-center gap-1">
                  <select value={v.channel} onChange={(e) => updateView(v.id, { channel: e.target.value })} className="rounded border border-slate-300 bg-white px-1.5 py-0.5 text-xs">
                    {mat.channels.map((c) => <option key={c} value={c}>{c}</option>)}
                  </select>
                  <select value={v.mode} onChange={(e) => updateView(v.id, { mode: e.target.value as View['mode'] })} className="rounded border border-slate-300 bg-white px-1.5 py-0.5 text-xs">
                    <option value="raw">cruda</option><option value="filtered">filtrada</option>
                  </select>
                  <button onClick={() => removeView(v.id)} className="rounded p-1 text-slate-400 hover:bg-slate-100 hover:text-red-500"><X size={14} /></button>
                </div>
              }>
              <ViewChart id={v.id} mode={v.mode} data={rawInit} overlay={overlay} register={register} />
            </Widget>
          ))}
        </div>

        <div className="space-y-4">
          <Widget title="Respuesta en frecuencia  |H(e^jω)|" accent="fir">
            {filter ? <UPlotChart data={freqData} options={freqOptions} height={170} /> : <div className="h-[170px]" />}
          </Widget>
          <Widget title="Filtro FIR (exploración)" accent="fir">
            <div className="space-y-3 text-sm">
              <Slider label={`Corte inferior: ${low} Hz`} min={1} max={high - 1} value={low} onChange={setLow} />
              <Slider label={`Corte superior: ${high} Hz`} min={low + 1} max={Math.floor(fs / 2)} value={high} onChange={setHigh} />
              <div>
                <label className="mb-1 block text-slate-500">Nº de taps (coeficientes): {taps}</label>
                <select value={taps} onChange={(e) => setTaps(Number(e.target.value))} className="w-full rounded-md border border-slate-300 px-2 py-1.5">
                  {[51, 75, 101, 151, 201].map((n) => <option key={n} value={n}>{n}</option>)}
                </select>
              </div>
              <button onClick={resetFilter} className="flex items-center gap-1.5 rounded-md border border-slate-300 px-2.5 py-1 text-xs hover:bg-slate-100"><RotateCcw size={13} /> Restaurar banda µ/β (8–30)</button>
            </div>
          </Widget>
        </div>
      </div>
    </PageShell>
  )
}

function ViewChart({ id, mode, data, overlay, register }: {
  id: number; mode: 'raw' | 'filtered'; data: uPlot.AlignedData
  overlay: React.RefObject<{ t: number; half: number; causal: boolean }>; register: (id: number, u: uPlot | null) => void
}) {
  const options = useMemo<Omit<uPlot.Options, 'width' | 'height'>>(() => ({
    legend: { show: false }, cursor: { y: false }, scales: { x: { time: false } },
    axes: [axisOpts('Tiempo (s)'), axisOpts('µV')],
    series: [{}, { stroke: mode === 'raw' ? '#2563eb' : '#0891b2', width: 1.3, spanGaps: false }],
    plugins: [{ hooks: { draw: (u: uPlot) => drawOverlay(u, overlay.current!, mode === 'raw') } }],
  }), [mode, overlay])
  useEffect(() => () => register(id, null), [id, register])
  return <UPlotChart data={data} options={options} height={170} onCreate={(u) => register(id, u)} />
}

function Slider({ label, min, max, value, onChange }: { label: string; min: number; max: number; value: number; onChange: (n: number) => void }) {
  return (
    <div>
      <label className="mb-1 block text-slate-500">{label}</label>
      <input type="range" min={min} max={max} value={value} onChange={(e) => onChange(Number(e.target.value))} className="accent-primary w-full" />
    </div>
  )
}

function drawOverlay(u: uPlot, st: { t: number; half: number; causal: boolean }, showGhost: boolean) {
  if (st.t <= 0) return
  const ctx = u.ctx; const { top, height } = u.bbox; const xp = u.valToPos(st.t, 'x', true)
  ctx.save()
  if (showGhost && st.half > 0) {
    const a = st.causal ? st.t - 2 * st.half : st.t - st.half
    const b = st.causal ? st.t : st.t + st.half
    ctx.fillStyle = 'rgba(37, 99, 235, 0.12)'
    ctx.fillRect(u.valToPos(Math.max(0, a), 'x', true), top, u.valToPos(b, 'x', true) - u.valToPos(Math.max(0, a), 'x', true), height)
  }
  ctx.strokeStyle = 'rgba(51, 65, 85, 0.85)'; ctx.lineWidth = Math.max(1, window.devicePixelRatio)
  ctx.beginPath(); ctx.moveTo(xp, top); ctx.lineTo(xp, top + height); ctx.stroke(); ctx.restore()
}
function drawBand(u: uPlot, band: { low: number; high: number }) {
  const ctx = u.ctx; const { top, height } = u.bbox
  const x0 = u.valToPos(band.low, 'x', true); const x1 = u.valToPos(band.high, 'x', true)
  ctx.save(); ctx.fillStyle = 'rgba(5, 150, 105, 0.10)'; ctx.fillRect(x0, top, x1 - x0, height); ctx.restore()
}
