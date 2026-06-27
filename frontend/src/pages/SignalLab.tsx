import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import type uPlot from 'uplot'
import { Radio, Plus, X, RotateCcw } from 'lucide-react'
import { PageShell } from '../components/PageShell'
import type { HelpContent } from '../components/HelpButton'
import { GridBoard, type GridWidget } from '../components/GridBoard'
import { FillChart } from '../components/charts/FillChart'
import { useStore } from '../store/useStore'
import { DATASETS } from '../lib/datasets'
import { openStream, getJSON } from '../api/client'
import { convolveCausal } from '../lib/dsp'
import { STAGE_COLORS } from '../lib/color'
import { progressFromFrame, type ProgressFrame } from '../lib/progress'

interface FilterResp { h: number[]; freqs: number[]; magnitude_db: number[]; group_delay: number }
interface CSPLite { channels: string[]; filters: number[][] }   // W (n_comp × n_canales): Z = W·X
interface View { id: number; channel: string; mode: 'raw' | 'filtered' }
// Frame del WS: la MISMA señal que reproducen las demás secciones en vivo. Trae todos
// los canales crudos del chunk (raw_all) cuando se pide `allch=1`, + info de progreso.
interface Msg extends ProgressFrame { raw_all?: number[][]; channels?: string[] }

/** ¿La vista apunta a un componente CSP (comp k) en vez de a un canal físico? */
const compIdx = (ch: string): number | null => (ch.startsWith('comp ') ? Number(ch.slice(5)) : null)

const WINDOW_SEC = 8
const DEFAULTS = { low: 8, high: 30, taps: 101 }

// Datos iniciales VACÍOS con referencia ESTABLE: estas gráficas se rellenan de forma
// imperativa (onCreate + setData en redraw) y la página re-renderiza en cada frame
// (el progreso actualiza el store). Un literal `[[], []]` nuevo por render dispararía el
// efecto `data` de UPlotChart y BORRARÍA la gráfica cada frame. Con una constante estable, no.
const EMPTY2: uPlot.AlignedData = [[], []]

function axisOpts(label: string) {
  return { label, labelSize: 32, labelFont: '12px Geist Variable', stroke: '#94a3b8', grid: { stroke: '#eef2f7', width: 1 }, ticks: { stroke: '#cbd5e1' }, font: '11px Geist Variable' }
}
const HELP: HelpContent = {
  pipeline: 'Etapas 1–2 del pipeline · Adquisición y filtrado FIR (dominio del tiempo)',
  intro: '🔬 Guía de experimentación en el laboratorio. Aquí recibes EXACTAMENTE la misma señal en vivo que clasifican las demás secciones (los trials reservados que llegan por streaming), y manipulas el filtro FIR en tiempo real para ver su efecto sobre ella, tanto en el tiempo como en la frecuencia. Pulsa Play en el panel lateral.',
  points: [
    { label: '1 · Cruda vs. filtrada (dominio del tiempo)', desc: 'La señal cruda contiene bajas frecuencias (deriva de la línea base por sudor o movimientos) y altas frecuencias (ruido muscular o de cables). Al activar el filtro FIR estándar (8–30 Hz), la señal filtrada oscila de forma mucho más limpia y armónica: estás aislando la actividad sensoriomotora pura.' },
    { label: '2 · Respuesta en frecuencia (dominio espectral)', desc: 'El gráfico de magnitud muestra el filtro en sí. La zona plana y elevada es la banda de paso (8–30 Hz, sombreada en verde). Todo lo que caiga en los valles laterales se atenúa drásticamente, por debajo de −50 dB.' },
    { label: '3 · Respuesta al impulso h[n]', desc: 'El stem plot («pines») son los coeficientes del filtro en el tiempo: literalmente la h[n] de la convolución y[n] = Σ h[k]·x[n−k]. Al mover los taps verás cambiar su forma (un sinc enventanado); su simetría es la que garantiza la fase lineal.' },
    { label: '4 · Sube los taps (p. ej. 151)', desc: 'Las «paredes» de la respuesta en frecuencia se vuelven casi verticales y perfectas (un filtro más selectivo). A cambio, la señal filtrada sufre un retraso mayor por la causalidad del sistema: más coeficientes ⇒ más retardo de grupo.' },
    { label: '5 · Baja los taps (p. ej. 21)', desc: 'El filtro se vuelve suave y redondeado en frecuencia, deja pasar más ruido, pero el retraso temporal cae casi a cero. Es el dilema clásico del ingeniero en BCIs: precisión espectral vs. velocidad en tiempo real.' },
    { label: 'Componentes CSP', desc: 'En el selector de cada gráfica puedes cambiar un canal físico (C3, C4…) por un componente CSP aprendido (comp 0, comp 1…) para ver cómo luce esa «señal virtual» en el tiempo, tras pasar por el filtro FIR y la combinación espacial de canales.' },
  ],
  terms: ['Convolución', 'FIR', 'Respuesta al impulso', 'Banda µ/β', 'Causalidad', 'Retardo de grupo', 'CSP'],
}

export default function SignalLab() {
  // Selectores por campo (no `useStore()` completo): evita re-renderizar el Laboratorio
  // en cada frame por `setProgress`/`setLatency`/`addLog`, que no usa.
  const dataset = useStore((s) => s.dataset)
  const subject = useStore((s) => s.subject)
  const clearToken = useStore((s) => s.clearToken)
  const playing = useStore((s) => s.playing)
  const fs = DATASETS[dataset].fs

  const [channels, setChannels] = useState<string[]>([])
  const [low, setLow] = useState(DEFAULTS.low)
  const [high, setHigh] = useState(DEFAULTS.high)
  const [taps, setTaps] = useState(DEFAULTS.taps)
  const [filter, setFilter] = useState<FilterResp | null>(null)
  const [csp, setCsp] = useState<CSPLite | null>(null)
  const [views, setViews] = useState<View[]>([
    { id: 0, channel: 'C3', mode: 'raw' }, { id: 1, channel: 'C3', mode: 'filtered' },
  ])
  const nextId = useRef(2)

  // --- buffers de la señal recibida por streaming (todos los canales, en µV) ---
  const rawBuf = useRef<number[][]>([])     // [canal][muestra], recortado a WINDOW
  const tBuf = useRef<number[]>([])         // eje de tiempo común
  const kRef = useRef(0)
  const uplots = useRef<Map<number, uPlot>>(new Map())
  const WINDOW = Math.round(fs * WINDOW_SEC)

  // refs para que el dibujado lea siempre lo último sin re-suscribir el WebSocket
  const viewsRef = useRef(views); viewsRef.current = views
  const channelsRef = useRef<string[]>([]); channelsRef.current = channels
  const hRef = useRef<number[] | null>(null); hRef.current = filter?.h ?? null
  const cspRef = useRef<CSPLite | null>(null); cspRef.current = csp
  const bandRef = useRef({ low, high })
  useEffect(() => { bandRef.current = { low, high } }, [low, high])

  // --- nombres de canal, filtro FIR y matriz CSP del sujeto ---
  useEffect(() => {
    setChannels([])
    getJSON<{ channels: string[] }>(`/info?dataset=${dataset}&subject=${subject}`)
      .then((r) => setChannels(r.channels)).catch(() => setChannels([]))
  }, [dataset, subject])
  useEffect(() => {
    getJSON<FilterResp>(`/filter?fs=${fs}&low=${low}&high=${high}&taps=${taps}`).then(setFilter).catch(() => setFilter(null))
  }, [fs, low, high, taps])
  useEffect(() => {
    setCsp(null)
    getJSON<CSPLite>(`/csp?dataset=${dataset}&subject=${subject}`)
      .then((r) => setCsp({ channels: r.channels, filters: r.filters })).catch(() => setCsp(null))
  }, [dataset, subject])

  // saneamiento de canales de las vistas al cambiar de dataset (los CSP se conservan)
  useEffect(() => {
    if (channels.length) setViews((vs) => vs.map((v) => (compIdx(v.channel) != null || channels.includes(v.channel) ? v : { ...v, channel: channels[0] })))
  }, [channels])

  // --- reinicio de buffers (cambio de sujeto/dataset o "limpiar vistas") ---
  const resetBuffers = useCallback(() => {
    rawBuf.current = []; tBuf.current = []; kRef.current = 0
    uplots.current.forEach((u) => u.setData([[], []]))
    useStore.getState().resetProgress()
  }, [])
  useEffect(() => { resetBuffers() }, [dataset, subject, clearToken, resetBuffers])
  // al salir del Laboratorio limpiamos el progreso del panel lateral
  useEffect(() => () => useStore.getState().resetProgress(), [])

  // --- redibujar todas las vistas desde el buffer actual ---
  // Tres casos por vista: canal crudo, canal filtrado (FIR causal en cliente con la banda
  // elegida), o componente CSP (FIR por canal + combinación espacial Z = W·X). Filtrar en
  // cliente es lo que hace interactivo al laboratorio (mover taps/banda y ver el efecto).
  const redraw = useCallback(() => {
    const chs = channelsRef.current, h = hRef.current, t = tBuf.current
    if (!t.length) return
    const fcache = new Map<number, number[]>()
    const filtCh = (ci: number): number[] => {
      const raw = rawBuf.current[ci] ?? []
      if (!h) return raw
      if (!fcache.has(ci)) fcache.set(ci, convolveCausal(raw, h))
      return fcache.get(ci)!
    }
    for (const v of viewsRef.current) {
      const u = uplots.current.get(v.id); if (!u) continue
      const k = compIdx(v.channel)
      let s: number[] | null = null
      if (k != null) {
        const cs = cspRef.current; if (!cs || !cs.filters[k]) continue
        const W = cs.filters[k]; const N = t.length; const out = new Array<number>(N).fill(0)
        for (let c = 0; c < cs.channels.length; c++) {
          const w = W[c]; if (!w) continue
          const ci = chs.indexOf(cs.channels[c]); if (ci < 0) continue
          const f = filtCh(ci)
          for (let n = 0; n < N; n++) out[n] += w * (f[n] ?? 0)
        }
        s = out
      } else {
        const ci = chs.indexOf(v.channel); if (ci < 0) continue
        s = v.mode === 'raw' ? (rawBuf.current[ci] ?? []) : filtCh(ci)
      }
      if (s) u.setData([t, s])
    }
  }, [])

  // --- WebSocket: la MISMA señal que las demás secciones (allch=1 trae todos los canales) ---
  useEffect(() => {
    if (!playing || !channels.length) return
    const ws = openStream(`/stream?dataset=${dataset}&subject=${subject}&allch=1`, (d) => {
      const m = d as Msg
      const prog = progressFromFrame(d as ProgressFrame)
      if (prog) useStore.getState().setProgress(prog[0], prog[1])
      if (!m.raw_all) return
      const nch = m.raw_all.length
      if (rawBuf.current.length !== nch) rawBuf.current = Array.from({ length: nch }, () => [])
      const clen = m.raw_all[0]?.length ?? 0
      for (let c = 0; c < nch; c++) {
        const buf = rawBuf.current[c]; const src = m.raw_all[c]
        for (let i = 0; i < src.length; i++) buf.push(src[i])
        if (buf.length > WINDOW) buf.splice(0, buf.length - WINDOW)
      }
      for (let i = 0; i < clen; i++) { tBuf.current.push(kRef.current / fs); kRef.current++ }
      if (tBuf.current.length > WINDOW) tBuf.current.splice(0, tBuf.current.length - WINDOW)
      redraw()
    })
    return () => ws.close()
  }, [playing, dataset, subject, channels, fs, WINDOW, redraw])

  // redibujar al cambiar filtro/vistas/CSP, sin esperar a un nuevo frame
  useEffect(() => { redraw() }, [filter, views, csp, redraw])

  const register = useCallback((id: number, u: uPlot | null) => { if (u) uplots.current.set(id, u); else uplots.current.delete(id) }, [])
  const addView = () => setViews((vs) => [...vs, { id: nextId.current++, channel: channels[0] ?? 'C3', mode: 'raw' }])
  const removeView = (id: number) => setViews((vs) => (vs.length > 1 ? vs.filter((v) => v.id !== id) : vs))
  const updateView = (id: number, patch: Partial<View>) => setViews((vs) => vs.map((v) => (v.id === id ? { ...v, ...patch } : v)))
  const resetFilter = () => { setLow(DEFAULTS.low); setHigh(DEFAULTS.high); setTaps(DEFAULTS.taps) }

  // Retardo de grupo del FIR (fase lineal): (N−1)/2 muestras. Lo da el backend; si aún
  // no cargó, se calcula a partir del nº de taps para que el indicador sea inmediato.
  const groupDelay = filter ? filter.group_delay : (taps - 1) / 2
  const groupDelayMs = Math.round((groupDelay / fs) * 1000)

  const freqData = useMemo<uPlot.AlignedData>(() => [filter?.freqs ?? [], filter?.magnitude_db ?? []], [filter])
  const freqOptions = useMemo<Omit<uPlot.Options, 'width' | 'height'>>(() => ({
    legend: { show: false }, cursor: { y: false }, scales: { x: { time: false }, y: { range: [-80, 5] } },
    axes: [axisOpts('Frecuencia (Hz)'), axisOpts('Magnitud (dB)')], series: [{}, { stroke: '#0891b2', width: 1.6 }],
    plugins: [{ hooks: { drawClear: (u: uPlot) => drawBand(u, bandRef.current) } }],
  }), [])

  const widgets: GridWidget[] = channels.length ? [
    ...views.map((v): GridWidget => {
      const k = compIdx(v.channel)
      const isComp = k != null
      const stroke = isComp ? STAGE_COLORS.disc : v.mode === 'raw' ? STAGE_COLORS.raw : STAGE_COLORS.filt
      return {
        i: `view-${v.id}`,
        title: isComp ? `Componente CSP (FIR + W)` : v.mode === 'raw' ? 'Cruda' : 'Filtrada (causal)',
        accent: isComp ? 'csp' : v.mode === 'raw' ? 'signal' : 'fir',
        w: 8, h: 4, minW: 4, minH: 3,
        actions: (
          <>
            <select value={v.channel} onChange={(e) => updateView(v.id, { channel: e.target.value })} className="rounded border border-slate-300 bg-white px-1.5 py-0.5 text-xs">
              <optgroup label="Canales">
                {channels.map((c) => <option key={c} value={c}>{c}</option>)}
              </optgroup>
              {csp && csp.filters.length > 0 && (
                <optgroup label="Componentes CSP">
                  {csp.filters.map((_, i) => <option key={`comp ${i}`} value={`comp ${i}`}>comp {i}</option>)}
                </optgroup>
              )}
            </select>
            {!isComp && (
              <select value={v.mode} onChange={(e) => updateView(v.id, { mode: e.target.value as View['mode'] })} className="rounded border border-slate-300 bg-white px-1.5 py-0.5 text-xs">
                <option value="raw">cruda</option><option value="filtered">filtrada</option>
              </select>
            )}
            <button onClick={() => removeView(v.id)} className="rounded p-1 text-slate-400 hover:bg-slate-100 hover:text-red-500"><X size={14} /></button>
          </>
        ),
        el: <ViewChart id={v.id} stroke={stroke} unit={isComp ? 'u.a.' : 'µV'} register={register} />,
      }
    }),
    {
      i: 'freq', title: 'Respuesta en frecuencia  |H(e^jω)|', accent: 'fir', w: 4, h: 4, minW: 3, minH: 3,
      el: filter ? <FillChart data={freqData} options={freqOptions} /> : <div className="h-full" />,
    },
    {
      i: 'impulse', title: 'Respuesta al impulso  h[n]', accent: 'fir', w: 4, h: 4, minW: 3, minH: 3,
      el: filter ? <ImpulseResponse h={filter.h} /> : <div className="h-full" />,
    },
    {
      i: 'filter', title: 'Filtro FIR (exploración)', accent: 'fir', w: 4, h: 5, minW: 3, minH: 4,
      el: (
        <div className="space-y-3 text-sm">
          <Slider label={`Corte inferior: ${low} Hz`} min={1} max={high - 1} value={low} onChange={setLow} />
          <Slider label={`Corte superior: ${high} Hz`} min={low + 1} max={Math.floor(fs / 2)} value={high} onChange={setHigh} />
          <div>
            <label className="mb-1 block text-slate-500">Nº de taps (coeficientes): {taps}</label>
            <select value={taps} onChange={(e) => setTaps(Number(e.target.value))} className="w-full rounded-md border border-slate-300 px-2 py-1.5">
              {[21, 51, 75, 101, 151, 201].map((n) => <option key={n} value={n}>{n}</option>)}
            </select>
            {/* Retardo de grupo: (N−1)/2 muestras, inevitable y causal. */}
            <p className="mt-1.5 rounded-md bg-slate-50 px-2 py-1.5 text-xs leading-snug text-slate-500">
              ⏱️ Retardo de grupo: <strong className="text-slate-700">{groupDelay} muestras</strong>{' '}
              (~{groupDelayMs} ms a {fs} Hz). Por eso la señal filtrada aparece un poco después que la cruda.
            </p>
          </div>
          <button onClick={resetFilter} className="flex items-center gap-1.5 rounded-md border border-slate-300 px-2.5 py-1 text-xs hover:bg-slate-100"><RotateCcw size={13} /> Restaurar banda µ/β (8–30)</button>
        </div>
      ),
    },
  ] : []

  return (
    <PageShell title="Laboratorio de Señales (LTI & CSP)"
      subtitle="Bienvenido al Laboratorio DSP. Recibes la MISMA señal en vivo que clasifican las demás secciones y manipulas el comportamiento temporal del filtro FIR en tiempo real. Al alterar los coeficientes (taps) y las frecuencias de corte de la respuesta al impulso, modificas la convolución digital y[n] = Σ h[k]·x[n−k], aislando los ritmos Mu y Beta (8–30 Hz) de forma puramente causal."
      help={HELP} world="online">
      {!channels.length ? (
        <div className="flex h-48 items-center justify-center text-slate-300">Cargando…</div>
      ) : (
        <GridBoard
          widgets={widgets}
          storageKey="signalLabLayout-v2"
          toolbar={
            <>
              <span className={`flex items-center gap-1.5 rounded-full px-2.5 py-0.5 text-xs ${playing ? 'bg-red-50 text-red-600' : 'bg-slate-100 text-slate-500'}`}>
                <Radio size={13} className={playing ? 'animate-pulse' : ''} /> {playing ? 'EN VIVO' : 'detenido'}
              </span>
              <button onClick={addView} className="flex items-center gap-1 rounded-md border border-slate-300 px-2.5 py-1 text-sm hover:bg-slate-100"><Plus size={15} /> Añadir gráfica</button>
            </>
          }
        />
      )}
    </PageShell>
  )
}

function ViewChart({ id, stroke, unit, register }: {
  id: number; stroke: string; unit: string; register: (id: number, u: uPlot | null) => void
}) {
  const options = useMemo<Omit<uPlot.Options, 'width' | 'height'>>(() => ({
    legend: { show: false }, cursor: { show: false }, scales: { x: { time: false } },
    axes: [axisOpts('Tiempo (s)'), axisOpts(unit)],
    series: [{}, { stroke, width: 1.3, spanGaps: false }],
  }), [stroke, unit])
  useEffect(() => () => register(id, null), [id, register])
  return <FillChart data={EMPTY2} options={options} onCreate={(u) => register(id, u)} />
}

/** Stem plot («pines») de la respuesta al impulso h[n]: los coeficientes del FIR en
 *  el tiempo. Se mide el contenedor para dibujar en píxeles reales (puntos redondos,
 *  trazos nítidos), igual que FillChart hace con uPlot. */
function ImpulseResponse({ h }: { h: number[] }) {
  const ref = useRef<HTMLDivElement | null>(null)
  const [size, setSize] = useState({ w: 0, h: 0 })
  useEffect(() => {
    const el = ref.current
    if (!el) return
    const ro = new ResizeObserver((es) => { const r = es[0].contentRect; setSize({ w: r.width, h: r.height }) })
    ro.observe(el)
    return () => ro.disconnect()
  }, [])

  const { w, h: H } = size
  const padX = 10, padY = 14
  const cy = H / 2
  const maxAbs = h.length ? Math.max(...h.map((v) => Math.abs(v))) || 1 : 1
  const N = h.length
  const xAt = (i: number) => padX + (N <= 1 ? (w - 2 * padX) / 2 : (i / (N - 1)) * (w - 2 * padX))
  const yAt = (v: number) => cy - (v / maxAbs) * (cy - padY)
  // Con muchos taps los puntos saturan; mostramos el marcador solo si caben.
  const showDots = N <= 121

  return (
    <div ref={ref} className="h-full w-full">
      {w > 0 && H > 0 && (
        <svg width={w} height={H}>
          <line x1={padX} y1={cy} x2={w - padX} y2={cy} stroke="#e2e8f0" strokeWidth={1} />
          <text x={padX} y={padY - 2} className="fill-slate-300" fontSize={10}>h[n]</text>
          {h.map((v, i) => (
            <line key={i} x1={xAt(i)} y1={cy} x2={xAt(i)} y2={yAt(v)} stroke="var(--accent-fir)" strokeWidth={1.1} strokeOpacity={0.85} />
          ))}
          {showDots && h.map((v, i) => (
            <circle key={`d${i}`} cx={xAt(i)} cy={yAt(v)} r={1.8} fill="var(--accent-fir)" />
          ))}
        </svg>
      )}
    </div>
  )
}

function Slider({ label, min, max, value, onChange }: { label: string; min: number; max: number; value: number; onChange: (n: number) => void }) {
  return (
    <div>
      <label className="mb-1 block text-slate-500">{label}</label>
      <input type="range" min={min} max={max} value={value} onChange={(e) => onChange(Number(e.target.value))} className="accent-primary w-full" />
    </div>
  )
}

function drawBand(u: uPlot, band: { low: number; high: number }) {
  const ctx = u.ctx; const { top, height } = u.bbox
  const x0 = u.valToPos(band.low, 'x', true); const x1 = u.valToPos(band.high, 'x', true)
  ctx.save(); ctx.fillStyle = 'rgba(5, 150, 105, 0.10)'; ctx.fillRect(x0, top, x1 - x0, height); ctx.restore()
}
