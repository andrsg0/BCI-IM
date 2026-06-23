import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import type uPlot from 'uplot'
import { Radio, Plus, X, RotateCcw } from 'lucide-react'
import { PageShell } from '../components/PageShell'
import type { HelpContent } from '../components/HelpButton'
import { GridBoard, type GridWidget } from '../components/GridBoard'
import { FillChart } from '../components/charts/FillChart'
import { useStore } from '../store/useStore'
import { DATASETS } from '../lib/datasets'
import { getJSON } from '../api/client'
import { convolveCausal, timeAxis } from '../lib/dsp'

interface ContResp { fs: number; channels: string[]; seconds: number; X: number[][] }
interface FilterResp { h: number[]; freqs: number[]; magnitude_db: number[]; group_delay: number }
interface Matrix { fs: number; channels: string[]; X: number[][]; label?: string; nTrials?: number }
interface CSPLite { channels: string[]; filters: number[][] }   // W (n_comp × n_canales): Z = W·X
interface View { id: number; channel: string; mode: 'raw' | 'filtered' }

/** ¿La vista apunta a un componente CSP (comp k) en vez de a un canal físico? */
const compIdx = (ch: string): number | null => (ch.startsWith('comp ') ? Number(ch.slice(5)) : null)

const WINDOW_SEC = 8
const LIVE_SECONDS = 30
const DEFAULTS = { low: 8, high: 30, taps: 101 }

function axisOpts(label: string) {
  return { label, labelSize: 32, labelFont: '12px Geist Variable', stroke: '#94a3b8', grid: { stroke: '#eef2f7', width: 1 }, ticks: { stroke: '#cbd5e1' }, font: '11px Geist Variable' }
}
const HELP: HelpContent = {
  pipeline: 'Etapas 1–2 del pipeline · Adquisición y filtrado FIR (dominio del tiempo)',
  intro: '🔬 Guía de experimentación en el laboratorio. Aquí manipulas el filtro FIR en tiempo real y observas su efecto sobre la señal, tanto en el tiempo como en la frecuencia. Juega con los controles para entender el compromiso clásico de todo filtro digital.',
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

function reveal(arr: number[], idx: number): (number | null)[] {
  const out = new Array<number | null>(arr.length)
  for (let i = 0; i < arr.length; i++) out[i] = i <= idx ? arr[i] : null
  return out
}

export default function SignalLab() {
  const { dataset, subject, channel, clearToken, playing, setProgress, resetProgress } = useStore()
  const fs = DATASETS[dataset].fs

  // El Laboratorio simula siempre la recepción en vivo del casco (mundo online).
  // El antiguo modo "Análisis (trial)" se ocultó por resultar confuso.
  const mode = 'live' as const
  const [mat, setMat] = useState<Matrix | null>(null)
  const [low, setLow] = useState(DEFAULTS.low)
  const [high, setHigh] = useState(DEFAULTS.high)
  const [taps, setTaps] = useState(DEFAULTS.taps)
  const [filter, setFilter] = useState<FilterResp | null>(null)
  const [csp, setCsp] = useState<CSPLite | null>(null)
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

  // matriz CSP (W) del sujeto: para proyectar componentes en el tiempo (comp 0, comp 1…)
  useEffect(() => {
    setCsp(null)
    getJSON<CSPLite>(`/csp?dataset=${dataset}&subject=${subject}`)
      .then((r) => setCsp({ channels: r.channels, filters: r.filters })).catch(() => setCsp(null))
  }, [dataset, subject])

  // saneamiento de canales al cambiar de dataset (los componentes CSP se conservan)
  useEffect(() => {
    if (mat) setViews((vs) => vs.map((v) => (compIdx(v.channel) != null || mat.channels.includes(v.channel) ? v : { ...v, channel: mat.channels[0] })))
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
    if (mat) { fsRef.current = mat.fs; totalRef.current = (mat.X[0].length - 1) / mat.fs; overlay.current.t = 0; setProgress(0, totalRef.current) }
  }, [mat, setProgress])
  useEffect(() => { overlay.current.t = 0; setProgress(0, totalRef.current); verRef.current++ }, [clearToken, setProgress])
  // al salir del Laboratorio limpiamos el progreso del panel lateral
  useEffect(() => () => resetProgress(), [resetProgress])

  // recalcular la serie de cada vista al cambiar vistas/filtro/datos/CSP.
  // Tres casos: canal crudo, canal filtrado (FIR causal), o componente CSP
  // (FIR causal por canal + combinación espacial Z = W·X).
  useEffect(() => {
    if (!mat || !filter) { viewSeries.current = new Map(); return }
    const fcache = new Map<string, number[]>()
    const filt = (chName: string): number[] | null => {
      const ci = mat.channels.indexOf(chName)
      if (ci < 0) return null
      if (!fcache.has(chName)) fcache.set(chName, convolveCausal(mat.X[ci], filter.h))
      return fcache.get(chName)!
    }
    const vs = new Map<number, number[]>()
    for (const v of views) {
      const k = compIdx(v.channel)
      if (k != null) {
        // Componente CSP: requiere W. Se filtra cada canal y se combina linealmente.
        if (!csp || !csp.filters[k]) continue
        const W = csp.filters[k]
        const N = mat.X[0].length
        const out = new Array<number>(N).fill(0)
        for (let c = 0; c < csp.channels.length; c++) {
          const w = W[c]; if (!w) continue
          const f = filt(csp.channels[c]); if (!f) continue
          for (let n = 0; n < N; n++) out[n] += w * f[n]
        }
        vs.set(v.id, out)
      } else {
        const ci = mat.channels.indexOf(v.channel)
        if (ci < 0) continue
        vs.set(v.id, v.mode === 'raw' ? mat.X[ci] : filt(v.channel)!)
      }
    }
    viewSeries.current = vs
    verRef.current++
  }, [views, filter, mat, csp])

  // bucle de animación
  useEffect(() => {
    let raf = 0, last = performance.now()
    const tick = (now: number) => {
      const dt = (now - last) / 1000; last = now
      const { playing, loop, setPlaying, setEnded } = useStore.getState()
      const total = totalRef.current
      if (playing && total > 0) {
        // si se reanuda tras haber terminado, se vuelve a empezar desde el inicio
        if (overlay.current.t >= total) overlay.current.t = 0
        overlay.current.t += dt
        if (overlay.current.t >= total) { if (loop) overlay.current.t = 0; else { overlay.current.t = total; setPlaying(false); setEnded(true) } }
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
      if (deci !== deciRef.current) { deciRef.current = deci; useStore.getState().setProgress(overlay.current.t, totalRef.current) }
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

  // Retardo de grupo del FIR (fase lineal): (N−1)/2 muestras. Lo da el backend; si aún
  // no cargó, se calcula a partir del nº de taps para que el indicador sea inmediato.
  const groupDelay = filter ? filter.group_delay : (taps - 1) / 2
  const groupDelayMs = Math.round((groupDelay / fs) * 1000)

  const freqOptions = useMemo<Omit<uPlot.Options, 'width' | 'height'>>(() => ({
    legend: { show: false }, cursor: { y: false }, scales: { x: { time: false }, y: { range: [-80, 5] } },
    axes: [axisOpts('Frecuencia (Hz)'), axisOpts('Magnitud (dB)')], series: [{}, { stroke: '#0891b2', width: 1.6 }],
    plugins: [{ hooks: { drawClear: (u: uPlot) => drawBand(u, bandRef.current) } }],
  }), [])

  const widgets: GridWidget[] = mat ? [
    ...views.map((v): GridWidget => {
      const k = compIdx(v.channel)
      const isComp = k != null
      const stroke = isComp ? '#7c3aed' : v.mode === 'raw' ? '#2563eb' : '#0891b2'
      return {
        i: `view-${v.id}`,
        title: isComp ? `Componente CSP (FIR + W)` : v.mode === 'raw' ? 'Cruda' : 'Filtrada (causal)',
        accent: isComp ? 'csp' : v.mode === 'raw' ? 'signal' : 'fir',
        w: 8, h: 4, minW: 4, minH: 3,
        actions: (
          <>
            <select value={v.channel} onChange={(e) => updateView(v.id, { channel: e.target.value })} className="rounded border border-slate-300 bg-white px-1.5 py-0.5 text-xs">
              <optgroup label="Canales">
                {mat.channels.map((c) => <option key={c} value={c}>{c}</option>)}
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
        el: <ViewChart id={v.id} stroke={stroke} unit={isComp ? 'u.a.' : 'µV'} ghost={!isComp && v.mode === 'raw'} data={rawInit} overlay={overlay} register={register} />,
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
      subtitle="Bienvenido al Laboratorio DSP. Aquí puedes manipular el comportamiento temporal de las señales bioeléctricas en tiempo real. Al alterar los coeficientes (taps) y las frecuencias de corte de la respuesta al impulso del filtro FIR, modificas matemáticamente la convolución digital y[n] = Σ h[k]·x[n−k], aislando los ritmos cerebrales Mu y Beta (8–30 Hz) de forma puramente causal."
      help={HELP} world="online">
      {!mat ? (
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

function ViewChart({ id, stroke, unit, ghost, data, overlay, register }: {
  id: number; stroke: string; unit: string; ghost: boolean; data: uPlot.AlignedData
  overlay: React.RefObject<{ t: number; half: number; causal: boolean }>; register: (id: number, u: uPlot | null) => void
}) {
  const options = useMemo<Omit<uPlot.Options, 'width' | 'height'>>(() => ({
    legend: { show: false }, cursor: { y: false }, scales: { x: { time: false } },
    axes: [axisOpts('Tiempo (s)'), axisOpts(unit)],
    series: [{}, { stroke, width: 1.3, spanGaps: false }],
    plugins: [{ hooks: { draw: (u: uPlot) => drawOverlay(u, overlay.current!, ghost) } }],
  }), [stroke, unit, ghost, overlay])
  useEffect(() => () => register(id, null), [id, register])
  return <FillChart data={data} options={options} onCreate={(u) => register(id, u)} />
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
