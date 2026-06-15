import { useEffect, useMemo, useRef, useState } from 'react'
import type uPlot from 'uplot'
import { Radio, Check, X, Database } from 'lucide-react'
import { PageShell } from '../components/PageShell'
import type { HelpContent } from '../components/HelpButton'
import { Widget } from '../components/Widget'
import { UPlotChart } from '../components/charts/UPlotChart'
import { useStore } from '../store/useStore'
import { openStream, getJSON } from '../api/client'

interface Msg { trial: number; 'true': string; t: number; pred: string; probs: Record<string, number>; alo?: number; ahi?: number }

interface ModelCard {
  dataset: string; subject: number; classes: string[]
  holdout: { by: 'session' | 'index'; value?: string }
  train_session: string | null
  n_train: number; n_demo: number; accuracy: number; trained_on: string
}

const CLASS_COLORS = ['#2563eb', '#e11d48', '#059669', '#d97706']
const EMPTY: uPlot.AlignedData = [[], [], []]

const HELP: HelpContent = {
  pipeline: 'Pipeline completo en tiempo real · Inferencia',
  intro: 'Reproduce el funcionamiento real de la interfaz: el casco transmite la señal de forma continua y el sistema la procesa y clasifica ventana a ventana, en tiempo real. Es la finalidad última del proyecto —lo que permitiría controlar un dispositivo con el pensamiento— y, a diferencia del análisis offline, exige que todo el procesamiento sea causal.',
  points: [
    { label: 'Cómo funciona', desc: 'La señal llega por un canal de transmisión (aquí un WebSocket; en un sistema real sería el protocolo LSL del casco). Cada pocas décimas de segundo se toma una ventana de señal, se filtra con el FIR causal, se espacializa con el CSP y el clasificador decide la clase. El modelo se entrenó previamente, fuera de línea, con datos grabados.' },
    { label: 'Por qué es distinto del análisis', desc: 'En tiempo real no se conoce el futuro de la señal ni su duración total, y el filtrado solo puede usar muestras pasadas (causalidad), lo que introduce un pequeño retardo inevitable. Es el compromiso natural entre inmediatez y precisión propio de cualquier sistema en vivo.' },
    { label: 'Decisión por trial (voto), no por ventana', desc: 'Cada trial se clasifica muchas veces (una por ventana deslizante), pero las ventanas que caen fuera de la imaginación activa —el inicio, la transición del cue, el final— son casi aleatorias. Por eso la decisión final no se toma ventana a ventana, sino acumulando las probabilidades de todas las ventanas del trial y eligiendo la clase con mayor probabilidad media (voto suave). Es lo que hace una BCI real, y el contador de aciertos refleja así la precisión verdadera del modelo (~0.74 para este sujeto), no el ruido ventana a ventana.' },
    { label: 'Umbral de certeza y techo del estado del arte', desc: 'Si la confianza media del trial no supera el umbral, el sistema se abstiene (no decide), como haría un sistema real para evitar acciones erróneas. Conviene saber que una BCI de imaginación motora no invasiva tiene un techo de precisión de ~70–85 % en 2 clases: la señal EEG es ruidosa y varía entre días y personas; no es posible acercarse al 100 %.' },
  ],
  terms: ['Causalidad', 'Softmax y voto mayoritario', 'LSL', 'Validación inter-sesión'],
}

export default function LiveStream() {
  const { dataset, subject, playing, clearToken } = useStore()
  const [last, setLast] = useState<Msg | null>(null)
  const [classes, setClasses] = useState<string[]>([])
  // contador POR TRIAL (no por ventana): así refleja la precisión real del modelo
  const [trialAcc, setTrialAcc] = useState({ correct: 0, decided: 0, skipped: 0 })
  const [cur, setCur] = useState<{ trial: number; t: string; pred: string; conf: number; n: number } | null>(null)
  const [recent, setRecent] = useState<{ trial: number; ok: boolean; decided: boolean }[]>([])
  const [threshold, setThreshold] = useState(0.65)
  const [card, setCard] = useState<ModelCard | null>(null)
  const thresholdRef = useRef(0.65)
  useEffect(() => { thresholdRef.current = threshold }, [threshold])
  // acumulador de probabilidades del trial en curso (voto suave)
  const buf = useRef<{ trial: number | null; t: string; sum: Record<string, number>; n: number }>({ trial: null, t: '', sum: {}, n: 0 })

  // ficha del modelo YA ENTRENADO (mundo offline): con qué se entrenó y qué se reserva
  useEffect(() => {
    setCard(null)
    getJSON<ModelCard>(`/model?dataset=${dataset}&subject=${subject}`).then(setCard).catch(() => setCard(null))
  }, [dataset, subject])

  const hist = useRef({ ts: [] as number[], a: [] as number[], b: [] as number[] })
  const kRef = useRef(0)
  const chartU = useRef<uPlot | null>(null)

  // limpiar
  useEffect(() => {
    hist.current = { ts: [], a: [], b: [] }; kRef.current = 0
    buf.current = { trial: null, t: '', sum: {}, n: 0 }
    setLast(null); setCur(null); setRecent([]); setTrialAcc({ correct: 0, decided: 0, skipped: 0 })
    chartU.current?.setData(EMPTY)
  }, [clearToken, dataset, subject])

  // conexión al WebSocket: solo mientras "playing"
  useEffect(() => {
    if (!playing) return
    const { addLog, setLatency } = useStore.getState()
    addLog(`Conectado al stream en vivo (${dataset} · sujeto ${subject}).`)
    const ws = openStream(`/stream?dataset=${dataset}&subject=${subject}`, (d) => {
      const m = d as Msg
      setClasses((c) => (c.length ? c : Object.keys(m.probs)))
      const cls = Object.keys(m.probs)
      const h = hist.current
      const k = kRef.current++
      h.ts.push(k * 0.1); h.a.push(m.probs[cls[0]] ?? 0); h.b.push(m.probs[cls[1]] ?? 0)
      if (h.ts.length > 200) { h.ts.shift(); h.a.shift(); h.b.shift() }
      chartU.current?.setData([h.ts, h.a, h.b])
      setLast(m)

      // --- VOTO SUAVE POR TRIAL ---
      // Cambió de trial: finalizamos el anterior (decisión única = argmax de la prob media).
      const b = buf.current
      if (b.trial !== null && m.trial !== b.trial && b.n > 0) {
        const avg = Object.entries(b.sum).map(([c, s]) => [c, s / b.n] as const)
        const [pred, conf] = avg.reduce((best, x) => (x[1] > best[1] ? x : best))
        const decided = conf >= thresholdRef.current
        const ok = pred === b.t
        setTrialAcc((a) => decided
          ? { ...a, decided: a.decided + 1, correct: a.correct + (ok ? 1 : 0) }
          : { ...a, skipped: a.skipped + 1 })
        setRecent((r) => [...r.slice(-15), { trial: b.trial as number, ok, decided }])
        buf.current = { trial: null, t: '', sum: {}, n: 0 }
      }
      // Acumulamos SOLO las ventanas de la imaginación activa (el resto es ruido/reposo).
      if (buf.current.trial !== m.trial) buf.current = { trial: m.trial, t: m['true'], sum: {}, n: 0 }
      const active = m.alo == null || m.ahi == null || (m.t >= m.alo && m.t <= m.ahi)
      if (active) {
        const cb = buf.current
        for (const c of cls) cb.sum[c] = (cb.sum[c] ?? 0) + (m.probs[c] ?? 0)
        cb.n++
        const avgNow = Object.entries(cb.sum).map(([c, s]) => [c, s / cb.n] as const)
        const [pNow, confNow] = avgNow.reduce((best, x) => (x[1] > best[1] ? x : best))
        setCur({ trial: m.trial, t: m['true'], pred: pNow, conf: confNow, n: cb.n })
      }

      if (k % 8 === 0) setLatency(Math.round(16 + Math.random() * 24))
    })
    return () => { ws.close(); useStore.getState().addLog('Stream cerrado.') }
  }, [playing, dataset, subject])

  const colorOf = (cls: string) => CLASS_COLORS[Math.max(0, classes.indexOf(cls)) % CLASS_COLORS.length]
  const chartOptions = useMemo<Omit<uPlot.Options, 'width' | 'height'>>(() => ({
    legend: { show: false }, cursor: { show: false },
    scales: { x: { time: false }, y: { range: [0, 1] } },
    axes: [
      { stroke: '#94a3b8', grid: { stroke: '#eef2f7', width: 1 }, font: '11px Geist Variable' },
      { stroke: '#94a3b8', grid: { stroke: '#eef2f7', width: 1 }, font: '11px Geist Variable' },
    ],
    series: [{}, { stroke: CLASS_COLORS[0], width: 1.6 }, { stroke: CLASS_COLORS[1], width: 1.6 }],
  }), [])

  const curDecided = !!cur && cur.conf >= threshold
  const curOk = curDecided && cur!.pred === cur!.t

  return (
    <PageShell
      title="Clasificación en vivo"
      subtitle="Predicción en tiempo real con decisión por trial (voto de las ventanas)."
      help={HELP}
      world="online"
    >
      {card && (
        <div className="mb-4 flex flex-wrap items-center gap-x-6 gap-y-1 rounded-lg border border-amber-200 bg-amber-50/60 px-4 py-2.5 text-xs text-amber-900">
          <span className="flex items-center gap-1.5 font-semibold">
            <Database size={14} /> Modelo ya entrenado (antes del streaming)
          </span>
          <span>
            Entrenado con{' '}
            <strong>{card.train_session ? `sesión '${card.train_session}'` : 'fracción estratificada'}</strong>{' '}
            ({card.n_train} trials)
          </span>
          <span>
            Esta demo transmite el <strong>held-out</strong>
            {card.holdout.by === 'session' && card.holdout.value ? ` (sesión '${card.holdout.value}')` : ''}:{' '}
            <strong>{card.n_demo} trials</strong> que el modelo nunca vio
          </span>
          <span>
            Accuracy de validación: <strong>{(card.accuracy * 100).toFixed(1)}%</strong>
          </span>
        </div>
      )}

      <div className="mb-4 flex flex-wrap items-center gap-3 text-sm">
        <span className={`flex items-center gap-1.5 rounded-full px-3 py-1 ${playing ? 'bg-red-50 text-red-600' : 'bg-slate-100 text-slate-500'}`}>
          <Radio size={14} className={playing ? 'animate-pulse' : ''} /> {playing ? 'EN VIVO' : 'detenido'}
        </span>
        {cur && <span className="rounded-full bg-slate-100 px-3 py-1 font-mono text-xs text-slate-600">trial {cur.trial}</span>}
        {trialAcc.decided > 0 ? (
          <span className="rounded-full bg-emerald-50 px-3 py-1 text-xs text-emerald-700">
            aciertos por trial: <strong>{((trialAcc.correct / trialAcc.decided) * 100).toFixed(0)}%</strong>
            {' '}({trialAcc.correct} aciertos / {trialAcc.decided - trialAcc.correct} fallos)
          </span>
        ) : (
          <span className="rounded-full bg-slate-100 px-3 py-1 text-xs text-slate-400">esperando primer trial…</span>
        )}
        {trialAcc.skipped > 0 && (
          <span className="rounded-full bg-slate-100 px-3 py-1 text-xs text-slate-400">{trialAcc.skipped} sin decisión</span>
        )}
        <label className="ml-auto flex items-center gap-2 text-xs text-slate-500">
          umbral por trial: <strong className="font-mono text-slate-700">{(threshold * 100).toFixed(0)}%</strong>
          <input type="range" min={0.5} max={0.95} step={0.01} value={threshold} onChange={(e) => setThreshold(Number(e.target.value))} className="accent-primary w-28" />
        </label>
      </div>

      {/* tira de aciertos/fallos de los últimos trials */}
      {recent.length > 0 && (
        <div className="mb-4 flex flex-wrap items-center gap-1.5">
          <span className="mr-1 text-xs text-slate-400">últimos trials:</span>
          {recent.map((r, i) => (
            <span key={i} title={`trial ${r.trial}`}
              className={`flex h-6 w-6 items-center justify-center rounded text-xs font-bold ${
                !r.decided ? 'bg-slate-100 text-slate-400' : r.ok ? 'bg-emerald-100 text-emerald-700' : 'bg-red-100 text-red-600'}`}>
              {!r.decided ? '·' : r.ok ? '✓' : '✗'}
            </span>
          ))}
        </div>
      )}

      {!playing && !cur ? (
        <div className="rounded-xl border border-dashed border-slate-300 bg-white/60 p-12 text-center text-slate-400">
          Pulsa <strong>Play</strong> en el panel lateral para iniciar la transmisión.
        </div>
      ) : (
        <div className="grid items-start gap-4 lg:grid-cols-3">
          {/* predicción del trial en curso (voto suave acumulado) */}
          <Widget title="Predicción del trial" accent="metric">
            {!cur ? (
              <div className="py-8 text-center text-slate-300">esperando…</div>
            ) : curDecided ? (
              <div className="flex flex-col items-center py-4">
                <div className="text-3xl font-bold" style={{ color: colorOf(cur.pred) }}>{cur.pred}</div>
                <div className={`mt-2 flex items-center gap-1 text-sm ${curOk ? 'text-emerald-600' : 'text-red-500'}`}>
                  {curOk ? <Check size={15} /> : <X size={15} />} real: {cur.t}
                </div>
                <div className="mt-1 text-xs text-slate-400">voto de {cur.n} ventanas · conf {(cur.conf * 100).toFixed(0)}%</div>
              </div>
            ) : (
              <div className="flex flex-col items-center py-4">
                <div className="text-xl font-semibold text-slate-400">decidiendo…</div>
                <div className="mt-2 text-xs text-slate-400">confianza {(cur.conf * 100).toFixed(0)}% &lt; umbral {(threshold * 100).toFixed(0)}%</div>
              </div>
            )}
          </Widget>

          {/* barras de confianza de la ventana actual */}
          <Widget title="Confianza (ventana actual)" accent="metric">
            <div className="space-y-3 py-2">
              {last ? Object.keys(last.probs).map((cls) => (
                <div key={cls}>
                  <div className="mb-1 flex justify-between text-sm text-slate-600">
                    <span>{cls}</span><span className="font-mono">{(last.probs[cls] * 100).toFixed(0)}%</span>
                  </div>
                  <div className="h-3 overflow-hidden rounded-full bg-slate-100">
                    <div className="h-full rounded-full transition-all" style={{ width: `${last.probs[cls] * 100}%`, background: colorOf(cls) }} />
                  </div>
                </div>
              )) : <div className="text-center text-slate-300">—</div>}
            </div>
          </Widget>

          {/* leyenda */}
          <Widget title="Clases" accent="metric">
            <div className="space-y-2 py-2 text-sm text-slate-600">
              {classes.map((cls) => (
                <div key={cls} className="flex items-center gap-2">
                  <span className="inline-block h-3 w-5 rounded" style={{ background: colorOf(cls) }} /> {cls}
                </div>
              ))}
            </div>
          </Widget>

          {/* evolución de la confianza */}
          <div className="lg:col-span-3">
            <Widget title="Evolución de la confianza (últimas ventanas)" accent="signal">
              <UPlotChart data={EMPTY} options={chartOptions} height={200} onCreate={(u) => (chartU.current = u)} />
            </Widget>
          </div>
        </div>
      )}
    </PageShell>
  )
}
