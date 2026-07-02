import { Fragment, useEffect, useMemo, useState } from 'react'
import type uPlot from 'uplot'
import { Widget } from './Widget'
import { UPlotChart } from './charts/UPlotChart'
import { Topomap, type Pos2D } from './Topomap'
import { getJSON } from '../api/client'

interface EEGNetResp {
  channels: string[]; fs: number; freqs: number[]
  temporal: number[][]    // (F1, n_freqs) formas de |H| normalizadas
  spatial: number[][]     // (F1*D, n_canales)
  classes: string[]; kern_length: number; pos2d: Pos2D
  accuracy_intersession: number; accuracy_kfold: number | null; folds: number | null
  // ficha de entrenamiento
  n_train: number; n_demo: number; kappa: number; trained_on: string
  epochs: number | null; n_temporal: number
  fir: { low_hz: number; high_hz: number; num_taps: number }
  csp_lda: { accuracy_intersession: number; kappa: number } | null
}
interface FilterResp { freqs: number[]; magnitude: number[] }

const PALETTE = ['#2563eb', '#e11d48', '#059669', '#d97706', '#7c3aed', '#0891b2', '#db2777', '#65a30d']
const MU_BETA = { low: 8, high: 30 }   // la banda que el método clásico impone a mano
const FREQ_MAX = 45                    // foco en la zona relevante (Hz)

/** Sombrea la banda µ/β (8–30 Hz): la referencia teórica para ambos métodos. */
function drawMuBeta(u: uPlot) {
  const ctx = u.ctx; const { top, height } = u.bbox
  const x0 = u.valToPos(MU_BETA.low, 'x', true); const x1 = u.valToPos(MU_BETA.high, 'x', true)
  ctx.save(); ctx.fillStyle = 'rgba(5, 150, 105, 0.12)'; ctx.fillRect(x0, top, x1 - x0, height); ctx.restore()
}
const freqAxis = (label: string) => ({
  label, labelSize: 30, stroke: '#94a3b8', grid: { stroke: '#eef2f7', width: 1 }, font: '11px Geist Variable',
})
const baseOpts = (nSeries: number): Omit<uPlot.Options, 'width' | 'height'> => ({
  legend: { show: false }, cursor: { y: false },
  scales: { x: { time: false, range: [0, FREQ_MAX] }, y: { range: [0, 1.05] } },
  axes: [freqAxis('Frecuencia (Hz)'), freqAxis('|H| (norm.)')],
  series: [{}, ...Array.from({ length: nSeries }, (_, i) => ({ stroke: PALETTE[i % PALETTE.length], width: 1.4 }))],
  plugins: [{ hooks: { drawClear: drawMuBeta } }],
})

/** Esquema del flujo de EEGNet, con su equivalente en el pipeline clásico (efecto espejo). */
function ArchitectureDiagram({ net }: { net: EEGNetResp }) {
  const C = net.channels.length
  const stages = [
    { name: 'Entrada EEG', sub: `${C} canales × tiempo`, eq: 'señal cruda (banda amplia)', color: 'var(--accent-signal)' },
    { name: 'Conv. Temporal', sub: `Conv2D · ${net.temporal.length} filtros (F1)`, eq: '≈ banco de filtros FIR (µ/β)', color: 'var(--accent-fir)' },
    { name: 'Conv. Espacial', sub: 'DepthwiseConv2D', eq: '≈ filtros espaciales CSP', color: 'var(--accent-csp)' },
    { name: 'Energía', sub: 'SeparableConv + Pooling', eq: '≈ log-varianza (band-power)', color: 'var(--accent-metric)' },
    { name: 'Clasificación', sub: `Densa + Softmax · ${net.classes.length} clases`, eq: '≈ clasificador LDA', color: 'var(--accent-metric)' },
  ]
  return (
    <div className="flex items-stretch gap-1.5 overflow-x-auto pb-1">
      {stages.map((s, i) => (
        <Fragment key={s.name}>
          <div
            className="flex min-w-[8rem] flex-1 flex-col rounded-lg border border-slate-200 bg-white p-2.5 shadow-sm"
            style={{ borderTop: `3px solid ${s.color}` }}
          >
            <div className="text-xs font-semibold text-slate-700">{s.name}</div>
            <div className="mt-0.5 font-mono text-[10px] leading-tight text-slate-500">{s.sub}</div>
            <div className="mt-auto pt-2 text-[10px] italic leading-tight text-slate-400">{s.eq}</div>
          </div>
          {i < stages.length - 1 && <div className="flex shrink-0 items-center text-lg text-slate-300">→</div>}
        </Fragment>
      ))}
    </div>
  )
}


export function EEGNetModel({ dataset, subject }: { dataset: string; subject: number }) {
  const [net, setNet] = useState<EEGNetResp | null>(null)
  const [fir, setFir] = useState<FilterResp | null>(null)

  useEffect(() => {
    setNet(null)
    getJSON<EEGNetResp>(`/eegnet?dataset=${dataset}&subject=${subject}`).then(setNet).catch(() => setNet(null))
  }, [dataset, subject])

  // tu FIR µ/β (8–30 Hz) para comparar con los filtros temporales aprendidos
  useEffect(() => {
    if (!net) return
    getJSON<FilterResp>(`/filter?fs=${net.fs}&low=8&high=30&taps=101`).then(setFir).catch(() => setFir(null))
  }, [net])

  const firData = useMemo<uPlot.AlignedData>(() => {
    if (!fir) return [[], []]
    const peak = Math.max(...fir.magnitude, 1e-9)
    return [fir.freqs, fir.magnitude.map((m) => m / peak)]   // normalizado a pico 1
  }, [fir])
  const firOptions = useMemo(() => {
    const o = baseOpts(1); o.series = [{}, { stroke: '#0f172a', width: 2 }]; return o
  }, [])
  const netData = useMemo<uPlot.AlignedData>(() => (net ? [net.freqs, ...net.temporal] : [[], []]), [net])
  const netOptions = useMemo(() => baseOpts(net ? net.temporal.length : 0), [net])

  if (!net) return <div className="flex h-64 items-center justify-center text-slate-300">Cargando EEGNet…</div>

  return (
    <div className="space-y-4">

      {/* --- Arquitectura: el flujo end-to-end y su espejo con el pipeline clásico --- */}
      <Widget title="El recorrido de la señal" accent="brain">
        <ArchitectureDiagram net={net} />
        <p className="mt-2 text-xs leading-relaxed text-slate-500">
          EEGNet no tiene etapas separadas como el método clásico: una única red procesa la señal de extremo a
          extremo. Aun así, cada bloque cumple el papel de una etapa del pipeline (la fila inferior marca
          el paralelismo): la convolución temporal ≈ el filtro <strong>FIR</strong> clásico, la depthwise ≈ los filtros espaciales{' '}
          <strong>CSP</strong>, el pooling ≈ la <strong>log-varianza</strong> y la capa densa ≈ el{' '}
          <strong>LDA</strong>. Por eso no se muestran subsecciones de varianza ni de clasificador: están
          integradas dentro de la propia red.
        </p>
      </Widget>

      {/* --- Comparación TEMPORAL: Filtro FIR vs el banco de filtros aprendido --- */}
      <Widget title="Filtro temporal — diseñado a mano vs aprendido" accent="fir">
        <p className="mb-3 text-xs leading-relaxed text-slate-500">
          El filtro FIR clásico actúa como un pasa-banda en la banda µ/β (zona verde). Cada curva
          representa la respuesta en frecuencia de uno de los filtros temporales aprendidos por EEGNet.
          Si los picos de las curvas se concentran en la zona verde, significa que la red neuronal
          <strong>redescubrió</strong> de forma autónoma la banda de frecuencia de interés establecida por la teoría.
        </p>
        <div className="grid gap-4 md:grid-cols-2">
          <div>
            <div className="mb-1 text-sm font-medium text-slate-600">Filtro FIR µ/β (diseñado)</div>
            <UPlotChart data={firData} options={firOptions} height={220} />
          </div>
          <div>
            <div className="mb-1 text-sm font-medium text-slate-600">Filtros temporales de EEGNet ({net.temporal.length}, aprendidos)</div>
            <UPlotChart data={netData} options={netOptions} height={220} />
          </div>
        </div>
      </Widget>

      {/* --- Filtros espaciales: filtros depthwise aprendidos por EEGNet --- */}
      <Widget title="Filtro espacial — filtros depthwise aprendidos" accent="csp">
        <p className="mb-3 text-xs leading-relaxed text-slate-500">
          La convolución <em>depthwise</em> de EEGNet realiza una combinación espacial de los canales de forma
          autónoma durante el entrenamiento, buscando patrones discriminativos similares a los obtenidos
          mediante métodos clásicos como el CSP (localizados frecuentemente sobre la corteza motora C3/C4).
        </p>
        <div className="flex flex-wrap content-start justify-around gap-4 py-2">
          {net.spatial.map((w, i) => (
            <div key={i} className="text-center">
              <Topomap channels={net.channels} pos2d={net.pos2d} values={w} size={130} />
              <div className="text-sm font-medium text-slate-700">filtro {i}</div>
              <div className="text-xs text-slate-400">aprendido</div>
            </div>
          ))}
        </div>
      </Widget>
    </div>
  )
}
