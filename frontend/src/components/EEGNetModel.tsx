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
// lo mínimo que necesitamos del CSP clásico para comparar (subconjunto de CSPResp)
interface CSPLike { channels: string[]; patterns: number[][]; eigenvalues: number[]; pos2d: Pos2D }

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

/** Celda de la ficha de entrenamiento: etiqueta pequeña + valor destacado. */
function Ficha({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <div className="text-[10px] uppercase tracking-wide text-amber-600">{label}</div>
      <div className="font-medium text-amber-900">{value}</div>
    </div>
  )
}

export function EEGNetModel({ dataset, subject, csp }: { dataset: string; subject: number; csp: CSPLike | null }) {
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
      <div className="space-y-3 rounded-lg border border-amber-200 bg-amber-50/60 px-4 py-3 text-xs text-amber-900">
        <p>
          <strong>EEGNet como espejo de tu teoría.</strong> No se usa como un segundo clasificador, sino
          para comprobar una idea: si dejamos que una red <strong>aprenda sola</strong> los filtros, ¿llega
          a lo mismo que tú diseñaste a mano? La banda µ/β (8–30 Hz) aparece sombreada en verde como
          referencia. Los filtros mostrados son de un modelo entrenado con todos los trials del sujeto
          (más datos ⇒ filtros más limpios).
        </p>

        {/* --- Ficha de entrenamiento: con qué y cómo se entrenó este modelo --- */}
        <div className="grid grid-cols-2 gap-x-6 gap-y-1.5 sm:grid-cols-3 lg:grid-cols-6">
          <Ficha label="Trials (entren.)" value={String(net.n_train)} />
          <Ficha label="Reservados demo" value={String(net.n_demo)} />
          <Ficha label="Banda de entrada" value={`${net.fir.low_hz}–${net.fir.high_hz} Hz`} />
          <Ficha label="Épocas" value={net.epochs != null ? String(net.epochs) : '—'} />
          <Ficha label="Filtros temporales" value={`${net.n_temporal} (F1)`} />
          <Ficha label="Entrenado" value={net.trained_on} />
        </div>
        <p className="text-amber-700">
          La red recibe una banda amplia ({net.fir.low_hz}–{net.fir.high_hz} Hz) y aprende dentro de ella sus
          propios filtros µ/β; el CSP+LDA, en cambio, recibe la banda 8–30 Hz ya impuesta a mano.
        </p>

        {/* --- Comparación honesta de precisión sobre el MISMO sujeto --- */}
        <div className="overflow-hidden rounded-md border border-amber-200">
          <table className="w-full text-left">
            <thead className="bg-amber-100/70 text-amber-800">
              <tr>
                <th className="px-3 py-1.5 font-medium">Método</th>
                <th className="px-3 py-1.5 font-medium">within-subject k-fold{net.folds ? ` (${net.folds})` : ''}</th>
                <th className="px-3 py-1.5 font-medium">inter-sesión (otro día)</th>
                <th className="px-3 py-1.5 font-medium">κ (inter-sesión)</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-amber-100">
              <tr>
                <td className="px-3 py-1.5">EEGNet (aprende los filtros)</td>
                <td className="px-3 py-1.5 font-mono">{net.accuracy_kfold != null ? (net.accuracy_kfold * 100).toFixed(1) + '%' : '—'}</td>
                <td className="px-3 py-1.5 font-mono">{(net.accuracy_intersession * 100).toFixed(1)}%</td>
                <td className="px-3 py-1.5 font-mono">{net.kappa.toFixed(2)}</td>
              </tr>
              <tr>
                <td className="px-3 py-1.5">CSP+LDA (filtros a mano)</td>
                <td className="px-3 py-1.5 font-mono text-amber-500">—</td>
                <td className="px-3 py-1.5 font-mono">{net.csp_lda ? (net.csp_lda.accuracy_intersession * 100).toFixed(1) + '%' : '—'}</td>
                <td className="px-3 py-1.5 font-mono">{net.csp_lda ? net.csp_lda.kappa.toFixed(2) : '—'}</td>
              </tr>
            </tbody>
          </table>
        </div>
        <p className="text-amber-700">
          Ambas precisiones son honestas (held-out / validación cruzada, sin fuga de datos). Con tan pocos
          trials por sujeto el método clásico suele igualar o superar al deep learning; la comparación
          completa entre sujetos y datasets está en <strong>Resultados</strong>.
        </p>
      </div>

      {/* --- Arquitectura: el flujo end-to-end y su espejo con el pipeline clásico --- */}
      <Widget title="Arquitectura de EEGNet — flujo de extremo a extremo" accent="brain">
        <ArchitectureDiagram net={net} />
        <p className="mt-2 text-xs text-slate-400">
          EEGNet no tiene pasos separados como el método clásico: una sola red aprende todo de extremo a
          extremo. Aun así, cada bloque cumple el papel de una etapa de tu pipeline (la fila inferior marca
          el paralelismo): la convolución temporal ≈ tu <strong>FIR</strong>, la depthwise ≈ tu{' '}
          <strong>CSP</strong>, el pooling ≈ la <strong>log-varianza</strong> y la capa densa ≈ el{' '}
          <strong>LDA</strong>. Por eso aquí no hay subsecciones de varianza ni de clasificador: están
          dentro de la red.
        </p>
      </Widget>

      {/* --- Comparación TEMPORAL: tu FIR vs el banco de filtros aprendido --- */}
      <Widget title="Filtro temporal — diseñado a mano vs aprendido" accent="fir">
        <div className="grid gap-4 md:grid-cols-2">
          <div>
            <div className="mb-1 text-sm font-medium text-slate-600">Tu FIR µ/β (diseñado)</div>
            <UPlotChart data={firData} options={firOptions} height={220} />
          </div>
          <div>
            <div className="mb-1 text-sm font-medium text-slate-600">Filtros temporales de EEGNet ({net.temporal.length}, aprendidos)</div>
            <UPlotChart data={netData} options={netOptions} height={220} />
          </div>
        </div>
        <p className="mt-2 text-xs text-slate-400">
          Tu FIR es un pasa-banda limpio en µ/β. Cada curva de la derecha es un filtro que la red ajustó
          sola (su |H(e^jω)|). La pregunta clave: ¿sus picos caen dentro de la zona verde? Si es así, la
          red <strong>redescubrió</strong> la banda que tú impusiste por teoría.
        </p>
      </Widget>

      {/* --- Comparación ESPACIAL: tus patrones CSP vs los filtros depthwise --- */}
      <Widget title="Filtro espacial — CSP vs depthwise aprendido" accent="csp">
        <div className="grid gap-4 md:grid-cols-2">
          <div>
            <div className="mb-2 text-sm font-medium text-slate-600">Tus patrones CSP (entrenados)</div>
            <div className="flex flex-wrap justify-around gap-2">
              {csp ? csp.patterns.map((pat, i) => (
                <div key={i} className="text-center">
                  <Topomap channels={csp.channels} pos2d={csp.pos2d} values={pat} size={120} />
                  <div className="text-xs text-slate-500">comp {i} · <span className="font-mono">λ={csp.eigenvalues[i].toFixed(2)}</span></div>
                </div>
              )) : <div className="text-slate-300">—</div>}
            </div>
          </div>
          <div>
            <div className="mb-2 text-sm font-medium text-slate-600">Filtros espaciales de EEGNet (aprendidos)</div>
            <div className="flex flex-wrap justify-around gap-2">
              {net.spatial.map((w, i) => (
                <div key={i} className="text-center">
                  <Topomap channels={net.channels} pos2d={net.pos2d} values={w} size={120} />
                  <div className="text-xs text-slate-500">filtro {i}</div>
                </div>
              ))}
            </div>
          </div>
        </div>
        <p className="mt-2 text-xs text-slate-400">
          El CSP combina los canales para maximizar la diferencia de varianza entre clases; típicamente
          se lateraliza sobre la corteza motora (C3/C4). La conv <em>depthwise</em> de EEGNet hace lo
          mismo, pero aprendido. ¿Alguno de sus filtros muestra esa misma lateralización?
        </p>
      </Widget>
    </div>
  )
}
