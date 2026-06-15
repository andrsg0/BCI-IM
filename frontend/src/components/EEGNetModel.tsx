import { useEffect, useMemo, useState } from 'react'
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
      <div className="space-y-2 rounded-lg border border-amber-200 bg-amber-50/60 px-4 py-2.5 text-xs text-amber-900">
        <p>
          <strong>EEGNet como espejo de tu teoría.</strong> No se usa como un segundo clasificador, sino
          para comprobar una idea: si dejamos que una red <strong>aprenda sola</strong> los filtros, ¿llega
          a lo mismo que tú diseñaste a mano? La banda µ/β (8–30 Hz) aparece sombreada en verde como
          referencia. Los filtros mostrados son de un modelo entrenado con todos los trials del sujeto
          (más datos ⇒ filtros más limpios).
        </p>
        <p className="flex flex-wrap gap-x-5 gap-y-1">
          <span>Accuracy EEGNet — within-subject k-fold{net.folds ? ` (${net.folds})` : ''}: <strong>{net.accuracy_kfold != null ? (net.accuracy_kfold * 100).toFixed(1) + '%' : '—'}</strong></span>
          <span>· inter-sesión (otro día): <strong>{(net.accuracy_intersession * 100).toFixed(1)}%</strong></span>
          <span className="text-amber-700">(CSP+LDA: 0.72 / 0.74 — el clásico sigue ganando con tan pocos datos)</span>
        </p>
      </div>

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
