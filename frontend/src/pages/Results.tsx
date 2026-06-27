import { useEffect, useMemo, useState } from 'react'
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  ReferenceLine, Cell, ScatterChart, Scatter, Legend, ZAxis,
} from 'recharts'
import { PageShell } from '../components/PageShell'
import type { HelpContent } from '../components/HelpButton'
import { DatasetRolesNote } from '../components/DatasetRolesNote'
import { ResultInterpretation } from '../components/ResultInterpretation'
import {
  fetchResultsIndex, fetchDatasetResult, fetchAggregate, pct, kappa, fmtItr, fmtGini,
  type DatasetResult, type SubjectRow, type AggregateResult,
} from '../lib/results'

const HELP: HelpContent = {
  pipeline: 'Evaluación comparativa del sistema',
  intro: 'Compara de forma honesta los dos métodos (CSP+LDA clásico y EEGNet) en dos escenarios: within-subject (calibrado en la misma persona) y cross-subject (usuario nuevo sin calibrar). Todos los números salen del pipeline real; no hay valores inventados.',
  points: [
    { label: 'Within vs cross', desc: 'Within-subject entrena y evalúa en el MISMO sujeto (techo con calibración). Cross-subject (LOSO) entrena con los demás y evalúa en el excluido: estima cómo rendiría con un usuario nuevo sin calibrar. La cross siempre es más baja.' },
    { label: 'Rango, no solo media', desc: 'La precisión varía muchísimo entre personas (BCI illiteracy). Por eso se muestra el rango min–max entre sujetos, no únicamente el promedio, que podría engañar.' },
    { label: 'Significancia', desc: 'El test de Wilcoxon pareado indica si la diferencia entre métodos es estadísticamente real o podría ser azar (p < 0.05 = significativa).' },
    { label: 'Línea de azar', desc: 'En clasificación binaria el azar es 50%. Cualquier barra debe leerse respecto a esa línea, no respecto a cero.' },
  ],
  terms: ['Validación cruzada', 'Accuracy y kappa', 'CSP', 'EEGNet'],
}

function Card({ title, children, right }: { title: string; children: React.ReactNode; right?: React.ReactNode }) {
  return (
    <section className="rounded-xl border border-slate-200 bg-white p-4 shadow-sm">
      <div className="mb-3 flex items-center justify-between gap-3">
        <h2 className="text-sm font-semibold uppercase tracking-wide text-slate-500">{title}</h2>
        {right}
      </div>
      {children}
    </section>
  )
}

// ---------------------------------------------------------------------------
// Resumen: leaderboard de datasets (within-subject CSP+LDA) con barras de rango.
// ---------------------------------------------------------------------------
function Overview({ index, selected, onSelect }: {
  index: DatasetResult[]
  selected: string
  onSelect: (id: string) => void
}) {
  return (
    <Card title="Resumen por dataset">
      <div className="space-y-3">
        {index.map((d) => {
          const s = d.summary['csp_within_acc']
          const active = d.id === selected
          return (
            <button
              key={d.id}
              onClick={() => onSelect(d.id)}
              className={`block w-full rounded-lg border p-3 text-left transition ${active ? 'border-emerald-400 bg-emerald-50/40' : 'border-slate-200 hover:border-slate-300'}`}
            >
              <div className="mb-1 flex items-center justify-between gap-2">
                <span className="flex items-center gap-2 text-sm font-medium text-slate-700">
                  {d.label}
                  {d.live && (
                    <span
                      className="rounded-full border border-sky-200 bg-sky-50 px-2 py-0.5 text-[10px] font-medium text-sky-600"
                      title="Este dataset tiene ≥2 sesiones, así que además sirve para la demo en vivo. Las cifras de aquí son su benchmark de población (within-subject k-fold y cross-subject LOSO), calculado con particiones independientes — no es el modelo desplegado. La partición concreta del modelo en vivo (0train → 1test) se ve en «Demo en vivo»."
                    >
                      {LIVE_TAG}
                    </span>
                  )}
                </span>
                <span className="font-semibold text-slate-800">{s ? pct(s.mean) : '—'}</span>
              </div>
              {s ? (
                <RangeBar min={s.min} max={s.max} mean={s.mean} chance={d.chance} />
              ) : (
                <p className="text-xs text-slate-400">Sin evaluación disponible.</p>
              )}
              <div className="mt-1 text-xs text-slate-400">
                {d.n_subjects_evaluated} sujetos evaluados · {d.fs} Hz
                {s && <> · rango {pct(s.min)}–{pct(s.max)}</>}
              </div>
            </button>
          )
        })}
      </div>
    </Card>
  )
}

/** Barra horizontal que muestra el rango min–max con un marcador en la media y la línea de azar. */
function RangeBar({ min, max, mean, chance }: { min: number; max: number; mean: number; chance: number }) {
  return (
    <div className="relative h-3 w-full rounded-full bg-slate-100">
      <div className="absolute top-0 h-3 rounded-full bg-emerald-200"
        style={{ left: `${min * 100}%`, width: `${(max - min) * 100}%` }} />
      <div className="absolute top-[-2px] h-[18px] w-[3px] rounded bg-emerald-600"
        style={{ left: `calc(${mean * 100}% - 1px)` }} title={`media ${pct(mean)}`} />
      <div className="absolute top-[-2px] h-[18px] w-px bg-slate-400"
        style={{ left: `${chance * 100}%` }} title={`azar ${pct(chance)}`} />
    </div>
  )
}

// ---------------------------------------------------------------------------
// Matriz 2×2: CSP+LDA vs EEGNet × within vs cross.
// ---------------------------------------------------------------------------
function MatrixTable({ r }: { r: DatasetResult }) {
  const m = r.matrix
  const itr = r.itr
  const km = r.kappa_matrix
  const cellCls = (v: number | null, best: boolean) =>
    `rounded-lg p-3 text-center ${v === null ? 'bg-slate-50 text-slate-300'
      : best ? 'bg-emerald-50 ring-1 ring-emerald-300' : 'bg-slate-50'}`
  const bestWithin = pickBest(m.csp.within, m.eegnet.within)
  const bestCross = pickBest(m.csp.cross, m.eegnet.cross)

  const CellContent = ({ acc, itrVal, kappaVal }: { acc: number | null; itrVal?: number | null; kappaVal?: number | null }) => (
    <>
      <div className="text-lg font-bold text-slate-800">{pct(acc)}</div>
      <div className="mt-0.5 flex items-center justify-center gap-2 text-[10px] text-slate-400">
        {kappaVal != null && <span>κ={kappa(kappaVal)}</span>}
        {itrVal != null && <span>{fmtItr(itrVal)} bit/min</span>}
      </div>
    </>
  )

  return (
    <Card title="Comparación de métodos">
      <div className="grid grid-cols-[auto_1fr_1fr] gap-2 text-sm">
        <div />
        <div className="text-center text-xs font-semibold uppercase text-slate-400">Within-subject</div>
        <div className="text-center text-xs font-semibold uppercase text-slate-400">Cross-subject (LOSO)</div>

        <div className="flex items-center text-xs font-semibold text-slate-500">CSP+LDA</div>
        <div className={cellCls(m.csp.within, bestWithin === 'csp')}>
          <CellContent acc={m.csp.within} itrVal={itr?.csp?.within} kappaVal={km?.csp?.within} />
        </div>
        <div className={cellCls(m.csp.cross, bestCross === 'csp')}>
          <CellContent acc={m.csp.cross} itrVal={itr?.csp?.cross} kappaVal={km?.csp?.cross} />
        </div>

        <div className="flex items-center text-xs font-semibold text-slate-500">EEGNet</div>
        <div className={cellCls(m.eegnet.within, bestWithin === 'eegnet')}>
          <CellContent acc={m.eegnet.within} itrVal={itr?.eegnet?.within} kappaVal={km?.eegnet?.within} />
        </div>
        <div className={cellCls(m.eegnet.cross, bestCross === 'eegnet')}>
          <CellContent acc={m.eegnet.cross} itrVal={itr?.eegnet?.cross} kappaVal={km?.eegnet?.cross} />
        </div>
      </div>

      {!r.has_compare && (
        <p className="mt-3 text-xs text-amber-600">
          EEGNet/cross no evaluado en este dataset todavía (solo CSP+LDA within).
        </p>
      )}
    </Card>
  )
}

// ---------------------------------------------------------------------------
// Métricas de comparación y rendimiento: significancia (Wilcoxon), condiciones
// del experimento y variabilidad inter-sujeto (Gini), agrupadas aparte para no
// recargar la matriz 2×2.
// ---------------------------------------------------------------------------
function ComparisonMetrics({ r }: { r: DatasetResult }) {
  return (
    <Card title="Métricas de comparación y rendimiento">
      <div className="space-y-1 text-xs text-slate-500">
        <p>Azar = {pct(r.chance)} · {r.classes.join(' vs ')} · T = {r.trial_time_s}s</p>
        <SignificanceNote label="Within" sig={r.significance.within} />
        <SignificanceNote label="Cross" sig={r.significance.cross} />
      </div>
      <GiniIndicator gini={r.gini} />
    </Card>
  )
}

// Gini: variabilidad inter-sujeto (0=uniforme, 1=concentrada).
const GINI_LABELS: Record<string, string> = {
  csp_within_acc: 'CSP within',
  csp_inter_acc: 'CSP inter',
  csp_cross_acc: 'CSP cross',
  eegnet_within_acc: 'EEGNet within',
  eegnet_cross_acc: 'EEGNet cross',
}

function GiniIndicator({ gini }: { gini: Record<string, number> }) {
  const entries = Object.entries(gini).filter(([k]) => k in GINI_LABELS)
  if (entries.length === 0) return null
  return (
    <div className="mt-3 rounded-lg bg-slate-50 px-3 py-2">
      <p className="mb-1.5 text-[10px] font-semibold uppercase tracking-wide text-slate-400">
        Gini · variabilidad inter-sujeto
      </p>
      <div className="flex flex-wrap gap-3">
        {entries.map(([k, v]) => {
          const color = v < 0.05 ? 'bg-emerald-400' : v < 0.12 ? 'bg-amber-400' : 'bg-red-400'
          return (
            <span key={k} className="flex items-center gap-1 text-[11px] text-slate-600"
              title={`Gini = ${v.toFixed(4)}: ${v < 0.05 ? 'homogéneo (baja variabilidad)' : v < 0.12 ? 'variabilidad moderada' : 'alta variabilidad (BCI illiteracy)'}`}>
              <span className={`inline-block h-2 w-2 rounded-full ${color}`} />
              {GINI_LABELS[k]}: {fmtGini(v)}
            </span>
          )
        })}
      </div>
      <p className="mt-1 text-[10px] text-slate-400">
        Gini bajo ({'<'}0.05) = rendimiento uniforme · alto ({'>'}0.12) = alta dispersión entre sujetos
      </p>
    </div>
  )
}

function pickBest(a: number | null, b: number | null): 'csp' | 'eegnet' | null {
  if (a === null && b === null) return null
  if (b === null) return 'csp'
  if (a === null) return 'eegnet'
  return a >= b ? 'csp' : 'eegnet'
}

/** Estadístico Wilcoxon destacado (p-valor grande + veredicto), para la vista agregada. */
function WilcoxonStat({ label, sig }: { label: string; sig?: { p: number; n: number } }) {
  if (!sig) return <div className="text-slate-300">{label}: —</div>
  const signif = sig.p < 0.05
  return (
    <div>
      <div className="text-[11px] text-slate-500">{label} · n = {sig.n}</div>
      <div className="flex flex-wrap items-baseline gap-1.5">
        <span className={`text-lg font-bold ${signif ? 'text-emerald-700' : 'text-slate-700'}`}>p = {sig.p.toFixed(3)}</span>
        <span className={`rounded px-1.5 py-0.5 text-[10px] font-medium ${signif ? 'bg-emerald-100 text-emerald-700' : 'bg-slate-100 text-slate-500'}`}>
          {signif ? 'significativa' : 'no significativa'}
        </span>
      </div>
    </div>
  )
}

function SignificanceNote({ label, sig }: { label: string; sig?: { p: number; n: number } }) {
  if (!sig) return null
  const signif = sig.p < 0.05
  return (
    <p>
      {label}: diferencia CSP+LDA vs EEGNet {signif
        ? <span className="font-semibold text-emerald-700">significativa</span>
        : <span className="text-slate-500">no significativa</span>} (Wilcoxon p = {sig.p.toFixed(3)}, n = {sig.n})
    </p>
  )
}

// ---------------------------------------------------------------------------
// Vista general agregada: matriz 2×2 sobre TODA la población (no por dataset).
// ---------------------------------------------------------------------------
const LIVE_TAG = 'demo'

function AggregateMatrix({ a }: { a: AggregateResult }) {
  const m = a.matrix
  const bestWithin = pickBest(m.csp.within, m.eegnet.within)
  const bestCross = pickBest(m.csp.cross, m.eegnet.cross)
  const cellCls = (v: number | null, best: boolean) =>
    `rounded-lg p-3 text-center ${v === null ? 'bg-slate-50 text-slate-300'
      : best ? 'bg-emerald-50 ring-1 ring-emerald-300' : 'bg-slate-50'}`
  const nOf = (metric: string) => a.summary[metric]?.n ?? 0

  const cols: { key: string; label: string }[] = [
    { key: 'csp_within_acc', label: 'CSP within' },
    { key: 'csp_cross_acc', label: 'CSP cross' },
    { key: 'eegnet_within_acc', label: 'EEGNet within' },
    { key: 'eegnet_cross_acc', label: 'EEGNet cross' },
  ]

  return (
    <Card title="Comparación general de métodos">
      <div className="grid gap-5 lg:grid-cols-[minmax(0,360px)_1fr]">
        {/* Matriz 2×2 agregada */}
        <div>
          <div className="grid grid-cols-[auto_1fr_1fr] gap-2 text-sm">
            <div />
            <div className="text-center text-xs font-semibold uppercase text-slate-400">Within</div>
            <div className="text-center text-xs font-semibold uppercase text-slate-400">Cross (LOSO)</div>

            <div className="flex items-center text-xs font-semibold text-slate-500">CSP+LDA</div>
            <div className={cellCls(m.csp.within, bestWithin === 'csp')}>
              <div className="text-lg font-bold text-slate-800">{pct(m.csp.within)}</div>
              <div className="text-[10px] text-slate-400">n={nOf('csp_within_acc')}</div>
            </div>
            <div className={cellCls(m.csp.cross, bestCross === 'csp')}>
              <div className="text-lg font-bold text-slate-800">{pct(m.csp.cross)}</div>
              <div className="text-[10px] text-slate-400">n={nOf('csp_cross_acc')}</div>
            </div>

            <div className="flex items-center text-xs font-semibold text-slate-500">EEGNet</div>
            <div className={cellCls(m.eegnet.within, bestWithin === 'eegnet')}>
              <div className="text-lg font-bold text-slate-800">{pct(m.eegnet.within)}</div>
              <div className="text-[10px] text-slate-400">n={nOf('eegnet_within_acc')}</div>
            </div>
            <div className={cellCls(m.eegnet.cross, bestCross === 'eegnet')}>
              <div className="text-lg font-bold text-slate-800">{pct(m.eegnet.cross)}</div>
              <div className="text-[10px] text-slate-400">n={nOf('eegnet_cross_acc')}</div>
            </div>
          </div>
          <div className="mt-3 space-y-2 text-xs text-slate-500">
            <p>Pooled por sujeto sobre {a.n_datasets} datasets evaluados · azar 50% · n = nº de sujetos.</p>
            <div className="rounded-lg border border-slate-200 bg-slate-50 p-2.5">
              <p className="mb-1.5 text-[10px] font-semibold uppercase tracking-wide text-slate-400">
                Significancia · Wilcoxon pareado (CSP+LDA vs EEGNet)
              </p>
              <div className="grid grid-cols-2 gap-2">
                <WilcoxonStat label="Within" sig={a.significance.within} />
                <WilcoxonStat label="Cross" sig={a.significance.cross} />
              </div>
            </div>
          </div>
        </div>

        {/* Desglose por dataset (honestidad: qué aporta cada uno) */}
        <div className="overflow-x-auto">
          <table className="w-full text-xs">
            <thead>
              <tr className="border-b border-slate-200 text-left uppercase text-slate-400">
                <th className="px-2 py-1.5">Dataset</th>
                <th className="px-2 py-1.5">Suj.</th>
                {cols.map((c) => <th key={c.key} className="px-2 py-1.5">{c.label}</th>)}
              </tr>
            </thead>
            <tbody>
              {a.per_dataset.filter((d) => d.n > 0).map((d) => (
                <tr key={d.id} className="border-b border-slate-100">
                  <td className="px-2 py-1.5 text-slate-700">
                    {d.label}
                    {d.live && <span className="ml-1 text-[10px] text-slate-400">· {LIVE_TAG}</span>}
                  </td>
                  <td className="px-2 py-1.5 tabular-nums text-slate-500">{d.n}</td>
                  {cols.map((c) => (
                    <td key={c.key} className="px-2 py-1.5 tabular-nums text-slate-700">
                      {c.key in d.cells ? pct(d.cells[c.key]) : '—'}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
          <p className="mt-2 text-[11px] text-slate-400">
            «—» = método/escenario aún no evaluado en ese dataset. La fila de demo en vivo (2a)
            se incluye solo aquí, como comparación científica de métodos.
          </p>
        </div>
      </div>
    </Card>
  )
}

// ---------------------------------------------------------------------------
// Gráfico por sujeto: barras de accuracy por sujeto con línea de azar.
// ---------------------------------------------------------------------------
type Metric = 'csp_within_acc' | 'csp_cross_acc' | 'eegnet_within_acc' | 'eegnet_cross_acc'
const METRIC_LABEL: Record<Metric, string> = {
  csp_within_acc: 'CSP+LDA within',
  csp_cross_acc: 'CSP+LDA cross',
  eegnet_within_acc: 'EEGNet within',
  eegnet_cross_acc: 'EEGNet cross',
}

/** Estadísticos de dispersión de una métrica entre sujetos (ignora nulos).
 *  σ muestral (ddof=1): la convención de los papers para "media ± σ". */
function metricStats(raw: (number | null | undefined)[]) {
  const a = raw.filter((v): v is number => typeof v === 'number').sort((x, y) => x - y)
  const n = a.length
  if (n === 0) return null
  const mean = a.reduce((s, v) => s + v, 0) / n
  const median = n % 2 ? a[(n - 1) / 2] : (a[n / 2 - 1] + a[n / 2]) / 2
  const std = n > 1 ? Math.sqrt(a.reduce((s, v) => s + (v - mean) ** 2, 0) / (n - 1)) : 0
  return { n, mean, median, std, min: a[0], max: a[n - 1] }
}

/** Resumen estadístico de la métrica seleccionada entre los sujetos del dataset.
 *  Complementa el min–max y el Gini con media ± σ y mediana (robusta al sesgo por
 *  «BCI illiteracy»). Se calcula en el cliente desde los datos por sujeto. */
function MetricStats({ r, metric }: { r: DatasetResult; metric: Metric }) {
  const stats = useMemo(
    () => metricStats((r.subjects ?? []).map((s) => s[metric] as number | null)),
    [r.subjects, metric],
  )
  if (!stats) return null
  const items: { label: string; v: string; hint?: string }[] = [
    { label: 'media ± σ', v: `${pct(stats.mean)} ± ${(stats.std * 100).toFixed(1)} pp`, hint: 'σ muestral, en puntos porcentuales' },
    { label: 'mediana', v: pct(stats.median), hint: 'robusta a sujetos cerca del azar' },
    { label: 'rango', v: `${pct(stats.min)}–${pct(stats.max)}` },
    { label: 'n', v: String(stats.n), hint: 'sujetos con esta métrica' },
  ]
  return (
    <div className="mt-3 flex flex-wrap gap-x-5 gap-y-1 rounded-lg bg-slate-50 px-3 py-2 text-xs">
      {items.map((it) => (
        <span key={it.label} className="flex items-baseline gap-1.5" title={it.hint}>
          <span className="text-slate-400">{it.label}</span>
          <span className="font-medium tabular-nums text-slate-700">{it.v}</span>
        </span>
      ))}
    </div>
  )
}

function SubjectChart({ r, metric, onPick, selected }: {
  r: DatasetResult
  metric: Metric
  onPick: (s: number) => void
  selected: number | null
}) {
  const data = (r.subjects ?? []).map((s) => ({ subject: `S${s.subject}`, id: s.subject, val: s[metric] ?? null }))
  return (
    <ResponsiveContainer width="100%" height={260}>
      <BarChart data={data} margin={{ top: 8, right: 8, bottom: 4, left: -16 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="#eef2f7" />
        <XAxis dataKey="subject" tick={{ fontSize: 11 }} interval={0} />
        <YAxis domain={[0, 1]} tickFormatter={(v) => `${Math.round(v * 100)}`} tick={{ fontSize: 11 }} />
        <Tooltip formatter={(v) => pct(typeof v === 'number' ? v : null)} labelFormatter={(l) => `Sujeto ${l}`} />
        <ReferenceLine y={r.chance} stroke="#94a3b8" strokeDasharray="4 4"
          label={{ value: 'azar', position: 'insideTopRight', fontSize: 10, fill: '#94a3b8' }} />
        <Bar dataKey="val" radius={[3, 3, 0, 0]} onClick={(_, i) => onPick(data[i].id)} cursor="pointer">
          {data.map((d) => (
            <Cell key={d.id} fill={d.id === selected ? '#059669' : '#34d399'} />
          ))}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  )
}

// ---------------------------------------------------------------------------
// Tabla por sujeto, ordenable. Clic en una fila abre la ficha de detalle.
// ---------------------------------------------------------------------------
const COLS: { key: keyof SubjectRow; label: string; fmt: (v: unknown) => string }[] = [
  { key: 'subject', label: 'Sujeto', fmt: (v) => `S${v}` },
  { key: 'n_trials', label: 'Trials', fmt: (v) => (v == null ? '—' : String(v)) },
  { key: 'csp_within_acc', label: 'CSP within', fmt: (v) => pct(v as number) },
  { key: 'csp_within_kappa', label: 'κ', fmt: (v) => kappa(v as number) },
  { key: 'csp_within_sens', label: 'Sens', fmt: (v) => pct(v as number) },
  { key: 'csp_within_spec', label: 'Spec', fmt: (v) => pct(v as number) },
  { key: 'csp_inter_acc', label: 'CSP inter-sesión', fmt: (v) => pct(v as number) },
  { key: 'csp_cross_acc', label: 'CSP cross', fmt: (v) => pct(v as number) },
  { key: 'csp_cross_kappa', label: 'κ cross', fmt: (v) => kappa(v as number) },
  { key: 'eegnet_within_acc', label: 'EEGNet within', fmt: (v) => pct(v as number) },
  { key: 'eegnet_within_kappa', label: 'κ EEG', fmt: (v) => kappa(v as number) },
  { key: 'eegnet_cross_acc', label: 'EEGNet cross', fmt: (v) => pct(v as number) },
  { key: 'eegnet_cross_kappa', label: 'κ EEG×', fmt: (v) => kappa(v as number) },
]

function SubjectTable({ r, selected, onPick }: {
  r: DatasetResult
  selected: number | null
  onPick: (s: number) => void
}) {
  const [sortKey, setSortKey] = useState<keyof SubjectRow>('subject')
  const [asc, setAsc] = useState(true)
  const rows = useMemo(() => {
    const arr = [...(r.subjects ?? [])]
    arr.sort((a, b) => {
      const av = a[sortKey], bv = b[sortKey]
      if (av == null) return 1
      if (bv == null) return -1
      return asc ? (av as number) - (bv as number) : (bv as number) - (av as number)
    })
    return arr
  }, [r.subjects, sortKey, asc])

  // Solo mostramos columnas que tengan algún dato en este dataset.
  const cols = COLS.filter((c) => c.key === 'subject' || c.key === 'n_trials'
    || rows.some((row) => row[c.key] != null))

  const click = (k: keyof SubjectRow) => {
    if (k === sortKey) setAsc(!asc)
    else { setSortKey(k); setAsc(k === 'subject') }
  }

  return (
    <div className="overflow-x-auto">
      <table className="w-full text-sm">
        <thead>
          <tr className="border-b border-slate-200 text-left text-xs uppercase text-slate-400">
            {cols.map((c) => (
              <th key={String(c.key)} className="cursor-pointer select-none px-2 py-2 hover:text-slate-600"
                onClick={() => click(c.key)}>
                {c.label}{sortKey === c.key ? (asc ? ' ▲' : ' ▼') : ''}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((row) => (
            <tr key={row.subject}
              onClick={() => onPick(row.subject)}
              className={`cursor-pointer border-b border-slate-100 ${row.subject === selected ? 'bg-emerald-50' : 'hover:bg-slate-50'}`}>
              {cols.map((c) => (
                <td key={String(c.key)} className="px-2 py-1.5 tabular-nums text-slate-700">
                  {c.fmt(row[c.key])}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

// ---------------------------------------------------------------------------
// Ficha de detalle de un sujeto.
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// Comparación accuracy vs κ (kappa de Cohen).
// κ corrige el acierto por azar: con 2 clases BALANCEADAS, κ ≈ 2·acc − 1. Un punto
// muy por debajo de esa recta delata accuracy "inflada" por desbalance de clases o
// acuerdo casual, no por una decisión realmente informada. Sirve para comparar los
// 4 regímenes (CSP+LDA / EEGNet × within / cross) sujeto a sujeto en un solo vistazo.
// ---------------------------------------------------------------------------
const KAPPA_SERIES: {
  accKey: keyof SubjectRow; kapKey: keyof SubjectRow; name: string; color: string
}[] = [
  { accKey: 'csp_within_acc', kapKey: 'csp_within_kappa', name: 'CSP+LDA within', color: '#0284c7' },
  { accKey: 'csp_cross_acc', kapKey: 'csp_cross_kappa', name: 'CSP+LDA cross', color: '#7dd3fc' },
  { accKey: 'eegnet_within_acc', kapKey: 'eegnet_within_kappa', name: 'EEGNet within', color: '#7c3aed' },
  { accKey: 'eegnet_cross_acc', kapKey: 'eegnet_cross_kappa', name: 'EEGNet cross', color: '#c4b5fd' },
]

interface KappaPoint { acc: number; kappa: number; subject: number; series: string }

function KappaTooltip({ active, payload }: {
  active?: boolean; payload?: { payload: KappaPoint }[]
}) {
  if (!active || !payload?.length) return null
  const p = payload[0].payload
  return (
    <div className="rounded-md border border-slate-200 bg-white px-2.5 py-1.5 text-xs shadow-sm">
      <div className="font-medium text-slate-700">{p.series} · S{p.subject}</div>
      <div className="text-slate-500">acc {pct(p.acc)} · κ {kappa(p.kappa)}</div>
    </div>
  )
}

function AccuracyKappaScatter({ r }: { r: DatasetResult }) {
  const subjects = r.subjects ?? []
  const series = KAPPA_SERIES.map((s) => ({
    ...s,
    data: subjects
      .filter((row) => row[s.accKey] != null && row[s.kapKey] != null)
      .map((row): KappaPoint => ({
        acc: row[s.accKey] as number,
        kappa: row[s.kapKey] as number,
        subject: row.subject,
        series: s.name,
      })),
  })).filter((s) => s.data.length > 0)

  if (series.length === 0) return null

  const allPts = series.flatMap((s) => s.data)
  const minAcc = Math.min(0.5, ...allPts.map((p) => p.acc))
  const minKap = Math.min(0, ...allPts.map((p) => p.kappa))

  return (
    <Card title="Accuracy vs κ (kappa de Cohen)">
      <div className="h-72">
        <ResponsiveContainer width="100%" height="100%">
          <ScatterChart margin={{ top: 8, right: 12, bottom: 24, left: 0 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
            <XAxis
              type="number" dataKey="acc" name="accuracy"
              domain={[Math.floor(minAcc * 20) / 20, 1]}
              tickFormatter={(v) => `${Math.round(v * 100)}%`}
              tick={{ fontSize: 11 }}
              label={{ value: 'accuracy', position: 'insideBottom', offset: -12, fontSize: 11, fill: '#64748b' }}
            />
            <YAxis
              type="number" dataKey="kappa" name="κ"
              domain={[Math.min(0, Math.floor(minKap * 10) / 10), 1]}
              tick={{ fontSize: 11 }}
              label={{ value: 'κ', angle: -90, position: 'insideLeft', fontSize: 11, fill: '#64748b' }}
            />
            <ZAxis range={[55, 55]} />
            {/* Recta de balance perfecto: κ = 2·acc − 1 (acc=0.5⇒κ=0, acc=1⇒κ=1). */}
            <ReferenceLine
              segment={[{ x: 0.5, y: 0 }, { x: 1, y: 1 }]}
              stroke="#94a3b8" strokeDasharray="5 4"
              ifOverflow="extendDomain"
            />
            {/* κ = 0: acuerdo no mejor que el azar. */}
            <ReferenceLine y={0} stroke="#cbd5e1" />
            <Tooltip content={<KappaTooltip />} cursor={{ strokeDasharray: '3 3' }} />
            <Legend verticalAlign="top" align="center" wrapperStyle={{ fontSize: 11, paddingBottom: 8 }} iconSize={9} />
            {series.map((s) => (
              <Scatter key={s.name} name={s.name} data={s.data} fill={s.color} />
            ))}
          </ScatterChart>
        </ResponsiveContainer>
      </div>
      <p className="mt-2 text-xs text-slate-400">
        Cada punto es un sujeto. La recta discontinua es <span className="font-mono">κ = 2·acc − 1</span>,
        la relación esperada con clases balanceadas: puntos por debajo señalan accuracy «inflada»
        por desbalance o acuerdo casual; cuanto más alto y a la derecha, mejor.
      </p>
    </Card>
  )
}

function SubjectDetail({ r, subject }: { r: DatasetResult; subject: number }) {
  const s = (r.subjects ?? []).find((x) => x.subject === subject)
  if (!s) return null
  const items: { label: string; v: string }[] = [
    { label: 'Trials', v: s.n_trials == null ? '—' : String(s.n_trials) },
    { label: 'CSP+LDA within (acc / κ)', v: `${pct(s.csp_within_acc)} / ${kappa(s.csp_within_kappa)}` },
    ...(r.has_intersession ? [{ label: 'CSP+LDA inter-sesión (acc / κ)', v: `${pct(s.csp_inter_acc)} / ${kappa(s.csp_inter_kappa)}` }] : []),
    ...(r.has_compare ? [
      { label: 'CSP+LDA cross (LOSO)', v: pct(s.csp_cross_acc) },
      { label: 'EEGNet within', v: pct(s.eegnet_within_acc) },
      { label: 'EEGNet cross (LOSO)', v: pct(s.eegnet_cross_acc) },
    ] : []),
  ]
  const best = s.csp_within_acc
  const illiteracy = best != null && best < r.chance + 0.1
  return (
    <Card title={`Ficha del sujeto S${subject}`}>
      <dl className="grid grid-cols-2 gap-x-4 gap-y-2 text-sm">
        {items.map((it) => (
          <div key={it.label} className="flex flex-col">
            <dt className="text-xs text-slate-400">{it.label}</dt>
            <dd className="font-medium tabular-nums text-slate-700">{it.v}</dd>
          </div>
        ))}
      </dl>
      {illiteracy && (
        <p className="mt-3 rounded-lg bg-amber-50 px-3 py-2 text-xs text-amber-700">
          Rendimiento cercano al azar ({pct(r.chance)}): posible caso de «BCI illiteracy»
          (desincronización µ/β débil), no necesariamente un fallo del método.
        </p>
      )}
    </Card>
  )
}

// ---------------------------------------------------------------------------
// Ficha del modelo pooled (generalización cross-subject) + provenance.
// ---------------------------------------------------------------------------
function PooledCard({ r }: { r: DatasetResult }) {
  const p = r.pooled
  if (!p) return null
  const prov: { label: string; v: string }[] = [
    { label: 'Media LOSO (cross-subject)', v: pct(p.loso_mean) },
    { label: 'Sujetos en el pool', v: String(p.n_subjects ?? '—') },
    { label: 'Trials de entrenamiento', v: String(p.n_train ?? '—') },
    { label: 'Épocas', v: String(p.epochs ?? '—') },
    { label: 'Aumentación', v: p.augment ? `sí (×${p.augment_copies})` : 'no' },
    { label: 'Dispositivo', v: p.device ?? '—' },
    { label: 'Banda FIR', v: p.fir ? `${p.fir.low_hz}–${p.fir.high_hz} Hz` : '—' },
    { label: 'Canales', v: String(p.n_channels ?? '—') },
    { label: 'Entrenado', v: p.trained_on ?? '—' },
  ]
  return (
    <Card title="Modelo EEGNet pooled · generalización a usuario nuevo">
      <dl className="grid grid-cols-2 gap-x-4 gap-y-2 text-sm sm:grid-cols-3">
        {prov.map((it) => (
          <div key={it.label} className="flex flex-col">
            <dt className="text-xs text-slate-400">{it.label}</dt>
            <dd className="font-medium tabular-nums text-slate-700">{it.v}</dd>
          </div>
        ))}
      </dl>
      <p className="mt-3 text-xs text-slate-500">
        El modelo base se entrena con TODOS los sujetos (punto de partida para un
        fine-tuning con calibración corta). La media LOSO es la estimación honesta de
        ponérselo a alguien que el modelo nunca vio, sin calibrar.
      </p>
    </Card>
  )
}

// ---------------------------------------------------------------------------
// Página.
// ---------------------------------------------------------------------------
export default function Results() {
  const [index, setIndex] = useState<DatasetResult[] | null>(null)
  const [aggregate, setAggregate] = useState<AggregateResult | null>(null)
  const [selected, setSelected] = useState<string>('')
  const [detail, setDetail] = useState<DatasetResult | null>(null)
  const [subject, setSubject] = useState<number | null>(null)
  const [metric, setMetric] = useState<Metric>('csp_within_acc')
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    // Resultados = benchmark de métodos sobre TODOS los datasets (cada uno autosuficiente
    // con sus 4 regímenes). Los de ≥2 sesiones (etiquetados «demo en vivo») aparecen aquí
    // como benchmark de población Y además en la sección «Demo en vivo» (su modelo
    // desplegado y su partición 0train→1test se ven allí).
    fetchResultsIndex()
      .then((all) => {
        setIndex(all)
        setSelected((cur) => cur || all[0]?.id || '')
      })
      .catch((e) => setError(String(e)))
    fetchAggregate().then(setAggregate).catch(() => { /* opcional */ })
  }, [])

  useEffect(() => {
    if (!selected) return
    setDetail(null)
    setSubject(null)
    fetchDatasetResult(selected).then((d) => {
      setDetail(d)
      // métrica por defecto: la cross si existe, si no la within
      setMetric(d.has_compare ? 'csp_cross_acc' : 'csp_within_acc')
    }).catch((e) => setError(String(e)))
  }, [selected])

  const availMetrics = useMemo<Metric[]>(() => {
    if (!detail) return ['csp_within_acc']
    return (Object.keys(METRIC_LABEL) as Metric[]).filter(
      (m) => (detail.subjects ?? []).some((s) => s[m] != null))
  }, [detail])

  return (
    <PageShell
      title="Resultados"
      help={HELP}
      world="offline"
    >
      {error && <p className="mb-4 rounded-lg bg-red-50 px-3 py-2 text-sm text-red-600">Error: {error}</p>}

      <div className="mb-4"><DatasetRolesNote /></div>

      <div className="mb-4"><ResultInterpretation /></div>

      {aggregate && <div className="mb-4"><AggregateMatrix a={aggregate} /></div>}

      <div className="grid grid-cols-1 gap-4 lg:grid-cols-[320px_1fr]">
        {/* Columna izquierda: resumen / selector */}
        <div className="space-y-4">
          {index ? (
            <Overview index={index} selected={selected} onSelect={setSelected} />
          ) : (
            <Card title="Resumen"><Skeleton /></Card>
          )}
        </div>

        {/* Columna derecha: detalle del dataset seleccionado */}
        <div className="space-y-4">
          {detail ? (
            <>
              <MatrixTable r={detail} />

              <ComparisonMetrics r={detail} />

              {(detail.subjects?.length ?? 0) > 0 && (
                <Card
                  title="Resultados por sujeto"
                  right={
                    <select
                      value={metric}
                      onChange={(e) => setMetric(e.target.value as Metric)}
                      className="rounded-md border border-slate-200 px-2 py-1 text-xs text-slate-600"
                    >
                      {availMetrics.map((m) => (
                        <option key={m} value={m}>{METRIC_LABEL[m]}</option>
                      ))}
                    </select>
                  }
                >
                  <SubjectChart r={detail} metric={metric} onPick={setSubject} selected={subject} />
                  <MetricStats r={detail} metric={metric} />
                  <div className="mt-4">
                    <SubjectTable r={detail} selected={subject} onPick={setSubject} />
                  </div>
                  <p className="mt-2 text-xs text-slate-400">Clic en una barra o fila para ver la ficha del sujeto.</p>
                </Card>
              )}

              {(detail.subjects?.length ?? 0) > 0 && <AccuracyKappaScatter r={detail} />}

              {subject != null && <SubjectDetail r={detail} subject={subject} />}

              <PooledCard r={detail} />
            </>
          ) : (
            <Card title="Detalle"><Skeleton /></Card>
          )}
        </div>
      </div>
    </PageShell>
  )
}

function Skeleton() {
  return <div className="h-32 animate-pulse rounded-lg bg-slate-100" />
}
