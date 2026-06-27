import { useEffect, useState } from 'react'
import { ShieldCheck } from 'lucide-react'
import { PageShell } from '../components/PageShell'
import type { HelpContent } from '../components/HelpButton'
import { useStore } from '../store/useStore'
import { DatasetRolesNote } from '../components/DatasetRolesNote'
import { getJSON } from '../api/client'
import { DATASETS, type DatasetId } from '../lib/datasets'
import { pct, kappa as fmtKappa, fetchResultsIndex } from '../lib/results'

const HELP: HelpContent = {
  pipeline: 'Benchmark por sujeto · los 4 regímenes sobre el held-out',
  intro: 'Compara, para UN sujeto, los cuatro regímenes del sistema (CSP+LDA y EEGNet × within/cross) sobre la partición que el modelo NUNCA vio (held-out). Cada número sale de evaluar el modelo ya entrenado: no se reentrena nada aquí. A diferencia de Resultados (que promedia toda la población), esto es el detalle honesto de una sola persona, incluida su matriz de confusión.',
  points: [
    { label: 'Train vs held-out', desc: 'El modelo se ajusta solo con la partición de entrenamiento. Los trials reservados (held-out) son los que se evalúan aquí (y los que reproduce la Clasificación en vivo): el modelo no los vio, así que su acierto sobre ellos no está inflado.' },
    { label: 'Within vs cross', desc: 'Within-subject está calibrado en esta misma persona (techo). Cross-subject entrena con OTROS sujetos y evalúa en este (usuario nuevo sin calibrar): siempre rinde más bajo. EEGNet aprende sus filtros; CSP+LDA los diseña a mano.' },
    { label: 'Matriz de confusión', desc: 'Filas = clase real, columnas = clase predicha. La diagonal son los aciertos; fuera de ella, los errores. Permite ver si el modelo confunde más una mano que la otra (sesgo), algo que la accuracy sola esconde.' },
    { label: 'Por qué difiere del benchmark', desc: 'La sección Resultados compara métodos por validación cruzada sobre toda la población. Aquí es UN sujeto concreto y el acierto real del modelo desplegado sobre sus trials reservados.' },
  ],
  terms: ['Accuracy, Matriz de confusión y Kappa de Cohen', 'CSP', 'EEGNet', 'Validación inter-sesión'],
}

// Los 4 regímenes (within/cross × CSP/EEGNet). Los cross deben estar pre-entrenados.
const METHODS = [
  { id: 'csp_lda', label: 'CSP+LDA', regime: 'within' },
  { id: 'csp_lda_cross', label: 'CSP+LDA', regime: 'cross' },
  { id: 'eegnet', label: 'EEGNet', regime: 'within' },
  { id: 'eegnet_cross', label: 'EEGNet', regime: 'cross' },
] as const

const CLASS_COLORS = ['#2563eb', '#e11d48', '#059669', '#d97706']

interface ModelCard {
  dataset: string; subject: number; method: string; fs: number
  classes: string[]; channels: string[]
  holdout: { by: string; value?: string; indices?: number[] }
  train_session: string | null
  n_train: number; n_demo: number; accuracy: number; kappa: number; trained_on: string
  n_components: number; fir: { low_hz: number; high_hz: number; num_taps: number }
  extra?: { n_train_subjects?: number } | null
}

interface EvalResp {
  dataset: string; subject: number; method: string
  classes: string[]; confusion: { labels: string[]; matrix: number[][] }
  accuracy: number; kappa: number; n_eval: number; holdout_kind: string
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

function Field({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex flex-col">
      <dt className="text-xs text-slate-400">{label}</dt>
      <dd className="font-medium tabular-nums text-slate-700">{value}</dd>
    </div>
  )
}

/** Matriz de confusión: filas = clase real, columnas = clase predicha. La intensidad
 *  del fondo crece con el conteo (verde en la diagonal, rojo fuera). */
function ConfusionMatrix({ conf }: { conf: { labels: string[]; matrix: number[][] } }) {
  const { labels, matrix } = conf
  const max = Math.max(1, ...matrix.flat())
  const bg = (v: number, diag: boolean) => {
    const a = v / max
    return diag ? `rgba(16,185,129,${0.12 + 0.5 * a})` : `rgba(244,63,94,${0.08 + 0.45 * a})`
  }
  return (
    <div className="overflow-x-auto">
      <table className="text-sm">
        <thead>
          <tr>
            <th className="px-2 py-1" />
            <th className="px-2 py-1 text-center text-[10px] font-medium uppercase tracking-wide text-slate-400" colSpan={labels.length}>
              Predicho →
            </th>
          </tr>
          <tr>
            <th className="px-2 py-1 text-left text-xs text-slate-400">Real ↓</th>
            {labels.map((l) => (
              <th key={l} className="px-3 py-1 text-center text-xs font-medium" style={{ color: CLASS_COLORS[labels.indexOf(l) % CLASS_COLORS.length] }}>{l}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {matrix.map((row, i) => {
            const total = row.reduce((s, v) => s + v, 0)
            return (
              <tr key={i}>
                <td className="px-2 py-1 text-xs font-medium" style={{ color: CLASS_COLORS[i % CLASS_COLORS.length] }}>{labels[i]}</td>
                {row.map((v, j) => (
                  <td key={j} className="px-3 py-2 text-center tabular-nums"
                    style={{ background: bg(v, i === j) }}
                    title={`real ${labels[i]} → predicho ${labels[j]}: ${v}`}>
                    <div className="font-semibold text-slate-800">{v}</div>
                    <div className="text-[10px] text-slate-500">{total ? `${Math.round((v / total) * 100)}%` : '—'}</div>
                  </td>
                ))}
              </tr>
            )
          })}
        </tbody>
      </table>
    </div>
  )
}

interface LiveDataset { id: string; label: string; subjects: number }

export default function LiveResults() {
  const { dataset, subject, setDataset, setSubject } = useStore()
  const [method, setMethod] = useState<string>('csp_lda')
  const [card, setCard] = useState<ModelCard | null>(null)
  const [evalData, setEvalData] = useState<EvalResp | null>(null)
  const [liveList, setLiveList] = useState<LiveDataset[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  // Solo datasets aptos para la demo en vivo (≥2 sesiones ⇒ d.live) y presentes en el selector (DATASETS).
  useEffect(() => {
    fetchResultsIndex()
      .then((all) => {
        const live = all
          .filter((d) => d.live && d.id in DATASETS)
          .map((d) => ({ id: d.id, label: d.label, subjects: d.n_subjects_declared ?? 9 }))
        setLiveList(live)
        if (live.length && !live.some((d) => d.id === dataset)) {
          setDataset(live[0].id as DatasetId)
        }
      })
      .catch(() => { /* el índice puede fallar; seguimos con el store */ })
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  useEffect(() => {
    setLoading(true)
    setError(null)
    setCard(null)
    setEvalData(null)
    Promise.all([
      getJSON<ModelCard>(`/model?dataset=${dataset}&subject=${subject}&method=${method}`),
      getJSON<EvalResp>(`/eval?dataset=${dataset}&subject=${subject}&method=${method}`),
    ])
      .then(([c, e]) => { setCard(c); setEvalData(e); setLoading(false) })
      .catch((err) => { setError(String(err)); setLoading(false) })
  }, [dataset, subject, method])

  const current = liveList.find((d) => d.id === dataset)
  const nSubjects = current?.subjects ?? DATASETS[dataset]?.subjects ?? 9
  const datasetLabel = current?.label ?? DATASETS[dataset]?.label ?? dataset
  const chance = card ? 1 / Math.max(card.classes.length, 2) : 0.5
  const methodInfo = METHODS.find((m) => m.id === method)!

  // Texto del tipo de partición.
  const holdoutLabel = card
    ? card.holdout.by === 'subject'
      ? `Cross-subject: entrenado con ${card.extra?.n_train_subjects ?? 'otros'} sujetos, evaluado en el ${card.subject} (usuario nuevo)`
      : card.holdout.by === 'session'
        ? `Inter-sesión: entrena en ${card.train_session ?? '0train'}, evalúa en ${card.holdout.value}`
        : 'Hold-out estratificado 30% (dataset de 1 sola sesión)'
    : ''

  const total = card ? card.n_train + card.n_demo : 0
  const demoLabel = card?.holdout.by === 'subject' ? 'Sujeto evaluado / held-out' : 'Demo en vivo / held-out'

  return (
    <PageShell
      title="Benchmark"
      subtitle="Comparación honesta por sujeto: los 4 regímenes (CSP+LDA y EEGNet × within/cross) sobre los trials reservados (held-out)."
      help={HELP}
      world="online"
    >
      <div className="mb-4"><DatasetRolesNote /></div>

      <div className="mb-4 flex flex-wrap items-center gap-3">
        <span className="text-sm text-slate-500">Modelo de:</span>
        {liveList.length > 1 ? (
          <select
            value={dataset}
            onChange={(e) => setDataset(e.target.value as DatasetId)}
            className="rounded-md border border-slate-200 px-2 py-1 text-sm font-medium text-slate-700"
          >
            {liveList.map((d) => <option key={d.id} value={d.id}>{d.label}</option>)}
          </select>
        ) : (
          <span className="rounded-lg bg-slate-100 px-3 py-1 text-sm font-medium text-slate-700">{datasetLabel}</span>
        )}
        <label className="text-sm text-slate-500">Sujeto</label>
        <select
          value={subject}
          onChange={(e) => setSubject(Number(e.target.value))}
          className="rounded-md border border-slate-200 px-2 py-1 text-sm text-slate-700"
        >
          {Array.from({ length: nSubjects }, (_, i) => i + 1).map((s) => (
            <option key={s} value={s}>S{s}</option>
          ))}
        </select>
        <span className="text-xs text-slate-400">(también se controla desde el panel lateral)</span>
      </div>

      {/* Selector de los 4 regímenes. */}
      <div className="mb-4 flex flex-wrap items-center gap-2 text-sm">
        <span className="text-xs font-semibold text-slate-500">Régimen:</span>
        {METHODS.map((m) => (
          <button
            key={m.id}
            onClick={() => setMethod(m.id)}
            className={`rounded-full px-3 py-1 text-xs transition ${
              method === m.id ? 'bg-slate-800 text-white' : 'bg-slate-100 text-slate-600 hover:bg-slate-200'
            }`}
          >
            {m.label} · {m.regime}
          </button>
        ))}
      </div>

      {error && (
        <p className="mb-4 rounded-lg border border-amber-200 bg-amber-50 px-3 py-2 text-sm text-amber-700">
          No se pudo evaluar el régimen <strong>{methodInfo.label} · {methodInfo.regime}</strong> para este sujeto.
          {' '}Probablemente no esté entrenado: córrelo con <code>scripts/train_all_regimes.py</code>.
        </p>
      )}
      {loading && <div className="h-40 animate-pulse rounded-xl bg-slate-100" />}

      {card && evalData && !loading && (
        <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
          {/* Acierto honesto sobre held-out */}
          <Card title="Acierto honesto (held-out)" right={<ShieldCheck size={18} className="text-emerald-500" />}>
            <div className="flex items-end gap-6">
              <div>
                <div className="text-4xl font-bold text-slate-800">{pct(evalData.accuracy)}</div>
                <div className="text-xs text-slate-400">accuracy · azar {pct(chance)}</div>
              </div>
              <div>
                <div className="text-2xl font-semibold text-slate-700">{fmtKappa(evalData.kappa)}</div>
                <div className="text-xs text-slate-400">kappa (κ)</div>
              </div>
              <div>
                <div className="text-2xl font-semibold text-slate-700">{evalData.n_eval}</div>
                <div className="text-xs text-slate-400">trials evaluados</div>
              </div>
            </div>
            {/* barra vs azar */}
            <div className="relative mt-4 h-3 w-full rounded-full bg-slate-100">
              <div className="absolute top-0 h-3 rounded-full bg-emerald-500" style={{ width: `${evalData.accuracy * 100}%` }} />
              <div className="absolute top-[-3px] h-[18px] w-px bg-slate-400" style={{ left: `${chance * 100}%` }} title={`azar ${pct(chance)}`} />
            </div>
            <p className="mt-3 text-xs text-slate-500">{holdoutLabel}.</p>
          </Card>

          {/* Matriz de confusión */}
          <Card title="Matriz de confusión (held-out)">
            <ConfusionMatrix conf={evalData.confusion} />
            <p className="mt-3 text-xs text-slate-500">
              Diagonal = aciertos; fuera de ella, los errores. El porcentaje es por fila (sobre cada clase real).
            </p>
          </Card>

          {/* Partición train / demo */}
          <Card title="Partición de datos">
            <div className="mb-2 flex h-6 w-full overflow-hidden rounded-lg text-xs font-medium text-white">
              <div className="flex items-center justify-center bg-amber-500" style={{ width: `${total ? (card.n_train / total) * 100 : 0}%` }}>
                {card.n_train}
              </div>
              <div className="flex items-center justify-center bg-emerald-500" style={{ width: `${total ? (card.n_demo / total) * 100 : 0}%` }}>
                {card.n_demo}
              </div>
            </div>
            <div className="flex justify-between text-xs text-slate-500">
              <span><span className="inline-block h-2 w-2 rounded-full bg-amber-500" /> Entrenamiento ({card.n_train})</span>
              <span>{demoLabel} ({card.n_demo}) <span className="inline-block h-2 w-2 rounded-full bg-emerald-500" /></span>
            </div>
            <p className="mt-3 text-xs text-slate-500">
              El modelo se ajustó solo con la partición de entrenamiento; los <strong>{card.n_demo}</strong> trials
              reservados (que nunca vio) son los que se evalúan aquí.
            </p>
          </Card>

          {/* Detalles del modelo */}
          <Card title="Cómo está construido">
            <dl className="grid grid-cols-2 gap-x-4 gap-y-2 text-sm sm:grid-cols-3">
              <Field label="Método" value={`${methodInfo.label} · ${methodInfo.regime}`} />
              <Field label="Clases" value={card.classes.join(' vs ')} />
              <Field label="Frecuencia" value={`${card.fs} Hz`} />
              <Field label="Banda FIR" value={`${card.fir.low_hz}–${card.fir.high_hz} Hz`} />
              <Field label="Coef. FIR" value={String(card.fir.num_taps)} />
              <Field label={methodInfo.label === 'EEGNet' ? 'Filtros temporales' : 'Componentes CSP'} value={String(card.n_components)} />
              <Field label="Canales" value={String(card.channels.length)} />
              <Field label="Entrenado" value={card.trained_on} />
            </dl>
          </Card>
        </div>
      )}
    </PageShell>
  )
}
