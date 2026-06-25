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
  pipeline: 'El modelo que clasifica en vivo',
  intro: 'Muestra la "ficha" honesta del modelo CSP+LDA que se usa en la página de Clasificación. La clave es que el modelo se entrena ANTES de la demo y la demo solo reproduce trials que el modelo NUNCA vio durante el entrenamiento (held-out). Así la precisión que ves es una estimación realista de cómo rendiría en vivo.',
  points: [
    { label: 'Train vs held-out', desc: 'El modelo se ajusta solo con la partición de entrenamiento. Los trials reservados (held-out) son los que la Clasificación reproduce: el modelo no los vio, así que su acierto sobre ellos no está inflado.' },
    { label: 'Inter-sesión', desc: 'Si el dataset tiene 2 sesiones, se entrena con la primera y se evalúa con la segunda («otro día»): la estimación más honesta de uso real. Si solo hay 1 sesión, se reserva un 30% estratificado.' },
    { label: 'Por qué difiere del benchmark', desc: 'La sección Resultados compara métodos por validación cruzada (within/cross). Aquí es UN modelo concreto desplegado y su acierto sobre los trials reales que reproduce la demo.' },
  ],
  terms: ['Accuracy y kappa', 'CSP', 'Validación cruzada'],
}

interface ModelCard {
  dataset: string
  subject: number
  method: string
  fs: number
  classes: string[]
  channels: string[]
  holdout: { by: string; value?: string; indices?: number[] }
  train_session: string | null
  n_train: number
  n_demo: number
  accuracy: number
  kappa: number
  trained_on: string
  n_components: number
  fir: { low_hz: number; high_hz: number; num_taps: number }
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

interface LiveDataset { id: string; label: string; subjects: number }

export default function LiveResults() {
  const { dataset, subject, setDataset, setSubject } = useStore()
  const [card, setCard] = useState<ModelCard | null>(null)
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
        // Si lo seleccionado no es de demo en vivo, saltar al primero que sí lo sea.
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
    getJSON<ModelCard>(`/model?dataset=${dataset}&subject=${subject}&method=csp_lda`)
      .then((c) => { setCard(c); setLoading(false) })
      .catch((e) => { setError(String(e)); setLoading(false) })
  }, [dataset, subject])

  const current = liveList.find((d) => d.id === dataset)
  const nSubjects = current?.subjects ?? DATASETS[dataset]?.subjects ?? 9
  const datasetLabel = current?.label ?? DATASETS[dataset]?.label ?? dataset
  const chance = card ? 1 / Math.max(card.classes.length, 2) : 0.5

  // Texto del tipo de partición (inter-sesión vs hold-out 30%).
  const holdoutLabel = card
    ? card.holdout.by === 'session'
      ? `Inter-sesión (entrena en ${card.train_session ?? '0train'}, evalúa en ${card.holdout.value})`
      : 'Hold-out estratificado 30% (dataset de 1 sola sesión)'
    : ''

  const total = card ? card.n_train + card.n_demo : 0

  return (
    <PageShell
      title="Demo en vivo"
      subtitle="Ficha honesta del modelo desplegado que clasifica en la página de Clasificación."
      help={HELP}
      world="online"
    >
      <div className="mb-4"><DatasetRolesNote /></div>

      <div className="mb-4 flex items-center gap-3">
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

      {error && <p className="mb-4 rounded-lg bg-red-50 px-3 py-2 text-sm text-red-600">Error: {error}</p>}
      {loading && <div className="h-40 animate-pulse rounded-xl bg-slate-100" />}

      {card && !loading && (
        <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
          {/* Acierto honesto sobre held-out */}
          <Card title="Acierto honesto (held-out)" right={<ShieldCheck size={18} className="text-emerald-500" />}>
            <div className="flex items-end gap-6">
              <div>
                <div className="text-4xl font-bold text-slate-800">{pct(card.accuracy)}</div>
                <div className="text-xs text-slate-400">accuracy · azar {pct(chance)}</div>
              </div>
              <div>
                <div className="text-2xl font-semibold text-slate-700">{fmtKappa(card.kappa)}</div>
                <div className="text-xs text-slate-400">kappa (κ)</div>
              </div>
            </div>
            {/* barra vs azar */}
            <div className="relative mt-4 h-3 w-full rounded-full bg-slate-100">
              <div className="absolute top-0 h-3 rounded-full bg-emerald-500" style={{ width: `${card.accuracy * 100}%` }} />
              <div className="absolute top-[-3px] h-[18px] w-px bg-slate-400" style={{ left: `${chance * 100}%` }} title={`azar ${pct(chance)}`} />
            </div>
            <p className="mt-3 text-xs text-slate-500">{holdoutLabel}.</p>
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
              <span>Demo en vivo / held-out ({card.n_demo}) <span className="inline-block h-2 w-2 rounded-full bg-emerald-500" /></span>
            </div>
            <p className="mt-3 text-xs text-slate-500">
              La página de Clasificación reproduce en bucle solo los <strong>{card.n_demo}</strong> trials
              reservados: el modelo nunca los vio al entrenar.
            </p>
          </Card>

          {/* Detalles del modelo */}
          <Card title="Cómo está construido">
            <dl className="grid grid-cols-2 gap-x-4 gap-y-2 text-sm sm:grid-cols-3">
              <Field label="Método" value="CSP + LDA" />
              <Field label="Clases" value={card.classes.join(' vs ')} />
              <Field label="Frecuencia" value={`${card.fs} Hz`} />
              <Field label="Banda FIR" value={`${card.fir.low_hz}–${card.fir.high_hz} Hz`} />
              <Field label="Coef. FIR" value={String(card.fir.num_taps)} />
              <Field label="Componentes CSP" value={String(card.n_components)} />
              <Field label="Canales" value={String(card.channels.length)} />
              <Field label="Entrenado" value={card.trained_on} />
            </dl>
          </Card>

          {/* Nota didáctica */}
          <Card title="Lectura honesta">
            <p className="text-sm text-slate-600">
              Este es el clasificador <strong>sujeto-específico</strong> (calibrado en esta persona)
              que corre la demo. Su acierto sobre el held-out estima el rendimiento real con ese
              usuario. Para comparar métodos (CSP+LDA vs EEGNet) y escenarios (within vs cross-subject)
              entre toda la población, ve a la sección <strong>Resultados</strong>.
            </p>
          </Card>
        </div>
      )}
    </PageShell>
  )
}
