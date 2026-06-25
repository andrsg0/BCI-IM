import { useEffect, useState } from 'react'
import { Radio, Database } from 'lucide-react'
import { fetchResultsIndex, type DatasetResult } from '../lib/results'

/** Explica cómo se usan los datasets y muestra, con datos reales del backend, qué tan
 *  completo está cada uno. Cada dataset es autosuficiente: TODOS se benchmarkean (4
 *  regímenes); los de ≥2 sesiones sirven ADEMÁS para la demo en vivo (inter-sesión).
 *  Hace visibles los huecos (p. ej. datasets aún sin la comparación EEGNet). */
export function DatasetRolesNote() {
  const [items, setItems] = useState<DatasetResult[] | null>(null)
  useEffect(() => { fetchResultsIndex().then(setItems).catch(() => setItems([])) }, [])
  if (!items || items.length === 0) return null

  const live = items.filter((d) => d.live)
  const train = items   // todos se benchmarkean (cada dataset es autosuficiente)

  // Estado de completitud para los datasets de benchmark.
  const completeness = (d: DatasetResult) => {
    if (d.has_compare) return { txt: 'CSP + EEGNet (2×2)', cls: 'bg-emerald-100 text-emerald-700' }
    if (d.n_subjects_evaluated > 0) return { txt: 'solo CSP within', cls: 'bg-amber-100 text-amber-700' }
    return { txt: 'sin medir', cls: 'bg-slate-100 text-slate-500' }
  }

  return (
    <div className="grid gap-3 sm:grid-cols-2">
      {/* EN VIVO -------------------------------------------------------------- */}
      <div className="rounded-xl border border-emerald-200 bg-emerald-50/40 p-3">
        <div className="mb-1 flex items-center gap-2 text-sm font-semibold text-emerald-800">
          <Radio size={15} /> Demo en vivo (calibrar un día → probar otro)
        </div>
        <p className="mb-2 text-[11px] leading-snug text-emerald-700">
          Requisito: <strong>≥ 2 sesiones</strong> (días distintos) para una estimación honesta inter-sesión.
        </p>
        <ul className="space-y-1 text-xs text-slate-600">
          {live.map((d) => (
            <li key={d.id} className="flex items-center justify-between gap-2">
              <span className="font-medium">{d.label}</span>
              <span className="text-slate-400">{d.n_subjects_declared ?? '?'} sujetos · {d.sessions} sesiones</span>
            </li>
          ))}
          {live.length === 0 && <li className="text-slate-400">—</li>}
        </ul>
        <p className="mt-2 text-[10px] text-emerald-600">
          Aparecen también en el benchmark de al lado (cada dataset es autosuficiente).
        </p>
      </div>

      {/* BENCHMARK DE POBLACIÓN (todos) ------------------------------------- */}
      <div className="rounded-xl border border-amber-200 bg-amber-50/40 p-3">
        <div className="mb-1 flex items-center gap-2 text-sm font-semibold text-amber-800">
          <Database size={15} /> Benchmark de población (todos)
        </div>
        <p className="mb-2 text-[11px] leading-snug text-amber-700">
          <strong>Todos</strong> los datasets se evalúan aquí: comparan métodos
          (within-subject vs cross-subject, CSP+LDA vs EEGNet) sobre su población.
        </p>
        <ul className="space-y-1 text-xs text-slate-600">
          {train.map((d) => {
            const c = completeness(d)
            return (
              <li key={d.id} className="flex items-center justify-between gap-2">
                <span className="font-medium">
                  {d.label}{' '}
                  <span className="text-slate-400">({d.n_subjects_evaluated}/{d.n_subjects_declared ?? '?'})</span>
                </span>
                <span className={`shrink-0 rounded px-1.5 py-0.5 text-[10px] ${c.cls}`}>{c.txt}</span>
              </li>
            )
          })}
          {train.length === 0 && <li className="text-slate-400">—</li>}
        </ul>
      </div>
    </div>
  )
}
