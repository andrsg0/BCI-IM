import { useEffect, useState } from 'react'
import { Radio } from 'lucide-react'
import { fetchResultsIndex, type DatasetResult } from '../lib/results'

/** Lista los datasets que ADEMÁS sirven para la demo en vivo (≥2 sesiones, estimación
 *  honesta inter-sesión). El benchmark de población lo cubren todos los datasets en las
 *  tablas de abajo, así que aquí solo se destacan los aptos para la demo. */
export function DatasetRolesNote() {
  const [items, setItems] = useState<DatasetResult[] | null>(null)
  useEffect(() => { fetchResultsIndex().then(setItems).catch(() => setItems([])) }, [])
  if (!items || items.length === 0) return null

  const live = items.filter((d) => d.live)
  if (live.length === 0) return null

  return (
    <div className="rounded-xl border border-emerald-200 bg-emerald-50/40 p-3">
      <div className="mb-2 flex items-center gap-2 text-sm font-semibold text-emerald-800">
        <Radio size={15} /> Demo en vivo
      </div>
      <ul className="grid gap-x-6 gap-y-1 text-xs text-slate-600 sm:grid-cols-2">
        {live.map((d) => (
          <li key={d.id} className="flex items-center justify-between gap-2">
            <span className="font-medium">{d.label}</span>
            <span className="text-slate-400">{d.n_subjects_declared ?? '?'} sujetos · {d.sessions} sesiones</span>
          </li>
        ))}
      </ul>
    </div>
  )
}
