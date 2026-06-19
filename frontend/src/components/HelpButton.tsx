import { useEffect, useMemo, useState } from 'react'
import { HelpCircle, X } from 'lucide-react'
import { Link } from 'react-router-dom'
import { useGlossary, findEntry, previewText, type GlossaryEntry } from '../lib/glossary'

export interface HelpContent {
  /** etapa del pipeline a la que pertenece la sección (badge superior) */
  pipeline?: string
  /** qué pasa en esta sección */
  intro: string
  /** qué significa cada cosa / cómo se configura / para qué sirve */
  points: { label: string; desc: string }[]
  /** términos importantes (se enlazan al glosario) */
  terms?: string[]
}

/** Chip de término clave: enlaza al glosario y muestra la definición al pasar el cursor. */
function GlossaryChip({ term, entries, onNavigate }: { term: string; entries: GlossaryEntry[]; onNavigate: () => void }) {
  const entry = useMemo(() => findEntry(term, entries), [term, entries])
  return (
    <span className="group relative inline-block">
      <Link to={`/glossary?q=${encodeURIComponent(term)}`} onClick={onNavigate}
        className="block rounded-full bg-primary/10 px-2.5 py-1 text-xs text-primary hover:bg-primary/20">
        {term}
      </Link>
      {entry && (
        <span className="pointer-events-none absolute bottom-full left-1/2 z-50 mb-2 hidden w-64 -translate-x-1/2 group-hover:block">
          <span className="block rounded-lg bg-slate-900 px-3 py-2 text-left text-xs leading-snug text-slate-100 shadow-xl">
            <span className="mb-0.5 block font-semibold text-white">{entry.term}</span>
            {previewText(entry.body)}
          </span>
        </span>
      )}
    </span>
  )
}

/** Botón "?" que abre un panel explicando la sección actual. */
export function HelpButton({ title, help }: { title: string; help: HelpContent }) {
  const [open, setOpen] = useState(false)
  const entries = useGlossary()

  useEffect(() => {
    if (!open) return
    const onKey = (e: KeyboardEvent) => e.key === 'Escape' && setOpen(false)
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [open])

  return (
    <>
      <button
        onClick={() => setOpen(true)}
        title="¿Qué es esta sección?"
        className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full border border-slate-300 text-slate-500 hover:bg-primary/10 hover:text-primary"
      >
        <HelpCircle size={18} />
      </button>

      {open && (
        <div className="fixed inset-0 z-50 flex items-start justify-center overflow-auto bg-slate-900/30 p-4 sm:p-8" onClick={() => setOpen(false)}>
          <div className="my-auto w-full max-w-lg rounded-2xl bg-white p-6 shadow-xl" onClick={(e) => e.stopPropagation()}>
            <div className="mb-3 flex items-start justify-between gap-4">
              <h2 className="text-lg font-bold text-slate-800">{title}</h2>
              <button onClick={() => setOpen(false)} className="rounded-md p-1 text-slate-400 hover:bg-slate-100"><X size={18} /></button>
            </div>

            {help.pipeline && (
              <div className="mb-3 inline-block rounded-full bg-primary/10 px-3 py-1 text-xs font-medium text-primary">
                {help.pipeline}
              </div>
            )}

            <p className="text-sm leading-relaxed text-slate-600">{help.intro}</p>

            <ul className="mt-4 space-y-3">
              {help.points.map((p, i) => (
                <li key={i} className="text-sm leading-relaxed text-slate-600">
                  <span className="font-semibold text-slate-700">{p.label}.</span> {p.desc}
                </li>
              ))}
            </ul>

            {help.terms && help.terms.length > 0 && (
              <div className="mt-5 border-t border-slate-100 pt-4">
                <div className="mb-2 text-xs font-semibold uppercase tracking-wide text-slate-400">Términos clave</div>
                <div className="flex flex-wrap gap-2">
                  {help.terms.map((term) => (
                    <GlossaryChip key={term} term={term} entries={entries} onNavigate={() => setOpen(false)} />
                  ))}
                </div>
                <p className="mt-2 text-[11px] text-slate-400">Pasa el cursor sobre un término para ver su definición; haz clic para abrirlo en el glosario.</p>
              </div>
            )}
          </div>
        </div>
      )}
    </>
  )
}
