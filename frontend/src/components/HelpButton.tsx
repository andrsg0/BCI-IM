import { useEffect, useState } from 'react'
import { HelpCircle, X } from 'lucide-react'
import { Link } from 'react-router-dom'

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

/** Botón "?" que abre un panel explicando la sección actual. */
export function HelpButton({ title, help }: { title: string; help: HelpContent }) {
  const [open, setOpen] = useState(false)

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
                    <Link key={term} to={`/glossary?q=${encodeURIComponent(term)}`} onClick={() => setOpen(false)}
                      className="rounded-full bg-primary/10 px-2.5 py-1 text-xs text-primary hover:bg-primary/20">
                      {term}
                    </Link>
                  ))}
                </div>
              </div>
            )}
          </div>
        </div>
      )}
    </>
  )
}
