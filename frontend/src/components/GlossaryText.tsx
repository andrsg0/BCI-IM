import { Fragment, useMemo } from 'react'
import { Link } from 'react-router-dom'
import {
  useGlossary, buildGlossaryMatcher, entryForAlias, previewText,
  type GlossaryEntry,
} from '../lib/glossary'

/** Término detectado en el cuerpo del texto: subrayado punteado + preview al hover. */
function GlossaryMark({ text, entry, onNavigate }: {
  text: string
  entry: GlossaryEntry
  onNavigate?: () => void
}) {
  return (
    <span className="group relative inline">
      <Link
        to={`/glossary?q=${encodeURIComponent(entry.term)}`}
        onClick={onNavigate}
        className="cursor-help font-medium text-primary underline decoration-dotted underline-offset-2 hover:decoration-solid"
      >
        {text}
      </Link>
      <span className="pointer-events-none absolute bottom-full left-1/2 z-50 mb-2 hidden w-64 -translate-x-1/2 group-hover:block">
        <span className="block rounded-lg bg-slate-900 px-3 py-2 text-left text-xs font-normal normal-case leading-snug text-slate-100 shadow-xl">
          <span className="mb-0.5 block font-semibold text-white">{entry.term}</span>
          {previewText(entry.body)}
        </span>
      </span>
    </span>
  )
}

/**
 * Renderiza `children` (texto plano) detectando términos del glosario y
 * enlazándolos in-situ. Solo se enlaza la PRIMERA aparición de cada término
 * para no saturar el párrafo. Si el glosario aún no cargó, devuelve el texto tal cual.
 */
export function GlossaryText({ children, onNavigate }: {
  children: string
  onNavigate?: () => void
}) {
  const entries = useGlossary()
  const matcher = useMemo(() => buildGlossaryMatcher(entries), [entries])

  const nodes = useMemo(() => {
    if (!matcher) return [children]
    const out: (string | { text: string; entry: GlossaryEntry })[] = []
    const seen = new Set<string>()       // term.term ya enlazado en este texto
    let last = 0
    matcher.regex.lastIndex = 0
    for (let mm = matcher.regex.exec(children); mm; mm = matcher.regex.exec(children)) {
      const entry = entryForAlias(mm[1], matcher)
      if (entry && !seen.has(entry.term)) {
        seen.add(entry.term)
        if (mm.index > last) out.push(children.slice(last, mm.index))
        out.push({ text: mm[1], entry })
        last = mm.index + mm[1].length
      }
    }
    if (last < children.length) out.push(children.slice(last))
    return out
  }, [children, matcher])

  return (
    <>
      {nodes.map((n, i) =>
        typeof n === 'string'
          ? <Fragment key={i}>{n}</Fragment>
          : <GlossaryMark key={i} text={n.text} entry={n.entry} onNavigate={onNavigate} />,
      )}
    </>
  )
}
