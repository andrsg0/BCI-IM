import { useEffect, useMemo, useState } from 'react'
import { useSearchParams } from 'react-router-dom'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import { Search } from 'lucide-react'
import { PageShell } from '../components/PageShell'
import type { HelpContent } from '../components/HelpButton'
import { getJSON } from '../api/client'

interface Entry { category: string; term: string; body: string }

const HELP: HelpContent = {
  pipeline: 'Referencia',
  intro: 'Diccionario de los términos del proyecto, tanto de matemáticas y procesamiento de señales como de neurofisiología. Cada entrada explica qué es el término y el concepto que hay detrás.',
  points: [
    { label: 'Búsqueda', desc: 'Escribe en el buscador para filtrar las entradas por su nombre o su contenido. Ignora mayúsculas y acentos.' },
    { label: 'Categorías', desc: 'Filtra por bloque temático: neurofisiología y EEG, teoría LTI/DSP, machine learning o herramientas.' },
    { label: 'Enlaces desde la ayuda', desc: 'Los términos clave de los paneles de ayuda (botón ?) de cada sección enlazan directamente aquí, abriendo el glosario ya filtrado por ese término.' },
  ],
}

const norm = (s: string) => s.toLowerCase().normalize('NFD').replace(/[̀-ͯ]/g, '')
// separa los sub-apartados "**Qué es:**", "**El concepto detrás:**"… en párrafos
const prep = (body: string) => body.replace(/\n(\*\*[^*\n]+:\*\*)/g, '\n\n$1')

export default function Glossary() {
  const [params] = useSearchParams()
  const [entries, setEntries] = useState<Entry[]>([])
  const [query, setQuery] = useState(params.get('q') ?? '')
  const [cat, setCat] = useState('all')

  useEffect(() => { getJSON<Entry[]>('/glossary').then(setEntries).catch(() => setEntries([])) }, [])

  const categories = useMemo(() => Array.from(new Set(entries.map((e) => e.category))), [entries])
  const filtered = useMemo(() => {
    // búsqueda por palabras (todas deben aparecer): tolera separadores y orden
    const tokens = norm(query).split(/[^a-z0-9µβ]+/).filter(Boolean)
    return entries.filter((e) => {
      if (cat !== 'all' && e.category !== cat) return false
      if (tokens.length === 0) return true
      const hay = norm(e.term + ' ' + e.body)
      return tokens.every((t) => hay.includes(t))
    })
  }, [entries, query, cat])

  return (
    <PageShell title="Glosario" subtitle="Términos de señales, machine learning y neurofisiología." help={HELP}>
      {/* buscador + categorías */}
      <div className="mb-5 space-y-3">
        <div className="relative max-w-md">
          <Search size={16} className="absolute left-3 top-1/2 -translate-y-1/2 text-slate-400" />
          <input
            value={query} onChange={(e) => setQuery(e.target.value)} placeholder="Buscar término…"
            className="w-full rounded-lg border border-slate-300 bg-white py-2 pl-9 pr-3 text-sm shadow-card focus:border-primary focus:outline-none"
          />
        </div>
        <div className="flex flex-wrap gap-2">
          {['all', ...categories].map((c) => (
            <button key={c} onClick={() => setCat(c)}
              className={`rounded-full px-3 py-1 text-xs ${cat === c ? 'bg-primary text-white' : 'bg-white text-slate-600 hover:bg-slate-100'}`}>
              {c === 'all' ? 'Todas' : c}
            </button>
          ))}
        </div>
        <p className="text-xs text-slate-400">{filtered.length} de {entries.length} términos</p>
      </div>

      {/* tarjetas */}
      <div className="grid items-start gap-4 lg:grid-cols-2">
        {filtered.map((e) => (
          <div key={e.term} className="shadow-card rounded-xl border border-slate-200 bg-white p-4">
            <div className="mb-1 text-xs font-medium uppercase tracking-wide text-primary">{e.category}</div>
            <h3 className="mb-2 font-semibold text-slate-800">{e.term}</h3>
            <div className="prose prose-sm prose-slate max-w-none prose-p:my-1.5 prose-li:my-0.5">
              <ReactMarkdown remarkPlugins={[remarkGfm]}>{prep(e.body)}</ReactMarkdown>
            </div>
          </div>
        ))}
        {filtered.length === 0 && entries.length > 0 && (
          <div className="col-span-full py-10 text-center text-slate-400">Sin resultados para «{query}».</div>
        )}
      </div>
    </PageShell>
  )
}
