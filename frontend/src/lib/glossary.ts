// Cache en cliente del glosario (servido por /api/glossary) + utilidades de
// búsqueda tolerante para mostrar previews al pasar el cursor sobre un término.
import { useEffect, useState } from 'react'
import { getJSON } from '../api/client'

export interface GlossaryEntry { category: string; term: string; body: string }

let cache: GlossaryEntry[] | null = null
let inflight: Promise<GlossaryEntry[]> | null = null

function load(): Promise<GlossaryEntry[]> {
  if (cache) return Promise.resolve(cache)
  if (!inflight) {
    inflight = getJSON<GlossaryEntry[]>('/glossary')
      .then((e) => (cache = e))
      .catch(() => (cache = []))
  }
  return inflight
}

/** Carga el glosario una sola vez y lo deja en caché para toda la app. */
export function useGlossary(): GlossaryEntry[] {
  const [entries, setEntries] = useState<GlossaryEntry[]>(cache ?? [])
  useEffect(() => {
    let on = true
    load().then((e) => { if (on) setEntries(e) })
    return () => { on = false }
  }, [])
  return entries
}

// normaliza para comparar: minúsculas, sin acentos ni símbolos (conserva µ y β)
const norm = (s: string) =>
  s.toLowerCase().normalize('NFD').replace(/[̀-ͯ]/g, '').replace(/[^a-z0-9µβ]/g, '')

/** Busca la entrada del glosario que mejor corresponde a un término corto. */
export function findEntry(term: string, entries: GlossaryEntry[]): GlossaryEntry | undefined {
  const t = norm(term)
  if (!t) return undefined
  let starts: GlossaryEntry | undefined
  let incl: GlossaryEntry | undefined
  for (const e of entries) {
    const et = norm(e.term)
    if (et === t) return e                       // coincidencia exacta
    if (!starts && et.startsWith(t)) starts = e  // el título empieza por el término
    if (!incl && et.includes(t)) incl = e        // el título contiene el término
  }
  return starts ?? incl
}

/** Texto corto de previsualización: quita markdown y recorta. */
export function previewText(body: string, max = 220): string {
  const txt = body.replace(/\*\*/g, '').replace(/`/g, '').replace(/\s+/g, ' ').trim()
  return txt.length > max ? txt.slice(0, max).trimEnd() + '…' : txt
}
