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

// ---------------------------------------------------------------------------
// Detección de términos del glosario dentro de texto libre (para auto-enlazar
// el cuerpo explicativo de cada sección, no solo los chips de "Términos clave").
// ---------------------------------------------------------------------------

// Palabras que, si abren un alias candidato, indican que NO es un término sino
// una descripción ("la jerarquía de una grabación") -> se descarta.
const ALIAS_STOP = new Set([
  'la', 'el', 'los', 'las', 'un', 'una', 'de', 'del', 'que', 'por', 'y', 'o',
  'con', 'para', 'su', 'sus', 'p', 'ej', 'al', 'en',
])

/** ¿`p` parece un término enlazable (no una frase descriptiva)? */
function isTermLike(p: string): boolean {
  const t = p.trim()
  if (t.length < 3 || t.length > 40) return false
  if (/[`[\]]/.test(t)) return false   // tokens tipo código (`h[n]`) no aparecen en prosa
  const words = t.split(/\s+/)
  if (words.length > 4) return false
  const first = words[0].toLowerCase().normalize('NFD').replace(/[̀-ͯ]/g, '').replace(/[^a-zµβ0-9]/g, '')
  return !ALIAS_STOP.has(first)
}

/**
 * Deriva los alias enlazables del título de una entrada del glosario.
 * Los títulos tienen forma "Corto (Largo / alias) — extra": se toma la parte
 * principal (antes del "("), partida por "/", más los tokens entre paréntesis.
 */
function aliasesFor(term: string): string[] {
  const out: string[] = []
  const main = term.split('(')[0].replace(/\s[—–-]\s.*$/, '')
  for (const part of main.split('/')) {
    if (isTermLike(part)) out.push(part.trim())
  }
  for (const m of term.matchAll(/\(([^)]*)\)/g)) {
    for (const part of m[1].split(/[/,]/)) {
      if (isTermLike(part)) out.push(part.trim())
    }
  }
  return out
}

const escapeRe = (s: string) => s.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')

export interface GlossaryMatcher {
  regex: RegExp
  /** alias normalizado -> entrada del glosario */
  lookup: Map<string, GlossaryEntry>
}

// Cache del matcher: reconstruir solo cuando cambia el array de entradas.
let matcherEntries: GlossaryEntry[] | null = null
let matcherCache: GlossaryMatcher | null = null

/** Construye (y cachea) un matcher que detecta términos del glosario en texto. */
export function buildGlossaryMatcher(entries: GlossaryEntry[]): GlossaryMatcher | null {
  if (matcherCache && matcherEntries === entries) return matcherCache
  matcherEntries = entries
  if (entries.length === 0) { matcherCache = null; return null }

  const lookup = new Map<string, GlossaryEntry>()
  const aliases: string[] = []
  for (const e of entries) {
    for (const a of aliasesFor(e.term)) {
      const key = norm(a)
      if (!key || lookup.has(key)) continue   // el primero gana (alias único)
      lookup.set(key, e)
      aliases.push(a)
    }
  }
  // Más largos primero: "Common Spatial Patterns" antes que "CSP".
  aliases.sort((a, b) => b.length - a.length)
  // Límite de palabra unicode (no atravesar acentos/dígitos/µ/β) vía lookarounds.
  const regex = new RegExp(
    `(?<![\\p{L}\\p{N}µβ])(${aliases.map(escapeRe).join('|')})(?![\\p{L}\\p{N}µβ])`,
    'giu',
  )
  matcherCache = { regex, lookup }
  return matcherCache
}

/** Busca la entrada que corresponde a un alias ya detectado en el texto. */
export function entryForAlias(alias: string, m: GlossaryMatcher): GlossaryEntry | undefined {
  return m.lookup.get(norm(alias))
}
