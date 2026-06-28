// Utilidades sobre nombres de electrodos del sistema 10-10/10-20.

/** Clasifica un electrodo por nombre: ¿es de la corteza motora? ¿qué hemisferio?
 *  (impar = izquierda, par = derecha, z = línea media). Ej.: C3→motor/L, FC4→motor/R, POz→no. */
export function elecInfo(ch: string): { motor: boolean; side: 'L' | 'R' | 'M' } {
  const m = ch.match(/^([A-Za-z]+?)(z|\d+)$/i)
  if (!m) return { motor: false, side: 'M' }
  const prefix = m[1].toUpperCase()
  const suf = m[2].toLowerCase()
  const motor = ['C', 'FC', 'CP', 'FCC', 'CCP'].includes(prefix)
  const side: 'L' | 'R' | 'M' = suf === 'z' ? 'M' : parseInt(suf, 10) % 2 === 1 ? 'L' : 'R'
  return { motor, side }
}

/** Promedio de los valores (p. ej. desviación µ/β) sobre cada corteza motora.
 *  Útil para visualizar la lateralización contralateral del ERD. */
export function motorLaterality(channels: string[], values: number[]): { left: number; right: number } {
  let lS = 0, lN = 0, rS = 0, rN = 0
  channels.forEach((ch, i) => {
    const e = elecInfo(ch)
    if (!e.motor) return
    const v = values[i] ?? 0
    if (e.side === 'L') { lS += v; lN++ } else if (e.side === 'R') { rS += v; rN++ }
  })
  return { left: lN ? lS / lN : 0, right: rN ? rS / rN : 0 }
}
