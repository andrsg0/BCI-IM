// Topomapa: cabeza vista desde arriba con los electrodos del sistema 10-20
// coloreados según un valor por canal (p. ej. el peso de un patrón CSP).
// Escala divergente azul(−) · blanco(0) · rojo(+).

import { divergingColor } from '../lib/color'

export type Pos2D = Record<string, [number, number] | null>

export function Topomap({ channels, pos2d, values, size = 170, showLabels = false }: {
  channels: string[]; pos2d: Pos2D; values: number[]; size?: number; showLabels?: boolean
}) {
  const R = size * 0.42
  const c = size / 2
  const maxAbs = Math.max(...values.map((v) => Math.abs(v)), 1e-9)
  const r = size / 30

  return (
    <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`} className="mx-auto">
      {/* nariz (anterior, arriba) */}
      <polygon points={`${c - 7},${c - R + 3} ${c},${c - R - 9} ${c + 7},${c - R + 3}`} fill="#f1f5f9" stroke="#cbd5e1" />
      {/* orejas */}
      <ellipse cx={c - R} cy={c} rx={size * 0.03} ry={size * 0.07} fill="#f1f5f9" stroke="#cbd5e1" />
      <ellipse cx={c + R} cy={c} rx={size * 0.03} ry={size * 0.07} fill="#f1f5f9" stroke="#cbd5e1" />
      {/* cabeza */}
      <circle cx={c} cy={c} r={R} fill="#f8fafc" stroke="#cbd5e1" strokeWidth={1.5} />
      {/* electrodos */}
      {channels.map((ch, i) => {
        const p = pos2d[ch]
        if (!p) return null
        const x = c + p[0] * R
        const y = c - p[1] * R   // +y = anterior → arriba en pantalla
        return (
          <g key={ch}>
            <circle cx={x} cy={y} r={r} fill={divergingColor(values[i] / maxAbs)} stroke="#475569" strokeWidth={0.6}>
              <title>{ch}: {values[i].toFixed(3)}</title>
            </circle>
            {showLabels && <text x={x} y={y + r + size / 22} fontSize={size / 20} textAnchor="middle" fill="#64748b">{ch}</text>}
          </g>
        )
      })}
    </svg>
  )
}
