import type { ReactNode } from 'react'

/** Familias de acento (la franja de color distintiva de cada tipo de widget). */
export type Accent = 'signal' | 'fir' | 'csp' | 'brain' | 'metric' | 'neutral'

const ACCENT_VAR: Record<Accent, string> = {
  signal: 'var(--accent-signal)',
  fir: 'var(--accent-fir)',
  csp: 'var(--accent-csp)',
  brain: 'var(--accent-brain)',
  metric: 'var(--accent-metric)',
  neutral: 'var(--accent-neutral)',
}

/**
 * Tarjeta-widget: rectángulo de bordes redondeados con una franja lateral de
 * color que lo hace distintivo (según su familia). Base de todo el sistema de
 * widgets tipo rompecabezas.
 */
export function Widget({
  title,
  accent = 'signal',
  actions,
  children,
}: {
  title: string
  accent?: Accent
  actions?: ReactNode
  children: ReactNode
}) {
  const color = ACCENT_VAR[accent]
  return (
    <div
      className="shadow-card flex h-full flex-col overflow-hidden rounded-xl border border-slate-200/80 bg-white"
      style={{ borderTop: `3px solid ${color}` }}
    >
      <div
        className="flex items-center gap-2 border-b border-slate-100 px-3 py-2"
        style={{ background: `color-mix(in srgb, ${color} 6%, white)` }}
      >
        <h3 className="text-sm font-semibold" style={{ color: `color-mix(in srgb, ${color} 55%, #334155)` }}>
          {title}
        </h3>
        <div className="ml-auto flex items-center gap-1">{actions}</div>
      </div>
      <div className="min-h-0 flex-1 overflow-auto p-3">{children}</div>
    </div>
  )
}
