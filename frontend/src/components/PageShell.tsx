import type { ReactNode } from 'react'
import { HelpButton, type HelpContent } from './HelpButton'
import type { World } from '../lib/nav'

/** Contenedor estándar de página: título, subtítulo, botón de ayuda y contenido. */
export function PageShell({
  title,
  subtitle,
  help,
  world,
  children,
}: {
  title: string
  subtitle?: string
  help?: HelpContent
  /** Mundo al que pertenece la página. Lo indica la agrupación del nav lateral; aquí
   *  se acepta por compatibilidad pero ya NO se muestra un distintivo junto al título. */
  world?: Exclude<World, 'general'>
  children: ReactNode
}) {
  void world
  return (
    <div className="mx-auto max-w-7xl">
      <header className="mb-5 flex items-start justify-between gap-4">
        <div>
          <h1 className="text-2xl font-bold text-slate-800">{title}</h1>
          {subtitle && <p className="mt-1 text-slate-500">{subtitle}</p>}
        </div>
        {help && <HelpButton title={title} help={help} />}
      </header>
      {children}
    </div>
  )
}

/** Nota temporal mientras un módulo está pendiente de implementar. */
export function ComingSoon({ hito }: { hito: string }) {
  return (
    <div className="rounded-xl border border-dashed border-slate-300 bg-white/60 p-10 text-center text-slate-400">
      Módulo en construcción — {hito}
    </div>
  )
}
