import { Database, Radio } from 'lucide-react'
import type { World } from '../lib/nav'

/**
 * Distintivo que recuerda en cuál de los dos mundos está la página:
 *  - offline: cálculos previos al streaming (entrenar/validar el modelo).
 *  - online : procesamiento en tiempo real, con el modelo ya entrenado.
 * Hace explícita y constante la separación entrenar / transmitir.
 */
const CONFIG: Record<Exclude<World, 'general'>, {
  icon: typeof Database; text: string; cls: string; dot: string
}> = {
  offline: {
    icon: Database,
    text: 'Cálculo previo al streaming · no es en tiempo real',
    cls: 'border-amber-200 bg-amber-50 text-amber-700',
    dot: 'bg-amber-500',
  },
  online: {
    icon: Radio,
    text: 'En tiempo real · usa el modelo ya entrenado',
    cls: 'border-emerald-200 bg-emerald-50 text-emerald-700',
    dot: 'bg-emerald-500',
  },
}

export function WorldBadge({ world }: { world: Exclude<World, 'general'> }) {
  const c = CONFIG[world]
  const Icon = c.icon
  return (
    <span className={`inline-flex items-center gap-1.5 rounded-full border px-2.5 py-1 text-xs font-medium ${c.cls}`}>
      <span className={`h-1.5 w-1.5 rounded-full ${c.dot}`} />
      <Icon size={13} />
      {c.text}
    </span>
  )
}
