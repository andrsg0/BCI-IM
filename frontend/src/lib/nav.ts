import type { ComponentType } from 'react'
import { Home, LayoutDashboard, Activity, Network, Brain, Radio, BarChart3, BookOpen } from 'lucide-react'

export interface NavItem {
  path: string
  label: string
  icon: ComponentType<{ size?: number; className?: string }>
}

/**
 * Los dos mundos del sistema (ver docs/presentacion.md):
 *  - 'offline': cálculos hechos ANTES del streaming (entrenar/validar el modelo).
 *  - 'online' : lo que ocurre AHORA, en tiempo real (la señal llega y se clasifica).
 *  - 'general': páginas transversales (inicio, glosario).
 */
export type World = 'general' | 'offline' | 'online'

export interface NavGroup {
  label: string | null   // encabezado del grupo (null = sin encabezado)
  world: World
  items: NavItem[]
}

// La navegación queda separada en dos mundos claramente etiquetados.
export const NAV_GROUPS: NavGroup[] = [
  {
    label: null,
    world: 'general',
    items: [
      { path: '/', label: 'Inicio', icon: Home },
      { path: '/dashboard', label: 'Dashboard', icon: LayoutDashboard },
    ],
  },
  {
    label: 'Entrenamiento',
    world: 'offline',
    items: [
      { path: '/csp', label: 'El Modelo', icon: Network },
      { path: '/results', label: 'Resultados', icon: BarChart3 },
    ],
  },
  {
    label: 'En vivo',
    world: 'online',
    items: [
      { path: '/lab', label: 'Laboratorio', icon: Activity },
      { path: '/live', label: 'Clasificación', icon: Radio },
      { path: '/brain', label: 'Cerebro 3D', icon: Brain },
    ],
  },
  {
    label: null,
    world: 'general',
    items: [{ path: '/glossary', label: 'Glosario', icon: BookOpen }],
  },
]

// Estilos por mundo (reutilizados por la barra superior y los distintivos de página).
export const WORLD_STYLE: Record<World, { dot: string; chip: string }> = {
  general: { dot: 'bg-slate-300', chip: 'text-slate-400' },
  offline: { dot: 'bg-amber-500', chip: 'text-amber-600' },
  online: { dot: 'bg-emerald-500', chip: 'text-emerald-600' },
}

// Lista plana (compatibilidad / usos puntuales).
export const NAV: NavItem[] = NAV_GROUPS.flatMap((g) => g.items)
