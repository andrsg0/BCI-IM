import { Fragment } from 'react'
import { NavLink } from 'react-router-dom'
import { PanelLeft } from 'lucide-react'
import { NAV_GROUPS, WORLD_STYLE } from '../../lib/nav'
import { useStore } from '../../store/useStore'

export function TopNav() {
  const toggleSidebar = useStore((s) => s.toggleSidebar)
  return (
    <header className="flex h-14 shrink-0 items-center gap-3 border-b border-slate-200 bg-white px-4">
      <button
        onClick={toggleSidebar}
        className="rounded-md p-2 text-slate-500 hover:bg-slate-100"
        title="Mostrar/ocultar panel"
      >
        <PanelLeft size={18} />
      </button>
      <span className="text-lg font-bold tracking-tight text-slate-800">
        BCI<span className="text-primary">·MI</span>
      </span>
      <nav className="ml-3 flex items-center gap-1 overflow-x-auto">
        {NAV_GROUPS.map((group, gi) => (
          <Fragment key={gi}>
            {gi > 0 && <span className="mx-1 h-6 w-px shrink-0 bg-slate-200" />}
            {group.label && (
              <span className={`flex shrink-0 items-center gap-1.5 pl-1 pr-0.5 text-[10px] font-semibold uppercase tracking-wide ${WORLD_STYLE[group.world].chip}`}>
                <span className={`h-1.5 w-1.5 rounded-full ${WORLD_STYLE[group.world].dot}`} />
                {group.label}
              </span>
            )}
            {group.items.map((item) => (
              <NavLink
                key={item.path}
                to={item.path}
                end={item.path === '/'}
                className={({ isActive }) =>
                  `flex items-center gap-1.5 whitespace-nowrap rounded-md px-3 py-1.5 text-sm transition-colors ${
                    isActive
                      ? 'bg-primary/10 font-medium text-primary'
                      : 'text-slate-600 hover:bg-slate-100'
                  }`
                }
              >
                <item.icon size={16} />
                {item.label}
              </NavLink>
            ))}
          </Fragment>
        ))}
      </nav>
    </header>
  )
}
