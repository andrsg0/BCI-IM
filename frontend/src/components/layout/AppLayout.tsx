import { Outlet } from 'react-router-dom'
import { TopNav } from './TopNav'
import { Sidebar } from './Sidebar'
import { useStore } from '../../store/useStore'

export function AppLayout() {
  const sidebarOpen = useStore((s) => s.sidebarOpen)
  return (
    <div className="flex h-full flex-col">
      <TopNav />
      <div className="flex min-h-0 flex-1">
        {sidebarOpen && <Sidebar />}
        <main className="min-h-0 flex-1 overflow-auto p-6">
          <Outlet />
        </main>
      </div>
    </div>
  )
}
