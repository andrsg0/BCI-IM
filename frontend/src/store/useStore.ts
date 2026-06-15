import { create } from 'zustand'
import { DATASETS, type DatasetId } from '../lib/datasets'

export interface LogEntry {
  id: number
  t: string
  level: 'info' | 'warn' | 'error'
  msg: string
}

export type SystemStatus = 'idle' | 'streaming' | 'paused'

interface AppState {
  // --- estado global (controlador maestro del sidebar) ---
  dataset: DatasetId
  subject: number
  channel: string
  playing: boolean
  loop: boolean
  sidebarOpen: boolean
  status: SystemStatus
  latencyMs: number
  logs: LogEntry[]
  /** se incrementa al pulsar "Limpiar vistas"; los widgets lo observan para resetear */
  clearToken: number
  /** color de acento principal de la UI (configurable en Apariencia) */
  primaryColor: string

  // --- acciones ---
  setLatency: (ms: number) => void
  setPrimaryColor: (c: string) => void
  setDataset: (d: DatasetId) => void
  setSubject: (s: number) => void
  setChannel: (c: string) => void
  togglePlay: () => void
  setPlaying: (p: boolean) => void
  toggleLoop: () => void
  toggleSidebar: () => void
  addLog: (msg: string, level?: LogEntry['level']) => void
  clearViews: () => void
}

let logId = 0
const now = () => new Date().toLocaleTimeString('es', { hour12: false })

const DEFAULT_PRIMARY = '#2563eb'
const savedPrimary =
  (typeof localStorage !== 'undefined' && localStorage.getItem('primaryColor')) || DEFAULT_PRIMARY
// aplica el color guardado al cargar (sobre la variable de tema)
if (typeof document !== 'undefined') {
  document.documentElement.style.setProperty('--color-primary', savedPrimary)
}

export const useStore = create<AppState>((set, get) => ({
  dataset: 'BNCI2014_001',
  subject: 1,
  channel: 'C3',
  playing: false,
  loop: false,
  sidebarOpen: true,
  status: 'idle',
  latencyMs: 0,
  logs: [{ id: logId++, t: now(), level: 'info', msg: 'Sistema iniciado.' }],
  clearToken: 0,
  primaryColor: savedPrimary,

  setLatency: (ms) => set({ latencyMs: ms }),

  setPrimaryColor: (c) => {
    set({ primaryColor: c })
    document.documentElement.style.setProperty('--color-primary', c)
    try { localStorage.setItem('primaryColor', c) } catch { /* sin persistencia */ }
  },

  setDataset: (d) => {
    const info = DATASETS[d]
    set({ dataset: d, subject: 1, channel: info.channels.includes('C3') ? 'C3' : info.channels[0] })
    get().addLog(`Dataset → ${info.label} (fs=${info.fs} Hz, ${info.subjects} sujetos)`)
  },
  setSubject: (s) => { set({ subject: s }); get().addLog(`Sujeto → ${s}`) },
  setChannel: (c) => set({ channel: c }),
  togglePlay: () => {
    const playing = !get().playing
    set({ playing, status: playing ? 'streaming' : 'paused' })
    get().addLog(playing ? 'Reproducción iniciada.' : 'Reproducción en pausa.')
  },
  setPlaying: (p) => set({ playing: p, status: p ? 'streaming' : 'paused' }),
  toggleLoop: () => { const loop = !get().loop; set({ loop }); get().addLog(`Loop ${loop ? 'activado' : 'desactivado'}.`) },
  toggleSidebar: () => set({ sidebarOpen: !get().sidebarOpen }),
  addLog: (msg, level = 'info') =>
    set((st) => ({ logs: [...st.logs.slice(-49), { id: logId++, t: now(), level, msg }] })),
  clearViews: () => { set((st) => ({ clearToken: st.clearToken + 1 })); get().addLog('Vistas limpiadas.') },
}))
