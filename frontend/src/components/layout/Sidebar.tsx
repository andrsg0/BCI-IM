import { useState } from 'react'
import { Play, Pause, Repeat, RotateCcw, Trash2, Info, Activity, Settings, Check } from 'lucide-react'
import { useStore } from '../../store/useStore'
import { DATASETS, DATASET_LIST } from '../../lib/datasets'

const LEVEL_COLOR = { info: 'text-slate-500', warn: 'text-amber-600', error: 'text-red-600' }

// paleta de colores de acento predefinidos
const ACCENT_PRESETS = ['#2563eb', '#0891b2', '#7c3aed', '#059669', '#e11d48', '#d97706', '#4f46e5', '#0f172a']

export function Sidebar() {
  const {
    dataset, subject, channel, playing, loop, ended, elapsedSec, totalSec, status, latencyMs, logs, primaryColor,
    setDataset, setSubject, setChannel, togglePlay, toggleLoop, clearViews, setPrimaryColor,
  } = useStore()
  const info = DATASETS[dataset]
  const [showSettings, setShowSettings] = useState(false)

  return (
    <aside className="flex w-72 shrink-0 flex-col border-r border-slate-200 bg-white">
      <div className="flex-1 space-y-5 overflow-auto p-4">
        {/* --- Fuente de datos --- */}
        <section className="space-y-2">
          <h2 className="text-xs font-semibold uppercase tracking-wide text-slate-400">Fuente de datos</h2>
          <label className="block text-xs text-slate-500">Dataset</label>
          <select
            value={dataset}
            onChange={(e) => setDataset(e.target.value as typeof dataset)}
            className="w-full rounded-md border border-slate-300 bg-white px-2 py-1.5 text-sm"
          >
            {DATASET_LIST.map((d) => (
              <option key={d.id} value={d.id}>{d.label} · {d.fs} Hz</option>
            ))}
          </select>

          <div className="grid grid-cols-2 gap-2">
            <div>
              <label className="block text-xs text-slate-500">Sujeto</label>
              <select
                value={subject}
                onChange={(e) => setSubject(Number(e.target.value))}
                className="w-full rounded-md border border-slate-300 bg-white px-2 py-1.5 text-sm"
              >
                {Array.from({ length: info.subjects }, (_, i) => i + 1).map((s) => (
                  <option key={s} value={s}>{s}</option>
                ))}
              </select>
            </div>
            <div>
              <label className="block text-xs text-slate-500">Canal</label>
              <select
                value={channel}
                onChange={(e) => setChannel(e.target.value)}
                className="w-full rounded-md border border-slate-300 bg-white px-2 py-1.5 text-sm"
              >
                {info.channels.map((c) => <option key={c} value={c}>{c}</option>)}
              </select>
            </div>
          </div>
        </section>

        {/* --- Reproducción --- */}
        <section className="space-y-2">
          <h2 className="text-xs font-semibold uppercase tracking-wide text-slate-400">Reproducción</h2>
          <div className="flex gap-2">
            <button
              onClick={togglePlay}
              className="bg-primary hover:bg-primary-hover flex flex-1 items-center justify-center gap-1.5 rounded-md px-3 py-2 text-sm font-medium text-white"
            >
              {playing ? <Pause size={16} /> : ended ? <RotateCcw size={16} /> : <Play size={16} />}
              {playing ? 'Pausa' : ended ? 'Volver a iniciar' : 'Play'}
            </button>
            <button
              onClick={toggleLoop}
              className={`rounded-md border px-3 py-2 ${loop ? 'border-primary/40 bg-primary/10 text-primary' : 'border-slate-300 text-slate-500 hover:bg-slate-100'}`}
              title="Repetir en bucle"
            >
              <Repeat size={16} />
            </button>
            <button
              onClick={clearViews}
              className="rounded-md border border-slate-300 px-3 py-2 text-slate-500 hover:bg-slate-100"
              title="Limpiar vistas"
            >
              <Trash2 size={16} />
            </button>
          </div>
          {/* Duración de la señal (solo informativa; el programa no la usa para clasificar) */}
          {totalSec > 0 && (
            <div className="space-y-1 pt-0.5">
              <div className="flex justify-between font-mono text-[11px] text-slate-500">
                <span>{elapsedSec.toFixed(1)} s</span>
                <span className="text-slate-400">{totalSec.toFixed(0)} s</span>
              </div>
              <div className="h-1.5 overflow-hidden rounded-full bg-slate-100">
                <div
                  className="bg-primary h-full rounded-full transition-[width] duration-100 ease-linear"
                  style={{ width: `${Math.min(100, (elapsedSec / totalSec) * 100)}%` }}
                />
              </div>
            </div>
          )}
          <p className="text-[11px] text-slate-400">La reproducción afecta solo a la página actual.</p>
        </section>

        {/* --- Estado del sistema --- */}
        <section className="space-y-2">
          <h2 className="text-xs font-semibold uppercase tracking-wide text-slate-400">Estado del sistema</h2>
          <div className="rounded-md bg-slate-50 p-2 text-xs text-slate-600">
            <div className="flex items-center justify-between">
              <span className="flex items-center gap-1.5">
                <span className={`h-2 w-2 rounded-full ${status === 'streaming' ? 'bg-green-500' : status === 'paused' ? 'bg-amber-500' : 'bg-slate-300'}`} />
                {status === 'streaming' ? 'Transmitiendo' : status === 'paused' ? 'En pausa' : 'Inactivo'}
              </span>
              <span className="flex items-center gap-1 text-slate-400"><Activity size={12} />{latencyMs} ms</span>
            </div>
          </div>
        </section>

        {/* --- Logs --- */}
        <section className="space-y-2">
          <h2 className="text-xs font-semibold uppercase tracking-wide text-slate-400">Logs</h2>
          <div className="h-40 overflow-auto rounded-md border border-slate-200 bg-slate-50 p-2 font-mono text-[11px] leading-relaxed">
            {logs.map((l) => (
              <div key={l.id} className={LEVEL_COLOR[l.level]}>
                <span className="text-slate-400">{l.t}</span> {l.msg}
              </div>
            ))}
          </div>
        </section>
      </div>

      {/* Panel de Apariencia (color de acento) */}
      {showSettings && (
        <div className="space-y-2 border-t border-slate-200 p-4">
          <div className="text-xs font-semibold uppercase tracking-wide text-slate-400">Color de acento</div>
          <div className="flex flex-wrap items-center gap-2">
            {ACCENT_PRESETS.map((c) => (
              <button
                key={c}
                onClick={() => setPrimaryColor(c)}
                className="flex h-7 w-7 items-center justify-center rounded-full ring-2 ring-offset-1 transition"
                style={{ background: c, ['--tw-ring-color' as string]: primaryColor === c ? c : 'transparent' }}
                title={c}
              >
                {primaryColor === c && <Check size={14} className="text-white" />}
              </button>
            ))}
            <label
              className="flex h-7 w-7 cursor-pointer items-center justify-center rounded-full border border-dashed border-slate-300 text-slate-400"
              title="Color personalizado"
            >
              +
              <input
                type="color"
                value={primaryColor}
                onChange={(e) => setPrimaryColor(e.target.value)}
                className="absolute h-0 w-0 opacity-0"
              />
            </label>
          </div>
        </div>
      )}

      <div className="flex items-center justify-between border-t border-slate-200 px-2 py-2">
        <a href="#about" className="flex items-center gap-2 rounded-md px-2 py-1.5 text-sm text-slate-500 hover:bg-slate-50">
          <Info size={15} /> Acerca de
        </a>
        <button
          onClick={() => setShowSettings((s) => !s)}
          className={`rounded-md p-2 ${showSettings ? 'bg-primary/10 text-primary' : 'text-slate-400 hover:bg-slate-100'}`}
          title="Apariencia"
        >
          <Settings size={16} />
        </button>
      </div>
    </aside>
  )
}
