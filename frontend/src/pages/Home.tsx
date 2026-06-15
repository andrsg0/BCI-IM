import { useNavigate } from 'react-router-dom'
import { ArrowRight } from 'lucide-react'
import { PageShell } from '../components/PageShell'
import type { HelpContent } from '../components/HelpButton'
import { DATASET_LIST } from '../lib/datasets'

const HELP: HelpContent = {
  pipeline: 'Visión general del proyecto',
  intro: 'Esta aplicación demuestra una Interfaz Cerebro-Computadora (BCI) para clasificar imaginación motora, concebida como aplicación práctica de la teoría de sistemas lineales e invariantes en el tiempo (LTI). La idea central es que cada etapa del procesamiento —el filtrado temporal, el filtrado espacial y la clasificación— es una operación lineal, y por tanto puede analizarse con las herramientas de la asignatura.',
  points: [
    { label: 'El diagrama del pipeline', desc: 'Resume el recorrido de la señal: adquisición del EEG, filtro FIR (banda µ/β), filtrado espacial CSP y clasificación. Cada bloque es una operación lineal y es interactivo: al pulsarlo se accede a la sección donde se explora en detalle.' },
    { label: 'Cómo navegar', desc: 'El menú superior ordena las secciones siguiendo el recorrido natural de la teoría a la práctica. El panel lateral funciona como control maestro: permite elegir el conjunto de datos, el sujeto y el canal, y reproducir la señal.' },
    { label: 'Las métricas', desc: 'Muestran la precisión media obtenida en cada conjunto de datos mediante validación cruzada, como referencia del rendimiento alcanzado por el sistema.' },
  ],
  terms: ['Sistema LTI', 'Convolución', 'FIR', 'CSP', 'Imaginación motora'],
}

const BLOCKS = [
  { label: 'Adquisición', sub: 'EEG  x[n]', to: '/lab', color: 'var(--accent-signal)' },
  { label: 'Filtro FIR', sub: 'convolución µ/β', to: '/lab', color: 'var(--accent-fir)' },
  { label: 'CSP', sub: 'filtro espacial', to: '/csp', color: 'var(--accent-csp)' },
  { label: 'LDA / EEGNet', sub: 'clasificación', to: '/live', color: 'var(--accent-metric)' },
]

export default function Home() {
  const nav = useNavigate()
  return (
    <PageShell
      title="Interfaz Cerebro-Computadora · Imaginación Motora"
      subtitle="Demostración práctica de sistemas LTI: convolución, filtros FIR, CSP y clasificación lineal."
      help={HELP}
    >
      {/* Diagrama del pipeline (bloques clicables) */}
      <section className="mb-8 rounded-xl border border-slate-200 bg-white p-6 shadow-card">
        <h2 className="mb-4 text-sm font-semibold uppercase tracking-wide text-slate-400">El pipeline</h2>
        <div className="flex flex-wrap items-center gap-2">
          {BLOCKS.map((b, i) => (
            <div key={b.label} className="flex items-center gap-2">
              <button
                onClick={() => nav(b.to)}
                className="group min-w-36 rounded-lg border border-slate-200 bg-slate-50 px-4 py-3 text-left transition-all hover:-translate-y-0.5 hover:shadow-md"
                style={{ borderTop: `3px solid ${b.color}` }}
              >
                <div className="font-semibold text-slate-700 group-hover:text-slate-900">{b.label}</div>
                <div className="text-xs text-slate-400">{b.sub}</div>
              </button>
              {i < BLOCKS.length - 1 && <ArrowRight className="text-slate-300" size={20} />}
            </div>
          ))}
        </div>
        <p className="mt-4 text-sm text-slate-500">
          Cada bloque es una operación <strong>lineal</strong>: el FIR mezcla muestras en el
          tiempo (<code className="rounded bg-slate-100 px-1">y[n]=Σ h[k]·x[n−k]</code>), el CSP
          mezcla canales en el espacio, y el LDA traza una frontera lineal. Haz clic para explorar.
        </p>
      </section>

      {/* Métricas clave por dataset */}
      <section>
        <h2 className="mb-3 text-sm font-semibold uppercase tracking-wide text-slate-400">Precisión por dataset (k-fold)</h2>
        <div className="grid gap-4 sm:grid-cols-3">
          {DATASET_LIST.map((d) => (
            <div key={d.id} className="rounded-xl border border-slate-200 bg-white p-4 shadow-card">
              <div className="text-sm text-slate-500">{d.label}</div>
              <div className="mt-1 text-3xl font-bold text-slate-800">{(d.accuracy * 100).toFixed(0)}%</div>
              <div className="mt-1 text-xs text-slate-400">{d.subjects} sujetos · {d.fs} Hz</div>
            </div>
          ))}
        </div>
      </section>
    </PageShell>
  )
}
