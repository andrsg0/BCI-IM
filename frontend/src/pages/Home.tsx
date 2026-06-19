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
      {/* Sobre el proyecto */}
      <section className="mb-8 rounded-xl border border-slate-200 bg-white p-6 shadow-card">
        <h2 className="mb-3 text-sm font-semibold uppercase tracking-wide text-slate-400">Sobre el proyecto</h2>
        <p className="text-sm leading-relaxed text-slate-600">
          Interfaz Cerebro-Computadora que clasifica <strong>imaginación motora</strong> (imaginar
          mover la mano izquierda o derecha) a partir de señales EEG. El objetivo académico es hacer
          la teoría de <strong>sistemas LTI</strong> explícita en el código —convolución, filtros FIR,
          filtrado espacial CSP y respuesta en frecuencia— en lugar de ocultarla tras llamadas a
          librerías. EEGNet (deep learning) aparece como espejo: sus capas imitan ese mismo pipeline,
          pero aprendido de los datos.
        </p>
        <div className="mt-4 grid gap-3 sm:grid-cols-3">
          {[
            { k: 'Foco', v: 'Teoría LTI explícita', d: 'Convolución, FIR, CSP y frecuencia, a mano' },
            { k: 'Datos', v: 'EEG público (MOABB)', d: 'Aún sin hardware; futuro casco vía LSL' },
            { k: 'Tarea', v: '2 clases (izq./der.)', d: 'Techo realista del estado del arte: ~70–85 %' },
          ].map((c) => (
            <div key={c.k} className="rounded-lg border border-slate-200 bg-slate-50 p-3">
              <div className="text-xs font-semibold uppercase tracking-wide text-slate-400">{c.k}</div>
              <div className="mt-0.5 font-semibold text-slate-700">{c.v}</div>
              <div className="mt-0.5 text-xs text-slate-400">{c.d}</div>
            </div>
          ))}
        </div>
      </section>

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

      {/* Los dos mundos */}
      <section className="mb-8">
        <h2 className="mb-3 text-sm font-semibold uppercase tracking-wide text-slate-400">Los dos mundos del sistema</h2>
        <div className="grid gap-4 sm:grid-cols-2">
          <div className="rounded-xl border border-amber-200 bg-amber-50/50 p-4">
            <div className="flex items-center gap-2">
              <span className="h-2.5 w-2.5 rounded-full bg-amber-500" />
              <span className="font-semibold text-amber-700">Offline · antes de transmitir</span>
            </div>
            <p className="mt-1.5 text-sm text-slate-600">
              Lo que se calcula <strong>una vez</strong> con datos etiquetados: entrenar el modelo (CSP + LDA)
              y validar su precisión. Secciones <strong>Entrenamiento</strong> y <strong>Resultados</strong>.
            </p>
          </div>
          <div className="rounded-xl border border-emerald-200 bg-emerald-50/50 p-4">
            <div className="flex items-center gap-2">
              <span className="h-2.5 w-2.5 rounded-full bg-emerald-500" />
              <span className="font-semibold text-emerald-700">Online · en tiempo real</span>
            </div>
            <p className="mt-1.5 text-sm text-slate-600">
              Lo que ocurre <strong>ahora</strong>, de forma causal: la señal llega y se clasifica ventana a
              ventana. Secciones <strong>Laboratorio</strong>, <strong>Clasificación</strong> y <strong>Cerebro 3D</strong>.
            </p>
          </div>
        </div>
      </section>

      {/* Métricas clave por dataset */}
      <section>
        <h2 className="mb-3 text-sm font-semibold uppercase tracking-wide text-slate-400">Precisión por dataset (k-fold, within-subject)</h2>
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
