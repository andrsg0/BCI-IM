import { useNavigate } from 'react-router-dom'
import { Activity, ArrowRight, BarChart3, Brain, Network, Radio, Scale } from 'lucide-react'
import { HelpButton, type HelpContent } from '../components/HelpButton'
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

// Pilares del proyecto: el qué, el cómo y la comparativa.
const PILLARS = [
  {
    icon: Activity,
    title: 'Decodificación en Tiempo Real',
    tag: 'El Desafío',
    color: 'var(--accent-signal)',
    desc: 'Clasificación binaria de Imaginación Motora (mano izquierda vs. derecha) simulada ventana a ventana con datos bioeléctricos reales.',
  },
  {
    icon: Scale,
    title: 'Teoría LTI Explícita',
    tag: 'La Filosofía',
    color: 'var(--accent-fir)',
    desc: 'Todo el procesamiento matemático —convolución, filtros FIR y patrones espaciales (CSP)— está programado a mano, sin librerías opacas.',
  },
  {
    icon: Network,
    title: 'El Espejo de Deep Learning',
    tag: 'La Comparativa',
    color: 'var(--accent-metric)',
    desc: 'Enfrentamos el pipeline matemático clásico contra EEGNet, analizando si la IA hereda la lógica humana al auto-aprender.',
  },
]

// Bloques del pipeline: cada uno enlaza a su análisis matemático.
const BLOCKS = [
  { label: 'Adquisición', sub: 'EEG  x[n]', dim: 'Entrada', to: '/lab', color: 'var(--accent-signal)' },
  { label: 'Filtro FIR', sub: 'convolución µ/β', dim: 'Tiempo', to: '/lab', color: 'var(--accent-fir)' },
  { label: 'CSP', sub: 'filtro espacial', dim: 'Espacio', to: '/csp', color: 'var(--accent-csp)' },
  { label: 'LDA', sub: 'frontera lineal', dim: 'Geometría', to: '/csp', color: 'var(--accent-metric)' },
]

export default function Home() {
  const nav = useNavigate()
  return (
    <div className="mx-auto max-w-7xl space-y-12">
      {/* 1 · Hero ------------------------------------------------------------ */}
      <section className="relative overflow-hidden rounded-3xl border border-slate-200 bg-gradient-to-br from-slate-900 via-slate-800 to-indigo-900 px-8 py-14 shadow-card sm:px-12 sm:py-16">
        <div className="absolute right-5 top-5">
          <HelpButton title="Inicio" help={HELP} />
        </div>
        <div className="pointer-events-none absolute -right-10 -top-10 opacity-10">
          <Brain size={220} className="text-white" />
        </div>
        <div className="relative max-w-3xl">
          <span className="inline-flex items-center gap-2 rounded-full bg-white/10 px-3 py-1 text-xs font-medium uppercase tracking-wide text-indigo-200">
            <Brain size={14} /> Interfaz Cerebro-Computadora · Imaginación Motora
          </span>
          <h1 className="mt-5 text-4xl font-extrabold leading-tight text-white sm:text-5xl">
            Abriendo la Caja Negra de las Interfaces Cerebro-Computadora
          </h1>
          <p className="mt-5 text-lg italic leading-relaxed text-slate-300">
            Un laboratorio interactivo para explorar el procesamiento de señales EEG: desde las
            matemáticas lineales clásicas hasta el poder de las Redes Convolucionales.
          </p>
          <div className="mt-8 flex flex-wrap gap-3">
            <button
              onClick={() => nav('/lab')}
              className="inline-flex items-center gap-2 rounded-lg bg-white px-5 py-2.5 text-sm font-semibold text-slate-900 transition hover:bg-slate-100"
            >
              Explorar el laboratorio <ArrowRight size={16} />
            </button>
            <button
              onClick={() => nav('/live')}
              className="inline-flex items-center gap-2 rounded-lg border border-white/30 px-5 py-2.5 text-sm font-semibold text-white transition hover:bg-white/10"
            >
              Ver demo en vivo <Radio size={16} />
            </button>
          </div>
        </div>
      </section>

      {/* 2 · Pilares -------------------------------------------------------- */}
      <section className="grid gap-5 md:grid-cols-3">
        {PILLARS.map((p) => {
          const Icon = p.icon
          return (
            <div
              key={p.title}
              className="rounded-2xl border border-slate-200 bg-white p-6 shadow-card transition hover:-translate-y-1 hover:shadow-md"
              style={{ borderTop: `4px solid ${p.color}` }}
            >
              <div className="flex h-12 w-12 items-center justify-center rounded-xl" style={{ background: `color-mix(in srgb, ${p.color} 15%, white)` }}>
                <Icon size={24} style={{ color: p.color }} />
              </div>
              <div className="mt-4 text-xs font-semibold uppercase tracking-wide text-slate-400">{p.tag}</div>
              <h3 className="mt-1 text-lg font-bold text-slate-800">{p.title}</h3>
              <p className="mt-2 text-sm leading-relaxed text-slate-600">{p.desc}</p>
            </div>
          )
        })}
      </section>

      {/* 3 · Pipeline interactivo ------------------------------------------- */}
      <section className="rounded-2xl border border-slate-200 bg-white p-6 shadow-card sm:p-8">
        <h2 className="mb-1 text-sm font-semibold uppercase tracking-wide text-slate-400">El pipeline</h2>
        <p className="mb-5 text-sm text-slate-500">Un viaje de transformación de datos. Haz clic en cualquier bloque para ir a su análisis.</p>
        <div className="flex flex-wrap items-stretch gap-2">
          {BLOCKS.map((b, i) => (
            <div key={b.label} className="flex items-stretch gap-2">
              <button
                onClick={() => nav(b.to)}
                className="group flex min-w-40 flex-col rounded-xl border border-slate-200 bg-slate-50 px-4 py-3 text-left transition-all hover:-translate-y-1 hover:shadow-md"
                style={{ borderTop: `3px solid ${b.color}` }}
              >
                <div className="text-[10px] font-semibold uppercase tracking-wider" style={{ color: b.color }}>{b.dim}</div>
                <div className="mt-0.5 font-semibold text-slate-700 group-hover:text-slate-900">{b.label}</div>
                <div className="text-xs text-slate-400">{b.sub}</div>
              </button>
              {i < BLOCKS.length - 1 && (
                <div className="flex items-center"><ArrowRight className="text-slate-300" size={20} /></div>
              )}
            </div>
          ))}
        </div>
        <p className="mt-5 max-w-3xl text-sm leading-relaxed text-slate-500">
          Cada bloque representa una operación lineal pura. El sistema transforma las micro-vibraciones
          del cerebro en comandos binarios en tres dimensiones críticas: el <strong>Tiempo</strong> (filtro
          FIR), el <strong>Espacio</strong> (CSP) y la <strong>Geometría</strong> de los datos (LDA).
        </p>
      </section>

      {/* 4 · Elige tu entorno ----------------------------------------------- */}
      <section>
        <h2 className="mb-4 text-sm font-semibold uppercase tracking-wide text-slate-400">Elige tu entorno</h2>
        <div className="grid gap-5 sm:grid-cols-2">
          {/* Laboratorio offline */}
          <button
            onClick={() => nav('/csp')}
            className="group rounded-2xl border border-amber-200 bg-amber-50/50 p-6 text-left transition hover:-translate-y-1 hover:shadow-md"
          >
            <div className="flex items-center gap-3">
              <div className="flex h-11 w-11 items-center justify-center rounded-xl bg-amber-100">
                <BarChart3 size={22} className="text-amber-600" />
              </div>
              <div>
                <div className="text-xs font-semibold uppercase tracking-wide text-amber-500">1 · Laboratorio Offline</div>
                <div className="font-bold text-amber-800">Cómo se entrena el modelo</div>
              </div>
            </div>
            <p className="mt-3 text-sm leading-relaxed text-slate-600">
              Explora cómo se entrenan y validan matemáticamente los modelos analizando poblaciones de
              5 datasets públicos.
            </p>
            <div className="mt-3 flex items-center gap-2 text-sm font-medium text-amber-700">
              📂 Entrenamiento y Resultados <ArrowRight size={15} className="transition group-hover:translate-x-1" />
            </div>
          </button>

          {/* Demo en vivo */}
          <button
            onClick={() => nav('/live')}
            className="group rounded-2xl border border-emerald-200 bg-emerald-50/50 p-6 text-left transition hover:-translate-y-1 hover:shadow-md"
          >
            <div className="flex items-center gap-3">
              <div className="flex h-11 w-11 items-center justify-center rounded-xl bg-emerald-100">
                <Radio size={22} className="text-emerald-600" />
              </div>
              <div>
                <div className="text-xs font-semibold uppercase tracking-wide text-emerald-500">2 · Demostración en Vivo</div>
                <div className="font-bold text-emerald-800">Procesamiento causal en tiempo real</div>
              </div>
            </div>
            <p className="mt-3 text-sm leading-relaxed text-slate-600">
              Experimenta el procesamiento causal en tiempo real. Una señal EEG entrante se procesa
              ventana a ventana, simulando un entorno clínico real.
            </p>
            <div className="mt-3 flex items-center gap-2 text-sm font-medium text-emerald-700">
              🟢 Laboratorio, Clasificación y Cerebro 3D <ArrowRight size={15} className="transition group-hover:translate-x-1" />
            </div>
          </button>
        </div>
      </section>

      {/* 5 · Rendimiento base ----------------------------------------------- */}
      <section>
        <h2 className="text-lg font-bold text-slate-800">Rendimiento Base (Within-Subject)</h2>
        <p className="mt-1 max-w-3xl text-sm text-slate-500">
          Un vistazo rápido a la precisión media del clasificador clásico (CSP+LDA) a través de las
          diferentes poblaciones del estado del arte, antes de explorar los modelos.
        </p>
        <div className="mt-4 grid gap-4 sm:grid-cols-3">
          {DATASET_LIST.map((d) => (
            <button
              key={d.id}
              onClick={() => nav('/results')}
              className="rounded-xl border border-slate-200 bg-white p-4 text-left shadow-card transition hover:-translate-y-1 hover:shadow-md"
            >
              <div className="text-sm text-slate-500">{d.label}</div>
              <div className="mt-1 text-3xl font-bold text-slate-800">{(d.accuracy * 100).toFixed(0)}%</div>
              <div className="mt-1 text-xs text-slate-400">{d.subjects} sujetos · {d.fs} Hz</div>
            </button>
          ))}
        </div>
      </section>
    </div>
  )
}
