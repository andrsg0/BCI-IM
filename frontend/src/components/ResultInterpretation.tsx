import { useState } from 'react'
import { Info, ChevronDown } from 'lucide-react'
import { GlossaryText } from './GlossaryText'

/**
 * Leyenda de interpretación HONESTA de los resultados: por qué la precisión varía
 * tanto entre sujetos y entre escenarios, sin inventar etiquetas que el dataset no
 * tiene. Es la mitad «datos reales» de la decisión «ambas cosas» del roadmap
 * (la otra mitad es el simulador sintético de señal degradada).
 *
 * Colapsable para no robar espacio: por defecto cerrada, se abre al pulsar.
 */
const REASONS: { title: string; body: string }[] = [
  {
    title: 'BCI illiteracy (variabilidad entre personas)',
    body: 'Entre un 15 % y un 30 % de las personas producen una desincronización µ/β débil o poco separable: su EEG de imaginación motora apenas se distingue del reposo. No es un fallo del método ni del montaje, es una característica fisiológica del sujeto. Por eso un mismo pipeline da 0.90 en una persona y 0.55 en otra.',
  },
  {
    title: 'Calidad de la señal y montaje',
    body: 'Impedancia de los electrodos, grosor del cuero cabelludo, artefactos oculares/musculares y el número de canales sobre la corteza motora cambian cuánta información útil llega. Un dataset con 64 canales y buena colocación parte con ventaja sobre uno de 3 canales, aunque el clasificador sea idéntico.',
  },
  {
    title: 'Pocos trials por sujeto',
    body: 'El CSP y el LDA estiman covarianzas y una frontera a partir de los trials de entrenamiento. Con pocos trials la estimación es ruidosa y el modelo sobreajusta: la accuracy medida tiene más varianza y suele quedar por debajo del techo real del sujeto.',
  },
  {
    title: 'Within vs cross-subject',
    body: 'Within-subject está calibrado en la misma persona (techo). Cross-subject (LOSO) se prueba en alguien que el modelo nunca vio: la anatomía y los patrones espaciales cambian de una cabeza a otra, así que la cross siempre cae. Que no se desplome al azar indica que hay estructura compartida entre sujetos.',
  },
  {
    title: 'No hay clase «reposo»',
    body: 'El clasificador es binario (izquierda vs derecha): no tiene una etiqueta para «no hacer nada». Lo que parece un error puede ser una ventana de transición o reposo forzada a elegir un lado. En la demo en vivo esto se mitiga con el umbral de confianza (abstención).',
  },
]

export function ResultInterpretation() {
  const [open, setOpen] = useState(false)
  return (
    <div className="rounded-xl border border-slate-200 bg-white shadow-sm">
      <button
        onClick={() => setOpen((v) => !v)}
        className="flex w-full items-center justify-between gap-3 px-4 py-3 text-left"
        aria-expanded={open}
      >
        <span className="flex items-center gap-2 text-sm font-semibold text-slate-700">
          <Info size={15} className="text-sky-500" />
          Cómo interpretar estos números (por qué varía tanto la precisión)
        </span>
        <ChevronDown size={16} className={`shrink-0 text-slate-400 transition-transform ${open ? 'rotate-180' : ''}`} />
      </button>

      {open && (
        <div className="border-t border-slate-100 px-4 py-3">
          <p className="mb-3 text-xs leading-relaxed text-slate-500">
            <GlossaryText>
              Estas cifras salen del pipeline real; no hay valores inventados ni etiquetas que el
              dataset no tenga. La precisión de una BCI de imaginación motora no invasiva tiene un
              techo de ~70–85 % en 2 clases y depende de varios factores honestos:
            </GlossaryText>
          </p>
          <ul className="space-y-2.5">
            {REASONS.map((r) => (
              <li key={r.title} className="rounded-lg bg-slate-50 px-3 py-2">
                <p className="text-xs font-semibold text-slate-700">{r.title}</p>
                <p className="mt-0.5 text-[11px] leading-relaxed text-slate-500">
                  <GlossaryText>{r.body}</GlossaryText>
                </p>
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  )
}
