import { PageShell } from '../components/PageShell'
import type { HelpContent } from '../components/HelpButton'
import { Widget } from '../components/Widget'
import { DATASET_LIST } from '../lib/datasets'

const HELP: HelpContent = {
  pipeline: 'Evaluación del pipeline completo',
  intro: 'Resume el rendimiento del sistema —el mismo pipeline LTI (FIR → CSP → LDA)— aplicado a distintos conjuntos de datos públicos de imaginación motora, sin modificar el código, únicamente la configuración. Sirve para valorar de forma objetiva la calidad del método y comparar entre poblaciones.',
  points: [
    { label: 'Cómo se mide', desc: 'La precisión (accuracy) se obtiene por validación cruzada y de forma independiente para cada sujeto (within-subject), porque los patrones EEG varían mucho de una persona a otra y no sería válido mezclarlas. Se reporta el promedio sobre los sujetos.' },
    { label: 'Por qué varía entre datasets', desc: 'Cada dataset emplea distinto número de sujetos, de canales y frecuencia de muestreo. La precisión tiende a disminuir cuando hay menos trials por sujeto: estimar la matriz de covarianza que necesita el CSP con pocos datos resulta inestable, un efecto conocido como maldición de la dimensionalidad.' },
    { label: 'Variabilidad entre personas', desc: 'Algunos sujetos producen una desincronización (ERD) clara y otros apenas la generan, fenómeno denominado «BCI illiteracy». Por ese motivo siempre se informa la media sobre sujetos y nunca un único caso, que podría ser engañoso.' },
  ],
  terms: ['Validación cruzada', 'Accuracy y kappa', 'CSP'],
}

export default function Results() {
  return (
    <PageShell
      title="Resultados"
      subtitle="El mismo pipeline en distintos datasets públicos."
      help={HELP}
      world="offline"
    >
      <div className="grid gap-4 lg:grid-cols-2">
        <Widget title="Comparativa de datasets (accuracy k-fold)" accent="metric">
          <div className="space-y-3 p-1">
            {DATASET_LIST.map((d) => (
              <div key={d.id}>
                <div className="mb-1 flex justify-between text-sm text-slate-600">
                  <span>{d.label}</span>
                  <span className="font-semibold">{(d.accuracy * 100).toFixed(1)}%</span>
                </div>
                <div className="h-3 overflow-hidden rounded-full bg-slate-100">
                  <div className="h-full rounded-full bg-emerald-500" style={{ width: `${d.accuracy * 100}%` }} />
                </div>
                <div className="mt-0.5 text-xs text-slate-400">{d.subjects} sujetos · {d.fs} Hz</div>
              </div>
            ))}
          </div>
        </Widget>
      </div>
    </PageShell>
  )
}
