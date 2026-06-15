import { useEffect, useMemo, useState } from 'react'
import { ScatterChart, Scatter, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend } from 'recharts'
import { PageShell } from '../components/PageShell'
import type { HelpContent } from '../components/HelpButton'
import { Widget } from '../components/Widget'
import { Topomap, type Pos2D } from '../components/Topomap'
import { EEGNetModel } from '../components/EEGNetModel'
import { useStore } from '../store/useStore'
import { getJSON } from '../api/client'

interface CSPResp {
  channels: string[]; eigenvalues: number[]; patterns: number[][]
  classes: string[]; features: number[][]; labels: string[]; pos2d: Pos2D
}

const CLASS_COLORS = ['#2563eb', '#e11d48', '#059669', '#d97706']

const HELP: HelpContent = {
  pipeline: 'Etapa 3 del pipeline · Filtrado espacial (CSP)',
  intro: 'Una vez filtrada la señal en el tiempo, el CSP (Common Spatial Patterns) combina los canales entre sí para resaltar las diferencias entre las dos clases. Es un filtro espacial lineal: el análogo del FIR, pero operando sobre el espacio (los electrodos) en lugar del tiempo. Su salida alimenta directamente al clasificador.',
  points: [
    { label: 'Por qué combinar canales', desc: 'Imaginar cada mano activa regiones cerebrales distintas (la mano izquierda se controla desde el hemisferio derecho, y viceversa). Un solo electrodo capta una mezcla de fuentes solapadas; al combinar linealmente todos los canales (Z = W·X) se construyen «señales virtuales» que aíslan mejor esa diferencia espacial entre clases.' },
    { label: 'Qué optimiza el CSP', desc: 'Busca las combinaciones de canales cuya potencia —la varianza de la señal— sea máxima para una clase y mínima para la otra. Esto se plantea como un problema de autovalores generalizados sobre las matrices de covarianza de cada clase, que se resuelve por blanqueo (whitening) y diagonalización conjunta.' },
    { label: 'Los topomapas y el autovalor λ', desc: 'Cada topomapa representa un filtro espacial (componente), con el peso de cada electrodo codificado en color. Su autovalor λ (entre 0 y 1) indica cómo reparte la varianza entre clases: cercano a 1 resalta una clase y cercano a 0 la otra. Se conservan los componentes de ambos extremos, los más discriminativos, que típicamente se lateralizan sobre la corteza motora (C3/C4).' },
    { label: 'La separabilidad', desc: 'El diagrama de dispersión sitúa cada trial según dos componentes CSP, coloreado por su clase real. Si las dos nubes de puntos aparecen separadas, significa que el CSP ha extraído características que permiten distinguir las clases: ese es exactamente el objetivo de esta etapa.' },
    { label: 'Se entrena una vez, se aplica en vivo', desc: 'Conviene distinguir dos fases. Aprender los filtros W —lo que se visualiza aquí— se hace una sola vez, fuera de línea, con datos etiquetados, porque necesita las covarianzas de cada clase. En cambio, aplicarlos es solo una multiplicación matricial Z = W·X, instantánea. Además, el CSP combina los canales en un mismo instante, sin mezclar muestras de distintos tiempos, por lo que es causal por naturaleza y no añade retardo alguno: por eso se aplica sin problema en tiempo real (es justo lo que ocurre en la sección En vivo). Esta sección, por tanto, muestra los filtros ya entrenados, no un proceso de transmisión.' },
  ],
  terms: ['CSP', 'Filtro espacial', 'Varianza y log-varianza', 'Problema de autovalores generalizados', 'Whitening'],
}

export default function SpatialCSP() {
  const { dataset, subject } = useStore()
  const [csp, setCsp] = useState<CSPResp | null>(null)
  const [tab, setTab] = useState<'csp' | 'eegnet'>('csp')

  useEffect(() => {
    setCsp(null)
    getJSON<CSPResp>(`/csp?dataset=${dataset}&subject=${subject}`).then(setCsp).catch(() => setCsp(null))
  }, [dataset, subject])

  // scatter de separabilidad: componente extremo alto (0) vs bajo (último)
  const scatter = useMemo(() => {
    if (!csp) return null
    const last = csp.features[0].length - 1
    const byClass = csp.classes.map((cls) =>
      csp.features.filter((_, i) => csp.labels[i] === cls).map((f) => ({ x: f[0], y: f[last] })),
    )
    return { last, byClass }
  }, [csp])

  return (
    <PageShell
      title="El Modelo · Filtrado espacial (CSP)"
      subtitle="Cómo se entrena el clasificador clásico y, como espejo, qué aprende EEGNet."
      help={HELP}
      world="offline"
    >
      {/* Pestañas: el clásico (lo que diseñas/entrenas) vs EEGNet como espejo */}
      <div className="mb-4 flex rounded-md border border-slate-300 p-0.5 text-sm w-fit">
        {([['csp', 'Clásico (FIR + CSP)'], ['eegnet', 'EEGNet (filtros aprendidos)']] as const).map(([id, label]) => (
          <button key={id} onClick={() => setTab(id)}
            className={`rounded px-3 py-1 ${tab === id ? 'bg-primary text-white' : 'text-slate-600 hover:bg-slate-100'}`}>
            {label}
          </button>
        ))}
      </div>

      {tab === 'eegnet' ? (
        <EEGNetModel dataset={dataset} subject={subject} csp={csp} />
      ) : !csp ? (
        <div className="flex h-64 items-center justify-center text-slate-300">Calculando CSP…</div>
      ) : (
        <div className="grid items-start gap-4 lg:grid-cols-3">
          <div className="lg:col-span-2">
            <Widget title="Patrones espaciales  (un filtro por componente)" accent="csp">
              <div className="flex flex-wrap justify-around gap-2">
                {csp.patterns.map((pat, ci) => (
                  <div key={ci} className="text-center">
                    <Topomap channels={csp.channels} pos2d={csp.pos2d} values={pat} size={160} />
                    <div className="text-xs text-slate-500">comp {ci} · <span className="font-mono">λ={csp.eigenvalues[ci].toFixed(2)}</span></div>
                  </div>
                ))}
              </div>
            </Widget>
          </div>

          <div className="space-y-4">
            <Widget title="Separabilidad de clases (log-varianza)" accent="csp">
              {scatter && (
                <ResponsiveContainer width="100%" height={250}>
                  <ScatterChart margin={{ top: 8, right: 8, bottom: 20, left: 0 }}>
                    <CartesianGrid stroke="#eef2f7" />
                    <XAxis type="number" dataKey="x" name={`comp 0`} tick={{ fontSize: 11 }}
                      label={{ value: 'log-var comp 0', position: 'bottom', fontSize: 11, fill: '#94a3b8' }} />
                    <YAxis type="number" dataKey="y" name={`comp ${scatter.last}`} tick={{ fontSize: 11 }}
                      label={{ value: `comp ${scatter.last}`, angle: -90, position: 'insideLeft', fontSize: 11, fill: '#94a3b8' }} />
                    <Tooltip cursor={{ strokeDasharray: '3 3' }} />
                    <Legend wrapperStyle={{ fontSize: 11 }} />
                    {scatter.byClass.map((pts, i) => (
                      <Scatter key={i} name={csp.classes[i]} data={pts} fill={CLASS_COLORS[i % CLASS_COLORS.length]} fillOpacity={0.6} />
                    ))}
                  </ScatterChart>
                </ResponsiveContainer>
              )}
            </Widget>
          </div>
        </div>
      )}
    </PageShell>
  )
}
