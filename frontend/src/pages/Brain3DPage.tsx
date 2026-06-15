import { useEffect, useRef, useState } from 'react'
import { Radio } from 'lucide-react'
import { PageShell } from '../components/PageShell'
import type { HelpContent } from '../components/HelpButton'
import { Widget } from '../components/Widget'
import { Brain3D, type Pos3D } from '../components/Brain3D'
import { useStore } from '../store/useStore'
import { openStream, getJSON } from '../api/client'

interface PosResp { channels: string[]; pos3d: Pos3D }
interface Msg { pred: string; probs: Record<string, number>; power: number[] }

const CLASS_COLORS = ['#2563eb', '#e11d48', '#059669', '#d97706']

const HELP: HelpContent = {
  pipeline: 'Visualización en vivo de la actividad cortical',
  intro: 'Representa sobre una cabeza tridimensional la actividad de la señal mientras se transmite en tiempo real. Cada electrodo del sistema 10-20 se ilumina según cuánto se desvía su potencia µ/β del promedio del cuero cabelludo en ese instante: así se ve qué regiones “se están usando” durante la imaginación motora.',
  points: [
    { label: 'Qué significa el brillo', desc: 'Cuanto más se aparta la potencia de un electrodo respecto a la media instantánea, más brilla. Durante la imaginación de una mano aparece una desincronización (ERD) sobre la corteza motora contraria (C3 para la mano derecha, C4 para la izquierda): esa lateralización es justo lo que el sistema explota para clasificar.' },
    { label: 'Por qué en vivo', desc: 'A diferencia de los pesos fijos del CSP (que se ven en El Modelo), aquí los colores cambian ventana a ventana con la señal causal que llega, igual que la predicción. Pulsa Play en el panel lateral para iniciar la transmisión.' },
    { label: 'Navegación', desc: 'Arrastra para rotar el modelo y usa la rueda para acercar o alejar. La nariz indica el frente de la cabeza.' },
  ],
  terms: ['ERD/ERS', 'Banda µ/β', 'Sistema 10-20', 'Causalidad'],
}

export default function Brain3DPage() {
  const { dataset, subject, playing, clearToken } = useStore()
  const [pos, setPos] = useState<PosResp | null>(null)
  const [values, setValues] = useState<number[]>([])
  const [last, setLast] = useState<Msg | null>(null)
  const ema = useRef<number[]>([])

  // posiciones de los electrodos (no depende del modelo)
  useEffect(() => {
    setPos(null); setLast(null); ema.current = []
    getJSON<PosResp>(`/positions?dataset=${dataset}&subject=${subject}`)
      .then((p) => { setPos(p); ema.current = new Array(p.channels.length).fill(0); setValues(ema.current.slice()) })
      .catch(() => setPos(null))
  }, [dataset, subject])

  useEffect(() => { ema.current = ema.current.map(() => 0); setValues(ema.current.slice()) }, [clearToken])

  // stream en vivo: ilumina los electrodos con la potencia por canal
  useEffect(() => {
    if (!playing || !pos) return
    const ws = openStream(`/stream?dataset=${dataset}&subject=${subject}`, (d) => {
      const m = d as Msg
      if (!m.power) return
      const mean = m.power.reduce((a, b) => a + b, 0) / m.power.length
      // suavizado exponencial para un brillo más estable
      ema.current = m.power.map((p, i) => 0.55 * (ema.current[i] ?? 0) + 0.45 * (p - mean))
      setValues(ema.current.slice())
      setLast(m)
    })
    return () => ws.close()
  }, [playing, pos, dataset, subject])

  const colorOf = (cls: string) => {
    const ks = last ? Object.keys(last.probs) : []
    return CLASS_COLORS[Math.max(0, ks.indexOf(cls)) % CLASS_COLORS.length]
  }

  return (
    <PageShell
      title="Cerebro 3D en vivo"
      subtitle="La cabeza se ilumina con la actividad µ/β de la señal mientras se transmite."
      help={HELP}
      world="online"
    >
      {!pos ? (
        <div className="flex h-64 items-center justify-center text-slate-300">Cargando modelo…</div>
      ) : (
        <div className="grid items-start gap-4 lg:grid-cols-4">
          <div className="lg:col-span-3">
            <Widget title="Actividad cortical en vivo" accent="brain">
              <div className="relative h-[520px] overflow-hidden rounded-lg">
                <Brain3D channels={pos.channels} pos3d={pos.pos3d} values={values} />
                <span className={`absolute left-3 top-3 flex items-center gap-1.5 rounded-full px-2.5 py-1 text-xs ${playing ? 'bg-red-50 text-red-600' : 'bg-slate-100 text-slate-500'}`}>
                  <Radio size={13} className={playing ? 'animate-pulse' : ''} /> {playing ? 'EN VIVO' : 'detenido'}
                </span>
              </div>
            </Widget>
          </div>

          <div className="space-y-4">
            <Widget title="Predicción" accent="metric">
              {!last ? (
                <div className="py-6 text-center text-sm text-slate-300">
                  Pulsa <strong>Play</strong> en el panel lateral
                </div>
              ) : (
                <div className="space-y-3 py-1">
                  <div className="text-center text-2xl font-bold" style={{ color: colorOf(last.pred) }}>{last.pred}</div>
                  {Object.keys(last.probs).map((cls) => (
                    <div key={cls}>
                      <div className="mb-1 flex justify-between text-sm text-slate-600">
                        <span>{cls}</span><span className="font-mono">{(last.probs[cls] * 100).toFixed(0)}%</span>
                      </div>
                      <div className="h-2.5 overflow-hidden rounded-full bg-slate-100">
                        <div className="h-full rounded-full transition-all" style={{ width: `${last.probs[cls] * 100}%`, background: colorOf(cls) }} />
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </Widget>

            <Widget title="Cómo leerlo" accent="brain">
              <div className="space-y-2 text-xs text-slate-600">
                <p>El brillo indica cuánto se desvía la potencia µ/β de cada electrodo respecto a la media instantánea.</p>
                <p>Busca la <strong>lateralización</strong> sobre C3 (derecha) y C4 (izquierda): es la firma de la imaginación motora.</p>
              </div>
            </Widget>
          </div>
        </div>
      )}
    </PageShell>
  )
}
