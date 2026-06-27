import { useEffect, useMemo, useRef, useState } from 'react'
import { Radio } from 'lucide-react'
import { PageShell } from '../components/PageShell'
import type { HelpContent } from '../components/HelpButton'
import { Widget } from '../components/Widget'
import { Brain3D, type Pos3D } from '../components/Brain3D'
import { divergingColor } from '../lib/color'
import { useStore } from '../store/useStore'
import { openStream, getJSON } from '../api/client'
import { progressFromFrame, type ProgressFrame } from '../lib/progress'

interface PosResp { channels: string[]; pos3d: Pos3D }
interface Msg { pred: string; probs: Record<string, number>; power: number[] }

const CLASS_COLORS = ['#2563eb', '#e11d48', '#059669', '#d97706']

// Clasifica un electrodo 10-10 por nombre: ¿es de la corteza motora? ¿qué hemisferio?
// (impar = izquierda, par = derecha, z = línea media). Ej.: C3→motor/L, FC4→motor/R, POz→no.
function elecInfo(ch: string): { motor: boolean; side: 'L' | 'R' | 'M' } {
  const m = ch.match(/^([A-Za-z]+?)(z|\d+)$/i)
  if (!m) return { motor: false, side: 'M' }
  const prefix = m[1].toUpperCase()
  const suf = m[2].toLowerCase()
  const motor = ['C', 'FC', 'CP', 'FCC', 'CCP'].includes(prefix)
  const side: 'L' | 'R' | 'M' = suf === 'z' ? 'M' : parseInt(suf, 10) % 2 === 1 ? 'L' : 'R'
  return { motor, side }
}

/** Barra divergente (azul − / rojo +) para comparar hemisferios. */
function DivBar({ label, value, scale }: { label: string; value: number; scale: number }) {
  const t = Math.max(-1, Math.min(1, value / scale))
  const pct = Math.abs(t) * 50
  return (
    <div className="flex items-center gap-2">
      <span className="w-8 text-[11px] text-slate-500">{label}</span>
      <div className="relative h-3 flex-1 rounded bg-slate-100">
        <div className="absolute bottom-0 left-1/2 top-0 w-px bg-slate-300" />
        <div className="absolute bottom-0 top-0 rounded" style={{ background: divergingColor(t), width: `${pct}%`, left: t < 0 ? `${50 - pct}%` : '50%' }} />
      </div>
    </div>
  )
}

const HELP: HelpContent = {
  pipeline: 'Visualización en vivo de la actividad cortical',
  intro: 'Esta es una “cámara térmica” del cerebro: pinta, en tiempo real, en qué zonas de la cabeza hay más o menos actividad eléctrica mientras la persona imagina mover una mano. Los anillos flotando sobre la cabeza son los electrodos (los sensores que tocan el cuero cabelludo); la corteza coloreada es ese mismo dato “rellenado” entre electrodos para que se vea como un mapa de calor continuo.',
  points: [
    { label: 'Qué mide el color', desc: 'El color NO es el voltaje crudo: es la POTENCIA en la banda µ/β (las ondas de ~8–30 Hz, las que se relacionan con el movimiento). Rojo = esa zona tiene MÁS potencia µ/β ahora; azul = tiene MENOS; blanco = está casi igual que la referencia (no se desvía). Es la misma escala que un termómetro: caliente/frío/neutro.' },
    { label: 'Por qué hay nodos casi “transparentes”', desc: 'Un electrodo se ve pálido o casi invisible cuando su valor es ≈ 0, es decir, cuando su potencia µ/β no cambia respecto a la referencia: se pinta blanco sobre un fondo blanco. No es un fallo ni un sensor roto. Pasa más o menos según el dataset porque cada uno tiene distinto montaje: los electrodos de la línea media (Cz) o los alejados de la corteza motora suelen quedarse cerca de 0 y por eso se ven apagados. (Si un canal tiene un nombre que no está en el montaje estándar 10-20, directamente no se dibuja.)' },
    { label: 'La idea clave: ERD (la “firma” del movimiento)', desc: 'Cuando imaginas mover una mano, la zona del cerebro que controla esa mano se “destapa”: BAJA su potencia µ/β. Eso se llama ERD (desincronización relacionada al evento). Y ocurre en el lado CONTRARIO a la mano: imaginar la mano DERECHA baja la actividad en la corteza IZQUIERDA (electrodo C3), y la mano IZQUIERDA la baja en la corteza DERECHA (electrodo C4). Esa caída cruzada (que se ve azul) es justo la pista que el clasificador aprovecha para adivinar qué mano imaginaste.' },
    { label: 'Modo “ERD (lateralización)”', desc: 'Compara cada electrodo consigo mismo: con su propia potencia de los segundos anteriores (su “línea base”). Así resalta el CAMBIO, no el nivel absoluto, y deja ver la caída contralateral (el azul en C3 o C4). Es el modo recomendado para entender qué hace el sistema. Se elige con el conmutador de arriba a la derecha.' },
    { label: 'Modo “Instantánea”', desc: 'Compara cada electrodo con el promedio de TODOS los electrodos en ese mismo instante. Muestra qué sensor destaca ahora mismo respecto al resto. Es la señal más “en bruto”: más nerviosa, salta en los dos lados a la vez y cuesta más leer la lateralización. Útil para ver lo cruda que es la señal antes de procesarla.' },
    { label: 'Por qué parpadea y no es un dibujo fijo', desc: 'El color cambia ventana a ventana porque usa la señal causal que va llegando (solo el pasado, como un casco real), igual que la predicción de la demo. Por eso verás parpadeo: la lateralización limpia es una TENDENCIA a lo largo de varios segundos, no un corte perfecto en cada fotograma. OJO: esto NO es el patrón fijo que aprendió el modelo (ese, estable, está en “El Modelo / CSP”); aquí ves la señal viva del momento.' },
    { label: 'Cómo usarlo', desc: 'Pulsa Play en el panel lateral para que empiece a transmitir. Arrastra con el ratón para girar la cabeza y usa la rueda para acercar/alejar. Pasa el cursor sobre un anillo para ver el nombre del electrodo y su valor exacto. El botón “Solo corteza motora” atenúa (deja semitransparentes) los electrodos que no están sobre la corteza motora, para enfocar la franja C/CP donde se decide la imaginación.' },
  ],
  terms: ['ERD/ERS', 'Banda µ/β', 'Sistema 10-20', 'Causalidad', 'CSP'],
}

// Modo de coloreo (ver docs/frontend-design.md · "Cerebro 3D EN VIVO"):
//  - 'erd':  desviación de cada canal respecto a su PROPIA línea base reciente
//            (EMA lenta). Un descenso = azul = desincronización (ERD). Muestra la
//            lateralización contralateral de la imaginación motora.
//  - 'inst': desviación respecto a la media ESPACIAL del instante (todos los canales).
//            Más cruda y nerviosa: cuánto destaca cada electrodo ahora mismo.
type ColorMode = 'erd' | 'inst'

export default function Brain3DPage() {
  const { dataset, subject, playing, clearToken } = useStore()
  const [pos, setPos] = useState<PosResp | null>(null)
  const [values, setValues] = useState<number[]>([])
  const [last, setLast] = useState<Msg | null>(null)
  const [mode, setMode] = useState<ColorMode>('erd')
  const [focusMotor, setFocusMotor] = useState(false)
  const hist = useRef<number[][]>([])   // ventana reciente de valores (agregación temporal)
  const base = useRef<number[]>([])     // línea base lenta por canal (modo ERD)
  const modeRef = useRef<ColorMode>(mode)
  // Al cambiar de modo, la ventana acumulada deja de ser comparable: reiniciar.
  useEffect(() => { modeRef.current = mode; hist.current = [] }, [mode])

  // posiciones de los electrodos (no depende del modelo)
  useEffect(() => {
    setPos(null); setLast(null); hist.current = []; base.current = []
    getJSON<PosResp>(`/positions?dataset=${dataset}&subject=${subject}`)
      .then((p) => { setPos(p); hist.current = []; base.current = []; setValues(new Array(p.channels.length).fill(0)) })
      .catch(() => setPos(null))
  }, [dataset, subject])

  useEffect(() => { hist.current = []; base.current = []; setValues((v) => v.map(() => 0)); useStore.getState().resetProgress() }, [clearToken])

  // Al salir de la página, ocultar la barra de progreso del panel lateral.
  useEffect(() => () => useStore.getState().resetProgress(), [])

  // Clasificación motora de cada canal → máscara para atenuar los no-motores (foco)
  // y para el cálculo de la barra de lateralización por hemisferio.
  const dimMask = useMemo(
    () => (focusMotor && pos ? pos.channels.map((ch) => !elecInfo(ch).motor) : undefined),
    [focusMotor, pos],
  )
  const lat = useMemo(() => {
    if (!pos) return null
    let lS = 0, lN = 0, rS = 0, rN = 0
    pos.channels.forEach((ch, i) => {
      const e = elecInfo(ch); if (!e.motor) return
      const v = values[i] ?? 0
      if (e.side === 'L') { lS += v; lN++ } else if (e.side === 'R') { rS += v; rN++ }
    })
    return { left: lN ? lS / lN : 0, right: rN ? rS / rN : 0 }
  }, [pos, values])

  // stream en vivo: colorea electrodos + heatmap con la potencia µ/β por canal.
  // El modo se lee por ref (modeRef) para no reconectar el WS al alternar vistas.
  useEffect(() => {
    if (!playing || !pos) return
    const ws = openStream(`/stream?dataset=${dataset}&subject=${subject}`, (d) => {
      const m = d as Msg
      const prog = progressFromFrame(d as ProgressFrame)
      if (prog) useStore.getState().setProgress(prog[0], prog[1])
      if (!m.power) return
      let target: number[]
      if (modeRef.current === 'inst') {
        const mean = m.power.reduce((a, b) => a + b, 0) / m.power.length
        target = m.power.map((p) => p - mean)
      } else {
        if (base.current.length !== m.power.length) base.current = m.power.slice()
        base.current = m.power.map((p, i) => 0.98 * base.current[i] + 0.02 * p) // baseline lento
        target = m.power.map((p, i) => p - base.current[i])
      }
      // Agregación temporal: promedio de la ventana reciente. El ruido frame a frame
      // se cancela y queda la TENDENCIA (la lateralización persiste; el ruido no).
      // ERD usa ventana larga (~2.5 s) para ver la tendencia; Instantánea, corta.
      const win = modeRef.current === 'erd' ? 25 : 6
      const h = hist.current
      h.push(target)
      while (h.length > win) h.shift()
      const agg = target.map((_, i) => {
        let s = 0
        for (const f of h) s += f[i] ?? 0
        return s / h.length
      })
      setValues(agg)   // BrainMesh normaliza por su máximo
      setLast(m)
    })
    return () => ws.close()
  }, [playing, pos, dataset, subject])

  const colorOf = (cls: string) => {
    const ks = last ? Object.keys(last.probs) : []
    return CLASS_COLORS[Math.max(0, ks.indexOf(cls)) % CLASS_COLORS.length]
  }

  // Texto de la leyenda según el modo de coloreo activo.
  const legend = mode === 'erd'
    ? { pos: 'más µ/β', neg: 'menos µ/β', note: 'Tendencia ~2,5 s (no cada frame). Azul = ERD → lado activo (opuesto a la mano).' }
    : { pos: 'más µ/β', neg: 'menos µ/β', note: 'Instante crudo: cambia rápido en ambos lados a la vez.' }
  const latScale = lat ? Math.max(Math.abs(lat.left), Math.abs(lat.right), 1e-6) : 1

  return (
    <PageShell
      title="Cerebro 3D en vivo"
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
                <Brain3D channels={pos.channels} pos3d={pos.pos3d} values={values} dimMask={dimMask} />
                <span className={`absolute left-3 top-3 flex items-center gap-1.5 rounded-full px-2.5 py-1 text-xs ${playing ? 'bg-red-50 text-red-600' : 'bg-slate-100 text-slate-500'}`}>
                  <Radio size={13} className={playing ? 'animate-pulse' : ''} /> {playing ? 'EN VIVO' : 'detenido'}
                </span>

                {/* Leyenda fija: significado del color + recordatorio de que es tendencia */}
                <div className="absolute bottom-3 left-3 rounded-lg bg-white/85 px-2.5 py-1.5 text-[10px] text-slate-600 shadow-sm backdrop-blur">
                  <div className="flex items-center gap-3">
                    <span className="flex items-center gap-1"><i className="inline-block h-2.5 w-2.5 rounded-full" style={{ background: '#e11d48' }} /> {legend.pos}</span>
                    <span className="flex items-center gap-1"><i className="inline-block h-2.5 w-2.5 rounded-full bg-slate-300" /> ≈ 0</span>
                    <span className="flex items-center gap-1"><i className="inline-block h-2.5 w-2.5 rounded-full" style={{ background: '#2563eb' }} /> {legend.neg}</span>
                  </div>
                  <div className="mt-0.5 max-w-[22rem] text-slate-400">{legend.note}</div>
                </div>
                {/* Conmutador de vista + foco motor */}
                <div className="absolute right-3 top-3 flex flex-col items-end gap-1">
                  <div className="flex gap-1 rounded-lg bg-white/85 p-1 text-xs shadow-sm backdrop-blur">
                    {([['erd', 'ERD (lateralización)'], ['inst', 'Instantánea']] as [ColorMode, string][]).map(([k, label]) => (
                      <button
                        key={k}
                        onClick={() => setMode(k)}
                        title={k === 'erd'
                          ? 'Desviación de cada canal respecto a su línea base reciente (ERD): el descenso de potencia se ve azul.'
                          : 'Desviación respecto a la media espacial del instante: cuánto destaca cada electrodo ahora mismo.'}
                        className={`rounded-md px-2 py-1 transition-colors ${mode === k ? 'bg-slate-800 text-white' : 'text-slate-500 hover:text-slate-800'}`}
                      >
                        {label}
                      </button>
                    ))}
                  </div>
                  <button
                    onClick={() => setFocusMotor((f) => !f)}
                    title="Atenúa los electrodos no motores para enfocar la franja C/CP, donde se decide la imaginación motora."
                    className={`rounded-lg px-2 py-1 text-xs shadow-sm backdrop-blur transition-colors ${focusMotor ? 'bg-slate-800 text-white' : 'bg-white/85 text-slate-600 hover:text-slate-800'}`}
                  >
                    {focusMotor ? '◉' : '○'} Solo corteza motora
                  </button>
                </div>
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

            <Widget title="Lateralización motora" accent="brain">
              {lat ? (
                <div className="space-y-2">
                  <DivBar label="Izq" value={lat.left} scale={latScale} />
                  <DivBar label="Der" value={lat.right} scale={latScale} />
                  <p className="text-[11px] leading-snug text-slate-500">
                    {mode === 'erd'
                      ? 'Promedio µ/β (vs. base) de cada corteza motora. Azul = más caída (ERD) = lado más activo → la mano imaginada es la del lado contrario.'
                      : 'Potencia µ/β relativa de cada corteza motora en el instante.'}
                  </p>
                </div>
              ) : <div className="text-sm text-slate-300">—</div>}
            </Widget>

            <Widget title="Cómo leerlo" accent="brain">
              <div className="space-y-2 text-xs text-slate-600">
                <p><span className="font-semibold text-rose-600">Rojo</span> = más potencia µ/β; <span className="font-semibold text-blue-600">azul</span> = menos. El cerebro pinta el mismo dato interpolado <strong>entre</strong> electrodos (heatmap cortical).</p>
                <p><strong>ERD (lateralización):</strong> cada canal se compara con su propia potencia reciente. Imaginar una mano <strong>baja</strong> la potencia (azul) sobre la corteza motora <strong>contraria</strong>: C3 (izquierda) para la mano derecha, C4 (derecha) para la izquierda.</p>
                <p><strong>Instantánea:</strong> compara cada electrodo con la media del instante. Es la señal cruda en vivo: más nerviosa y cambia en ambos lados a la vez.</p>
              </div>
            </Widget>
          </div>
        </div>
      )}
    </PageShell>
  )
}
