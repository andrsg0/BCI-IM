import { Fragment, useEffect, useMemo, useState } from 'react'
import { ChevronDown, ChevronRight } from 'lucide-react'
import {
  ScatterChart, Scatter, LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, Legend, ReferenceLine,
} from 'recharts'
import { PageShell } from '../components/PageShell'
import type { HelpContent } from '../components/HelpButton'
import { Widget } from '../components/Widget'
import { GlossaryText } from '../components/GlossaryText'
import { Topomap, type Pos2D } from '../components/Topomap'
import { EEGNetModel } from '../components/EEGNetModel'
import { getJSON } from '../api/client'

interface CSPResp {
  channels: string[]; eigenvalues: number[]; patterns: number[][]
  filters: number[][]   // W (n_componentes, n_canales): Z = W·X
  classes: string[]; features: number[][]; labels: string[]; pos2d: Pos2D
}

interface DatasetInfo { id: string; label: string; subjects: number; fs: number; sessions?: number; live?: boolean }

interface CspSignal {
  fs: number; component: number; n_components: number; favored_class: string
  eigenvalue: number; channel: string; trial: number
  t: number[]; raw: number[]; csp: number[]
}

interface TrainConfig {
  dataset: {
    id: string; label: string; fs: number; sessions: number | null; live: boolean
    n_subjects: number | null; subject: number; classes: string[]
    n_channels: number | null; channels: string[] | null; n_trials: number | null
  }
  preprocessing: {
    epoching: { tmin: number | null; tmax: number | null }
    classification_window: { tmin_rel: number | null; tmax_rel: number | null; abs_s: number[] | null; len_s: number | null }
    fir: { low_hz: number | null; high_hz: number | null; num_taps: number | null; window: string | null; group_delay_samples: number | null; group_delay_ms: number | null }
    csp: { n_components: number | null; log_variance: boolean | null; shrinkage: number | null }
  }
  validation: {
    classifier: string | null; cv_folds: number | null
    holdout_kind: string | null; holdout_desc: string | null
    n_train: number | null; n_demo: number | null
    accuracy_intersession: number | null; kappa: number | null; trained_on: string | null
  }
  has_model: boolean
}

interface LdaResp {
  classes: string[]; positive_class: string
  weights: number[]; bias: number; n_features: number
  boundary2d: { comp_x: number; comp_y: number; w: number[]; b: number }
  confusion: { labels: string[]; matrix: number[][] }
  accuracy: number; kappa: number; n_eval: number
  holdout_kind: string; cv_folds: number | null; has_model: boolean
}

const CLASS_COLORS = ['#2563eb', '#e11d48', '#059669', '#d97706']

// Ayuda única para toda la pestaña CSP + LDA (las tres etapas, en orden).
const HELP_PIPELINE: HelpContent = {
  pipeline: 'El modelo clásico · CSP → log-varianza → LDA',
  intro: '¿Cómo se entrena el modelo clásico? La señal, ya filtrada en la banda µ/β, recorre tres etapas lineales encadenadas que ves aquí en orden, con datos reales del sujeto: primero el filtro espacial (CSP), luego la extracción de características (log-varianza) y por último el clasificador (LDA).',
  points: [
    { label: 'Filtro espacial (CSP)', desc: 'Combina los electrodos para que la diferencia de energía entre imaginar una mano y la otra se note al máximo. Cada topomapa muestra el patrón de un componente: los colores fuertes (típicamente sobre C3/C4, la corteza motora) marcan los electrodos que más pesan; el blanco, los que aporta poco. El número λ sale del problema de autovalores generalizados e indica a qué mano responde el componente (λ≈1 una, λ≈0 la otra).' },
    { label: 'El color es un peso, NO la energía', desc: 'Cuidado con leer "azul = poca energía aquí": el color es el PESO del electrodo en el filtro, y su signo es arbitrario (matemáticamente el mapa entero podría aparecer con los colores invertidos sin cambiar nada). La caída de potencia del hemisferio contrario (el ERD que esperas ver al mover una mano) NO está en estos colores, sino en la log-varianza del componente (el paso siguiente). Por eso no busques aquí la regla "lado contrario a la mano".' },
    { label: '¿Por qué los mapas se ven ruidosos?', desc: 'Son patrones de UN solo sujeto, con pocos trials y regularización: rara vez salen tan limpios como en los libros. Aun así suele haber cierta lateralización —los componentes de una mano se apoyan algo más en los electrodos de su mismo lado—, pero no esperes un mapa de manual.' },
    { label: 'Características (log-varianza)', desc: 'Un clasificador no entiende ondas, solo números. Por eso cada componente del CSP se resume en su log-varianza (su energía). Así cada intento se vuelve un punto: si las nubes de cada mano se separan, las clases son distinguibles; si se mezclan, habrá errores.' },
    { label: 'Clasificación (LDA)', desc: 'Traza una frontera de decisión recta (un hiperplano) que parte el espacio en dos regiones, una por mano. En vivo, la decisión es simplemente de qué lado cae el punto.' },
    { label: 'Validación honesta', desc: 'El modelo se evalúa con intentos que nunca vio durante el entrenamiento (partición held-out), para no engañarnos. Las cifras de precisión (accuracy, κ) y la matriz de confusión se muestran en la sección Resultados.' },
  ],
  terms: ['CSP', 'Filtro espacial', 'Problema de autovalores generalizados', 'Varianza y log-varianza', 'LDA', 'Frontera de decisión / hiperplano', 'Held-out (partición reservada)', 'Accuracy, Matriz de confusión y Kappa de Cohen'],
}

const HELP_EEGNET: HelpContent = {
  pipeline: 'Comparación · teoría (CSP) vs datos (EEGNet)',
  intro: 'A diferencia del enfoque clásico por pasos, EEGNet es una red neuronal convolucional compacta, de extremo a extremo (end-to-end): recibe la señal temporal y sus capas aprenden solas los filtros temporales y espaciales, sin calcular la varianza a mano. ¿Cómo interpretar esos filtros aprendidos? Lo que ves aquí es el interior de la caja negra: estos mapas topográficos no se calcularon con la fórmula del CSP, son los pesos que la red auto-aprendió por retropropagación (backpropagation) durante el entrenamiento.',
  points: [
    { label: 'El Efecto Espejo', desc: 'Nota cómo la red neuronal, por sí sola y tras procesar miles de datos, llegó a una conclusión muy similar a la del CSP: descubrió de forma autónoma que para identificar la imaginación motora tiene que prestarle atención casi exclusiva a la corteza motora (puntos de color intenso sobre C3 y C4).' },
    { label: '¿Por qué no hay sección de Varianza o LDA?', desc: 'Porque EEGNet integra todo en su estructura. Sus capas convolucionales actúan como los filtros (CSP), sus funciones de activación y capas de agrupación (pooling de la red) extraen la información de la energía (log-varianza), y su última capa densa (Softmax) actúa como el clasificador final (LDA).' },
    { label: 'Filtro temporal', desc: 'Tu FIR es un pasa-banda limpio en µ/β (8–30 Hz, zona verde). Cada curva aprendida por EEGNet es la respuesta en frecuencia de uno de sus filtros temporales. Si sus picos caen en la zona verde, la red redescubrió la banda que tú impusiste por teoría.' },
    { label: 'Ficha de entrenamiento', desc: 'Bajo los gráficos se resume cómo se entrenó la red (trials, banda, épocas) y su accuracy honesta frente al CSP+LDA del mismo sujeto. Con pocos trials, el método clásico suele igualar o superar al deep learning.' },
  ],
  terms: ['CSP+LDA vs EEGNet (teoría vs datos)', 'EEGNet', 'Respuesta en frecuencia'],
}

// --- Ficha de configuración (dataset + preprocesamiento + validación) -------
/** Fila etiqueta→valor dentro de un bloque de la ficha de configuración. */
function Spec({ label, value, hint }: { label: string; value: React.ReactNode; hint?: string }) {
  return (
    <div className="flex items-baseline justify-between gap-3 border-b border-slate-100 py-1 last:border-0">
      <dt className="text-xs text-slate-500" title={hint}>{label}</dt>
      <dd className="text-right text-xs font-medium tabular-nums text-slate-700">{value ?? '—'}</dd>
    </div>
  )
}

/** Bloque de la ficha (Dataset / Preprocesamiento / Validación). */
function SpecBlock({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div className="rounded-lg border border-slate-200 bg-slate-50/50 p-3">
      <div className="mb-1.5 text-[11px] font-semibold uppercase tracking-wide text-slate-400">{title}</div>
      <dl>{children}</dl>
    </div>
  )
}

/** Ficha "Datos del dataset y preprocesamiento": dataset y preprocesamiento.
 *  La validación/métricas vive ahora en la subsección LDA (no se duplica aquí). */
function ConfigFicha({ cfg }: { cfg: TrainConfig }) {
  const { dataset: d, preprocessing: p } = cfg
  const fir = p.fir
  const cw = p.classification_window
  return (
    <div>
      <div className="grid gap-3 sm:grid-cols-2">
        <SpecBlock title="Dataset">
          <Spec label="Nombre" value={d.label} />
          <Spec label="Sujeto" value={`${d.subject}${d.n_subjects ? ` de ${d.n_subjects}` : ''}`} />
          <Spec label="Clases" value={d.classes.join(' vs ')} />
          <Spec label="Canales EEG" value={d.n_channels ?? '—'} />
          <Spec label="Frec. muestreo" value={`${d.fs} Hz`} />
          <Spec label="Trials (totales)" value={d.n_trials ?? '—'} hint="entrenamiento + held-out de demo" />
        </SpecBlock>

        <SpecBlock title="Preprocesamiento">
          <Spec label="FIR pasa-banda" value={fir.low_hz != null ? `${fir.low_hz}–${fir.high_hz} Hz` : '—'} hint="banda µ/β" />
          <Spec label="Coef. FIR (taps)" value={fir.num_taps ?? '—'} hint="orden del filtro; impar ⇒ fase lineal" />
          <Spec label="Ventana FIR" value={fir.window ?? '—'} />
          <Spec label="Retardo de grupo" value={fir.group_delay_ms != null ? `${fir.group_delay_ms.toFixed(0)} ms (${fir.group_delay_samples} m)` : '—'} hint="(N−1)/2 muestras; compensado offline" />
          <Spec label="Ventana clasif." value={cw.abs_s ? `${cw.abs_s[0]}–${cw.abs_s[1]} s` : '—'} hint="sub-ventana activa del trial tras el cue" />
          <Spec label="Comp. CSP" value={p.csp.n_components ?? '—'} hint="pares extremos del espectro de autovalores" />
          <Spec label="Shrinkage CSP" value={p.csp.shrinkage != null ? p.csp.shrinkage : '—'} hint="regularización de covarianza [0,1]" />
        </SpecBlock>
      </div>
      {!cfg.has_model && (
        <p className="mt-2 text-[11px] text-amber-600">
          No hay modelo CSP+LDA entrenado en disco para este sujeto: se muestran solo los parámetros de
          configuración (sin métricas ni partición). Entrénalo con <code>scripts/train_model.py</code>.
        </p>
      )}
    </div>
  )
}

/** Panel plegable para texto/tablas auxiliares debajo de la introducción. */
function Collapsible({ title, defaultOpen = false, children }: { title: string; defaultOpen?: boolean; children: React.ReactNode }) {
  const [open, setOpen] = useState(defaultOpen)
  return (
    <div className="rounded-xl border border-slate-200 bg-white shadow-sm">
      <button
        onClick={() => setOpen((o) => !o)}
        className="flex w-full items-center gap-2 px-4 py-2.5 text-left text-sm font-medium text-slate-600 hover:bg-slate-50"
      >
        {open ? <ChevronDown size={16} /> : <ChevronRight size={16} />}
        {title}
      </button>
      {open && <div className="border-t border-slate-100 p-4">{children}</div>}
    </div>
  )
}

// --- Selector de página (dataset + sujeto), independiente del panel lateral --
function DataSelector({ datasets, dataset, subject, onDataset, onSubject }: {
  datasets: DatasetInfo[]; dataset: string; subject: number
  onDataset: (id: string) => void; onSubject: (s: number) => void
}) {
  const cur = datasets.find((d) => d.id === dataset)
  const nSubjects = cur?.subjects ?? 9
  return (
    <div className="flex flex-wrap items-center gap-3 rounded-xl border border-slate-200 bg-white px-4 py-2.5 shadow-sm">
      <label className="text-sm text-slate-500">Dataset</label>
      <select
        value={dataset}
        onChange={(e) => onDataset(e.target.value)}
        className="rounded-md border border-slate-200 px-2 py-1 text-sm font-medium text-slate-700"
      >
        {datasets.map((d) => <option key={d.id} value={d.id}>{d.label}</option>)}
      </select>
      <label className="text-sm text-slate-500">Sujeto</label>
      <select
        value={subject}
        onChange={(e) => onSubject(Number(e.target.value))}
        className="rounded-md border border-slate-200 px-2 py-1 text-sm text-slate-700"
      >
        {Array.from({ length: nSubjects }, (_, i) => i + 1).map((s) => (
          <option key={s} value={s}>{s}</option>
        ))}
      </select>
    </div>
  )
}

// --- Ecuación interactiva Z = W·X -------------------------------------------
/** Color divergente azul(−)–blanco(0)–rojo(+) normalizado por |máx|. */
function divColor(v: number, maxAbs: number) {
  const t = maxAbs ? Math.max(-1, Math.min(1, v / maxAbs)) : 0
  if (t >= 0) { const k = Math.round(255 * (1 - t)); return `rgb(255,${k},${k})` }
  const k = Math.round(255 * (1 + t)); return `rgb(${k},${k},255)`
}

type EqSymId = 'Z' | 'W' | 'X'

/** Letra clicable de la ecuación Z = W·X (resaltada si está seleccionada). */
function EqSym({ id, sel, onSelect, children }: { id: EqSymId; sel: EqSymId; onSelect: (id: EqSymId) => void; children: React.ReactNode }) {
  return (
    <button
      onClick={() => onSelect(id)}
      className={`rounded-md px-2 py-0.5 font-mono text-2xl transition ${sel === id ? 'bg-primary text-white' : 'text-slate-700 hover:bg-slate-100'}`}
    >
      {children}
    </button>
  )
}

function CspEquation({ csp }: { csp: CSPResp }) {
  const [sel, setSel] = useState<EqSymId>('W')
  const maxAbs = useMemo(
    () => Math.max(1e-9, ...csp.filters.flat().map((x) => Math.abs(x))),
    [csp.filters],
  )
  return (
    <div className="flex flex-col">
      <div className="mb-2 flex items-center justify-center gap-1">
        <EqSym id="Z" sel={sel} onSelect={setSel}>Z</EqSym>
        <span className="font-mono text-2xl text-slate-400">=</span>
        <EqSym id="W" sel={sel} onSelect={setSel}>W</EqSym>
        <span className="font-mono text-2xl text-slate-400">·</span>
        <EqSym id="X" sel={sel} onSelect={setSel}>X</EqSym>
      </div>
      <p className="mb-2 text-center text-[11px] text-slate-400">Pulsa una letra para ver qué representa.</p>

      <div className="flex-1">
        {sel === 'Z' && (
          <div className="text-sm leading-relaxed text-slate-600">
            <strong>Z — señales virtuales (componentes CSP).</strong> La salida del filtro espacial:
            {' '}{csp.filters.length} «canales virtuales», cada uno una combinación lineal de los{' '}
            {csp.channels.length} electrodos. Su <span className="font-mono">log-varianza</span> es lo que
            alimenta al clasificador LDA. A cada componente le corresponde un mapa de arriba (su patrón).
          </div>
        )}
        {sel === 'X' && (
          <div className="text-sm leading-relaxed text-slate-600">
            <strong>X — señal de entrada.</strong> El EEG ya filtrado en la banda µ/β:
            una matriz de <span className="font-mono">{csp.channels.length} canales × tiempo</span>.
            Cada fila es un electrodo; cada columna, un instante. El gráfico «Cruda vs filtrada» muestra
            una de esas filas (un electrodo) frente a su componente Z.
          </div>
        )}
        {sel === 'W' && (
          <div>
            <p className="mb-2 text-xs leading-relaxed text-slate-500">
              <strong>W — matriz de filtros espaciales</strong> ({csp.filters.length} componentes ×{' '}
              {csp.channels.length} canales). Cada <strong>fila</strong> es un filtro: los pesos con los que
              se combinan los electrodos. Así se obtiene cada componente, instante a instante, como una suma
              ponderada: <span className="font-mono">Z<sub>i</sub>[n] = Σ<sub>j</sub> W<sub>i,j</sub>·X<sub>j</sub>[n]</span>{' '}
              (el color es el peso: rojo +, azul −).
            </p>
            <p className="mb-2 text-[11px] leading-relaxed text-slate-400">
              Estos números <strong>no</strong> son los de los mapas topográficos. Los mapas muestran los
              <em> patrones</em> (la pseudo-inversa de W); W son los pesos que de verdad multiplican la señal.
              Que difieran es normal.
            </p>
            <div className="overflow-x-auto">
              <table className="border-separate" style={{ borderSpacing: 2 }}>
                <thead>
                  <tr>
                    <th className="sticky left-0 bg-white px-1 text-left text-[10px] text-slate-400">comp</th>
                    {csp.channels.map((ch) => (
                      <th key={ch} className="px-0.5 text-[9px] font-normal text-slate-400">{ch}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {csp.filters.map((row, ci) => (
                    <tr key={ci}>
                      <td className="sticky left-0 bg-white pr-1 text-[10px] font-medium text-slate-500">{ci}</td>
                      {row.map((v, j) => (
                        <td
                          key={j}
                          title={`${csp.channels[j]}: ${v.toFixed(3)}`}
                          className="px-1 py-0.5 text-center font-mono text-[9px] text-slate-700"
                          style={{ background: divColor(v, maxAbs) }}
                        >
                          {v.toFixed(2)}
                        </td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}

// --- Fórmula de la log-varianza --------------------------------------------
function LogVarFormula() {
  return (
    <div className="flex h-full flex-col items-center justify-center gap-4 text-slate-700">
      <div className="flex items-center gap-2 text-lg">
        <span className="font-mono italic">f<sub>i</sub></span>
        <span className="text-slate-400">=</span>
        <span className="font-mono">ln</span>
        <span className="text-3xl font-light text-slate-400">(</span>
        <span className="inline-flex flex-col items-center leading-tight">
          <span className="px-2 py-0.5 font-mono text-sm">Var(Z<sub>i</sub>)</span>
          <span className="my-0.5 h-px w-full bg-slate-500" />
          <span className="px-2 py-0.5 font-mono text-sm">Σ<sub>j</sub> Var(Z<sub>j</sub>)</span>
        </span>
        <span className="text-3xl font-light text-slate-400">)</span>
      </div>
      <p className="max-w-md text-center text-xs leading-relaxed text-slate-500">
        Cada componente <span className="font-mono">Z<sub>i</sub></span> (una «señal virtual» del CSP) se
        resume en un solo número: el <strong>logaritmo</strong> de su <strong>varianza</strong> (energía)
        relativa a la energía total de todos los componentes. El cociente normaliza, y el logaritmo iguala
        las escalas y hace la distribución más gaussiana —ideal para el clasificador lineal (LDA)—. Estos{' '}
        <span className="font-mono">f<sub>i</sub></span> son las coordenadas de cada punto del gráfico de la derecha.
      </p>
    </div>
  )
}

// --- Señal cruda vs componente CSP -----------------------------------------
function SignalCompare({ dataset, subject, nComp }: { dataset: string; subject: number; nComp: number }) {
  const [comp, setComp] = useState(0)
  const [sig, setSig] = useState<CspSignal | null>(null)

  useEffect(() => { setComp(0) }, [dataset, subject])
  useEffect(() => {
    setSig(null)
    getJSON<CspSignal>(`/csp_signal?dataset=${dataset}&subject=${subject}&component=${comp}`)
      .then(setSig).catch(() => setSig(null))
  }, [dataset, subject, comp])

  const data = useMemo(() => {
    if (!sig) return []
    return sig.t.map((t, i) => ({ t: t.toFixed(2), raw: sig.raw[i], csp: sig.csp[i] }))
  }, [sig])

  return (
    <div className="flex h-full flex-col">
      <div className="mb-1 flex items-center justify-between gap-2">
        <p className="text-[11px] leading-snug text-slate-500">
          {sig
            ? <>Electrodo crudo <strong>{sig.channel}</strong> (el de mayor peso en este componente) frente a la salida del <strong>comp {sig.component}</strong>. Normalizadas para comparar forma, no amplitud.</>
            : 'Cargando señal…'}
        </p>
        <select
          value={comp}
          onChange={(e) => setComp(Number(e.target.value))}
          className="shrink-0 rounded-md border border-slate-200 px-1.5 py-0.5 text-xs text-slate-600"
          title="Componente CSP"
        >
          {Array.from({ length: nComp }, (_, i) => i).map((i) => <option key={i} value={i}>comp {i}</option>)}
        </select>
      </div>
      <div className="min-h-0 flex-1">
        <ResponsiveContainer width="100%" height="100%" minHeight={140}>
          <LineChart data={data} margin={{ top: 6, right: 12, bottom: 18, left: 0 }}>
            <CartesianGrid stroke="#eef2f7" />
            <XAxis dataKey="t" tick={{ fontSize: 10 }} interval="preserveStartEnd" minTickGap={40}
              label={{ value: 'tiempo (s)', position: 'bottom', fontSize: 10, fill: '#94a3b8' }} />
            <YAxis tick={{ fontSize: 10 }} width={28} />
            <Tooltip contentStyle={{ fontSize: 11 }} />
            <Legend verticalAlign="top" wrapperStyle={{ fontSize: 11 }} />
            <Line name="Cruda (1 canal)" dataKey="raw" stroke="#94a3b8" dot={false} strokeWidth={1.2} isAnimationActive={false} />
            <Line name="Filtrada (CSP)" dataKey="csp" stroke="#7c3aed" dot={false} strokeWidth={1.8} isAnimationActive={false} />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  )
}

// --- Clasificación lineal (LDA) --------------------------------------------
/** Parte un rectángulo (en coords de datos) por la recta w0·x+w1·y+b=0.
 *  Devuelve los dos sub-polígonos (lado positivo/negativo) y los puntos de corte. */
function splitRect(xMin: number, xMax: number, yMin: number, yMax: number, w0: number, w1: number, b: number) {
  const corners = [{ x: xMin, y: yMin }, { x: xMax, y: yMin }, { x: xMax, y: yMax }, { x: xMin, y: yMax }]
  const f = (p: { x: number; y: number }) => w0 * p.x + w1 * p.y + b
  const pos: { x: number; y: number }[] = []
  const neg: { x: number; y: number }[] = []
  const cut: { x: number; y: number }[] = []
  for (let i = 0; i < 4; i++) {
    const cur = corners[i], nxt = corners[(i + 1) % 4]
    const fc = f(cur), fn = f(nxt)
    if (fc >= 0) pos.push(cur); else neg.push(cur)
    if ((fc >= 0) !== (fn >= 0)) {
      const t = fc / (fc - fn)
      const ip = { x: cur.x + t * (nxt.x - cur.x), y: cur.y + t * (nxt.y - cur.y) }
      pos.push(ip); neg.push(ip); cut.push(ip)
    }
  }
  return { pos, neg, cut }
}

/** Scatter de separabilidad con la frontera de decisión del LDA dibujada encima.
 *  La recta se dibuja con <ReferenceLine segment> (recharts 3 ya no expone las
 *  escalas a <Customized>, así que calculamos los extremos en coords de datos). */
function DecisionScatter({ csp, lda }: { csp: CSPResp; lda: LdaResp }) {
  const last = csp.features[0].length - 1
  const pts = csp.features.map((f, i) => ({ x: f[0], y: f[last], cls: csp.labels[i] }))
  const xsv = pts.map((q) => q.x), ysv = pts.map((q) => q.y)
  const pad = (a: number[], g: number) => {
    const lo = Math.min(...a), hi = Math.max(...a), m = (hi - lo) * g || 1
    return [lo - m, hi + m] as const
  }
  const [xMin, xMax] = pad(xsv, 0.08)
  const [yMin, yMax] = pad(ysv, 0.08)
  const byClass = csp.classes.map((cls) => pts.filter((q) => q.cls === cls))
  const [w0, w1] = lda.boundary2d.w
  const b = lda.boundary2d.b
  // Extremos de la frontera (intersección de la recta con el rectángulo visible).
  const cut = splitRect(xMin, xMax, yMin, yMax, w0, w1, b).cut
  // El lado positivo (f>0) es positive_class; le toca su color de clase.
  const posIdx = csp.classes.indexOf(lda.positive_class)
  const posColor = CLASS_COLORS[(posIdx >= 0 ? posIdx : 0) % CLASS_COLORS.length]
  const negColor = CLASS_COLORS[(posIdx === 0 ? 1 : 0) % CLASS_COLORS.length]

  return (
    <div className="flex h-full flex-col">
      <p className="mb-1 text-[11px] leading-snug text-slate-500">
        La <strong>línea discontinua</strong> es la frontera del LDA: a un lado quedan los{' '}
        <span style={{ color: posColor }}>{lda.positive_class}</span>, al otro los{' '}
        <span style={{ color: negColor }}>{csp.classes.find((c) => c !== lda.positive_class)}</span>.
        En vivo, la decisión es de qué lado cae el punto.
      </p>
      <div className="min-h-0 flex-1">
        <ResponsiveContainer width="100%" height="100%" minHeight={200}>
          <ScatterChart margin={{ top: 4, right: 12, bottom: 26, left: 4 }}>
            <CartesianGrid stroke="#eef2f7" />
            <XAxis type="number" dataKey="x" domain={[xMin, xMax]} tick={{ fontSize: 11 }} allowDataOverflow
              tickFormatter={(v: number) => v.toFixed(1)}
              label={{ value: `log-var comp 0  →  ${csp.classes[0]}`, position: 'bottom', fontSize: 10, fill: '#94a3b8' }} />
            <YAxis type="number" dataKey="y" domain={[yMin, yMax]} tick={{ fontSize: 11 }} allowDataOverflow width={36}
              tickFormatter={(v: number) => v.toFixed(1)}
              label={{ value: `log-var comp ${last} → ${csp.classes[1]}`, angle: -90, position: 'insideLeft', fontSize: 10, fill: '#94a3b8' }} />
            {cut.length === 2 && (
              <ReferenceLine
                segment={[{ x: cut[0].x, y: cut[0].y }, { x: cut[1].x, y: cut[1].y }]}
                stroke="#0f172a" strokeWidth={2} strokeDasharray="6 3" ifOverflow="visible"
              />
            )}
            <Tooltip cursor={{ strokeDasharray: '3 3' }} />
            <Legend verticalAlign="top" wrapperStyle={{ fontSize: 11 }} />
            {byClass.map((data, i) => (
              <Scatter key={i} name={csp.classes[i]} data={data} fill={CLASS_COLORS[i % CLASS_COLORS.length]} fillOpacity={0.6} />
            ))}
          </ScatterChart>
        </ResponsiveContainer>
      </div>
    </div>
  )
}

/** Fórmula del clasificador  y = w·F + b  con su regla de decisión. */
function LdaFormula({ lda, classes }: { lda: LdaResp; classes: string[] }) {
  const neg = classes.find((c) => c !== lda.positive_class) ?? classes[1]
  return (
    <div className="flex h-full flex-col items-center justify-center gap-4 text-slate-700">
      <div className="flex items-center gap-2 text-2xl">
        <span className="font-mono italic">y</span>
        <span className="text-slate-400">=</span>
        <span className="font-mono italic">w</span>
        <span className="text-slate-400">·</span>
        <span className="font-mono italic">F</span>
        <span className="text-slate-400">+</span>
        <span className="font-mono italic">b</span>
      </div>
      <div className="flex gap-2 text-sm">
        <span className="rounded-md px-3 py-1" style={{ background: 'rgba(37,99,235,0.12)' }}>
          si <span className="font-mono">y &gt; 0</span> → <strong style={{ color: CLASS_COLORS[0] }}>{lda.positive_class}</strong>
        </span>
        <span className="rounded-md px-3 py-1" style={{ background: 'rgba(225,29,72,0.12)' }}>
          si <span className="font-mono">y ≤ 0</span> → <strong style={{ color: CLASS_COLORS[1] }}>{neg}</strong>
        </span>
      </div>
      <p className="max-w-md text-center text-xs leading-relaxed text-slate-500">
        <span className="font-mono">F</span> es el vector de {lda.n_features} características (log-varianzas del paso anterior),
        {' '}<span className="font-mono">w</span> los pesos que aprende el LDA y <span className="font-mono">b</span> el sesgo
        (= {lda.bias.toFixed(2)}). La frontera es justo donde <span className="font-mono">y = 0</span>: un hiperplano que
        parte el espacio en las dos regiones de comando.
      </p>
    </div>
  )
}

// --- Diagrama del recorrido de la señal (espejo del de EEGNet) -------------
/** Esquema cronológico de las 4 etapas lineales del modelo clásico. */
function PipelineDiagram({ nCh, nComp, nClasses }: { nCh: number; nComp: number; nClasses: number }) {
  const stages = [
    { name: 'Entrada', sub: `${nCh} canales × tiempo`, eq: 'EEG filtrado en µ/β', color: 'var(--accent-signal)' },
    { name: 'Filtro espacial (CSP)', sub: `${nComp} componentes`, eq: 'combina los electrodos', color: 'var(--accent-csp)' },
    { name: 'Log-varianza', sub: `${nComp} números`, eq: 'energía por componente', color: 'var(--accent-metric)' },
    { name: 'Clasificación (LDA)', sub: `${nClasses} clases`, eq: 'frontera de decisión', color: 'var(--accent-metric)' },
  ]
  return (
    <div className="flex items-stretch gap-1.5 overflow-x-auto pb-1">
      {stages.map((s, i) => (
        <Fragment key={s.name}>
          <div
            className="flex min-w-[8rem] flex-1 flex-col rounded-lg border border-slate-200 bg-white p-2.5 shadow-sm"
            style={{ borderTop: `3px solid ${s.color}` }}
          >
            <div className="text-xs font-semibold text-slate-700">{s.name}</div>
            <div className="mt-0.5 font-mono text-[10px] leading-tight text-slate-500">{s.sub}</div>
            <div className="mt-auto pt-2 text-[10px] italic leading-tight text-slate-400">{s.eq}</div>
          </div>
          {i < stages.length - 1 && <div className="flex shrink-0 items-center text-lg text-slate-300">→</div>}
        </Fragment>
      ))}
    </div>
  )
}

// --- Scatter de separabilidad (sin frontera; la frontera va en la etapa LDA) -
function SeparabilityScatter({ csp }: { csp: CSPResp }) {
  const last = csp.features[0].length - 1
  const byClass = csp.classes.map((cls) =>
    csp.features.filter((_, i) => csp.labels[i] === cls).map((f) => ({ x: f[0], y: f[last] })),
  )
  return (
    <div className="flex h-full flex-col">
      <p className="mb-1 text-[11px] leading-snug text-slate-500">
        Cada punto es un <strong>trial</strong>. Usamos los dos componentes más discriminativos (los extremos del
        espectro): eje X = energía del <strong>comp 0</strong> (sube con{' '}
        <span style={{ color: CLASS_COLORS[0] }}>{csp.classes[0]}</span>); eje Y = <strong>comp {last}</strong> (sube con{' '}
        <span style={{ color: CLASS_COLORS[1] }}>{csp.classes[1]}</span>). Si las dos nubes se <strong>separan</strong>, las clases son distinguibles.
      </p>
      <div className="min-h-0 flex-1">
        <ResponsiveContainer width="100%" height="100%" minHeight={200}>
          <ScatterChart margin={{ top: 4, right: 12, bottom: 26, left: 4 }}>
            <CartesianGrid stroke="#eef2f7" />
            <XAxis type="number" dataKey="x" name="log-var comp 0" tick={{ fontSize: 11 }}
              label={{ value: `log-var comp 0  →  ${csp.classes[0]}`, position: 'bottom', fontSize: 10, fill: '#94a3b8' }} />
            <YAxis type="number" dataKey="y" name={`log-var comp ${last}`} tick={{ fontSize: 11 }}
              label={{ value: `log-var comp ${last} → ${csp.classes[1]}`, angle: -90, position: 'insideLeft', fontSize: 10, fill: '#94a3b8' }} />
            <Tooltip cursor={{ strokeDasharray: '3 3' }} />
            <Legend verticalAlign="top" wrapperStyle={{ fontSize: 11 }} />
            {byClass.map((pts, i) => (
              <Scatter key={i} name={csp.classes[i]} data={pts} fill={CLASS_COLORS[i % CLASS_COLORS.length]} fillOpacity={0.6} />
            ))}
          </ScatterChart>
        </ResponsiveContainer>
      </div>
    </div>
  )
}

// --- Flujo cronológico del modelo clásico (CSP → log-varianza → LDA) --------
/** Reemplaza los antiguos 3 GridBoards: un único recorrido en cajas, en orden,
 *  con explicaciones concisas (los términos se auto-enlazan al glosario). */
function CspLdaPipeline({ dataset, subject, csp, lda }: {
  dataset: string; subject: number; csp: CSPResp; lda: LdaResp | null
}) {
  // qué clase resalta cada componente: λ alto (≥0.5) → primera clase; λ bajo → segunda.
  const favoredClass = (ci: number) => (csp.eigenvalues[ci] >= 0.5 ? csp.classes[0] : csp.classes[1])

  return (
    <div className="space-y-4">
      {/* 0 · Recorrido de la señal (ancla la cronología) */}
      <Widget title="El recorrido de la señal" accent="brain">
        <PipelineDiagram nCh={csp.channels.length} nComp={csp.filters.length} nClasses={csp.classes.length} />
        <p className="mt-2 text-xs leading-relaxed text-slate-500">
          <GlossaryText>Cada intento de imaginación motora recorre cuatro etapas, todas operaciones lineales. La señal ya llega filtrada en la banda µ/β; desde ahí el CSP combina los electrodos, la log-varianza resume cada componente en un número y el LDA decide. Abajo se ve cada etapa en orden, con datos reales del sujeto.</GlossaryText>
        </p>
        <p className="mt-2 text-[11px] leading-relaxed text-slate-400">
          Para no confundirse: el <strong>mapa topográfico es fijo</strong> (son los pesos del filtro, no un
          instante). Cada uno de los {csp.filters.length} componentes es una <strong>señal completa en el
          tiempo</strong>; su log-varianza la resume en <strong>un número por componente y por intento</strong>.
          Así cada intento se vuelve un punto ({csp.filters.length} coordenadas), y por eso el gráfico de
          separabilidad tiene muchos puntos: <strong>uno por intento</strong>.
        </p>
      </Widget>

      {/* 1 · Filtro espacial (CSP): mapas + ecuación */}
      <Widget title="CSP (Filtro espacial)" accent="csp">
        <p className="mb-3 text-xs leading-relaxed text-slate-500">
          <GlossaryText>El casco capta una mezcla de toda la actividad cerebral. El CSP funciona como un lente: combina los electrodos para que la diferencia entre imaginar la mano izquierda y la derecha resalte al máximo. Cada topomapa es uno de esos filtros espaciales y los colores fuertes marcan los electrodos que más pesan. El número λ sale del problema de autovalores generalizados e indica a qué mano responde cada componente (λ≈1 una, λ≈0 la otra).</GlossaryText>
        </p>
        <div className="grid gap-4 md:grid-cols-2">
          <div>
            <div className="mb-2 text-sm font-medium text-slate-600">Mapas topográficos</div>
            <div className="flex flex-wrap content-start justify-around gap-2">
              {csp.patterns.map((pat, ci) => (
                <div key={ci} className="text-center">
                  <Topomap channels={csp.channels} pos2d={csp.pos2d} values={pat} size={150} />
                  <div className="text-sm font-medium text-slate-700">comp {ci}</div>
                  <div className="text-xs text-slate-500">
                    favorece <span style={{ color: CLASS_COLORS[csp.eigenvalues[ci] >= 0.5 ? 0 : 1] }}>{favoredClass(ci)}</span>
                  </div>
                  <div className="font-mono text-[11px] text-slate-400">λ = {csp.eigenvalues[ci].toFixed(2)}</div>
                </div>
              ))}
            </div>
          </div>
          <div className="min-h-[320px]">
            <div className="mb-2 text-sm font-medium text-slate-600">Ecuación de proyección</div>
            <CspEquation csp={csp} />
          </div>
        </div>
      </Widget>

      {/* 2 · Señal cruda vs filtrada por CSP */}
      <Widget title="Señal: cruda vs filtrada por CSP" accent="csp">
        <p className="mb-2 text-xs leading-relaxed text-slate-500">
          <GlossaryText>Comparamos un electrodo tal cual (ruidoso) con la salida de un componente CSP: la misma actividad, pero con el ruido de los demás canales cancelado. Ambas se normalizan para comparar su forma, no su amplitud.</GlossaryText>
        </p>
        <div className="h-72">
          <SignalCompare dataset={dataset} subject={subject} nComp={csp.filters.length} />
        </div>
      </Widget>

      {/* 3 · Características (log-varianza): fórmula + scatter */}
      <Widget title="Log-var (Características)" accent="metric">
        <p className="mb-3 text-xs leading-relaxed text-slate-500">
          <GlossaryText>Un clasificador no entiende ondas, solo números. Por eso resumimos cada componente del CSP en su log-varianza (cuánta energía tiene). Así cada intento se vuelve un punto: si los puntos de cada mano se separan, el modelo podrá distinguirlas.</GlossaryText>
        </p>
        <div className="grid gap-4 md:grid-cols-2">
          <div className="min-h-[260px]"><LogVarFormula /></div>
          <div className="min-h-[260px]"><SeparabilityScatter csp={csp} /></div>
        </div>
      </Widget>

      {/* 4 · Clasificación (LDA): frontera + regla */}
      {lda ? (
        <Widget title="LDA (Clasificación)" accent="metric">
          <p className="mb-3 text-xs leading-relaxed text-slate-500">
            <GlossaryText>El LDA traza una frontera de decisión recta (un hiperplano) que parte el espacio en dos regiones, una por mano. En vivo, la decisión es simplemente de qué lado cae el punto. (Su precisión sobre datos no vistos se mide en Resultados.)</GlossaryText>
          </p>
          <div className="grid gap-4 md:grid-cols-2">
            <div className="min-h-[280px]"><DecisionScatter csp={csp} lda={lda} /></div>
            <div className="min-h-[280px]"><LdaFormula lda={lda} classes={csp.classes} /></div>
          </div>
        </Widget>
      ) : (
        <div className="flex h-32 items-center justify-center text-slate-300">Calculando frontera de decisión…</div>
      )}
    </div>
  )
}

// ---------------------------------------------------------------------------
export default function SpatialCSP() {
  const [datasets, setDatasets] = useState<DatasetInfo[]>([])
  const [dataset, setDataset] = useState<string>('BNCI2014_001')
  const [subject, setSubject] = useState<number>(1)
  const [csp, setCsp] = useState<CSPResp | null>(null)
  const [cfg, setCfg] = useState<TrainConfig | null>(null)
  const [lda, setLda] = useState<LdaResp | null>(null)
  const [tab, setTab] = useState<'pipeline' | 'eegnet'>('pipeline')

  // Lista de datasets para el selector de página (offline = todos, no solo demo en vivo).
  useEffect(() => {
    getJSON<DatasetInfo[]>('/datasets').then(setDatasets).catch(() => setDatasets([]))
  }, [])

  // Al cambiar de dataset, volver al sujeto 1 (los conteos de sujetos difieren).
  const onDataset = (id: string) => { setDataset(id); setSubject(1) }

  useEffect(() => {
    setCsp(null)
    getJSON<CSPResp>(`/csp?dataset=${dataset}&subject=${subject}`).then(setCsp).catch(() => setCsp(null))
  }, [dataset, subject])

  useEffect(() => {
    setCfg(null)
    getJSON<TrainConfig>(`/train_config?dataset=${dataset}&subject=${subject}`).then(setCfg).catch(() => setCfg(null))
  }, [dataset, subject])

  useEffect(() => {
    setLda(null)
    getJSON<LdaResp>(`/lda?dataset=${dataset}&subject=${subject}`).then(setLda).catch(() => setLda(null))
  }, [dataset, subject])

  // Dos "métodos" enfrentados: el modelo clásico (CSP+LDA) y el aprendido (EEGNet).
  const section = tab === 'eegnet'
    ? { name: 'EEGNet', help: HELP_EEGNET }
    : { name: 'CSP + LDA', help: HELP_PIPELINE }

  return (
    <PageShell
      title={`Entrenamiento · ${section.name}`}
      help={section.help}
      world="offline"
    >
      {/* Selector de datos de la página (independiente del panel lateral) */}
      <div className="mb-4">
        <DataSelector
          datasets={datasets}
          dataset={dataset}
          subject={subject}
          onDataset={onDataset}
          onSubject={setSubject}
        />
      </div>

      {/* Datos del dataset y preprocesamiento (plegable) */}
      <div className="mb-4">
        <Collapsible title="Información del dataset y preprocesamiento" defaultOpen>
          {cfg ? <ConfigFicha cfg={cfg} /> : <div className="text-sm text-slate-300">Cargando configuración…</div>}
        </Collapsible>
      </div>

      {/* Selector de método: modelo clásico (flujo único) vs EEGNet */}
      <div className="mb-4 flex rounded-md border border-slate-300 p-0.5 text-sm w-fit">
        {([['pipeline', 'CSP + LDA'], ['eegnet', 'EEGNet']] as const).map(([id, label]) => (
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
        <CspLdaPipeline dataset={dataset} subject={subject} csp={csp} lda={lda} />
      )}
    </PageShell>
  )
}
