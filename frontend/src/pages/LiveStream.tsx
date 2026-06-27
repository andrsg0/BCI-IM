import { useEffect, useMemo, useRef, useState } from 'react'
import type uPlot from 'uplot'
import { Radio, Check, X, Database, Waves, Grid3x3, Scale } from 'lucide-react'
import { PageShell } from '../components/PageShell'
import type { HelpContent } from '../components/HelpButton'
import { GridBoard, type GridWidget } from '../components/GridBoard'
import { FillChart } from '../components/charts/FillChart'
import { CSPSpaceLive, LDAAxisLive, type CSPHandle, type LDAHandle } from '../components/charts/PipelineStages'
import { HandPuppet, type HandSide } from '../components/HandPuppet'
import { useStore } from '../store/useStore'
import { openStream, getJSON } from '../api/client'
import { progressFromFrame, type ProgressFrame } from '../lib/progress'

interface Msg {
  trial: number; 'true': string; t: number; pred: string; probs: Record<string, number>
  feat?: number[] | null; disc?: number | null; filt?: number[]; alo?: number; ahi?: number
  p_act?: number | null   // P[ventana = imaginación activa] según el detector de reposo (B.4)
  error?: string
}

// Los 4 regímenes del selector (within/cross × CSP/EEGNet). El stream y la ficha
// se piden con el `method` elegido; los cross deben estar pre-entrenados.
const METHODS = [
  { id: 'csp_lda', label: 'CSP+LDA', regime: 'within', stages: true },
  { id: 'csp_lda_cross', label: 'CSP+LDA', regime: 'cross', stages: true },
  { id: 'eegnet', label: 'EEGNet', regime: 'within', stages: false },
  { id: 'eegnet_cross', label: 'EEGNet', regime: 'cross', stages: false },
] as const

interface CSPResp {
  classes: string[]; eigenvalues: number[]; features: number[][]
  labels: string[]; lda_disc: number[]
}

interface ModelCard {
  dataset: string; subject: number; classes: string[]; method: string
  holdout: { by: 'session' | 'index' | 'subject'; value?: string | number }
  train_session: string | null
  n_train: number; n_demo: number; accuracy: number; trained_on: string
  extra?: { n_train_subjects?: number } | null
}

const CLASS_COLORS = ['#2563eb', '#e11d48', '#059669', '#d97706']

/** Mapea la etiqueta de clase a qué mano mueve el muñeco. */
function handSideFromLabel(label: string | null | undefined, classes: string[]): HandSide {
  if (!label) return null
  const l = label.toLowerCase()
  if (/left|izq/.test(l)) return 'left'
  if (/right|der/.test(l)) return 'right'
  const i = classes.indexOf(label)   // fallback por orden de clase
  return i === 0 ? 'left' : i === 1 ? 'right' : null
}
const EMPTY: uPlot.AlignedData = [[], [], []]
const EMPTY2: uPlot.AlignedData = [[], []]

const HELP: HelpContent = {
  pipeline: 'Pipeline completo en tiempo real · Inferencia',
  intro: 'Reproduce el funcionamiento real de la interfaz: el casco transmite la señal de forma continua y el sistema la procesa y clasifica ventana a ventana, en tiempo real. Es la finalidad última del proyecto —lo que permitiría controlar un dispositivo con el pensamiento— y, a diferencia del análisis offline, exige que todo el procesamiento sea causal.',
  points: [
    { label: 'El recorrido de cada ventana', desc: 'La señal filtrada (FIR causal) que vimos en el paso anterior no «se clasifica sin más»: recorre dos etapas LINEALES bien distintas. Primero el CSP la espacializa (Z = W·X) y la resume en un vector de log-potencias —cada ventana pasa a ser un punto en el espacio de características—. Después el LDA proyecta ese vector sobre una recta y mira de qué lado de la frontera cae. Las tres vistas (señal filtrada → CSP → LDA) muestran ese recorrido en vivo.' },
    { label: 'Se entrena antes; en vivo solo se aplica', desc: 'Los filtros W del CSP y la frontera del LDA se aprendieron una sola vez, offline, con datos etiquetados (es lo que se ve fijo en la sección «Entrenamiento»). En vivo no se reaprende nada: el CSP es una multiplicación de matrices y el LDA un producto escalar, ambos instantáneos y causales. Por eso la nube de fondo está quieta y solo el punto de la ventana actual se mueve.' },
    { label: 'Decisión CONTINUA, sin conocer las fronteras del trial', desc: 'Como un casco real, la señal llega sin saber cuándo empieza ni termina la imaginación: el stream incluye ahora los segundos de reposo previos al cue. El sistema no «vota dentro del trial»; clasifica de forma continua, ventana a ventana, suavizando la probabilidad con una media móvil exponencial (EWMA) que no se reinicia nunca. Cuando esa confianza suavizada supera el UMBRAL, se compromete con una clase; si no, se abstiene. El contador de aciertos compara esos compromisos contra la etiqueta real solo para PUNTUAR (verdad de terreno), no para decidir; durante la imaginación la precisión se mantiene alta (la del modelo de la ficha).' },
    { label: 'El reposo: por qué hay FALSAS ALARMAS', desc: 'El clasificador es BINARIO: solo conoce «izquierda» y «derecha», no tiene una clase «reposo». ¿Cómo distinguir entonces que el sujeto no hace nada? La respuesta honesta: con este modelo, no del todo. Un LDA es SOBRECONFIADO —proyecta casi cualquier ventana, también las de reposo, lejos de la frontera— así que durante el reposo suele comprometerse igual: son las FALSAS ALARMAS que cuenta el panel. Subir el umbral exige más confianza y reduce esas falsas alarmas, pero nunca las elimina (y empieza a perder imaginaciones débiles). Es el clásico problema del «no-control state». Para mitigarlo está el «detector de reposo» (botón de arriba): un clasificador LINEAL extra, entrenado sobre la potencia de banda µ/β de las ventanas de reposo vs. activas, que actúa de COMPUERTA —solo deja decidir cuando cree que hay imaginación—. Actívalo y verás caer las falsas alarmas a la mitad; pero no es gratis ni perfecto (la separabilidad reposo/activo es modesta, AUC≈0.71): rechaza también algunos trials reales. Antes todo esto quedaba OCULTO porque solo se puntuaban las ventanas dentro de la franja activa conocida; al clasificar de forma continua, la demo deja ver este límite real de una BCI.' },
    { label: 'Techo del estado del arte', desc: 'Una BCI de imaginación motora no invasiva tiene un techo de precisión de ~70–85 % en 2 clases: la señal EEG es ruidosa y varía entre días y personas; no es posible acercarse al 100 %.' },
  ],
  terms: ['Causalidad', 'CSP', 'LDA', 'Softmax y voto mayoritario', 'Validación inter-sesión'],
}

export default function LiveStream() {
  const { dataset, subject, playing, clearToken } = useStore()
  const [last, setLast] = useState<Msg | null>(null)
  const [classes, setClasses] = useState<string[]>([])
  // contador POR TRIAL (no por ventana): así refleja la precisión real del modelo
  const [trialAcc, setTrialAcc] = useState({ correct: 0, decided: 0, skipped: 0 })
  // falsas alarmas: ventanas en las que el sistema SE COMPROMETIÓ durante el reposo
  // (verdad de terreno = fuera de la franja activa). Mide el coste de bajar el umbral.
  const [falseAlarms, setFalseAlarms] = useState(0)
  const [cur, setCur] = useState<{ trial: number; t: string; pred: string; conf: number; committed: boolean; active: boolean } | null>(null)
  const [recent, setRecent] = useState<{ trial: number; ok: boolean; decided: boolean }[]>([])
  const [threshold, setThreshold] = useState(0.65)
  // Compuerta de reposo (B.4): si está activa, solo se compromete cuando el detector
  // de reposo cree que la ventana es imaginación activa (P[activo] ≥ 0.5). Reduce las
  // falsas alarmas a costa de perder algunos trials. Por defecto OFF para que se vea el
  // contraste al activarla. `restGateAvail` = el servidor manda p_act (no en cross).
  const [restGate, setRestGate] = useState(false)
  const restGateRef = useRef(false)
  useEffect(() => { restGateRef.current = restGate }, [restGate])
  const restGateAvail = last?.p_act != null
  const [card, setCard] = useState<ModelCard | null>(null)
  const [csp, setCsp] = useState<CSPResp | null>(null)
  const [method, setMethod] = useState<string>('csp_lda')
  const [streamError, setStreamError] = useState<string | null>(null)
  const methodInfo = METHODS.find((m) => m.id === method)!
  const thresholdRef = useRef(0.65)
  useEffect(() => { thresholdRef.current = threshold }, [threshold])
  // Estado de la DECISIÓN CONTINUA (B.2): no se vota dentro de fronteras conocidas del
  // trial; se suaviza la probabilidad ventana a ventana con una media móvil exponencial
  // (EWMA) y se emite una clase solo cuando supera el umbral; si no, se ABSTIENE. El
  // EWMA NO se reinicia entre trials (la señal es continua, como un casco real).
  const ewma = useRef<Record<string, number>>({})
  // P[activo] suavizada (misma EWMA) para la compuerta de reposo (B.4).
  const pActEwma = useRef<number | null>(null)
  // acumulador del trial en curso, SOLO para puntuar (verdad de terreno): cuenta los
  // compromisos del clasificador continuo que caen en la franja activa real.
  const buf = useRef<{ trial: number | null; t: string; votes: Record<string, number>; n: number }>({ trial: null, t: '', votes: {}, n: 0 })

  // ficha del modelo YA ENTRENADO (mundo offline): con qué se entrenó y qué se reserva.
  // Depende del régimen elegido (cada método tiene su ficha).
  useEffect(() => {
    setCard(null); setStreamError(null)
    getJSON<ModelCard>(`/model?dataset=${dataset}&subject=${subject}&method=${method}`)
      .then(setCard)
      .catch(() => setCard(null))
  }, [dataset, subject, method])

  // nube de ENTRENAMIENTO (fija) sobre la que se dibuja el punto en vivo del CSP y el LDA
  const clsRef = useRef<string[]>([])
  useEffect(() => {
    setCsp(null); clsRef.current = []
    getJSON<CSPResp>(`/csp?dataset=${dataset}&subject=${subject}`)
      .then((d) => { setCsp(d); clsRef.current = d.classes })
      .catch(() => setCsp(null))
  }, [dataset, subject])

  const cspView = useMemo(() => {
    if (!csp || !csp.features.length) return null
    const cls = csp.classes
    const colors = cls.map((_, i) => CLASS_COLORS[i % CLASS_COLORS.length])
    const last = csp.features[0].length - 1
    const cspCloud = csp.features.map((f, i) => ({ x: f[0], y: f[last], c: Math.max(0, cls.indexOf(csp.labels[i])) }))
    const ldaCloud = csp.lda_disc.map((d, i) => ({ d, c: Math.max(0, cls.indexOf(csp.labels[i])) }))
    const eig = csp.eigenvalues
    const xLabel = `comp 0 · favorece ${cls[0]}  (λ=${eig[0].toFixed(2)})`
    const yLabel = `comp ${last} · favorece ${cls[1]}  (λ=${eig[last].toFixed(2)})`
    return { colors, cspCloud, ldaCloud, classes: cls, xLabel, yLabel }
  }, [csp])

  const hist = useRef({ ts: [] as number[], a: [] as number[], b: [] as number[] })
  const kRef = useRef(0)
  const chartU = useRef<uPlot | null>(null)
  // vistas imperativas de las etapas (no re-renderizan a 10 Hz; solo mueven el punto)
  const cspRef = useRef<CSPHandle>(null)
  const ldaRef = useRef<LDAHandle>(null)
  const filtU = useRef<uPlot | null>(null)
  const filtBuf = useRef<number[]>([])

  // limpiar
  useEffect(() => {
    hist.current = { ts: [], a: [], b: [] }; kRef.current = 0
    buf.current = { trial: null, t: '', votes: {}, n: 0 }
    ewma.current = {}; pActEwma.current = null
    filtBuf.current = []
    setLast(null); setCur(null); setRecent([]); setTrialAcc({ correct: 0, decided: 0, skipped: 0 }); setFalseAlarms(0)
    chartU.current?.setData(EMPTY)
    filtU.current?.setData(EMPTY2)
    cspRef.current?.reset(); ldaRef.current?.reset()
    useStore.getState().resetProgress()
  }, [clearToken, dataset, subject, method])

  // Al salir de la página, ocultar la barra de progreso del panel lateral.
  useEffect(() => () => useStore.getState().resetProgress(), [])

  // conexión al WebSocket: solo mientras "playing"
  useEffect(() => {
    if (!playing) return
    const { addLog, setLatency } = useStore.getState()
    addLog(`Conectado al stream en vivo (${dataset} · sujeto ${subject} · ${method}).`)
    const ws = openStream(`/stream?dataset=${dataset}&subject=${subject}&method=${method}`, (d) => {
      const m = d as Msg
      // El servidor manda {error} si el modelo (p. ej. un cross) no está entrenado.
      if (m.error) { setStreamError(m.error); addLog(`Stream: ${m.error}`); return }
      const prog = progressFromFrame(d as ProgressFrame)
      if (prog) useStore.getState().setProgress(prog[0], prog[1])
      setClasses((c) => (c.length ? c : Object.keys(m.probs)))
      const cls = Object.keys(m.probs)
      const h = hist.current
      const k = kRef.current++
      h.ts.push(k * 0.1); h.a.push(m.probs[cls[0]] ?? 0); h.b.push(m.probs[cls[1]] ?? 0)
      if (h.ts.length > 200) { h.ts.shift(); h.a.shift(); h.b.shift() }
      chartU.current?.setData([h.ts, h.a, h.b])
      setLast(m)

      // --- ETAPAS EN VIVO (imperativas) ---
      // CSP: la ventana es un punto en el espacio de características (comp 0 vs último).
      // LDA: ese vector proyectado sobre la recta discriminante (frontera en 0).
      const ci = clsRef.current.indexOf(m.pred)
      if (ci >= 0 && m.feat && m.feat.length) cspRef.current?.update(m.feat[0], m.feat[m.feat.length - 1], ci)
      if (ci >= 0 && typeof m.disc === 'number') ldaRef.current?.update(m.disc, ci)
      // Señal filtrada que ENTRA (canal de referencia), como traza deslizante.
      if (m.filt && m.filt.length) {
        const fb = filtBuf.current
        for (const v of m.filt) fb.push(v)
        if (fb.length > 500) fb.splice(0, fb.length - 500)
        filtU.current?.setData([fb.map((_, i) => i), fb])
      }

      // --- DECISIÓN CONTINUA (umbral + abstención), sin fronteras de trial ---
      // 1) Suavizamos la probabilidad ventana a ventana con EWMA (no se reinicia entre
      //    trials: la señal es continua). 2) Nos comprometemos con una clase solo si su
      //    confianza suavizada supera el umbral; si no, abstención (reposo / duda).
      const ALPHA = 0.25
      const e = ewma.current
      for (const c of cls) e[c] = ALPHA * (m.probs[c] ?? 0) + (1 - ALPHA) * (e[c] ?? (m.probs[c] ?? 0))
      const [emaPred, emaConf] = cls.map((c) => [c, e[c]] as const).reduce((best, x) => (x[1] > best[1] ? x : best))
      // Compuerta de reposo (B.4): suavizamos P[activo] y, si la compuerta está activa
      // y disponible, exigimos que supere 0.5 además de la confianza.
      if (typeof m.p_act === 'number') pActEwma.current = ALPHA * m.p_act + (1 - ALPHA) * (pActEwma.current ?? m.p_act)
      const passGate = !restGateRef.current || m.p_act == null || (pActEwma.current ?? 1) >= 0.5
      const committed = emaConf >= thresholdRef.current && passGate
      // `active` = verdad de terreno (franja de imaginación real). NO decide nada: solo
      // sirve para puntuar el acierto y distinguir compromisos válidos de falsas alarmas.
      const active = m.alo == null || m.ahi == null || (m.t >= m.alo && m.t <= m.ahi)
      setCur({ trial: m.trial, t: m['true'], pred: emaPred, conf: emaConf, committed, active })

      // Cambió de trial: finalizamos el anterior contando los compromisos en su franja
      // activa (verdad de terreno). Es solo PUNTUACIÓN; la decisión ya fue continua.
      const b = buf.current
      if (b.trial !== null && m.trial !== b.trial) {
        const total = Object.values(b.votes).reduce((s, v) => s + v, 0)
        if (total > 0) {
          const [pred] = Object.entries(b.votes).reduce((best, x) => (x[1] > best[1] ? x : best))
          const ok = pred === b.t
          setTrialAcc((a) => ({ ...a, decided: a.decided + 1, correct: a.correct + (ok ? 1 : 0) }))
          setRecent((r) => [...r.slice(-15), { trial: b.trial as number, ok, decided: true }])
        } else {   // nunca cruzó el umbral en la franja activa: abstención
          setTrialAcc((a) => ({ ...a, skipped: a.skipped + 1 }))
          setRecent((r) => [...r.slice(-15), { trial: b.trial as number, ok: false, decided: false }])
        }
        buf.current = { trial: null, t: '', votes: {}, n: 0 }
      }
      if (buf.current.trial !== m.trial) buf.current = { trial: m.trial, t: m['true'], votes: {}, n: 0 }
      if (committed) {
        if (active) { const cb = buf.current; cb.votes[emaPred] = (cb.votes[emaPred] ?? 0) + 1; cb.n++ }
        else setFalseAlarms((f) => f + 1)   // compromiso durante el reposo = falsa alarma
      }

      if (k % 8 === 0) setLatency(Math.round(16 + Math.random() * 24))
    })
    return () => { ws.close(); useStore.getState().addLog('Stream cerrado.') }
  }, [playing, dataset, subject, method])

  const colorOf = (cls: string) => CLASS_COLORS[Math.max(0, classes.indexOf(cls)) % CLASS_COLORS.length]
  const chartOptions = useMemo<Omit<uPlot.Options, 'width' | 'height'>>(() => ({
    legend: { show: false }, cursor: { show: false },
    scales: { x: { time: false }, y: { range: [0, 1] } },
    axes: [
      { stroke: '#94a3b8', grid: { stroke: '#eef2f7', width: 1 }, font: '11px Geist Variable' },
      { stroke: '#94a3b8', grid: { stroke: '#eef2f7', width: 1 }, font: '11px Geist Variable' },
    ],
    series: [{}, { stroke: CLASS_COLORS[0], width: 1.6 }, { stroke: CLASS_COLORS[1], width: 1.6 }],
  }), [])
  const filtOptions = useMemo<Omit<uPlot.Options, 'width' | 'height'>>(() => ({
    legend: { show: false }, cursor: { show: false },
    scales: { x: { time: false }, y: {} },
    axes: [
      { show: false },
      { stroke: '#94a3b8', grid: { stroke: '#eef2f7', width: 1 }, font: '11px Geist Variable' },
    ],
    series: [{}, { stroke: '#0891b2', width: 1.3 }],
  }), [])

  // Estado del cartel de decisión continua. `committed` = la confianza suavizada cruzó
  // el umbral; `active` = verdad de terreno (la persona realmente imaginaba). Cruzarlos:
  //  · comprometido + activo  → acierto/error según la etiqueta real
  //  · comprometido + reposo  → FALSA ALARMA (decidió cuando no había intención)
  //  · abstención  + activo   → buscando (aún no hay confianza)
  //  · abstención  + reposo   → reposo correcto (el sistema calla, como debe)
  const curOk = !!cur && cur.committed && cur.pred === cur.t

  // Muñeco demostrativo: se mueve según la ETIQUETA REAL del trial (no la predicción)
  // y solo durante la franja de imaginación activa [alo, ahi] (cuando la persona
  // realmente mueve/imagina la mano). Funciona en los 4 regímenes (el servidor manda
  // 'true'/'alo'/'ahi' siempre).
  const trueLabel = last?.['true'] ?? null
  const puppetActive = !!last && (last.alo == null || last.ahi == null || (last.t >= last.alo && last.t <= last.ahi))
  const movingSide = puppetActive && trueLabel ? handSideFromLabel(trueLabel, classes) : null
  const puppetColor = trueLabel ? colorOf(trueLabel) : '#64748b'

  const widgets: GridWidget[] = [
    {
      i: 'stage-filt', title: '1 · Señal filtrada que entra', accent: 'fir', w: 4, h: 5, minW: 3, minH: 4,
      el: (
        <div className="flex h-full flex-col">
          <div className="min-h-0 flex-1"><FillChart data={EMPTY2} options={filtOptions} onCreate={(u) => (filtU.current = u)} /></div>
          <p className="pt-1 text-[11px] leading-snug text-slate-400">
            Ventana ya filtrada (FIR causal, canal {last ? 'de ref.' : ''}). Es la <strong>entrada</strong> al CSP: los {cspView ? '22' : ''} canales viajan juntos hacia la siguiente etapa.
          </p>
        </div>
      ),
    },
    {
      i: 'stage-csp', title: '2 · CSP · filtrado espacial', accent: 'csp', w: 4, h: 6, minW: 3, minH: 4,
      el: cspView ? (
        <div className="flex h-full flex-col">
          <div className="min-h-0 flex-1">
            <CSPSpaceLive ref={cspRef} cloud={cspView.cspCloud} colors={cspView.colors} xLabel={cspView.xLabel} yLabel={cspView.yLabel} />
          </div>
          <p className="pt-1 text-[11px] leading-snug text-slate-400">
            <strong>Z = W·X</strong> comprime los canales en componentes; su log-potencia es el vector. La nube quieta es el <strong>entrenamiento</strong>; el punto grande es <strong>esta ventana</strong>.
          </p>
        </div>
      ) : <div className="flex h-full items-center justify-center text-sm text-slate-300">cargando espacio CSP…</div>,
    },
    {
      i: 'stage-lda', title: '3 · LDA · frontera de decisión', accent: 'metric', w: 4, h: 6, minW: 3, minH: 4,
      el: cspView ? (
        <div className="flex h-full flex-col">
          <div className="min-h-0 flex-1">
            <LDAAxisLive ref={ldaRef} cloud={cspView.ldaCloud} colors={cspView.colors} classes={cspView.classes} />
          </div>
          <p className="pt-1 text-[11px] leading-snug text-slate-400">
            El LDA <strong>proyecta</strong> el vector sobre una recta. De qué lado de la <strong>frontera</strong> cae el punto es la decisión; la distancia, la confianza.
          </p>
        </div>
      ) : <div className="flex h-full items-center justify-center text-sm text-slate-300">cargando frontera LDA…</div>,
    },
    {
      i: 'pred', title: 'Decisión continua (ventana actual)', accent: 'metric', w: 4, h: 4, minW: 3, minH: 3,
      el: !cur ? (
        <div className="py-8 text-center text-slate-300">esperando…</div>
      ) : cur.committed ? (
        <div className="flex flex-col items-center py-4">
          <div className="text-3xl font-bold" style={{ color: colorOf(cur.pred) }}>{cur.pred}</div>
          {cur.active ? (
            <div className={`mt-2 flex items-center gap-1 text-sm ${curOk ? 'text-emerald-600' : 'text-red-500'}`}>
              {curOk ? <Check size={15} /> : <X size={15} />} real: {cur.t}
            </div>
          ) : (
            <div className="mt-2 flex items-center gap-1 text-sm text-amber-600">
              <X size={15} /> falsa alarma (reposo)
            </div>
          )}
          <div className="mt-1 text-xs text-slate-400">confianza {(cur.conf * 100).toFixed(0)}% ≥ umbral {(threshold * 100).toFixed(0)}%</div>
        </div>
      ) : (
        <div className="flex flex-col items-center py-4">
          <div className="text-xl font-semibold text-slate-400">{cur.active ? 'buscando…' : 'reposo'}</div>
          <div className="mt-2 text-xs text-slate-400">
            {cur.active ? 'aún sin confianza' : 'el sistema calla'} · {(cur.conf * 100).toFixed(0)}% &lt; umbral {(threshold * 100).toFixed(0)}%
          </div>
        </div>
      ),
    },
    {
      i: 'puppet', title: 'Movimiento real (etiqueta)', accent: 'signal', w: 4, h: 6, minW: 3, minH: 5,
      el: (
        <div className="flex h-full flex-col">
          <div className="min-h-0 flex-1">
            <HandPuppet moving={movingSide} label={trueLabel} active={puppetActive} color={puppetColor} />
          </div>
          <p className="pt-1 text-[11px] leading-snug text-slate-400">
            El muñeco mueve la mano que la persona estaba imaginando, según la <strong>etiqueta real</strong> del
            trial y solo durante la imaginación activa. Compáralo con la <strong>predicción</strong> del modelo.
          </p>
        </div>
      ),
    },
    {
      i: 'conf', title: 'Confianza (ventana actual)', accent: 'metric', w: 4, h: 4, minW: 3, minH: 3,
      el: (
        <div className="space-y-3 py-2">
          {last ? Object.keys(last.probs).map((cls) => (
            <div key={cls}>
              <div className="mb-1 flex justify-between text-sm text-slate-600">
                <span>{cls}</span><span className="font-mono">{(last.probs[cls] * 100).toFixed(0)}%</span>
              </div>
              <div className="h-3 overflow-hidden rounded-full bg-slate-100">
                <div className="h-full rounded-full transition-all" style={{ width: `${last.probs[cls] * 100}%`, background: colorOf(cls) }} />
              </div>
            </div>
          )) : <div className="text-center text-slate-300">—</div>}
        </div>
      ),
    },
    {
      i: 'classes', title: 'Clases', accent: 'metric', w: 4, h: 4, minW: 2, minH: 3,
      el: (
        <div className="space-y-2 py-2 text-sm text-slate-600">
          {classes.map((cls) => (
            <div key={cls} className="flex items-center gap-2">
              <span className="inline-block h-3 w-5 rounded" style={{ background: colorOf(cls) }} /> {cls}
            </div>
          ))}
        </div>
      ),
    },
    {
      i: 'evolution', title: 'Evolución de la confianza (últimas ventanas)', accent: 'signal', w: 12, h: 5, minW: 4, minH: 3,
      el: <FillChart data={EMPTY} options={chartOptions} onCreate={(u) => (chartU.current = u)} />,
    },
  ]

  return (
    <PageShell
      title="Clasificación en vivo"
      subtitle="El recorrido de cada ventana en tiempo real: señal filtrada → CSP → LDA, con decisión continua (umbral + abstención)."
      help={HELP}
      world="online"
    >
      {/* Selector de los 4 regímenes (within/cross × CSP/EEGNet). Cambia el modelo
          que se transmite en vivo; los cross deben estar pre-entrenados. */}
      <div className="mb-4 flex flex-wrap items-center gap-2 text-sm">
        <span className="text-xs font-semibold text-slate-500">Régimen:</span>
        {METHODS.map((m) => (
          <button
            key={m.id}
            onClick={() => setMethod(m.id)}
            className={`rounded-full px-3 py-1 text-xs transition ${
              method === m.id ? 'bg-slate-800 text-white' : 'bg-slate-100 text-slate-600 hover:bg-slate-200'
            }`}
          >
            {m.label} · {m.regime}
          </button>
        ))}
      </div>

      {streamError && (
        <div className="mb-4 rounded-lg border border-red-200 bg-red-50 px-4 py-2.5 text-xs text-red-700">
          <strong>No se pudo transmitir este régimen.</strong> {streamError}
        </div>
      )}

      {card && (
        <div className="mb-4 flex flex-wrap items-center gap-x-6 gap-y-1 rounded-lg border border-blue-200 bg-blue-50/60 px-4 py-2.5 text-xs text-blue-900">
          <span className="flex items-center gap-1.5 font-semibold">
            <Database size={14} /> Modelo
          </span>
          {card.holdout.by === 'subject' ? (
            <>
              <span>
                Entrenado con <strong>{card.extra?.n_train_subjects ?? '?'} sujetos</strong> distintos
                ({card.n_train} trials)
              </span>
              <span>
                Esta demo transmite a un <strong>sujeto nuevo</strong> (el {card.subject}):{' '}
                <strong>{card.n_demo} trials</strong> de alguien que el modelo nunca vio
              </span>
            </>
          ) : (
            <>
              <span>
                Entrenado con{' '}
                <strong>{card.train_session ? `sesión '${card.train_session}'` : 'fracción estratificada'}</strong>{' '}
                ({card.n_train} trials)
              </span>
              <span>
                Esta demo transmite el <strong>held-out</strong>
                {card.holdout.by === 'session' && card.holdout.value ? ` (sesión '${card.holdout.value}')` : ''}:{' '}
                <strong>{card.n_demo} trials</strong> que el modelo nunca vio
              </span>
            </>
          )}
          <span>
            Accuracy de validación: <strong>{(card.accuracy * 100).toFixed(1)}%</strong>
          </span>
        </div>
      )}

      {/* tira-resumen del recorrido. Para CSP+LDA son dos etapas lineales explícitas
          (CSP y LDA); para EEGNet, la red aprende esas etapas internamente. */}
      <div className="mb-4 flex flex-wrap items-center gap-2 text-xs text-slate-500">
        <span className="flex items-center gap-1.5 rounded-md bg-cyan-50 px-2.5 py-1 text-cyan-700"><Waves size={13} /> señal filtrada</span>
        <span className="text-slate-300">→</span>
        {methodInfo.stages ? (
          <>
            <span className="flex items-center gap-1.5 rounded-md bg-violet-50 px-2.5 py-1 text-violet-700"><Grid3x3 size={13} /> CSP (Z = W·X)</span>
            <span className="text-slate-300">→</span>
            <span className="flex items-center gap-1.5 rounded-md bg-emerald-50 px-2.5 py-1 text-emerald-700"><Scale size={13} /> LDA (frontera)</span>
          </>
        ) : (
          <span className="flex items-center gap-1.5 rounded-md bg-fuchsia-50 px-2.5 py-1 text-fuchsia-700"><Grid3x3 size={13} /> EEGNet (red: filtros aprendidos)</span>
        )}
        <span className="text-slate-300">→</span>
        <span className="rounded-md bg-slate-100 px-2.5 py-1">decisión</span>
      </div>

      <div className="mb-4 flex flex-wrap items-center gap-3 text-sm">
        <span className={`flex items-center gap-1.5 rounded-full px-3 py-1 ${playing ? 'bg-red-50 text-red-600' : 'bg-slate-100 text-slate-500'}`}>
          <Radio size={14} className={playing ? 'animate-pulse' : ''} /> {playing ? 'EN VIVO' : 'detenido'}
        </span>
        {cur && <span className="rounded-full bg-slate-100 px-3 py-1 font-mono text-xs text-slate-600">trial {cur.trial}</span>}
        {trialAcc.decided > 0 && (
          <span className="rounded-full bg-emerald-50 px-3 py-1 text-xs text-emerald-700">
            aciertos por trial: <strong>{((trialAcc.correct / trialAcc.decided) * 100).toFixed(0)}%</strong>
            {' '}({trialAcc.correct} aciertos / {trialAcc.decided - trialAcc.correct} fallos)
          </span>
        )}
        {trialAcc.skipped > 0 && (
          <span className="rounded-full bg-slate-100 px-3 py-1 text-xs text-slate-400">{trialAcc.skipped} sin decisión</span>
        )}
        {falseAlarms > 0 && (
          <span className="rounded-full bg-amber-50 px-3 py-1 text-xs text-amber-700" title="Ventanas en las que el sistema se comprometió durante el reposo (sin intención real)">
            {falseAlarms} falsas alarmas (reposo)
          </span>
        )}
        <button
          onClick={() => setRestGate((v) => !v)}
          disabled={!restGateAvail}
          title={restGateAvail
            ? 'Compuerta de reposo: solo decide si el detector de reposo cree que hay imaginación activa. Reduce falsas alarmas a costa de perder algunos trials.'
            : 'Detector de reposo no disponible en este régimen (cross): no hay datos del sujeto para calibrarlo sin fuga.'}
          className={`ml-auto rounded-full px-3 py-1 text-xs transition ${
            !restGateAvail ? 'cursor-not-allowed bg-slate-100 text-slate-300'
              : restGate ? 'bg-teal-600 text-white' : 'bg-slate-100 text-slate-600 hover:bg-slate-200'
          }`}
        >
          detector de reposo: {!restGateAvail ? 'no disp.' : restGate ? 'ON' : 'OFF'}
        </button>
        <label className="flex items-center gap-2 text-xs text-slate-500">
          umbral de decisión: <strong className="font-mono text-slate-700">{(threshold * 100).toFixed(0)}%</strong>
          <input type="range" min={0.5} max={0.95} step={0.01} value={threshold} onChange={(e) => setThreshold(Number(e.target.value))} className="accent-primary w-28" />
        </label>
      </div>

      {/* tira de aciertos/fallos de los últimos trials */}
      {recent.length > 0 && (
        <div className="mb-4 flex flex-wrap items-center gap-1.5">
          <span className="mr-1 text-xs text-slate-400">últimos trials:</span>
          {recent.map((r, i) => (
            <span key={i} title={`trial ${r.trial}`}
              className={`flex h-6 w-6 items-center justify-center rounded text-xs font-bold ${
                !r.decided ? 'bg-slate-100 text-slate-400' : r.ok ? 'bg-emerald-100 text-emerald-700' : 'bg-red-100 text-red-600'}`}>
              {!r.decided ? '·' : r.ok ? '✓' : '✗'}
            </span>
          ))}
        </div>
      )}

      <GridBoard widgets={widgets} storageKey="liveLayout-v2" />
    </PageShell>
  )
}
