// Vistas en vivo de las dos etapas del clasificador, dibujadas en SVG y
// actualizadas de forma IMPERATIVA (sin re-render de React a 10 Hz): el fondo es la
// nube de ENTRENAMIENTO (fija) y un punto se mueve ventana a ventana.
//
//  · CSPSpaceLive — espacio de características CSP (log-varianza): cada ventana de la
//    señal filtrada se convierte en un punto (comp 0 vs comp último). Diferencia la
//    etapa de FILTRADO ESPACIAL: comprime los canales en componentes.
//  · LDAAxisLive  — recta discriminante del LDA: el vector se proyecta a un escalar;
//    la FRONTERA está en 0. Diferencia la etapa de DECISIÓN: de qué lado cae.
import { forwardRef, memo, useImperativeHandle, useMemo, useRef } from 'react'

export interface CSPHandle { update(x: number, y: number, predIdx: number): void; reset(): void }
export interface LDAHandle { update(disc: number, predIdx: number): void; reset(): void }

interface CSPPt { x: number; y: number; c: number }
interface LDAPt { d: number; c: number }

// Lienzos en coordenadas fijas (el SVG escala al contenedor). A nivel de módulo
// para que sean estables y no entren en las dependencias de los hooks.
const CW = 320, CH = 250, CPAD = { l: 42, r: 14, t: 14, b: 40 }
const LW = 340, LH = 170, LPAD = { l: 16, r: 16, t: 30, b: 36 }

// ---- CSP: nube 2D + punto en vivo -----------------------------------------
export const CSPSpaceLive = memo(forwardRef<CSPHandle, {
  cloud: CSPPt[]; colors: string[]; xLabel: string; yLabel: string
}>(function CSPSpaceLive({ cloud, colors, xLabel, yLabel }, ref) {
  const dot = useRef<SVGCircleElement>(null)
  const W = CW, H = CH, pad = CPAD

  const sc = useMemo(() => {
    if (!cloud.length) return null
    const xs = cloud.map((p) => p.x), ys = cloud.map((p) => p.y)
    let x0 = Math.min(...xs), x1 = Math.max(...xs)
    let y0 = Math.min(...ys), y1 = Math.max(...ys)
    const mx = (x1 - x0) * 0.14 || 1, my = (y1 - y0) * 0.14 || 1
    x0 -= mx; x1 += mx; y0 -= my; y1 += my
    const sx = (v: number) => CPAD.l + ((v - x0) / (x1 - x0)) * (CW - CPAD.l - CPAD.r)
    const sy = (v: number) => CH - CPAD.b - ((v - y0) / (y1 - y0)) * (CH - CPAD.t - CPAD.b)
    return { sx, sy }
  }, [cloud])

  useImperativeHandle(ref, () => ({
    update(x, y, predIdx) {
      if (!sc || !dot.current) return
      const cx = Math.max(CPAD.l, Math.min(CW - CPAD.r, sc.sx(x)))
      const cy = Math.max(CPAD.t, Math.min(CH - CPAD.b, sc.sy(y)))
      dot.current.setAttribute('cx', String(cx))
      dot.current.setAttribute('cy', String(cy))
      dot.current.setAttribute('fill', colors[predIdx] ?? '#0f172a')
      dot.current.style.opacity = '1'
    },
    reset() { if (dot.current) dot.current.style.opacity = '0' },
  }), [sc, colors])

  if (!sc) return <div className="flex h-full items-center justify-center text-sm text-slate-300">sin datos CSP…</div>
  const cxAxis = (pad.l + W - pad.r) / 2
  const cyAxis = (pad.t + H - pad.b) / 2
  return (
    <svg viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="xMidYMid meet" className="h-full w-full">
      <line x1={pad.l} y1={H - pad.b} x2={W - pad.r} y2={H - pad.b} stroke="#cbd5e1" />
      <line x1={pad.l} y1={pad.t} x2={pad.l} y2={H - pad.b} stroke="#cbd5e1" />
      {cloud.map((p, i) => (
        <circle key={i} cx={sc.sx(p.x)} cy={sc.sy(p.y)} r={3}
          fill={colors[p.c] ?? '#94a3b8'} fillOpacity={0.25} />
      ))}
      <circle ref={dot} cx={-99} cy={-99} r={7} fill="#0f172a" stroke="#fff" strokeWidth={2}
        style={{ opacity: 0, transition: 'cx .08s linear, cy .08s linear' }} />
      <text x={cxAxis} y={H - 8} textAnchor="middle" fontSize={10.5} fill="#64748b">{xLabel}</text>
      <text x={13} y={cyAxis} textAnchor="middle" fontSize={10.5} fill="#64748b"
        transform={`rotate(-90 13 ${cyAxis})`}>{yLabel}</text>
    </svg>
  )
}))

// ---- LDA: recta discriminante 1D + punto en vivo --------------------------
export const LDAAxisLive = memo(forwardRef<LDAHandle, {
  cloud: LDAPt[]; colors: string[]; classes: string[]
}>(function LDAAxisLive({ cloud, colors, classes }, ref) {
  const marker = useRef<SVGGElement>(null)
  const W = LW, H = LH, pad = LPAD
  const axisY = H - pad.b

  const sc = useMemo(() => {
    if (!cloud.length) return null
    const D = (Math.max(...cloud.map((p) => Math.abs(p.d))) || 1) * 1.18
    const sx = (v: number) => LPAD.l + ((v + D) / (2 * D)) * (LW - LPAD.l - LPAD.r)
    return { sx, D }
  }, [cloud])

  useImperativeHandle(ref, () => ({
    update(disc, predIdx) {
      if (!sc || !marker.current) return
      const x = Math.max(LPAD.l, Math.min(LW - LPAD.r, sc.sx(disc)))
      marker.current.setAttribute('transform', `translate(${x} 0)`)
      const c = colors[predIdx] ?? '#0f172a'
      marker.current.querySelectorAll('[data-fill]').forEach((el) => el.setAttribute('fill', c))
      marker.current.querySelectorAll('[data-stroke]').forEach((el) => el.setAttribute('stroke', c))
      marker.current.style.opacity = '1'
    },
    reset() { if (marker.current) marker.current.style.opacity = '0' },
  }), [sc, colors])

  if (!sc) return <div className="flex h-full items-center justify-center text-sm text-slate-300">sin datos LDA…</div>
  const x0 = sc.sx(0)
  return (
    <svg viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="xMidYMid meet" className="h-full w-full">
      {/* regiones de decisión a cada lado de la frontera */}
      <rect x={pad.l} y={pad.t} width={x0 - pad.l} height={axisY - pad.t} fill={colors[0]} fillOpacity={0.07} />
      <rect x={x0} y={pad.t} width={W - pad.r - x0} height={axisY - pad.t} fill={colors[1]} fillOpacity={0.07} />
      {/* nube de entrenamiento proyectada (jitter vertical determinista) */}
      {cloud.map((p, i) => (
        <circle key={i} cx={sc.sx(p.d)} cy={pad.t + 6 + ((i * 53) % 100) / 100 * (axisY - pad.t - 12)} r={2.6}
          fill={colors[p.c] ?? '#94a3b8'} fillOpacity={0.3} />
      ))}
      {/* eje + frontera */}
      <line x1={pad.l} y1={axisY} x2={W - pad.r} y2={axisY} stroke="#cbd5e1" />
      <line x1={x0} y1={pad.t - 6} x2={x0} y2={axisY + 4} stroke="#475569" strokeWidth={1.5} strokeDasharray="4 3" />
      <text x={x0} y={pad.t - 10} textAnchor="middle" fontSize={10} fill="#475569">frontera</text>
      {/* etiquetas de clase a cada extremo */}
      <text x={pad.l} y={H - 8} textAnchor="start" fontSize={11} fill={colors[0]}>◀ {classes[0]}</text>
      <text x={W - pad.r} y={H - 8} textAnchor="end" fontSize={11} fill={colors[1]}>{classes[1]} ▶</text>
      {/* marcador en vivo */}
      <g ref={marker} transform={`translate(${-99} 0)`} style={{ opacity: 0, transition: 'transform .08s linear' }}>
        <line data-stroke x1={0} y1={pad.t - 4} x2={0} y2={axisY + 4} stroke="#0f172a" strokeWidth={2} />
        <circle data-fill cx={0} cy={axisY} r={6.5} fill="#0f172a" stroke="#fff" strokeWidth={2} />
      </g>
    </svg>
  )
}))
