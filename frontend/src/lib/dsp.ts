// DSP en el cliente para el Laboratorio: convolución en TypeScript (la misma
// operación LTI que el backend hace en Python), para filtrar en vivo al mover
// los controles sin ir al servidor por cada cambio.

/**
 * Convolución discreta y[n] = Σ_k h[k]·x[n−k], modo "same" (longitud = x,
 * centrada para compensar el retardo de grupo del FIR de fase lineal).
 */
export function convolveSame(x: number[], h: number[]): number[] {
  const nx = x.length
  const nh = h.length
  const full = new Float64Array(nx + nh - 1)
  for (let k = 0; k < nh; k++) {
    const hk = h[k]
    for (let n = 0; n < nx; n++) full[k + n] += hk * x[n]
  }
  const start = (nh - 1) >> 1
  const out = new Array<number>(nx)
  for (let i = 0; i < nx; i++) out[i] = full[start + i]
  return out
}

/**
 * Convolución CAUSAL: y[n] = Σ_{k≥0} h[k]·x[n−k], usando SOLO el pasado.
 * Es lo que puede hacer un sistema en tiempo real (un casco en vivo): introduce
 * un retardo de grupo de (M−1)/2 muestras y un transitorio al inicio. Contrasta
 * con convolveSame (offline, centrada, no causal).
 */
export function convolveCausal(x: number[], h: number[]): number[] {
  const nx = x.length
  const nh = h.length
  const out = new Array<number>(nx)
  for (let n = 0; n < nx; n++) {
    let acc = 0
    const kmax = Math.min(nh - 1, n)
    for (let k = 0; k <= kmax; k++) acc += h[k] * x[n - k]
    out[n] = acc
  }
  return out
}

/** Eje de tiempo en segundos para una señal de N muestras a fs Hz. */
export function timeAxis(n: number, fs: number): number[] {
  const t = new Array<number>(n)
  for (let i = 0; i < n; i++) t[i] = i / fs
  return t
}
