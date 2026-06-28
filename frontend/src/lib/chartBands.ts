import type uPlot from 'uplot'

/** Sombrea (hook drawClear) los tramos del gráfico donde la ventana estaba en la franja
 *  de imaginación ACTIVA (verdad de terreno). `act` se alinea por índice con `u.data[0]`.
 *  Se usa en los gráficos de confianza de Clasificación y del Dashboard para mostrar
 *  cuándo "debería" imaginar la persona (la confianza suele moverse antes por la ventana). */
export function drawActiveBands(u: uPlot, act: boolean[]) {
  const xs = u.data[0] as number[] | undefined
  if (!xs || !xs.length) return
  const ctx = u.ctx
  const { top, height } = u.bbox
  ctx.save()
  ctx.fillStyle = 'rgba(16, 185, 129, 0.12)'   // emerald, sutil
  let i = 0
  while (i < xs.length) {
    if (act[i]) {
      let j = i
      while (j + 1 < xs.length && act[j + 1]) j++
      const x0 = u.valToPos(xs[i] - 0.05, 'x', true)
      const x1 = u.valToPos(xs[j] + 0.05, 'x', true)
      ctx.fillRect(x0, top, Math.max(1, x1 - x0), height)
      i = j + 1
    } else {
      i++
    }
  }
  ctx.restore()
}
