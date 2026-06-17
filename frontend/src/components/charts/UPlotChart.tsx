import { useEffect, useRef } from 'react'
import uPlot from 'uplot'
import 'uplot/dist/uPlot.min.css'

export interface UPlotChartProps {
  data: uPlot.AlignedData
  /** opciones de uPlot SIN width/height (se calculan automáticamente). Memoizar en el padre. */
  options: Omit<uPlot.Options, 'width' | 'height'>
  height?: number
  /** acceso a la instancia (para redibujar overlays animados). */
  onCreate?: (u: uPlot) => void
}

/** Wrapper React de uPlot: rápido para señales con miles de puntos y streaming. */
export function UPlotChart({ data, options, height = 200, onCreate }: UPlotChartProps) {
  const containerRef = useRef<HTMLDivElement>(null)
  const plotRef = useRef<uPlot | null>(null)
  const dataRef = useRef(data)
  dataRef.current = data
  const heightRef = useRef(height)
  heightRef.current = height

  // Crear / destruir el plot SOLO cuando cambian las opciones (memoizadas).
  // El alto se aplica con setSize (sin recrear) para que el redimensionado en la
  // cuadrícula sea suave y no invalide las referencias (gráficos en vivo).
  useEffect(() => {
    const el = containerRef.current
    if (!el) return
    const width = el.clientWidth || 600
    const u = new uPlot({ ...options, width, height: heightRef.current } as uPlot.Options, dataRef.current, el)
    plotRef.current = u
    onCreate?.(u)
    const ro = new ResizeObserver(() => {
      if (el.clientWidth) u.setSize({ width: el.clientWidth, height: heightRef.current })
    })
    ro.observe(el)
    return () => {
      ro.disconnect()
      u.destroy()
      plotRef.current = null
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [options])

  // Aplicar cambios de alto sin recrear el plot.
  useEffect(() => {
    const u = plotRef.current
    const el = containerRef.current
    if (u && el?.clientWidth) u.setSize({ width: el.clientWidth, height })
  }, [height])

  // Actualizar datos sin recrear el plot.
  useEffect(() => {
    plotRef.current?.setData(data)
  }, [data])

  return <div ref={containerRef} className="w-full" />
}
