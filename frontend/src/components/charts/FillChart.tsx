import { useLayoutEffect, useRef, useState } from 'react'
import type uPlot from 'uplot'
import { UPlotChart } from './UPlotChart'

/**
 * Gráfico uPlot que rellena la altura disponible de su contenedor (la mide con
 * un ResizeObserver). Pensado para paneles redimensionables de la cuadrícula:
 * al cambiar el tamaño del panel, el gráfico se ajusta solo.
 */
export function FillChart({ data, options, onCreate }: {
  data: uPlot.AlignedData
  options: Omit<uPlot.Options, 'width' | 'height'>
  onCreate?: (u: uPlot) => void
}) {
  const ref = useRef<HTMLDivElement>(null)
  const [h, setH] = useState(0)
  useLayoutEffect(() => {
    const el = ref.current
    if (!el) return
    const ro = new ResizeObserver(() => setH(el.clientHeight))
    ro.observe(el)
    setH(el.clientHeight)
    return () => ro.disconnect()
  }, [])
  return (
    <div ref={ref} className="h-full w-full">
      {h > 20 && <UPlotChart data={data} options={options} height={h} onCreate={onCreate} />}
    </div>
  )
}
