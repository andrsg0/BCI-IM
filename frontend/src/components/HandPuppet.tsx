import { useId } from 'react'

export type HandSide = 'left' | 'right' | null

/**
 * Muñeco demostrativo de la señal: mueve el brazo según la ETIQUETA REAL del
 * trial (no la predicción del modelo), y solo durante la franja de imaginación
 * activa (cuando la persona realmente está moviendo/imaginando la mano).
 *
 * Es la salida intuitiva de la demo: "esto es lo que la persona estaba haciendo".
 * Vista POR DETRÁS (como si fueras tú): la mano izquierda del muñeco queda a la
 * izquierda del espectador, sin espejo que confunda.
 *
 * El brazo activo se levanta con una animación de saludo; el color es el de la
 * clase (mismo que el resto de la página). Si no hay señal, ambos brazos reposan.
 */
export function HandPuppet({ moving, label, active, color = '#2563eb' }: {
  /** qué mano se mueve según la etiqueta real (o null = reposo) */
  moving: HandSide
  /** texto de la etiqueta real (ej. "left_hand") */
  label?: string | null
  /** ¿estamos dentro de la ventana de imaginación activa? */
  active: boolean
  /** color de la clase que se está moviendo */
  color?: string
}) {
  const uid = useId().replace(/[:]/g, '')
  // Hombros (origen de rotación de cada brazo).
  const LS: [number, number] = [78, 96]
  const RS: [number, number] = [142, 96]
  // Reposo: brazos colgando con ligera apertura. Levantado: brazo arriba+afuera.
  const leftAngle = moving === 'left' ? -118 : 6
  const rightAngle = moving === 'right' ? 118 : -6
  const leftWave = moving === 'left'
  const rightWave = moving === 'right'

  const restColor = '#94a3b8'   // slate-400 (brazo en reposo)
  const bodyColor = '#cbd5e1'   // slate-300

  return (
    <div className="flex h-full flex-col items-center justify-between gap-2">
      <style>{`
        .hp-arm-${uid}{transition:transform .45s cubic-bezier(.4,0,.2,1);transform-box:view-box}
        .hp-wave-${uid}{animation:hp-wave-${uid} .9s ease-in-out infinite}
        @keyframes hp-wave-${uid}{
          0%,100%{transform:rotate(calc(var(--a) - 13deg))}
          50%{transform:rotate(calc(var(--a) + 13deg))}
        }
      `}</style>

      <svg viewBox="0 0 220 270" className="min-h-0 w-full flex-1" role="img"
        aria-label={moving ? `muñeco moviendo la mano ${moving === 'left' ? 'izquierda' : 'derecha'}` : 'muñeco en reposo'}>
        {/* Suelo */}
        <ellipse cx="110" cy="256" rx="64" ry="9" fill="#e2e8f0" />

        {/* Piernas */}
        <path d="M96 196 L92 250" stroke={bodyColor} strokeWidth="13" strokeLinecap="round" />
        <path d="M124 196 L128 250" stroke={bodyColor} strokeWidth="13" strokeLinecap="round" />

        {/* Torso (vista de espaldas) */}
        <path d="M82 96 Q110 84 138 96 L132 200 Q110 210 88 200 Z" fill={bodyColor} />

        {/* Cabeza (de espaldas: sin cara, con casquete de pelo) */}
        <circle cx="110" cy="56" r="27" fill="#e2e8f0" />
        <path d="M85 50 Q110 26 135 50 Q110 40 85 50 Z" fill={bodyColor} />

        {/* Brazo izquierdo (del muñeco = izquierda del espectador) */}
        <g
          className={`hp-arm-${uid} ${leftWave ? `hp-wave-${uid}` : ''}`}
          style={{
            // @ts-expect-error custom property para el keyframe del saludo
            '--a': `${leftAngle}deg`,
            transformOrigin: `${LS[0]}px ${LS[1]}px`,
            transform: `rotate(${leftAngle}deg)`,
          }}
        >
          <path d={`M${LS[0]} ${LS[1]} q -6 44 -8 84`} stroke={leftWave ? color : restColor}
            strokeWidth="12" strokeLinecap="round" fill="none" />
          <circle cx={LS[0] - 8} cy={LS[1] + 86} r="10" fill={leftWave ? color : restColor} />
        </g>

        {/* Brazo derecho (del muñeco = derecha del espectador) */}
        <g
          className={`hp-arm-${uid} ${rightWave ? `hp-wave-${uid}` : ''}`}
          style={{
            // @ts-expect-error custom property para el keyframe del saludo
            '--a': `${rightAngle}deg`,
            transformOrigin: `${RS[0]}px ${RS[1]}px`,
            transform: `rotate(${rightAngle}deg)`,
          }}
        >
          <path d={`M${RS[0]} ${RS[1]} q 6 44 8 84`} stroke={rightWave ? color : restColor}
            strokeWidth="12" strokeLinecap="round" fill="none" />
          <circle cx={RS[0] + 8} cy={RS[1] + 86} r="10" fill={rightWave ? color : restColor} />
        </g>

        {/* Marcas I / D bajo cada lado (vista de espaldas) */}
        <text x="40" y="262" fontSize="13" fontWeight="700" fill="#94a3b8" textAnchor="middle">I</text>
        <text x="180" y="262" fontSize="13" fontWeight="700" fill="#94a3b8" textAnchor="middle">D</text>
      </svg>

      <div className="w-full text-center">
        {moving ? (
          <p className="text-sm font-semibold" style={{ color }}>
            Moviendo la mano {moving === 'left' ? 'izquierda' : 'derecha'}
          </p>
        ) : (
          <p className="text-sm font-medium text-slate-400">{active ? '—' : 'En reposo'}</p>
        )}
        <p className="mt-0.5 text-[11px] text-slate-400">
          Etiqueta real del trial{label ? <> · <span className="font-mono">{label}</span></> : ''} (no la predicción)
        </p>
      </div>
    </div>
  )
}
