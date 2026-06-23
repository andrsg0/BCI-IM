import { Suspense, useMemo, useState } from 'react'
import { Canvas } from '@react-three/fiber'
import { OrbitControls, Html } from '@react-three/drei'
import { Vector3, Quaternion } from 'three'
import { divergingColor } from '../lib/color'
import { BrainMesh, type SurfacePoint } from './BrainMesh'

export type Pos3D = Record<string, [number, number, number] | null>

const AXIS_Z = new Vector3(0, 0, 1)

// Quiralidad del modelo .glb. Los electrodos se colocan en su dirección REAL del
// montaje (no dependen de esto); este flag solo voltea la anatomía del cerebro. Si
// los hemisferios salen al revés (C1/C3/C5 deberían caer sobre el lóbulo IZQUIERDO),
// cámbialo a true.
const MIRROR_LR = false

// Electrodos como ANILLOS (no esferas): en la cabeza real el electrodo es un contacto
// sobre el cuero cabelludo, no clavado en la corteza. BrainMesh los coloca flotando
// sobre una cáscara por encima del cerebro (GAP); aquí cada anillo se orienta con su
// eje hacia afuera (radial). El color sigue la misma escala divergente que el heatmap.
function Electrodes({ channels, points, values, dim }: { channels: string[]; points: SurfacePoint[]; values: number[]; dim?: boolean[] }) {
  const [hover, setHover] = useState<number | null>(null)
  const maxAbs = Math.max(...values.map((v) => Math.abs(v)), 1e-9)

  // Orientación de cada anillo (su eje +z mira hacia afuera). Solo depende de los
  // puntos, no de los valores: se recalcula al cambiar de sujeto, no por frame.
  const quats = useMemo(
    () => points.map((p) => {
      if (!p) return null
      const dir = new Vector3(p[0], p[1], p[2])
      if (dir.length() < 1e-9) return null
      return new Quaternion().setFromUnitVectors(AXIS_Z, dir.normalize())
    }),
    [points],
  )

  return (
    <>
      {channels.map((ch, i) => {
        const p = points[i]
        const q = quats[i]
        if (!p || !q) return null
        const t = values[i] / maxAbs
        const col = divergingColor(t)
        const on = hover === i
        const dd = !!dim?.[i] && !on   // electrodo atenuado (no motor, foco activo)
        return (
          <mesh
            key={ch}
            position={p}
            quaternion={q}
            scale={on ? 1.3 : 1}
            onPointerOver={(e) => { e.stopPropagation(); setHover(i); document.body.style.cursor = 'pointer' }}
            onPointerOut={() => { setHover(null); document.body.style.cursor = 'auto' }}
          >
            <torusGeometry args={[0.05, 0.016, 10, 28]} />
            <meshStandardMaterial color={col} emissive={col} emissiveIntensity={dd ? 0.04 : 0.25 + Math.abs(t) * 1.5} roughness={0.4} transparent opacity={dd ? 0.16 : 1} />
            {on && (
              <Html center distanceFactor={3.2} zIndexRange={[100, 0]} style={{ pointerEvents: 'none' }}>
                <div className="whitespace-nowrap rounded bg-slate-900/90 px-1.5 py-0.5 text-center text-white shadow-md" style={{ transform: 'translateY(-160%)' }}>
                  <div className="text-[10px] font-semibold leading-tight">{ch}</div>
                  <div className="font-mono text-[8px] leading-tight text-slate-300">
                    {values[i] >= 0 ? '+' : ''}{values[i].toFixed(2)}
                  </div>
                </div>
              </Html>
            )}
          </mesh>
        )
      })}
    </>
  )
}

export function Brain3D({ channels, pos3d, values, dimMask }: { channels: string[]; pos3d: Pos3D; values: number[]; dimMask?: boolean[] }) {
  // Puntos de cada electrodo proyectados sobre la corteza (los reporta BrainMesh
  // tras cargar el .glb y hacer raycast). Hasta entonces no dibujamos nodos.
  const [points, setPoints] = useState<SurfacePoint[] | null>(null)

  return (
    <Canvas camera={{ position: [0, 0.5, 3], fov: 45 }} dpr={[1, 2]}>
      <color attach="background" args={['#f8fafc']} />
      <ambientLight intensity={0.75} />
      <directionalLight position={[4, 5, 6]} intensity={0.8} />
      {/* malla anatómica de cerebro: hace de corteza, lleva el heatmap por shader y
          proyecta los electrodos sobre su superficie (onPoints).
          - rotation +90° sobre y: el modelo trae el frente en -x; así el lóbulo
            frontal mira hacia +z (el frente de la cabeza).
          - scale x: voltea la anatomía del cerebro (ver MIRROR_LR). Los electrodos NO
            cuelgan de este grupo (se colocan por dirección de montaje en BrainMesh),
            así que voltear esto recoloca los lóbulos bajo unos nodos que no se mueven. */}
      <group scale={[MIRROR_LR ? -1 : 1, 1, 1]}>
        <Suspense fallback={null}>
          <BrainMesh
            rotation={[0, Math.PI / 2, 0]}
            channels={channels}
            pos3d={pos3d}
            values={values}
            onPoints={setPoints}
          />
        </Suspense>
      </group>
      {points && <Electrodes channels={channels} points={points} values={values} dim={dimMask} />}
      <OrbitControls enablePan={false} minDistance={1.8} maxDistance={6} />
    </Canvas>
  )
}
