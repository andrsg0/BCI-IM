import { Suspense, useState } from 'react'
import { Canvas } from '@react-three/fiber'
import { OrbitControls, Html } from '@react-three/drei'
import { DoubleSide } from 'three'
import { divergingColor } from '../lib/color'
import { BrainMesh, type SurfacePoint } from './BrainMesh'

export type Pos3D = Record<string, [number, number, number] | null>

// Nodos de electrodos colocados sobre los puntos que el raycast proyectó sobre la
// corteza (BrainMesh), no sobre una esfera. El color discreto sigue la misma escala
// divergente que el heatmap, normalizado por maxAbs del frame.
function Electrodes({ channels, points, values }: { channels: string[]; points: SurfacePoint[]; values: number[] }) {
  const [hover, setHover] = useState<number | null>(null)
  const maxAbs = Math.max(...values.map((v) => Math.abs(v)), 1e-9)

  return (
    <>
      {channels.map((ch, i) => {
        const p = points[i]
        if (!p) return null
        const t = values[i] / maxAbs
        const col = divergingColor(t)
        const on = hover === i
        return (
          <mesh
            key={ch}
            position={p}
            onPointerOver={(e) => { e.stopPropagation(); setHover(i); document.body.style.cursor = 'pointer' }}
            onPointerOut={() => { setHover(null); document.body.style.cursor = 'auto' }}
          >
            <sphereGeometry args={[on ? 0.06 : 0.045, 20, 20]} />
            <meshStandardMaterial color={col} emissive={col} emissiveIntensity={0.25 + Math.abs(t) * 1.5} roughness={0.4} />
            {on && (
              <Html center distanceFactor={6} zIndexRange={[100, 0]} style={{ pointerEvents: 'none' }}>
                <div className="whitespace-nowrap rounded-md bg-slate-900/90 px-2 py-1 text-center text-white shadow-lg">
                  <div className="text-sm font-semibold leading-tight">{ch}</div>
                  <div className="font-mono text-[11px] text-slate-300">
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

export function Brain3D({ channels, pos3d, values }: { channels: string[]; pos3d: Pos3D; values: number[] }) {
  // Puntos de cada electrodo proyectados sobre la corteza (los reporta BrainMesh
  // tras cargar el .glb y hacer raycast). Hasta entonces no dibujamos nodos.
  const [points, setPoints] = useState<SurfacePoint[] | null>(null)

  return (
    <Canvas camera={{ position: [0, 0.5, 3], fov: 45 }} dpr={[1, 2]}>
      <color attach="background" args={['#f8fafc']} />
      <ambientLight intensity={0.75} />
      <directionalLight position={[4, 5, 6]} intensity={0.8} />
      {/* cabeza / scalp: solo una referencia muy tenue (los electrodos van ahora
          apoyados sobre la corteza, no sobre esta esfera). */}
      <mesh>
        <sphereGeometry args={[1, 48, 48]} />
        <meshStandardMaterial color="#93c5fd" transparent opacity={0.05} side={DoubleSide} roughness={0.6} />
      </mesh>
      {/* malla anatómica de cerebro: hace de corteza, lleva el heatmap por shader y
          proyecta los electrodos sobre su superficie (onPoints).
          rotation: el modelo viene con el frente hacia +x; lo giramos -90° sobre el
          eje vertical (y) para que mire al frente (+z, hacia la nariz). */}
      <Suspense fallback={null}>
        <BrainMesh
          rotation={[0, -Math.PI / 2, 0]}
          channels={channels}
          pos3d={pos3d}
          values={values}
          onPoints={setPoints}
        />
      </Suspense>
      {/* nariz (referencia de orientación, hacia +z) */}
      <mesh position={[0, 0.1, 1.02]} rotation={[Math.PI / 2, 0, 0]}>
        <coneGeometry args={[0.07, 0.18, 16]} />
        <meshStandardMaterial color="#cbd5e1" />
      </mesh>
      {points && <Electrodes channels={channels} points={points} values={values} />}
      <OrbitControls enablePan={false} minDistance={1.8} maxDistance={6} />
    </Canvas>
  )
}
