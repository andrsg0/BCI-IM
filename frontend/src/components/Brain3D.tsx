import { Canvas } from '@react-three/fiber'
import { OrbitControls } from '@react-three/drei'
import { DoubleSide } from 'three'
import { divergingColor } from '../lib/color'

export type Pos3D = Record<string, [number, number, number] | null>

// Mapeo de coordenadas MNE (x=derecha, y=anterior, z=arriba) a three.js
// (x=derecha, y=arriba, z=frente). La nariz queda hacia +z.
function Electrodes({ channels, pos3d, values }: { channels: string[]; pos3d: Pos3D; values: number[] }) {
  const maxAbs = Math.max(...values.map((v) => Math.abs(v)), 1e-9)
  let maxNorm = 1e-9
  for (const ch of channels) {
    const p = pos3d[ch]
    if (p) { const n = Math.hypot(p[0], p[1], p[2]); if (n > maxNorm) maxNorm = n }
  }
  const s = 0.96 / maxNorm

  return (
    <>
      {channels.map((ch, i) => {
        const p = pos3d[ch]
        if (!p) return null
        const t = values[i] / maxAbs
        const col = divergingColor(t)
        return (
          <mesh key={ch} position={[p[0] * s, p[2] * s, p[1] * s]}>
            <sphereGeometry args={[0.045, 20, 20]} />
            <meshStandardMaterial color={col} emissive={col} emissiveIntensity={0.25 + Math.abs(t) * 1.5} roughness={0.4} />
          </mesh>
        )
      })}
    </>
  )
}

export function Brain3D({ channels, pos3d, values }: { channels: string[]; pos3d: Pos3D; values: number[] }) {
  return (
    <Canvas camera={{ position: [0, 0.5, 3], fov: 45 }} dpr={[1, 2]}>
      <color attach="background" args={['#f8fafc']} />
      <ambientLight intensity={0.75} />
      <directionalLight position={[4, 5, 6]} intensity={0.8} />
      {/* cabeza / scalp transparente */}
      <mesh>
        <sphereGeometry args={[1, 48, 48]} />
        <meshStandardMaterial color="#93c5fd" transparent opacity={0.12} side={DoubleSide} roughness={0.6} />
      </mesh>
      {/* nariz (referencia de orientación, hacia +z) */}
      <mesh position={[0, 0.1, 1.02]} rotation={[Math.PI / 2, 0, 0]}>
        <coneGeometry args={[0.07, 0.18, 16]} />
        <meshStandardMaterial color="#cbd5e1" />
      </mesh>
      <Electrodes channels={channels} pos3d={pos3d} values={values} />
      <OrbitControls enablePan={false} minDistance={1.8} maxDistance={6} />
    </Canvas>
  )
}
