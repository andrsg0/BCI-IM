import { useEffect, useMemo, useRef } from 'react'
import { useGLTF } from '@react-three/drei'
import {
  Box3,
  Vector3,
  Mesh,
  MeshStandardMaterial,
  DoubleSide,
  type Group,
  type IUniform,
} from 'three'

const MODEL_URL = '/models/brain.glb'

// Máximo de electrodos que admite el shader (los datasets van de 22 a 64 canales).
const MAX_E = 64

export type SurfacePoint = [number, number, number] | null

// GLSL: misma escala divergente que lib/color.ts (azul − · blanco 0 · rojo +).
const GLSL_DIVERGING = /* glsl */ `
vec3 diverging(float t) {
  t = clamp(t, -1.0, 1.0);
  if (t >= 0.0) {
    float c = (255.0 - t * 185.0) / 255.0;
    return vec3(1.0, c, c);            // blanco -> rojo
  }
  float k = -t;
  return vec3((255.0 - k * 185.0) / 255.0, (255.0 - k * 130.0) / 255.0, 1.0); // blanco -> azul
}
`

/**
 * Malla anatómica de cerebro (.glb) que hace de "corteza" dentro del scalp.
 * Hace tres cosas:
 *  1. Auto-centra y auto-escala el modelo por su bounding box (radio objetivo).
 *  2. Inyecta un HEATMAP CORTICAL en el material vía onBeforeCompile: interpola
 *     en GPU, por píxel, la potencia µ/β de cada electrodo con un kernel gaussiano
 *     sobre la DISTANCIA ANGULAR (la generalización 3D del topomapa). Resolución-
 *     independiente: degradado suave aunque la malla sea low-poly. Por frame solo
 *     se actualiza el uniform uValues[] (~64 floats), sin recrear geometría.
 *  3. Proyecta cada electrodo sobre la superficie por raycast (dirección radial
 *     -> primer impacto con la malla) y reporta esos puntos vía onPoints, para que
 *     los nodos descansen sobre la corteza en vez de flotar/hundirse.
 *
 * El color sale del `power` real del stream: misma normalización por maxAbs que
 * los nodos, sin datos inventados.
 */
export function BrainMesh({
  channels,
  pos3d,
  values,
  onPoints,
  rotation = [0, 0, 0],
  targetRadius = 0.8,
  color = '#e6b3bf',
  sigma = 0.45,
  strength = 0.9,
}: {
  channels: string[]
  pos3d: Record<string, [number, number, number] | null>
  values: number[]
  onPoints: (pts: SurfacePoint[]) => void
  rotation?: [number, number, number]
  targetRadius?: number
  color?: string
  /** ancho angular (rad) del kernel gaussiano de interpolación */
  sigma?: number
  /** cuánto tiñe el heatmap el tejido [0..1] */
  strength?: number
}) {
  const { scene } = useGLTF(MODEL_URL)
  const groupRef = useRef<Group>(null)

  // Buffers vivos que comparten JS y GPU (se mutan in-place; el shader lee la
  // misma referencia cada frame, así no recompilamos nada).
  const valuesBuf = useRef<Float32Array>(new Float32Array(MAX_E))
  const uniforms = useRef<Record<string, IUniform> | null>(null)

  // Direcciones unitarias de cada electrodo desde el centro de la cabeza, en el
  // marco de three.js (MNE x=der, y=frente, z=arriba  ->  three x, z, y).
  const dirs = useMemo(() => {
    return channels.map((ch) => {
      const p = pos3d[ch]
      if (!p) return null
      const v = new Vector3(p[0], p[2], p[1])
      const n = v.length()
      return n > 1e-9 ? v.multiplyScalar(1 / n) : null
    })
  }, [channels, pos3d])

  // Clon con material de tejido + heatmap inyectado. Se reconstruye solo cuando
  // cambian el modelo, el color o el set de canales (NO con cada frame de values).
  const cloned = useMemo(() => {
    const c = scene.clone(true)

    // uDirs como array fijo de MAX_E vec3 (padding con ceros).
    const uDirs = Array.from({ length: MAX_E }, () => new Vector3())
    dirs.forEach((d, i) => { if (d && i < MAX_E) uDirs[i].copy(d) })
    const count = Math.min(dirs.filter(Boolean).length ? dirs.length : 0, MAX_E)

    // DoubleSide: el raycast respeta material.side, así el rayo impacta SIEMPRE la
    // cara cercana (no la atraviesa hacia la interna) y el render del grupo espejado
    // de Brain3D no se ve "del revés".
    const mat = new MeshStandardMaterial({ color, roughness: 0.85, metalness: 0, side: DoubleSide })
    mat.onBeforeCompile = (shader) => {
      shader.uniforms.uCount = { value: count }
      shader.uniforms.uDirs = { value: uDirs }
      shader.uniforms.uValues = { value: valuesBuf.current }
      shader.uniforms.uMaxAbs = { value: 1 }
      shader.uniforms.uSigma = { value: sigma }
      shader.uniforms.uStrength = { value: strength }
      uniforms.current = shader.uniforms

      // --- Vertex: dirección del fragmento desde el centro (mundo) ---
      shader.vertexShader = shader.vertexShader
        .replace('#include <common>', '#include <common>\nvarying vec3 vHeatDir;')
        .replace(
          '#include <begin_vertex>',
          '#include <begin_vertex>\n  vHeatDir = (modelMatrix * vec4(position, 1.0)).xyz;',
        )

      // --- Fragment: interpolación angular + mezcla en color y emisivo ---
      shader.fragmentShader = shader.fragmentShader
        .replace(
          '#include <common>',
          `#include <common>
          #define MAX_E ${MAX_E}
          varying vec3 vHeatDir;
          uniform int uCount;
          uniform vec3 uDirs[MAX_E];
          uniform float uValues[MAX_E];
          uniform float uMaxAbs;
          uniform float uSigma;
          uniform float uStrength;
          ${GLSL_DIVERGING}`,
        )
        .replace(
          '#include <color_fragment>',
          `#include <color_fragment>
          vec3 hm_dir = normalize(vHeatDir);
          float hm_num = 0.0;
          float hm_den = 0.0;
          for (int i = 0; i < MAX_E; i++) {
            if (i >= uCount) break;
            float d = clamp(dot(hm_dir, uDirs[i]), -1.0, 1.0);
            float a = acos(d);                                   // distancia angular
            float w = exp(-(a * a) / (2.0 * uSigma * uSigma));   // kernel gaussiano
            hm_num += w * uValues[i];
            hm_den += w;
          }
          float hm_val = hm_den > 0.0 ? hm_num / hm_den : 0.0;
          float hm_t = clamp(hm_val / max(uMaxAbs, 1e-6), -1.0, 1.0);
          vec3 hm_col = diverging(hm_t);
          float hm_mag = abs(hm_t);
          diffuseColor.rgb = mix(diffuseColor.rgb, hm_col, clamp(hm_mag * uStrength, 0.0, 1.0));`,
        )
        .replace(
          '#include <emissivemap_fragment>',
          '#include <emissivemap_fragment>\n  totalEmissiveRadiance += hm_col * hm_mag * 0.5;',
        )
    }

    c.traverse((o) => {
      const mesh = o as Mesh
      if (mesh.isMesh) mesh.material = mat
    })
    return c
  }, [scene, color, dirs, sigma, strength])

  // Centrar en el origen y escalar (lado mayor = 2*targetRadius).
  const { scale, offset } = useMemo(() => {
    const box = new Box3().setFromObject(cloned)
    const size = box.getSize(new Vector3())
    const center = box.getCenter(new Vector3())
    const maxDim = Math.max(size.x, size.y, size.z) || 1
    return { scale: (targetRadius * 2) / maxDim, offset: center }
  }, [cloned, targetRadius])

  // Heatmap en vivo: solo actualiza uValues[] y uMaxAbs (sin recompilar).
  useEffect(() => {
    const buf = valuesBuf.current
    let maxAbs = 1e-9
    for (let i = 0; i < buf.length; i++) {
      const v = i < values.length ? values[i] : 0
      buf[i] = v
      if (Math.abs(v) > maxAbs) maxAbs = Math.abs(v)
    }
    if (uniforms.current) uniforms.current.uMaxAbs.value = maxAbs
  }, [values])

  // Colocar electrodos sobre una CÁSCARA ELIPSOIDAL suave (el "cuero cabelludo"),
  // no sobre la corteza bache a bache: el scalp real pasa por encima de los surcos
  // —en especial la fisura interhemisférica de la línea media (Fz, Cz, POz…)—. Se
  // toma la caja envolvente del cerebro en mundo (invariante al espejo de Brain3D) y
  // se sitúa cada electrodo donde su dirección del montaje corta el elipsoide, con un
  // pequeño margen para que floten justo por encima. Así NINGUNO se hunde y quedan en
  // su dirección real del montaje (independiente de la quiralidad del modelo).
  useEffect(() => {
    const grp = groupRef.current
    if (!grp) return
    grp.updateWorldMatrix(true, true)
    const box = new Box3().setFromObject(grp)
    const half = box.getSize(new Vector3()).multiplyScalar(0.5)
    const ctr = box.getCenter(new Vector3())
    const MARGIN = 1.1 // 10 % por encima de la superficie del cerebro
    const pts: SurfacePoint[] = dirs.map((d) => {
      if (!d) return null
      // s tal que ctr + s·d cae sobre el elipsoide de semiejes `half`
      const inv = Math.hypot(d.x / (half.x || 1), d.y / (half.y || 1), d.z / (half.z || 1))
      const s = (inv > 1e-9 ? 1 / inv : 1) * MARGIN
      return [ctr.x + d.x * s, ctr.y + d.y * s, ctr.z + d.z * s]
    })
    onPoints(pts)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [cloned, dirs, scale, offset, targetRadius])

  return (
    <group ref={groupRef} rotation={rotation}>
      <group scale={scale} position={[-offset.x * scale, -offset.y * scale, -offset.z * scale]}>
        <primitive object={cloned} />
      </group>
    </group>
  )
}

useGLTF.preload(MODEL_URL)
