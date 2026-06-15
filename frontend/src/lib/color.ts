// Escala de color divergente para pesos espaciales: azul(−) · blanco(0) · rojo(+).
// t debe venir normalizado a [-1, 1].
export function divergingColor(t: number): string {
  const v = Math.max(-1, Math.min(1, t))
  if (v >= 0) {
    const c = Math.round(255 - v * 185)
    return `rgb(255, ${c}, ${c})`        // blanco → rojo
  }
  const k = -v
  return `rgb(${Math.round(255 - k * 185)}, ${Math.round(255 - k * 130)}, 255)` // blanco → azul
}
