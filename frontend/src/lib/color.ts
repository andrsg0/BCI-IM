// ---------------------------------------------------------------------------
// Código de color del sistema (fuente única). La idea: un color SIGNIFICA siempre
// lo mismo en todos los gráficos, para que la demo se lea de un vistazo.
//
//  · Colores por CLASE (identidad de lo que se predice): azul = 1ª clase,
//    rojo = 2ª, verde = 3ª, ámbar = 4ª. Se usan en predicción, confianza,
//    decisión, matriz de confusión… siempre con el MISMO orden de clases.
//  · Colores por ETAPA de la señal (la misma señal a lo largo del pipeline):
//    cruda = gris, filtrada = cian, discriminante LDA = violeta.
//  · ACIERTO/ERROR es semántica aparte: verde = acierto, rojo = error (iconos),
//    no se mezcla con el color de clase.
// ---------------------------------------------------------------------------
export const CLASS_COLORS = ['#2563eb', '#e11d48', '#059669', '#d97706'] as const

/** Color estable de una clase según el orden de clases del dataset. */
export function classColor(classes: string[], cls: string): string {
  const i = classes.indexOf(cls)
  return CLASS_COLORS[(i < 0 ? 0 : i) % CLASS_COLORS.length]
}

/** Colores por etapa del pipeline (no dependen de la clase). */
export const STAGE_COLORS = {
  raw: '#64748b',   // señal cruda — gris pizarra (sin procesar)
  filt: '#0891b2',  // señal filtrada µ/β — cian (procesada por el FIR)
  disc: '#7c3aed',  // discriminante LDA — violeta
} as const

/** Semántica de acierto/error (independiente del color de clase). */
export const OUTCOME_COLORS = { ok: '#059669', bad: '#e11d48' } as const

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
