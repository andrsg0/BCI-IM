// Progreso de la señal en vivo. La demo reproduce los trials reservados (held-out)
// en bucle: cada pasada es FINITA (demo_n trials × trial_s segundos). El servidor manda
// la posición en la tanda (demo_i), el total (demo_n), la duración de trial (trial_s) y
// el tiempo dentro del trial actual (t). Con eso reconstruimos una barra que se llena en
// cada pasada y se reinicia al repetir el bucle.

export interface ProgressFrame {
  demo_i?: number
  demo_n?: number
  trial_s?: number
  t?: number
}

/** Devuelve [elapsedSec, totalSec] de la pasada actual, o null si el frame no trae
 *  la info de progreso (p. ej. streams antiguos). */
export function progressFromFrame(f: ProgressFrame): [number, number] | null {
  const { demo_i, demo_n, trial_s } = f
  if (demo_i == null || demo_n == null || trial_s == null || demo_n <= 0) return null
  const total = demo_n * trial_s
  const within = Math.max(0, Math.min(f.t ?? 0, trial_s))
  const elapsed = demo_i * trial_s + within
  return [elapsed, total]
}
