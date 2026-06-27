// Tira de verificación por trial: un cuadrito por trial reciente con su resultado
// (✓ acierto · ✗ fallo · · sin decisión/abstención). Presentacional puro: el estado
// lo calcula quien la usa (la página de Clasificación o el panel del Dashboard).

export interface TrialOutcome {
  trial: number
  ok: boolean
  /** ¿el sistema llegó a comprometerse con una clase? (false = abstención/sin decisión) */
  decided: boolean
}

export function RecentTrialsStrip({ recent }: { recent: TrialOutcome[] }) {
  if (recent.length === 0) {
    return (
      <div className="flex h-full items-center justify-center text-center text-sm text-slate-300">
        Aún no hay trials evaluados.
      </div>
    )
  }
  return (
    <div className="flex h-full flex-wrap content-center items-center gap-1.5">
      {recent.map((r, i) => (
        <span
          key={i}
          title={`trial ${r.trial}`}
          className={`flex h-6 w-6 items-center justify-center rounded text-xs font-bold ${
            !r.decided ? 'bg-slate-100 text-slate-400' : r.ok ? 'bg-emerald-100 text-emerald-700' : 'bg-red-100 text-red-600'
          }`}
        >
          {!r.decided ? '·' : r.ok ? '✓' : '✗'}
        </span>
      ))}
    </div>
  )
}
