// Tipos y acceso a la API de resultados (/api/results). El backend ensambla los
// CSV por sujeto + fichas ModelCard que ya existen en disco (ver server/results.py),
// así que aquí NO se hardcodea ningún número: todo viene del pipeline real.
import { getJSON } from '../api/client'

export interface Stat {
  mean: number
  std: number
  min: number
  max: number
  n: number
}

export interface SubjectRow {
  subject: number
  n_trials: number | null
  csp_within_acc?: number | null
  csp_within_kappa?: number | null
  csp_within_sens?: number | null
  csp_within_spec?: number | null
  csp_inter_acc?: number | null
  csp_inter_kappa?: number | null
  csp_inter_sens?: number | null
  csp_inter_spec?: number | null
  csp_cross_acc?: number | null
  csp_cross_kappa?: number | null
  eegnet_within_acc?: number | null
  eegnet_within_kappa?: number | null
  eegnet_cross_acc?: number | null
  eegnet_cross_kappa?: number | null
}

export interface Significance {
  stat: number
  p: number
  n: number
}

export interface Pooled {
  loso_mean: number | null
  loso_per_subject: Record<string, number>
  n_subjects?: number | null
  n_train?: number | null
  epochs?: number | null
  augment?: boolean | null
  augment_copies?: number | null
  device?: string | null
  trained_on?: string | null
  n_channels?: number | null
  fir?: { low_hz: number; high_hz: number; num_taps: number } | null
}

export type ResultStatus = 'measured' | 'partial' | 'pending'

export interface DatasetResult {
  id: string
  label: string
  /** nº de sesiones reales; ``live`` se deriva (≥2 ⇒ apto demo en vivo). */
  sessions: number
  live: boolean
  fs: number | null
  n_subjects_declared?: number | null
  n_subjects_evaluated: number
  classes: string[]
  chance: number
  status: ResultStatus
  has_intersession: boolean
  has_compare: boolean
  subjects?: SubjectRow[]
  summary: Record<string, Stat>
  matrix: {
    csp: { within: number | null; cross: number | null }
    eegnet: { within: number | null; cross: number | null }
  }
  /** ITR (bits/min) por método/escenario — fórmula de Wolpaw (2000). */
  itr: {
    csp: { within: number | null; cross: number | null }
    eegnet: { within: number | null; cross: number | null }
  }
  /** Kappa matrix (espejo de accuracy matrix). */
  kappa_matrix: {
    csp: { within: number | null; cross: number | null }
    eegnet: { within: number | null; cross: number | null }
  }
  /** Gini coefficient por métrica de accuracy (dispersión inter-sujeto). */
  gini: Record<string, number>
  /** Tiempo por decisión usado para el cálculo de ITR (streaming.window_s). */
  trial_time_s: number
  significance: { within?: Significance; cross?: Significance }
  pooled: Pooled | null
}

export interface AggregateResult {
  matrix: {
    csp: { within: number | null; cross: number | null }
    eegnet: { within: number | null; cross: number | null }
  }
  summary: Record<string, Stat>
  gini: Record<string, number>
  significance: { within?: Significance; cross?: Significance }
  per_dataset: {
    id: string
    label: string
    live: boolean | null
    n: number
    cells: Record<string, number>
    gini?: Record<string, number>
    itr?: {
      csp: { within: number | null; cross: number | null }
      eegnet: { within: number | null; cross: number | null }
    }
  }[]
  n_datasets: number
}

export const fetchResultsIndex = () => getJSON<DatasetResult[]>('/results')
export const fetchDatasetResult = (id: string) => getJSON<DatasetResult>(`/results/${id}`)
export const fetchAggregate = () => getJSON<AggregateResult>('/results_aggregate')

// --- helpers de formato ----------------------------------------------------
export const pct = (x?: number | null) =>
  x === null || x === undefined || Number.isNaN(x) ? '—' : `${(x * 100).toFixed(1)}%`

export const kappa = (x?: number | null) =>
  x === null || x === undefined || Number.isNaN(x) ? '—' : x.toFixed(2)

/** ITR en bits/min formateado con 1 decimal. */
export const fmtItr = (x?: number | null) =>
  x === null || x === undefined || Number.isNaN(x) ? '—' : `${x.toFixed(1)}`

/** Gini coefficient formateado con 3 decimales. */
export const fmtGini = (x?: number | null) =>
  x === null || x === undefined || Number.isNaN(x) ? '—' : x.toFixed(3)

export const STATUS_LABEL: Record<ResultStatus, string> = {
  measured: 'Medido (2×2 completo)',
  partial: 'Parcial (solo CSP+LDA within)',
  pending: 'Pendiente de evaluar',
}
