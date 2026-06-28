// Catálogo de datasets (espejo de los configs del backend). El backend dará la
// lista real de canales/sujetos; aquí hay valores por defecto para los selectores.

export type DatasetId = 'BNCI2014_001' | 'BNCI2014_004' | 'Kumar2024'

export interface DatasetInfo {
  id: DatasetId
  label: string
  subjects: number
  fs: number
  channels: string[]
  /** accuracy media k-fold medida en la Etapa 1 */
  accuracy: number
  /** nº de sesiones reales (espejo del REGISTRY del backend). Se DERIVA de aquí si el
   *  dataset sirve para la demo en vivo: ≥2 sesiones ⇒ estimación honesta inter-sesión.
   *  Reemplaza al antiguo campo manual 'role'. Ver docs/informe/01-datos.md. */
  sessions: number
}

const MOTOR_CHANNELS = [
  'FC3', 'FC1', 'FCz', 'FC2', 'FC4',
  'C3', 'C1', 'Cz', 'C2', 'C4',
  'CP3', 'CPz', 'CP4', 'Pz',
]

export const DATASETS: Record<DatasetId, DatasetInfo> = {
  BNCI2014_001: { id: 'BNCI2014_001', label: 'BCI IV 2a', subjects: 9, fs: 250, channels: MOTOR_CHANNELS, accuracy: 0.688, sessions: 2 },
  BNCI2014_004: { id: 'BNCI2014_004', label: 'BCI IV 2b', subjects: 9, fs: 250, channels: ['C3', 'Cz', 'C4'], accuracy: 0.604, sessions: 5 },
  Kumar2024: { id: 'Kumar2024', label: 'Kumar2024', subjects: 18, fs: 512, channels: MOTOR_CHANNELS, accuracy: 0.644, sessions: 6 },
}

export const DATASET_LIST = Object.values(DATASETS)

// ≥2 sesiones ⇒ apto para la demo en vivo (estimación honesta inter-sesión).
export const isLive = (d: DatasetInfo) => d.sessions >= 2

// Solo los datasets aptos para la demo en vivo (lo que ofrece el panel lateral).
export const LIVE_DATASET_LIST = DATASET_LIST.filter(isLive)
