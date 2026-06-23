// Catálogo de datasets (espejo de los configs del backend). El backend dará la
// lista real de canales/sujetos; aquí hay valores por defecto para los selectores.

export type DatasetId = 'BNCI2014_001' | 'PhysionetMI' | 'Liu2024'

export interface DatasetInfo {
  id: DatasetId
  label: string
  subjects: number
  fs: number
  channels: string[]
  /** accuracy media k-fold medida en la Etapa 1 */
  accuracy: number
  /** uso del dataset (espejo del REGISTRY del backend): 'live' = apto para la demo en
   *  vivo (≥2 sesiones); 'training' = solo benchmark de población. Ver docs/datasets.md. */
  role: 'live' | 'training'
}

const MOTOR_CHANNELS = [
  'FC3', 'FC1', 'FCz', 'FC2', 'FC4',
  'C3', 'C1', 'Cz', 'C2', 'C4',
  'CP3', 'CPz', 'CP4', 'Pz',
]

export const DATASETS: Record<DatasetId, DatasetInfo> = {
  BNCI2014_001: { id: 'BNCI2014_001', label: 'BCI IV 2a', subjects: 9, fs: 250, channels: MOTOR_CHANNELS, accuracy: 0.688, role: 'live' },
  PhysionetMI: { id: 'PhysionetMI', label: 'PhysioNet MMI', subjects: 109, fs: 160, channels: MOTOR_CHANNELS, accuracy: 0.608, role: 'training' },
  Liu2024: { id: 'Liu2024', label: 'Liu2024', subjects: 50, fs: 500, channels: MOTOR_CHANNELS, accuracy: 0.536, role: 'training' },
}

export const DATASET_LIST = Object.values(DATASETS)

// Solo los datasets aptos para la demo en vivo (lo que ofrece el panel lateral).
export const LIVE_DATASET_LIST = DATASET_LIST.filter((d) => d.role === 'live')
