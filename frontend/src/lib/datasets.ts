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
}

const MOTOR_CHANNELS = [
  'FC3', 'FC1', 'FCz', 'FC2', 'FC4',
  'C3', 'C1', 'Cz', 'C2', 'C4',
  'CP3', 'CPz', 'CP4', 'Pz',
]

export const DATASETS: Record<DatasetId, DatasetInfo> = {
  BNCI2014_001: { id: 'BNCI2014_001', label: 'BCI IV 2a', subjects: 9, fs: 250, channels: MOTOR_CHANNELS, accuracy: 0.688 },
  PhysionetMI: { id: 'PhysionetMI', label: 'PhysioNet MMI', subjects: 109, fs: 160, channels: MOTOR_CHANNELS, accuracy: 0.608 },
  Liu2024: { id: 'Liu2024', label: 'Liu2024', subjects: 50, fs: 500, channels: MOTOR_CHANNELS, accuracy: 0.536 },
}

export const DATASET_LIST = Object.values(DATASETS)
