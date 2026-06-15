// Cliente del backend FastAPI (Etapa 2). En dev, Vite hace de proxy de /api y /ws.

const API_BASE = '/api'

export async function getJSON<T>(path: string): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`)
  if (!res.ok) throw new Error(`HTTP ${res.status} en ${path}`)
  return res.json() as Promise<T>
}

/** Abre un WebSocket al backend (p. ej. para la simulación en vivo). */
export function openStream(path: string, onMessage: (data: unknown) => void): WebSocket {
  const proto = location.protocol === 'https:' ? 'wss' : 'ws'
  const ws = new WebSocket(`${proto}://${location.host}/ws${path}`)
  ws.onmessage = (e) => {
    try { onMessage(JSON.parse(e.data)) } catch { /* ignora frames no-JSON */ }
  }
  return ws
}
