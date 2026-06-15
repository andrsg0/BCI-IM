"""Arranca el servidor API del backend (FastAPI + uvicorn).

Uso:
    python scripts/run_server.py            # http://localhost:8000
    python scripts/run_server.py --reload   # recarga en caliente (desarrollo)

El frontend (Vite, :5173) hace de proxy de /api y /ws hacia este servidor.
"""
from __future__ import annotations

import argparse

import uvicorn


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--host', default='0.0.0.0')
    ap.add_argument('--port', type=int, default=8000)
    ap.add_argument('--reload', action='store_true')
    args = ap.parse_args()
    uvicorn.run('bci.server.app:app', host=args.host, port=args.port, reload=args.reload)


if __name__ == '__main__':
    main()
