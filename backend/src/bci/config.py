"""Carga de la configuración del proyecto desde un YAML.

Centraliza el acceso a `configs/default.yaml` para que todos los módulos
(scripts, pipeline, tests) compartan los mismos parámetros sin hardcodearlos.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

# Raíz de `backend/` = dos niveles por encima de este archivo (src/bci/config.py).
BACKEND_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = BACKEND_ROOT / "configs" / "default.yaml"


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Lee el YAML de configuración y lo devuelve como diccionario.

    Parameters
    ----------
    path
        Ruta al YAML. Si es ``None`` se usa ``configs/default.yaml``.
    """
    cfg_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    if not cfg_path.exists():
        raise FileNotFoundError(f"No se encontró el archivo de configuración: {cfg_path}")
    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_path(relative: str) -> Path:
    """Convierte una ruta relativa del YAML (p. ej. 'data/raw') en absoluta.

    Las rutas del YAML se interpretan relativas a la raíz de ``backend/``.
    """
    p = Path(relative)
    return p if p.is_absolute() else BACKEND_ROOT / p
