from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Iterable, Literal

import pandas as pd


@dataclass(frozen=True)
class ChartSpec:
    """Structured description of a chart helper exposed to the LLM."""

    name: str
    helper_name: str
    required_args: tuple[str, ...]
    optional_args: tuple[str, ...] = ()
    sql_strategy: Literal["aggregate", "raw", "derived"] = "raw"
    description: str = ""


class ChartBackend(ABC):
    """Base interface for chart rendering backends."""

    name: str
    supported_charts: tuple[str, ...] = ()

    @abstractmethod
    def render(self, chart_type: str, df: pd.DataFrame, output_file: str, **kwargs) -> str:
        """Render *chart_type* for *df* into *output_file*."""

    def get_chart_specs(self) -> tuple[ChartSpec, ...]:
        """Return chart specifications supported by this backend."""
        return tuple(get_chart_spec(chart_name) for chart_name in self.supported_charts)


_BACKENDS: Dict[str, ChartBackend] = {}
_CHART_SPECS: Dict[str, ChartSpec] = {}


def register_backend(backend: ChartBackend) -> None:
    """Register a chart backend implementation."""
    _BACKENDS[backend.name] = backend


def register_chart_spec(spec: ChartSpec) -> None:
    """Register a chart specification."""
    _CHART_SPECS[spec.name] = spec


def get_backend(name: str) -> ChartBackend:
    """Return a registered chart backend by name."""
    try:
        return _BACKENDS[name]
    except KeyError as exc:
        supported = ", ".join(sorted(_BACKENDS)) or "none"
        raise ValueError(
            f"Unsupported chart backend: {name}. Supported values: {supported}"
        ) from exc


def get_chart_spec(name: str) -> ChartSpec:
    """Return a registered chart specification by name."""
    try:
        return _CHART_SPECS[name]
    except KeyError as exc:
        supported = ", ".join(sorted(_CHART_SPECS)) or "none"
        raise ValueError(
            f"Unsupported chart type: {name}. Supported values: {supported}"
        ) from exc


def list_chart_specs() -> tuple[ChartSpec, ...]:
    """List all registered chart specifications."""
    return tuple(_CHART_SPECS[name] for name in sorted(_CHART_SPECS))


def get_backend_chart_specs(name: str) -> tuple[ChartSpec, ...]:
    """List chart specifications supported by a backend."""
    return get_backend(name).get_chart_specs()


def list_backends() -> Iterable[str]:
    """List registered backend names."""
    return tuple(sorted(_BACKENDS))