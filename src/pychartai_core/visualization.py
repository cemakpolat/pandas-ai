"""
Plugin-oriented visualization facade for LLM-generated code.

The public chart helpers remain stable for LLM-generated imports:

    from pychartai_core.visualization import bar_chart

Internally, rendering is delegated to registered backend implementations.
This keeps the LLM-facing API small while making backend support modular.
"""

from __future__ import annotations

import pandas as pd

from .visualization_backends import (
    get_backend,
    get_backend_chart_specs,
    list_backends,
    list_chart_specs,
    register_backend,
    register_chart_spec,
)
from .visualization_backends.utils import ensure_dir


def get_chart_helper_names(backend: str | None = None) -> tuple[str, ...]:
    """Return chart helper names, optionally filtered by backend."""
    specs = get_backend_chart_specs(backend) if backend else list_chart_specs()
    return tuple(spec.helper_name for spec in specs)


def build_import_statement(backend: str | None = None) -> str:
    """Build an explicit import statement for chart helpers."""
    helpers = ", ".join(get_chart_helper_names(backend))
    return f"from pychartai_core.visualization import {helpers}"


def render_chart(
    chart_type: str,
    df: pd.DataFrame,
    *,
    backend: str = "seaborn",
    output_file: str = "exports/charts/chart.png",
    theme: str | None = None,
    **kwargs,
) -> str:
    """Render *chart_type* through the selected backend.

    Args:
        theme: Optional theme name ('light', 'dark', 'corporate', 'minimal')
               or a :class:`~pychartai_core.themes.ChartTheme` instance.
    """
    from .themes import resolve_theme

    # Apply theme if provided; also check global config
    if theme is None:
        from .config import config as global_config
        theme = global_config.get('chart_theme')
    resolved_theme = resolve_theme(theme)

    if backend == 'plotly':
        # Plotly theme applied after render via fig.update_layout
        pass
    elif backend == 'seaborn':
        resolved_theme.apply_seaborn()
    else:
        resolved_theme.apply_matplotlib()

    ensure_dir(output_file)
    implementation = get_backend(backend)
    return implementation.render(chart_type=chart_type, df=df, output_file=output_file, **kwargs)


def describe_backend(backend: str) -> tuple[str, ...]:
    """Return human-readable chart capability lines for *backend*."""
    lines: list[str] = []
    for spec in get_backend_chart_specs(backend):
        required = ", ".join(spec.required_args) if spec.required_args else "no positional args"
        optional = ", ".join(spec.optional_args) if spec.optional_args else "none"
        lines.append(
            f"{spec.helper_name}({required}) optional=[{optional}] strategy={spec.sql_strategy} - {spec.description}"
        )
    return tuple(lines)


def bar_chart(
    df: pd.DataFrame,
    x: str,
    y: str,
    *,
    title: str = "",
    output_file: str = "exports/charts/chart.png",
    hue: str | None = None,
    backend: str = "seaborn",
    **kwargs,
) -> str:
    return render_chart("bar", df, x=x, y=y, title=title, output_file=output_file, hue=hue, backend=backend, **kwargs)


def line_chart(
    df: pd.DataFrame,
    x: str,
    y: str,
    *,
    title: str = "",
    output_file: str = "exports/charts/chart.png",
    hue: str | None = None,
    backend: str = "seaborn",
    **kwargs,
) -> str:
    return render_chart("line", df, x=x, y=y, title=title, output_file=output_file, hue=hue, backend=backend, **kwargs)


def scatter_chart(
    df: pd.DataFrame,
    x: str,
    y: str,
    *,
    title: str = "",
    output_file: str = "exports/charts/chart.png",
    hue: str | None = None,
    size: str | None = None,
    backend: str = "seaborn",
    **kwargs,
) -> str:
    return render_chart("scatter", df, x=x, y=y, title=title, output_file=output_file, hue=hue, size=size, backend=backend, **kwargs)


def histogram(
    df: pd.DataFrame,
    column: str,
    *,
    title: str = "",
    output_file: str = "exports/charts/chart.png",
    bins: int = 20,
    hue: str | None = None,
    backend: str = "seaborn",
    **kwargs,
) -> str:
    return render_chart("histogram", df, column=column, title=title, output_file=output_file, bins=bins, hue=hue, backend=backend, **kwargs)


def heatmap(
    df: pd.DataFrame,
    *,
    title: str = "Correlation Heatmap",
    output_file: str = "exports/charts/chart.png",
    backend: str = "seaborn",
    **kwargs,
) -> str:
    return render_chart("heatmap", df, title=title, output_file=output_file, backend=backend, **kwargs)


def box_chart(
    df: pd.DataFrame,
    x: str,
    y: str,
    *,
    title: str = "",
    output_file: str = "exports/charts/chart.png",
    hue: str | None = None,
    backend: str = "seaborn",
    **kwargs,
) -> str:
    return render_chart("box", df, x=x, y=y, title=title, output_file=output_file, hue=hue, backend=backend, **kwargs)


def violin_chart(
    df: pd.DataFrame,
    x: str,
    y: str,
    *,
    title: str = "",
    output_file: str = "exports/charts/chart.png",
    hue: str | None = None,
    backend: str = "seaborn",
    **kwargs,
) -> str:
    return render_chart("violin", df, x=x, y=y, title=title, output_file=output_file, hue=hue, backend=backend, **kwargs)


def pie_chart(
    df: pd.DataFrame,
    labels: str,
    values: str,
    *,
    title: str = "",
    output_file: str = "exports/charts/chart.png",
    backend: str = "seaborn",
    **kwargs,
) -> str:
    return render_chart("pie", df, labels=labels, values=values, title=title, output_file=output_file, backend=backend, **kwargs)


def area_chart(
    df: pd.DataFrame,
    x: str,
    y: str,
    *,
    title: str = "",
    output_file: str = "exports/charts/chart.png",
    hue: str | None = None,
    backend: str = "seaborn",
    **kwargs,
) -> str:
    return render_chart("area", df, x=x, y=y, title=title, output_file=output_file, hue=hue, backend=backend, **kwargs)


def count_chart(
    df: pd.DataFrame,
    x: str,
    *,
    title: str = "",
    output_file: str = "exports/charts/chart.png",
    hue: str | None = None,
    backend: str = "seaborn",
    **kwargs,
) -> str:
    return render_chart("count", df, x=x, title=title, output_file=output_file, hue=hue, backend=backend, **kwargs)


def kde_chart(
    df: pd.DataFrame,
    column: str,
    *,
    title: str = "",
    output_file: str = "exports/charts/chart.png",
    hue: str | None = None,
    backend: str = "seaborn",
    **kwargs,
) -> str:
    return render_chart("kde", df, column=column, title=title, output_file=output_file, hue=hue, backend=backend, **kwargs)


def strip_chart(
    df: pd.DataFrame,
    x: str,
    y: str,
    *,
    title: str = "",
    output_file: str = "exports/charts/chart.png",
    hue: str | None = None,
    backend: str = "seaborn",
    **kwargs,
) -> str:
    return render_chart("strip", df, x=x, y=y, title=title, output_file=output_file, hue=hue, backend=backend, **kwargs)


def regression_chart(
    df: pd.DataFrame,
    x: str,
    y: str,
    *,
    title: str = "",
    output_file: str = "exports/charts/chart.png",
    hue: str | None = None,
    backend: str = "seaborn",
    **kwargs,
) -> str:
    return render_chart("regression", df, x=x, y=y, title=title, output_file=output_file, hue=hue, backend=backend, **kwargs)


def pairplot_chart(
    df: pd.DataFrame,
    *,
    title: str = "",
    output_file: str = "exports/charts/chart.png",
    hue: str | None = None,
    columns: list | None = None,
    backend: str = "seaborn",
    **kwargs,
) -> str:
    return render_chart("pairplot", df, title=title, output_file=output_file, hue=hue, columns=columns, backend=backend, **kwargs)


def stacked_bar_chart(
    df: pd.DataFrame,
    x: str,
    y: str,
    stack: str,
    *,
    title: str = "",
    output_file: str = "exports/charts/chart.png",
    normalize: bool = False,
    backend: str = "seaborn",
    **kwargs,
) -> str:
    return render_chart("stacked_bar", df, x=x, y=y, stack=stack, title=title, output_file=output_file, normalize=normalize, backend=backend, **kwargs)


def bubble_chart(
    df: pd.DataFrame,
    x: str,
    y: str,
    size: str,
    *,
    title: str = "",
    output_file: str = "exports/charts/chart.png",
    hue: str | None = None,
    backend: str = "seaborn",
    **kwargs,
) -> str:
    return render_chart("bubble", df, x=x, y=y, size=size, title=title, output_file=output_file, hue=hue, backend=backend, **kwargs)


def step_chart(
    df: pd.DataFrame,
    x: str,
    y: str,
    *,
    title: str = "",
    output_file: str = "exports/charts/chart.png",
    hue: str | None = None,
    backend: str = "seaborn",
    **kwargs,
) -> str:
    return render_chart("step", df, x=x, y=y, title=title, output_file=output_file, hue=hue, backend=backend, **kwargs)


def swarm_chart(
    df: pd.DataFrame,
    x: str,
    y: str,
    *,
    title: str = "",
    output_file: str = "exports/charts/chart.png",
    hue: str | None = None,
    backend: str = "seaborn",
    **kwargs,
) -> str:
    return render_chart("swarm", df, x=x, y=y, title=title, output_file=output_file, hue=hue, backend=backend, **kwargs)


def ecdf_chart(
    df: pd.DataFrame,
    column: str,
    *,
    title: str = "",
    output_file: str = "exports/charts/chart.png",
    hue: str | None = None,
    backend: str = "seaborn",
    **kwargs,
) -> str:
    return render_chart("ecdf", df, column=column, title=title, output_file=output_file, hue=hue, backend=backend, **kwargs)


def funnel_chart(
    df: pd.DataFrame,
    labels: str,
    values: str,
    *,
    title: str = "",
    output_file: str = "exports/charts/chart.png",
    backend: str = "seaborn",
    **kwargs,
) -> str:
    return render_chart("funnel", df, labels=labels, values=values, title=title, output_file=output_file, backend=backend, **kwargs)


__all__ = [
    "area_chart",
    "bar_chart",
    "build_import_statement",
    "box_chart",
    "bubble_chart",
    "count_chart",
    "describe_backend",
    "ecdf_chart",
    "funnel_chart",
    "get_backend",
    "get_chart_helper_names",
    "heatmap",
    "histogram",
    "kde_chart",
    "line_chart",
    "list_backends",
    "list_chart_specs",
    "pairplot_chart",
    "pie_chart",
    "register_backend",
    "register_chart_spec",
    "regression_chart",
    "render_chart",
    "scatter_chart",
    "stacked_bar_chart",
    "step_chart",
    "strip_chart",
    "swarm_chart",
    "violin_chart",
]
