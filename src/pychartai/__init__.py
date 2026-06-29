"""
pychartai — AI-powered data analysis and chart generation.

Single public API.  Everything a user needs is available directly under
``import pychartai as pai``.  The ``pychartai_core`` sub-package is an
implementation detail and should not be imported directly.

Quick start::

    import pychartai as pai

    llm = pai.OllamaLLM(model='llama3.2')
    pai.config.set({'llm': llm})

    df = pai.read_csv('data/sales.csv')
    print(df.chat('What is the average revenue by region?'))
    df.chat('Plot a bar chart of revenue by region', chart_library='plotly')

    # Data profiling — no LLM needed
    report = df.profile()
    print(report.summary)

    # Conversation memory
    df.enable_memory()
    df.chat('Total sales by region')
    df.chat('Now show that as a percentage')
"""

from __future__ import annotations

import sys
from typing import Any, Optional
import pandas as pd

# ---------------------------------------------------------------------------
# Core imports from implementation package
# ---------------------------------------------------------------------------

from pychartai_core import config as global_config
from pychartai_core.providers import (
    PyChartLLM,
    OllamaLLM,
    GitHubLLM,
    OpenAILLM,
    GeminiLLM,
    AnthropicLLM,
    QwenLLM,
    DeepSeekLLM,
    GenericLLM,
    PandasAILLM,
)
from pychartai_core.smart_df import SmartDataFrame
from pychartai_core.agent import PyChartAgent, TransformationLog, QueryIntent
from pychartai_core.sandbox import RestrictedSandbox, DockerSandbox
from pychartai_core.streaming import StreamEvent
from pychartai_core.reporter import InsightReporter
from pychartai_core.skills import Skill, skill
from pychartai_core.schema import Schema, Column
from pychartai_core.cache import ResponseCache
from pychartai_core.memory import ConversationMemory, Turn
from pychartai_core.profiler import DataProfiler, ProfileReport
from pychartai_core.themes import ChartTheme, resolve_theme, BUILTIN_THEMES
from pychartai_core.error_hints import get_hint, format_error_with_hint
from pychartai_core.redactor import DataRedactor
from pychartai_core.logging import (
    get_logger, configure_logger, set_log_level, suppress_pandasai_logging,
)
from pychartai_core.pipeline import (
    Pipeline, PipelineStep, PipelineContext,
    ValidateInput, InjectSchema, InjectSkills,
    CacheLookup, CallAnalyzer, CallOwnAgent, CacheStore,
    default_pipeline,
)
from pychartai_core.connections import (
    BaseConnection, CSVConnection, ExcelConnection,
    JSONConnection, ParquetConnection, SQLConnection,
    connect,
)
from pychartai_core.db_connectors import (
    PostgreSQLConnection,
    MySQLConnection,
    SnowflakeConnection,
    BigQueryConnection,
    RedshiftConnection,
)
from pychartai_core.cloud_connectors import (
    S3Connection,
    GCSConnection,
    AzureBlobConnection,
    GoogleSheetsConnection,
)
from pychartai_core.visualization import (
    get_chart_helper_names,
    list_backends,
    area_chart,
    bar_chart,
    box_chart,
    bubble_chart,
    count_chart,
    ecdf_chart,
    funnel_chart,
    heatmap,
    histogram,
    kde_chart,
    line_chart,
    pairplot_chart,
    pie_chart,
    regression_chart,
    scatter_chart,
    stacked_bar_chart,
    step_chart,
    strip_chart,
    swarm_chart,
    violin_chart,
)

# pandasai-dependent imports are guarded — only available when
# pychartai[pandasai] extra is installed.
try:
    from pychartai_core.analyzer import DataAnalyzer, CustomLLM
    _PANDASAI_AVAILABLE = True
except ImportError:
    _PANDASAI_AVAILABLE = False
    DataAnalyzer = None   # type: ignore[assignment,misc]
    CustomLLM = None      # type: ignore[assignment,misc]

__version__ = '0.4.0'

# Expose global config singleton under the expected name
config = global_config


# ---------------------------------------------------------------------------
# Convenience helpers
# ---------------------------------------------------------------------------

def available_backends() -> tuple[str, ...]:
    """Return the names of all supported chart backends."""
    return tuple(list_backends())


def available_charts(chart_library: Optional[str] = None) -> tuple[str, ...]:
    """Return all supported chart helper names, optionally filtered by backend."""
    return get_chart_helper_names(chart_library)


# ---------------------------------------------------------------------------
# SmartDataFrame alias — canonical class is SmartDataFrame (capital F).
# SmartDataframe (lower-case f) is kept as a backward-compatible alias.
# ---------------------------------------------------------------------------

# SmartDataFrame is already imported above; re-export the alias
SmartDataframe = SmartDataFrame   # backward-compat alias


# ---------------------------------------------------------------------------
# I/O helpers — return SmartDataFrame directly
# ---------------------------------------------------------------------------

def DataFrame(
    data: Any = None,
    *,
    config: Optional[dict[str, Any]] = None,
    chart_library: Optional[str] = None,
    **kwargs: Any,
) -> SmartDataFrame:
    """Create a :class:`SmartDataFrame` from any pandas-compatible input.

    Args:
        data:          Data passed through to :class:`pandas.DataFrame`.
        config:        Optional dict of pychartai config overrides to apply.
        chart_library: Default chart backend for this instance.
        **kwargs:      Forwarded to :class:`pandas.DataFrame`.
    """
    if isinstance(data, pd.DataFrame):
        raw = data
    elif isinstance(data, SmartDataFrame):
        raw = data._df
    else:
        raw = pd.DataFrame(data, **kwargs)
    sdf = SmartDataFrame(raw)
    if chart_library:
        global_config.set({'chart_backend': chart_library})
    if config:
        normalized = dict(config)
        if 'chart_library' in normalized and 'chart_backend' not in normalized:
            normalized['chart_backend'] = normalized.pop('chart_library')
        global_config.set(normalized)
    return sdf


def read_csv(
    filepath_or_buffer: Any,
    *,
    config: Optional[dict[str, Any]] = None,
    chart_library: Optional[str] = None,
    **kwargs: Any,
) -> SmartDataFrame:
    """Read a CSV file and return a :class:`SmartDataFrame`."""
    return DataFrame(pd.read_csv(filepath_or_buffer, **kwargs),
                     config=config, chart_library=chart_library)


def read_excel(
    filepath_or_buffer: Any,
    *,
    config: Optional[dict[str, Any]] = None,
    chart_library: Optional[str] = None,
    **kwargs: Any,
) -> SmartDataFrame:
    """Read an Excel file and return a :class:`SmartDataFrame`."""
    return DataFrame(pd.read_excel(filepath_or_buffer, **kwargs),
                     config=config, chart_library=chart_library)


def read_json(
    filepath_or_buffer: Any,
    *,
    config: Optional[dict[str, Any]] = None,
    chart_library: Optional[str] = None,
    **kwargs: Any,
) -> SmartDataFrame:
    """Read a JSON file and return a :class:`SmartDataFrame`."""
    return DataFrame(pd.read_json(filepath_or_buffer, **kwargs),
                     config=config, chart_library=chart_library)


def read_parquet(
    filepath_or_buffer: Any,
    *,
    config: Optional[dict[str, Any]] = None,
    chart_library: Optional[str] = None,
    **kwargs: Any,
) -> SmartDataFrame:
    """Read a Parquet file and return a :class:`SmartDataFrame`."""
    return DataFrame(pd.read_parquet(filepath_or_buffer, **kwargs),
                     config=config, chart_library=chart_library)


# ---------------------------------------------------------------------------
# Module-level chat() — multi-DataFrame queries
# ---------------------------------------------------------------------------

def _normalize_df(value: Any) -> pd.DataFrame:
    """Coerce a SmartDataFrame or plain DataFrame to pd.DataFrame."""
    if isinstance(value, SmartDataFrame):
        return object.__getattribute__(value, '_df')
    if isinstance(value, pd.DataFrame):
        return value
    raise TypeError(
        f'pychartai.chat() expects pandas DataFrame or SmartDataFrame, got {type(value).__name__}'
    )


def _resolve_llm_for_agent(llm: Any, chart_backend: str) -> Any:
    """Resolve a pychartai LLM wrapper to a pandasai-compatible LLM object."""
    if isinstance(llm, PandasAILLM):
        return llm.get_inner()
    if isinstance(llm, (OllamaLLM, GitHubLLM, OpenAILLM, GeminiLLM,
                        AnthropicLLM, QwenLLM, DeepSeekLLM, GenericLLM)):
        if CustomLLM is None:
            raise ImportError(
                'pandasai is required to use the module-level chat() function '
                'without a sandbox.  Install it with: pip install pychartai[pandasai]\n'
                'Or use SmartDataFrame.chat() with the default agent.'
            )
        provider = llm._get_provider()
        return CustomLLM(provider, chart_backend=chart_backend)
    return llm


def chat(
    query: str,
    *dataframes: Any,
    sandbox: Optional[Any] = None,
    chart_library: Optional[str] = None,
    plot_type: Optional[str] = None,
    chart_options: Optional[dict[str, Any]] = None,
    **chart_kwargs: Any,
) -> str:
    """Run a natural-language query over one or more DataFrames.

    When called without a *sandbox*, the ``pandasai`` extra is required.
    To use the standalone (no-pandasai) path pass an explicit sandbox::

        pai.chat('Who earns the most?', df, sandbox=pai.RestrictedSandbox())

    Or call ``.chat()`` directly on a :class:`SmartDataFrame` (recommended)::

        df = pai.read_csv('data.csv')
        df.chat('Who earns the most?')

    Args:
        query:        Natural-language question.
        *dataframes:  One or more :class:`pandas.DataFrame` or
                      :class:`SmartDataFrame` objects.
        sandbox:      Optional :class:`RestrictedSandbox` or
                      :class:`DockerSandbox`.  When provided, pandasai is
                      **not** required.
        chart_library: Override the visualization backend
                       ('seaborn', 'matplotlib', 'plotly').
        plot_type:    Chart type hint (e.g. 'bar', 'line').
        chart_options: Extra chart keyword options.
        **chart_kwargs: Additional chart parameters.

    Returns:
        Analysis result as a string, or a chart file path.

    Raises:
        ValueError:   If no DataFrames are passed.
        RuntimeError: If no LLM is configured.
        ImportError:  If pandasai is absent and no sandbox is provided.
    """
    if not dataframes:
        raise ValueError('pychartai.chat() requires at least one DataFrame.')

    llm = config.get('llm')
    if llm is None:
        raise RuntimeError(
            "No LLM configured. Call pai.config.set({'llm': pai.OllamaLLM(model='llama3.2')}) first."
        )

    backend = chart_library or config.get('chart_backend', 'seaborn')
    output_dir = config.get('charts_output_dir', 'exports/charts')
    directive = SmartDataFrame._build_chart_directive(plot_type, chart_options, chart_kwargs)
    effective_query = query + directive
    frames = [_normalize_df(item) for item in dataframes]

    if sandbox is not None:
        from pychartai_core.analyzer import DataAnalyzer as _DA
        analyzer = _DA(charts_output_dir=output_dir, chart_backend=backend, llm=llm)
        if len(frames) == 1:
            return analyzer.analyze(frames[0], effective_query, sandbox=sandbox)
        return analyzer._analyze_with_sandbox_multi(effective_query, frames, backend, sandbox)

    resolved_llm = _resolve_llm_for_agent(llm, chart_backend=backend)

    try:
        from pandasai import Agent as _PandasAIAgent
    except ImportError as exc:
        raise ImportError(
            'The module-level chat() function requires pandasai when no sandbox is '
            'provided.  Install it with: pip install pychartai[pandasai]\n'
            'Or use SmartDataFrame.chat() for the standalone agent path.'
        ) from exc

    agent = _PandasAIAgent(
        frames,
        config={
            'llm': resolved_llm,
            'verbose': config.get('verbose', False),
            'save_charts': False,
            'open_charts': False,
        },
        memory_size=config.get('memory_size', 100),
    )
    result = agent.chat(effective_query)
    if hasattr(result, 'value'):
        return str(result.value)
    return str(result)


async def achat(
    query: str,
    *dataframes: Any,
    sandbox: Optional[Any] = None,
    chart_library: Optional[str] = None,
    plot_type: Optional[str] = None,
    chart_options: Optional[dict[str, Any]] = None,
    **chart_kwargs: Any,
) -> str:
    """Async wrapper around :func:`chat`.

    Runs the synchronous call in a thread-pool executor so it can be awaited
    without blocking the event loop::

        result = await pai.achat('Who earns the most?', df)
    """
    import asyncio
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None,
        lambda: chat(
            query,
            *dataframes,
            sandbox=sandbox,
            chart_library=chart_library,
            plot_type=plot_type,
            chart_options=chart_options,
            **chart_kwargs,
        ),
    )


# ---------------------------------------------------------------------------
# Public API surface
# ---------------------------------------------------------------------------

__all__ = [
    '__version__',
    # Config
    'config',
    # LLM providers
    'PyChartLLM',
    'OllamaLLM',
    'GitHubLLM',
    'OpenAILLM',
    'GeminiLLM',
    'AnthropicLLM',
    'QwenLLM',
    'DeepSeekLLM',
    'GenericLLM',
    'PandasAILLM',
    # DataFrame
    'SmartDataFrame',
    'SmartDataframe',        # backward-compat alias
    'DataFrame',
    'read_csv',
    'read_excel',
    'read_json',
    'read_parquet',
    # Agents
    'PyChartAgent',
    'TransformationLog',
    'QueryIntent',
    # Module-level chat
    'chat',
    'achat',
    # Sandbox & streaming
    'RestrictedSandbox',
    'DockerSandbox',
    'StreamEvent',
    # Memory
    'ConversationMemory',
    'Turn',
    # Profiling
    'DataProfiler',
    'ProfileReport',
    # Themes
    'ChartTheme',
    'resolve_theme',
    'BUILTIN_THEMES',
    # Skills
    'Skill',
    'skill',
    # Semantic layer
    'Schema',
    'Column',
    # Cache
    'ResponseCache',
    # PII redaction
    'DataRedactor',
    # Error hints
    'get_hint',
    'format_error_with_hint',
    # Logging
    'get_logger',
    'configure_logger',
    'set_log_level',
    'suppress_pandasai_logging',
    # Pipeline
    'Pipeline',
    'PipelineStep',
    'PipelineContext',
    'ValidateInput',
    'InjectSchema',
    'InjectSkills',
    'CacheLookup',
    'CallAnalyzer',
    'CallOwnAgent',
    'CacheStore',
    'default_pipeline',
    # Connections
    'BaseConnection',
    'CSVConnection',
    'ExcelConnection',
    'JSONConnection',
    'ParquetConnection',
    'SQLConnection',
    'connect',
    # Database connectors
    'PostgreSQLConnection',
    'MySQLConnection',
    'SnowflakeConnection',
    'BigQueryConnection',
    'RedshiftConnection',
    # Cloud storage connectors
    'S3Connection',
    'GCSConnection',
    'AzureBlobConnection',
    'GoogleSheetsConnection',
    # Chart helpers (all 20)
    'area_chart',
    'bar_chart',
    'box_chart',
    'bubble_chart',
    'count_chart',
    'ecdf_chart',
    'funnel_chart',
    'heatmap',
    'histogram',
    'kde_chart',
    'line_chart',
    'pairplot_chart',
    'pie_chart',
    'regression_chart',
    'scatter_chart',
    'stacked_bar_chart',
    'step_chart',
    'strip_chart',
    'swarm_chart',
    'violin_chart',
    # Utilities
    'available_backends',
    'available_charts',
    'get_hint',
    'format_error_with_hint',
    # Reporting
    'InsightReporter',
    # pandasai integration (None when pandasai not installed)
    'DataAnalyzer',
    'CustomLLM',
]
