"""
pychartai: AI-powered data analysis with 20 chart types, 3 backends, and PandasAI-compatible features.

Public API (new)::

    import pychartai as pai

    llm = pai.OllamaLLM(model="llama3.2")
    pai.config.set({"llm": llm})

    df = pai.read_csv("data/sales.csv")
    print(df.chat("What is the average revenue by region?"))
    df.chat("Plot bar chart of revenue by region", chart_type="plotly")

Legacy API (unchanged)::

    from pychartai import DataAnalyzer, DataManager
    analyzer = DataAnalyzer(model_name="llama3.2")
"""

# ---- Initialize pychartai logging (overrides PandasAI logging) ----
from .logging import get_logger, configure_logger, set_log_level, suppress_pandasai_logging
try:
    suppress_pandasai_logging()
except Exception:
    # Suppress error if pandasai not installed
    pass
_pychartai_logger = get_logger()

# ---- Legacy / internal API ----
# CustomLLM and DataAnalyzer require pandasai; make imports conditional
try:
    from .analyzer import DataAnalyzer, CustomLLM
except ImportError:
    # pandasai not installed; provide stub that raises helpful error
    class CustomLLM:
        def __init__(self, *args, **kwargs):
            raise ImportError(
                "CustomLLM requires pandasai. Install it with: pip install pychartai[pandasai]"
            )
    class DataAnalyzer:
        def __init__(self, *args, **kwargs):
            raise ImportError(
                "DataAnalyzer requires pandasai. Install it with: pip install pychartai[pandasai]"
            )

from .data_manager import DataManager
from .model_manager import LLMProvider, LiteLLMProvider

# ---- New public API ----
from .config import config                                            # GlobalConfig singleton
from .providers import (                                             # LLM wrappers
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
from .smart_df import SmartDataFrame                                  # Smart DataFrame
from .agent import PyChartAgent                                       # Own agent (pandasai-independent)
from .io import read_csv, read_excel, read_json, read_parquet, DataFrame  # I/O helpers

# ---- Advanced features ----
from .skills import Skill, skill                                      # Skills
from .schema import Schema, Column                                    # Semantic layer
from .cache import ResponseCache                                      # Response cache
from .pipeline import (                                               # Pipeline
    Pipeline, PipelineStep, PipelineContext,
    ValidateInput, InjectSchema, InjectSkills,
    CacheLookup, CallAnalyzer, CallOwnAgent, CacheStore,
    default_pipeline,
)
from .connections import (                                            # Data connectors
    BaseConnection, CSVConnection, ExcelConnection,
    JSONConnection, ParquetConnection, SQLConnection,
    connect,
)
from .sandbox import RestrictedSandbox, DockerSandbox                 # Sandboxes
from .streaming import StreamEvent                                    # Streaming types

# ---- Phase 3+ features ----
from .memory import ConversationMemory, Turn                          # Conversation memory
from .profiler import DataProfiler, ProfileReport                     # Data profiling
from .themes import ChartTheme, resolve_theme, BUILTIN_THEMES         # Themes
from .error_hints import get_hint, format_error_with_hint             # Error hints
from .agent import TransformationLog                                  # Transformation tracking

__version__ = "0.4.0"

__all__ = [
    # --- Logging ---
    "get_logger",
    "configure_logger",
    "set_log_level",
    # --- Legacy ---
    "DataAnalyzer",
    "CustomLLM",
    "DataManager",
    "LLMProvider",
    "LiteLLMProvider",
    # --- New public API ---
    "config",
    "PyChartLLM",
    "OllamaLLM",
    "GitHubLLM",
    "OpenAILLM",
    "GeminiLLM",
    "AnthropicLLM",
    "QwenLLM",
    "DeepSeekLLM",
    "GenericLLM",
    "PandasAILLM",
    "SmartDataFrame",
    "read_csv",
    "read_excel",
    "read_json",
    "read_parquet",
    "DataFrame",
    # --- Skills ---
    "Skill",
    "skill",
    # --- Semantic Layer ---
    "Schema",
    "Column",
    # --- Cache ---
    "ResponseCache",
    # --- Pipeline ---
    "Pipeline",
    "PipelineStep",
    "PipelineContext",
    "ValidateInput",
    "InjectSchema",
    "InjectSkills",
    "CacheLookup",
    "CallAnalyzer",
    "CallOwnAgent",
    "CacheStore",
    "default_pipeline",
    # --- Connections ---
    "BaseConnection",
    "CSVConnection",
    "ExcelConnection",
    "JSONConnection",
    "ParquetConnection",
    "SQLConnection",
    "connect",
    # --- Sandbox ---
    "RestrictedSandbox",
    "DockerSandbox",
    # --- Streaming ---
    "StreamEvent",
    # --- Conversation memory ---
    "ConversationMemory",
    "Turn",
    # --- Data profiling ---
    "DataProfiler",
    "ProfileReport",
    # --- Themes ---
    "ChartTheme",
    "resolve_theme",
    "BUILTIN_THEMES",
    # --- Error hints ---
    "get_hint",
    "format_error_with_hint",
    # --- Transformation tracking ---
    "TransformationLog",
]
