"""Tests for pychartai-compatible public API behavior."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pandas as pd

from pychartai_core.config import config
from pychartai_core.smart_df import SmartDataFrame
import pychartai_core.analyzer as analyzer_module
from pychartai_core.visualization import (
    build_import_statement,
    describe_backend,
    get_chart_helper_names,
    list_backends,
    list_chart_specs,
    register_backend,
    render_chart,
)
from pychartai_core.visualization_backends import ChartBackend

import pychartai as chartai


class _FakeAnalyzer:
    last_init = None
    last_query = None

    def __init__(self, *args, **kwargs):
        _FakeAnalyzer.last_init = kwargs

    def analyze(self, df, query, **kwargs):
        _FakeAnalyzer.last_query = query
        return "ok"


def test_smart_df_accepts_chart_library_and_plot_directives(monkeypatch):
    monkeypatch.setattr(analyzer_module, "DataAnalyzer", _FakeAnalyzer)

    config.reset()
    config.set({"llm": object(), "chart_backend": "seaborn"})

    sdf = SmartDataFrame(pd.DataFrame({"x": [1, 2], "y": [3, 4]}))
    result = sdf.chat(
        "Create a chart",
        chart_library="plotly",
        plot_type="scatter",
        chart_options={"hue": "region"},
        bins=15,
        agent='sandbox',
    )

    assert result == "ok"
    assert _FakeAnalyzer.last_init["chart_backend"] == "plotly"
    assert "Chart directives:" in _FakeAnalyzer.last_query
    assert "plot type: scatter" in _FakeAnalyzer.last_query
    assert "hue='region'" in _FakeAnalyzer.last_query
    assert "bins=15" in _FakeAnalyzer.last_query


def test_chartai_dataframe_uses_chart_library_default(monkeypatch):
    monkeypatch.setattr(analyzer_module, "DataAnalyzer", _FakeAnalyzer)

    config.reset()
    config.set({"llm": object(), "chart_backend": "seaborn"})

    sdf = chartai.DataFrame(
        {"region": ["A", "B"], "revenue": [10, 20]},
        chart_library="matplotlib",
    )

    result = sdf.chat("Show revenue by region", plot_type="bar", agent='sandbox')

    assert result == "ok"
    assert _FakeAnalyzer.last_init["chart_backend"] == "matplotlib"
    assert "plot type: bar" in _FakeAnalyzer.last_query


def test_chartai_config_accepts_chart_library_key(monkeypatch):
    monkeypatch.setattr(analyzer_module, "DataAnalyzer", _FakeAnalyzer)

    config.reset()
    config.set({"llm": object()})

    df = pd.DataFrame({"a": [1], "b": [2]})
    sdf = chartai.SmartDataframe(df, config={"chart_library": "plotly"})

    sdf.chat("plot a vs b", agent='sandbox')

    assert _FakeAnalyzer.last_init["chart_backend"] == "plotly"


class _FakeAgent:
    last_frames = None
    last_config = None
    last_query = None

    def __init__(self, frames, config, memory_size):
        _FakeAgent.last_frames = frames
        _FakeAgent.last_config = config

    def chat(self, query):
        _FakeAgent.last_query = query
        return "agent-ok"


def test_chartai_module_chat_supports_multiple_dataframes(monkeypatch):
    import unittest.mock as _mock
    # Inject a fake pandasai module so the lazy import inside chat() resolves correctly
    fake_pandasai = _mock.MagicMock()
    fake_pandasai.Agent = _FakeAgent
    monkeypatch.setitem(sys.modules, "pandasai", fake_pandasai)
    monkeypatch.setattr(chartai, "_resolve_llm_for_agent", lambda llm, chart_backend: llm)

    raw_llm = object()
    config.reset()
    config.set({"llm": raw_llm, "chart_backend": "seaborn"})

    employees_df = chartai.DataFrame(
        {
            "EmployeeID": [1, 2],
            "Name": ["John", "Emma"],
        }
    )
    salaries_df = chartai.DataFrame(
        {
            "EmployeeID": [1, 2],
            "Salary": [5000, 6000],
        }
    )

    result = chartai.chat(
        "Who gets paid the most?",
        employees_df,
        salaries_df,
        plot_type="bar",
        chart_options={"x": "Name"},
    )

    assert result == "agent-ok"
    assert len(_FakeAgent.last_frames) == 2
    assert _FakeAgent.last_config["llm"] is raw_llm
    assert "plot type: bar" in _FakeAgent.last_query
    assert "x='Name'" in _FakeAgent.last_query


def test_data_analyzer_accepts_raw_pandasai_llm():
    raw_llm = object()
    analyzer = analyzer_module.DataAnalyzer(llm=raw_llm)
    assert analyzer._passthrough_llm is raw_llm


def test_visualization_lists_builtin_backends():
    backends = set(list_backends())
    assert {"seaborn", "matplotlib", "plotly"}.issubset(backends)


def test_visualization_exposes_chart_specs_and_backend_descriptions():
    chart_specs = {spec.name: spec for spec in list_chart_specs()}

    assert chart_specs["bar"].helper_name == "bar_chart"
    assert chart_specs["bar"].sql_strategy == "aggregate"
    assert chart_specs["histogram"].required_args == ("column",)

    backend_lines = describe_backend("seaborn")
    assert any("bar_chart(x, y)" in line for line in backend_lines)
    assert any("strategy=raw" in line for line in backend_lines)


def test_visualization_builds_explicit_import_statement():
    helper_names = get_chart_helper_names("plotly")
    import_statement = build_import_statement("plotly")

    assert "import *" not in import_statement
    assert "bar_chart" in import_statement
    assert set(helper_names) >= {"bar_chart", "scatter_chart", "heatmap"}


class _TestBackend(ChartBackend):
    name = "test-backend"

    def render(self, chart_type, df, output_file, **kwargs):
        assert chart_type == "bar"
        assert list(df.columns) == ["x", "y"]
        return output_file


def test_visualization_allows_custom_backend_registration(tmp_path):
    register_backend(_TestBackend())

    output_path = tmp_path / "custom.txt"
    result = render_chart(
        "bar",
        pd.DataFrame({"x": [1, 2], "y": [3, 4]}),
        backend="test-backend",
        output_file=str(output_path),
    )

    assert result == str(output_path)


class _EchoProvider:
    def generate(self, prompt_str, **kwargs):
        return prompt_str


class _PromptInstruction:
    def to_string(self):
        return "Summarize the data"


def test_custom_llm_builds_chart_prompt_from_registry_metadata():
    llm = analyzer_module.CustomLLM(
        provider=_EchoProvider(),
        chart_backend="plotly",
        charts_output_dir="exports/charts",
    )

    prompt = llm.call(_PromptInstruction())

    assert "Available helpers:" in prompt
    assert "bar_chart" in prompt
    assert "histogram" in prompt
    assert "strategy=aggregate" in prompt
    assert "strategy=derived" in prompt
    assert "backend='plotly'" in prompt
    assert "import *" not in prompt


def test_custom_llm_ensure_result_format_uses_explicit_helper_imports():
    code = analyzer_module.CustomLLM._ensure_result_format(
        "result = 'exports/charts/chart.png'",
        chart_backend="seaborn",
        charts_output_dir="exports/custom-charts",
    )

    assert "from pychartai_core.visualization import" in code
    assert "bar_chart" in code
    assert "import *" not in code
    assert "exports/custom-charts" in code


def test_chartai_exposes_backend_and_chart_discovery():
    assert {"seaborn", "matplotlib", "plotly"}.issubset(set(chartai.available_backends()))
    assert "bar_chart" in chartai.available_charts()
    assert "line_chart" in chartai.available_charts("plotly")
