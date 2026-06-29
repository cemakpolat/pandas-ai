"""tests/test_new_features.py

Tests for features added in v0.4+:
  - on_progress callbacks
  - last_usage token tracking
  - exponential backoff (mocked)
  - select_dtypes string dtype compatibility
"""

import time
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from pychartai_core.agent import PyChartAgent


# ---------------------------------------------------------------------------
# Fixtures / stubs
# ---------------------------------------------------------------------------

class _StubProvider:
    def __init__(self, code: str, usage=None) -> None:
        self._code = code
        self.last_usage = usage or {}

    def generate(self, prompt: str, **kwargs) -> str:
        return f'```python\n{self._code}\n```'


class _StubSandbox:
    """Minimal sandbox stub — executes result = eval(code's RHS) trivially."""

    def execute(self, code: str, context: dict) -> object:
        # Parse 'result = <expr>' and evaluate the expr against context
        for line in code.strip().splitlines():
            if line.startswith('result = ') or line.startswith('result='):
                expr = line.split('=', 1)[1].strip()
                try:
                    return eval(expr, {**context})  # noqa: S307
                except Exception:
                    return expr
        return None


@pytest.fixture()
def sales_df():
    return pd.DataFrame({
        'region': ['North', 'South', 'East', 'West'],
        'revenue': [1000.0, 800.0, 1200.0, 950.0],
        'units': [50, 40, 60, 45],
    })


def _make_agent(code: str, usage=None) -> PyChartAgent:
    provider = _StubProvider(code, usage)
    agent = PyChartAgent.__new__(PyChartAgent)
    agent._provider = provider
    agent._llm = None
    agent._memory = None
    agent.chart_backend = 'seaborn'
    agent.charts_output_dir = 'exports/charts'
    agent.verbose = False
    agent.max_retries = 3
    agent._llm_timeout = 60
    agent.last_transformation = None
    agent.last_usage = {}
    return agent


# ---------------------------------------------------------------------------
# on_progress callbacks
# ---------------------------------------------------------------------------

class TestProgressCallbacks:

    def test_progress_stages_are_fired(self, sales_df):
        agent = _make_agent("result = df['revenue'].mean()")
        stages = []

        def cb(stage, detail):
            stages.append(stage)

        agent.chat("average revenue", sales_df, sandbox=_StubSandbox(), on_progress=cb)

        assert 'classifying' in stages
        assert 'generating' in stages
        assert 'executing' in stages
        assert 'formatting' in stages

    def test_progress_fires_in_order(self, sales_df):
        agent = _make_agent("result = df['revenue'].sum()")
        stages = []

        agent.chat("total revenue", sales_df, sandbox=_StubSandbox(),
                   on_progress=lambda s, d: stages.append(s))

        expected_order = ['classifying', 'generating', 'executing', 'formatting']
        # All expected stages present and in the right relative order
        indices = [stages.index(s) for s in expected_order if s in stages]
        assert indices == sorted(indices)

    def test_crashing_callback_does_not_break_chat(self, sales_df):
        agent = _make_agent("result = 42")

        def bad_callback(stage, detail):
            raise RuntimeError("callback crash")

        # Should not propagate the callback exception
        result = agent.chat("test", sales_df, sandbox=_StubSandbox(), on_progress=bad_callback)
        assert result is not None

    def test_none_callback_is_accepted(self, sales_df):
        agent = _make_agent("result = df['units'].max()")
        result = agent.chat("max units", sales_df, sandbox=_StubSandbox(), on_progress=None)
        assert result is not None


# ---------------------------------------------------------------------------
# Token / cost tracking (last_usage)
# ---------------------------------------------------------------------------

class TestLastUsage:

    def test_last_usage_populated_from_provider(self, sales_df):
        usage = {
            'prompt_tokens': 120,
            'completion_tokens': 45,
            'total_tokens': 165,
            'model': 'openai/gpt-4o',
        }
        agent = _make_agent("result = df['revenue'].mean()", usage=usage)
        agent.chat("mean revenue", sales_df, sandbox=_StubSandbox())

        assert agent.last_usage['prompt_tokens'] == 120
        assert agent.last_usage['completion_tokens'] == 45
        assert agent.last_usage['total_tokens'] == 165

    def test_last_usage_empty_for_local_models(self, sales_df):
        # Local models (Ollama) don't return usage — last_usage stays empty
        agent = _make_agent("result = df['revenue'].mean()", usage={})
        agent.chat("mean revenue", sales_df, sandbox=_StubSandbox())
        # No error; last_usage may be empty dict
        assert isinstance(agent.last_usage, dict)

    def test_last_usage_initialised_on_construction(self):
        agent = _make_agent("result = 1")
        assert hasattr(agent, 'last_usage')
        assert isinstance(agent.last_usage, dict)


# ---------------------------------------------------------------------------
# Exponential backoff (verify sleep is called on retry)
# ---------------------------------------------------------------------------

class TestExponentialBackoff:

    def test_backoff_sleep_called_on_retry(self, sales_df):
        """Verify time.sleep is called with increasing delays on retry."""
        call_count = [0]

        class _FailTwiceProvider:
            last_usage = {}

            def generate(self, prompt: str, **kwargs) -> str:
                call_count[0] += 1
                if call_count[0] < 3:
                    raise RuntimeError("transient error")
                return "```python\nresult = 42\n```"

        agent = PyChartAgent.__new__(PyChartAgent)
        agent._provider = _FailTwiceProvider()
        agent._llm = None
        agent._memory = None
        agent.chart_backend = 'seaborn'
        agent.charts_output_dir = 'exports/charts'
        agent.verbose = False
        agent.max_retries = 3
        agent._llm_timeout = 60
        agent.last_transformation = None
        agent.last_usage = {}

        sleep_calls = []
        with patch('pychartai_core.agent.time.sleep', side_effect=lambda s: sleep_calls.append(s)):
            agent.chat("test", sales_df, sandbox=_StubSandbox())

        assert len(sleep_calls) == 2          # two retries → two sleeps
        assert sleep_calls[0] == 1            # 2^0
        assert sleep_calls[1] == 2            # 2^1

    def test_backoff_not_called_on_success(self, sales_df):
        agent = _make_agent("result = df['revenue'].mean()")

        sleep_calls = []
        with patch('pychartai_core.agent.time.sleep', side_effect=lambda s: sleep_calls.append(s)):
            agent.chat("mean revenue", sales_df, sandbox=_StubSandbox())

        assert sleep_calls == []   # no retries needed → no sleep


# ---------------------------------------------------------------------------
# select_dtypes string dtype compatibility (profiler / reporter)
# ---------------------------------------------------------------------------

class TestSelectDtypesCompat:

    def test_profiler_handles_string_dtype(self):
        """DataFrames with pd.StringDtype columns should profile without error."""
        from pychartai_core.profiler import DataProfiler

        df = pd.DataFrame({
            'name': pd.array(['Alice', 'Bob', 'Carol'], dtype='string'),
            'score': [90.0, 85.0, 92.0],
        })
        profiler = DataProfiler()
        report = profiler.profile(df)
        assert report is not None

    def test_profiler_handles_object_dtype(self):
        from pychartai_core.profiler import DataProfiler

        df = pd.DataFrame({
            'name': ['Alice', 'Bob', 'Carol'],    # object dtype
            'score': [90.0, 85.0, 92.0],
        })
        profiler = DataProfiler()
        report = profiler.profile(df)
        assert report is not None
