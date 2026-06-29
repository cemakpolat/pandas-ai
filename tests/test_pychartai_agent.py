"""tests/test_pychartai_agent.py

Tests for PyChartAgent — the pandasai-independent NL → code → sandbox agent.

Run with:
    pytest tests/test_pychartai_agent.py -v
"""

import pandas as pd
import pytest

import pychartai as pai
from pychartai_core.agent import PyChartAgent
from pychartai_core.sandbox import RestrictedSandbox


# ---------------------------------------------------------------------------
# Minimal stub LLM provider — returns hardcoded Python code
# ---------------------------------------------------------------------------

class _StubProvider:
	"""Fake LLM provider that echoes a pre-set code snippet."""

	def __init__(self, code: str) -> None:
		self._code = code

	def generate(self, prompt: str, **kwargs) -> str:
		return f'```python\n{self._code}\n```'


# _StubLLM is an alias for _StubProvider — the agent accepts any object with
# a .generate() method directly, so no extra wrapper is needed.
_StubLLM = _StubProvider


# ---------------------------------------------------------------------------
# Sample DataFrame
# ---------------------------------------------------------------------------

@pytest.fixture()
def sales_df():
	return pd.DataFrame({
		'region': ['North', 'South', 'East', 'West'],
		'revenue': [1000, 800, 1200, 950],
		'units': [50, 40, 60, 45],
	})


# ---------------------------------------------------------------------------
# PyChartAgent unit tests
# ---------------------------------------------------------------------------

class TestPyChartAgentExtractCode:
	"""Code extraction from raw LLM responses."""

	def test_fenced_python_block(self):
		raw = '```python\nresult = 42\n```'
		assert PyChartAgent._extract_code(raw) == 'result = 42'

	def test_fenced_no_language(self):
		raw = '```\nresult = "hello"\n```'
		assert PyChartAgent._extract_code(raw) == 'result = "hello"'

	def test_strips_think_blocks(self):
		raw = '<think>reasoning here</think>\n```python\nresult = 1\n```'
		assert PyChartAgent._extract_code(raw) == 'result = 1'

	def test_fallback_raw_text(self):
		raw = 'result = df["revenue"].mean()'
		assert PyChartAgent._extract_code(raw) == raw


class TestPyChartAgentSanitizeCode:
	"""Code sanitisation steps."""

	def test_removes_execute_sql_redefinition(self):
		code = 'def execute_sql_query(q):\n    return None\nresult = 1'
		sanitized = PyChartAgent._sanitize_code(code)
		assert 'def execute_sql_query' not in sanitized
		assert 'result = 1' in sanitized

	def test_removes_plt_object_result(self):
		code = "result = {'type': 'plot', 'value': plt}"
		sanitized = PyChartAgent._sanitize_code(code)
		assert "result = {'type': 'plot', 'value': plt}" not in sanitized

	def test_clean_code_unchanged(self):
		code = "result = df['revenue'].mean()"
		assert PyChartAgent._sanitize_code(code) == code

	def test_removes_plotting_call_for_non_chart_queries(self):
		code = "result = df.plot(kind='bar')"
		sanitized = PyChartAgent._sanitize_code(code, allow_charts=False)
		assert 'df.plot' not in sanitized

	def test_renames_restricted_underscore_variables(self):
		code = '_result_path = "x.png"\nresult = _result_path'
		sanitized = PyChartAgent._sanitize_code(code)
		assert '_result_path' not in sanitized
		assert 'result_path' in sanitized

	def test_strips_backend_imports(self):
		code = "from seaborn import pie_chart\nimport matplotlib.pyplot as plt\nresult = pie_chart(df, labels='region', values='revenue', output_file=chart_path)"
		sanitized = PyChartAgent._sanitize_code(code)
		assert 'from seaborn import pie_chart' not in sanitized
		assert 'import matplotlib.pyplot as plt' not in sanitized

	def test_strips_fake_chart_module_imports(self):
		code = "from chart import histogram\nimport charts\nresult = histogram(df, column='revenue', output_file=chart_path)"
		sanitized = PyChartAgent._sanitize_code(code)
		assert 'from chart import histogram' not in sanitized
		assert 'import charts' not in sanitized


class TestPyChartAgentIntent:
	"""Intent classification and validation rules."""

	def test_detects_chart_query(self, sales_df):
		intent = PyChartAgent._classify_query_intent('Create a pie chart of revenue by region', sales_df)
		assert intent.kind == 'chart'
		assert intent.chart_helper == 'pie_chart'

	def test_detects_histogram_query(self, sales_df):
		intent = PyChartAgent._classify_query_intent('Create a histogram of revenue', sales_df)
		assert intent.kind == 'chart'
		assert intent.chart_helper == 'histogram'

	def test_detects_generic_chart_query(self, sales_df):
		intent = PyChartAgent._classify_query_intent('Plot revenue by region', sales_df)
		assert intent.kind == 'chart'

	def test_detects_filter_query(self, sales_df):
		intent = PyChartAgent._classify_query_intent('Show all rows where revenue > 900', sales_df)
		assert intent.kind == 'filter'
		assert intent.expects_table is True

	def test_detects_grouped_aggregation(self, sales_df):
		intent = PyChartAgent._classify_query_intent('What is the total revenue by region?', sales_df)
		assert intent.kind == 'aggregation'
		assert intent.needs_grouping is True
		assert intent.group_by_hint == 'region'

	def test_detects_per_grouped_aggregation(self, sales_df):
		intent = PyChartAgent._classify_query_intent('What is the average revenue per region?', sales_df)
		assert intent.kind == 'aggregation'
		assert intent.needs_grouping is True
		assert intent.group_by_hint == 'region'

	def test_detects_per_product_grouping(self, sales_df):
		df = sales_df.assign(product=['Laptop', 'Phone', 'Laptop', 'Tablet'])
		intent = PyChartAgent._classify_query_intent('What is the average revenue per product?', df)
		assert intent.kind == 'aggregation'
		assert intent.needs_grouping is True
		assert intent.group_by_hint == 'product'

	def test_detects_count_query(self, sales_df):
		intent = PyChartAgent._classify_query_intent('How many unique regions are there?', sales_df)
		assert intent.kind == 'count'
		assert intent.expects_scalar is True

	def test_chart_validation_rejects_non_chart_result(self, sales_df):
		intent = PyChartAgent._classify_query_intent('Create a pie chart of revenue by region', sales_df)
		with pytest.raises(ValueError, match='Chart query'):
			PyChartAgent._validate_result('hello', 'Create a pie chart of revenue by region', intent)

	def test_filter_validation_rejects_scalar(self, sales_df):
		intent = PyChartAgent._classify_query_intent('Show all rows where revenue > 900', sales_df)
		with pytest.raises(ValueError, match='Filter query'):
			PyChartAgent._validate_result(3, 'Show all rows where revenue > 900', intent)

	def test_count_validation_rejects_dataframe(self, sales_df):
		intent = PyChartAgent._classify_query_intent('How many unique regions are there?', sales_df)
		with pytest.raises(ValueError, match='Count query'):
			PyChartAgent._validate_result(sales_df, 'How many unique regions are there?', intent)

	def test_grouped_aggregation_rejects_scalar(self, sales_df):
		df = sales_df.assign(product=['Laptop', 'Phone', 'Laptop', 'Tablet'])
		intent = PyChartAgent._classify_query_intent('What is the average revenue per product?', df)
		with pytest.raises(ValueError, match='Grouped aggregation'):
			PyChartAgent._validate_result(123.4, 'What is the average revenue per product?', intent)

	def test_time_bucketing_rejects_short_result(self):
		df = pd.DataFrame({
			'date': pd.to_datetime(['2024-01-01', '2024-02-01']),
			'pressure': [1000, 1010],
		})
		intent = PyChartAgent._classify_query_intent('What is the average pressure per month?', df)
		with pytest.raises(ValueError, match='too little grouped output'):
			PyChartAgent._validate_result(pd.Series([1000.0]), 'What is the average pressure per month?', intent)

	def test_normalizes_pie_chart_hallucination(self, sales_df):
		intent = PyChartAgent._classify_query_intent('Create a pie chart of revenue by region', sales_df)
		code = (
			"df_grouped = df.groupby('region')['revenue'].sum().reset_index()\n"
			"result = pd.pie(df_grouped['revenue'], labels=df_grouped['region'])"
		)
		normalized = PyChartAgent._normalize_chart_code(code, intent, 'seaborn', sales_df, 'Create a pie chart of revenue by region')
		assert 'pie_chart(' in normalized
		assert "labels='region'" in normalized
		assert "values='revenue'" in normalized

	def test_normalizes_histogram_series_call(self, sales_df):
		intent = PyChartAgent._classify_query_intent('Create a histogram of revenue', sales_df)
		code = "result = histogram(df['revenue'], title='T', output_file='chart.png', backend='seaborn')"
		normalized = PyChartAgent._normalize_chart_code(code, intent, 'seaborn', sales_df, 'Create a histogram of revenue')
		assert "histogram(df, column='revenue'" in normalized
		assert 'output_file=chart_path' in normalized

	def test_normalizes_fake_chart_import(self, sales_df):
		intent = PyChartAgent._classify_query_intent('Create a pie chart of revenue by region', sales_df)
		code = "from chart import pie_chart\nresult = pie_chart(df, labels='region', values='revenue', output_file='x.png')"
		normalized = PyChartAgent._normalize_chart_code(code, intent, 'seaborn', sales_df, 'Create a pie chart of revenue by region')
		assert 'from chart import pie_chart' not in normalized

	def test_infers_missing_pie_values(self, sales_df):
		intent = PyChartAgent._classify_query_intent('Create a pie chart of revenue by region', sales_df)
		code = "result = pie_chart(df, labels='region', output_file=chart_path, backend='seaborn')"
		normalized = PyChartAgent._normalize_chart_code(code, intent, 'seaborn', sales_df, 'Create a pie chart of revenue by region')
		assert "values='revenue'" in normalized

	def test_normalizes_pie_positional_arguments(self, sales_df):
		intent = PyChartAgent._classify_query_intent('Create a pie chart of revenue by region', sales_df)
		code = "result = pie_chart(df, 'region', labels='region', output_file=chart_path, backend='seaborn')"
		normalized = PyChartAgent._normalize_chart_code(code, intent, 'seaborn', sales_df, 'Create a pie chart of revenue by region')
		assert "labels='region'" in normalized
		assert normalized.count("labels='region'") == 1
		assert "values='revenue'" in normalized

	def test_infers_value_column_from_query(self, sales_df):
		assert PyChartAgent._infer_value_column('Plot revenue by region', sales_df) == 'revenue'

	def test_locks_chart_helper_family(self, sales_df):
		code = "result = area_chart(df, x='region', y='revenue', output_file=chart_path, backend='seaborn')"
		locked = PyChartAgent._lock_chart_helper_calls(code, 'pie_chart')
		assert 'area_chart(' not in locked
		assert 'pie_chart(' in locked

	def test_injects_fallback_when_target_helper_missing(self, sales_df):
		intent = PyChartAgent._classify_query_intent('Create a pie chart of revenue by region', sales_df)
		code = "result = str(df.head())"
		repaired = PyChartAgent._ensure_chart_helper_present(
			code=code,
			intent=intent,
			chart_backend='seaborn',
			df=sales_df,
			query='Create a pie chart of revenue by region',
		)
		assert 'pie_chart(' in repaired
		assert '[repair]' in repaired


class TestPyChartAgentFormatResult:
	"""Result normalisation."""

	def test_plot_dict(self):
		assert PyChartAgent._format_result({'type': 'plot', 'value': 'out.png'}) == 'out.png'

	def test_dataframe(self, sales_df):
		result = PyChartAgent._format_result(sales_df)
		assert 'North' in result

	def test_none(self):
		assert PyChartAgent._format_result(None) == 'No result returned.'

	def test_string(self):
		assert PyChartAgent._format_result('hello') == 'hello'

	def test_number(self):
		assert PyChartAgent._format_result(42) == '42'

	def test_series(self, sales_df):
		result = PyChartAgent._format_result(sales_df['revenue'])
		assert '1000' in result


class TestPyChartAgentChat:
	"""End-to-end chat via stub LLM + real RestrictedSandbox."""

	def test_text_query(self, sales_df):
		stub = _StubLLM("result = str(df['revenue'].sum())")
		agent = PyChartAgent(llm=stub)
		result = agent.chat('What is total revenue?', sales_df)
		assert result == '3950'

	def test_dataframe_result(self, sales_df):
		stub = _StubLLM("result = df[df['revenue'] > 900]")
		agent = PyChartAgent(llm=stub)
		result = agent.chat('Show high revenue regions', sales_df)
		assert 'North' in result
		assert 'East' in result

	def test_empty_dataframe(self, sales_df):
		stub = _StubLLM("result = 'ok'")
		agent = PyChartAgent(llm=stub)
		empty_df = pd.DataFrame()
		with pytest.raises(ValueError, match='empty'):
			agent.chat('anything', empty_df)

	def test_bad_code_returns_error(self, sales_df):
		stub = _StubLLM('raise ValueError("boom")')
		agent = PyChartAgent(llm=stub, max_retries=1)
		with pytest.raises(RuntimeError, match='Query failed'):
			agent.chat('cause error', sales_df)

	def test_custom_sandbox(self, sales_df):
		stub = _StubLLM("result = len(df)")
		agent = PyChartAgent(llm=stub)
		sandbox = RestrictedSandbox()
		result = agent.chat('Count rows', sales_df, sandbox=sandbox)
		assert result == '4'

	def test_filter_query_retries_when_first_attempt_returns_chart(self, sales_df):
		class _SequenceProvider:
			def __init__(self):
				self.responses = [
					"result = bar_chart(df, x='region', y='revenue', output_file=chart_path, backend='seaborn')",
					"result = df[df['revenue'] > 900]",
				]
				self.index = 0

			def generate(self, prompt: str, **kwargs) -> str:
				response = self.responses[min(self.index, len(self.responses) - 1)]
				self.index += 1
				return f'```python\n{response}\n```'

		agent = PyChartAgent(llm=_SequenceProvider(), max_retries=2)
		result = agent.chat('Show all rows where revenue > 900', sales_df)
		assert 'North' in result
		assert 'East' in result

	def test_grouped_query_retries_when_first_attempt_returns_scalar(self, sales_df):
		df = sales_df.assign(product=['Laptop', 'Phone', 'Laptop', 'Tablet'])

		class _SequenceProvider:
			def __init__(self):
				self.responses = [
					"result = df['revenue'].mean()",
					"result = df.groupby('product', as_index=False)['revenue'].mean()",
				]
				self.index = 0

			def generate(self, prompt: str, **kwargs) -> str:
				response = self.responses[min(self.index, len(self.responses) - 1)]
				self.index += 1
				return f'```python\n{response}\n```'

		agent = PyChartAgent(llm=_SequenceProvider(), max_retries=2)
		result = agent.chat('What is the average revenue per product?', df)
		assert 'Laptop' in result
		assert 'product' in result


class TestSmartDataFrameAgentSwitch:
	"""SmartDataFrame.chat() agent= parameter dispatch."""

	def test_own_agent_route(self, sales_df, monkeypatch):
		"""agent='own' must call _chat_with_own_agent."""
		sdf = pai.SmartDataFrame(sales_df)
		calls = []

		def _fake_own(self_inner, query, *args, **kwargs):
			calls.append('own')
			return 'own_result'

		from pychartai_core import smart_df as sdf_mod
		monkeypatch.setattr(sdf_mod.SmartDataFrame, '_chat_with_own_agent', _fake_own)

		# Need a (fake) LLM in config so RuntimeError is not raised
		from pychartai_core.config import config
		config.set({'llm': _StubLLM("result = 1")})

		result = sdf.chat('test', agent='own')
		assert result == 'own_result'
		assert calls == ['own']

	def test_invalid_agent_raises(self, sales_df):
		sdf = pai.SmartDataFrame(sales_df)
		from pychartai_core.config import config
		config.set({'llm': _StubLLM("result = 1")})
		with pytest.raises(ValueError, match='agent must be'):
			sdf.chat('test', agent='unknown')


# ---------------------------------------------------------------------------
# Multi-DataFrame coercion tests (PyChartAgent._coerce_dataframes)
# ---------------------------------------------------------------------------

class TestCoerceDataframes:
	"""Unit tests for PyChartAgent._coerce_dataframes()."""

	def test_single_dataframe_passthrough(self):
		df = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})
		out, note = PyChartAgent._coerce_dataframes(df)
		assert out is df
		assert note == ''

	def test_list_single_element_passthrough(self):
		df = pd.DataFrame({'a': [1, 2]})
		out, note = PyChartAgent._coerce_dataframes([df])
		assert out is df

	def test_dict_single_element_passthrough(self):
		df = pd.DataFrame({'x': [9]})
		out, note = PyChartAgent._coerce_dataframes({'only': df})
		assert out is df

	def test_list_two_dataframes_merged_on_common_column(self):
		df1 = pd.DataFrame({'id': [1, 2], 'revenue': [100, 200]})
		df2 = pd.DataFrame({'id': [1, 2], 'cost': [50, 80]})
		out, note = PyChartAgent._coerce_dataframes([df1, df2])
		assert 'revenue' in out.columns
		assert 'cost' in out.columns
		assert len(out) == 2
		assert 'id' in note

	def test_dict_two_dataframes_merged(self):
		orders = pd.DataFrame({'order_id': [1, 2], 'amount': [50, 75]})
		products = pd.DataFrame({'order_id': [1, 2], 'name': ['A', 'B']})
		out, note = PyChartAgent._coerce_dataframes({'orders': orders, 'products': products})
		assert 'amount' in out.columns
		assert 'name' in out.columns

	def test_no_common_columns_concat(self):
		df1 = pd.DataFrame({'a': [1, 2]})
		df2 = pd.DataFrame({'b': [3, 4]})
		out, note = PyChartAgent._coerce_dataframes([df1, df2])
		# Should have both columns after concat+fill
		assert 'a' in out.columns or 'b' in out.columns
		assert 'concatenated' in note.lower() or 'concat' in note.lower()

	def test_empty_list_returns_empty_df(self):
		out, note = PyChartAgent._coerce_dataframes([])
		assert isinstance(out, pd.DataFrame)
		assert out.empty

	def test_note_contains_schema_lines(self):
		df1 = pd.DataFrame({'id': [1], 'val': [10]})
		df2 = pd.DataFrame({'id': [1], 'extra': [99]})
		_, note = PyChartAgent._coerce_dataframes([df1, df2])
		assert 'df0' in note
		assert 'df1' in note

	def test_chat_accepts_list_of_dataframes(self, sales_df):
		"""PyChartAgent.chat() should accept a list without raising."""
		df2 = pd.DataFrame({'region': ['North', 'South'], 'target': [900, 700]})

		agent = PyChartAgent(llm=_StubLLM("result = df['revenue'].sum()"), max_retries=1)
		# Should not raise; merged df will have region/revenue/units/target columns
		result = agent.chat('What is the total revenue?', [sales_df, df2])
		assert result is not None

	def test_chat_accepts_dict_of_dataframes(self, sales_df):
		df2 = pd.DataFrame({'region': ['North', 'South'], 'target': [900, 700]})
		agent = PyChartAgent(llm=_StubLLM("result = df['units'].sum()"), max_retries=1)
		result = agent.chat('What are total units?', {'sales': sales_df, 'targets': df2})
		assert result is not None

	def test_smart_df_extra_dfs_forwarded(self, sales_df):
		"""SmartDataFrame.chat(extra_dfs=) merges and passes to PyChartAgent."""
		df2 = pd.DataFrame({'region': ['North', 'South'], 'budget': [500, 300]})

		calls = []

		def _fake_own(self_inner, query, *args, **kwargs):
			calls.append(kwargs.get('extra_dfs'))
			return 'merged_result'

		import pychartai_core.smart_df as sdf_mod
		from pychartai_core.config import config
		config.set({'llm': _StubLLM("result = 1")})

		sdf = pai.SmartDataFrame(sales_df)
		import pytest
		import unittest.mock as mock
		with mock.patch.object(sdf_mod.SmartDataFrame, '_chat_with_own_agent', _fake_own):
			result = sdf.chat('Compare', extra_dfs=df2)
		assert result == 'merged_result'
		assert calls[0] is df2


@pytest.fixture()
def sales_df():
	return pd.DataFrame({
		'region': ['North', 'South', 'East', 'West'],
		'revenue': [1000, 800, 1200, 950],
		'units': [50, 40, 60, 45],
	})


# ---------------------------------------------------------------------------
# InsightReporter / sdf.report() tests
# ---------------------------------------------------------------------------

class TestInsightReporter:
	"""Tests for the HTML report generator."""

	@pytest.fixture()
	def simple_df(self):
		import numpy as np
		rng = np.random.default_rng(42)
		return pd.DataFrame({
			'category': ['A', 'B', 'C', 'D'] * 25,
			'value': rng.integers(10, 200, size=100).astype(float),
			'count': rng.integers(1, 50, size=100),
		})

	def test_report_creates_html_file(self, simple_df, tmp_path):
		sdf = pai.SmartDataFrame(simple_df)
		out = sdf.report(str(tmp_path / 'report.html'), llm_narrative=False)
		assert out.endswith('.html')
		import os
		assert os.path.isfile(out)

	def test_report_html_contains_overview(self, simple_df, tmp_path):
		sdf = pai.SmartDataFrame(simple_df)
		out = sdf.report(str(tmp_path / 'r.html'), llm_narrative=False)
		content = open(out, encoding='utf-8').read()
		assert 'Overview' in content
		assert '100' in content  # n_rows

	def test_report_html_contains_chart_section(self, simple_df, tmp_path):
		sdf = pai.SmartDataFrame(simple_df)
		out = sdf.report(str(tmp_path / 'r.html'), llm_narrative=False)
		content = open(out, encoding='utf-8').read()
		# At least one chart should be base64-encoded PNG
		assert 'data:image/png;base64,' in content

	def test_report_returns_absolute_path(self, simple_df, tmp_path):
		import os
		sdf = pai.SmartDataFrame(simple_df)
		out = sdf.report(str(tmp_path / 'sub' / 'r.html'), llm_narrative=False)
		assert os.path.isabs(out)

	def test_report_no_llm_narrative_skips_llm(self, simple_df, tmp_path):
		"""When llm_narrative=False, no LLM calls are made."""
		from pychartai_core.reporter import _llm_narrative
		called = []

		def _fake_narrative(llm, context, max_sentences=3):
			called.append(context)
			return None

		import pychartai_core.reporter as reporter_mod
		import unittest.mock as mock
		with mock.patch.object(reporter_mod, '_llm_narrative', _fake_narrative):
			sdf = pai.SmartDataFrame(simple_df)
			sdf.report(str(tmp_path / 'r.html'), llm_narrative=False)
		assert called == []

	def test_report_with_date_column_adds_timeseries(self, tmp_path):
		import numpy as np
		df = pd.DataFrame({
			'date': pd.date_range('2023-01-01', periods=50, freq='D'),
			'sales': np.random.default_rng(1).integers(100, 500, size=50).astype(float),
		})
		sdf = pai.SmartDataFrame(df)
		out = sdf.report(str(tmp_path / 'ts.html'), llm_narrative=False)
		content = open(out, encoding='utf-8').read()
		assert 'data:image/png;base64,' in content

	def test_reporter_direct_instantiation(self, simple_df, tmp_path):
		from pychartai_core.reporter import InsightReporter
		sdf = pai.SmartDataFrame(simple_df)
		reporter = InsightReporter(sdf)
		out = reporter.generate(str(tmp_path / 'direct.html'), llm_narrative=False)
		assert out.endswith('.html')
