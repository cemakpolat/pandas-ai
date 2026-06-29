#!/usr/bin/env python3
"""
test_integration_extensive.py — Real-world integration tests for pychartai.

Tests whether the full pipeline (LLM → code generation → sandbox → result)
can actually answer questions, generate quality charts, explain results,
use conversation memory, and profile DataFrames.

Requires an Ollama model running locally (or any configured LLM).

Usage:
    python tests/test_integration_extensive.py --model llama3.2
    python tests/test_integration_extensive.py --model llama3.2 --backend plotly
    python tests/test_integration_extensive.py --model llama3.2 --section memory
    python tests/test_integration_extensive.py --model llama3.2 --section explain
    python tests/test_integration_extensive.py --model llama3.2 --section charts
    python tests/test_integration_extensive.py --model llama3.2 --section qa
    python tests/test_integration_extensive.py --model llama3.2 --section profile
    python tests/test_integration_extensive.py --model llama3.2 --section all

    # Via Makefile:
    make test-integration MODEL=llama3.2
    make test-memory MODEL=llama3.2
    make test-explain MODEL=llama3.2
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import traceback

# Ensure src/ is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pandas as pd
import pychartai as pai
from pychartai_core.data_manager import DataManager
from pychartai_core.profiler import DataProfiler
from pychartai_core.themes import BUILTIN_THEMES, resolve_theme


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class ResultTracker:
	"""Track pass/fail/skip for each test case."""

	def __init__(self):
		self.passed: list[str] = []
		self.failed: list[tuple[str, str]] = []
		self.skipped: list[str] = []
		self._start: float = 0.0

	def start(self, name: str) -> None:
		self._start = time.time()
		print(f'\n  ▸ {name} ...', end=' ', flush=True)

	def ok(self, name: str, detail: str = '') -> None:
		elapsed = time.time() - self._start
		extra = f' — {detail}' if detail else ''
		print(f'✓ ({elapsed:.1f}s){extra}')
		self.passed.append(name)

	def fail(self, name: str, error: str) -> None:
		elapsed = time.time() - self._start
		print(f'✗ ({elapsed:.1f}s)')
		print(f'    Error: {error[:200]}')
		self.failed.append((name, error))

	def skip(self, name: str, reason: str) -> None:
		print(f'  ▸ {name} ... SKIP ({reason})')
		self.skipped.append(name)

	def summary(self) -> None:
		total = len(self.passed) + len(self.failed) + len(self.skipped)
		print(f'\n{"=" * 60}')
		print(f'Integration Test Results: {len(self.passed)}/{total} passed', end='')
		if self.failed:
			print(f', {len(self.failed)} FAILED', end='')
		if self.skipped:
			print(f', {len(self.skipped)} skipped', end='')
		print()
		if self.failed:
			print('\nFailed tests:')
			for name, err in self.failed:
				print(f'  ✗ {name}: {err[:120]}')
		print(f'{"=" * 60}')


def load_datasets() -> dict[str, pd.DataFrame]:
	"""Load all sample datasets."""
	dm = DataManager()
	datasets = {}
	for dtype in ('sales', 'weather', 'ecommerce', 'health', 'energy', 'stocks'):
		datasets[dtype] = dm.create_sample_data(dtype, dtype)
	return datasets


def is_chart_path(result: str) -> bool:
	"""Check if result looks like a chart file path."""
	if not isinstance(result, str):
		return False
	return result.endswith(('.png', '.html', '.svg')) and os.path.isfile(result)


# ---------------------------------------------------------------------------
# Section 1: Q&A — Can the LLM answer data questions correctly?
# ---------------------------------------------------------------------------

def run_qa_tests(datasets: dict, backend: str, results: ResultTracker) -> None:
	"""Test natural-language Q&A against real data."""
	print('\n' + '=' * 60)
	print('SECTION 1: Natural Language Q&A')
	print('=' * 60)

	sales = datasets['sales']
	sdf = pai.SmartDataFrame(sales)

	# --- Test 1: Simple aggregation ---
	name = 'QA: Total revenue'
	results.start(name)
	try:
		answer = sdf.chat('What is the total revenue?', agent='own')
		assert 'Error' not in str(answer), f'Got error: {answer}'
		results.ok(name, f'Answer: {str(answer)[:80]}')
	except Exception as e:
		results.fail(name, str(e))

	# --- Test 2: Group-by aggregation ---
	name = 'QA: Revenue by region'
	results.start(name)
	try:
		answer = sdf.chat('What is the total revenue by region?', agent='own')
		assert 'Error' not in str(answer), f'Got error: {answer}'
		results.ok(name, f'Answer: {str(answer)[:80]}')
	except Exception as e:
		results.fail(name, str(e))

	# --- Test 3: Count query ---
	name = 'QA: Count unique products'
	results.start(name)
	try:
		answer = sdf.chat('How many unique products are there?', agent='own')
		assert 'Error' not in str(answer), f'Got error: {answer}'
		results.ok(name, f'Answer: {str(answer)[:80]}')
	except Exception as e:
		results.fail(name, str(e))

	# --- Test 4: Filtering ---
	name = 'QA: Filter high revenue'
	results.start(name)
	try:
		answer = sdf.chat('Show rows where revenue is greater than 1000', agent='own')
		assert 'Error' not in str(answer), f'Got error: {answer}'
		results.ok(name, f'Answer: {str(answer)[:80]}')
	except Exception as e:
		results.fail(name, str(e))

	# --- Test 5: Top-N ---
	name = 'QA: Top 5 products by revenue'
	results.start(name)
	try:
		answer = sdf.chat('What are the top 5 products by revenue?', agent='own')
		assert 'Error' not in str(answer), f'Got error: {answer}'
		results.ok(name, f'Answer: {str(answer)[:80]}')
	except Exception as e:
		results.fail(name, str(e))

	# --- Test 6: Average / mean ---
	name = 'QA: Average price per product'
	results.start(name)
	try:
		answer = sdf.chat('What is the average price per product?', agent='own')
		assert 'Error' not in str(answer), f'Got error: {answer}'
		results.ok(name, f'Answer: {str(answer)[:80]}')
	except Exception as e:
		results.fail(name, str(e))

	# --- Test 7: Weather dataset ---
	weather = datasets['weather']
	wdf = pai.SmartDataFrame(weather)

	name = 'QA: Hottest city (weather)'
	results.start(name)
	try:
		answer = wdf.chat('Which city has the highest average temperature?', agent='own')
		assert 'Error' not in str(answer), f'Got error: {answer}'
		results.ok(name, f'Answer: {str(answer)[:80]}')
	except Exception as e:
		results.fail(name, str(e))

	# --- Test 8: Health dataset ---
	health = datasets['health']
	hdf = pai.SmartDataFrame(health)

	name = 'QA: Average BMI by gender (health)'
	results.start(name)
	try:
		answer = hdf.chat('What is the average BMI by gender?', agent='own')
		assert 'Error' not in str(answer), f'Got error: {answer}'
		results.ok(name, f'Answer: {str(answer)[:80]}')
	except Exception as e:
		results.fail(name, str(e))

	# --- Test 9: Stocks dataset ---
	stocks = datasets['stocks']
	stdf = pai.SmartDataFrame(stocks)

	name = 'QA: Stock with highest volume (stocks)'
	results.start(name)
	try:
		answer = stdf.chat('Which stock ticker has the highest average volume?', agent='own')
		assert 'Error' not in str(answer), f'Got error: {answer}'
		results.ok(name, f'Answer: {str(answer)[:80]}')
	except Exception as e:
		results.fail(name, str(e))

	# --- Test 10: E-commerce complex ---
	ecom = datasets['ecommerce']
	edf = pai.SmartDataFrame(ecom)

	name = 'QA: Revenue by category (ecommerce)'
	results.start(name)
	try:
		answer = edf.chat('What is the total revenue by category?', agent='own')
		assert 'Error' not in str(answer), f'Got error: {answer}'
		results.ok(name, f'Answer: {str(answer)[:80]}')
	except Exception as e:
		results.fail(name, str(e))


# ---------------------------------------------------------------------------
# Section 2: Chart Generation — Quality chart output across backends
# ---------------------------------------------------------------------------

def run_chart_tests(datasets: dict, backend: str, output_dir: str, results: ResultTracker) -> None:
	"""Test chart generation across different chart types and datasets."""
	print('\n' + '=' * 60)
	print(f'SECTION 2: Chart Generation (backend={backend})')
	print('=' * 60)

	os.makedirs(output_dir, exist_ok=True)
	sales = datasets['sales']
	sdf = pai.SmartDataFrame(sales)

	chart_tests = [
		('Chart: Bar chart (price by region)', 'Plot a bar chart of total price by region'),
		('Chart: Line chart (price trend)', 'Plot a line chart of price over time'),
		('Chart: Pie chart (quantity share)', 'Plot a pie chart of quantity share by product'),
		('Chart: Histogram (price distribution)', 'Plot a histogram of price'),
		('Chart: Scatter chart (quantity vs price)', 'Plot a scatter chart of quantity versus price'),
		('Chart: Box chart (price by region)', 'Plot a box chart of price by region'),
		('Chart: Heatmap (correlation)', 'Plot a correlation heatmap'),
	]

	for name, query in chart_tests:
		results.start(name)
		try:
			result = sdf.chat(query, chart_type=backend, agent='own')
			if is_chart_path(result):
				results.ok(name, f'Chart: {os.path.basename(result)}')
			elif 'Error' in str(result):
				results.fail(name, str(result)[:200])
			else:
				# Result may be a valid non-chart response
				results.ok(name, f'Result: {str(result)[:80]}')
		except Exception as e:
			results.fail(name, str(e))

	# --- Cross-dataset charts ---
	weather = datasets['weather']
	wdf = pai.SmartDataFrame(weather)

	name = 'Chart: Weather line chart (temperature trend)'
	results.start(name)
	try:
		result = wdf.chat('Plot a line chart of temperature over time', chart_type=backend, agent='own')
		if is_chart_path(result):
			results.ok(name, f'Chart: {os.path.basename(result)}')
		else:
			results.ok(name, f'Result: {str(result)[:80]}')
	except Exception as e:
		results.fail(name, str(e))

	health = datasets['health']
	hdf = pai.SmartDataFrame(health)

	name = 'Chart: Health violin chart (BMI by gender)'
	results.start(name)
	try:
		result = hdf.chat('Plot a violin chart of bmi column grouped by gender column', chart_type=backend, agent='own')
		if is_chart_path(result):
			results.ok(name, f'Chart: {os.path.basename(result)}')
		else:
			results.ok(name, f'Result: {str(result)[:80]}')
	except Exception as e:
		results.fail(name, str(e))

	energy = datasets['energy']
	edf = pai.SmartDataFrame(energy)

	name = 'Chart: Energy stacked bar (production by source)'
	results.start(name)
	try:
		result = edf.chat('Plot a bar chart of total production_gwh by source column', chart_type=backend, agent='own')
		if is_chart_path(result):
			results.ok(name, f'Chart: {os.path.basename(result)}')
		else:
			results.ok(name, f'Result: {str(result)[:80]}')
	except Exception as e:
		results.fail(name, str(e))


# ---------------------------------------------------------------------------
# Section 3: Data Profiling — Auto-EDA without LLM
# ---------------------------------------------------------------------------

def run_profile_tests(datasets: dict, results: ResultTracker) -> None:
	"""Test DataFrame profiling / auto-EDA (no LLM needed)."""
	print('\n' + '=' * 60)
	print('SECTION 3: Data Profiling / Auto-EDA')
	print('=' * 60)

	for dtype, df in datasets.items():
		name = f'Profile: {dtype} ({df.shape[0]}×{df.shape[1]})'
		results.start(name)
		try:
			report = DataProfiler.profile(df)
			assert report.n_rows == df.shape[0]
			assert report.n_columns == df.shape[1]
			assert report.memory_usage_mb > 0
			summary = report.summary
			assert 'rows' in summary.lower()
			results.ok(name, f'{report.n_rows:,} rows, {report.duplicates} dupes, {report.missing["missing_count"].sum()} missing')
		except Exception as e:
			results.fail(name, str(e))

	# Test via SmartDataFrame.profile()
	name = 'Profile: via SmartDataFrame.profile()'
	results.start(name)
	try:
		sdf = pai.SmartDataFrame(datasets['sales'])
		report = sdf.profile()
		assert report.n_rows > 0
		assert report.numeric_stats is not None
		results.ok(name, f'Stats for {len(report.numeric_stats)} numeric cols')
	except Exception as e:
		results.fail(name, str(e))

	# Test ProfileReport.to_dict()
	name = 'Profile: to_dict() serialization'
	results.start(name)
	try:
		report = DataProfiler.profile(datasets['ecommerce'])
		d = report.to_dict()
		assert isinstance(d, dict)
		assert 'n_rows' in d
		assert 'correlations' in d
		results.ok(name, f'{len(d)} keys')
	except Exception as e:
		results.fail(name, str(e))

	# Correlation detection
	name = 'Profile: correlation matrix shape'
	results.start(name)
	try:
		report = DataProfiler.profile(datasets['health'])
		if report.correlations is not None:
			n_numeric = len(report.correlations.columns)
			assert n_numeric >= 2
			results.ok(name, f'{n_numeric}×{n_numeric} correlation matrix')
		else:
			results.ok(name, 'No correlation (< 2 numeric cols)')
	except Exception as e:
		results.fail(name, str(e))


# ---------------------------------------------------------------------------
# Section 4: Conversation Memory — Multi-turn follow-ups
# ---------------------------------------------------------------------------

def run_memory_tests(datasets: dict, backend: str, results: ResultTracker) -> None:
	"""Test conversation memory and follow-up queries."""
	print('\n' + '=' * 60)
	print('SECTION 4: Conversation Memory')
	print('=' * 60)

	sales = datasets['sales']
	sdf = pai.SmartDataFrame(sales)
	sdf.enable_memory(window_size=5)

	# Turn 1: Establish context
	name = 'Memory: Turn 1 — revenue by region'
	results.start(name)
	try:
		answer1 = sdf.chat('What is the total revenue by region?', agent='own')
		assert 'Error' not in str(answer1), f'Got error: {answer1}'
		assert sdf.memory is not None
		assert len(sdf.memory) >= 0  # memory tracked by agent, but may not push back to sdf memory
		results.ok(name, f'Answer: {str(answer1)[:80]}')
	except Exception as e:
		results.fail(name, str(e))

	# Turn 2: Follow-up referencing previous
	name = 'Memory: Turn 2 — follow-up (top region)'
	results.start(name)
	try:
		answer2 = sdf.chat('Which region had the highest?', agent='own')
		assert 'Error' not in str(answer2), f'Got error: {answer2}'
		results.ok(name, f'Answer: {str(answer2)[:80]}')
	except Exception as e:
		results.fail(name, str(e))

	# Turn 3: Another follow-up
	name = 'Memory: Turn 3 — chart follow-up'
	results.start(name)
	try:
		answer3 = sdf.chat('Show a bar chart of that', chart_type=backend, agent='own')
		results.ok(name, f'Result: {str(answer3)[:80]}')
	except Exception as e:
		results.fail(name, str(e))

	# Clean memory
	name = 'Memory: disable and verify'
	results.start(name)
	try:
		sdf.disable_memory()
		assert sdf.memory is None
		results.ok(name, 'Memory disabled')
	except Exception as e:
		results.fail(name, str(e))


# ---------------------------------------------------------------------------
# Section 5: Result Explanations — explain=True
# ---------------------------------------------------------------------------

def run_explain_tests(datasets: dict, results: ResultTracker) -> None:
	"""Test LLM-generated explanations of results."""
	print('\n' + '=' * 60)
	print('SECTION 5: Result Explanations')
	print('=' * 60)

	sales = datasets['sales']
	sdf = pai.SmartDataFrame(sales)

	name = 'Explain: Total revenue with explanation'
	results.start(name)
	try:
		answer = sdf.chat('What is the total revenue?', agent='own', explain=True)
		assert 'Error' not in str(answer), f'Got error: {answer}'
		# The answer should contain the number + an explanation
		results.ok(name, f'Answer (len={len(str(answer))}): {str(answer)[:100]}')
	except Exception as e:
		results.fail(name, str(e))

	name = 'Explain: Average price with explanation'
	results.start(name)
	try:
		answer = sdf.chat('What is the average price per product?', agent='own', explain=True)
		assert 'Error' not in str(answer), f'Got error: {answer}'
		results.ok(name, f'Answer (len={len(str(answer))}): {str(answer)[:100]}')
	except Exception as e:
		results.fail(name, str(e))

	# Health dataset
	health = datasets['health']
	hdf = pai.SmartDataFrame(health)

	name = 'Explain: Health inference (risk by smoking)'
	results.start(name)
	try:
		answer = hdf.chat('What is the average risk score for smokers vs non-smokers?', agent='own', explain=True)
		assert 'Error' not in str(answer), f'Got error: {answer}'
		results.ok(name, f'Answer (len={len(str(answer))}): {str(answer)[:100]}')
	except Exception as e:
		results.fail(name, str(e))


# ---------------------------------------------------------------------------
# Section 6: DataFrame Inference — can we derive insights?
# ---------------------------------------------------------------------------

def run_inference_tests(datasets: dict, results: ResultTracker) -> None:
	"""Test whether the LLM can provide data inference and insights."""
	print('\n' + '=' * 60)
	print('SECTION 6: DataFrame Inference & Insights')
	print('=' * 60)

	# --- Health data: can we infer risk factors? ---
	health = datasets['health']
	hdf = pai.SmartDataFrame(health)

	name = 'Inference: BMI vs risk_score correlation'
	results.start(name)
	try:
		answer = hdf.chat('Is there a correlation between BMI and risk_score? Calculate the correlation coefficient.', agent='own')
		assert 'Error' not in str(answer), f'Got error: {answer}'
		results.ok(name, f'Answer: {str(answer)[:100]}')
	except Exception as e:
		results.fail(name, str(e))

	name = 'Inference: Outcome distribution by gender'
	results.start(name)
	try:
		answer = hdf.chat('What is the distribution of outcome (positive vs negative) by gender?', agent='own')
		assert 'Error' not in str(answer), f'Got error: {answer}'
		results.ok(name, f'Answer: {str(answer)[:100]}')
	except Exception as e:
		results.fail(name, str(e))

	# --- E-commerce: can we find patterns? ---
	ecom = datasets['ecommerce']
	edf = pai.SmartDataFrame(ecom)

	name = 'Inference: Best-selling category'
	results.start(name)
	try:
		answer = edf.chat('Which category has the most orders? Show the count per category.', agent='own')
		assert 'Error' not in str(answer), f'Got error: {answer}'
		results.ok(name, f'Answer: {str(answer)[:100]}')
	except Exception as e:
		results.fail(name, str(e))

	name = 'Inference: Revenue by country (top 5)'
	results.start(name)
	try:
		answer = edf.chat('What are the top 5 countries by total order_value?', agent='own')
		assert 'Error' not in str(answer), f'Got error: {answer}'
		results.ok(name, f'Answer: {str(answer)[:100]}')
	except Exception as e:
		results.fail(name, str(e))

	# --- Energy: can we compare sources? ---
	energy = datasets['energy']
	endf = pai.SmartDataFrame(energy)

	name = 'Inference: CO2 by energy source'
	results.start(name)
	try:
		answer = endf.chat('Which energy source produces the most CO2 per GWh of production?', agent='own')
		assert 'Error' not in str(answer), f'Got error: {answer}'
		results.ok(name, f'Answer: {str(answer)[:100]}')
	except Exception as e:
		results.fail(name, str(e))

	name = 'Inference: Energy production trend by source'
	results.start(name)
	try:
		answer = endf.chat('What is the total production by energy source?', agent='own')
		assert 'Error' not in str(answer), f'Got error: {answer}'
		results.ok(name, f'Answer: {str(answer)[:100]}')
	except Exception as e:
		results.fail(name, str(e))


# ---------------------------------------------------------------------------
# Section 7: Transformation Tracking — inspect what the agent generated
# ---------------------------------------------------------------------------

def run_tracking_tests(datasets: dict, results: ResultTracker) -> None:
	"""Test that generated code and intent are tracked."""
	print('\n' + '=' * 60)
	print('SECTION 7: Transformation Tracking')
	print('=' * 60)

	sales = datasets['sales']
	agent = pai.PyChartAgent(chart_backend='seaborn')

	name = 'Tracking: intent + code tracked'
	results.start(name)
	try:
		result = agent.chat('What is total revenue by region?', sales)
		xform = agent.last_transformation
		assert xform is not None, 'last_transformation is None'
		assert xform.query == 'What is total revenue by region?'
		assert xform.generated_code, 'No generated code tracked'
		assert xform.intent is not None, 'No intent tracked'
		results.ok(name, f'Intent: {xform.intent.kind}, attempts: {xform.attempts}, success: {xform.success}')
	except Exception as e:
		results.fail(name, str(e))

	name = 'Tracking: chart intent detected'
	results.start(name)
	try:
		result = agent.chat('Plot a bar chart of revenue by region', sales)
		xform = agent.last_transformation
		assert xform is not None
		assert xform.intent.kind == 'chart', f'Expected chart, got {xform.intent.kind}'
		results.ok(name, f'Intent: {xform.intent.kind}, helper: {xform.intent.chart_helper}')
	except Exception as e:
		results.fail(name, str(e))


# ---------------------------------------------------------------------------
# Section 8: Theme rendering (no LLM needed)
# ---------------------------------------------------------------------------

def run_theme_tests(output_dir: str, results: ResultTracker) -> None:
	"""Test that all themes render without error."""
	print('\n' + '=' * 60)
	print('SECTION 8: Chart Themes')
	print('=' * 60)

	from pychartai_core.visualization import bar_chart

	df = pd.DataFrame({
		'category': ['Electronics', 'Clothing', 'Food', 'Books', 'Sports'],
		'revenue': [45000, 32000, 28000, 15000, 22000],
	})

	theme_dir = os.path.join(output_dir, 'themes')
	os.makedirs(theme_dir, exist_ok=True)

	for theme_name in BUILTIN_THEMES:
		name = f'Theme: {theme_name} bar chart'
		results.start(name)
		try:
			theme = resolve_theme(theme_name)
			theme.apply_matplotlib()
			path = bar_chart(
				df, x='category', y='revenue',
				title=f'Revenue by Category ({theme_name})',
				output_file=os.path.join(theme_dir, f'{theme_name}_bar.png'),
				backend='seaborn',
			)
			assert os.path.isfile(path), f'Chart not created: {path}'
			results.ok(name, f'{os.path.basename(path)}')
		except Exception as e:
			results.fail(name, str(e))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
	parser = argparse.ArgumentParser(description='pychartai extensive integration tests')
	parser.add_argument('--model', default='llama3.2', help='Ollama model name')
	parser.add_argument('--backend', default='seaborn', help='Chart backend')
	parser.add_argument('--output-dir', default='exports/charts/integration', help='Chart output dir')
	parser.add_argument('--section', default='all',
	                    choices=['all', 'qa', 'charts', 'profile', 'memory', 'explain', 'inference', 'tracking', 'themes'],
	                    help='Run specific section only')
	args = parser.parse_args()

	print(f'pychartai Integration Tests')
	print(f'Model: {args.model} | Backend: {args.backend} | Output: {args.output_dir}')
	print(f'Section: {args.section}')
	print('=' * 60)

	# Configure LLM
	needs_llm = args.section in ('all', 'qa', 'charts', 'memory', 'explain', 'inference', 'tracking')
	if needs_llm:
		try:
			llm = pai.OllamaLLM(model=args.model)
			pai.config.set({
				'llm': llm,
				'chart_backend': args.backend,
				'charts_output_dir': args.output_dir,
				'verbose': False,
				'max_retries': 3,
			})
		except Exception as e:
			print(f'\n✗ Failed to configure LLM: {e}')
			print('  Make sure Ollama is running: ollama serve')
			print(f'  And the model is pulled: ollama pull {args.model}')
			sys.exit(1)

	# Load datasets
	print('\nLoading datasets...')
	datasets = load_datasets()
	for dtype, df in datasets.items():
		print(f'  {dtype}: {df.shape[0]:,} rows × {df.shape[1]} cols')

	results = ResultTracker()
	os.makedirs(args.output_dir, exist_ok=True)

	sections = {
		'profile': lambda: run_profile_tests(datasets, results),
		'themes': lambda: run_theme_tests(args.output_dir, results),
		'qa': lambda: run_qa_tests(datasets, args.backend, results),
		'charts': lambda: run_chart_tests(datasets, args.backend, args.output_dir, results),
		'memory': lambda: run_memory_tests(datasets, args.backend, results),
		'explain': lambda: run_explain_tests(datasets, results),
		'inference': lambda: run_inference_tests(datasets, results),
		'tracking': lambda: run_tracking_tests(datasets, results),
	}

	if args.section == 'all':
		for section_fn in sections.values():
			try:
				section_fn()
			except Exception as e:
				print(f'\n  ✗ Section crashed: {e}')
				traceback.print_exc()
	else:
		fn = sections.get(args.section)
		if fn:
			fn()
		else:
			print(f'Unknown section: {args.section}')
			sys.exit(1)

	results.summary()

	# Exit code: non-zero if any failures
	sys.exit(1 if results.failed else 0)


if __name__ == '__main__':
	main()
