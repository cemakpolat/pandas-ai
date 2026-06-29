"""
advanced_features_demo.py — Runnable demo for all five advanced pychartai features.

Features demonstrated:
  1. Skills   — @skill decorator, add_skill(), LLM-invocable custom functions
  2. Schema   — Schema + Column semantic layer, set_schema()
  3. Cache    — ResponseCache, pai.config.set({'cache': True}), cache.size()
  4. Pipeline — PipelineStep / PipelineContext, sdf.pipeline.add_step()
  5. Connections — CSVConnection / ExcelConnection / JSONConnection, connect()

Usage:
  python examples/advanced_features_demo.py                             # all features (Ollama default)
  python examples/advanced_features_demo.py --feature skills            # one feature
  python examples/advanced_features_demo.py --model mistral --backend plotly
  python examples/advanced_features_demo.py --provider openai --model gpt-4o
  python examples/advanced_features_demo.py --provider github --model gpt-4.1
  python examples/advanced_features_demo.py --provider qwen --model qwen-plus
  python examples/advanced_features_demo.py --provider gemini --model gemini-2.0-flash
  python examples/advanced_features_demo.py --provider anthropic --model claude-3-5-sonnet-20241022
  python examples/advanced_features_demo.py --provider deepseek --model deepseek-chat
  python examples/advanced_features_demo.py --no-llm                   # unit mode (no LLM needed)

Environment variables by provider:
  Ollama:    (local server at http://localhost:11434, no auth required)
  OpenAI:    OPENAI_API_KEY=sk-...
  GitHub:    GITHUB_TOKEN=github_pat_...
  Qwen:      DASHSCOPE_API_KEY=...
  Gemini:    GEMINI_API_KEY=...
  Anthropic: ANTHROPIC_API_KEY=sk-ant-...
  DeepSeek:  DEEPSEEK_API_KEY=sk-...

Make targets:
  make demo-advanced                                  # equivalent to --feature all with Ollama
  make demo-advanced PROVIDER=openai MODEL=gpt-4o
  make demo-advanced PROVIDER=github MODEL=gpt-4.1
  make demo-advanced PROVIDER=qwen MODEL=qwen-plus
"""

import argparse
import os
import sys
import tempfile

# ---------------------------------------------------------------------------
# Ensure the src/ layout is on the path when running from the repo root.
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, os.path.join(_ROOT, 'src'))

import pandas as pd
import pychartai_core as pai
from pychartai_core.pipeline import PipelineStep, PipelineContext

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SEPARATOR = '\n' + '=' * 70 + '\n'


def _header(title: str) -> None:
	print(SEPARATOR)
	print(f'  DEMO: {title}')
	print('=' * 70)


def _make_sales_df() -> pd.DataFrame:
	"""Return a realistic 24-row sales DataFrame for demo use.

	Columns:
	  product, category, region, revenue, units, cost, profit,
	  discount_pct, month, quarter, sales_rep, customer_segment
	"""
	return pd.DataFrame({
		'product': [
			'Widget A', 'Widget B', 'Gadget X', 'Gadget Y', 'Widget A',
			'Gadget X', 'Widget C', 'Gadget Z', 'Widget B', 'Widget A',
			'Gadget X', 'Widget C', 'Gadget Y', 'Widget B', 'Gadget Z',
			'Widget A', 'Gadget X', 'Widget C', 'Gadget Z', 'Widget B',
			'Gadget Y', 'Widget A', 'Widget C', 'Gadget X',
		],
		'category': [
			'Widgets', 'Widgets', 'Gadgets', 'Gadgets', 'Widgets',
			'Gadgets', 'Widgets', 'Gadgets', 'Widgets', 'Widgets',
			'Gadgets', 'Widgets', 'Gadgets', 'Widgets', 'Gadgets',
			'Widgets', 'Gadgets', 'Widgets', 'Gadgets', 'Widgets',
			'Gadgets', 'Widgets', 'Widgets', 'Gadgets',
		],
		'region': [
			'North', 'South', 'East', 'West', 'East',
			'North', 'South', 'East', 'West', 'South',
			'West', 'North', 'East', 'East', 'North',
			'West', 'South', 'East', 'West', 'North',
			'South', 'North', 'West', 'South',
		],
		'revenue': [
			12000, 8500, 17000, 6000,  9500,
			15000, 7200, 11000, 5500, 13500,
			18000, 6800, 7500, 9000, 12500,
			10500, 16000, 8000, 9800,  7000,
			5800, 14000, 7800, 19000,
		],
		'units': [
			118,  79, 152,  58,  94,
			143,  74, 108,  52, 132,
			157,  66,  73,  88, 119,
			103, 148,  81,  95,  68,
			 56, 127,  77, 169,
		],
		'cost': [
			7000, 5000, 9500, 3800, 5600,
			8500, 4300, 6500, 3400, 7800,
			10000, 4100, 4600, 5400, 7200,
			6200, 9200, 4700, 5800, 4200,
			3600, 8200, 4600, 11000,
		],
		'profit': [
			5000, 3500, 7500, 2200, 3900,
			6500, 2900, 4500, 2100, 5700,
			8000, 2700, 2900, 3600, 5300,
			4300, 6800, 3300, 4000, 2800,
			2200, 5800, 3200, 8000,
		],
		'discount_pct': [
			 5,  8, 10,  5,  7,
			 5, 12,  8, 10,  5,
			 5, 10,  8, 12,  5,
			 8, 10,  5,  7, 12,
			10,  5,  8,  5,
		],
		'month': [
			1, 2, 1, 2, 3,
			4, 5, 4, 5, 6,
			7, 8, 7, 8, 9,
			10, 11, 10, 11, 12,
			3, 6, 9, 12,
		],
		'quarter': [
			1, 1, 1, 1, 1,
			2, 2, 2, 2, 2,
			3, 3, 3, 3, 3,
			4, 4, 4, 4, 4,
			1, 2, 3, 4,
		],
		'sales_rep': [
			'Alice', 'Bob', 'Alice', 'Carol', 'Bob',
			'Alice', 'Carol', 'Bob', 'Alice', 'Carol',
			'Alice', 'Bob', 'Carol', 'Alice', 'Bob',
			'Carol', 'Alice', 'Bob', 'Carol', 'Alice',
			'Bob', 'Carol', 'Alice', 'Bob',
		],
		'customer_segment': [
			'Enterprise', 'SMB', 'Enterprise', 'Consumer', 'SMB',
			'Enterprise', 'Consumer', 'SMB', 'Enterprise', 'Enterprise',
			'SMB', 'Consumer', 'Enterprise', 'SMB', 'Consumer',
			'SMB', 'Enterprise', 'Consumer', 'SMB', 'Consumer',
			'Enterprise', 'SMB', 'Consumer', 'Enterprise',
		],
	})


# ---------------------------------------------------------------------------
# 1. Skills demo
# ---------------------------------------------------------------------------

def demo_skills(llm_available: bool) -> None:
	_header('Skills — @skill decorator')

	@pai.skill
	def top_products(df, n: int = 5):
		'''Return the top-N products by revenue.'''
		return df.nlargest(n, 'revenue')[['product', 'revenue']]

	@pai.skill
	def revenue_per_unit(df):
		'''Calculate revenue per unit sold for each product.'''
		result = df.copy()
		result['rev_per_unit'] = (result['revenue'] / result['units']).round(2)
		return result[['product', 'rev_per_unit']]

	print('  Registered skills:')
	print(f'    - top_products: {top_products.description}')
	print(f'    - revenue_per_unit: {revenue_per_unit.description}')

	df = _make_sales_df()
	sdf = pai.SmartDataFrame(df)
	sdf.add_skill(top_products)
	sdf.add_skill(revenue_per_unit)

	print(f'\n  Active skills on SmartDataFrame: {[s.name for s in sdf.skills]}')
	print('  top_products result (called directly):')
	print(top_products(df, n=3).to_string(index=False))
	print('\n  revenue_per_unit result (called directly):')
	print(revenue_per_unit(df).to_string(index=False))

	if llm_available:
		print('\n  Asking LLM: "Show the top 2 products by revenue"')
		result = sdf.chat('Show the top 2 products by revenue')
		print(f'  LLM result: {result}')
	else:
		print('\n  [LLM skipped — pass --no-llm to skip or provide a running Ollama]')


# ---------------------------------------------------------------------------
# 2. Schema demo
# ---------------------------------------------------------------------------

def demo_schema(llm_available: bool) -> None:
	_header('Schema — Semantic Layer')

	schema = pai.Schema(
		name='Monthly Sales',
		description='Aggregated sales records from the ERP system.',
		columns={
			'product':  pai.Column(description='Product name', dtype='str'),
			'region':   pai.Column(
				description='Geographic sales region',
				values=['North', 'South', 'East', 'West'],
			),
			'revenue':  pai.Column(description='Monthly revenue in USD', unit='USD'),
			'units':    pai.Column(description='Units sold in the month', dtype='int'),
			'month':    pai.Column(description='Month number (1 = January)', dtype='int'),
		},
	)

	print('  Schema created:')
	print(f'    name: {schema.name}')
	print(f'    description: {schema.description}')
	print(f'    columns: {list(schema.columns.keys())}')
	print('\n  Prompt fragment that will be injected into every LLM call:')
	print(schema.to_prompt_fragment())

	df = _make_sales_df()
	sdf = pai.SmartDataFrame(df)
	sdf.set_schema(schema)
	print(f'\n  Schema attached to SmartDataFrame: {sdf.schema.name}')

	if llm_available:
		print('\n  Asking LLM: "Which region has the highest total revenue?"')
		result = sdf.chat('Which region has the highest total revenue?')
		print(f'  LLM result: {result}')
	else:
		print('\n  [LLM skipped — pass --no-llm to skip or provide a running Ollama]')


# ---------------------------------------------------------------------------
# 3. Cache demo
# ---------------------------------------------------------------------------

def demo_cache(llm_available: bool) -> None:
	_header('Cache — ResponseCache')

	df = _make_sales_df()

	with tempfile.TemporaryDirectory() as tmpdir:
		cache = pai.ResponseCache(tmpdir)
		print(f'  Cache directory: {tmpdir}')
		print(f'  Initial cache size: {cache.size()} entries')

		fingerprint = pai.ResponseCache.fingerprint(df)
		print(f'  DataFrame fingerprint (first 16 chars): {fingerprint[:16]}...')

		query = 'What is the total revenue?'

		# Manually put a value
		cache.put(query, fingerprint, 'Total revenue is $53,000')
		print(f'  After put: cache size = {cache.size()} entries')

		hit = cache.get(query, fingerprint)
		print(f'  Cache hit: "{hit}"')

		miss = cache.get('unknown question', fingerprint)
		print(f'  Cache miss (different query): {miss}')

		cache.clear()
		print(f'  After clear: cache size = {cache.size()} entries')

		if llm_available:
			pai.config.set({'cache': cache})
			sdf = pai.SmartDataFrame(df)
			print('\n  First call (LLM invoked):')
			r1 = sdf.chat(query)
			print(f'    Result: {r1}')
			print(f'    Cache size after first call: {cache.size()}')
			print('\n  Second call (cache hit, no LLM call):')
			r2 = sdf.chat(query)
			print(f'    Result: {r2}')
			assert r1 == r2, 'Cache should return the exact same value'
			print('    OK — both calls returned identical result')

	if llm_available:
		pai.config.set({'cache': None})


# ---------------------------------------------------------------------------
# LLM Provider Factory
# ---------------------------------------------------------------------------

def _create_llm(provider: str, model: str):
	"""Create an LLM instance for the given provider and model.

	Args:
		provider: Provider name (ollama, openai, github, qwen, gemini, anthropic, deepseek)
		model:    Model name or alias

	Returns:
		LLM instance configured for the provider

	Raises:
		ValueError: If provider is not supported or required credentials are missing
	"""
	provider = provider.lower()

	if provider == 'ollama':
		return pai.OllamaLLM(model=model)
	elif provider == 'openai':
		return pai.OpenAILLM(model=model)
	elif provider == 'github':
		return pai.GitHubLLM(model=model)
	elif provider == 'qwen':
		return pai.QwenLLM(model=model)
	elif provider == 'gemini':
		return pai.GeminiLLM(model=model)
	elif provider == 'anthropic':
		return pai.AnthropicLLM(model=model)
	elif provider == 'deepseek':
		return pai.DeepSeekLLM(model=model)
	else:
		raise ValueError(
			f'Unknown provider: {provider}. '
			f'Supported: ollama, openai, github, qwen, gemini, anthropic, deepseek'
		)


def _get_provider_display_name(provider: str) -> str:
	"""Get a friendly display name for the provider."""
	provider_names = {
		'ollama': 'Ollama (local)',
		'openai': 'OpenAI',
		'github': 'GitHub Models',
		'qwen': 'Alibaba Qwen (DashScope)',
		'gemini': 'Google Gemini',
		'anthropic': 'Anthropic Claude',
		'deepseek': 'DeepSeek',
	}
	return provider_names.get(provider.lower(), provider)


# ---------------------------------------------------------------------------
# 4. Pipeline demo
# ---------------------------------------------------------------------------

def demo_pipeline(llm_available: bool) -> None:
	_header('Pipeline — Custom Steps')

	df = _make_sales_df()
	sdf = pai.SmartDataFrame(df)

	print('  Default pipeline steps:')
	for step in sdf.pipeline._steps:
		print(f'    - {type(step).__name__}')

	# Define a custom logging step
	log_output = []

	class QueryLogger(PipelineStep):
		name = 'QueryLogger'

		def run(self, ctx: PipelineContext) -> PipelineContext:
			q = ctx.get('query', '')
			log_output.append(q)
			print(f'    [QueryLogger] intercepted query: {q!r}')
			return ctx

	# Insert before CallAnalyzer
	sdf.pipeline.add_step(QueryLogger(), before='CallAnalyzer')

	print(f'\n  Pipeline after inserting QueryLogger ({len(sdf.pipeline)} steps):')
	for step in sdf.pipeline._steps:
		print(f'    - {type(step).__name__}')

	if llm_available:
		print('\n  Running sdf.chat("What is the average revenue?") through the pipeline:')
		result = sdf.chat('What is the average revenue?')
		print(f'  Result: {result}')
		assert len(log_output) == 1, 'QueryLogger should have fired once'
		print(f'  Queries intercepted by QueryLogger: {log_output}')
	else:
		print('\n  [LLM skipped — pass --no-llm to skip or provide a running Ollama]')
		print(f'  Pipeline has {len(sdf.pipeline)} steps (1 custom + 6 default)')


# ---------------------------------------------------------------------------
# 5. Connections demo
# ---------------------------------------------------------------------------

def demo_connections(llm_available: bool) -> None:
	_header('Connections — Data Source Connectors')

	df = _make_sales_df()

	with tempfile.TemporaryDirectory() as tmpdir:
		# Write sample files
		csv_path     = os.path.join(tmpdir, 'sales.csv')
		excel_path   = os.path.join(tmpdir, 'sales.xlsx')
		json_path    = os.path.join(tmpdir, 'sales.json')
		parquet_path = os.path.join(tmpdir, 'sales.parquet')

		df.to_csv(csv_path, index=False)
		df.to_excel(excel_path, index=False)
		df.to_json(json_path, orient='records', indent=2)
		df.to_parquet(parquet_path, index=False)

		print(f'  Sample files written to: {tmpdir}')

		# CSV
		sdf_csv = pai.connect(pai.CSVConnection(csv_path))
		print(f'\n  CSVConnection → SmartDataFrame: {len(sdf_csv)} rows, columns={list(sdf_csv.columns)}')

		# Excel
		sdf_excel = pai.connect(pai.ExcelConnection(excel_path))
		print(f'  ExcelConnection → SmartDataFrame: {len(sdf_excel)} rows')

		# JSON
		sdf_json = pai.connect(pai.JSONConnection(json_path))
		print(f'  JSONConnection → SmartDataFrame: {len(sdf_json)} rows')

		# Parquet
		sdf_parquet = pai.connect(pai.ParquetConnection(parquet_path))
		print(f'  ParquetConnection → SmartDataFrame: {len(sdf_parquet)} rows')

		# With schema
		schema = pai.Schema(
			name='Sales',
			columns={'revenue': pai.Column(description='Monthly revenue', unit='USD')},
		)
		sdf_with_schema = pai.connect(pai.CSVConnection(csv_path), schema=schema)
		print(f'\n  connect() with schema: schema.name = {sdf_with_schema.schema.name}')

		# SQL (optional — requires sqlalchemy)
		try:
			import sqlalchemy  # noqa: F401
			db_url = f'sqlite:///{os.path.join(tmpdir, "sales.db")}'
			import sqlalchemy as sa
			engine = sa.create_engine(db_url)
			df.to_sql('sales', engine, index=False, if_exists='replace')
			sdf_sql = pai.connect(
				pai.SQLConnection(db_url, query='SELECT * FROM sales'),
			)
			print(f'  SQLConnection → SmartDataFrame: {len(sdf_sql)} rows')
		except ImportError:
			print('  SQLConnection: skipped (sqlalchemy not installed)')

		if llm_available:
			print('\n  Asking CSV-backed SmartDataFrame: "Which product has the highest revenue?"')
			result = sdf_csv.chat('Which product has the highest revenue?')
			print(f'  Result: {result}')


# ---------------------------------------------------------------------------
# Combined: all features wired together
# ---------------------------------------------------------------------------

def demo_all_together(llm_available: bool, backend: str) -> None:
	_header('Combined — Skills + Schema + Cache + Pipeline + Connection')

	@pai.skill
	def top_products(df, n: int = 3):
		'''Return the top-N products by revenue.'''
		return df.nlargest(n, 'revenue')[['product', 'revenue']]

	schema = pai.Schema(
		name='Sales',
		description='Monthly product sales data.',
		columns={
			'product': pai.Column(description='Product name'),
			'region':  pai.Column(description='Sales region', values=['North', 'South', 'East', 'West']),
			'revenue': pai.Column(description='Monthly revenue', unit='USD'),
		},
	)

	log_output = []

	class AuditStep(PipelineStep):
		name = 'AuditStep'

		def run(self, ctx: PipelineContext) -> PipelineContext:
			q = ctx.get('query', '')
			log_output.append(q)
			return ctx

	df = _make_sales_df()

	with tempfile.TemporaryDirectory() as tmpdir:
		csv_path = os.path.join(tmpdir, 'sales.csv')
		df.to_csv(csv_path, index=False)
		cache = pai.ResponseCache(tmpdir)

		if llm_available:
			pai.config.set({'cache': cache})

		sdf = pai.connect(pai.CSVConnection(csv_path), schema=schema)
		sdf.add_skill(top_products)
		sdf.pipeline.add_step(AuditStep())

		print(f'  Connected via CSVConnection: {len(sdf)} rows')
		print(f'  Schema: {sdf.schema.name}')
		print(f'  Skills: {[s.name for s in sdf.skills]}')
		print(f'  Pipeline steps: {[type(s).__name__ for s in sdf.pipeline._steps]}')

		# Direct skill invocation
		print('\n  top_products (direct call, n=2):')
		print(top_products(df, n=2).to_string(index=False))

		if llm_available:
			q = 'Show top 2 products by revenue'
			print(f'\n  First LLM call: {q!r}')
			r1 = sdf.chat(q)
			print(f'  Result: {r1}')
			print(f'  Audit log: {log_output}')
			print(f'  Cache size: {cache.size()}')

			print(f'\n  Second LLM call (same query — should hit cache): {q!r}')
			r2 = sdf.chat(q)
			print(f'  Result: {r2}')
			assert r1 == r2


# ---------------------------------------------------------------------------
# 7. Charts demo — chart-generating queries across backends
# ---------------------------------------------------------------------------

def demo_charts(llm_available: bool, backend: str) -> None:
	_header(f'Charts — Visualization Queries ({backend} backend)')

	df = _make_sales_df()
	sdf = pai.SmartDataFrame(df)

	print(f'  Dataset: {len(df)} rows × {len(df.columns)} columns')
	print(f'  Columns: {list(df.columns)}')

	_chart_questions = [
		('bar chart — revenue by region',
		 'Plot a bar chart of total revenue by region'),
		('bar chart — revenue by category',
		 'Plot a bar chart of total revenue by product category'),
		('line chart — monthly revenue trend',
		 'Plot a line chart of total revenue by month'),
		('bar chart — profit by sales rep',
		 'Plot a bar chart of total profit by sales rep'),
		('bar chart — revenue by customer segment',
		 'Plot a bar chart of total revenue by customer segment'),
	]

	_text_questions = [
		'Which region has the highest total revenue?',
		'Show total profit per sales rep ranked from highest to lowest',
		'What is the average profit per row in the dataset?',
		'Which customer segment generates the most total revenue?',
		'Show the average discount percentage for each product category',
		'Show total revenue grouped by quarter, sorted by quarter number',
	]

	print(f'\n  --- Text queries ({len(_text_questions)} questions) ---')
	for q in _text_questions:
		if llm_available:
			result = sdf.chat(q)
			print(f'  Q: {q}')
			print(f'  A: {result}\n')
		else:
			print(f'  Q: {q}  [skipped — no LLM]')

	print(f'\n  --- Chart queries ({len(_chart_questions)} charts, backend={backend}) ---')
	chart_paths = []
	for label, q in _chart_questions:
		if llm_available:
			result = sdf.chat(q, chart_library=backend)
			print(f'  [{label}]')
			if isinstance(result, str) and (result.endswith('.png') or result.endswith('.html')):
				chart_paths.append(result)
				import os as _os
				size_kb = round(_os.path.getsize(result) / 1024) if _os.path.isfile(result) else 0
				print(f'    → saved: {result} ({size_kb} KB)')
			else:
				print(f'    → result: {result}')
		else:
			print(f'  [{label}]  [skipped — no LLM]')

	if chart_paths:
		print(f'\n  Generated {len(chart_paths)} chart(s):')
		for p in chart_paths:
			print(f'    {p}')
		if any(p.endswith('.html') for p in chart_paths):
			print('  (HTML files are fully self-contained — open directly in any browser)')


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

_FEATURE_MAP = {
	'skills':      demo_skills,
	'schema':      demo_schema,
	'cache':       demo_cache,
	'pipeline':    demo_pipeline,
	'connections': demo_connections,
}


def main() -> None:
	parser = argparse.ArgumentParser(
		description='pychartai advanced features demo',
		formatter_class=argparse.RawDescriptionHelpFormatter,
		epilog=__doc__,
	)
	parser.add_argument(
		'--feature',
		choices=list(_FEATURE_MAP.keys()) + ['all', 'combined', 'charts'],
		default='all',
		help='Which feature demo to run (default: all)',
	)
	parser.add_argument(
		'--provider',
		default='ollama',
		choices=['ollama', 'openai', 'github', 'qwen', 'gemini', 'anthropic', 'deepseek'],
		help='LLM provider (default: ollama)',
	)
	parser.add_argument('--model',   default='llama3.2', help='Model name or alias for the provider')
	parser.add_argument('--backend', default='seaborn',  help='Chart backend (seaborn/matplotlib/plotly)')
	parser.add_argument(
		'--no-llm',
		action='store_true',
		help='Run in unit mode — skip LLM calls, only show feature APIs',
	)
	args = parser.parse_args()

	if not args.no_llm:
		try:
			llm = _create_llm(args.provider, args.model)
			pai.config.set({'llm': llm, 'chart_backend': args.backend, 'verbose': False})
			llm_available = True
			provider_name = _get_provider_display_name(args.provider)
			print(f'LLM: {provider_name}/{args.model}  |  backend: {args.backend}')
		except Exception as exc:
			print(f'[WARNING] Could not initialise LLM ({exc}). Running in --no-llm mode.')
			llm_available = False
	else:
		llm_available = False
		print('Running in no-LLM mode (unit demo only).')

	if args.feature == 'all':
		for name, fn in _FEATURE_MAP.items():
			fn(llm_available)
		demo_all_together(llm_available, args.backend)
		demo_charts(llm_available, args.backend)
	elif args.feature == 'combined':
		demo_all_together(llm_available, args.backend)
	elif args.feature == 'charts':
		demo_charts(llm_available, args.backend)
	else:
		_FEATURE_MAP[args.feature](llm_available)

	print(SEPARATOR)
	print('  Demo complete.')
	print(SEPARATOR)


if __name__ == '__main__':
	main()
