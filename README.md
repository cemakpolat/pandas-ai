# pychartai

> AI-powered natural-language data analysis and chart generation for Python.
> Runs fully standalone — no pandasai required.

**20 chart types · 3 visualization backends · Unified LiteLLM backend · RestrictedSandbox · Skills · Schema · Cache · Pipeline · Connections · Conversation Memory · Auto-EDA · Chart Themes · Result Explanations**

---

## What's New in v0.4.0

- **Conversation memory** — `df.enable_memory()` stores last N query-result turns for follow-up queries
- **Data profiling / auto-EDA** — `df.profile()` returns summary stats, missing values, correlations, duplicate detection — no LLM needed
- **Natural language explanations** — `df.chat('revenue by region', explain=True, agent='own')` appends an LLM-generated plain-English explanation
- **Chart themes** — `pai.config.set({'chart_theme': 'dark'})` — 4 built-in themes (light, dark, corporate, minimal) + custom themes
- **Error suggestions** — 40+ error patterns mapped to actionable fix messages (e.g. "Ollama not running → Run: `ollama serve`")
- **Transformation tracking** — `agent.last_transformation` exposes generated code, intent, attempt count for debugging
- **Improved intent classification** — all 20 chart types recognized in keyword matching + optional LLM classifier fallback
- **Export formats** — PDF/SVG export for matplotlib/seaborn, JSON data export from Plotly
- **Jupyter integration** — `_repr_html_()` for rich notebook rendering of SmartDataFrame
- All 379+ tests pass (281 unit tests + 98 chart backend tests, no LLM required)

---

## Features

| Category | Details |
|---|---|
| **LLM Providers** | Unified LiteLLM backend: Ollama (local) · OpenAI · DeepSeek · GitHub Models · Gemini · Anthropic · Qwen · Custom |
| **Execution modes** | RestrictedSandbox (default) · DockerSandbox · opt-in pandasai.Agent |
| **Chart backends** | Seaborn · Matplotlib · Plotly (interactive HTML) |
| **Chart types** | 20: area · bar · box · bubble · count · ecdf · funnel · heatmap · histogram · kde · line · pairplot · pie · regression · scatter · stacked_bar · step · strip · swarm · violin |
| **Skills** | `@skill` decorator — inject custom Python callables into LLM context |
| **Schema** | `Schema` + `Column` — semantic layer: column descriptions, units, allowed values |
| **Cache** | SHA-256 keyed file-based `ResponseCache` — skip redundant LLM calls |
| **Pipeline** | 6-step extensible `Pipeline`: ValidateInput → InjectSchema → InjectSkills → CacheLookup → CallAnalyzer → CacheStore |
| **Connections** | CSV · Excel · JSON · Parquet · SQL · PostgreSQL · MySQL · Snowflake · BigQuery · Redshift · S3 · GCS · Azure Blob · Google Sheets |
| **Streaming** | `chat_stream()` yields `StreamEvent` tokens as they arrive |
| **Memory** | `enable_memory(window_size=10)` — sliding-window conversation context for follow-up queries |
| **Profiling** | `df.profile()` — auto-EDA: stats, missing values, correlations, duplicates (no LLM) |
| **Themes** | 4 built-in (light/dark/corporate/minimal) + custom `ChartTheme` |
| **Explanations** | `explain=True` — LLM-generated plain-English result explanations |
| **Error hints** | 40+ error patterns → actionable fix suggestions |
| **Tracking** | `agent.last_transformation` — generated code, intent, retries |
| **Jupyter** | `_repr_html_()` for rich notebook rendering |
| **Datasets** | 7 built-ins: sales · weather · ecommerce · health · energy · analytics · stocks |
| **Tests** | 379+ tests (281 unit + 98 chart) — no LLM required |

---

## Quick Start

```bash
pip install pychartai
```

```python
import pychartai as pai

llm = pai.OllamaLLM(model='llama3.2')
pai.config.set({'llm': llm})

df = pai.read_csv('data/use_cases/sales.csv')

# Text query — RestrictedSandbox by default (no pandasai needed)
print(df.chat('What is revenue by region?'))

# Chart generation
path = df.chat('Plot a bar chart of revenue by product', chart_type='seaborn')
print(path)  # exports/charts/seaborn_20260327_....png

# Data profiling — no LLM needed
report = df.profile()
print(report.summary)

# Conversation memory — follow-up queries
df.enable_memory()
df.chat('Total revenue by region', agent='own')
df.chat('Now show that as a percentage', agent='own')

# Dark theme charts
pai.config.set({'chart_theme': 'dark'})
df.chat('Bar chart of revenue by region', chart_type='seaborn')
```

Or via Makefile:

```bash
git clone https://github.com/yourusername/pychartai
cd pychartai
make venv && make prepare-data
make demo-advanced MODEL=llama3.2
```

---

## Table of Contents

1. [Installation](#installation)
2. [Execution Modes](#execution-modes)
3. [API Overview](#api-overview)
4. [Advanced Features](#advanced-features)
5. [LLM Providers](#llm-providers)
6. [Visualization Backends and Chart Types](#visualization-backends-and-chart-types)
7. [Streaming](#streaming)
8. [pychartai vs pandasai](#pychartai-vs-pandasai)
9. [Benchmarks](#benchmarks)
10. [Architecture](#architecture)
11. [Project Structure](#project-structure)
12. [Makefile Targets](#makefile-targets)
13. [Built-in Datasets](#built-in-datasets)
14. [Configuration](#configuration)
15. [Troubleshooting](#troubleshooting)

---

## Installation

```bash
# Standalone (no pandasai required):
pip install pychartai

# With optional pandasai.Agent orchestration:
pip install pychartai[pandasai]

# From source:
git clone https://github.com/yourusername/pychartai
cd pychartai
pip install -e .
```

**System requirements:** Python 3.9+

For local inference, install [Ollama](https://ollama.ai):

```bash
ollama serve            # keep running
ollama pull llama3.2   # ~2 GB
```

For cloud providers, set the relevant environment variable:

| Provider | Variable |
|---|---|
| OpenAI | `OPENAI_API_KEY` |
| GitHub Models | `GITHUB_TOKEN` |
| DeepSeek | `DEEPSEEK_API_KEY` |
| Google Gemini | `GEMINI_API_KEY` |
| Anthropic | `ANTHROPIC_API_KEY` |
| Alibaba Qwen | `DASHSCOPE_API_KEY` |

---

## Execution Modes

### Mode 1 — RestrictedSandbox (default)

No pandasai required. LLM generates Python code; `RestrictedPython` blocks dangerous built-ins.

```python
sdf.chat('What is revenue by region?')
# LLM → code → RestrictedPython sandbox → result
```

### Mode 2 — DockerSandbox (production)

Full container isolation — no filesystem, network, or process access outside the container.

```python
with pai.DockerSandbox() as sb:
    result = sdf.chat('Revenue by region', sandbox=sb)
```

### Mode 3 — PyChartAgent (built-in autonomous agent)

pychartai's own NL→code→sandbox agent with semantic intent classification, chart repair, and helper locking.

```python
agent = pai.PyChartAgent(
    llm=pai.OllamaLLM(model='llama3.2'),
    chart_backend='seaborn',
    charts_output_dir='exports/charts',
    max_retries=2,
)
result = agent.chat('Plot revenue by region', df)
```

### Mode 4 — pandasai.Agent (opt-in)

Use pandasai's SQL-first orchestrator if you already have it installed.

```python
sdf.chat('Revenue by region', use_agent=True)   # requires pip install pychartai[pandasai]
```

---

## API Overview

### 1. SmartDataFrame (recommended)

```python
import pychartai as pai

pai.config.set({'llm': pai.OllamaLLM(model='llama3.2')})

df = pai.read_csv('data/use_cases/sales.csv')             # SmartDataFrame
print(df.chat('What is the top product by revenue?'))     # text answer

path = df.chat('Bar chart of revenue by region', chart_type='seaborn')    # PNG
path = df.chat('Revenue trend over time',         chart_type='plotly')     # HTML
path = df.chat('Histogram of order values',       chart_type='matplotlib') # PNG

# Streaming
for event in df.chat_stream('Summarize this dataset'):
    if event.type == 'token':
        print(event.value, end='', flush=True)

# Advanced features
df.add_skill(my_function)
df.set_schema(my_schema)
```

### 2. PyChartAgent (direct)

```python
import pandas as pd
import pychartai as pai

llm = pai.OllamaLLM(model='llama3.2')
df = pd.read_csv('data/use_cases/sales.csv')

agent = pai.PyChartAgent(llm=llm, chart_backend='seaborn')
print(agent.chat('Top 3 products by revenue?', df))
print(agent.chat('Bar chart of revenue by region', df))
```

### 3. pychartai Compatibility Layer

```python
import pychartai, pandas as pd

pychartai.config.set({'llm': pychartai.OllamaLLM(model='llama3.2')})
df = pd.read_csv('data/use_cases/sales.csv')
sdf = pychartai.SmartDataframe(df, chart_library='seaborn')

sdf.chat('What is total revenue by product?')
sdf.chat('Plot a bar chart of sales by region', chart_library='plotly')

print(pychartai.available_backends())         # ('matplotlib', 'plotly', 'seaborn')
print(pychartai.available_charts('seaborn'))  # ('area_chart', 'bar_chart', ...)
```

### 4. Legacy DataAnalyzer API

```python
from pychartai import DataAnalyzer, DataManager

manager  = DataManager()
analyzer = DataAnalyzer(model_name='llama3.2', verbose=True)

df = manager.create_sample_data('sales', 'sales')
print(analyzer.analyze(df, 'Top 3 products by revenue?'))

for insight in analyzer.generate_insights(df, num_insights=3):
    print('-', insight)

analyzer.switch_model('mistral')
analyzer.switch_provider('gpt-4o', provider_type='openai')
```

### 5. CLI

```bash
python main.py --demo --model llama3.2
python main.py --example sales --model llama3.2
```

> Note: `main.py` is not included — use the Makefile targets instead.

Or use the Makefile:

```bash
make demo-advanced MODEL=llama3.2
make demo-advanced MODEL=llama3.2
```

---

## Advanced Features

### Skills

```python
import pychartai as pai

@pai.skill
def top_products(df, n: int = 5):
	'''Return the top-N products by revenue.'''
	return df.nlargest(n, 'revenue')[['product', 'revenue']]

pai.config.set({'llm': pai.OllamaLLM(model='llama3.2')})
sdf = pai.read_csv('data/use_cases/sales.csv')
sdf.add_skill(top_products)

print(sdf.chat('Show the top 3 products'))
```

Skills inject the function signature + docstring into the LLM system prompt and the full source into the code preamble.

### Schema / Semantic Layer

```python
import pychartai as pai

schema = pai.Schema(
	name='Monthly Sales',
	description='Aggregated sales from the ERP system.',
	columns={
		'revenue': pai.Column(description='Monthly revenue', unit='USD'),
		'region':  pai.Column(
			description='Geographic region',
			values=['North', 'South', 'East', 'West'],
		),
	},
)

sdf = pai.read_csv('data/use_cases/sales.csv')
sdf.set_schema(schema)
print(sdf.chat('Which region had the highest revenue in Q3?'))
```

### Response Cache

```python
import pychartai as pai

pai.config.set({'llm': pai.OllamaLLM(model='llama3.2'), 'cache': True})

sdf = pai.read_csv('data/use_cases/sales.csv')
sdf.chat('Average revenue?')   # calls LLM
sdf.chat('Average revenue?')   # cache hit — instant

cache = pai.ResponseCache('.my_cache')
print(cache.size())
cache.clear()
```

### Pipeline

```python
import pychartai as pai
from pychartai_core.pipeline import PipelineStep, PipelineContext

class QueryLogger(PipelineStep):
	name = 'QueryLogger'
	def run(self, ctx: PipelineContext) -> PipelineContext:
		print(f'[LOG] {ctx.get("query")}')
		return ctx

pai.config.set({'llm': pai.OllamaLLM(model='llama3.2')})
sdf = pai.read_csv('data/use_cases/sales.csv')
sdf.pipeline.add_step(QueryLogger(), before='CallAnalyzer')
sdf.chat('Total revenue?')
# prints: [LOG] Total revenue?
```

Default order: `ValidateInput → InjectSchema → InjectSkills → CacheLookup → CallAnalyzer → CacheStore`

Individual steps can be skipped: `step.skip()` / `step.enable()`.

### Connections

```python
import pychartai as pai

pai.config.set({'llm': pai.OllamaLLM(model='llama3.2')})

sdf = pai.connect(pai.CSVConnection('data/use_cases/sales.csv'))
sdf = pai.connect(pai.ExcelConnection('data/sales.xlsx', sheet_name='Q1'))
sdf = pai.connect(pai.JSONConnection('data/records.json'))
sdf = pai.connect(pai.ParquetConnection('data/events.parquet'))
sdf = pai.connect(pai.SQLConnection('sqlite:///db.sqlite3', query='SELECT * FROM sales'))

# Attach schema at load time
schema = pai.Schema(name='Sales', columns={'revenue': pai.Column(unit='USD')})
sdf = pai.connect(pai.CSVConnection('data/use_cases/sales.csv'), schema=schema)
```

### Database connectors

Dedicated connectors for the major databases. Each handles the connection
string, authentication, and schema inspection (`.list_tables()`).

```python
import pychartai as pai

pai.config.set({'llm': pai.OllamaLLM(model='llama3.2')})

# PostgreSQL — pip install pychartai[db-postgres]
conn = pai.PostgreSQLConnection(
    host='localhost', database='analytics',
    user='postgres', password='secret', table='sales',
)
sdf = pai.SmartDataFrame(conn.load())
print(conn.list_tables())

# MySQL / MariaDB — pip install pychartai[db-mysql]
conn = pai.MySQLConnection(
    host='localhost', database='shop',
    user='root', password='secret',
    query='SELECT * FROM orders WHERE year = 2024',
)

# Snowflake — pip install pychartai[db-snowflake]
conn = pai.SnowflakeConnection(
    account='xy12345.us-east-1', warehouse='COMPUTE_WH',
    database='ANALYTICS', schema='PUBLIC',
    user='admin', password='secret', table='SALES',
)

# Google BigQuery — pip install pychartai[db-bigquery]
conn = pai.BigQueryConnection(
    project_id='my-gcp-project', dataset_id='analytics',
    credentials_path='/path/to/service-account.json', table='sales',
)

# Amazon Redshift — pip install pychartai[db-redshift]
conn = pai.RedshiftConnection(
    host='cluster.xxxx.us-east-1.redshift.amazonaws.com',
    database='analytics', user='admin', password='secret', table='sales',
)

# All drivers at once
# pip install pychartai[db-all]
```

| Database | Connector | Extra |
|---|---|---|
| PostgreSQL | `PostgreSQLConnection` | `pychartai[db-postgres]` |
| MySQL / MariaDB | `MySQLConnection` | `pychartai[db-mysql]` |
| Snowflake | `SnowflakeConnection` | `pychartai[db-snowflake]` |
| Google BigQuery | `BigQueryConnection` | `pychartai[db-bigquery]` |
| Amazon Redshift | `RedshiftConnection` | `pychartai[db-redshift]` |
| SQLite / generic | `SQLConnection` | (built-in, needs `sqlalchemy`) |

### Cloud storage connectors

Load CSV / Parquet / JSON / Excel files directly from cloud object storage,
plus Google Sheets. File format is auto-detected from the extension (override
with `file_format=`).

```python
import pychartai as pai

# AWS S3 — pip install pychartai[cloud-s3]
conn = pai.S3Connection('s3://my-bucket/data/sales.csv')
sdf = pai.SmartDataFrame(conn.load())

# S3-compatible stores (MinIO, Cloudflare R2, Wasabi) via endpoint_url
conn = pai.S3Connection(
    's3://my-bucket/sales.csv',
    endpoint_url='https://<accountid>.r2.cloudflarestorage.com',
    aws_access_key_id='...', aws_secret_access_key='...',
)

# Google Cloud Storage — pip install pychartai[cloud-gcs]
conn = pai.GCSConnection('gs://my-bucket/events.parquet')

# Azure Blob Storage — pip install pychartai[cloud-azure]
conn = pai.AzureBlobConnection(
    container='data', blob='sales.csv',
    account_url='https://acct.blob.core.windows.net',
)

# Google Sheets — pip install pychartai[cloud-gsheets]
conn = pai.GoogleSheetsConnection(
    spreadsheet_id='1AbC...', sheet_name='Sheet1',
    credentials_path='/path/to/service-account.json',
)

# All cloud connectors at once
# pip install pychartai[cloud-all]
```

| Source | Connector | Extra |
|---|---|---|
| AWS S3 | `S3Connection` | `pychartai[cloud-s3]` |
| Google Cloud Storage | `GCSConnection` | `pychartai[cloud-gcs]` |
| Azure Blob Storage | `AzureBlobConnection` | `pychartai[cloud-azure]` |
| Google Sheets | `GoogleSheetsConnection` | `pychartai[cloud-gsheets]` |

---

## LLM Providers

| Class | Default model | Auth env var |
|---|---|---|
| `OllamaLLM` | `llama3.2` | None (local server at `localhost:11434`) |
| `GitHubLLM` | `gpt-4.1` | `GITHUB_TOKEN` |
| `OpenAILLM` | `gpt-4o` | `OPENAI_API_KEY` |
| `GeminiLLM` | `gemini-2.0-flash` | `GEMINI_API_KEY` |
| `AnthropicLLM` | `claude-3-5-sonnet-20241022` | `ANTHROPIC_API_KEY` |
| `QwenLLM` | `qwen-plus` | `DASHSCOPE_API_KEY` |
| `DeepSeekLLM` | `deepseek-chat` | `DEEPSEEK_API_KEY` |
| `GenericLLM` | user-defined | configurable |

```python
# Configure once, use everywhere
pai.config.set({'llm': pai.GitHubLLM(model='gpt-4.1')})

# Switch at runtime
analyzer.switch_provider('gpt-4o', provider_type='openai')
analyzer.switch_model('mistral')
```

**Model selector:**

| Model | Provider | Cost | RAM | Best for |
|---|---|---|---|---|
| `llama3.2` | Ollama | Free | ~4 GB | Best overall local |
| `mistral` | Ollama | Free | ~5 GB | Speed-first local |
| `gemma:2b` | Ollama | Free | ~1.6 GB | Low-memory |
| `gpt-4.1` | GitHub Models | Token-based | Cloud | Best cloud reasoning |
| `gpt-4o` | OpenAI | ~$0.01/1K | Cloud | High-quality cloud |
| `deepseek-chat` | DeepSeek | ~$0.0014/1K | Cloud | Cheapest cloud |
| `gemini-2.0-flash` | Gemini | Token-based | Cloud | Google ecosystem |

See [docs/MULTI_PROVIDER_GUIDE.md](docs/MULTI_PROVIDER_GUIDE.md) for full provider setup and cost comparison.

---

## Visualization Backends and Chart Types

```python
pai.config.set({'chart_backend': 'plotly'})     # global default
df.chat('Plot revenue', chart_type='seaborn')   # per-call override
```

| Backend | Output | Best for |
|---|---|---|
| `seaborn` | PNG | Publication-quality static images |
| `matplotlib` | PNG | Fully customizable static images |
| `plotly` | Interactive HTML | Dashboards and exploration |

### 20 Built-in Chart Types

| Helper | Description | Key args |
|---|---|---|
| `area_chart` | Filled area trends | x, y |
| `bar_chart` | Category comparisons | x, y |
| `box_chart` | Distribution + outliers | x, y |
| `bubble_chart` | Scatter + size-encoded 3rd variable | x, y, size |
| `count_chart` | Categorical frequency | x |
| `ecdf_chart` | Empirical CDF | column |
| `funnel_chart` | Conversion stages | labels, values |
| `heatmap` | Correlation matrix | (auto-inferred) |
| `histogram` | Value distribution | column |
| `kde_chart` | Smooth density estimate | column |
| `line_chart` | Time series / ordered trends | x, y |
| `pairplot_chart` | Multi-variable scatter matrix | (auto-inferred) |
| `pie_chart` | Part-to-whole proportions | labels, values |
| `regression_chart` | Scatter + linear trend + CI band | x, y |
| `scatter_chart` | Two-variable correlation | x, y |
| `stacked_bar_chart` | Stacked / normalized bars | x, y, stack |
| `step_chart` | Discrete / staircase intervals | x, y |
| `strip_chart` | Individual points per category | x, y |
| `swarm_chart` | Non-overlapping jittered scatter | x, y |
| `violin_chart` | Distribution shape by category | x, y |

All helpers accept `title`, `output_file`, `hue`, `backend` as optional kwargs and return the saved file path.

**Direct chart API:**

```python
from pychartai_core.visualization import bar_chart, line_chart, heatmap

path = bar_chart(df, x='product', y='revenue', title='Revenue by Product',
	output_file='exports/charts/rev.png', backend='seaborn')

path = line_chart(df, x='date', y='sales',
	output_file='exports/charts/trend.html', backend='plotly')
```

**Custom backend:**

```python
from pychartai_core.visualization_backends.base import ChartBackend, register_backend

class MyBackend(ChartBackend):
	name = 'mybackend'
	supported_charts = ('bar_chart', 'line_chart')
	def render(self, chart_type, df, output_file, **kwargs): ...

register_backend(MyBackend)
```

---

## Streaming

```python
import pychartai as pai

pai.config.set({'llm': pai.OllamaLLM(model='llama3.2')})
sdf = pai.read_csv('data/use_cases/sales.csv')

for event in sdf.chat_stream('Summarize this dataset'):
	if event.type == 'token':
		print(event.value, end='', flush=True)
	elif event.type == 'result':
		final = event.value
	elif event.type == 'error':
		print(f'Error: {event.value}')
```

`StreamEvent` fields: `type` (`'token'` | `'result'` | `'error'`), `value`.

---

## pychartai vs pandasai

| | pandasai | pychartai |
|---|---|---|
| Default execution | pandasai.Agent (required) | RestrictedSandbox — **no pandasai needed** |
| pandasai dependency | Core | Optional (`use_agent=True` only) |
| LLM abstraction | via LiteLLM | via LiteLLM (unified `PyChartLLM` + `LiteLLMProvider`, SOLID) |
| Chart backends | matplotlib | seaborn · matplotlib · plotly |
| Chart types | ~6 types | 20 types — same API on all backends |
| Interactive HTML charts | No | Yes (Plotly backend) |
| RestrictedPython sandbox | No | Yes — **default execution mode** |
| Docker sandbox | No | Yes |
| Skills | Yes (built-in) | Yes (custom implementation) |
| Schema layer | Yes (built-in) | Yes (custom `Schema` + `Column`) |
| Response cache | Yes (built-in) | Yes (SHA-256 file cache) |
| Pipeline extension | Yes (built-in) | Yes (custom 6-step, fully extensible) |
| Data connectors | Yes (built-in) | CSV · Excel · JSON · Parquet · SQL · PostgreSQL · MySQL · Snowflake · BigQuery · Redshift |
| Streaming | Yes | Yes — `chat_stream()` → `StreamEvent` |
| pandasai.Agent integration | Core | Opt-in via `use_agent=True` |
| Hosted cloud UI | Yes (BambooLLM) | No — self-hosted only |

---

## Benchmarks

> **Note:** Reproducible benchmark scripts are provided below.  Run them yourself on your hardware and model — results vary significantly by LLM, hardware, and dataset.  The numbers below are from a single development run with `llama3.2` on Apple M2 and should be treated as illustrative, not definitive.

```bash
python compare_agents_extensive.py   # 12-case NL analytics
python compare_agents_charts.py      # 5-case chart generation
```

### NL analytics correctness (12 cases, llama3.2, Apple M2)

| Metric | pychartai | pandasai.Agent |
|---|---|---|
| Correctness | 10 / 12 | 5 / 12 |
| Avg latency | 1.36s | 6.45s |

### Chart generation (5 cases, llama3.2, Apple M2)

| Metric | pychartai | pandasai.Agent |
|---|---|---|
| Pass rate | 5 / 5 | 3 / 5 |
| Avg latency | 2.97s | 13.85s |

---

## Architecture

```
pychartai v0.4.0
================

  Entry Points
    main.py (CLI)  ──  import pychartai as pai

  Public API
  ├── pai.read_csv()  pai.connect()  pai.config.set()
  ├── SmartDataFrame.chat()  .chat_stream()  .profile()  .enable_memory()
  └── PyChartAgent.chat()  .explain_result()  .last_transformation

  Advanced Feature Layer
  ├── Skills    @pai.skill / sdf.add_skill()
  ├── Schema    pai.Schema + pai.Column
  ├── Cache     pai.ResponseCache
  ├── Pipeline  ValidateInput → InjectSchema → InjectSkills → CacheLookup → CallAnalyzer → CacheStore
  ├── Conns     CSVConnection · ExcelConnection · JSONConnection · ParquetConnection · SQLConnection
  ├── Memory    ConversationMemory (sliding window)
  ├── Profiler  DataProfiler.profile(df) → ProfileReport
  └── Themes    ChartTheme (light · dark · corporate · minimal · custom)

  Core Analysis Engine  (DataAnalyzer + PyChartAgent)
  ├── Intent classification  (QueryIntent)
  ├── Semantic validation
  ├── NL → Python code generation
  ├── Code sanitization + chart repair
  └── Sandbox execution  (RestrictedSandbox / DockerSandbox)

  LLM Provider Layer (SOLID design + LiteLLM backend)
  ├── PyChartLLM (universal API — central abstraction)
  ├── LiteLLMProvider (only concrete impl — delegates to litellm.completion())
  ├── OllamaAvailabilityChecker (SRP — health checks only)
  └── Named subclasses for convenience
      ├── OllamaLLM · GitHubLLM · OpenAILLM · GeminiLLM
      └── AnthropicLLM · QwenLLM · DeepSeekLLM · GenericLLM

  Visualization System
  ├── 20 chart helpers
  ├── SeabornBackend     → PNG
  ├── MatplotlibBackend  → PNG
  └── PlotlyBackend      → interactive HTML
```

---

## Project Structure

```
pychartai/
├── Makefile                       # All make targets (no main.py — use make demo-advanced)
├── pyproject.toml                 # Build config (v0.4.0)
├── requirements.txt               # Pinned dependencies
│
├── src/
│   ├── pychartai_core/
│   │   ├── agent.py               # PyChartAgent (standalone NL→code→sandbox)
│   │   ├── analyzer.py            # DataAnalyzer + sandbox context builder
│   │   ├── providers.py           # 8 LLM provider classes
│   │   ├── smart_df.py            # SmartDataFrame.chat() / .chat_stream()
│   │   ├── sandbox.py             # RestrictedSandbox + DockerSandbox
│   │   ├── skills.py              # @skill decorator + Skill dataclass
│   │   ├── schema.py              # Schema + Column semantic layer
│   │   ├── cache.py               # ResponseCache (SHA-256 keyed file cache)
│   │   ├── pipeline.py            # Pipeline + 6 built-in steps
│   │   ├── connections.py         # Data source connectors + connect()
│   │   ├── visualization.py       # 20 chart helper functions
│   │   ├── streaming.py           # StreamEvent + chat_stream()
│   │   ├── memory.py              # ConversationMemory (sliding window)
│   │   ├── profiler.py            # DataProfiler + ProfileReport (auto-EDA)
│   │   ├── themes.py              # ChartTheme + 4 built-in themes
│   │   ├── error_hints.py         # 40+ error → suggestion mappings
│   │   └── visualization_backends/
│   │       ├── base.py            # ChartBackend ABC + ChartSpec
│   │       ├── catalog.py         # 20 built-in registrations
│   │       ├── seaborn_backend.py
│   │       ├── matplotlib_backend.py
│   │       └── plotly_backend.py
│   └── pychartai/
│       └── __init__.py            # pychartai compatibility layer
│
├── examples/
│   ├── advanced_features_demo.py      # Skills · Schema · Cache · Pipeline · Connections
│   ├── agent_comparison_demo.py       # pychartai vs pandasai.Agent (text)
│   ├── chart_comparison_demo.py       # Chart generation comparison
│   ├── execution_modes_examples.py    # All 3 execution modes
│   ├── llm_chart_examples.py          # LLM chart examples (CLI)
│   ├── pychartai_github_models_demo.py# GitHub AI Models demo
│   ├── pychartai_style_examples.py    # pychartai-style scenarios
│   └── basic_examples.py
│
├── tests/
│   ├── test_charts.py             # 98 chart backend tests (no LLM)
│   ├── test_features.py           # 70 feature unit tests (no LLM)
│   ├── test_chartai_api.py        # public API tests
│   ├── test_sandbox.py            # RestrictedSandbox tests
│   ├── test_streaming.py          # streaming tests
│   ├── test_docker_sandbox.py
│   ├── test_memory.py             # ConversationMemory tests
│   ├── test_profiler.py           # DataProfiler tests
│   ├── test_themes.py             # ChartTheme tests
│   ├── test_error_hints.py        # error hint tests
│   ├── test_smart_df_features.py  # memory/profile/repr_html tests
│   └── test_integration_extensive.py  # real-world LLM integration suite
│
├── docs/
│   ├── ARCHITECTURE_DEEPDIVE.md
│   ├── EXECUTION_FLOWS.md
│   ├── MULTI_PROVIDER_GUIDE.md
│   ├── CODE_GENERATION_EXPLAINED.md
│   ├── MODES_CHEATSHEET.md
│   └── DOCUMENTATION_INDEX.md
│
├── data/
│   ├── use_cases/             # 7 built-in CSV datasets
│   └── generators/
│       └── sample_data.py # generator functions (lazy-loaded by DataManager)
└── exports/charts/            # Generated chart output
```

---

## Makefile Targets

### Setup

| Target | Description |
|---|---|
| `make venv` | Create `.venv` and install all deps |
| `make install` | Install into active virtualenv |
| `make prepare-data` | Generate 7 CSV datasets |

### Tests — no LLM required

| Target | Description |
|---|---|
| `make test` | Comprehensive test suite: API + charts + new-features |
| `make test-api` | Public API pytest suite (12 tests) |
| `make test-charts` | All 3 backends visualization tests (98 tests) |
| `make test-charts-seaborn` | Seaborn backend only |
| `make test-charts-plotly` | Plotly backend only |
| `make test-charts-matplotlib` | Matplotlib backend only |
| `make test-unit-features` | Skills/Schema/Cache/Pipeline/Connections (70 tests) |
| `make test-new-features` | Memory/profiler/themes/error-hints/smart_df (51 tests) |

### Demos and LLM examples

| Target | Description |
|---|---|
| `make demo-advanced [MODEL=...] [BACKEND=...] [FEATURE=...]` | All advanced features demo |
| `make demo-advanced FEATURE=skills` | Skills decorator demo |
| `make demo-advanced FEATURE=schema` | Semantic layer demo |
| `make demo-advanced FEATURE=cache` | Response cache demo |
| `make demo-advanced FEATURE=pipeline` | 6-step pipeline demo |
| `make demo-advanced PROVIDER=github MODEL=gpt-4.1` | GitHub AI Models |
| `make demo-advanced PROVIDER=openai MODEL=gpt-4o` | OpenAI |
| `make demo-advanced PROVIDER=gemini MODEL=gemini-2.0-flash` | Gemini |
| `make test-llm-text [MODEL=...]` | LLM text-only queries (3 tests) |
| `make test-llm-seaborn [MODEL=...]` | LLM chart generation — seaborn (7 tests) |
| `make test-llm-plotly [MODEL=...]` | LLM chart generation — plotly (7 tests) |
| `make test-llm-matplotlib [MODEL=...]` | LLM chart generation — matplotlib (7 tests) |
| `make test-llm [MODEL=...] [BACKEND=...]` | All LLM+ chart tests |
| `make test-pychartai-style` | pychartai-style scenarios |
| `make test-pychartai-style-seaborn` | pychartai-style — seaborn |
| `make test-pychartai-style-plotly` | pychartai-style — plotly |
| `make gpt5` | GitHub AI Models integration (requires `.env`) |

### Variables

| Variable | Default | Values |
|---|---|---|
| `MODEL` | `llama3.2` | any Ollama model or cloud model name |
| `PROVIDER` | `ollama` | `ollama` · `openai` · `github` · `gemini` · `anthropic` · `qwen` · `deepseek` |
| `BACKEND` | `all` | `seaborn` · `matplotlib` · `plotly` · `all` |
| `DATASET` | `sales` | `sales` · `weather` · `ecommerce` · `health` · `energy` · `analytics` · `stocks` · `all` |
| `KEEP` | unset | `KEEP=1` to preserve chart files |
| `FEATURE` | `all` | `skills` · `schema` · `cache` · `pipeline` · `connections` · `charts` · `combined` · `all` |

---

## Built-in Datasets

```bash
make prepare-data
```

| Dataset | Key columns |
|---|---|
| `sales` | product, region, quantity, price, revenue, date |
| `weather` | city, temperature, humidity, pressure, date |
| `ecommerce` | product, category, orders, revenue, customer_id |
| `health` | age, bmi, risk_score, outcome, age_group |
| `energy` | source, production_gwh, co2_tonnes, region |
| `analytics` | page, users, sessions, bounce_rate, date |
| `stocks` | ticker, open, close, volume, date |

---

## Configuration

```python
pai.config.set({
	'llm':               pai.OllamaLLM(model='llama3.2'),
	'chart_backend':     'seaborn',           # 'seaborn' | 'matplotlib' | 'plotly'
	'charts_output_dir': 'exports/charts',
	'verbose':           False,
	'cache':             True,
	'cache_dir':         '.pychartai_cache',
})
```

```bash
# .env — placed in project root, auto-loaded on import
OPENAI_API_KEY=sk-your-openai-key
DEEPSEEK_API_KEY=your-deepseek-key
GITHUB_TOKEN=ghp-your-github-token
GEMINI_API_KEY=your-gemini-key
ANTHROPIC_API_KEY=sk-ant-your-key
DASHSCOPE_API_KEY=your-qwen-key
OLLAMA_BASE_URL=http://localhost:11434   # optional override
```

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `Cannot connect to Ollama` | Run `ollama serve` in another terminal |
| `Model not found on Ollama` | Run `ollama pull <model-name>` |
| Sandbox: invalid variable name | Context variables must not start with `_` |
| `OPENAI_API_KEY not set` | `export OPENAI_API_KEY="sk-..."` |
| `GITHUB_TOKEN not set` | `export GITHUB_TOKEN="ghp-..."` |
| Charts not saving | Run `make test-charts`; check `exports/charts/` |
| Out of memory (local) | Use `ollama pull gemma:2b` (~1.6 GB) |
| Slow responses | Use `mistral` locally or `deepseek-chat` for cloud |

---

## License

MIT License

---

## Resources

- [Ollama](https://ollama.ai) — local LLM runner
- [docs/MULTI_PROVIDER_GUIDE.md](docs/MULTI_PROVIDER_GUIDE.md) — provider setup
- [docs/ARCHITECTURE_DEEPDIVE.md](docs/ARCHITECTURE_DEEPDIVE.md) — component design
- [docs/EXECUTION_FLOWS.md](docs/EXECUTION_FLOWS.md) — execution pipeline diagrams
- [QUICKSTART.md](QUICKSTART.md) — 2-minute setup guide
