# Getting Started

Get up and running with pychartai in 5 minutes.

## Install

```bash
# Core (no visualization backend)
pip install pychartai

# With Plotly interactive charts
pip install pychartai[viz-plotly]

# With a specific database connector
pip install pychartai[db-postgres]      # PostgreSQL
pip install pychartai[db-mysql]         # MySQL
pip install pychartai[cloud-s3]         # AWS S3
pip install pychartai[cloud-gcs]        # Google Cloud Storage

# Everything at once
pip install pychartai[dev,db-all,cloud-all,viz-plotly,api]
```

## Setup an LLM

### Local (Ollama — free, no API key)

```bash
# Install Ollama from https://ollama.com
ollama pull llama3.2
ollama serve    # start in background
```

### Cloud (OpenAI, Anthropic, Gemini, etc.)

Set an environment variable with your API key:

```bash
export OPENAI_API_KEY=sk-...
export ANTHROPIC_API_KEY=sk-ant-...
export GEMINI_API_KEY=AIza...
```

## First query

```python
import pychartai as pai

# Configure the LLM
llm = pai.OllamaLLM(model='llama3.2')  # or OpenAILLM, etc.
pai.config.set({'llm': llm})

# Load data
df = pai.read_csv('data/sales.csv')

# Ask a question
print(df.chat('What is the average revenue by region?'))

# Generate a chart
path = df.chat('Bar chart of revenue by region', chart_library='seaborn')
print(f'Chart saved to: {path}')

# Data profiling (no LLM needed)
report = df.profile()
print(report.summary)
```

## Multi-turn conversation

```python
# Enable memory for follow-up questions
df.enable_memory()

df.chat('Total revenue by region')
df.chat('Now show that as a percentage')  # remembers prior context
df.chat('Which is the top region?')
```

## From a database

```python
# PostgreSQL
conn = pai.PostgreSQLConnection(
    host='localhost', database='analytics',
    user='postgres', password='secret', table='sales',
)
df = pai.SmartDataFrame(conn.load())

# BigQuery
conn = pai.BigQueryConnection(
    project_id='my-project', dataset_id='analytics',
    credentials_path='/path/to/service-account.json', table='sales',
)
df = pai.SmartDataFrame(conn.load())

# S3 (AWS, MinIO, Cloudflare R2, etc.)
conn = pai.S3Connection('s3://my-bucket/data/sales.csv')
df = pai.SmartDataFrame(conn.load())
```

## Key features

| Feature | How to use |
|---|---|
| **Conversation memory** | `df.enable_memory()` — multi-turn context |
| **Data profiling** | `df.profile()` — auto-EDA without LLM |
| **Progress tracking** | `df.chat(..., on_progress=lambda s, d: print(s, d))` |
| **PII redaction** | `redactor = pai.DataRedactor(); pai.config.set({'redactor': redactor})` |
| **Result explanations** | `df.chat(..., explain=True)` |
| **Dashboard** | `df.dashboard('overview of sales')` — multiple related charts |
| **Token tracking** | `agent.last_usage` — cost accounting |
| **Streaming** | `df.chat_stream(...)` — token-by-token streaming |

## Next steps

- **[API Reference](api_reference.md)** — full documentation of all classes and methods
- **[Migration from pandasai](migration.md)** — side-by-side examples
- **[Installation Guide](installation.md)** — all optional extras explained
- **[Contributing](../CONTRIBUTING.md)** — report issues, submit PRs

## Troubleshooting

**"No LLM configured"** → call `pai.config.set({'llm': ...})` first

**"Ollama not running"** → run `ollama serve` in another terminal

**"Module not found" (e.g. boto3, psycopg2)** → install the extra: `pip install pychartai[cloud-s3]`, `pip install pychartai[db-postgres]`

**"RestrictedPython not installed"** → unit tests skip RestrictedPython; integration tests require the dev extras: `pip install pychartai[dev]`
