# API Reference

Full reference for all public symbols exported by `import pychartai as pai`.

---

## Configuration

### `pai.config`

Global configuration singleton.  Thread-safe (RLock-protected).

```python
pai.config.set({'llm': llm, 'verbose': True})
value = pai.config.get('llm')
pai.config.reset()   # restore defaults
```

**Recognised keys:**

| Key | Type | Default | Description |
|---|---|---|---|
| `llm` | `PyChartLLM` | `None` | Active LLM provider |
| `chart_backend` | `str` | `'seaborn'` | Default chart backend (`'seaborn'`, `'matplotlib'`, `'plotly'`) |
| `chart_theme` | `str` | `'light'` | Chart theme (`'light'`, `'dark'`, `'corporate'`, `'minimal'`) |
| `charts_output_dir` | `str` | `'exports/charts'` | Directory for saved chart files |
| `max_retries` | `int` | `3` | LLM call retry count |
| `llm_timeout` | `int` | `60` | Per-call LLM timeout in seconds |
| `verbose` | `bool` | `False` | Print generated code and retry messages |

---

## LLM Providers

All providers inherit from `PyChartLLM` and are interchangeable.

### `pai.PyChartLLM(model, *, api_key=None, base_url=None, temperature=0.1, max_tokens=2048)`

Universal provider.  `model` must be `'provider/model_name'`.

```python
llm = pai.PyChartLLM(model='openai/gpt-4o')
llm = pai.PyChartLLM(model='ollama/llama3.2')
```

**Properties:**

- `llm.last_usage` → `dict` with `prompt_tokens`, `completion_tokens`, `total_tokens`, `model` (empty for local models)

### Named convenience classes

| Class | Default model | Key env var |
|---|---|---|
| `OllamaLLM` | `llama3.2` | — |
| `OpenAILLM` | `gpt-4o` | `OPENAI_API_KEY` |
| `AnthropicLLM` | `claude-3-5-sonnet-20241022` | `ANTHROPIC_API_KEY` |
| `GeminiLLM` | `gemini-2.0-flash` | `GEMINI_API_KEY` |
| `GitHubLLM` | `gpt-4.1` | `GITHUB_TOKEN` |
| `DeepSeekLLM` | `deepseek-chat` | `DEEPSEEK_API_KEY` |
| `QwenLLM` | `qwen-turbo` | `DASHSCOPE_API_KEY` |
| `GenericLLM` | — | `OPENAI_API_KEY` |

---

## SmartDataFrame

### `pai.SmartDataFrame(df, *, config=None, schema=None, chart_library=None)`

A transparent proxy around a `pd.DataFrame` with `.chat()` and `.profile()` methods.

```python
sdf = pai.SmartDataFrame(raw_df)
sdf = pai.SmartDataFrame(raw_df, schema=my_schema, chart_library='plotly')
```

### `.chat(query, *, agent=None, chart_library=None, explain=False, on_progress=None, extra_dfs=None, sandbox=None)`

Ask a natural-language question or request a chart.

| Param | Type | Default | Description |
|---|---|---|---|
| `query` | `str` | required | Natural-language question |
| `agent` | `str\|None` | `None` | Execution backend: `None`/`'own'` (default), `'pandasai'`, `'sandbox'` |
| `chart_library` | `str` | from config | Override backend: `'seaborn'`, `'matplotlib'`, `'plotly'` |
| `explain` | `bool` | `False` | Append LLM explanation to result |
| `on_progress` | `callable` | `None` | `fn(stage, detail)` — stages: `'classifying'`, `'generating'`, `'executing'`, `'formatting'` |
| `extra_dfs` | `list\|dict` | `None` | Additional DataFrames to include (own agent only) |
| `sandbox` | `Sandbox` | `None` | Custom sandbox instance |

**Returns:** result string or chart file path.

### `.chat_stream(query, *, sandbox=None, chart_library=None)`

Stream result tokens.  Yields `StreamEvent` objects.

### `.profile()`

Auto-EDA without LLM.  Returns a `ProfileReport`.

```python
report = df.profile()
print(report.summary)        # dict of overall stats
print(report.numeric_stats)  # per-column numeric stats
print(report.missing)        # missing value counts
print(report.correlations)   # correlation matrix
```

### `.enable_memory(window_size=10)`

Enable sliding-window conversation memory for multi-turn queries.

### `.add_skill(fn)`

Inject a `@pai.skill`-decorated callable into the agent context.

---

## PyChartAgent

### `pai.PyChartAgent(*, llm=None, chart_backend='seaborn', charts_output_dir='exports/charts', verbose=False, max_retries=None, memory=None)`

Pandasai-independent NL → code → sandbox agent.

### `.chat(query, df, *, sandbox=None, chart_backend=None, explain=False, on_progress=None)`

Execute a query.  Raises `ValueError` (empty df), `TimeoutError` (LLM timeout), `RuntimeError` (max retries exceeded).

**Properties:**

- `.last_transformation` → `TransformationLog(query, generated_code, intent, attempts, success, error)`
- `.last_usage` → `dict` — token counts from the most recent LLM call

---

## Sandboxes

### `pai.RestrictedSandbox(*, allow_imports=None)`

Default in-process sandbox using RestrictedPython.

```python
sandbox = pai.RestrictedSandbox()
sandbox = pai.RestrictedSandbox(allow_imports=('numpy', 'pandas', 'scipy'))
```

### `pai.DockerSandbox(image='pychartai-sandbox:latest', memory='512m', timeout=60)`

Isolated Docker container sandbox.  Requires Docker Desktop.

```python
sandbox = pai.DockerSandbox()
sandbox.start()
result = agent.chat('...', df, sandbox=sandbox)
sandbox.stop()

# or as a context manager
with pai.DockerSandbox() as sandbox:
    result = agent.chat('...', df, sandbox=sandbox)
```

---

## Skills

### `@pai.skill`

Decorator that marks a function as injectable into the LLM context.

```python
@pai.skill
def top_n_products(df, n: int = 5) -> str:
    """Return the top N products by revenue."""
    return df.nlargest(n, 'revenue')[['product', 'revenue']].to_string()

sdf.add_skill(top_n_products)
```

---

## Schema

### `pai.Schema(columns=None)`
### `pai.Column(name, *, dtype=None, description=None, unit=None, allowed_values=None)`

Semantic layer that describes DataFrame columns to the LLM.

```python
schema = pai.Schema(columns=[
    pai.Column('revenue', dtype='float', description='Monthly revenue in USD', unit='USD'),
    pai.Column('region',  dtype='str',   description='Sales region',
               allowed_values=['North', 'South', 'East', 'West']),
])
sdf = pai.SmartDataFrame(df, schema=schema)
```

---

## Chart Themes

### `pai.ChartTheme(name, *, palette=None, style=None, context=None, font_scale=1.0)`

Built-in themes: `'light'`, `'dark'`, `'corporate'`, `'minimal'`.

```python
pai.config.set({'chart_theme': 'dark'})
```

---

## Connections

### `pai.connect(source, *, type=None)`
### `pai.read_csv(path, **kwargs)` → `SmartDataFrame`
### `pai.read_excel(path, **kwargs)` → `SmartDataFrame`
### `pai.read_json(path, **kwargs)` → `SmartDataFrame`
### `pai.read_parquet(path, **kwargs)` → `SmartDataFrame`
### `pai.SQLConnection(connection_string, *, query=None, table=None, **kwargs)`

Generic SQLAlchemy connection. `.load()` returns a `pd.DataFrame`.

---

## Database connectors

All connectors subclass `SQLConnection`. Provide either `table=` or `query=`.
Call `.load()` to fetch a `pd.DataFrame`, or `.list_tables()` to inspect the schema.

### `pai.PostgreSQLConnection(host, database, user, password, *, port=5432, schema='public', table=None, query=None)`
Requires `pip install pychartai[db-postgres]`.

### `pai.MySQLConnection(host, database, user, password, *, port=3306, table=None, query=None)`
Requires `pip install pychartai[db-mysql]`.

### `pai.SnowflakeConnection(account, user, password, warehouse, database, *, schema='PUBLIC', table=None, query=None)`
Requires `pip install pychartai[db-snowflake]`.

### `pai.BigQueryConnection(project_id, dataset_id, *, credentials_path=None, table=None, query=None)`
Requires `pip install pychartai[db-bigquery]`. Uses Application Default Credentials when `credentials_path` is omitted.

### `pai.RedshiftConnection(host, database, user, password, *, port=5439, cluster=None, schema='public', table=None, query=None)`
Requires `pip install pychartai[db-redshift]`. Redshift is Postgres-compatible.

```python
conn = pai.PostgreSQLConnection(
    host='localhost', database='analytics',
    user='postgres', password='secret', table='sales',
)
df = conn.load()
tables = conn.list_tables()
sdf = pai.SmartDataFrame(df)
```

---

## Cloud storage connectors

All connectors subclass `BaseConnection`. File format is auto-detected from the
key/extension; override with `file_format=` ('csv', 'parquet', 'json', 'xlsx').
Call `.load()` to fetch a `pd.DataFrame`.

### `pai.S3Connection(uri=None, *, bucket=None, key=None, file_format=None, aws_access_key_id=None, aws_secret_access_key=None, region_name=None, **kwargs)`
Load from AWS S3. Accepts `s3://bucket/key` or `bucket=`/`key=`. Requires `pip install pychartai[cloud-s3]`.

### `pai.GCSConnection(uri=None, *, bucket=None, blob=None, file_format=None, credentials_path=None, project=None, **kwargs)`
Load from Google Cloud Storage. Accepts `gs://bucket/blob`. Requires `pip install pychartai[cloud-gcs]`.

### `pai.AzureBlobConnection(container, blob, *, account_url=None, connection_string=None, credential=None, file_format=None, **kwargs)`
Load from Azure Blob Storage. Requires `pip install pychartai[cloud-azure]`.

### `pai.GoogleSheetsConnection(spreadsheet_id, *, sheet_name=None, credentials_path=None, header_row=0, **kwargs)`
Load a worksheet from Google Sheets. Requires `pip install pychartai[cloud-gsheets]`.

```python
conn = pai.S3Connection('s3://my-bucket/data/sales.csv')
sdf = pai.SmartDataFrame(conn.load())
```

---

## Cache

### `pai.ResponseCache(path='cache/', *, ttl=None)`

SHA-256 keyed file-based response cache.

```python
cache = pai.ResponseCache()
pai.config.set({'cache': cache})
```

---

## Pipeline

### `pai.Pipeline(steps=None)`

6-step extensible pipeline: `ValidateInput → InjectSchema → InjectSkills → CacheLookup → CallAnalyzer → CacheStore`.

---

## Streaming

### `pai.chat_stream(query, df, *, llm=None, sandbox=None)`

Standalone streaming function.  Yields `StreamEvent(type, value)` where `type` is `'token'`, `'result'`, or `'error'`.

---

## ConversationMemory

### `pai.ConversationMemory(window_size=10, *, max_result_chars=2000)`

Sliding-window multi-turn context store.

---

## DataProfiler

### `pai.DataProfiler()`
### `.profile(df)` → `ProfileReport`

---

## Error hints

### `pai.get_hint(error_message)` → `str | None`

Map a raw exception message to an actionable fix suggestion.

```python
try:
    df.chat('...')
except Exception as exc:
    hint = pai.get_hint(str(exc))
    if hint:
        print(f'Suggestion: {hint}')
```
