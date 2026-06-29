# Migrating from pandasai

This guide covers the differences between pandasai and pychartai so you can migrate existing code with minimal friction.

---

## Core differences

| | pandasai v3 | pychartai v0.4 |
|---|---|---|
| Dependency | Required as core | Optional (`pip install pychartai[pandasai]`) |
| Default execution | `pandasai.Agent` (raw exec) | `RestrictedSandbox` (RestrictedPython) |
| Import | `from pandasai import SmartDataframe` | `import pychartai as pai` |
| Chart backends | matplotlib | seaborn · matplotlib · plotly |
| Chart types | ~6 | 20 |
| Conversation memory | Enterprise tier | Free, built-in |
| Auto-EDA | Basic | Full (`df.profile()`) |

---

## Side-by-side migration

### Basic query

```python
# pandasai
from pandasai import SmartDataframe
from pandasai.llm import OpenAI

llm = OpenAI(api_token='sk-...')
df = SmartDataframe(df, config={'llm': llm})
result = df.chat('What is the average revenue by region?')
```

```python
# pychartai — equivalent
import pychartai as pai

pai.config.set({'llm': pai.OpenAILLM(model='gpt-4o')})
df = pai.SmartDataFrame(df)
result = df.chat('What is the average revenue by region?')
```

### Chart generation

```python
# pandasai
df.chat('Plot a bar chart of revenue by region')
```

```python
# pychartai — explicitly choose a backend
df.chat('Plot a bar chart of revenue by region', chart_library='seaborn')
df.chat('Plot a bar chart of revenue by region', chart_library='plotly')   # interactive HTML
```

### Reading a CSV

```python
# pandasai
from pandasai import read_csv
df = read_csv('data/sales.csv')
```

```python
# pychartai
df = pai.read_csv('data/sales.csv')   # identical API
```

### Config

```python
# pandasai
config = {
    'llm': llm,
    'verbose': True,
    'save_charts': True,
    'save_charts_path': 'exports/charts',
}
df = SmartDataframe(df, config=config)
```

```python
# pychartai — global config (set once, applies everywhere)
pai.config.set({
    'llm': llm,
    'verbose': True,
    'charts_output_dir': 'exports/charts',
})
df = pai.SmartDataFrame(df)
```

### Skills

```python
# pandasai
from pandasai.skills import skill

@skill
def top_n(df, n: int) -> str:
    return df.nlargest(n, 'revenue').to_string()
```

```python
# pychartai — identical decorator, different import
import pychartai as pai

@pai.skill
def top_n(df, n: int) -> str:
    return df.nlargest(n, 'revenue').to_string()

df = pai.SmartDataFrame(raw_df)
df.add_skill(top_n)
```

### Schema / semantic layer

```python
# pandasai
from pandasai.schemas import Schema
schema = Schema(...)
```

```python
# pychartai
schema = pai.Schema(columns=[
    pai.Column('revenue', dtype='float', description='Monthly revenue in USD'),
    pai.Column('region', dtype='str', description='Sales region'),
])
df = pai.SmartDataFrame(raw_df, schema=schema)
```

---

## Features only in pychartai

These have no pandasai equivalent:

- **`df.profile()`** — auto-EDA without LLM (stats, correlations, missing values, duplicates)
- **`DockerSandbox`** — fully isolated Docker container execution
- **`chart_library='plotly'`** — interactive HTML charts
- **`on_progress=` callback** — hook into generating / executing / formatting stages
- **`agent.last_usage`** — token counts from the most recent LLM call
- **`agent.last_transformation`** — generated code, intent, attempt count
- **Error hints** — 40+ error patterns mapped to actionable fix suggestions
- **Chart themes** — `pai.config.set({'chart_theme': 'dark'})`
- **FastAPI REST wrapper** — `pip install pychartai[api]`
- **`DataRedactor`** — PII detection/redaction before LLM calls (hash/mask/drop)
- **Dashboard generation** — `df.dashboard('overview of sales')` → multiple charts
- **Database connectors** — `PostgreSQLConnection`, `MySQLConnection`, `SnowflakeConnection`, `BigQueryConnection`, `RedshiftConnection`
- **Cloud storage connectors** — `S3Connection`, `GCSConnection`, `AzureBlobConnection`, `GoogleSheetsConnection`

---

## Keeping pandasai as a fallback

pychartai's own agent is the default, but you can opt in to pandasai orchestration per-call:

```python
pip install pychartai[pandasai]
```

```python
# Use pandasai.Agent for this specific call
result = df.chat('Revenue by region', agent='pandasai')

# Use pychartai's own agent (default)
result = df.chat('Revenue by region')
result = df.chat('Revenue by region', agent='own')  # explicit
```

---

## Removed / renamed APIs

| pandasai | pychartai equivalent |
|---|---|
| `SmartDataframe` (lowercase f) | `SmartDataFrame` (both work — alias provided) |
| `pandasai.Agent(dfs=[...])` | `pai.PyChartAgent()` + list of DataFrames passed to `.chat()` |
| `df.chat(..., config={...})` | `pai.config.set({...})` (global) or `SmartDataFrame(df, config={...})` |
