# PyChartAI

> AI-powered natural-language data analysis and chart generation for Python.
> Runs fully standalone — no pandasai required.

**20 chart types · 3 visualization backends · 8 LLM providers · RestrictedSandbox**

---

## Install

```bash
pip install pychartai
```

With optional extras:

```bash
pip install pychartai[pandasai]   # pandasai execution path
pip install pychartai[api]        # FastAPI REST server
```

## Quick Start

```python
import pychartai as pai

llm = pai.OllamaLLM(model="llama3.2")
pai.config.set({"llm": llm})

df = pai.read_csv("data/sales.csv")

# Natural-language query
print(df.chat("What is the average revenue by region?"))

# Chart generation
path = df.chat("Plot a bar chart of revenue by region", chart_type="plotly")

# Data profiling — no LLM needed
report = df.profile()
print(report.summary)

# Conversation memory
df.enable_memory()
df.chat("Total sales by region", agent="own")
df.chat("Now show that as a percentage", agent="own")
```

See the [Quick Start guide](QUICKSTART.md) for more examples.
