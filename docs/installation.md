# Installation

## Requirements

- Python 3.8–3.12
- pip 21+

## Standard install

```bash
pip install pychartai
```

This installs the core package with seaborn and matplotlib backends.  
No pandasai dependency required.

## Optional extras

| Extra | What it adds | Install command |
|---|---|---|
| `viz-plotly` | Interactive HTML charts via Plotly + static export via kaleido (~80 MB) | `pip install pychartai[viz-plotly]` |
| `pandasai` | Optional `pandasai.Agent` execution path | `pip install pychartai[pandasai]` |
| `api` | FastAPI REST server (`/chat`, `/profile`, `/health`) | `pip install pychartai[api]` |
| `dev` | Test suite, linting, all optional deps | `pip install pychartai[dev]` |
| `docs` | MkDocs + Material theme for building these docs | `pip install pychartai[docs]` |

Install multiple extras at once:

```bash
pip install "pychartai[viz-plotly,api]"
```

## Local LLM (Ollama — no API key needed)

```bash
# macOS / Linux
curl -fsSL https://ollama.com/install.sh | sh
ollama pull llama3.2
ollama serve          # keep running in a terminal tab
```

```python
import pychartai as pai

llm = pai.OllamaLLM(model='llama3.2')
pai.config.set({'llm': llm})
```

## Cloud LLM

Set the relevant environment variable before running:

```bash
export OPENAI_API_KEY=sk-...          # OpenAI
export ANTHROPIC_API_KEY=sk-ant-...   # Anthropic
export GEMINI_API_KEY=AIza...         # Google Gemini
export GITHUB_TOKEN=ghp_...           # GitHub Models (free tier available)
```

```python
import pychartai as pai

pai.config.set({'llm': pai.OpenAILLM(model='gpt-4o')})
# or
pai.config.set({'llm': pai.AnthropicLLM(model='claude-3-5-sonnet-20241022')})
```

## Verifying the install

```python
import pychartai as pai
print(pai.__version__)    # 0.4.0
```

No errors means the install is clean.  You do not need a live LLM to import the library.
