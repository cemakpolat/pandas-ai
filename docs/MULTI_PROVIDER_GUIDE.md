# Multi-Provider LLM Support for Advanced Demo

The advanced features demo supports multiple LLM providers via a **unified LiteLLM backend**. All providers share the same `LiteLLMProvider` implementation — adding a new provider requires only a new `PyChartLLM` subclass.

## Architecture

```
Your code          Provider class    LiteLLM model string       Backend
────────────────   ────────────────   ───────────────────────   ───────────────
OllamaLLM()     → PyChartLLM     → 'ollama/llama3.2'       → Ollama server
OpenAILLM()     → PyChartLLM     → 'gpt-4o'                → OpenAI API
GitHubLLM()     → PyChartLLM     → 'openai/gpt-4.1'        → models.github.ai
GeminiLLM()     → PyChartLLM     → 'gemini/gemini-2.0-flash'→ Google AI
AnthropicLLM()  → PyChartLLM     → 'anthropic/claude-...'  → Anthropic API
DeepSeekLLM()   → PyChartLLM     → 'deepseek/deepseek-chat'→ DeepSeek API
QwenLLM()       → PyChartLLM     → 'qwen-plus' + api_base  → DashScope API
GenericLLM()    → PyChartLLM     → custom model + base_url → Any endpoint
```

API keys are resolved in order: explicit `api_key=` argument → environment variable → litellm fallback.

## Supported Providers

- **ollama** — Local Ollama server (default)
- **openai** — OpenAI API (GPT-4, GPT-4o, etc.)
- **github** — GitHub AI Models API (GPT-4.1, GPT-5)
- **qwen** — Alibaba Qwen via DashScope
- **gemini** — Google Gemini
- **anthropic** — Anthropic Claude
- **deepseek** — DeepSeek API

## Setting Up Environment Variables

Each provider requires specific authentication credentials via environment variables:

### Ollama (Local)
No authentication required. Just ensure Ollama server is running:
```bash
ollama serve
```

### OpenAI
```bash
export OPENAI_API_KEY="sk-..."
```
Get your key from: https://platform.openai.com/api-keys

### GitHub Models
```bash
export GITHUB_TOKEN="github_pat_..."
```
Get your token from: https://github.com/settings/tokens (create a token with `repo` scope)

### Qwen (DashScope)
```bash
export DASHSCOPE_API_KEY="sk-..."
```
Get your key from: https://dashscope.console.aliyun.com/

### Gemini
```bash
export GEMINI_API_KEY="..."
```
Get your key from: https://aistudio.google.com/app/apikey

### Anthropic
```bash
export ANTHROPIC_API_KEY="sk-ant-..."
```
Get your key from: https://console.anthropic.com/account/keys

### DeepSeek
```bash
export DEEPSEEK_API_KEY="sk-..."
```
Get your key from: https://platform.deepseek.com/api/keys

## Usage Examples

### Command Line

**Default (Ollama/llama3.2):**
```bash
make demo-advanced
```

**Ollama with different model:**
```bash
make demo-advanced MODEL=llama3.1 BACKEND=seaborn
```

**OpenAI GPT-4o:**
```bash
make demo-advanced PROVIDER=openai MODEL=gpt-4o
```

**GitHub AI Models (GPT-4.1):**
```bash
make demo-advanced PROVIDER=github MODEL=gpt-4.1
```

**Qwen (Alibaba):**
```bash
make demo-advanced PROVIDER=qwen MODEL=qwen-plus
```

**Gemini (Google):**
```bash
make demo-advanced PROVIDER=gemini MODEL=gemini-2.0-flash
```

**Anthropic Claude:**
```bash
make demo-advanced PROVIDER=anthropic MODEL=claude-3-5-sonnet-20241022
```

**DeepSeek:**
```bash
make demo-advanced PROVIDER=deepseek MODEL=deepseek-chat
```

### Python Script

Direct script invocation with provider selection:

```bash
# OpenAI
python examples/advanced_features_demo.py \
  --provider openai \
  --model gpt-4o \
  --backend seaborn \
  --feature all

# GitHub AI Models
python examples/advanced_features_demo.py \
  --provider github \
  --model gpt-4.1 \
  --backend plotly \
  --feature skills

# Qwen
python examples/advanced_features_demo.py \
  --provider qwen \
  --model qwen-plus \
  --feature cache
```

## Run Unit Mode (No LLM Required)

To test the advanced features without calling any LLM, use the offline unit tests:

```bash
make test-unit-features
```

Or run feature-specific pytest suites directly:
```bash
.venv/bin/python -m pytest tests/test_features.py -v
```

## Troubleshooting

### "Could not initialise LLM" / "LLM error"
1. **Verify credentials:** Check that your `*_API_KEY` environment variable is set correctly.
2. **API key format:** Ensure you're using the correct API key (not API ID or other credentials).
3. **Network connectivity:** Verify internet connection for cloud-based providers.
4. **Ollama:** If using Ollama, ensure the server is running (`ollama serve`) and the model is pulled (`ollama pull llama3.2`).
5. **LiteLLM error messages:** Errors are wrapped as `RuntimeError: LLM error (model): ...` — the inner message from litellm identifies the root cause.

### "Unknown provider"
Check that you've spelled the provider name correctly (lowercase):
- Valid: `ollama`, `openai`, `github`, `qwen`, `gemini`, `anthropic`, `deepseek`

### Model Not Found
Each provider has different available models. Common models:

| Provider | Model Examples |
|----------|--------|
| Ollama | llama3.2, llama3.1, mistral, neural-chat |
| OpenAI | gpt-4o, gpt-4-turbo, gpt-4o-mini |
| GitHub | gpt-4.1, gpt-5 (preview) |
| Qwen | qwen-plus, qwen-turbo, qwen-max |
| Gemini | gemini-2.0-flash, gemini-1.5-pro |
| Anthropic | claude-3-5-sonnet-20241022, claude-3-haiku-20240307 |
| DeepSeek | deepseek-chat, deepseek-coder |

## Feature Demos

You can run individual features with different providers:

```bash
# Skills demo with Gemini
make demo-advanced PROVIDER=gemini MODEL=gemini-2.0-flash FEATURE=skills

# Schema demo with Claude
make demo-advanced PROVIDER=anthropic MODEL=claude-3-5-sonnet-20241022 FEATURE=schema

# Cache demo with Qwen
make demo-advanced PROVIDER=qwen MODEL=qwen-plus FEATURE=cache

# Combined demo with GPT-4o
make demo-advanced PROVIDER=openai MODEL=gpt-4o FEATURE=combined
```

## Cost Considerations

- **Ollama** — Free (local processing)
- **GitHub Models** — Free (preview)
- **DeepSeek** — Very cheap (~$0.27 per 1M input tokens)
- **Qwen** — Very affordable
- **Gemini** — Affordable
- **Anthropic** — Moderate cost
- **OpenAI** — Higher cost (GPT-4o ~ $0.03 per 1K input tokens)

## Notes

- The demo automatically handles API key management via environment variables.
- Each provider's response quality may vary; some features might work better with certain models.
- For reproducibility, consider fixing the model name rather than using latest/default versions.
