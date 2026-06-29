# PyChartAI v0.4.0 — Production-Ready Release

## Status: Ready for PyPI

**Date:** 2026-06-08  
**Version:** 0.4.0  
**License:** MIT  
**Python:** 3.8–3.12

---

## What's Included

A production-ready, security-first AI data analysis library offering:

### Core Features
- **PyChartAgent** — standalone NL→code→sandbox agent (no pandasai required)
- **20 chart types** across 3 backends (seaborn, matplotlib, plotly interactive)
- **8 LLM providers** directly via PyChartLLM + LiteLLM unified backend
- **14 data connectors** — local files + 5 databases + 4 cloud stores
- **RestrictedSandbox** (default) + **DockerSandbox** for execution isolation
- **Conversation memory** — multi-turn context with sliding-window
- **Auto-EDA profiling** — `df.profile()` without LLM

### Production Hardening
- **Rate-limit handling** with exponential backoff (4 retries, 2^n delay)
- **PII/data redaction** — built-in `DataRedactor` with hash/mask/drop strategies
- **Prompt-injection guards** — null-byte stripping, max-length enforcement, pattern detection
- **Cost tracking** — `agent.last_usage` exposes token counts
- **Structured logging** — `DEBUG`/`INFO`/`WARNING` records with query/intent/duration
- **Progress callbacks** — `on_progress=` hooks into generating/executing/formatting stages
- **Dashboard generation** — LLM decomposes queries into N complementary charts
- **Error hints** — 40+ patterns mapped to actionable fixes

### Data Sources
| Category | Connectors |
|---|---|
| Local files | CSV, Excel, JSON, Parquet |
| Databases | PostgreSQL, MySQL, Snowflake, BigQuery, Redshift, SQLite |
| Cloud storage | S3 (S3-compatible: MinIO/R2/Wasabi), Google Cloud Storage, Azure Blob Storage |
| SaaS | Google Sheets |

### Documentation
- **Getting Started** — 5-minute setup guide
- **Installation Guide** — all optional extras explained
- **Migration from pandasai** — side-by-side examples
- **API Reference** — complete class/method documentation
- **Architecture & Guides** — execution modes, LLM providers
- **Community** — CONTRIBUTING, SECURITY, CODE_OF_CONDUCT

### Test Coverage
- **260 unit tests pass** (core, connectors, features)
- **6 integration tests** for cloud connectors (gated behind `PYCHARTAI_CLOUD_IT=1`)
- **Zero external LLM required** for test suite
- **Integration tests use real emulators** (MinIO for S3, Azurite for Azure Blob, fake-gcs-server for GCS)

---

## How It Compares

### vs pandas-ai

| Dimension | pychartai | pandas-ai |
|---|---|---|
| Standalone | ✅ Yes | ❌ No |
| Default sandbox | ✅ RestrictedPython | ❌ raw exec() |
| Chart types | ✅ 20 | ⚠️ ~6 |
| Chart backends | ✅ 3 (interactive HTML) | ❌ 1 (matplotlib) |
| Data connectors | ✅ 14 | ⚠️ ~5 |
| Cost tracking | ✅ Yes | ❌ No |
| PII handling | ✅ Built-in | ❌ No |
| Rate-limit handling | ✅ Yes | ❌ No |
| Conversation memory | ✅ Free | ❌ Enterprise |
| Community size | ⚠️ Growing | ✅ Large |

**Positioning:** "Production-ready, security-first alternative to pandas-ai. Choose us for data privacy, flexible data sources, and rich visualization."

---

## Getting Started

```bash
pip install pychartai
ollama pull llama3.2 && ollama serve  # or set OPENAI_API_KEY
```

```python
import pychartai as pai

pai.config.set({'llm': pai.OllamaLLM(model='llama3.2')})
df = pai.read_csv('sales.csv')

# Analysis
print(df.chat('Average revenue by region?'))

# Visualization
df.chat('Bar chart of revenue by region', chart_library='seaborn')

# Profiling (no LLM)
print(df.profile().summary)

# Multi-turn
df.enable_memory()
df.chat('Total revenue')
df.chat('Show that as a percentage')  # remembers context
```

---

## Before First Deployment

- [ ] **Create PyPI account** at https://pypi.org/account/register/ (if needed)
- [ ] **Configure OIDC trusted publisher** in PyPI settings (Points to GitHub repo + `.github/workflows/publish.yml`)
- [ ] **Push to GitHub** and tag `v0.4.0`
- [ ] **GitHub Actions auto-publishes** to PyPI via OIDC (no API token in secrets)
- [ ] **Rotate/revoke** `.env` PAT if repo was private before

---

## Architecture Highlights

```
User query
  ↓
PyChartAgent (pandasai-independent)
  ├─ Intent classification (20 chart types + NL analysis)
  ├─ Prompt engineering (schema, sample rows, chart specs)
  ├─ LLM call with timeout + retry (exponential backoff)
  ├─ Code extraction & sanitization
  └─ Sandbox execution (RestrictedPython or Docker)
       ├─ RestrictedSandbox (default) — in-process, fast
       └─ DockerSandbox — isolated, secure
  ├─ Result validation
  ├─ Conversation memory recording
  └─ Optional LLM explanation
  ↓
Formatted result + metadata
  (tokens, duration, transformation log, visual asset path)
```

**Key design principle:** Fail fast with clear errors. Raise exceptions (not strings). Hint every error with 40+ pattern-matched suggestions.

---

## Optional Extras

```bash
# Visualization
pip install pychartai[viz-plotly]      # Interactive Plotly charts + kaleido export

# Databases
pip install pychartai[db-all]          # All 5 DB connectors
pip install pychartai[db-postgres]     # PostgreSQL only
pip install pychartai[db-mysql]        # MySQL / MariaDB
pip install pychartai[db-snowflake]    # Snowflake
pip install pychartai[db-bigquery]     # Google BigQuery
pip install pychartai[db-redshift]     # Amazon Redshift

# Cloud Storage
pip install pychartai[cloud-all]       # All cloud + SaaS
pip install pychartai[cloud-s3]        # AWS S3 + S3-compatible (MinIO, R2, Wasabi)
pip install pychartai[cloud-gcs]       # Google Cloud Storage
pip install pychartai[cloud-azure]     # Azure Blob Storage
pip install pychartai[cloud-gsheets]   # Google Sheets

# pandasai integration
pip install pychartai[pandasai]        # Use pandasai.Agent as alternate backend

# REST API
pip install pychartai[api]             # FastAPI server

# Documentation
pip install pychartai[docs]            # Build docs locally

# Full dev
pip install pychartai[dev,db-all,cloud-all,viz-plotly]
```

---

## Testing

```bash
# Unit tests (no LLM needed)
pytest tests/ --ignore=tests/integration

# With cloud integration (requires emulators)
docker compose -f docker-compose.cloud-test.yml up -d
PYCHARTAI_CLOUD_IT=1 pytest tests/integration/
docker compose -f docker-compose.cloud-test.yml down -v

# Full suite (CI-style)
pytest tests/

# Coverage report
pytest --cov=src --cov-report=html
```

**Current:** 260 unit tests + 4 integration tests (cloud connectors against local emulators).

---

## Known Limitations & Roadmap

### Current Release (v0.4.0)
- ✅ Standalone agent, RestrictedSandbox, DockerSandbox
- ✅ 20 chart types, 3 backends, 8 LLM providers
- ✅ 14 data connectors (local, DB, cloud, SaaS)
- ✅ Production hardening (rate limits, PII, prompt injection, cost tracking, logging)
- ✅ Conversation memory, auto-EDA, error hints, progress callbacks
- ✅ Cloud integration tests with emulators

### Planned (v0.5+)
- [ ] Dashboard generation UI (web component)
- [ ] Real-time data source updates (push-based invalidation)
- [ ] Fine-tuning & model caching (token embedding cache)
- [ ] More LLM providers (Groq, Together, etc.)
- [ ] SQL dialect-specific optimizations (window functions, CTEs)
- [ ] Async API (`async def chat()`)

### Not Planned
- Hosted cloud UI (BambooLLM-style) — stay self-hosted
- pandasai embedding/RAG — that's pandas-ai's focus

---

## Security Notes

### Default Behavior
- **Code execution is sandboxed** by default (RestrictedPython)
- **DataFrame contents are NOT redacted** by default — optional `DataRedactor` must be wired in
- **API keys are never logged** (env vars only)
- **LLM input is sanitized** against common prompt-injection patterns

### Data Privacy
- **In-process execution** (RestrictedSandbox) keeps data local
- **Optional PII redaction** before cloud LLM calls
- **No telemetry** — all events are your responsibility to capture

### Reporting
See [SECURITY.md](SECURITY.md) for vulnerability disclosure.

---

## License & Attribution

MIT License. Copyright Cem Akpolat.

Built on [LiteLLM](https://github.com/BerriAI/litellm), [RestrictedPython](https://restrictedpython.readthedocs.io), and the pandas ecosystem.

---

## Contact & Support

- **GitHub:** https://github.com/cemakpolat/pychartai
- **Email:** cem.akpolat@eficode.com
- **Issues:** Use GitHub Issues for bug reports and feature requests
- **Contributing:** See [CONTRIBUTING.md](CONTRIBUTING.md)

---

**Ready to ship.** 🚀
