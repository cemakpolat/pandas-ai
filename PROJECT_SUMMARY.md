# PyChartAI — Project Summary & Architecture

**Version:** 0.4.0  
**Status:** Production Ready  
**License:** MIT

---

## One-Sentence Summary

A **production-ready, security-first AI data analysis library** with 20 chart types, 14 data connectors, and built-in PII redaction — positioned as a commercially viable alternative to pandas-ai.

---

## Why pychartai?

### Problem
pandas-ai is convenient for prototyping but has production gaps:
- No default sandboxing (raw `exec()`)
- Limited data connectors (~5)
- No cost tracking for LLM calls
- No PII/data privacy controls
- Enterprise-only conversation memory
- Limited chart types (6) and backends (1)

### Solution
pychartai is built from first principles as a production product:
- ✅ **Default RestrictedPython sandbox** (fast, in-process)
- ✅ **14 data connectors** (local + 5 databases + 4 cloud + SaaS)
- ✅ **Cost tracking** (`agent.last_usage`)
- ✅ **PII redaction built-in** (3 strategies: hash/mask/drop)
- ✅ **Free conversation memory** (multi-turn, sliding-window)
- ✅ **20 chart types × 3 backends** (seaborn, matplotlib, plotly interactive)
- ✅ **Zero pandasai dependency** (independent agent)

---

## Architecture at a Glance

```
┌─────────────────────────────────────────────────────────┐
│ User Query                                              │
│ "Average revenue by region?" / "Bar chart of sales"    │
└────────────────────┬────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────┐
│ PyChartAgent (Standalone NL→Code Agent)                │
├────────────────────────────────────────────────────────┤
│ 1. Intent Classification (20 chart types + NL analysis) │
│    → Determine if chart, raw analysis, or dashboard    │
│                                                         │
│ 2. Prompt Engineering                                  │
│    • DataFrame schema + sample rows                    │
│    • Chart specs + data field hints                    │
│    • Execution constraints (RestrictedPython safety)   │
│                                                         │
│ 3. LLM Call (with retry logic)                         │
│    • Rate-limit handling: exponential backoff 4x       │
│    • Timeout: 60s (configurable)                       │
│    • Fallback: return cached result on failure         │
│                                                         │
│ 4. Code Extraction & Sanitization                      │
│    • Extract Python code from LLM response             │
│    • Remove dangerous imports                          │
│    • Validate syntax                                   │
│                                                         │
│ 5. Sandbox Execution                                   │
│    ├─ RestrictedSandbox (DEFAULT)                     │
│    │  • RestrictedPython in-process                   │
│    │  • Fast, no Docker overhead                       │
│    │  • Prevents dangerous builtins (__import__, etc.) │
│    │                                                    │
│    └─ DockerSandbox (OPTIONAL)                        │
│       • Full isolation via container                   │
│       • Slower but more secure                         │
│       • For untrusted LLM code                         │
│                                                         │
│ 6. Result Validation                                   │
│    • Check output type (DataFrame, chart path, etc.)  │
│    • Validate chart file exists + readable            │
│    • Extract & record token counts                     │
│                                                         │
│ 7. Conversation Memory Recording                       │
│    • Store query, intent, result in ConversationMemory │
│    • Sliding-window (keep last N turns for context)   │
│                                                         │
│ 8. Optional LLM Explanation                           │
│    • If explain=True, ask LLM to summarize result      │
│    • Include in metadata                               │
└────────────────────┬────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────┐
│ Formatted Result + Metadata                             │
├────────────────────────────────────────────────────────┤
│ • Output (chart path, DataFrame, string, or HTML)      │
│ • Metadata:                                             │
│   - last_usage: {prompt_tokens, completion_tokens}     │
│   - duration: execution time (sec)                      │
│   - transformation_log: steps taken                     │
│   - memory: conversation history                        │
└─────────────────────────────────────────────────────────┘
```

---

## Directory Structure

```
pychartai/
├── src/pychartai/                    # Public API shim
│   ├── __init__.py                   # Exports 99 public symbols
│   └── __version__.py                # "0.4.0"
│
├── src/pychartai_core/               # Implementation
│   ├── agent.py                      # PyChartAgent (main orchestrator)
│   ├── model_manager.py              # LiteLLMProvider (8 LLM backends)
│   ├── smart_df.py                   # SmartDataFrame (pandas wrapper)
│   ├── sandbox.py                    # RestrictedSandbox + DockerSandbox
│   ├── visualization.py              # ChartAI (20 chart types)
│   ├── visualization_backends/       # seaborn, matplotlib, plotly
│   ├── memory.py                     # ConversationMemory
│   ├── profiler.py                   # DataProfiler (auto-EDA)
│   ├── pipeline.py                   # 6-step composable Pipeline
│   ├── cache.py                      # ResponseCache (SHA-256 keyed)
│   ├── skills.py                     # @skill decorator
│   ├── schema.py                     # DataFrame schema introspection
│   ├── streaming.py                  # chat_stream() generator
│   ├── error_hints.py                # 40+ error pattern → fix suggestions
│   ├── reporter.py                   # InsightReporter (HTML reports)
│   ├── redactor.py                   # DataRedactor (PII removal)
│   ├── db_connectors.py              # 5 database connectors
│   ├── cloud_connectors.py           # 4 cloud storage + 1 SaaS
│   ├── themes.py                     # 4 built-in chart themes
│   ├── api.py                        # FastAPI REST wrapper
│   └── config.py                     # Global configuration
│
├── tests/                             # 260+ unit tests
│   ├── test_features.py              # Core functionality
│   ├── test_db_connectors.py         # Database connector unit tests
│   ├── test_cloud_connectors.py      # Cloud connector unit tests
│   ├── test_redactor.py              # PII redaction tests
│   ├── test_new_features.py          # Production features
│   ├── test_query_validation.py      # Input validation
│   ├── integration/                  # Cloud integration tests
│   │   └── test_cloud_integration.py # S3/Azure/GCS vs emulators
│   └── smoke_examples.py             # Example syntax validation
│
├── examples/                          # 2 kept, 9 removed
│   ├── basic_examples.py             # Essential examples
│   └── advanced_features_demo.py     # Power features
│
├── docs/                              # mkdocs Material theme
│   ├── index.md                      # Homepage
│   ├── getting-started.md            # 5-minute setup
│   ├── installation.md               # All optional extras
│   ├── migration.md                  # pandas-ai → pychartai
│   ├── api_reference.md              # Full API docs
│   ├── MODES_CHEATSHEET.md           # Execution modes
│   └── MULTI_PROVIDER_GUIDE.md       # LLM provider setup
│
├── data/                              # Sample datasets
│   └── use_cases/                    # 7 CSV files for examples
│
├── .github/workflows/                 # GitHub Actions
│   ├── test.yml                      # pytest on PR/push
│   ├── cloud-integration.yml         # Cloud connector integration tests
│   └── publish.yml                   # Auto-publish to PyPI on tag
│
├── .env.example                       # Template for API keys
├── pyproject.toml                    # Package metadata + optional extras
├── docker-compose.cloud-test.yml     # Local emulator stack (MinIO, Azurite, fake-gcs)
├── mkdocs.yml                        # Documentation site config
│
├── README.md                         # Project homepage
├── CHANGELOG.md                      # Version history
├── CONTRIBUTING.md                  # Developer guide
├── SECURITY.md                       # Vulnerability disclosure
├── CODE_OF_CONDUCT.md                # Community guidelines
├── LICENSE                           # MIT
├── RELEASE_NOTES.md                  # This release summary
├── SHIPPING_CHECKLIST.md             # Pre-release & deployment checklist
└── PROJECT_SUMMARY.md                # This file
```

---

## Key Design Decisions

### 1. Standalone (No pandasai Required)
- pandasai is an optional extra `[pandasai]`
- Core agent (`PyChartAgent`) is completely independent
- Can swap backends: `SmartDataFrame.with_analyzer(pandasai.Agent)` for legacy code
- **Why:** Avoid pandasai bugs, licensing confusion, bloated dependency tree

### 2. RestrictedPython by Default
- Execution sandbox is active by default, not optional
- Uses `RestrictedPython.compile_restricted()` + safe globals
- Can escalate to `DockerSandbox` if needed (isolated container)
- **Why:** Data safety > convenience. Users expect guardrails.

### 3. Exponential Backoff for Rate Limits
- `delay = min(2^attempt, 30)` with jitter
- 4 retry attempts before failing
- Separate from agent-level retries (model_manager.py)
- **Why:** Prevent thundering herd in distributed systems. Cap at 30s to avoid excessive waits.

### 4. Optional Extras (No Bloat)
- Base install: `pip install pychartai` — minimal deps
- Cloud support: `pip install pychartai[cloud-all]` — boto3, google-cloud-storage, etc.
- Databases: `pip install pychartai[db-postgres]` — psycopg2, sqlalchemy
- Visualization: `pip install pychartai[viz-plotly]` — plotly, kaleido
- **Why:** Users shouldn't pay for unused dependencies. Reduces install time & security surface.

### 5. S3-Compatible Endpoints
- `S3Connection(endpoint_url='http://minio:9000')` works with MinIO, Cloudflare R2, Wasabi
- Enables local testing with docker-compose + MinIO emulator
- **Why:** Unifies S3-compatible stores under one connector. Easier to test.

### 6. Lazy Imports for Cloud SDKs
- boto3, google-cloud-storage, azure-storage-blob imported only in `.load()` method
- Clear `ImportError` with hint: "Install: pip install pychartai[cloud-s3]"
- **Why:** Fast import time for users who don't use cloud. Better error messages.

### 7. PII Redaction Is Opt-In
- `DataRedactor` exists but NOT wired in by default
- Users must explicitly: `pai.config.set({'redactor': redactor})`
- 3 strategies: hash (SHA-256[:12]), mask ('***'), drop (column removal)
- **Why:** Default behavior shouldn't surprise users. Redaction loses data fidelity; let users choose.

### 8. Prompt-Injection Detection Without Blocking
- 14 regex patterns: "ignore previous instructions", "act as", "new system prompt", etc.
- Logs `WARNING` but allows query through
- **Why:** False positives break legitimate queries. Warn-not-block is better for UX.

### 9. Conversation Memory with Sliding Window
- Keeps last N turns (default 10) for multi-turn context
- Stored in `SmartDataFrame.memory` (in-memory only, no persistence)
- **Why:** Free, simple, solves 80% of use case. Database persistence is future work.

### 10. Cost Tracking via `agent.last_usage`
- Dict: `{'prompt_tokens': int, 'completion_tokens': int, 'total_tokens': int}`
- Set after every LLM call by model_manager.py
- **Why:** Users need to track spend. Expose it; don't hide it.

---

## Technology Choices

| Layer | Choice | Why |
|---|---|---|
| **LLM Backend** | LiteLLM (8 providers) | Unified API, no vendor lock-in |
| **Code Sandbox** | RestrictedPython | Fast, in-process, safe defaults |
| **Alternative Sandbox** | Docker | Full isolation for max security |
| **Data Frame** | pandas | Industry standard, community support |
| **Visualization** | seaborn + matplotlib + plotly | Rich types, 3 backends for flexibility |
| **Config** | dict-based `config.set()` | Simple, no YAML/env complexity |
| **Logging** | stdlib `logging` module | No dependencies, standard interface |
| **Testing** | pytest | De facto standard for Python |
| **CI/CD** | GitHub Actions | Free, integrated, OIDC for PyPI |
| **Docs** | mkdocs + Material theme | Beautiful, searchable, GitHub Pages ready |
| **Build** | pyproject.toml only | Modern Python packaging (PEP 517) |

---

## Testing Strategy

### Unit Tests (260 passing)
- Core functionality (agent, smartdf, visualization)
- Data connectors (database & cloud, URI parsing, format detection)
- Features (redaction, rate limits, callbacks, memory)
- Input validation (null-byte stripping, injection patterns)
- Error hints (40+ patterns)
- **No external LLM required** — all mocked

### Integration Tests (4 cloud tests, gated)
- S3 (MinIO emulator) — upload, load CSV via boto3
- Azure Blob (Azurite emulator) — upload, load CSV
- GCS (fake-gcs-server emulator) — upload, load CSV
- Gated behind `PYCHARTAI_CLOUD_IT=1` environment variable
- **CI runs them automatically** via `.github/workflows/cloud-integration.yml`

### Smoke Tests
- All 2 example files parse without syntax errors
- `pytest tests/smoke_examples.py`

### Manual Testing Checklist (before release)
```bash
# Local LLM
ollama pull llama3.2 && ollama serve

# Quick test
python -c "
import pychartai as pai
pai.config.set({'llm': pai.OllamaLLM(model='llama3.2')})
df = pai.read_csv('data/use_cases/sales.csv')
print(df.chat('Total revenue?'))
"
```

---

## Release & Deployment

### Automated Pipeline (GitHub Actions)
1. **test.yml** — pytest on every PR, push to main, Python 3.9–3.12
2. **cloud-integration.yml** — spins up MinIO, Azurite, fake-gcs; runs cloud tests
3. **publish.yml** — on `v0.4.0` tag, builds + publishes to PyPI (OIDC, no secrets)

### Manual Steps to Release
```bash
# 1. Tag
git tag -a v0.4.0 -m "Production release"
git push origin v0.4.0

# 2. Automated publish via GitHub Actions
# (no manual `twine upload` needed)

# 3. Verify
pip install pychartai==0.4.0
python -c "import pychartai; print(pychartai.__version__)"
```

---

## Performance Characteristics

| Operation | Time | Notes |
|---|---|---|
| **Import** | ~100ms | pandas, numpy, matplotlib loaded lazily |
| **SmartDataFrame creation** | ~10ms | Just wraps pandas.DataFrame |
| **RestrictedPython sandbox** | <1ms overhead | No startup cost |
| **LLM call** | 2–30s | Depends on model & input |
| **Chart generation** | 100–500ms | Seaborn/matplotlib render |
| **Conversation memory lookup** | <1ms | In-memory list |

**Bottleneck:** Always the LLM call. Everything else is fast.

---

## Known Limitations

1. **No real-time data invalidation** — data loaded once; live updates require manual refresh
2. **No async API** — everything is synchronous; async support planned for v0.5
3. **No fine-tuning** — can't train custom models; uses public LLMs only
4. **No RAG/embedding** — focuses on generative queries, not retrieval-augmented
5. **No hosted cloud UI** — self-hosted only; no BambooLLM-style SaaS
6. **Small community** — new project; pandas-ai has larger user base

---

## Future Roadmap (v0.5+)

- [ ] **Async API** — `async def chat()` for concurrent queries
- [ ] **Dashboard UI** — web component to visualize multi-chart dashboards
- [ ] **Fine-tuning** — domain-specific model tuning
- [ ] **More LLM providers** — Groq, Together, Perplexity
- [ ] **SQL optimizations** — window functions, CTEs, pushdown predicates
- [ ] **Real-time invalidation** — detect data changes, refresh caches
- [ ] **Embedding cache** — cache token embeddings for faster LLM calls

---

## Support & Contribution

- **Issues:** https://github.com/cemakpolat/pychartai/issues
- **Email:** cem.akpolat@eficode.com
- **Contributing:** See CONTRIBUTING.md for dev setup, code style, PR process
- **Security:** See SECURITY.md for vulnerability disclosure

---

## License & Attribution

MIT License. Copyright Cem Akpolat.

Built on:
- [pandas](https://pandas.pydata.org/) — data manipulation
- [LiteLLM](https://github.com/BerriAI/litellm) — unified LLM API
- [RestrictedPython](https://restrictedpython.readthedocs.io/) — safe code sandbox
- [matplotlib/seaborn/plotly](https://matplotlib.org/) — visualization
- [mkdocs](https://www.mkdocs.org/) + [Material](https://squidfunk.github.io/mkdocs-material/) — documentation

---

**Maintained by:** Cem Akpolat  
**Last updated:** 2026-06-08  
**Status:** Production Ready ✅
