# Changelog

All notable changes to pychartai are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Versioning follows [Semantic Versioning](https://semver.org/).

---

## [Unreleased]

### Added
- **Cloud storage connectors** — `S3Connection` (AWS S3), `GCSConnection` (Google Cloud Storage), `AzureBlobConnection` (Azure Blob), `GoogleSheetsConnection` (Google Sheets); auto-detect file format (csv/parquet/json/xlsx) from extension; new optional extras `[cloud-s3]`, `[cloud-gcs]`, `[cloud-azure]`, `[cloud-gsheets]`, `[cloud-all]`
- **`S3Connection(endpoint_url=...)`** — point at S3-compatible stores (MinIO, Cloudflare R2, Wasabi) or local emulators
- **Cloud integration tests** — `tests/integration/test_cloud_integration.py` runs S3/Azure/GCS connectors against MinIO, Azurite, and fake-gcs-server emulators; `docker-compose.cloud-test.yml` for local runs + `.github/workflows/cloud-integration.yml` CI job (opt-in via `PYCHARTAI_CLOUD_IT=1`)
- **Database connectors** — dedicated adapters `PostgreSQLConnection`, `MySQLConnection`, `SnowflakeConnection`, `BigQueryConnection`, `RedshiftConnection`; each handles connection-string construction, auth, and `.list_tables()` schema inspection; new optional extras `[db-postgres]`, `[db-mysql]`, `[db-snowflake]`, `[db-bigquery]`, `[db-redshift]`, `[db-all]`
- **`DataRedactor`** — PII detection and redaction before LLM calls; 30+ built-in column-name patterns; strategies: `'hash'` (SHA-256[:12], preserves groupability), `'mask'` (`'***'`), `'drop'`; opt-in via `SmartDataFrame(df, redactor=redactor)` or `pai.config.set({'redactor': redactor})`
- **Dashboard generation** — `agent.dashboard(query, df, n_charts=4)` and `df.dashboard(query)` decompose a high-level request into multiple complementary charts; LLM generates sub-queries, each executed independently with partial-failure tolerance
- **Prompt-injection guards** — `_validate_query()` strips null bytes and control chars, enforces configurable max length (`max_query_len`, default 4000), logs `WARNING` when injection patterns detected (warn-not-block policy)
- **Rate-limit / 429 handling** — `LiteLLMProvider.generate()` retries up to 4 times with full-jitter exponential back-off (base 2s, cap 60s) before raising a clear `RuntimeError` with actionable message
- **Structured observability logging** — `PyChartAgent.chat()` emits `DEBUG`/`INFO`/`WARNING` log records with structured fields: `query`, `intent`, `attempt`, `duration`, `tokens`; use standard Python logging to route to any handler
- **Benchmark scripts** — `compare_agents_extensive.py` (12-case NL analytics) and `compare_agents_charts.py` (5-case chart generation) added to repo; run locally with `--model` and `--pandasai` flags
- **Examples in CI** — new `examples-import-check` GitHub Actions job parses all 11 example files for syntax errors and top-level LLM calls; `tests/smoke_examples.py` runs standalone
- **Progress callbacks** — `agent.chat(..., on_progress=fn)` and `df.chat(..., on_progress=fn)` fires `fn(stage, detail)` at `'classifying'`, `'generating'`, `'executing'`, `'formatting'` stages
- **Token / cost tracking** — `agent.last_usage` and `llm.last_usage` expose `prompt_tokens`, `completion_tokens`, `total_tokens`; empty dict for local models (Ollama)
- **`CONTRIBUTING.md`**, **`SECURITY.md`**, **`CODE_OF_CONDUCT.md`** — OSS community essentials
- **Docs: installation guide, migration guide, API reference** — `docs/installation.md`, `docs/migration.md`, `docs/api_reference.md`
- **`[viz-plotly]` extra** — `plotly` + `kaleido` (~80 MB) moved from core deps to optional; install with `pip install pychartai[viz-plotly]`
- **37 new tests** — covering PII redactor, prompt-injection guards, progress callbacks, token tracking, exponential backoff, `select_dtypes` compat

### Changed
- **Exponential backoff on LLM retry** — `PyChartAgent.chat()` sleeps `min(2^attempt, 30)s` between retries (was immediate)
- **`select_dtypes` forward-compat** — `profiler.py` and `reporter.py` include `'string'` dtype for `pd.StringDtype` columns (pandas 2+/3+)
- **Benchmarks** — README table now includes hardware/model context; removes unverified perfect-score framing

### Fixed
- `str | None` syntax in `analyzer.py:172` → `Optional[str]` for Python 3.9 compatibility
- `SmartDataFrame.__init__` now accepts `schema=` and `redactor=` keyword arguments directly

---

## [0.4.0] — 2026-06-07

### Security
- **DockerSandbox hardening** — containers now run with `--cap-drop NET_ADMIN,NET_RAW,SYS_ADMIN,SYS_PTRACE,SYS_MODULE` and `--security-opt no-new-privileges`
- **RestrictedSandbox whitelist tightened** — `scipy` removed from default allowed imports (its advanced submodules can expose shell/subprocess capabilities); users who need scipy must opt in explicitly via `RestrictedSandbox(allow_imports=(..., 'scipy'))`
- **`.env.example` added** — real secrets must never be committed; `.gitignore` already excludes `.env`

### Added (production hardening)
- **Thread-safe `GlobalConfig`** — `set()`/`get()`/`reset()` are now protected by `threading.RLock`; safe for multi-threaded use
- **`ConversationMemory.max_result_chars`** — bounded per-turn result size (default 2000 chars) prevents memory growth in long conversations
- **Consolidated public API** — all classes exported under `pychartai` namespace: `PyChartAgent`, `ConversationMemory`, `DataProfiler`, `ChartTheme`, `get_hint`, `PyChartLLM` and more were previously only reachable via `pychartai_core`
- **`SmartDataframe` alias** — backward-compat alias for `SmartDataFrame` to avoid typo confusion
- **`SmartDataFrame.__init__(config=, chart_library=)`** — accepts config dict and chart-library override directly on construction

### Changed (production hardening)
- **Errors raise exceptions** — `PyChartAgent.chat()` now raises `ValueError`/`RuntimeError`/`TimeoutError` instead of returning error strings; callers can use proper `try/except` flow
- **Logging side-effect removed from import** — `suppress_pandasai_logging()` is no longer called at module-level of `logging.py`; instead it is called explicitly from `pychartai_core/__init__.py`, giving users a chance to configure logging before pychartai imports
- **Dependency version caps** — `litellm>=1.35.0,<2`, `RestrictedPython>=7.0,<9` to prevent major-version regressions

### Fixed
- Silent `except Exception: pass` blocks in `smart_df.py` chart-output handling now log warnings to `pychartai` logger instead of swallowing errors
- `DockerSandbox.stop()` silent exception now logs to debug

### Added
- **Conversation memory** — `df.enable_memory(window_size=N)` stores prior query-result pairs for follow-up questions
- **Data profiling / auto-EDA** — `df.profile()` returns summary stats, missing values, correlations, duplicate detection; no LLM required
- **Natural language explanations** — `df.chat(..., explain=True, agent='own')` appends plain-English LLM explanation to results
- **Chart themes** — 4 built-in themes (`light`, `dark`, `corporate`, `minimal`) + custom `ChartTheme`; set via `pai.config.set({'chart_theme': 'dark'})`
- **Comprehensive error hints** — 40+ error patterns mapped to actionable fix suggestions (e.g. "Ollama not running → `ollama serve`")
- **Transformation tracking** — `agent.last_transformation` exposes generated code, detected intent, and retry count
- **Improved intent classification** — all 20 chart types in keyword map + optional LLM fallback for ambiguous queries
- **Export formats** — PDF/SVG export for matplotlib/seaborn backends; JSON data export for Plotly
- **Jupyter integration** — `_repr_html_()` on `SmartDataFrame` for rich notebook rendering
- **FastAPI REST wrapper** — `/chat`, `/chat/upload`, `/profile`, `/chart/{path}`, `/health` endpoints (install with `pip install pychartai[api]`)
- **PyChartAgent** — standalone pandasai-independent NL→code→sandbox agent

### Changed
- `pychartai/__init__.py`: `pandasai` import is now lazy (only required when `pandasai` execution path is used)
- `chat()` raises a clear `ImportError` with install hint when `pandasai` is absent
- Build backend fixed: `setuptools.backends.legacy:build` → `setuptools.build_meta`
- Version unified to `0.4.0` across both packages
- Author email corrected; GitHub URLs updated

### Fixed
- `from X import *` was incorrectly matched by the SQL sanitizer regex, injecting `SELECT *` before import statements
- `GROUP BY` sanitizer dropped multi-column group-by clauses — removed entirely
- `ChartResponse.__str__()` calling `show()` → `Image.open()` → `FileNotFoundError`; fixed by using `result.value`
- Scatter chart type shim let `type: 'scatter_chart'` pass through; now validated and corrected
- `RestrictedSandbox`: `_write_` guard was a pass-through (`lambda x: x`); now blocks writes to types/functions
- `os` module removed from sandbox whitelist (blocked `os.system()` path)

### Removed
- `setup.py` — redundant with `pyproject.toml`
- `.bak` backup files from source tree

---

## [0.3.0] — 2026-04-01

### Added
- `PyChartLLM` universal provider + 8 named convenience classes (`OllamaLLM`, `OpenAILLM`, `GitHubLLM`, `GeminiLLM`, `AnthropicLLM`, `QwenLLM`, `DeepSeekLLM`, `GenericLLM`)
- `SmartDataFrame` with `.chat()`, skills, schema, pipeline, and streaming support
- `RestrictedSandbox` (RestrictedPython) and `DockerSandbox`
- `ResponseCache` (SHA-256 keyed file-based)
- `Pipeline` with 6 built-in steps
- Data connectors: CSV, Excel, JSON, Parquet, SQL
- `@skill` decorator for injectable callables
- `Schema` + `Column` semantic layer
- `chat_stream()` → `StreamEvent` tokens
- 20 chart types across seaborn, matplotlib, and plotly backends

---

## [0.2.0] — 2026-02-15

### Added
- Initial multi-backend visualization (seaborn + plotly)
- Ollama provider with `/api/chat` endpoint support
- `DataAnalyzer` + `CustomLLM` (pandasai-compatible legacy path)
- 7 built-in datasets: sales, weather, ecommerce, health, energy, analytics, stocks

---

## [0.1.0] — 2026-01-10

### Added
- Initial release: basic pandas-ai wrapper with chart generation
