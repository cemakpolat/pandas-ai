# PyChartAI Shipping Checklist

**Status:** ✅ Ready to Ship  
**Date:** 2026-06-08  
**Version:** 0.4.0

---

## Pre-Release Verification

- [x] **Core functionality**
  - [x] PyChartAgent standalone (no pandasai dependency)
  - [x] 20 chart types across 3 backends (seaborn, matplotlib, plotly)
  - [x] 8 LLM providers (Ollama, OpenAI, Anthropic, Gemini, Qwen, DeepSeek, GitHub Models, Generic)
  - [x] RestrictedSandbox (default) + DockerSandbox
  - [x] Conversation memory with sliding-window
  - [x] Auto-EDA profiling (`df.profile()`)

- [x] **Data connectors (14 total)**
  - [x] Local files (CSV, Excel, JSON, Parquet)
  - [x] PostgreSQL, MySQL, Snowflake, BigQuery, Redshift (5 databases)
  - [x] S3 (with S3-compatible endpoint_url for MinIO/R2/Wasabi)
  - [x] Google Cloud Storage (GCS)
  - [x] Azure Blob Storage
  - [x] Google Sheets

- [x] **Production hardening**
  - [x] Rate-limit handling with exponential backoff
  - [x] PII/data redaction (DataRedactor with hash/mask/drop strategies)
  - [x] Prompt-injection guards (null-byte stripping, max-length enforcement, pattern detection)
  - [x] Cost tracking (`agent.last_usage` with token counts)
  - [x] Structured logging (DEBUG/INFO/WARNING)
  - [x] Progress callbacks (`on_progress=`)
  - [x] Dashboard generation (LLM decomposes queries into N charts)
  - [x] Error hints (40+ patterns)

- [x] **Testing**
  - [x] 260 unit tests passing
  - [x] 4 cloud integration tests (gated behind `PYCHARTAI_CLOUD_IT=1`)
  - [x] Integration tests for S3/Azure/GCS connectors with real emulators
  - [x] CI workflows (test.yml, cloud-integration.yml, publish.yml)
  - [x] No external LLM required for test suite

- [x] **Documentation**
  - [x] Getting Started guide (5-minute setup)
  - [x] Installation guide (all optional extras)
  - [x] Migration guide (from pandasai)
  - [x] API Reference (complete class/method docs)
  - [x] Architecture & LLM provider guides
  - [x] CONTRIBUTING.md
  - [x] SECURITY.md
  - [x] CODE_OF_CONDUCT.md
  - [x] CHANGELOG.md
  - [x] RELEASE_NOTES.md (this repo)

- [x] **Code quality**
  - [x] Type hints on all public APIs
  - [x] No bare `except:` clauses
  - [x] No `print()` in library code (logging instead)
  - [x] Black-formatted
  - [x] isort-organized imports
  - [x] No TODO/FIXME in core modules
  - [x] Unused code removed
  - [x] Deep-dive docs removed (unmaintained exploratory docs deleted)

---

## Before Publishing to PyPI

### Phase 1: Repository Setup (ONE TIME)
- [ ] **GitHub**
  - [ ] Create repo at https://github.com/cemakpolat/pychartai
  - [ ] Set description: "Production-ready, security-first AI data analysis library"
  - [ ] Enable GitHub Pages (Settings → Pages → Build from `/docs` or auto-deploy from workflow)
  - [ ] Enable branch protection on `main` (require PR reviews, pass CI)

- [ ] **PyPI Account**
  - [ ] Register at https://pypi.org/account/register/ (if not already done)
  - [ ] Verify email
  - [ ] Create API token for local testing (TestPyPI)

- [ ] **OIDC Trusted Publisher** (GitHub → PyPI, no token in secrets)
  - [ ] Log in to PyPI
  - [ ] Go to Account Settings → Publishing
  - [ ] Add trusted publisher:
    - **Provider:** GitHub
    - **Owner:** cemakpolat
    - **Repo:** pychartai
    - **Workflow:** `.github/workflows/publish.yml`
    - **Environment:** (leave empty or use "production")

### Phase 2: First Release to TestPyPI (OPTIONAL, but recommended)
```bash
# Build locally
pip install build twine
python -m build
twine check dist/*

# Test upload to TestPyPI
twine upload --repository testpypi dist/*

# Test install from TestPyPI
python -m venv test_env
source test_env/bin/activate
pip install --index-url https://test.pypi.org/simple/ pychartai
```

### Phase 3: Git Push & Release Tag
```bash
# Ensure main is clean
git status

# Commit any remaining changes
git add .
git commit -m "v0.4.0: Production release"

# Push to GitHub
git push origin main

# Create release tag (triggers publish.yml)
git tag -a v0.4.0 -m "Production-ready release: security-first alternative to pandas-ai"
git push origin v0.4.0
```

This triggers `.github/workflows/publish.yml` → auto-publish to PyPI (OIDC).

### Phase 4: Verify PyPI Publication
- [ ] Visit https://pypi.org/project/pychartai/ and verify v0.4.0 is live
- [ ] Check package metadata, files, and installation instructions
- [ ] Test install: `pip install pychartai`
- [ ] Verify it works:
  ```bash
  python -c "import pychartai; print(pychartai.__version__)"
  ```

---

## Ongoing Maintenance

### After Each Release
- [ ] Create GitHub Release (manually or via action)
  - Title: `v0.4.0 — Production Release`
  - Body: Copy from RELEASE_NOTES.md
  - Assets: none needed (auto-download from PyPI)

- [ ] Update CHANGELOG.md
  - Move `[Unreleased]` items to `[0.4.0] — 2026-06-08`
  - Start new `[Unreleased]` section for v0.5+ work

- [ ] Announce on channels
  - GitHub Discussion
  - Email (cem.akpolat@eficode.com)

### Handling Issues
- **Bug reports:** Triage in GitHub Issues, assign to v0.4.x or v0.5+
- **Security vulnerabilities:** See SECURITY.md (private disclosure)
- **Feature requests:** Accept but prioritize v0.5+ roadmap

---

## Known Risks & Mitigation

| Risk | Mitigation |
|---|---|
| **pandas-ai compatibility** | We're a new product, not a drop-in replacement. Docs are clear. |
| **LLM hallucination** | Code goes through RestrictedSandbox first — won't execute dangerous code. |
| **Data privacy** | PII redaction is opt-in. Users must explicitly enable. Documented. |
| **S3-compatible stores** | Tested with MinIO emulator in CI. endpoint_url param tested. |
| **Community size** | Small at launch, but solid foundation. Grow via word-of-mouth. |
| **Inactive contrib** | Maintained by Cem Akpolat. No bus-factor risk yet. |

---

## Optional Future Enhancements (v0.5+)

These are deliberately NOT in v0.4.0 (ship on time):
- [ ] Dashboard generation UI (web component)
- [ ] Real-time data source invalidation
- [ ] Fine-tuning & embedding caching
- [ ] More LLM providers (Groq, Together, Perplexity)
- [ ] Async API (`async def chat()`)
- [ ] SQL optimizer (window functions, CTEs)
- [ ] Hosted cloud UI (if demand emerges)

---

## Communication Template

**For announcement (email, forum, etc.):**

```
🚀 Introducing PyChartAI v0.4.0 — Production-Ready AI Data Analysis

We're shipping a security-first, commercially viable alternative to pandas-ai.

✨ Highlights:
• 20 chart types across 3 backends (seaborn, matplotlib, interactive plotly HTML)
• 14 data connectors (5 databases + 4 cloud stores + local files + SaaS)
• Production hardening: rate limits, PII redaction, prompt-injection guards, cost tracking
• Conversation memory & multi-turn context for free
• Sandbox execution by default (RestrictedPython) — safe code generation
• 260 unit tests + cloud integration tests with real emulators

📦 Install: pip install pychartai
🐳 Local LLM: ollama pull llama3.2 && ollama serve
💬 First query: df.chat('Average revenue by region?')

📚 Docs: https://cemakpolat.github.io/pychartai/
🐛 Issues: https://github.com/cemakpolat/pychartai/issues

Built for teams that care about data security, flexible data sources, and rich visualization.

---
MIT License. Copyright Cem Akpolat.
```

---

## Final Checklist Before Clicking "Publish"

- [ ] README.md is finalized and tested (links work)
- [ ] RELEASE_NOTES.md is clear and honest
- [ ] All 260 unit tests pass
- [ ] All 4 cloud integration tests pass (or skipped cleanly)
- [ ] CI workflows are enabled and green
- [ ] mkdocs.yml nav is correct (no broken links)
- [ ] Docs site builds without errors
- [ ] .gitignore excludes all build artifacts + generated outputs
- [ ] No `.env` or API keys in git history
- [ ] Version bumped to 0.4.0 in `src/pychartai/__init__.py`
- [ ] Git tag `v0.4.0` created
- [ ] OIDC trusted publisher configured in PyPI
- [ ] GitHub repo settings are public + Pages enabled

---

**✅ Ready. Ship it.** 🚀
