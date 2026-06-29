.PHONY: help install setup venv prepare-data demo demo-advanced \
	test test-api test-unit-features test-new-features \
        test-charts test-charts-seaborn test-charts-plotly test-charts-matplotlib \
	test-llm test-llm-seaborn test-llm-plotly test-llm-matplotlib test-llm-text \
	test-pychartai-style test-pychartai-style-seaborn test-pychartai-style-plotly test-pychartai-style-matplotlib \
	test-pychartai-style-single test-pychartai-style-read-csv test-pychartai-style-multi \
	test-integration test-profile test-themes test-memory test-explain \
        demo-agent-gemma4 demo-agent-gemma4-full \
        gpt5 gpt5-demo \
        demo-docker demo-docker-openai demo-docker-github demo-docker-gemini demo-docker-anthropic demo-docker-deepseek demo-docker-qwen \
        clean clean-pyc clean-charts info

# Python interpreter — prefer venv, fall back to system python3
PYTHON ?= $(shell [ -f .venv/bin/python3 ] && echo .venv/bin/python3 || echo python3)
PYTEST ?= $(PYTHON) -m pytest

help:
	@echo "pychartai - Available Commands"
	@echo "========================================="
	@echo ""
	@echo "Setup & Installation:"
	@echo "  make install        - Install dependencies from requirements.txt"
	@echo "  make venv           - Create virtualenv (.venv) and install dependencies"
	@echo "  make prepare-data   - Generate CSV datasets under data/use_cases"
	@echo "  make setup          - Complete setup (install dependencies)"
	@echo ""
	@echo "Running the Application:"
	@echo "  make demo [MODEL=llama3.2]                    - Quick demo"
	@echo "  make demo-advanced [MODEL=...] [BACKEND=...] - Advanced features demo"
	@echo "  make demo-advanced PROVIDER=openai MODEL=gpt-4o      - OpenAI"
	@echo "  make demo-advanced PROVIDER=github MODEL=gpt-4.1      - GitHub Models"
	@echo "  make demo-advanced PROVIDER=gemini MODEL=gemini-2.0-flash - Gemini"
	@echo "  make demo-advanced PROVIDER=anthropic MODEL=claude-3-5-sonnet - Claude"
	@echo ""
	@echo "Local Validation (no LLM required):"
	@echo "  make test                - Run all offline tests (API + charts + new features)"
	@echo "  make test-api            - Run public API pytest suite"
	@echo "  make test-unit-features  - Run Skills/Schema/Cache/Pipeline/Connections unit tests"
	@echo "  make test-new-features   - Run memory/profiler/themes/error-hints tests"
	@echo ""
	@echo "Chart Testing (no LLM required):"
	@echo "  make test-charts              - All 3 backends (seaborn, plotly, matplotlib)"
	@echo "  make test-charts-seaborn      - Seaborn backend only"
	@echo "  make test-charts-plotly       - Plotly backend only"
	@echo "  make test-charts-matplotlib   - Matplotlib backend only"
	@echo "  make test-charts KEEP=1       - Keep generated charts"
	@echo ""
	@echo "LLM Examples (requires Ollama):"
	@echo "  make test-llm [MODEL=llama3.2]        - Run LLM examples (all backends)"
	@echo "  make test-llm-seaborn [MODEL=...]     - LLM examples with seaborn backend"
	@echo "  make test-llm-plotly [MODEL=...]      - LLM examples with plotly backend"
	@echo "  make test-llm-matplotlib [MODEL=...]  - LLM examples with matplotlib backend"
	@echo "  make test-llm-text [MODEL=...]        - LLM text queries only (no charts)"
	@echo "  make test-llm DATASET=all             - Test across all 7 datasets"
	@echo "  make test-llm KEEP=1                  - Keep generated charts"
	@echo ""
	@echo "Autonomous Client Agent (requires Ollama):"
	@echo "  make demo-agent [MODEL=llama3.2] [BACKEND=seaborn] [DATASET=sales|employee|both]"
	@echo "  make demo-agent PHASES=1,2 WORKERS=4 EXTRA=4"
	@echo "  make demo-agent PARALLEL=1           - Agent + background app in parallel"
	@echo "  make demo-agent-gemma4               - Gemma4 + complex charts (Phase 3), all datasets"
	@echo "  make demo-agent-gemma4-full          - Gemma4 + phases 1,2,3,4 + all backends + all datasets"
	@echo "  make demo-agent-gemma4-full KEEP_NL_CHARTS=1  - Keep NL phase chart files under output/nl"
	@echo ""
	@echo "Integration Testing (requires Ollama):"
	@echo "  make test-integration [MODEL=llama3.2]   - Extensive real-world Q&A + charts + profiling"
	@echo "  make test-profile                        - Data profiling demo (no LLM)"
	@echo "  make test-themes                         - Chart theme demo (no LLM)"
	@echo "  make test-memory [MODEL=llama3.2]        - Conversation memory demo"
	@echo "  make test-explain [MODEL=llama3.2]       - Result explanation demo"
	@echo ""
	@echo "pychartai Style Examples (requires Ollama):"
	@echo "  make test-pychartai-style                   - All scenarios (all backends)"
	@echo "  make test-pychartai-style-seaborn           - Scenarios with seaborn backend"
	@echo "  make test-pychartai-style-plotly            - Scenarios with plotly backend"
	@echo "  make test-pychartai-style-matplotlib        - Scenarios with matplotlib backend"
	@echo "  make test-pychartai-style-single            - Single DataFrame df.chat()"
	@echo "  make test-pychartai-style-read-csv          - read_csv() + df.chat()"
	@echo "  make test-pychartai-style-multi             - pai.chat() multi-DataFrame"
	@echo "  make test-pychartai-style KEEP=1            - Keep all generated files"
	@echo ""
	@echo "GitHub AI Models (requires GITHUB_TOKEN):"
	@echo "  make gpt5           - Run GPT-5 integration demo"
	@echo "  make gpt5-demo      - Alias for 'make gpt5'"
	@echo ""
	@echo "Docker Sandbox + LLM (requires Docker running + an LLM provider):"
	@echo "  make demo-docker [PROVIDER=ollama] [MODEL=llama3.2] [DATASET=sales]  - Run Docker+LLM demo"
	@echo "  make demo-docker DATASET=all                                         - Run across all 7 datasets"
	@echo "  make demo-docker MEMORY=1g TIMEOUT=120                               - Custom resource limits"
	@echo "  make demo-docker-openai   [MODEL=gpt-4o]                             - OpenAI provider"
	@echo "  make demo-docker-github   [MODEL=gpt-4.1]                            - GitHub Models provider"
	@echo "  make demo-docker-gemini   [MODEL=gemini-2.0-flash]                   - Gemini provider"
	@echo "  make demo-docker-anthropic [MODEL=claude-3-5-sonnet-20241022]        - Anthropic provider"
	@echo "  make demo-docker-deepseek [MODEL=deepseek-chat]                      - DeepSeek provider"
	@echo "  make demo-docker-qwen     [MODEL=qwen-plus]                          - Qwen provider"
	@echo ""
	@echo "Maintenance:"
	@echo "  make clean          - Remove cache and temporary files"
	@echo "  make clean-pyc      - Remove Python cache files"
	@echo "  make clean-charts   - Remove generated chart files"
	@echo "  make info           - Show project information"
	@echo ""
	@echo "Examples:"
	@echo "  make demo MODEL=llama3.2"
	@echo "  make demo-advanced MODEL=llama3.2 BACKEND=seaborn"
	@echo "  make test-llm MODEL=llama3.2 DATASET=all KEEP=1"
	@echo ""

install:
	@echo "Installing dependencies..."
	pip install -r requirements.txt
	@echo "✓ Dependencies installed"

venv:
	@echo "Creating virtualenv in .venv (if missing) and installing dependencies..."
	@if [ ! -d ".venv" ]; then python3 -m venv .venv || python -m venv .venv; fi
	.venv/bin/pip install -r requirements.txt
	@echo "✓ Virtual environment ready (.venv)"

setup: install
	@echo "✓ Setup complete! Run 'make demo' to start"

prepare-data:
	@echo "Generating use-case CSV datasets under data/use_cases ..."
	$(PYTHON) scripts/prepare_use_case_data.py
	@echo "✓ Use-case datasets are ready"

demo:
	@echo "Running demo with default model..."
	python3 main.py --demo

# ---------------------------------------------------------------------------
# Advanced Features demo (Skills, Schema, Cache, Pipeline, Connections)
# ---------------------------------------------------------------------------
# Pass MODEL=<name> to choose the model (default: llama3.2 for Ollama).
# Pass PROVIDER=<ollama|openai|github|qwen|gemini|anthropic|deepseek> to choose the LLM provider.
# Pass BACKEND=<seaborn|matplotlib|plotly> to choose the chart backend.
# Pass FEATURE=<skills|schema|cache|pipeline|connections|combined|charts> for a single feature.
#
# Examples:
#   make demo-advanced                                  # Ollama/llama3.2 (default)
#   make demo-advanced MODEL=mistral BACKEND=plotly
#   make demo-advanced PROVIDER=openai MODEL=gpt-4o
#   make demo-advanced PROVIDER=github MODEL=gpt-4.1
#   make demo-advanced PROVIDER=gemini MODEL=gemini-2.0-flash
#   make demo-advanced FEATURE=skills

_DEMO_FEATURE ?= $(if $(FEATURE),$(FEATURE),all)
_DEMO_PROVIDER ?= $(if $(PROVIDER),$(PROVIDER),ollama)

demo-advanced:
	@echo "Running advanced features demo (provider=$(_DEMO_PROVIDER), model=$(or $(MODEL),$(_MODEL)), backend=$(or $(BACKEND),seaborn), feature=$(_DEMO_FEATURE))..."
	@mkdir -p data/use_cases
	$(PYTHON) examples/advanced_features_demo.py \
		--provider $(_DEMO_PROVIDER) \
		--model $(or $(MODEL),$(_MODEL)) \
		--backend $(or $(BACKEND),seaborn) \
		--feature $(_DEMO_FEATURE)

# ---------------------------------------------------------------------------
# Autonomous multi-phase client agent demo
# ---------------------------------------------------------------------------
# Runs an Ollama-powered agent across 4 phases:
#   Phase 1 — Simple charts (bar, pie, histogram)
#   Phase 2 — Medium charts (scatter, line, grouped bar)
#   Phase 3 — Complex charts (heatmap, box plot, stacked bar, bubble)
#   Phase 4 — Deep text analysis (rankings, ratios, trend summaries)
#
# Options:
#   MODEL=llama3.2          Ollama model name
#   BACKEND=seaborn         primary backend for NL phases (seaborn | matplotlib | plotly)
#   BACKENDS=seaborn,matplotlib,plotly   Phase-3 multi-library showcase (overrides BACKEND)
#   DATASET=both            sales | employee | both
#   PHASES=1,2,3            comma-separated phases to run (default: 1,2,3)
#   WORKERS=3               parallel worker threads (phases 1–2)
#   EXTRA=0                 extra LLM-generated questions per dataset
#
# Examples:
#   make demo-agent                              # seaborn, both datasets, phases 1,2,3
#   make demo-agent BACKENDS=seaborn,matplotlib,plotly PHASES=3
#   make demo-agent BACKEND=plotly PHASES=1,2
#   make demo-agent PHASES=1,2,3,4 WORKERS=4

# Use ?= so the variable is only set here if not already defined on the command line.
# Avoids the Make or() function, which misinterprets commas as argument separators.
PHASES     ?= 1,2,3
DATASET    ?= both
WORKERS    ?= 3
EXTRA      ?= 0
MODEL      ?= $(_MODEL)
BACKEND    ?= seaborn
BACKENDS   ?=

_BACKENDS_ARG := $(if $(BACKENDS),--backends $(BACKENDS),)
_COMPLEX_BACKENDS ?= $(if $(BACKENDS),$(BACKENDS),seaborn,matplotlib,plotly)
_GEMMA_MODEL := $(if $(filter command line,$(origin MODEL)),$(MODEL),gemma4)
_GEMMA_PHASES := $(if $(filter command line,$(origin PHASES)),$(PHASES),3)
_GEMMA_PHASES_FULL := $(if $(filter command line,$(origin PHASES)),$(PHASES),1,2,3,4)
_GEMMA_WORKERS := $(if $(filter command line,$(origin WORKERS)),$(WORKERS),4)
_GEMMA_EXTRA := $(if $(filter command line,$(origin EXTRA)),$(EXTRA),4)
_GEMMA_OUTPUT := $(if $(filter command line,$(origin OUTPUT)),$(OUTPUT),agent_output_gemma4)
_GEMMA_OUTPUT_FULL := $(if $(filter command line,$(origin OUTPUT)),$(OUTPUT),agent_output_gemma4_full)
_KEEP_NL_CHARTS_ARG := $(if $(KEEP_NL_CHARTS), --keep-nl-charts,)

demo-agent:
	@echo "Starting client agent (model=$(MODEL), backend=$(BACKEND), dataset=$(DATASET), phases=$(PHASES))..."
	@mkdir -p agent_output
	$(PYTHON) examples/client_agent.py \
		--model $(MODEL) \
		--backend $(BACKEND) \
		--dataset $(DATASET) \
		--phases $(PHASES) \
		--workers $(WORKERS) \
		--extra $(EXTRA) \
		--output agent_output \
		$(_BACKENDS_ARG)

# Gemma4-focused autonomous agent presets for multi-plot and complex diagrams.
# - Uses BOTH datasets (sales + employee) for diverse data types.
# - Runs Phase 3 direct chart suite (20 chart types) across all 3 backends.
# - "full" preset also runs NL phases and text-analysis phase.
demo-agent-gemma4:
	@echo "Starting Gemma4 complex-diagram demo (Phase 3, all backends, both datasets)..."
	@mkdir -p $(_GEMMA_OUTPUT)
	$(PYTHON) examples/client_agent.py \
		--model $(_GEMMA_MODEL) \
		--dataset all \
		--phases $(_GEMMA_PHASES) \
		--backends $(_COMPLEX_BACKENDS) \
		--workers $(_GEMMA_WORKERS) \
		--output $(_GEMMA_OUTPUT)

demo-agent-gemma4-full:
	@echo "Starting Gemma4 full autonomous demo (phases 1,2,3,4 + all backends + extras)..."
	@mkdir -p $(_GEMMA_OUTPUT_FULL)
	$(PYTHON) examples/client_agent.py \
		--model $(_GEMMA_MODEL) \
		--dataset all \
		--phases $(_GEMMA_PHASES_FULL) \
		--backends $(_COMPLEX_BACKENDS) \
		--workers $(_GEMMA_WORKERS) \
		--extra $(_GEMMA_EXTRA) \
		--output $(_GEMMA_OUTPUT_FULL)$(_KEEP_NL_CHARTS_ARG)

# Runs Phase 3 only across all 3 backends and saves to agent_output_backends/.
# Demonstrates pychartai's multi-library claim: same 20 chart types rendered
# by seaborn, matplotlib, and plotly from a single API call.
demo-agent-backends:
	@echo "Phase-3 multi-backend showcase (seaborn + matplotlib + plotly)..."
	@mkdir -p agent_output_backends
	$(PYTHON) examples/client_agent.py \
		--model $(MODEL) \
		--backends seaborn,matplotlib,plotly \
		--dataset $(DATASET) \
		--phases 3 \
		--output agent_output_backends

test: test-api test-charts test-new-features

test-unit-features:
	@echo "Running Skills / Schema / Cache / Pipeline / Connections unit tests (no LLM required)..."
	$(PYTEST) tests/test_features.py -v

test-new-features:
	@echo "Running Phase 3/4 feature tests (memory, profiler, themes, error hints, smart_df)..."
	$(PYTEST) tests/test_memory.py tests/test_profiler.py tests/test_themes.py tests/test_error_hints.py tests/test_smart_df_features.py -v

test-api:
	@echo "Running chartai API and compatibility tests..."
	$(PYTEST) tests/test_chartai_api.py

# ---------------------------------------------------------------------------
# Chart backend tests (no LLM required — tests visualization functions directly)
# ---------------------------------------------------------------------------
# Pass KEEP=1 to preserve generated PNG/HTML files after the test run.

_KEEP_FLAG = $(if $(KEEP),--keep,)

test-charts:
	@echo "Testing all chart backends (seaborn, plotly, matplotlib)..."
	$(PYTHON) tests/test_charts.py $(_KEEP_FLAG)

test-charts-seaborn:
	@echo "Testing seaborn backend..."
	$(PYTHON) tests/test_charts.py --backend seaborn $(_KEEP_FLAG)

test-charts-plotly:
	@echo "Testing plotly backend..."
	$(PYTHON) tests/test_charts.py --backend plotly $(_KEEP_FLAG)

test-charts-matplotlib:
	@echo "Testing matplotlib backend..."
	$(PYTHON) tests/test_charts.py --backend matplotlib $(_KEEP_FLAG)

# ---------------------------------------------------------------------------
# LLM examples (full pipeline: Ollama → pychartai Agent → visualization)
# ---------------------------------------------------------------------------
# Requires Ollama running locally with the model pulled.
# Pass MODEL=<name> to choose a model, KEEP=1 to preserve chart files,
# DATASET=weather or DATASET=all to change dataset.
# Examples:
#   make test-llm
#   make test-llm MODEL=llama3.2 BACKEND=seaborn KEEP=1
#   make test-llm-text
#   make test-llm DATASET=all

_MODEL   ?= llama3.2
_DATASET ?= $(if $(DATASET),$(DATASET),sales)
_BACKEND ?= $(if $(BACKEND),$(BACKEND),all)
_OUTPUT_DIR ?= $(if $(OUTPUT_DIR),$(OUTPUT_DIR),exports/charts/manual)
_SCENARIO ?= $(if $(SCENARIO),$(SCENARIO),all)
_STYLE_OUTPUT_DIR ?= $(if $(OUTPUT_DIR),$(OUTPUT_DIR),exports/charts/pychartai-style)

test-llm:
	@echo "Running LLM examples (model=$(or $(MODEL),$(_MODEL)), backends=$(_BACKEND), dataset=$(_DATASET))..."
	@mkdir -p data/use_cases
	@mkdir -p $(_OUTPUT_DIR)
	$(PYTHON) examples/llm_chart_examples.py \
		--model $(or $(MODEL),$(_MODEL)) \
		--backend $(_BACKEND) \
		--dataset $(_DATASET) \
		--data-dir data/use_cases \
		--output-dir $(_OUTPUT_DIR) \
		$(_KEEP_FLAG)

test-llm-seaborn:
	@echo "Running LLM examples with seaborn backend..."
	@mkdir -p data/use_cases
	@mkdir -p $(_OUTPUT_DIR)
	$(PYTHON) examples/llm_chart_examples.py \
		--model $(or $(MODEL),$(_MODEL)) \
		--backend seaborn \
		--dataset $(_DATASET) \
		--data-dir data/use_cases \
		--output-dir $(_OUTPUT_DIR) \
		$(_KEEP_FLAG)

test-llm-plotly:
	@echo "Running LLM examples with plotly backend..."
	@mkdir -p data/use_cases
	@mkdir -p $(_OUTPUT_DIR)
	$(PYTHON) examples/llm_chart_examples.py \
		--model $(or $(MODEL),$(_MODEL)) \
		--backend plotly \
		--dataset $(_DATASET) \
		--data-dir data/use_cases \
		--output-dir $(_OUTPUT_DIR) \
		$(_KEEP_FLAG)

test-llm-matplotlib:
	@echo "Running LLM examples with matplotlib backend..."
	@mkdir -p data/use_cases
	@mkdir -p $(_OUTPUT_DIR)
	$(PYTHON) examples/llm_chart_examples.py \
		--model $(or $(MODEL),$(_MODEL)) \
		--backend matplotlib \
		--dataset $(_DATASET) \
		--data-dir data/use_cases \
		--output-dir $(_OUTPUT_DIR) \
		$(_KEEP_FLAG)

test-llm-text:
	@echo "Running LLM text-only examples (no charts)..."
	@mkdir -p data/use_cases
	@mkdir -p $(_OUTPUT_DIR)
	$(PYTHON) examples/llm_chart_examples.py \
		--model $(or $(MODEL),$(_MODEL)) \
		--dataset $(_DATASET) \
		--data-dir data/use_cases \
		--output-dir $(_OUTPUT_DIR) \
		--text-only

test-pychartai-style:
	@echo "Running pychartai-style scenarios (model=$(or $(MODEL),$(_MODEL)), backend=$(or $(BACKEND),seaborn), scenario=$(_SCENARIO))..."
	@mkdir -p $(_STYLE_OUTPUT_DIR)
	$(PYTHON) examples/pychartai_style_examples.py \
		--model $(or $(MODEL),$(_MODEL)) \
		--backend $(or $(BACKEND),seaborn) \
		--scenario $(_SCENARIO) \
		--output-dir $(_STYLE_OUTPUT_DIR) \
		$(_KEEP_FLAG)

test-pychartai-style-seaborn:
	@echo "Running pychartai-style scenarios with seaborn backend..."
	$(MAKE) test-pychartai-style MODEL=$(or $(MODEL),$(_MODEL)) BACKEND=seaborn SCENARIO=$(or $(SCENARIO),$(_SCENARIO)) OUTPUT_DIR=$(_STYLE_OUTPUT_DIR) KEEP=$(KEEP)

test-pychartai-style-plotly:
	@echo "Running pychartai-style scenarios with plotly backend..."
	$(MAKE) test-pychartai-style MODEL=$(or $(MODEL),$(_MODEL)) BACKEND=plotly SCENARIO=$(or $(SCENARIO),$(_SCENARIO)) OUTPUT_DIR=$(_STYLE_OUTPUT_DIR) KEEP=$(KEEP)

test-pychartai-style-matplotlib:
	@echo "Running pychartai-style scenarios with matplotlib backend..."
	$(MAKE) test-pychartai-style MODEL=$(or $(MODEL),$(_MODEL)) BACKEND=matplotlib SCENARIO=$(or $(SCENARIO),$(_SCENARIO)) OUTPUT_DIR=$(_STYLE_OUTPUT_DIR) KEEP=$(KEEP)

test-pychartai-style-single:
	@echo "Running pychartai-style single DataFrame scenario..."
	$(MAKE) test-pychartai-style MODEL=$(or $(MODEL),$(_MODEL)) BACKEND=$(or $(BACKEND),seaborn) SCENARIO=single OUTPUT_DIR=$(_STYLE_OUTPUT_DIR) KEEP=$(KEEP)

test-pychartai-style-read-csv:
	@echo "Running pychartai-style read_csv scenario..."
	$(MAKE) test-pychartai-style MODEL=$(or $(MODEL),$(_MODEL)) BACKEND=$(or $(BACKEND),seaborn) SCENARIO=read-csv OUTPUT_DIR=$(_STYLE_OUTPUT_DIR) KEEP=$(KEEP)

test-pychartai-style-multi:
	@echo "Running pychartai-style multi-DataFrame scenario..."
	$(MAKE) test-pychartai-style MODEL=$(or $(MODEL),$(_MODEL)) BACKEND=$(or $(BACKEND),seaborn) SCENARIO=multi OUTPUT_DIR=$(_STYLE_OUTPUT_DIR) KEEP=$(KEEP)

# ---------------------------------------------------------------------------
# Integration testing — extensive real-world validation (requires Ollama)
# ---------------------------------------------------------------------------
# Tests Q&A correctness, chart generation, profiling, memory, and explanations.
#   make test-integration MODEL=llama3.2
#   make test-integration MODEL=mistral BACKEND=plotly

test-integration:
	@echo "Running extensive integration tests (model=$(or $(MODEL),$(_MODEL)), backend=$(or $(BACKEND),seaborn))..."
	@mkdir -p data/use_cases exports/charts/integration
	$(PYTHON) tests/test_integration_extensive.py \
		--model $(or $(MODEL),$(_MODEL)) \
		--backend $(or $(BACKEND),seaborn) \
		--output-dir exports/charts/integration

test-profile:
	@echo "Running data profiling demo (no LLM required)..."
	$(PYTHON) -c "\
import sys; sys.path.insert(0, 'src'); \
import pandas as pd; \
from pychartai_core.profiler import DataProfiler; \
from pychartai_core.data_manager import DataManager; \
dm = DataManager(); \
for dt in ['sales', 'weather', 'ecommerce', 'health', 'energy']: \
    df = dm.create_sample_data(dt, dt); \
    report = DataProfiler.profile(df); \
    print(f'\n{\"=\" * 60}'); \
    print(report.summary); \
print('\n✓ All profiles generated successfully') \
"

test-themes:
	@echo "Running chart theme demo (no LLM required)..."
	@mkdir -p exports/charts/themes
	$(PYTHON) -c "\
import sys; sys.path.insert(0, 'src'); \
import pandas as pd; \
from pychartai_core.visualization import bar_chart; \
from pychartai_core.themes import BUILTIN_THEMES, resolve_theme; \
df = pd.DataFrame({'category': ['A','B','C','D'], 'value': [10,25,15,30]}); \
for name in BUILTIN_THEMES: \
    theme = resolve_theme(name); \
    if name == 'dark': theme.apply_seaborn(); \
    else: theme.apply_matplotlib(); \
    path = bar_chart(df, x='category', y='value', title=f'Theme: {name}', \
                     output_file=f'exports/charts/themes/{name}_bar.png', backend='seaborn'); \
    print(f'  ✓ {name}: {path}'); \
print('\n✓ All themes rendered successfully') \
"

test-memory:
	@echo "Running conversation memory demo (model=$(or $(MODEL),$(_MODEL)))..."
	@mkdir -p data/use_cases
	$(PYTHON) tests/test_integration_extensive.py \
		--model $(or $(MODEL),$(_MODEL)) \
		--backend $(or $(BACKEND),seaborn) \
		--output-dir exports/charts/integration \
		--section memory

test-explain:
	@echo "Running explanation demo (model=$(or $(MODEL),$(_MODEL)))..."
	@mkdir -p data/use_cases
	$(PYTHON) tests/test_integration_extensive.py \
		--model $(or $(MODEL),$(_MODEL)) \
		--backend $(or $(BACKEND),seaborn) \
		--output-dir exports/charts/integration \
		--section explain

clean-charts:
	@echo "Removing generated chart files..."
	@rm -f exports/charts/*.png exports/charts/*.html
	@rm -rf exports/charts/manual
	@rm -rf exports/charts/pychartai-style
	@rm -rf exports/charts/test
	@echo "✓ Charts cleaned"

clean: clean-pyc clean-charts
	@echo "Cleaning project..."
	@rm -rf build/
	@rm -rf dist/
	@rm -rf *.egg-info
	@rm -rf .pytest_cache/
	@rm -rf .coverage
	@rm -rf htmlcov/
	@rm -rf .mypy_cache/
	@find . -type d -name "__pycache__" -exec rm -r {} + 2>/dev/null || true
	@echo "✓ Clean complete"

clean-pyc:
	@echo "Removing Python cache files..."
	@find . -type f -name "*.py[cod]" -delete
	@find . -type f -name "*~" -delete

info:
	@echo "pychartai — Project Information"
	@echo "=========================================="
	@echo ""
	@echo "Structure:"
	@echo "  src/pychartai_core/           Core implementation (SOLID principles)"
	@echo "  examples/                      Example scripts"
	@echo "  tests/                         Test suites (281+ unit tests + 98 chart tests, no LLM required)"
	@echo "  data/use_cases/                7 built-in CSV datasets"
	@echo "  exports/charts/                Generated chart outputs"
	@echo ""
	@echo "Python: 3.9+    |    Main: pandas, seaborn, plotly, RestrictedPython"
	@echo ""
	@echo "Key Features:"
	@echo "  • Unified LiteLLM backend: Ollama, OpenAI, GitHub, Gemini, Anthropic, DeepSeek, Qwen, Custom"
	@echo "  • 3 execution modes: RestrictedSandbox (default), DockerSandbox, PyChartAgent"
	@echo "  • 20 chart types × 3 backends (seaborn, matplotlib, plotly)"
	@echo "  • Skills, Schema, Cache, Pipeline, Connectors"
	@echo "  • Conversation memory, Data profiling, Chart themes"
	@echo "  • Error hints, Result explanations, Transformation tracking"
	@echo "  • Streaming: chat_stream() → StreamEvent"
	@echo ""
	@echo "Run 'make help' for all commands."

# ---------------------------------------------------------------------------
# GitHub AI Models - GPT-5 Integration
# ---------------------------------------------------------------------------
# Requires GITHUB_TOKEN environment variable (set in .env)

.env-check:
	@if [ ! -f .env ]; then \
		echo "❌ .env file not found"; \
		exit 1; \
	fi
	@if ! grep -q "GITHUB_TOKEN" .env; then \
		echo "❌ GITHUB_TOKEN not found in .env"; \
		exit 1; \
	fi
	@echo "✓ .env configuration verified"

gpt5: .env-check
	@echo "Running GitHub AI Models integration demo..."
	@model=$$(grep '^GITHUB_MODEL=' .env | cut -d'=' -f2); \
	echo "Model: openai/$$model (from GitHub AI Models API)"; \
	echo ""; \
	export $$(grep -v '^\s*#' .env | xargs) && $(PYTHON) examples/pychartai_github_models_demo.py

gpt5-demo: gpt5

# ---------------------------------------------------------------------------
# Docker Sandbox + LLM examples
# ---------------------------------------------------------------------------
# Runs PyChartAgent queries inside a Docker container as the execution
# backend.  Charts are disabled in Docker mode (no display server) — all
# results are text / tabular data.
#
# Requires:
#   - Docker daemon running  (docker info must succeed)
#   - An LLM provider reachable from the host (default: Ollama)
#
# Pass any of:
#   PROVIDER=<ollama|openai|github|gemini|anthropic|deepseek|qwen>
#   MODEL=<name>        (default: llama3.2)
#   DATASET=<name|all>  (default: sales)
#   MEMORY=<docker-mem> (default: 512m)
#   TIMEOUT=<seconds>   (default: 60)
#
# Examples:
#   make demo-docker
#   make demo-docker PROVIDER=openai MODEL=gpt-4o
#   make demo-docker PROVIDER=github MODEL=gpt-4.1
#   make demo-docker DATASET=all MEMORY=1g TIMEOUT=120

_DOCKER_DATASET  ?= $(if $(DATASET),$(DATASET),sales)
_DOCKER_MEMORY   ?= $(if $(MEMORY),$(MEMORY),512m)
_DOCKER_TIMEOUT  ?= $(if $(TIMEOUT),$(TIMEOUT),60)
_DOCKER_PROVIDER ?= $(if $(PROVIDER),$(PROVIDER),ollama)

demo-docker:
	@echo "Running Docker+LLM demo (provider=$(_DOCKER_PROVIDER), model=$(or $(MODEL),$(_MODEL)), dataset=$(_DOCKER_DATASET))..."
	@echo "Docker sandbox: memory=$(_DOCKER_MEMORY), timeout=$(_DOCKER_TIMEOUT)s"
	@mkdir -p data/use_cases
	$(PYTHON) examples/docker_llm_demo.py \
		--provider $(_DOCKER_PROVIDER) \
		--model $(or $(MODEL),$(_MODEL)) \
		--dataset $(_DOCKER_DATASET) \
		--memory $(_DOCKER_MEMORY) \
		--timeout $(_DOCKER_TIMEOUT)

demo-docker-openai:
	@echo "Running Docker+OpenAI demo..."
	$(MAKE) demo-docker PROVIDER=openai MODEL=$(or $(MODEL),gpt-4o) DATASET=$(_DOCKER_DATASET) MEMORY=$(_DOCKER_MEMORY) TIMEOUT=$(_DOCKER_TIMEOUT)

demo-docker-github:
	@echo "Running Docker+GitHub Models demo..."
	$(MAKE) demo-docker PROVIDER=github MODEL=$(or $(MODEL),gpt-4.1) DATASET=$(_DOCKER_DATASET) MEMORY=$(_DOCKER_MEMORY) TIMEOUT=$(_DOCKER_TIMEOUT)

demo-docker-gemini:
	@echo "Running Docker+Gemini demo..."
	$(MAKE) demo-docker PROVIDER=gemini MODEL=$(or $(MODEL),gemini-2.0-flash) DATASET=$(_DOCKER_DATASET) MEMORY=$(_DOCKER_MEMORY) TIMEOUT=$(_DOCKER_TIMEOUT)

demo-docker-anthropic:
	@echo "Running Docker+Anthropic demo..."
	$(MAKE) demo-docker PROVIDER=anthropic MODEL=$(or $(MODEL),claude-3-5-sonnet-20241022) DATASET=$(_DOCKER_DATASET) MEMORY=$(_DOCKER_MEMORY) TIMEOUT=$(_DOCKER_TIMEOUT)

demo-docker-deepseek:
	@echo "Running Docker+DeepSeek demo..."
	$(MAKE) demo-docker PROVIDER=deepseek MODEL=$(or $(MODEL),deepseek-chat) DATASET=$(_DOCKER_DATASET) MEMORY=$(_DOCKER_MEMORY) TIMEOUT=$(_DOCKER_TIMEOUT)

demo-docker-qwen:
	@echo "Running Docker+Qwen demo..."
	$(MAKE) demo-docker PROVIDER=qwen MODEL=$(or $(MODEL),qwen-plus) DATASET=$(_DOCKER_DATASET) MEMORY=$(_DOCKER_MEMORY) TIMEOUT=$(_DOCKER_TIMEOUT)

.DEFAULT_GOAL := help
