"""
Analyzer module for pandas-ai integration.
"""

from typing import Optional, List, Dict, Any
import pandas as pd
try:
    from pandasai import Agent
    from pandasai.llm.base import LLM as PandasAILLM
    from pandasai.core.prompts.base import BasePrompt
    from pandasai.agent.state import AgentState
    _PANDASAI_AVAILABLE = True
except ImportError:
    _PANDASAI_AVAILABLE = False
    Agent = None  # type: ignore[assignment]
    PandasAILLM = object  # type: ignore[assignment,misc]
    BasePrompt = object  # type: ignore[assignment,misc]
    AgentState = object  # type: ignore[assignment,misc]
from .model_manager import LLMProvider
from .visualization import build_import_statement, describe_backend
from .visualization_backends import get_backend_chart_specs, list_backends


class CustomLLM(PandasAILLM):
    """Custom LLM wrapper that integrates Ollama with pandas-ai."""

    def __init__(
        self,
        provider: LLMProvider,
        chart_backend: str = "seaborn",
        charts_output_dir: str = "exports/charts",
        skills: Optional[List] = None,
    ):
        """Initialize with an LLMProvider.

        Args:
            provider: An instance of LLMProvider (Ollama, OpenAI-compat, etc.)
            chart_backend: Default backend ('seaborn', 'matplotlib', or 'plotly')
            charts_output_dir: Directory where generated charts should be stored
            skills: Optional list of Skill objects to inject into prompts
        """
        super().__init__()
        self.provider = provider
        self.chart_backend = chart_backend
        self.charts_output_dir = charts_output_dir
        self.skills: List = list(skills) if skills else []

    @property
    def type(self) -> str:
        """Return the type of this LLM."""
        return "ollama"

    def call(
        self,
        instruction: BasePrompt,
        context: AgentState = None,
        **kwargs
    ) -> str:
        """Execute a call to the LLM."""
        prompt_str = instruction.to_string()

        chart_specs = get_backend_chart_specs(self.chart_backend)
        chart_lines = "\n".join(
            f"#    - {line}" for line in describe_backend(self.chart_backend)
        )
        available_helpers = ", ".join(spec.helper_name for spec in chart_specs)
        import_statement = build_import_statement(self.chart_backend)
        aggregate_helpers = ", ".join(
            spec.helper_name for spec in chart_specs if spec.sql_strategy == "aggregate"
        ) or "none"
        raw_helpers = ", ".join(
            spec.helper_name for spec in chart_specs if spec.sql_strategy == "raw"
        ) or "none"
        derived_helpers = ", ".join(
            spec.helper_name for spec in chart_specs if spec.sql_strategy == "derived"
        ) or "none"

        sql_directive = (
            "\n\n"
            "# INSTRUCTIONS (follow exactly):\n"
            "# 1. NEVER redefine execute_sql_query — just call it.\n"
            f"# 2. For aggregation charts ({aggregate_helpers}): use GROUP BY in SQL.\n"
            "#    CRITICAL: alias aggregated columns using the ORIGINAL column name so the chart helper can find them.\n"
            "#    df = execute_sql_query('SELECT x_col, AVG(y_col) AS y_col FROM TABLE GROUP BY x_col')\n"
            "#    Include ALL columns referenced by x=, y=, and hue= in SELECT and GROUP BY as needed.\n"
            f"# 3. For raw-distribution charts ({raw_helpers}): NO GROUP BY, select ALL needed columns.\n"
            "#    df = execute_sql_query('SELECT x_col, y_col, hue_col FROM TABLE')  -- include hue_col if used\n"
            f"# 4. For derived charts ({derived_helpers}): query the needed numeric columns, then let the helper derive the final view.\n"
            "# 5. Always set: result = {'type': 'plot', 'value': path} for charts.\n"
            "# 6. For chart queries:\n"
            f"#    {import_statement}\n"
            f"#    import os, datetime as _dt, uuid as _uuid\n"
            f"#    os.makedirs('{self.charts_output_dir}', exist_ok=True)\n"
            f"#    chart_path = '{self.charts_output_dir}/{self.chart_backend}_' + _dt.datetime.now().strftime('%Y%m%d_%H%M%S') + '_' + _uuid.uuid4().hex[:6] + '.png'\n"
            f"#    path = bar_chart(df, x='col', y='col', title='T', output_file=chart_path, backend='{self.chart_backend}')\n"
            "#    result = {'type': 'plot', 'value': path}\n"
            f"# 7. For ALL chart calls, pass backend='{self.chart_backend}' as a keyword argument.\n"
            f"# 8. Available helpers: {available_helpers}\n"
            f"# 9. Helper schemas for backend='{self.chart_backend}':\n"
            f"{chart_lines}\n"
        )
        # If skills are registered, inject their descriptions BEFORE the SQL/chart
        # directive so the LLM's attention is drawn to them first.
        if self.skills:
            from .skills import build_skills_prompt
            skills_hint = build_skills_prompt(self.skills)
            if skills_hint:
                prompt_str = prompt_str + '\n\n' + skills_hint

        prompt_str = prompt_str + sql_directive

        # Inject skills description BEFORE the SQL directive so the LLM
        # sees the available helpers before the chart-generation instructions.
        return self.provider.generate(prompt_str, **kwargs)

    def generate_code(self, instruction: BasePrompt, context: AgentState = None) -> str:
        """Generate and post-process Python code from the LLM."""
        import re as _re

        prompt_str = instruction.to_string()
        table_match = _re.search(r'table_name=["\']([^"\' >]+)', prompt_str)
        table_name = table_match.group(1) if table_match else None

        raw_response = self.call(instruction, context)
        code = self._extract_code_robust(raw_response)

        if table_name and table_name.startswith('table_'):
            code = _re.sub(r'table_[a-f0-9]{32}', table_name, code)

        # Prepend skill source code so skill functions are in-scope at execution time
        if self.skills:
            from .skills import build_skills_preamble
            preamble = build_skills_preamble(self.skills)
            if preamble:
                code = preamble + code

        code = self._sanitize_code(code, table_name=table_name)
        code = self._ensure_result_format(
            code,
            chart_backend=self.chart_backend,
            charts_output_dir=self.charts_output_dir,
        )
        return code

    def _extract_code_robust(self, response: str) -> str:
        """Extract Python code from raw LLM output."""
        import re as _re

        clean = _re.sub(r'<think>.*?</think>', '', response, flags=_re.DOTALL).strip()
        code = self._try_extract(clean)
        if code:
            return code

        for block in _re.findall(r'<think>(.*?)</think>', response, _re.DOTALL):
            code = self._try_extract(block)
            if code:
                return code

        return self._try_extract(response)

    def _try_extract(self, text: str) -> str:
        """Run PandasAI's _extract_code on *text*."""
        if not text.strip():
            return ''
        try:
            result = self._extract_code(text)
            return result if result.strip() else ''
        except Exception:
            return ''

    @staticmethod
    def _sanitize_code(code: str, table_name: 'Optional[str]' = None) -> str:
        """Remove common LLM hallucination patterns."""
        import re

        lines = code.splitlines()
        cleaned: list[str] = []

        i = 0
        while i < len(lines):
            line = lines[i]
            stripped = line.strip()

            if re.match(r"^,\s*\w", stripped):
                fixed = re.sub(r"^,\s*", "SELECT ", stripped)
                cleaned.append(line[:len(line)-len(stripped)] + fixed)
                i += 1
                continue
            
            if re.match(r"^\w+[_\w]*\s*\)", stripped) or re.match(r"^\w+[_\w]*\s+as\s+\w", stripped, re.IGNORECASE):
                prev_idx = i - 1
                while prev_idx >= 0 and lines[prev_idx].strip() == '':
                    prev_idx -= 1
                if prev_idx >= 0:
                    prev_line = lines[prev_idx].strip()
                    if ('SELECT' in prev_line.upper() or ',' in prev_line) and ')' not in prev_line:
                        cleaned.append(line)
                        i += 1
                        continue
                fixed = "SELECT " + stripped
                cleaned.append(line[:len(line)-len(stripped)] + fixed)
                i += 1
                continue
            
            if re.match(r"^FROM\s+", stripped, re.IGNORECASE) and not re.match(r"^from\s+\S+\s+import\b", stripped):
                prev_idx = i - 1
                while prev_idx >= 0 and lines[prev_idx].strip() == '':
                    prev_idx -= 1
                prev_line = lines[prev_idx].strip() if prev_idx >= 0 else ''
                if not re.match(r"^SELECT\s+", prev_line, re.IGNORECASE):
                    cleaned.append(line[:len(line)-len(stripped)] + "SELECT *")
                    cleaned.append(line)
                    i += 1
                    continue

            # Strip wildcard imports — RestrictedPython blocks them at compile time
            if re.match(r'^from\s+\S+\s+import\s+\*', stripped):
                cleaned.append(f'# [sanitized] removed wildcard import')
                i += 1
                continue

            # Strip problematic chart library imports that create AttributeErrors
            # (e.g., "from plotly import figure_factory" → figure_factory doesn't exist in sandbox)
            if re.match(r'^from\s+plotly\s+import\s+figure_factory', stripped, re.IGNORECASE):
                cleaned.append('# [sanitized] removed plotly.figure_factory import (use bar_chart helper instead)')
                i += 1
                continue

            # Strip imports of chart helpers from the wrong module
            # (e.g. "from plotly.graph_objects import bar_chart")
            # Our helpers are already injected into the sandbox context.
            _CHART_HELPERS = (
                'bar_chart', 'line_chart', 'scatter_chart', 'pie_chart',
                'histogram', 'heatmap', 'box_chart', 'violin_chart',
                'area_chart', 'count_chart', 'bubble_chart', 'ecdf_chart',
                'funnel_chart', 'kde_chart', 'pairplot_chart', 'regression_chart',
                'stacked_bar_chart', 'step_chart', 'strip_chart', 'swarm_chart',
            )
            if re.match(r'^from\s+(?!pychartai_core)', stripped) and any(
                h in stripped for h in _CHART_HELPERS
            ):
                cleaned.append('# [sanitized] removed chart helper import (already in scope)')
                i += 1
                continue

            if re.match(r'def\s+execute_sql_query\s*\(', stripped):
                cleaned.append('# [sanitized] stripped execute_sql_query redefinition')
                func_indent = len(line) - len(line.lstrip())
                i += 1
                while i < len(lines):
                    next_line = lines[i]
                    next_stripped = next_line.strip()
                    if next_stripped == '':
                        i += 1
                        continue
                    next_indent = len(next_line) - len(next_line.lstrip())
                    if next_indent <= func_indent:
                        break
                    i += 1
                continue

            if table_name and re.search(r'\.groupby\s*\([^)]+\)\s*\[', stripped):
                cleaned.append('# [sanitized] removed invalid post-SQL groupby')
                i += 1
                continue

            # Strip assignments to chart_path that shadow the injected variable
            # (e.g., chart_path = 'average_price_by_product.png')
            if re.match(r'^chart_path\s*=\s*[\'"][^\'"]*[\'"]', stripped):
                cleaned.append('# [sanitized] removed local chart_path assignment (using injected chart_path)')
                i += 1
                continue

            # Strip/convert plotly figure_factory calls — they don't exist in sandbox
            # Convert "figure_factory.bar(...)" → "bar_chart(...)"
            if 'figure_factory.' in stripped:
                # Map figure_factory methods to our chart helpers
                _FF_MAP = {
                    'bar': 'bar_chart',
                    'line': 'line_chart',
                    'scatter': 'scatter_chart',
                    'pie': 'pie_chart',
                    'histogram': 'histogram',
                    'histogram_2d': 'heatmap',
                    'box': 'box_chart',
                    'violin': 'violin_chart',
                }
                fixed = stripped
                for ff_func, our_func in _FF_MAP.items():
                    fixed = re.sub(rf'figure_factory\.{ff_func}\s*\(', f'{our_func}(', fixed)
                if fixed != stripped:
                    cleaned.append(line[: len(line) - len(stripped)] + fixed)
                    i += 1
                    continue
                # If no mapping found, strip the line
                cleaned.append('# [sanitized] removed unsupported figure_factory call')
                i += 1
                continue

            if re.search(r"result\s*=\s*\{[^}]*['\"]plot['\"][^}]*\bplt\b[^}]*\}", stripped):
                cleaned.append('# [sanitized] removed plt-object result')
                i += 1
                continue

            cleaned.append(line)
            i += 1

        sanitized = "\n".join(cleaned)

        if table_name and not re.search(r'execute_sql_query\s*\(', sanitized):
            if not re.search(r'plotsai\.\w+\(', sanitized):
                injection = (
                    f"# [sanitized] injected mandatory execute_sql_query call\n"
                    f"_df = execute_sql_query('SELECT * FROM {table_name}')\n"
                    f"result = _df\n"
                )
                sanitized = injection + sanitized

        if table_name:
            sanitized = re.sub(
                r"(FROM\s+)(\{\{?table_name\}\}?|\{\{?\}\}?)",
                lambda m: f"{m.group(1)}{table_name}",
                sanitized,
                flags=re.IGNORECASE,
            )

        sanitized = CustomLLM._fix_unclosed_braces(sanitized)
        return sanitized

    @staticmethod
    def _fix_unclosed_braces(code: str) -> str:
        """Fix unclosed braces, brackets, and parentheses."""
        try:
            compile(code, '<string>', 'exec')
            return code
        except SyntaxError:
            pass
        
        braces_count = code.count('{') - code.count('}')
        brackets_count = code.count('[') - code.count(']')
        parens_count = code.count('(') - code.count(')')
        
        if braces_count > 0:
            code += '\n' + '}' * braces_count + '  # [auto-closed]'
        if brackets_count > 0:
            code += '\n' + ']' * brackets_count + '  # [auto-closed]'
        if parens_count > 0:
            code += '\n' + ')' * parens_count + '  # [auto-closed]'
        
        try:
            compile(code, '<string>', 'exec')
            return code
        except SyntaxError:
            return (
                "try:\n"
                + "\n".join("    " + line for line in code.splitlines()) + "\n"
                + "except Exception as e:\n"
                + f"    result = {{'type': 'string', 'value': f'Code execution error: {{str(e)}}'}}\n"
            )

    @staticmethod
    def _ensure_result_format(
        code: str,
        chart_backend: str = "seaborn",
        charts_output_dir: str = "exports/charts",
    ) -> str:
        """Wrap result into PandasAI's dict format."""
        import re
        import_statement = build_import_statement(chart_backend)
        import_pattern = re.escape(import_statement)
        
        # Replace any standalone import plotsai with the correct import
        code = re.sub(
            r'^\s*import\s+plotsai\s*$',
            import_statement,
            code,
            flags=re.MULTILINE
        )

        # Fix invalid "import * as plotsai" syntax the LLM sometimes generates
        code = re.sub(
            r'from\s+pychartai_core\.visualization\s+import\s+\*\s+as\s+\w+',
            import_statement,
            code,
            flags=re.MULTILINE
        )
        code = re.sub(
            r'from\s+pychartai_core\.visualization\s+import\s+\*',
            import_statement,
            code,
            flags=re.MULTILINE,
        )
        
        # If visualization is not imported at all, add it
        _has_viz_import = re.search(
            rf'{import_pattern}|import\s+pychartai_core[.\w]*\s+as\s+plotsai|from\s+pychartai_core[.\w]*\s+import',
            code, re.MULTILINE
        )
        if not _has_viz_import:
            code = (
                f"{import_statement}\n"
                "import os\n"
                f"os.makedirs({charts_output_dir!r}, exist_ok=True)\n"
            ) + code
        
        # Always append shim — it normalises the result dict even when one is already set
        shim = (
            "\n"
            "# --- PandasAI result-format shim (auto-added) ---\n"
            "import pandas as _pd_shim\n"
            "_VALID_TYPES = {'dataframe', 'string', 'number', 'plot'}\n"
            "try:\n"
            "    _r = result\n"
            "except NameError:\n"
            "    _r = None\n"
            "if isinstance(_r, dict) and _r.get('type') in _VALID_TYPES:\n"
            "    # Validate that value actually matches the claimed type\n"
            "    _t, _v = _r.get('type'), _r.get('value')\n"
            "    if _t == 'plot' and not (isinstance(_v, str) and _v.endswith(('.png', '.jpg', '.svg', '.html'))):\n"
            "        result = {'type': 'string', 'value': str(_v)}\n"
            "    elif _t == 'dataframe' and isinstance(_v, str) and _v.endswith(('.png', '.jpg', '.svg', '.html')):\n"
            "        result = {'type': 'plot', 'value': _v}\n"
            "    # else: already correct\n"
            "elif isinstance(_r, dict) and 'type' in _r and 'value' in _r:\n"
            "    # invalid type (e.g. 'scatter_chart') — fix it\n"
            "    _val = _r['value']\n"
            "    if isinstance(_val, str) and _val.endswith(('.png', '.jpg', '.svg', '.html')):\n"
            "        result = {'type': 'plot', 'value': _val}\n"
            "    elif isinstance(_val, _pd_shim.DataFrame):\n"
            "        result = {'type': 'dataframe', 'value': _val}\n"
            "    else:\n"
            "        result = {'type': 'string', 'value': str(_val)}\n"
            "elif isinstance(_r, dict) and not _r:\n"
            "    result = {'type': 'string', 'value': 'Query completed but returned no data'}\n"
            "elif isinstance(_r, _pd_shim.DataFrame):\n"
            "    result = {'type': 'dataframe', 'value': _r}\n"
            "elif isinstance(_r, (int, float)):\n"
            "    result = {'type': 'number', 'value': _r}\n"
            "elif isinstance(_r, str) and _r.endswith(('.png', '.jpg', '.svg', '.html')):\n"
            "    result = {'type': 'plot', 'value': _r}\n"
            "elif _r is not None:\n"
            "    result = {'type': 'string', 'value': str(_r)}\n"
            "else:\n"
            "    result = {'type': 'string', 'value': 'No result generated'}\n"
        )
        return code + shim

    @staticmethod
    def _fix_sandbox_chart_code(code: str) -> str:
        """Fix common LLM patterns in sandbox-generated chart code.

        1. Wrap bare chart-helper calls (without ``result =``) so that the
           return value is captured: ``bar_chart(...)`` → ``result = bar_chart(...)``
        2. Replace any hardcoded ``output_file='...'`` string with the
           ``chart_path`` variable that is pre-injected into the sandbox context.
        """
        import re

        _CHART_FUNC_PAT = (
            r'(?:bar_chart|line_chart|scatter_chart|pie_chart|histogram|heatmap'
            r'|box_chart|violin_chart|area_chart|count_chart|bubble_chart'
            r'|ecdf_chart|funnel_chart|kde_chart|pairplot_chart|regression_chart'
            r'|stacked_bar_chart|step_chart|strip_chart|swarm_chart)'
        )

        lines = code.splitlines()
        new_lines = []
        for line in lines:
            stripped = line.strip()
            # Wrap bare chart calls that have no assignment prefix
            if (
                re.match(rf'^{_CHART_FUNC_PAT}\s*\(', stripped)
                and not re.match(r'^result\s*=', stripped)
            ):
                indent = line[: len(line) - len(line.lstrip())]
                new_lines.append(f'{indent}result = {stripped}')
                continue
            new_lines.append(line)

        code = '\n'.join(new_lines)

        # Replace hardcoded output_file values with the injected chart_path variable
        code = re.sub(
            r"output_file\s*=\s*['\"][^'\"]*['\"]",
            'output_file=chart_path',
            code,
        )

        return code


class DataAnalyzer:
    """Main analyzer class for pandas-ai queries."""

    def __init__(
        self,
        model_name: str = "llama3.2",
        memory_size: int = 100,
        verbose: bool = False,
        charts_output_dir: str = "exports/charts",
        chart_backend: str = "seaborn",
        provider_type: str = "ollama",
        api_key: str = None,
        *,
        llm=None,        # OllamaLLM or PandasAILLM instance (keyword-only)
        skills=None,     # list of Skill objects to inject (keyword-only)
    ):
        """
        Initialize DataAnalyzer.

        Args:
            model_name:        Model alias (e.g., 'llama3.2', 'gpt-3.5', 'deepseek-chat').
            memory_size:       PandasAI agent memory size.
            verbose:           Enable verbose output.
            charts_output_dir: Directory where generated charts are saved.
            chart_backend:     Visualization backend ('seaborn', 'matplotlib', or 'plotly').
            provider_type:     LLM provider ('ollama', 'openai', 'deepseek', 'github').
            api_key:           API key for cloud providers (OpenAI, DeepSeek, GitHub Models).
            llm:               Optional ``OllamaLLM`` or ``PandasAILLM`` instance
                               from the public API.  When provided, *model_name*
                               is derived from the LLM object and the legacy
                               model-name path is skipped.
        """
        import os

        supported_backends = set(list_backends())
        if chart_backend not in supported_backends:
            supported = ", ".join(sorted(supported_backends))
            raise ValueError(f"Unsupported chart backend: {chart_backend}. Supported values: {supported}")

        os.makedirs(charts_output_dir, exist_ok=True)

        self.model_name = model_name
        self.provider_type = provider_type
        self.api_key = api_key
        self.memory_size = memory_size
        self.verbose = verbose
        self.charts_output_dir = charts_output_dir
        self.chart_backend = chart_backend
        self.agent: Optional[Agent] = None
        self.llm: Optional[CustomLLM] = None
        self._passthrough_llm = None  # holds native pandasai LLM for PandasAILLM path
        self._skills = list(skills) if skills else []
        self._initialize_llm(llm_override=llm)

    def _initialize_llm(self, llm_override=None) -> None:
        """Initialize the LLM from an override object or a model-name string.

        Supports two input forms (Liskov-compatible: any PyChartLLM subclass works):
          * llm_override is None     → build from provider_type + model_name strings
          * llm_override is PyChartLLM → use its provider directly
          * llm_override is PandasAILLM → pass through to pandasai Agent
          * anything else            → treat as a native pandasai LLM
        """
        from .providers import PyChartLLM as _PyChartLLM, PandasAILLM as _PandasAILLM

        try:
            if llm_override is None:
                # Build a PyChartLLM from the string arguments (OCP: new providers
                # need no changes here — just add them to providers._REGISTRY)
                llm_override = _PyChartLLM(
                    model=f'{self.provider_type}/{self.model_name}',
                    api_key=self.api_key,
                )

            if isinstance(llm_override, _PyChartLLM):
                self.model_name = llm_override.model_name
                provider = llm_override._get_provider()
                self.llm = CustomLLM(
                    provider,
                    chart_backend=self.chart_backend,
                    charts_output_dir=self.charts_output_dir,
                    skills=self._skills,
                )

            elif isinstance(llm_override, _PandasAILLM):
                self._passthrough_llm = llm_override.get_inner()
                self.llm = None

            else:
                # Native pandasai-compatible LLM passed directly
                self._passthrough_llm = llm_override
                self.llm = None

            if self.verbose:
                print(f"✓ Initialized LLM (model={self.model_name}, chart_backend={self.chart_backend})")

        except (RuntimeError, TypeError):
            raise
        except Exception as e:
            raise RuntimeError(f"Failed to initialize LLM: {str(e)}")
    
    def analyze(
        self,
        df: pd.DataFrame,
        query: str,
        chart_backend: Optional[str] = None,
        sandbox=None,
        **kwargs
    ) -> str:
        """
        Analyze DataFrame with a natural language query.

        Args:
            df:            DataFrame to analyze.
            query:         Natural language query.
            chart_backend: Per-call visualization backend override
                           ('seaborn' or 'plotly').  Falls back to the
                           instance-level ``chart_backend`` when *None*.
            sandbox:       Optional sandbox instance (:class:`RestrictedSandbox`
                           or :class:`DockerSandbox`).  When provided the
                           pandasai Agent is bypassed and the generated code
                           is executed inside the sandbox instead.
            **kwargs:      Additional parameters (currently unused).

        Returns:
            Analysis result as string, or a file path when a chart is produced.
        """
        if df.empty:
            return 'Error: DataFrame is empty'

        effective_backend = chart_backend or self.chart_backend

        if sandbox is not None:
            return self._analyze_with_sandbox(df, query, effective_backend, sandbox)

        # Resolve active LLM — create a lightweight CustomLLM copy only when
        # the per-call backend differs from the default (avoids mutation).
        if self._passthrough_llm is not None:
            active_llm = self._passthrough_llm
        elif self.llm is not None and effective_backend != self.chart_backend:
            active_llm = CustomLLM(
                self.llm.provider,
                chart_backend=effective_backend,
                charts_output_dir=self.charts_output_dir,
                skills=self._skills,
            )
        else:
            active_llm = self.llm

        from .config import config as _global_config
        max_retries = _global_config.get('max_retries', 3)
        agent = Agent(
            [df],
            config={
                "llm": active_llm,
                "verbose": self.verbose,
                "save_charts": False,
                "open_charts": False,
            },
            memory_size=self.memory_size
        )

        for attempt in range(max_retries):
            try:
                import io as _io
                import contextlib as _cl
                if self.verbose:
                    result = agent.chat(query)
                else:
                    with _cl.redirect_stdout(_io.StringIO()):
                        result = agent.chat(query)
                # ChartResponse.__str__ calls show() → Image.open() which can fail.
                # Use .value directly to get the file path without triggering display.
                if hasattr(result, 'value'):
                    return str(result.value)
                return str(result)
            except Exception as e:
                error_msg = str(e)
                retryable = ("No code found" in error_msg or 
                            "Invalid expression" in error_msg or
                            "Unexpected token" in error_msg or
                            "Unable to get your answer" in error_msg or
                            "was never closed" in error_msg or
                            "SyntaxError" in error_msg or
                            "The code must execute SQL" in error_msg)
                
                if retryable and attempt < max_retries - 1:
                    if self.verbose:
                        print(f"⚠ Retry {attempt + 1}/{max_retries - 1}: {error_msg[:80]}")
                    continue
                
                return f"Error during analysis: {error_msg}"

        return "Error: Failed after maximum retries"

    # ------------------------------------------------------------------
    # Sandbox execution path
    # ------------------------------------------------------------------

    def _analyze_with_sandbox(
        self,
        df: pd.DataFrame,
        query: str,
        chart_backend: str,
        sandbox,
    ) -> str:
        """Execute *query* against *df* using *sandbox* instead of pandasai."""
        if self.llm is None:
            raise RuntimeError(
                'Sandbox execution requires a pychartai LLM provider '
                '(OllamaLLM, OpenAILLM, etc.). '
                'PandasAILLM pass-through is not supported in sandbox mode.'
            )
        from .sandbox import DockerSandbox
        is_docker = isinstance(sandbox, DockerSandbox)
        prompt = self._build_sandbox_prompt(
            df, query, chart_backend, self.charts_output_dir,
            allow_charts=not is_docker,
        )
        from .config import config as _global_config
        max_retries = _global_config.get('max_retries', 3)
        for attempt in range(max_retries):
            try:
                raw = self.llm.provider.generate(prompt)
                code = self.llm._extract_code_robust(raw)
                code = CustomLLM._sanitize_code(code, table_name=None)
                code = CustomLLM._fix_sandbox_chart_code(code)
                if self._skills:
                    from .skills import build_skills_preamble
                    preamble = build_skills_preamble(self._skills)
                    if preamble:
                        code = preamble + '\n' + code
                if self.verbose:
                    print(f'[sandbox] generated code:\n{code}')
                if is_docker:
                    result = sandbox.execute(code, {'df': df})
                else:
                    context = self._build_sandbox_context(df, chart_backend, self.charts_output_dir)
                    result = sandbox.execute(code, context)
                return self._format_sandbox_result(result)
            except (TimeoutError, ConnectionError) as exc:
                return f'Error in sandbox execution: {exc}'
            except Exception as exc:
                if attempt < max_retries - 1:
                    if self.verbose:
                        print(f'⚠ Sandbox retry {attempt + 1}: {exc}')
                    continue
                return f'Error in sandbox execution: {exc}'
        return 'Error: Max retries exceeded in sandbox mode.'

    def _analyze_with_sandbox_multi(
        self,
        query: str,
        frames: List[pd.DataFrame],
        chart_backend: str,
        sandbox,
    ) -> str:
        """Execute *query* against multiple *frames* using *sandbox*."""
        if self.llm is None:
            raise RuntimeError(
                'Sandbox execution requires a pychartai LLM provider.'
            )
        from .sandbox import DockerSandbox
        is_docker = isinstance(sandbox, DockerSandbox)
        prompt = self._build_sandbox_prompt_multi(
            frames, query, chart_backend, self.charts_output_dir,
            allow_charts=not is_docker,
        )
        var_names = ['df'] + [f'df{i}' for i in range(1, len(frames))]
        from .config import config as _global_config
        max_retries = _global_config.get('max_retries', 3)
        for attempt in range(max_retries):
            try:
                raw = self.llm.provider.generate(prompt)
                code = self.llm._extract_code_robust(raw)
                code = CustomLLM._sanitize_code(code, table_name=None)
                code = CustomLLM._fix_sandbox_chart_code(code)
                if self._skills:
                    from .skills import build_skills_preamble
                    preamble = build_skills_preamble(self._skills)
                    if preamble:
                        code = preamble + '\n' + code
                if is_docker:
                    docker_ctx = {name: frame for name, frame in zip(var_names, frames)}
                    result = sandbox.execute(code, docker_ctx)
                else:
                    context = self._build_sandbox_context(frames[0], chart_backend, self.charts_output_dir)
                    for name, frame in zip(var_names, frames):
                        context[name] = frame
                    context['dfs'] = frames
                    result = sandbox.execute(code, context)
                return self._format_sandbox_result(result)
            except Exception as exc:
                if attempt < max_retries - 1:
                    if self.verbose:
                        print(f'⚠ Sandbox (multi-df) retry {attempt + 1}: {exc}')
                    continue
                return f'Error in sandbox execution (multi-df): {exc}'
        return 'Error: Max retries exceeded.'

    def _build_sandbox_prompt(
        self,
        df: pd.DataFrame,
        query: str,
        chart_backend: str,
        charts_output_dir: str,
        allow_charts: bool = True,
    ) -> str:
        """Build a direct code-gen prompt for sandbox execution."""
        col_desc = ', '.join(f'{c} ({dt})' for c, dt in df.dtypes.items())
        sample_str = df.head(3).to_string()
        chart_portion = ''
        if allow_charts:
            chart_specs = get_backend_chart_specs(chart_backend)
            helper_names = ', '.join(s.helper_name for s in chart_specs)
            chart_lines = '\n'.join(f'#   {l}' for l in describe_backend(chart_backend))
            chart_portion = (
                f'4. For charts — IMPORTANT:\n'
                f'   - The helper functions ({helper_names}) are ALREADY in scope. Do NOT import them.\n'
                f'   - Use pre-computed `chart_path` as the output file path.\n'
                f'   - CORRECT: `result = bar_chart(df, x=\'col\', y=\'col\', title=\'Title\', output_file=chart_path, backend=\'{chart_backend}\')`\n'
                f'   - WRONG: `import seaborn as sns; ...` or `from plotly import ...` or `df.plot(...)`\n'
                f'   - Always pass `backend=\'{chart_backend}\'` and `output_file=chart_path`.\n'
                f'   - Chart helper signatures:\n{chart_lines}\n'
            )
        else:
            chart_portion = (
                '4. Do NOT generate charts — return text or DataFrame results only.\n'
            )
        return (
            f"You have a pandas DataFrame `df` with {len(df)} rows.\n"
            f"Columns: {col_desc}\n"
            f"Sample (first 3 rows):\n{sample_str}\n\n"
            f"Task: {query}\n\n"
            "Instructions:\n"
            "1. Use direct pandas operations on `df`. Do NOT use SQL or `execute_sql_query`.\n"
            "2. `df` is already available in scope. Do not re-import or redefine it.\n"
            "3. Set `result` to the final answer:\n"
            "   - Text/number: `result = str(answer)` or `result = number`\n"
            "   - DataFrame: `result = df_answer`\n"
            f"{chart_portion}"
            "5. Do NOT use `subprocess`, `open()`, or network calls.\n"
            "6. Return ONLY the Python code block, no explanations.\n"
        )

    def _build_sandbox_prompt_multi(
        self,
        frames: List[pd.DataFrame],
        query: str,
        chart_backend: str,
        charts_output_dir: str,
        allow_charts: bool = True,
    ) -> str:
        """Build a sandbox prompt for multiple DataFrames."""
        var_names = ['df'] + [f'df{i}' for i in range(1, len(frames))]
        sections = []
        for var, frame in zip(var_names, frames):
            col_desc = ', '.join(f'{c} ({dt})' for c, dt in frame.dtypes.items())
            sample_str = frame.head(2).to_string()
            sections.append(
                f"DataFrame `{var}` with {len(frame)} rows.\n"
                f"Columns: {col_desc}\n"
                f"Sample:\n{sample_str}"
            )
        df_block = '\n\n'.join(sections)
        var_list = ', '.join(f'`{v}`' for v in var_names)
        chart_portion = (
            '4. Do NOT generate charts — return text or DataFrame results only.\n'
            if not allow_charts else
            '4. For charts use `chart_path` as output_file and pass `backend=\''
            + chart_backend + '\'` to chart helpers.\n'
        )
        return (
            "You have the following pandas DataFrames:\n\n"
            f"{df_block}\n\n"
            f"Task: {query}\n\n"
            "Instructions:\n"
            f"1. All DataFrames are available as {var_list}. Do not redefine them.\n"
            "2. Use direct pandas operations. Do NOT use SQL or `execute_sql_query`.\n"
            "3. Set `result` to the final answer (str, number, or DataFrame).\n"
            f"{chart_portion}"
            "5. Return ONLY the Python code block, no explanations.\n"
        )

    def _build_sandbox_context(
        self,
        df: pd.DataFrame,
        chart_backend: str,
        charts_output_dir: str,
    ) -> dict:
        """Build the execution context injected into RestrictedSandbox."""
        import os
        import datetime as _dt
        import uuid as _uuid
        import numpy as np
        import matplotlib as _mpl
        _mpl.use('Agg')
        import matplotlib.pyplot as _plt
        _plt.show = lambda: None  # suppress interactive display inside sandbox
        # Guard against LLM calling plt.set_title / plt.set_xlabel etc. which
        # don't exist on the pyplot module (they live on Axes objects).
        # guarded_getattr returns None for missing attrs, then None(...) would
        # crash with 'NoneType' not callable.  Stub them as no-ops instead.
        for _stub in ('set_title', 'set_xlabel', 'set_ylabel',
                      'set_xticks', 'set_yticks', 'set_xticklabels',
                      'set_yticklabels'):
            if not hasattr(_plt, _stub):
                setattr(_plt, _stub, lambda *a, **kw: None)
        import seaborn as _sns
        from . import visualization as _viz

        os.makedirs(charts_output_dir, exist_ok=True)
        chart_path = os.path.join(
            charts_output_dir,
            f'{chart_backend}_{_dt.datetime.now().strftime("%Y%m%d_%H%M%S")}_{_uuid.uuid4().hex[:6]}.png',
        )
        return {
            'df': df,
            'pd': pd,
            'np': np,
            'os': os,
            'plt': _plt,
            'sns': _sns,
            'chart_path': chart_path,
            # Chart helpers (called as trusted functions; not restricted)
            'area_chart': _viz.area_chart,
            'bar_chart': _viz.bar_chart,
            'box_chart': _viz.box_chart,
            'bubble_chart': _viz.bubble_chart,
            'count_chart': _viz.count_chart,
            'ecdf_chart': _viz.ecdf_chart,
            'funnel_chart': _viz.funnel_chart,
            'heatmap': _viz.heatmap,
            'histogram': _viz.histogram,
            'kde_chart': _viz.kde_chart,
            'line_chart': _viz.line_chart,
            'pairplot_chart': _viz.pairplot_chart,
            'pie_chart': _viz.pie_chart,
            'regression_chart': _viz.regression_chart,
            'scatter_chart': _viz.scatter_chart,
            'stacked_bar_chart': _viz.stacked_bar_chart,
            'step_chart': _viz.step_chart,
            'strip_chart': _viz.strip_chart,
            'swarm_chart': _viz.swarm_chart,
            'violin_chart': _viz.violin_chart,
        }

    @staticmethod
    def _format_sandbox_result(result) -> str:
        """Normalise the raw sandbox result into a string."""
        if isinstance(result, dict) and result.get('type') == 'plot':
            return str(result.get('value', ''))
        if isinstance(result, pd.DataFrame):
            return result.to_string()
        if result is None:
            return 'No result returned.'
        return str(result)

    def get_statistics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Get basic statistics from DataFrame."""
        return {
            "shape": df.shape,
            "columns": df.columns.tolist(),
            "dtypes": df.dtypes.to_dict(),
            "null_count": df.isnull().sum().to_dict(),
            "description": df.describe().to_dict(),
        }
    
    def generate_insights(
        self,
        df: pd.DataFrame,
        num_insights: int = 3
    ) -> List[str]:
        """Generate automatic insights from DataFrame."""
        insights = []
        
        prompts = [
            f"What are the key statistics and summary for this data? Provide {num_insights} important insights.",
            f"What patterns or trends do you observe in this dataset? Highlight {num_insights} key findings.",
            f"What are the highest and lowest values, and what would you recommend to analyze further?",
        ]
        
        try:
            for prompt in prompts:
                result = self.analyze(df, prompt)
                if not result.startswith("Error"):
                    insights.append(result)
        except Exception as e:
            if self.verbose:
                print(f"Error generating insights: {str(e)}")
        
        return insights[:num_insights]
    
    def switch_model(self, model_name: str) -> None:
        """Switch to a different Ollama model (legacy path only)."""
        self.model_name = model_name
        self._passthrough_llm = None
        self._initialize_llm(llm_override=None)
        if self.verbose:
            print(f"✓ Switched to {model_name} model")
    
    def switch_provider(
        self,
        model_name: str,
        provider_type: str = "ollama",
        api_key: str = None
    ) -> None:
        """
        Switch to a different LLM provider/model.
        
        Args:
            model_name: Model to switch to
            provider_type: Provider type ('ollama', 'openai', 'deepseek', 'github')
            api_key: API key for cloud providers
        """
        self.model_name = model_name
        self.provider_type = provider_type
        self.api_key = api_key
        self._passthrough_llm = None
        self._initialize_llm(llm_override=None)
        if self.verbose:
            print(f"✓ Switched to {provider_type}/{model_name}")
