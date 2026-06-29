"""agent.py — PyChartAgent: a pandasai-independent NL → code → sandbox agent.

This module provides a drop-in alternative to ``pandasai.Agent`` that works
entirely within the pychartai ecosystem — no pandasai dependency required.

Architecture::

    User query
        │
        ▼
    _build_prompt()         ← DataFrame schema + sample + chart specs
        │
        ▼
    provider.generate()     ← any pychartai LLM provider (Ollama, OpenAI, …)
        │
        ▼
    _extract_code()         ← strip markdown fences / <think> blocks
        │
        ▼
    _sanitize_code()        ← remove hallucinated patterns, fix syntax
        │
        ▼
    sandbox.execute()       ← RestrictedSandbox (or DockerSandbox)
        │
        ▼
    _format_result()        ← normalise to str / chart file path

Usage::

    import pychartai as pai

    llm = pai.OllamaLLM(model='llama3.2')
    pai.config.set({'llm': llm})

    agent = pai.PyChartAgent()                      # picks up global LLM
    df = pai.read_csv('data/sales.csv')
    print(agent.chat('Average revenue by region?', df._df))

    # Or via SmartDataFrame:
    sdf = pai.SmartDataFrame(raw_df)
    sdf.chat('Plot bar chart of revenue by region', agent='own')
"""

from __future__ import annotations

import datetime as _dt
import logging as _logging
import os
import re
import time
import uuid as _uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

_log = _logging.getLogger('pychartai')

import pandas as pd


@dataclass
class TransformationLog:
	"""Log of operations applied during a query execution."""

	query: str = ''
	generated_code: str = ''
	intent: Optional['QueryIntent'] = None
	attempts: int = 0
	success: bool = False
	error: Optional[str] = None


_CHART_HELPERS: tuple[str, ...] = (
	'area_chart', 'bar_chart', 'box_chart', 'bubble_chart', 'count_chart',
	'ecdf_chart', 'funnel_chart', 'heatmap', 'histogram', 'kde_chart',
	'line_chart', 'pairplot_chart', 'pie_chart', 'regression_chart',
	'scatter_chart', 'stacked_bar_chart', 'step_chart', 'strip_chart',
	'swarm_chart', 'violin_chart',
)


@dataclass(frozen=True)
class QueryIntent:
	"""Lightweight classification of the user's query."""

	kind: str
	needs_grouping: bool = False
	needs_time_bucketing: bool = False
	expects_scalar: bool = False
	expects_table: bool = False
	chart_helper: Optional[str] = None
	group_by_hint: Optional[str] = None


class PyChartAgent:
	"""A pandasai-independent agent for natural-language DataFrame analysis.

	Generates Python code via an LLM provider, then executes it inside a
	:class:`~pychartai_core.sandbox.RestrictedSandbox` (or DockerSandbox).
	All 20 visualization helpers are injected into the sandbox context so
	chart queries work out-of-the-box.

	Args:
		llm:               LLM instance (``OllamaLLM``, ``OpenAILLM``, etc.).
		                   Falls back to ``pai.config`` global LLM if omitted.
		chart_backend:     Visualization backend — ``'seaborn'``, ``'matplotlib'``,
		                   or ``'plotly'``.  Defaults to ``'seaborn'``.
		charts_output_dir: Directory where generated charts are saved.
		verbose:           Print generated code and retry messages.
		max_retries:       Number of code-generation retries on failure.
	"""

	def __init__(
		self,
		llm=None,
		chart_backend: str = 'seaborn',
		charts_output_dir: str = 'exports/charts',
		verbose: bool = False,
		max_retries: Optional[int] = None,
		memory: Optional['ConversationMemory'] = None,
	) -> None:
		from .config import config as global_config
		self.chart_backend = chart_backend
		self.charts_output_dir = charts_output_dir
		self.verbose = verbose
		self.max_retries = max_retries if max_retries is not None else global_config.get('max_retries', 3)
		self._llm_timeout = global_config.get('llm_timeout', 60)
		self._provider = self._resolve_provider(llm)
		self._llm = llm  # keep a reference so we can read last_usage
		self._memory = memory
		self.last_transformation: Optional[TransformationLog] = None
		self.last_usage: Dict[str, Any] = {}  # token counts from the most recent LLM call

	# ------------------------------------------------------------------
	# Provider resolution
	# ------------------------------------------------------------------

	@staticmethod
	def _resolve_provider(llm):
		"""Return a low-level provider that exposes ``.generate(prompt)``.

		DIP: this method depends on the PyChartLLM abstraction, not on any
		concrete provider class.
		"""
		if llm is None:
			from .config import config as global_config
			llm = global_config.get('llm')
		if llm is None:
			raise RuntimeError(
				'No LLM configured. '
				"Call pai.config.set({'llm': pai.OllamaLLM(model='llama3.2')}) first."
			)
		# All pychartai public LLM classes (PyChartLLM + subclasses) expose _get_provider()
		from .providers import PyChartLLM as _PyChartLLM
		if isinstance(llm, _PyChartLLM):
			return llm._get_provider()
		# Raw provider passed directly (already has .generate())
		if hasattr(llm, 'generate'):
			return llm
		raise TypeError(
			f'Unsupported LLM type: {type(llm).__name__}. '
			'Pass a PyChartLLM (OllamaLLM, OpenAILLM, etc.) or any object with a .generate() method.'
		)

	# ------------------------------------------------------------------
	# LLM call with timeout
	# ------------------------------------------------------------------

	def _generate_with_timeout(self, prompt: str) -> str:
		"""Call the provider with a timeout guard and capture token usage."""
		import concurrent.futures
		with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
			future = pool.submit(self._provider.generate, prompt)
			try:
				result = future.result(timeout=self._llm_timeout)
				# Capture token usage if the provider exposes it (cloud models only)
				usage = getattr(self._provider, 'last_usage', {})
				if usage:
					self.last_usage = dict(usage)
				return result
			except concurrent.futures.TimeoutError:
				raise TimeoutError(
					f'LLM generation timed out after {self._llm_timeout}s. '
					'Increase llm_timeout in config or use a faster model.'
				)

	# ------------------------------------------------------------------
	# Primary interface
	# ------------------------------------------------------------------

	def chat(
		self,
		query: str,
		df,           # pd.DataFrame | list[pd.DataFrame] | dict[str, pd.DataFrame]
		sandbox=None,
		chart_backend: Optional[str] = None,
		explain: bool = False,
		on_progress=None,   # Optional[Callable[[str, str], None]]
	) -> str:
		"""Execute a natural-language query against one or more DataFrames.

		Args:
			query:         Natural-language question or chart instruction.
			df:            A single ``pd.DataFrame``, a list of DataFrames, or a
			               dict mapping names to DataFrames.  When multiple frames
			               are passed they are merged (auto-joined on shared columns)
			               into a single working frame before analysis.
			sandbox:       Sandbox instance.  Defaults to
			               :class:`~pychartai_core.sandbox.RestrictedSandbox`.
			chart_backend: Override the visualization backend for this call.
			explain:       If True, append an LLM-generated explanation of the result.
			on_progress:   Optional callback ``fn(stage: str, detail: str)`` called at
			               each processing stage.  Stages: ``'classifying'``,
			               ``'generating'``, ``'executing'``, ``'formatting'``.

		Returns:
			Analysis result as a string, or a chart file path.
		"""
		from .error_hints import get_hint

		def _progress(stage: str, detail: str = '') -> None:
			if on_progress is not None:
				try:
					on_progress(stage, detail)
				except Exception:
					pass  # never let a progress callback crash the pipeline

		# --- normalise multi-DataFrame input ---
		df, merge_note = self._coerce_dataframes(df)

		if df.empty:
			raise ValueError('DataFrame is empty — cannot analyse an empty table.')

		# --- input validation & prompt-injection guard ---
		query = self._validate_query(query)

		if sandbox is None:
			from .sandbox import RestrictedSandbox
			sandbox = RestrictedSandbox()

		effective_backend = chart_backend or self.chart_backend

		from .sandbox import DockerSandbox
		is_docker = isinstance(sandbox, DockerSandbox)
		_progress('classifying', query)
		intent = self._classify_query_intent(query, df)
		prompt = self._build_prompt(df, query, effective_backend, intent, allow_charts=not is_docker)
		if merge_note:
			prompt = merge_note + '\n' + prompt

		# Inject conversation memory context
		if self._memory and len(self._memory) > 0:
			memory_ctx = self._memory.get_context(max_turns=5)
			prompt = memory_ctx + '\n\n' + prompt

		# Transformation tracking
		xform = TransformationLog(query=query, intent=intent)
		last_error: Optional[str] = None
		last_exc: Optional[Exception] = None
		_t0 = time.monotonic()

		_log.debug(
			'pychartai.chat.start query=%r intent=%s backend=%s',
			query[:120], intent.kind, effective_backend,
		)

		for attempt in range(self.max_retries):
			xform.attempts = attempt + 1
			try:
				active_prompt = prompt if attempt == 0 else self._build_retry_prompt(prompt, last_error)
				_progress('generating', f'attempt {attempt + 1}/{self.max_retries}')
				raw = self._generate_with_timeout(active_prompt)
				code = self._extract_code(raw)
				code = self._sanitize_code(code, allow_charts=intent.kind == 'chart')
				code = self._normalize_chart_code(code, intent, effective_backend, df, query)
				xform.generated_code = code

				if self.verbose:
					print(f'[PyChartAgent] attempt={attempt + 1} generated code:\n{code}\n')

				_progress('executing', intent.kind)
				if is_docker:
					result = sandbox.execute(code, {'df': df})
				else:
					context = self._build_context(df, effective_backend)
					result = sandbox.execute(code, context)
				self._validate_result(result, query, intent)

				_progress('formatting', '')
				formatted = self._format_result(result)
				xform.success = True
				self.last_transformation = xform

				# Record turn in memory
				chart_path = formatted if isinstance(result, str) and result.endswith(('.png', '.html', '.svg')) else None
				if self._memory is not None:
					self._memory.add(query, formatted, chart_path=chart_path)

				# Optionally explain the result
				if explain and intent.kind != 'chart':
					explanation = self._explain_result(query, formatted, df)
					if explanation:
						formatted = f'{formatted}\n\nExplanation: {explanation}'

				_log.info(
					'pychartai.chat.success query=%r intent=%s attempt=%d duration=%.3fs tokens=%s',
					query[:80], intent.kind, attempt + 1,
					time.monotonic() - _t0,
					self.last_usage.get('total_tokens', 'n/a'),
				)
				return formatted

			except Exception as exc:
				last_error = str(exc)
				last_exc = exc
				xform.error = last_error
				# Non-retryable errors: re-raise immediately with hint
				is_auth_error = (
					'API key' in last_error
					or 'api_key' in last_error
					or 'authentication' in last_error.lower()
				)
				if isinstance(exc, (TimeoutError, ConnectionError, PermissionError)) or is_auth_error:
					self.last_transformation = xform
					hint = get_hint(last_error)
					msg = f'{exc}' + (f'\nHint: {hint}' if hint else '')
					raise type(exc)(msg) from exc
				if attempt < self.max_retries - 1:
					delay = min(2 ** attempt, 30)  # 1s, 2s, 4s, … capped at 30s
					if self.verbose:
						print(f'[PyChartAgent] retry {attempt + 1}/{self.max_retries - 1} in {delay}s: {exc}')
					time.sleep(delay)
					continue

		self.last_transformation = xform
		hint = get_hint(last_error or '')
		msg = f'Query failed after {self.max_retries} attempt(s): {last_error}'
		if hint:
			msg += f'\nHint: {hint}'
		_log.warning(
			'pychartai.chat.failed query=%r attempts=%d duration=%.3fs error=%r',
			query[:80], self.max_retries, time.monotonic() - _t0, last_error,
		)
		raise RuntimeError(msg) from last_exc

	def _explain_result(self, query: str, result: str, df: pd.DataFrame) -> Optional[str]:
		"""Ask the LLM to explain a result in plain English."""
		try:
			prompt = (
				f'The user asked: "{query}"\n'
				f'The computed answer is:\n{result[:500]}\n\n'
				f'Explain this result concisely in 1-3 sentences for a non-technical audience. '
				f'No code, no markdown, just a plain English explanation.'
			)
			return self._generate_with_timeout(prompt).strip()
		except Exception:
			return None

	# ------------------------------------------------------------------
	# Input validation
	# ------------------------------------------------------------------

	# Phrases that commonly appear in prompt-injection attempts.
	# We warn (not block) so legitimate queries that happen to contain
	# these words are not silently dropped.
	_INJECTION_PATTERNS: List[str] = [
		r'ignore\s+(the\s+)?(previous|above|prior|all)\s+instructions?',
		r'forget\s+(the\s+)?(previous|above|prior|all)\s+instructions?',
		r'disregard\s+(the\s+)?(previous|above|prior|all)',
		r'you\s+are\s+now\s+(?:a|an)\s+\w+',
		r'act\s+as\s+(?:a|an)\s+\w+',
		r'new\s+system\s+prompt',
		r'override\s+(your\s+)?instructions?',
	]
	_INJECTION_RE = [re.compile(p, re.IGNORECASE) for p in _INJECTION_PATTERNS]
	_MAX_QUERY_LEN = 4000  # chars — larger prompts are almost always injection attempts

	def _validate_query(self, query: str) -> str:
		"""Sanitize and validate user query before it reaches the LLM.

		- Strips null bytes and ASCII control characters.
		- Enforces a maximum length (configurable via ``config.set({'max_query_len': N})``).
		- Logs a warning when injection patterns are detected (does not block).

		Returns the cleaned query string.
		"""
		from .config import config as _cfg
		max_len = _cfg.get('max_query_len', self._MAX_QUERY_LEN)

		# Strip null bytes and non-printable ASCII control chars (except \n \t)
		cleaned = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', query)

		if len(cleaned) > max_len:
			_log.warning(
				'pychartai.query.truncated original_len=%d max_len=%d',
				len(cleaned), max_len,
			)
			cleaned = cleaned[:max_len]

		for pattern in self._INJECTION_RE:
			if pattern.search(cleaned):
				_log.warning(
					'pychartai.query.injection_suspect pattern=%r query_preview=%r',
					pattern.pattern[:60], cleaned[:120],
				)
				break  # one warning is enough

		return cleaned

	# ------------------------------------------------------------------
	# Prompt construction
	# ------------------------------------------------------------------

	def _build_prompt(
		self,
		df: pd.DataFrame,
		query: str,
		chart_backend: str,
		intent: QueryIntent,
		allow_charts: bool = True,
	) -> str:
		"""Construct a code-generation prompt tailored to the detected intent."""
		col_desc = ', '.join(f'{c} ({dt})' for c, dt in df.dtypes.items())
		sample_str = df.head(3).to_string()
		date_columns = [col for col in df.columns if pd.api.types.is_datetime64_any_dtype(df[col])]

		header = (
			f'You have a pandas DataFrame `df` with {len(df)} rows.\n'
			f'Columns: {col_desc}\n'
			f'Sample (first 3 rows):\n{sample_str}\n\n'
			f'Task: {query}\n\n'
		)

		# --- per-intent template ---
		if intent.kind == 'filter':
			body = self._prompt_filter(df, intent)
		elif intent.kind == 'count':
			body = self._prompt_count(df, intent)
		elif intent.kind == 'aggregation':
			body = self._prompt_aggregation(df, intent, date_columns)
		elif intent.kind == 'chart' and allow_charts:
			body = self._prompt_chart(df, intent, chart_backend)
		else:
			body = self._prompt_generic(df, intent, date_columns)

		footer = (
			'RULES:\n'
			'- `df` is already in scope. Do NOT re-import or redefine it.\n'
			'- Use ONLY direct pandas operations. Do NOT call execute_sql_query.\n'
			'- To compute a derived value, use df.assign():\n'
			"    e.g. `tmp = df.assign(revenue=df['price'] * df['quantity'])`\n"
			'- Existing column names (case-sensitive): ' + ', '.join(f'`{c}`' for c in df.columns) + '\n'
			'- Do NOT call subprocess, open(), or network functions.\n'
			'- Return ONLY the Python code block. No explanation, no commentary.\n'
		)

		return header + body + footer

	@staticmethod
	def _prompt_filter(df: pd.DataFrame, intent: QueryIntent) -> str:
		return (
			'Instructions (filter):\n'
			'1. Filter `df` and store the matching rows in `result`.\n'
			'2. result must be a DataFrame (or Series), not a scalar.\n'
			'3. Example:\n'
			"   result = df[df['col'] > value]\n\n"
		)

	@staticmethod
	def _prompt_count(df: pd.DataFrame, intent: QueryIntent) -> str:
		return (
			'Instructions (count):\n'
			'1. Compute a single numeric count and store it in `result`.\n'
			'2. result must be a scalar (int or string), not a DataFrame.\n'
			'3. Examples:\n'
			"   result = df['col'].nunique()\n"
			"   result = len(df[df['col'] > value])\n\n"
		)

	@staticmethod
	def _prompt_aggregation(df: pd.DataFrame, intent: QueryIntent, date_columns: list) -> str:
		groupby_hint = f'`{intent.group_by_hint}`' if intent.group_by_hint else 'the appropriate column'
		date_hint = ''
		if intent.needs_time_bucketing and date_columns:
			date_hint = (
				f'   - For time bucketing, use `{date_columns[0]}.dt.to_period("M")` or similar.\n'
			)
		scalar_hint = 'a single scalar (e.g. a float)' if intent.expects_scalar else f'a grouped table grouped by {groupby_hint}'
		grp_col = intent.group_by_hint or 'col'
		return (
			f'Instructions (aggregation):\n'
			f'1. Compute the aggregation and store it in `result`.\n'
			f'2. result should be {scalar_hint}.\n'
			f'3. If a column you need does not exist, compute it first with df.assign():\n'
			f"   tmp = df.assign(revenue=df['price'] * df['quantity'])\n"
			f"   result = tmp.groupby('region')['revenue'].sum().reset_index()\n"
			f'{date_hint}'
			f'4. Examples:\n'
			f"   result = df.groupby('{grp_col}')['value_col'].sum().reset_index()\n"
			f"   result = df['col'].mean()\n\n"
		)

	def _prompt_chart(self, df: pd.DataFrame, intent: QueryIntent, chart_backend: str) -> str:
		from .visualization_backends import get_backend_chart_specs
		from .visualization import describe_backend

		chart_specs = get_backend_chart_specs(chart_backend)
		helper_names = ', '.join(s.helper_name for s in chart_specs)
		chart_lines = '\n'.join(f'#   {line}' for line in describe_backend(chart_backend))
		chart_example = self._build_chart_example(intent, chart_backend)
		helper = intent.chart_helper or 'bar_chart'
		return (
			f'Instructions (chart):\n'
			f'1. Prepare the data `df` needs to be structured for the chart function.\n'
			f"   - If a column you need doesn't exist, compute it with df.assign():\n"
			f"     tmp = df.assign(revenue=df['price'] * df['quantity'])\n"
			f'2. Call the chart helper and assign its return value to `result`.\n'
			f'3. Use ONLY the `{helper}` helper. Never `df.plot()`, never seaborn/plotly directly.\n'
			f'4. All chart helpers are already in scope. Do NOT import them.\n'
			f'5. Use the pre-defined variable `chart_path` as the output file.\n'
			f"6. Always pass `backend='{chart_backend}'` to the helper.\n"
			f'7. Example:\n'
			f'   {chart_example}\n'
			f'8. Available helpers: {helper_names}\n'
			f'9. Helper signatures:\n{chart_lines}\n\n'
		)

	@staticmethod
	def _prompt_generic(df: pd.DataFrame, intent: QueryIntent, date_columns: list) -> str:
		date_hint = ''
		if date_columns:
			date_hint = f'   - Date columns available: {", ".join(date_columns)}\n'
		return (
			'Instructions:\n'
			'1. Analyse `df` and compute the answer to the task.\n'
			'2. Store the result in `result` (string, number, Series, or DataFrame).\n'
			f'3. Tips:\n'
			f'   - If a computed column is needed, use df.assign():\n'
			f"     tmp = df.assign(revenue=df['price'] * df['quantity'])\n"
			f'{date_hint}'
			'\n'
		)

	@staticmethod
	def _build_retry_prompt(base_prompt: str, error: Optional[str]) -> str:
		"""Inject execution feedback for a retry attempt."""
		if not error:
			return base_prompt
		return (
			base_prompt
			+ '\nPrevious attempt failed with this error:\n'
			+ error
			+ '\nFix the code and return only corrected Python code.\n'
		)

	# ------------------------------------------------------------------
	# Multi-DataFrame helpers
	# ------------------------------------------------------------------

	@staticmethod
	def _coerce_dataframes(df) -> tuple:
		"""Normalise *df* into a single :class:`pandas.DataFrame`.

		Accepts:
		  * a single ``pd.DataFrame`` → returned as-is, no note.
		  * a ``list`` of DataFrames   → named ``df0``, ``df1``, … auto-merged.
		  * a ``dict[str, DataFrame]`` → merged using provided names.

		Auto-merge strategy:
		  1. Find columns shared by all DataFrames.
		  2. If any common column exists, perform successive ``pd.merge`` (inner
		     join) on those shared columns.
		  3. If no common column exists, ``pd.concat`` along axis=0 instead.

		Returns (merged_df, note_str).  *note_str* is a string prepended to the
		prompt when multiple DataFrames were supplied, or '' otherwise.
		"""
		if isinstance(df, pd.DataFrame):
			return df, ''

		if isinstance(df, list):
			if not df:
				return pd.DataFrame(), ''
			if len(df) == 1:
				return df[0], ''
			dfs: dict = {f'df{i}': d for i, d in enumerate(df)}
		elif isinstance(df, dict):
			if not df:
				return pd.DataFrame(), ''
			if len(df) == 1:
				return next(iter(df.values())), ''
			dfs = dict(df)
		else:
			# unsupported type — let the caller deal with it
			return df, ''  # type: ignore[return-value]

		# Build a schema note for the prompt
		schema_lines = ['# Multiple DataFrames provided:']
		for name, d in dfs.items():
			cols = ', '.join(f'{c} ({d[c].dtype})' for c in d.columns)
			schema_lines.append(f'#   {name}: {d.shape[0]} rows × {d.shape[1]} cols — {cols}')
		merge_note = '\n'.join(schema_lines)

		frames = list(dfs.values())

		# Attempt merge on common columns
		common = set(frames[0].columns)
		for frame in frames[1:]:
			common &= set(frame.columns)
		common_cols = sorted(common)

		try:
			if common_cols:
				merged = frames[0]
				for frame in frames[1:]:
					merged = pd.merge(merged, frame, on=common_cols, how='inner')
				merge_note += f'\n# Auto-merged on columns: {common_cols}'
			else:
				merged = pd.concat(frames, axis=0, ignore_index=True)
				merge_note += '\n# No common columns found — DataFrames concatenated by row.'
		except Exception as exc:
			# Fallback: just use the first frame
			merged = frames[0]
			merge_note += f'\n# Merge failed ({exc}); using first DataFrame only.'

		return merged, merge_note

	@staticmethod
	def _classify_query_intent(query: str, df: pd.DataFrame) -> QueryIntent:
		"""Infer broad execution intent from query text.

		Uses a fast keyword-based check first.  Falls back to a generic
		intent when the query is ambiguous — in the future an LLM classifier
		can refine these cases.
		"""
		lowered = query.lower()
		chart_map = {
			'pie': 'pie_chart',
			'histogram': 'histogram',
			'hist': 'histogram',
			'bar': 'bar_chart',
			'line': 'line_chart',
			'scatter': 'scatter_chart',
			'box': 'box_chart',
			'violin': 'violin_chart',
			'area': 'area_chart',
			'heatmap': 'heatmap',
			'bubble': 'bubble_chart',
			'funnel': 'funnel_chart',
			'kde': 'kde_chart',
			'ecdf': 'ecdf_chart',
			'pairplot': 'pairplot_chart',
			'regression': 'regression_chart',
			'stacked': 'stacked_bar_chart',
			'step': 'step_chart',
			'strip': 'strip_chart',
			'swarm': 'swarm_chart',
		}
		for keyword, helper in chart_map.items():
			if keyword in lowered:
				return QueryIntent(kind='chart', chart_helper=helper)
		if any(token in lowered for token in ('chart', 'plot', 'graph')):
			return QueryIntent(kind='chart', chart_helper=None)

		group_by_hint = PyChartAgent._extract_group_by_hint(lowered, df)

		needs_time = any(token in lowered for token in ('per month', 'monthly', 'per year', 'yearly', 'per week', 'weekly', 'trend'))
		if any(token in lowered for token in ('show all', 'rows where', 'filter', 'greater than', 'less than')):
			return QueryIntent(kind='filter', expects_table=True)
		if any(token in lowered for token in ('how many', 'count', 'number of', 'unique')):
			return QueryIntent(kind='count', expects_scalar=True)
		if any(token in lowered for token in ('average', 'mean', 'sum', 'total', 'highest', 'lowest', 'top', 'max', 'min')):
			return QueryIntent(
				kind='aggregation',
				needs_grouping=bool(group_by_hint),
				needs_time_bucketing=needs_time,
				expects_scalar=not bool(group_by_hint or needs_time),
				expects_table=bool(group_by_hint or needs_time),
				group_by_hint=group_by_hint,
			)
		return QueryIntent(kind='generic', needs_time_bucketing=needs_time)

	def classify_with_llm(self, query: str, df: pd.DataFrame) -> QueryIntent:
		"""Use the LLM for intent classification when keyword matching is uncertain.

		This is opt-in — call it explicitly for ambiguous queries.  The fast
		keyword-based classifier is always tried first.
		"""
		cols = ', '.join(f'{c} ({dt})' for c, dt in df.dtypes.items())
		prompt = (
			f'Classify this query into exactly one category.\n'
			f'Columns: {cols}\n'
			f'Query: "{query}"\n'
			f'Categories: chart, filter, count, aggregation, generic\n'
			f'If chart, also state the chart type from: bar, line, scatter, pie, histogram, '
			f'box, violin, area, heatmap, bubble, funnel, kde, ecdf, pairplot, regression, '
			f'stacked_bar, step, strip, swarm.\n'
			f'Reply with ONLY the category (and chart type if applicable) like: chart/bar or filter'
		)
		try:
			raw = self._generate_with_timeout(prompt).strip().lower()
			if raw.startswith('chart'):
				parts = raw.split('/')
				helper = parts[1].strip() + '_chart' if len(parts) > 1 else None
				if helper and helper not in _CHART_HELPERS:
					helper = None
				return QueryIntent(kind='chart', chart_helper=helper)
			if raw in ('filter', 'count', 'aggregation'):
				return QueryIntent(kind=raw, expects_scalar=(raw == 'count'), expects_table=(raw == 'filter'))
			return QueryIntent(kind='generic')
		except Exception:
			return self._classify_query_intent(query, df)

	@staticmethod
	def _extract_group_by_hint(lowered_query: str, df: pd.DataFrame) -> Optional[str]:
		"""Infer grouping column from phrases like 'by region' or 'per product'."""
		for pattern in (r'\bby\s+([a-zA-Z_][a-zA-Z0-9_]*)', r'\bper\s+([a-zA-Z_][a-zA-Z0-9_]*)'):
			match = re.search(pattern, lowered_query)
			if not match:
				continue
			candidate = match.group(1).rstrip('?.!,;:')
			if candidate in {'month', 'year', 'week', 'day'}:
				continue
			for column in df.columns:
				if column.lower() == candidate:
					return column
		return None

	@staticmethod
	def _build_intent_instructions(intent: QueryIntent, df: pd.DataFrame) -> str:
		"""Add targeted guidance based on detected query shape."""
		if intent.kind == 'filter':
			return (
				'Intent guidance: this is a filtering task. Return the filtered rows, '
				'not a chart and not a scalar summary.\n'
			)
		if intent.kind == 'count':
			return (
				'Intent guidance: this is a counting task. Return a single scalar value, '
				'not a chart and not a full table.\n'
			)
		if intent.kind == 'aggregation':
			parts = ['Intent guidance: this is an aggregation task.']
			if intent.needs_grouping and intent.group_by_hint:
				parts.append(f'Group by `{intent.group_by_hint}` and return the grouped result.')
			if intent.needs_time_bucketing:
				date_columns = [col for col in df.columns if pd.api.types.is_datetime64_any_dtype(df[col])]
				if date_columns:
					parts.append(f'Use `{date_columns[0]}` to bucket data by time.')
			if intent.expects_scalar:
				parts.append('Return a single scalar or short text answer.')
			else:
				parts.append('Return a grouped table or Series, not a single scalar.')
			return ' '.join(parts) + '\n'
		if intent.kind == 'chart':
			return (
				'Intent guidance: this is a chart task. Build the data needed for the chart first, '
				'then assign the helper return value directly to `result`.\n'
			)
		return ''

	@staticmethod
	def _build_chart_example(intent: QueryIntent, chart_backend: str) -> str:
		"""Return a helper-specific example line for chart queries."""
		helper = intent.chart_helper or 'bar_chart'
		b = chart_backend

		# Helpers that take (df, labels=, values=)
		if helper == 'pie_chart':
			return (
				"tmp = df.groupby('category')['value'].sum().reset_index(); "
				f"result = pie_chart(tmp, labels='category', values='value', title='Title', output_file=chart_path, backend='{b}')"
			)
		if helper == 'funnel_chart':
			return (
				"tmp = df.groupby('stage')['count'].sum().reset_index(); "
				f"result = funnel_chart(tmp, labels='stage', values='count', title='Title', output_file=chart_path, backend='{b}')"
			)
		# Helpers that take (df, column=)
		if helper in ('histogram', 'kde_chart', 'ecdf_chart'):
			return f"result = {helper}(df, column='value', title='Title', output_file=chart_path, backend='{b}')"
		# Helpers that take (df) only
		if helper in ('heatmap', 'pairplot_chart'):
			return f"result = {helper}(df, title='Title', output_file=chart_path, backend='{b}')"
		# Helpers that take (df, x=)
		if helper == 'count_chart':
			return f"result = count_chart(df, x='category', title='Title', output_file=chart_path, backend='{b}')"
		# Helpers that take (df, x=, y=, size=)
		if helper == 'bubble_chart':
			return f"result = bubble_chart(df, x='col_x', y='col_y', size='col_size', title='Title', output_file=chart_path, backend='{b}')"
		# Helpers that take (df, x=, y=, stack=)
		if helper == 'stacked_bar_chart':
			return f"result = stacked_bar_chart(df, x='category', y='value', stack='group', title='Title', output_file=chart_path, backend='{b}')"
		# Default: x/y chart helpers (bar, line, scatter, box, violin, area, step, strip, swarm, regression)
		return f"result = {helper}(df, x='col_x', y='col_y', title='Title', output_file=chart_path, backend='{b}')"

	# ------------------------------------------------------------------
	# Code extraction (pandasai-independent)
	# ------------------------------------------------------------------

	@staticmethod
	def _extract_code(response: str) -> str:
		"""Extract Python code from an LLM response, stripping markdown fences."""
		# Strip chain-of-thought / reasoning blocks
		clean = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL).strip()

		# Prefer fenced ```python … ``` blocks
		for pattern in (r'```python\s*(.*?)```', r'```\s*(.*?)```'):
			match = re.search(pattern, clean, re.DOTALL)
			if match:
				code = match.group(1).strip()
				if code:
					return code

		# Fall back to the full (cleaned) text
		return clean.strip()

	# ------------------------------------------------------------------
	# Code sanitisation (pandasai-independent)
	# ------------------------------------------------------------------

	@staticmethod
	def _sanitize_code(code: str, allow_charts: bool = True) -> str:
		"""Remove common LLM hallucination patterns and fix obvious syntax errors."""
		code = re.sub(r'^\s*from\s+(?:seaborn|matplotlib(?:\.pyplot)?|plotly(?:\.express)?)\s+import\s+.*$', '', code, flags=re.MULTILINE)
		code = re.sub(r'^\s*import\s+(?:seaborn|matplotlib(?:\.pyplot)?|plotly(?:\.express)?)\b.*$', '', code, flags=re.MULTILINE)
		code = re.sub(r'^\s*from\s+charts?\s+import\s+.*$', '', code, flags=re.MULTILINE)
		code = re.sub(r'^\s*import\s+charts?\b.*$', '', code, flags=re.MULTILINE)
		# Rename underscore-prefixed user variables (forbidden by RestrictedPython)
		# e.g. `_result` → `result`, `_result_path` → `result_path`
		# Exclude Python dunder names and RestrictedPython's own `_getattr_`-style names.
		code = re.sub(
			r'\b(_)([a-zA-Z][a-zA-Z0-9_]*)\b',
			lambda m: m.group(2) if not m.group(0).endswith('_') else m.group(0),
			code,
		)
		# Catch trailing-underscore names that still have a leading underscore
		# (e.g. `_var_`) — just strip both underscores.
		code = re.sub(r'\b_([a-zA-Z][a-zA-Z0-9]*)_\b', r'\1', code)

		lines = code.splitlines()
		cleaned: List[str] = []

		i = 0
		while i < len(lines):
			line = lines[i]
			stripped = line.strip()

			if stripped.startswith('print('):
				i += 1
				continue

			# Strip any attempt to redefine execute_sql_query
			if re.match(r'def\s+execute_sql_query\s*\(', stripped):
				cleaned.append('# [sanitized] removed execute_sql_query redefinition')
				func_indent = len(line) - len(line.lstrip())
				i += 1
				while i < len(lines):
					tline = lines[i]
					if tline.strip() and (len(tline) - len(tline.lstrip())) <= func_indent:
						break
					i += 1
				continue

			# Remove result = {'type': 'plot', 'value': plt}  (plt is not serialisable)
			if re.search(r"result\s*=\s*\{[^}]*['\"]plot['\"][^}]*\bplt\b[^}]*\}", stripped):
				cleaned.append('# [sanitized] removed plt-object result dict')
				i += 1
				continue

			if not allow_charts and re.search(r'\b(?:bar_chart|line_chart|scatter_chart|pie_chart|histogram|heatmap|box_chart|violin_chart|count_chart|kde_chart|area_chart|pairplot_chart|regression_chart|stacked_bar_chart|bubble_chart|funnel_chart|ecdf_chart|step_chart|strip_chart|swarm_chart)\s*\(', stripped):
				cleaned.append('# [sanitized] removed chart helper call for non-chart query')
				i += 1
				continue

			if not allow_charts and re.search(r'\.plot\s*\(', stripped):
				cleaned.append('# [sanitized] removed plotting call for non-chart query')
				i += 1
				continue

			cleaned.append(line)
			i += 1

		sanitized = '\n'.join(cleaned)

		# Auto-close mismatched brackets so the code at least compiles
		try:
			compile(sanitized, '<string>', 'exec')
		except SyntaxError:
			open_braces = sanitized.count('{') - sanitized.count('}')
			open_brackets = sanitized.count('[') - sanitized.count(']')
			open_parens = sanitized.count('(') - sanitized.count(')')
			if open_braces > 0:
				sanitized += '\n' + '}' * open_braces
			if open_brackets > 0:
				sanitized += '\n' + ']' * open_brackets
			if open_parens > 0:
				sanitized += '\n' + ')' * open_parens

		return sanitized

	@staticmethod
	def _normalize_chart_code(
		code: str,
		intent: QueryIntent,
		chart_backend: str,
		df: pd.DataFrame,
		query: str,
	) -> str:
		"""Rewrite common chart hallucinations into pychartai helper calls."""
		if intent.kind != 'chart':
			return code

		code = re.sub(r'^\s*from\s+charts?\s+import\s+.*$', '', code, flags=re.MULTILINE)
		code = re.sub(r'^\s*import\s+charts?\b.*$', '', code, flags=re.MULTILINE)

		code = re.sub(r"result\s*=\s*pd\.pie\(df_grouped\['([^']+)'\][^\n]*labels\s*=\s*df_grouped\['([^']+)'\][^\n]*", lambda m: (
			f"result = pie_chart(df_grouped, labels='{m.group(2)}', values='{m.group(1)}', title='Pie Chart', output_file=chart_path, backend='{chart_backend}')"
		), code)
		code = re.sub(r"result\s*=\s*plt\.pie\(df_grouped\['([^']+)'\][^\n]*labels\s*=\s*df_grouped\['([^']+)'\][^\n]*", lambda m: (
			f"result = pie_chart(df_grouped, labels='{m.group(2)}', values='{m.group(1)}', title='Pie Chart', output_file=chart_path, backend='{chart_backend}')"
		), code)

		code = re.sub(
			r"result\s*=\s*histogram\(df\['([^']+)'\]",
			lambda m: f"result = histogram(df, column='{m.group(1)}'",
			code,
		)

		code = re.sub(r"output_file\s*=\s*['\"][^'\"]+['\"]", 'output_file=chart_path', code)
		code = code.replace("figure.savefig('chart_path')", 'figure.savefig(chart_path)\nresult = chart_path')
		code = code.replace('pd.pie(', 'pie_chart(')
		code = code.replace('plt.pie(', 'pie_chart(')
		code = code.replace('sns.histplot(', 'histogram(')
		code = code.replace('px.histogram(', 'histogram(')

		value_column = PyChartAgent._infer_value_column(query, df)
		label_column = PyChartAgent._infer_label_column(intent, df)

		def _fix_pie_args(match: re.Match[str]) -> str:
			args = match.group(1)
			parts = [part.strip() for part in args.split(',') if part.strip()]
			frame_arg = parts[0] if parts else 'df'
			positional = [part for part in parts[1:] if '=' not in part]
			keyword_parts = [part for part in parts[1:] if '=' in part]
			labels_value = None
			values_value = None
			other_keywords: list[str] = []

			for part in keyword_parts:
				if part.startswith('labels='):
					labels_value = part.split('=', 1)[1].strip()
				elif part.startswith('values='):
					values_value = part.split('=', 1)[1].strip()
				else:
					other_keywords.append(part)

			if positional:
				labels_value = labels_value or positional[0]
			if len(positional) > 1:
				values_value = values_value or positional[1]

			if labels_value is None and label_column:
				labels_value = repr(label_column)
			if values_value is None and value_column:
				values_value = repr(value_column)

			rebuilt = [frame_arg]
			if labels_value is not None:
				rebuilt.append(f'labels={labels_value}')
			if values_value is not None:
				rebuilt.append(f'values={values_value}')
			rebuilt.extend(other_keywords)
			return f"pie_chart({', '.join(rebuilt)})"

		code = re.sub(r'pie_chart\(([^\n]+)\)', _fix_pie_args, code)

		if intent.chart_helper:
			code = PyChartAgent._lock_chart_helper_calls(code, intent.chart_helper)
			code = PyChartAgent._ensure_chart_helper_present(
				code=code,
				intent=intent,
				chart_backend=chart_backend,
				df=df,
				query=query,
			)

		return code

	@staticmethod
	def _lock_chart_helper_calls(code: str, target_helper: str) -> str:
		"""Rewrite chart helper calls so chart intent uses only *target_helper*."""
		for helper in _CHART_HELPERS:
			if helper == target_helper:
				continue
			code = re.sub(rf'\b{helper}\s*\(', f'{target_helper}(', code)
		return code

	@staticmethod
	def _ensure_chart_helper_present(
		code: str,
		intent: QueryIntent,
		chart_backend: str,
		df: pd.DataFrame,
		query: str,
	) -> str:
		"""Inject a deterministic fallback when the expected helper call is missing."""
		target = intent.chart_helper
		if not target:
			return code
		if re.search(rf'\b{target}\s*\(', code):
			return code

		label_col = PyChartAgent._infer_label_column(intent, df)
		value_col = PyChartAgent._infer_value_column(query, df)

		# For numeric-column helpers, ensure we have a value column
		if value_col is None:
			for column in df.columns:
				if pd.api.types.is_numeric_dtype(df[column]):
					value_col = column
					break

		# For label-column helpers, ensure we have a label column
		if label_col is None:
			for column in df.columns:
				if not pd.api.types.is_numeric_dtype(df[column]):
					label_col = column
					break

		# Fallback label: use first column as last resort
		if label_col is None and len(df.columns) > 0:
			label_col = df.columns[0]

		if target == 'pie_chart':
			fallback = (
				"\n# [repair] injected fallback pie-chart generation\n"
				f"_chart_df = df.groupby('{label_col}', as_index=False)['{value_col}'].sum()\n"
				f"result = pie_chart(_chart_df, labels='{label_col}', values='{value_col}', title='Pie Chart', output_file=chart_path, backend='{chart_backend}')\n"
			)
			return code + fallback

		if target in ('histogram', 'kde_chart', 'ecdf_chart'):
			col = value_col or (df.columns[0] if len(df.columns) > 0 else 'value')
			fallback = (
				f"\n# [repair] injected fallback {target} generation\n"
				f"result = {target}(df, column='{col}', title='{target.replace('_', ' ').title()}', output_file=chart_path, backend='{chart_backend}')\n"
			)
			return code + fallback

		if target in ('heatmap', 'pairplot_chart'):
			fallback = (
				f"\n# [repair] injected fallback {target} generation\n"
				f"result = {target}(df, title='{target.replace('_', ' ').title()}', output_file=chart_path, backend='{chart_backend}')\n"
			)
			return code + fallback

		if target == 'count_chart':
			col = label_col or (df.columns[0] if len(df.columns) > 0 else 'category')
			fallback = (
				f"\n# [repair] injected fallback count_chart generation\n"
				f"result = count_chart(df, x='{col}', title='Count Chart', output_file=chart_path, backend='{chart_backend}')\n"
			)
			return code + fallback

		if target == 'funnel_chart':
			fallback = (
				f"\n# [repair] injected fallback funnel_chart generation\n"
				f"_chart_df = df.groupby('{label_col}', as_index=False)['{value_col}'].sum()\n"
				f"result = funnel_chart(_chart_df, labels='{label_col}', values='{value_col}', title='Funnel Chart', output_file=chart_path, backend='{chart_backend}')\n"
			)
			return code + fallback

		# Generic fallback for x/y chart helpers (bar, line, scatter, box, violin, area, step, strip, swarm, regression, stacked_bar, bubble).
		x_col = label_col or (df.columns[0] if len(df.columns) > 0 else 'x')
		y_col = value_col or x_col
		fallback = (
			f"\n# [repair] injected fallback chart generation\n"
			f"result = {target}(df, x='{x_col}', y='{y_col}', title='Chart', output_file=chart_path, backend='{chart_backend}')\n"
		)
		return code + fallback

	@staticmethod
	def _infer_value_column(query: str, df: pd.DataFrame) -> Optional[str]:
		"""Choose a numeric value column for chart helpers when the LLM omits one."""
		lowered = query.lower()
		preferred = []
		for column in df.columns:
			if not pd.api.types.is_numeric_dtype(df[column]):
				continue
			if column.lower() in lowered:
				preferred.append(column)
		for common in ('revenue', 'sales', 'value', 'amount', 'price', 'quantity', 'count', 'total_revenue'):
			for column in df.columns:
				if column.lower() == common and pd.api.types.is_numeric_dtype(df[column]):
					preferred.append(column)
		if preferred:
			return preferred[0]
		for column in df.columns:
			if pd.api.types.is_numeric_dtype(df[column]):
				return column
		return None

	@staticmethod
	def _infer_label_column(intent: QueryIntent, df: pd.DataFrame) -> Optional[str]:
		"""Choose a categorical label column for chart helpers when omitted."""
		if intent.group_by_hint:
			return intent.group_by_hint
		for column in df.columns:
			if not pd.api.types.is_numeric_dtype(df[column]):
				return column
		return None

	# ------------------------------------------------------------------
	# Execution context
	# ------------------------------------------------------------------

	def _build_context(self, df: pd.DataFrame, chart_backend: str) -> Dict[str, Any]:
		"""Build the variable namespace injected into the sandbox."""
		import numpy as np
		from . import visualization as _viz

		os.makedirs(self.charts_output_dir, exist_ok=True)
		ext = '.html' if chart_backend == 'plotly' else '.png'
		chart_path = os.path.join(
			self.charts_output_dir,
			f'{chart_backend}_{_dt.datetime.now().strftime("%Y%m%d_%H%M%S")}_{_uuid.uuid4().hex[:6]}{ext}',
		)
		return {
			'df': df,
			'pd': pd,
			'np': np,
			'chart_path': chart_path,
			# All 20 chart helpers (trusted — run outside restricted code)
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

	# ------------------------------------------------------------------
	# Result formatting
	# ------------------------------------------------------------------

	@staticmethod
	def _format_result(result: Any) -> str:
		"""Normalise a raw sandbox result to a string or chart file path."""
		if isinstance(result, dict) and result.get('type') == 'plot':
			return str(result.get('value', ''))
		if isinstance(result, pd.DataFrame):
			return result.to_string()
		if isinstance(result, pd.Series):
			return result.to_string()
		if result is None:
			return 'No result returned.'
		return str(result)

	@staticmethod
	def _validate_result(result: Any, query: str, intent: QueryIntent) -> None:
		"""Reject results that clearly do not satisfy the query shape."""
		if intent.kind == 'chart':
			if isinstance(result, dict) and result.get('type') == 'plot' and result.get('value'):
				return
			if isinstance(result, str) and result.endswith(('.png', '.html', '.svg')):
				if not os.path.isfile(result):
					raise ValueError(f'Chart file was not created: {result}')
				return
			raise ValueError('Chart query must return a chart file path or plot result.')

		if intent.kind == 'filter' and not isinstance(result, (pd.DataFrame, pd.Series)):
			raise ValueError('Filter query must return tabular rows, not a scalar or chart.')

		if intent.kind == 'count' and isinstance(result, (pd.DataFrame, pd.Series)):
			raise ValueError('Count query must return a scalar value, not a table.')

		if intent.needs_time_bucketing and not isinstance(result, (pd.DataFrame, pd.Series)):
			raise ValueError('Time-bucketing query must return grouped tabular output.')
		if intent.needs_time_bucketing and PyChartAgent._result_size(result) < 2:
			raise ValueError('Time-bucketing query returned too little grouped output.')

		if intent.needs_grouping and not isinstance(result, (pd.DataFrame, pd.Series)):
			raise ValueError('Grouped aggregation must return grouped tabular output.')
		if intent.needs_grouping and PyChartAgent._result_size(result) < 2:
			raise ValueError('Grouped aggregation returned too little grouped output.')

	@staticmethod
	def _result_size(result: Any) -> int:
		"""Estimate the number of rows/items in a result for validation."""
		if isinstance(result, (pd.DataFrame, pd.Series)):
			return len(result)
		return 1

	# ------------------------------------------------------------------
	# Dashboard generation
	# ------------------------------------------------------------------

	def dashboard(
		self,
		query: str,
		df,
		n_charts: int = 4,
		sandbox=None,
		chart_backend: Optional[str] = None,
	) -> List[Dict[str, str]]:
		"""Generate multiple complementary charts from a single natural-language query.

		The agent decomposes *query* into up to *n_charts* distinct chart sub-queries
		using the LLM, then executes each one independently.  Failed sub-queries are
		skipped (logged as warnings) so a partial dashboard is still returned.

		Args:
			query:         High-level description, e.g. ``'dashboard of sales data'``.
			df:            DataFrame (or list/dict of DataFrames).
			n_charts:      Maximum number of charts to generate (default 4).
			sandbox:       Optional sandbox override.
			chart_backend: Visualization backend for all charts.

		Returns:
			List of dicts, each with keys ``'title'``, ``'path'``, ``'query'``.
			Chart paths can be ``.png`` or ``.html`` depending on the backend.
		"""
		import json as _json

		n_charts = max(1, min(n_charts, 8))  # clamp to 1–8
		df_coerced, _ = self._coerce_dataframes(df)

		# Build a condensed schema string for the decomposition prompt
		cols_info = ', '.join(
			f'{c}({str(dt)[:8]})'
			for c, dt in df_coerced.dtypes.items()
		)
		decompose_prompt = (
			f'You are a data visualization assistant. '
			f'Given the user request and dataset schema, produce exactly {n_charts} '
			f'distinct chart queries that together form an informative dashboard.\n\n'
			f'User request: "{query}"\n'
			f'Dataset columns: {cols_info}\n\n'
			f'Respond with a JSON array of objects, each with "title" and "chart_query". '
			f'Each chart_query must be a complete standalone instruction like '
			f'"Plot a bar chart of revenue by region". '
			f'No markdown, no explanation — raw JSON only.'
		)

		try:
			raw = self._generate_with_timeout(decompose_prompt)
			# Strip markdown fences if the model wraps the JSON
			raw = re.sub(r'^```(?:json)?\s*', '', raw.strip(), flags=re.IGNORECASE)
			raw = re.sub(r'\s*```$', '', raw.strip())
			sub_queries: List[Dict[str, str]] = _json.loads(raw)
			if not isinstance(sub_queries, list):
				raise ValueError('LLM did not return a JSON array')
		except Exception as exc:
			_log.warning('pychartai.dashboard.decompose_failed error=%r — using fallback queries', str(exc))
			# Fallback: generic chart types applied to the first numeric columns
			num_cols = df_coerced.select_dtypes(include='number').columns.tolist()[:2]
			cat_cols = df_coerced.select_dtypes(include=['object', 'category', 'string']).columns.tolist()[:1]
			sub_queries = []
			if num_cols and cat_cols:
				sub_queries += [
					{'title': f'Bar chart of {num_cols[0]} by {cat_cols[0]}',
					 'chart_query': f'Plot a bar chart of {num_cols[0]} by {cat_cols[0]}'},
					{'title': f'Distribution of {num_cols[0]}',
					 'chart_query': f'Plot a histogram of {num_cols[0]}'},
				]
			if len(num_cols) >= 2:
				sub_queries.append({
					'title': f'{num_cols[0]} vs {num_cols[1]}',
					'chart_query': f'Scatter plot of {num_cols[0]} vs {num_cols[1]}',
				})
			sub_queries = sub_queries[:n_charts]

		results: List[Dict[str, str]] = []
		for item in sub_queries[:n_charts]:
			title = item.get('title', 'Chart')
			sub_q = item.get('chart_query', item.get('query', ''))
			if not sub_q:
				continue
			try:
				path = self.chat(
					sub_q, df,
					sandbox=sandbox,
					chart_backend=chart_backend or self.chart_backend,
				)
				results.append({'title': title, 'path': str(path), 'query': sub_q})
			except Exception as exc:
				_log.warning(
					'pychartai.dashboard.chart_failed title=%r query=%r error=%r',
					title, sub_q[:80], str(exc),
				)

		_log.info(
			'pychartai.dashboard.done requested=%d generated=%d',
			n_charts, len(results),
		)
		return results
