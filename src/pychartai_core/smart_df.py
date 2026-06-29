"""
smart_df.py
"""
from __future__ import annotations
from typing import Any, List, Optional
import pandas as pd


class SmartDataFrame:
	"""Transparent pd.DataFrame wrapper with .chat(), skills, schema, pipeline.

	Args:
		df:            A :class:`pandas.DataFrame` to wrap.
		config:        Optional dict of pychartai config overrides applied to
		               the global :attr:`pychartai.config` singleton.
		chart_library: Default chart backend for this instance
		               ('seaborn', 'matplotlib', 'plotly').
	"""

	def __init__(
		self,
		df: pd.DataFrame,
		config: Optional[dict] = None,
		chart_library: Optional[str] = None,
		schema=None,
		redactor=None,
	) -> None:
		if not isinstance(df, pd.DataFrame):
			raise TypeError(f'SmartDataFrame requires a pd.DataFrame, got {type(df).__name__}')
		object.__setattr__(self, '_df', df)
		object.__setattr__(self, '_skills', [])
		object.__setattr__(self, '_schema', schema)
		object.__setattr__(self, '_pipeline', None)
		object.__setattr__(self, '_memory', None)
		object.__setattr__(self, '_redactor', redactor)

		# Apply optional config overrides eagerly so they take effect immediately
		if config or chart_library:
			from .config import config as _global_config
			merged: dict = dict(config) if config else {}
			if chart_library:
				merged.setdefault('chart_backend', chart_library)
			# Normalise legacy key name
			if 'chart_library' in merged and 'chart_backend' not in merged:
				merged['chart_backend'] = merged.pop('chart_library')
			elif 'chart_library' in merged:
				del merged['chart_library']
			_global_config.set(merged)

	def __getattr__(self, name: str) -> Any:
		return getattr(object.__getattribute__(self, '_df'), name)

	def __setattr__(self, name: str, value: Any) -> None:
		# Check class-level property descriptors before delegating
		for cls in type(self).__mro__:
			if name in cls.__dict__:
				desc = cls.__dict__[name]
				if isinstance(desc, property) and desc.fset is not None:
					desc.fset(self, value)
					return
				break
		if name in ('_df', '_skills', '_schema', '_pipeline', '_memory', '_redactor'):
			object.__setattr__(self, name, value)
		else:
			object.__getattribute__(self, '_df').__setattr__(name, value)

	def __getitem__(self, key: Any) -> Any:
		return object.__getattribute__(self, '_df')[key]

	def __setitem__(self, key: Any, value: Any) -> None:
		object.__getattribute__(self, '_df')[key] = value

	def __len__(self) -> int:
		return len(object.__getattribute__(self, '_df'))

	def __contains__(self, item: Any) -> bool:
		return item in object.__getattribute__(self, '_df')

	def __iter__(self):
		return iter(object.__getattribute__(self, '_df'))

	def __repr__(self) -> str:
		df = object.__getattribute__(self, '_df')
		return f'SmartDataFrame({df.shape[0]}x{df.shape[1]})\n{repr(df)}'

	@staticmethod
	def _build_chart_directive(plot_type, chart_options, chart_kwargs):
		parts = []
		if plot_type:
			parts.append(f'plot type: {plot_type}')
		merged = {}
		if chart_options:
			merged.update(chart_options)
		if chart_kwargs:
			merged.update(chart_kwargs)
		if merged:
			parts.append('chart options: ' + ', '.join(f'{k}={repr(v)}' for k, v in merged.items()))
		if not parts:
			return ''
		return '\n\nChart directives: ' + '; '.join(parts)

	def chat(self, query, chart_type=None, *, chart_library=None, plot_type=None,
	         chart_options=None, sandbox=None, use_agent=False,
	         agent=None, explain=False, extra_dfs=None, on_progress=None, **chart_kwargs):
		"""Ask a natural-language question about the DataFrame.

		Execution modes via the *agent* parameter:

		* ``agent=None`` (default): :class:`~pychartai_core.agent.PyChartAgent` —
		                            pandasai-independent code generation + sandbox.
		* ``agent='own'``:          Explicit alias for the default own-agent mode.
		* ``agent='pandasai'``:     ``pandasai.Agent`` orchestration (requires pandasai).
		  ``use_agent=True`` is a backward-compatible alias for ``agent='pandasai'``.
		* ``agent='sandbox'``:      Legacy direct-sandbox path (no PyChartAgent).

		Args:
		    query:        Natural-language question or instruction.
		    chart_type:   Legacy alias for *chart_library*.
		    chart_library: Visualization backend ('seaborn', 'plotly', 'matplotlib').
		    plot_type:    Hint to guide the chart style.
		    chart_options: Additional chart keyword options.
		    sandbox:      Optional :class:`RestrictedSandbox` or :class:`DockerSandbox`.
		    use_agent:    Deprecated alias for ``agent='pandasai'``.
		    agent:        Execution backend — ``None``, ``'own'``, ``'sandbox'``, or ``'pandasai'``.
		    explain:      If True, append an LLM-generated plain-English explanation.
		    extra_dfs:    Additional DataFrames to include in the analysis.  Can be a
		                  ``list`` of DataFrames or a ``dict`` mapping names to DataFrames.
		                  Only supported with ``agent=None`` / ``'own'``.
		    **chart_kwargs: Additional chart parameters.
		"""
		from .config import config as global_config
		llm = global_config.get('llm')
		if llm is None:
			raise RuntimeError(
				'No LLM configured. '
				"Call pai.config.set({'llm': pai.OllamaLLM(model='llama3.2')}) first."
			)

		_VALID_AGENTS = (None, 'own', 'sandbox', 'pandasai')
		if agent not in _VALID_AGENTS:
			raise ValueError(f"agent must be one of {_VALID_AGENTS}, got {agent!r}")

		# Backward-compat: use_agent=True → agent='pandasai'
		if use_agent and agent is None:
			agent = 'pandasai'

		if agent == 'pandasai' and sandbox is not None:
			raise ValueError("Cannot combine agent='pandasai' with sandbox parameter.")

		if agent == 'pandasai':
			return self._chat_with_agent(query, chart_type, chart_library, plot_type, chart_options, **chart_kwargs)

		# Legacy direct-sandbox path (no LLM prompt engineering)
		if agent == 'sandbox':
			if sandbox is None:
				from .sandbox import RestrictedSandbox
				sandbox = RestrictedSandbox()
			return self._chat_with_sandbox(query, chart_type, chart_library, plot_type, chart_options, sandbox, **chart_kwargs)

		# Default (agent=None or agent='own'): PyChartAgent
		return self._chat_with_own_agent(query, chart_type, chart_library, plot_type, chart_options, sandbox, explain=explain, extra_dfs=extra_dfs, on_progress=on_progress, **chart_kwargs)
	
	def _chat_with_agent(self, query, chart_type=None, chart_library=None, plot_type=None,
	                       chart_options=None, **chart_kwargs):
		"""Execute query using pandasai.Agent orchestration."""
		from .config import config as global_config
		
		effective_backend = chart_library or chart_type or global_config.get('chart_backend', 'seaborn')
		directive = self._build_chart_directive(plot_type, chart_options, chart_kwargs)
		effective_query = query + directive
		df = object.__getattribute__(self, '_df')
		skills = object.__getattribute__(self, '_skills')
		schema = object.__getattribute__(self, '_schema')
		raw_cache = global_config.get('cache')
		if raw_cache is True:
			from .cache import ResponseCache
			raw_cache = ResponseCache()
		elif not raw_cache:
			raw_cache = None
		ctx_init = {
			'df': df,
			'query': effective_query,
			'original_query': effective_query,
			'skills': list(skills),
			'schema': schema,
			'backend': effective_backend,
			'cache': raw_cache,
			'cache_hit': False,
			'sandbox': None,
			'config': {
				'llm': global_config.get('llm'),
				'memory_size': global_config.get('memory_size', 100),
				'verbose': global_config.get('verbose', False),
				'charts_output_dir': global_config.get('charts_output_dir', 'exports/charts'),
			},
		}
		ctx = self.pipeline.run(ctx_init)
		result = ctx.get('result', '')
		import os, shutil, time, uuid as _uuid
		_is_temp = isinstance(result, str) and (
			'temp_chart_' in result or os.path.basename(result).startswith('chart')
		)
		if _is_temp and os.path.isfile(result):
			import logging as _logging
			_log = _logging.getLogger('pychartai')
			try:
				base_dir = global_config.get('charts_output_dir', 'exports/charts')
				os.makedirs(base_dir, exist_ok=True)
				ext = '.html' if effective_backend == 'plotly' else '.png'
				new_path = os.path.join(
					base_dir,
					f'{effective_backend}_{time.strftime("%Y%m%d_%H%M%S")}_{_uuid.uuid4().hex[:6]}{ext}'
				)
				shutil.move(result, new_path)
				result = new_path
			except OSError as _e:
				_log.warning('Could not move temp chart to output dir: %s', _e)
		try:
			base_dir = global_config.get('charts_output_dir', 'exports/charts')
			if os.path.isdir(base_dir):
				for item in os.listdir(base_dir):
					item_path = os.path.join(base_dir, item)
					if os.path.isdir(item_path):
						shutil.rmtree(item_path)
		except OSError as _e:
			import logging as _logging
			_logging.getLogger('pychartai').debug('Chart dir cleanup warning: %s', _e)
		return result
	
	def _chat_with_sandbox(self, query, chart_type=None, chart_library=None, plot_type=None,
	                        chart_options=None, sandbox=None, **chart_kwargs):
		"""Execute query using sandbox (pandasai-independent), routed through the pipeline."""
		from .config import config as global_config

		effective_backend = chart_library or chart_type or global_config.get('chart_backend', 'seaborn')
		directive = self._build_chart_directive(plot_type, chart_options, chart_kwargs)
		effective_query = query + directive
		df = object.__getattribute__(self, '_df')
		skills = object.__getattribute__(self, '_skills')
		schema = object.__getattribute__(self, '_schema')

		raw_cache = global_config.get('cache')
		if raw_cache is True:
			from .cache import ResponseCache
			raw_cache = ResponseCache()
		elif not raw_cache:
			raw_cache = None

		ctx_init = {
			'df': df,
			'query': effective_query,
			'original_query': effective_query,
			'skills': list(skills),
			'schema': schema,
			'backend': effective_backend,
			'cache': raw_cache,
			'cache_hit': False,
			'sandbox': sandbox,
			'config': {
				'llm': global_config.get('llm'),
				'memory_size': global_config.get('memory_size', 100),
				'verbose': global_config.get('verbose', False),
				'charts_output_dir': global_config.get('charts_output_dir', 'exports/charts'),
			},
		}
		ctx = self.pipeline.run(ctx_init)
		return ctx.get('result', '')

	def _chat_with_own_agent(self, query, chart_type=None, chart_library=None, plot_type=None,
	                          chart_options=None, sandbox=None, explain=False, extra_dfs=None,
	                          on_progress=None, **chart_kwargs):
		"""Execute query using PyChartAgent (pandasai-independent, switchable).

		Routes through ``self.pipeline`` so user-added custom steps fire.
		``CallAnalyzer`` is replaced with ``CallOwnAgent`` which wraps
		``PyChartAgent``; all other steps (cache, schema, skills, custom) run
		unchanged.
		"""
		from .config import config as global_config
		from .pipeline import CallAnalyzer, CallOwnAgent, Pipeline

		effective_backend = chart_library or chart_type or global_config.get('chart_backend', 'seaborn')
		directive = self._build_chart_directive(plot_type, chart_options, chart_kwargs)
		effective_query = query + directive
		df = object.__getattribute__(self, '_df')
		memory = object.__getattribute__(self, '_memory')
		skills = object.__getattribute__(self, '_skills')
		schema = object.__getattribute__(self, '_schema')

		# Determine the df that PyChartAgent will actually receive.
		# extra_dfs → multi-df input; PII redaction → redacted copy.
		# We always keep the original `df` in ctx so the pipeline steps
		# (CacheLookup, ValidateInput, InjectSchema) see a plain DataFrame.
		agent_input = None  # None → CallOwnAgent will use ctx['df']
		if extra_dfs is not None:
			if isinstance(extra_dfs, dict):
				agent_input = {'primary': df, **extra_dfs}
			elif isinstance(extra_dfs, list):
				agent_input = [df] + list(extra_dfs)
			else:
				agent_input = [df, extra_dfs]

		redactor = object.__getattribute__(self, '_redactor')
		if redactor is None:
			from .config import config as _cfg
			redactor = _cfg.get('redactor')
		if redactor is not None:
			source = df if agent_input is None or not isinstance(agent_input, pd.DataFrame) else agent_input
			agent_input = redactor.redact(source)

		raw_cache = global_config.get('cache')
		if raw_cache is True:
			from .cache import ResponseCache
			raw_cache = ResponseCache()
		elif not raw_cache:
			raw_cache = None

		call_step = CallOwnAgent(
			llm=global_config.get('llm'),
			chart_backend=effective_backend,
			charts_output_dir=global_config.get('charts_output_dir', 'exports/charts'),
			verbose=global_config.get('verbose', False),
			memory=memory,
			agent_input=agent_input,
			sandbox=sandbox,
			explain=explain,
			on_progress=on_progress,
		)

		# Build a run-time pipeline: keep all user-configured steps but
		# replace CallAnalyzer with CallOwnAgent so the own-agent path is used.
		steps = [call_step if isinstance(s, CallAnalyzer) else s for s in self.pipeline._steps]
		if not any(isinstance(s, CallOwnAgent) for s in steps):
			steps.append(call_step)
		run_pipeline = Pipeline(steps)

		ctx_init = {
			'df': df,
			'query': effective_query,
			'original_query': effective_query,
			'skills': list(skills),
			'schema': schema,
			'backend': effective_backend,
			'cache': raw_cache,
			'cache_hit': False,
			'sandbox': sandbox,
			'config': {
				'llm': global_config.get('llm'),
				'memory_size': global_config.get('memory_size', 100),
				'verbose': global_config.get('verbose', False),
				'charts_output_dir': global_config.get('charts_output_dir', 'exports/charts'),
			},
		}
		ctx = run_pipeline.run(ctx_init)
		return ctx.get('result', '')

	def dashboard(
		self,
		query: str,
		n_charts: int = 4,
		*,
		sandbox=None,
		chart_library: Optional[str] = None,
	) -> List[dict]:
		"""Generate multiple complementary charts from a single query.

		Args:
			query:         High-level description, e.g. ``'dashboard of sales data'``.
			n_charts:      Number of charts to generate (default 4, max 8).
			sandbox:       Optional sandbox override.
			chart_library: Visualization backend for all charts.

		Returns:
			List of dicts with keys ``'title'``, ``'path'``, ``'query'``.
		"""
		from .config import config as global_config
		from .agent import PyChartAgent

		df = object.__getattribute__(self, '_df')
		effective_backend = chart_library or global_config.get('chart_backend', 'seaborn')

		own_agent = PyChartAgent(
			llm=global_config.get('llm'),
			chart_backend=effective_backend,
			charts_output_dir=global_config.get('charts_output_dir', 'exports/charts'),
			verbose=global_config.get('verbose', False),
		)
		return own_agent.dashboard(query, df, n_charts=n_charts, sandbox=sandbox,
		                           chart_backend=effective_backend)

	def chat_stream(self, query: str, *, sandbox=None, chart_library=None,
	               plot_type=None, chart_options=None, **chart_kwargs):
		"""Stream analysis tokens and then yield the final result.

		Yields :class:`~pychartai_core.streaming.StreamEvent` objects:

		- ``type='token'`` — a raw LLM token emitted during code generation.
		- ``type='result'`` — the final analysis answer (last event).
		- ``type='error'`` — if an error occurred.

		This method always uses the sandbox execution path (direct code
		generation + execution) instead of the pandasai Agent pipeline.  A
		:class:`~pychartai_core.sandbox.RestrictedSandbox` is created
		automatically when *sandbox* is ``None``.

		Requires a streaming-capable LLM provider (all built-in providers
		support streaming).

		Example::

		    for event in df.chat_stream('What is the average revenue by region?'):
		        if event.type == 'token':
		            print(event.text, end='', flush=True)
		        elif event.type == 'result':
		            print('\\nAnswer:', event.value)
		"""
		from .streaming import StreamEvent
		from .config import config as global_config
		from .analyzer import DataAnalyzer, CustomLLM

		llm = global_config.get('llm')
		if llm is None:
			yield StreamEvent(type='error', error='No LLM configured.')
			return

		effective_backend = chart_library or global_config.get('chart_backend', 'seaborn')
		charts_output_dir = global_config.get('charts_output_dir', 'exports/charts')
		directive = self._build_chart_directive(plot_type, chart_options, chart_kwargs)
		effective_query = query + directive
		df = object.__getattribute__(self, '_df')

		# Resolve the underlying provider
		from .providers import PyChartLLM as _PyChartLLM
		if not isinstance(llm, _PyChartLLM):
			yield StreamEvent(
				type='error',
				error='chat_stream() requires a PyChartLLM instance (OllamaLLM, OpenAILLM, etc.). '
				      'PandasAILLM pass-through does not support streaming.',
			)
			return

		provider = llm._get_provider()

		# Prepare sandbox
		if sandbox is None:
			from .sandbox import RestrictedSandbox
			sandbox = RestrictedSandbox()

		# Build a temporary DataAnalyzer to reuse prompt/context helpers
		analyzer = DataAnalyzer(
			charts_output_dir=charts_output_dir,
			chart_backend=effective_backend,
			llm=llm,
		)
		from .sandbox import DockerSandbox
		is_docker = isinstance(sandbox, DockerSandbox)
		prompt = analyzer._build_sandbox_prompt(
			df, effective_query, effective_backend, charts_output_dir,
			allow_charts=not is_docker,
		)

		# Stream code-generation tokens
		tokens: list = []
		try:
			for token in provider.generate_stream(prompt):
				tokens.append(token)
				yield StreamEvent(type='token', text=token)
		except Exception as exc:
			yield StreamEvent(type='error', error=f'Streaming error: {exc}')
			return

		# Execute collected code in sandbox
		try:
			full_response = ''.join(tokens)
			code = analyzer.llm._extract_code_robust(full_response)
			code = CustomLLM._sanitize_code(code, table_name=None)
			if is_docker:
				raw_result = sandbox.execute(code, {'df': df})
			else:
				context = analyzer._build_sandbox_context(df, effective_backend, charts_output_dir)
				raw_result = sandbox.execute(code, context)
			yield StreamEvent(type='result', value=analyzer._format_sandbox_result(raw_result))
		except Exception as exc:
			yield StreamEvent(type='error', error=f'Execution error: {exc}')

	async def achat(
		self,
		query: str,
		chart_library=None,
		*,
		sandbox=None,
		plot_type=None,
		chart_options=None,
		**chart_kwargs,
	):
		"""Async version of :meth:`chat`.  Runs the synchronous chat in a
		thread-pool executor so it can be awaited without blocking the event
		loop.

		Example::

		    result = await df.achat('What is the average revenue?')
		"""
		import asyncio
		loop = asyncio.get_event_loop()
		return await loop.run_in_executor(
			None,
			lambda: self.chat(
				query,
				chart_library=chart_library,
				sandbox=sandbox,
				plot_type=plot_type,
				chart_options=chart_options,
				**chart_kwargs,
			),
		)

	def add_skill(self, skill_or_func):
		"""Register a skill (Skill instance or callable). Returns self."""
		from .skills import Skill, skill as make_skill
		if isinstance(skill_or_func, Skill):
			entry = skill_or_func
		elif callable(skill_or_func):
			entry = make_skill(skill_or_func)
		else:
			raise TypeError(
				f'add_skill expects callable or Skill, got {type(skill_or_func).__name__}'
			)
		object.__getattribute__(self, '_skills').append(entry)
		return self

	def remove_skill(self, name: str):
		"""Remove a skill by name."""
		skills = object.__getattribute__(self, '_skills')
		object.__setattr__(self, '_skills', [s for s in skills if s.name != name])
		return self

	@property
	def skills(self) -> list:
		"""Registered skills list (copy)."""
		return list(object.__getattribute__(self, '_skills'))

	def set_schema(self, schema):
		"""Attach a Schema for LLM context. Returns self."""
		object.__setattr__(self, '_schema', schema)
		return self

	@property
	def schema(self):
		"""Attached schema or None."""
		return object.__getattribute__(self, '_schema')

	@property
	def pipeline(self):
		"""Pipeline for this dataframe (created lazily)."""
		p = object.__getattribute__(self, '_pipeline')
		if p is None:
			from .pipeline import default_pipeline
			p = default_pipeline()
			object.__setattr__(self, '_pipeline', p)
		return p

	@pipeline.setter
	def pipeline(self, value):
		object.__setattr__(self, '_pipeline', value)

	@property
	def memory(self):
		"""Conversation memory (or None if disabled)."""
		return object.__getattribute__(self, '_memory')

	def enable_memory(self, window_size: int = 10):
		"""Enable conversation memory with the given window size. Returns self."""
		from .memory import ConversationMemory
		object.__setattr__(self, '_memory', ConversationMemory(window_size=window_size))
		return self

	def disable_memory(self):
		"""Disable conversation memory. Returns self."""
		object.__setattr__(self, '_memory', None)
		return self

	def profile(self):
		"""Generate an auto-EDA profile report (no LLM call).

		Returns:
			:class:`~pychartai_core.profiler.ProfileReport` with summary stats,
			missing values, correlations, and more.
		"""
		from .profiler import DataProfiler
		return DataProfiler.profile(object.__getattribute__(self, '_df'))

	def report(
		self,
		output_file: str = 'report.html',
		*,
		llm_narrative: bool = True,
	) -> str:
		"""Generate a self-contained HTML insight report for this DataFrame.

		The report includes overview metrics, numeric statistics, and up to five
		automatically selected charts (top-category bar, time-series, distributions,
		correlation heatmap).  When an LLM is configured and *llm_narrative* is
		True, a three-sentence narrative is generated for each section.

		Args:
		    output_file:    Destination path for the HTML file (default
		                    ``'report.html'``).
		    llm_narrative:  If True and an LLM is configured, append auto-generated
		                    narrative text to each chart section.

		Returns:
		    The absolute path to the generated HTML file.

		Example::

		    import pandas as pd
		    import pychartai as pai

		    df = pd.read_csv('data/sales.csv')
		    sdf = pai.SmartDataFrame(df)
		    path = sdf.report('reports/sales_insights.html')
		    print('Report written to', path)
		"""
		from .reporter import InsightReporter
		return InsightReporter(self).generate(output_file, llm_narrative=llm_narrative)

	@property
	def dataframe(self):
		"""The underlying pd.DataFrame."""
		return object.__getattribute__(self, '_df')

	def _repr_html_(self) -> str:
		"""Rich HTML rendering for Jupyter notebooks."""
		df = object.__getattribute__(self, '_df')
		rows, cols = df.shape
		header = (
			f'<div style="margin-bottom:8px">'
			f'<strong>SmartDataFrame</strong> — {rows:,} rows × {cols} columns'
			f'</div>'
		)
		return header + df._repr_html_()
