"""
pipeline.py — Composable processing pipeline for SmartDataFrame queries.

A :class:`Pipeline` is an ordered list of :class:`PipelineStep` objects.
Each step receives and returns a :class:`PipelineContext` (a plain ``dict``
subclass) so steps can read/write shared state freely.

Provided built-in steps (used by :meth:`~SmartDataFrame.chat` internally):

=================  =========================================================
Step               Responsibility
=================  =========================================================
ValidateInput      Assert df is non-empty and query is non-empty string.
InjectSchema       Prepend semantic layer context to the query string.
InjectSkills       Append skill descriptions to the query; store sources.
CacheLookup        Check the ResponseCache; short-circuit on a cache hit.
CallAnalyzer       Delegate to :class:`~DataAnalyzer` to call the LLM.
CacheStore         Persist the result to cache (skipped on cache hit).
=================  =========================================================

You can extend or replace steps::

    from pychartai_core.pipeline import Pipeline, PipelineStep, PipelineContext

    class LogStep(PipelineStep):
        def run(self, ctx: PipelineContext) -> PipelineContext:
            print(f'[LOG] query={ctx[\"query\"]!r}')
            return ctx

    sdf = pai.read_csv('data.csv')
    sdf.pipeline.add_step(LogStep())   # appended before execution
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, TYPE_CHECKING

if TYPE_CHECKING:
	import pandas as pd
	from .skills import Skill
	from .schema import Schema
	from .cache import ResponseCache


class PipelineContext(dict):
	"""Mutable state container passed through every pipeline step.

	Standard keys:

	- ``'df'``             : pandas DataFrame
	- ``'query'``          : current (possibly enriched) query string
	- ``'original_query'`` : unmodified user query string
	- ``'result'``         : output of the LLM/analysis step
	- ``'skills'``         : list[Skill]
	- ``'skill_sources'``  : dict[name, source_code] – populated by InjectSkills
	- ``'schema'``         : Schema | None
	- ``'cache'``          : ResponseCache | None
	- ``'cache_hit'``      : bool
	- ``'backend'``        : chart backend name
	- ``'config'``         : global config dict (read-only reference)
	"""


class PipelineStep(ABC):
	"""Abstract base class for a single processing step.

	Subclass and implement :meth:`run`.  Return the (possibly mutated)
	context; the return value replaces the context for subsequent steps.
	"""

	enabled: bool = True

	@abstractmethod
	def run(self, ctx: PipelineContext) -> PipelineContext:
		"""Execute this step and return the updated context."""

	def skip(self) -> 'PipelineStep':
		"""Disable this step at runtime."""
		self.enabled = False
		return self

	def enable(self) -> 'PipelineStep':
		"""Re-enable this step."""
		self.enabled = True
		return self


# ---------------------------------------------------------------------------
# Built-in steps
# ---------------------------------------------------------------------------

class ValidateInput(PipelineStep):
	"""Raise ValueError for obviously invalid inputs."""

	def run(self, ctx: PipelineContext) -> PipelineContext:
		df = ctx.get('df')
		query = ctx.get('query', '')
		if df is None or (hasattr(df, 'empty') and df.empty):
			raise ValueError('Pipeline received an empty DataFrame.')
		if not isinstance(query, str) or not query.strip():
			raise ValueError('Pipeline received an empty query string.')
		ctx['original_query'] = ctx.get('original_query', query)
		return ctx


class InjectSchema(PipelineStep):
	"""Prepend semantic layer metadata to the query."""

	def run(self, ctx: PipelineContext) -> PipelineContext:
		schema = ctx.get('schema')
		if schema is not None and hasattr(schema, 'to_prompt_fragment'):
			fragment = schema.to_prompt_fragment()
			if fragment:
				ctx['query'] = fragment + '\n\n' + ctx['query']
		return ctx


class InjectSkills(PipelineStep):
	"""Append skill descriptions to the query and store source mappings."""

	def run(self, ctx: PipelineContext) -> PipelineContext:
		from .skills import build_skills_prompt
		skills = ctx.get('skills') or []
		if skills:
			snippet = build_skills_prompt(skills)
			if snippet:
				ctx['query'] = ctx['query'] + '\n\n' + snippet
			ctx['skill_sources'] = {s.name: s.to_source() for s in skills}
		return ctx


class CacheLookup(PipelineStep):
	"""Return a cached result immediately if available."""

	def run(self, ctx: PipelineContext) -> PipelineContext:
		cache = ctx.get('cache')
		if cache is None:
			return ctx
		from .cache import ResponseCache
		fp = self._extended_fingerprint(ctx)
		cached = cache.get(ctx.get('original_query', ctx['query']), fp)
		if cached is not None:
			ctx['result'] = cached
			ctx['cache_hit'] = True
		return ctx

	@staticmethod
	def _extended_fingerprint(ctx: PipelineContext) -> str:
		"""Build a fingerprint that includes df shape, schema, and backend."""
		from .cache import ResponseCache
		fp = ResponseCache.fingerprint(ctx['df'])
		backend = ctx.get('backend', '')
		schema = ctx.get('schema')
		schema_id = ''
		if schema is not None and hasattr(schema, 'name'):
			schema_id = str(schema.name or '')
		return f'{fp}|{backend}|{schema_id}'


class CallAnalyzer(PipelineStep):
	"""Delegate to the DataAnalyzer (LLM backend)."""

	def run(self, ctx: PipelineContext) -> PipelineContext:
		if ctx.get('cache_hit'):
			return ctx  # already resolved

		from .analyzer import DataAnalyzer
		cfg = ctx.get('config', {})
		analyzer = DataAnalyzer(
			memory_size=cfg.get('memory_size', 100),
			verbose=cfg.get('verbose', False),
			charts_output_dir=cfg.get('charts_output_dir', 'exports/charts'),
			chart_backend=ctx.get('backend', 'seaborn'),
			llm=cfg.get('llm'),
			skills=ctx.get('skills') or [],
		)
		ctx['result'] = analyzer.analyze(ctx['df'], ctx['query'], sandbox=ctx.get('sandbox'))
		return ctx


class CacheStore(PipelineStep):
	"""Persist the result to cache after a successful LLM call."""

	def run(self, ctx: PipelineContext) -> PipelineContext:
		cache = ctx.get('cache')
		if cache is None or ctx.get('cache_hit') or 'result' not in ctx:
			return ctx
		fp = CacheLookup._extended_fingerprint(ctx)
		query_key = ctx.get('original_query') or ctx.get('query', '')
		cache.put(query_key, fp, str(ctx['result']))
		return ctx


class CallOwnAgent(PipelineStep):
	"""Delegate to PyChartAgent (pandasai-independent path).

	Used by :meth:`SmartDataFrame._chat_with_own_agent` so that custom
	pipeline steps added by the user still fire even on the default
	(non-pandasai) execution path.
	"""

	def __init__(self, llm, chart_backend: str, charts_output_dir: str,
	             verbose: bool, memory=None, agent_input=None,
	             sandbox=None, explain: bool = False, on_progress=None) -> None:
		self._llm = llm
		self._chart_backend = chart_backend
		self._charts_output_dir = charts_output_dir
		self._verbose = verbose
		self._memory = memory
		self._agent_input = agent_input  # None → use ctx['df']; set for redacted/multi-df input
		self._sandbox = sandbox
		self._explain = explain
		self._on_progress = on_progress

	def run(self, ctx: PipelineContext) -> PipelineContext:
		if ctx.get('cache_hit'):
			return ctx
		from .agent import PyChartAgent
		agent = PyChartAgent(
			llm=self._llm,
			chart_backend=self._chart_backend,
			charts_output_dir=self._charts_output_dir,
			verbose=self._verbose,
			memory=self._memory,
		)
		df_for_agent = self._agent_input if self._agent_input is not None else ctx['df']
		ctx['result'] = agent.chat(
			ctx['query'],
			df_for_agent,
			sandbox=self._sandbox,
			explain=self._explain,
			on_progress=self._on_progress,
		)
		return ctx


# ---------------------------------------------------------------------------
# Pipeline orchestrator
# ---------------------------------------------------------------------------

class Pipeline:
	"""Ordered chain of :class:`PipelineStep` objects.

	Args:
		steps: Initial list of steps.  Steps marked ``enabled=False`` are
		       skipped during execution.

	Example::

		from pychartai_core.pipeline import Pipeline, ValidateInput, CallAnalyzer
		pipeline = Pipeline([ValidateInput(), CallAnalyzer()])
		ctx = pipeline.run({'df': df, 'query': 'What is the average revenue?'})
		print(ctx['result'])
	"""

	def __init__(self, steps: List[PipelineStep]) -> None:
		self._steps: List[PipelineStep] = list(steps)

	@property
	def steps(self) -> List[PipelineStep]:
		return self._steps

	def run(self, initial: dict) -> PipelineContext:
		"""Execute all enabled steps in order and return the final context."""
		ctx = PipelineContext(initial)
		for step in self._steps:
			if step.enabled:
				step_name = type(step).__name__
				try:
					result = step.run(ctx)
					if result is not None:
						ctx = result
				except (ValueError, TypeError) as exc:
					raise type(exc)(f'[Pipeline/{step_name}] {exc}') from exc
				except Exception as exc:
					raise RuntimeError(f'[Pipeline/{step_name}] {exc}') from exc
		return ctx

	def add_step(self, step: PipelineStep, *, index: int = -1, before: str = '') -> 'Pipeline':
		"""Append (or insert) a step.

		Args:
			step:   The pipeline step to add.
			index:  Integer position to insert at (negative = append).
			before: Name of an existing step class to insert before.
			        Takes precedence over *index* when provided.
		"""
		if before:
			idx = next(
				(i for i, s in enumerate(self._steps) if type(s).__name__ == before),
				-1,
			)
			if idx < 0:
				raise ValueError(f"No step named {before!r} found in pipeline")
			self._steps.insert(idx, step)
		elif index < 0:
			self._steps.append(step)
		else:
			self._steps.insert(index, step)
		return self

	def remove_step(self, step_type: type) -> 'Pipeline':
		"""Remove all steps of a given type."""
		self._steps = [s for s in self._steps if not isinstance(s, step_type)]
		return self

	def __len__(self) -> int:
		return len(self._steps)

	def __repr__(self) -> str:
		names = [type(s).__name__ for s in self._steps]
		return f'Pipeline([{", ".join(names)}])'


def default_pipeline() -> Pipeline:
	"""Return the standard pipeline used by :meth:`SmartDataFrame.chat`.

	Steps in order:

	1. :class:`ValidateInput`
	2. :class:`InjectSchema`
	3. :class:`InjectSkills`
	4. :class:`CacheLookup`
	5. :class:`CallAnalyzer`
	6. :class:`CacheStore`
	"""
	return Pipeline([
		ValidateInput(),
		InjectSchema(),
		InjectSkills(),
		CacheLookup(),
		CallAnalyzer(),
		CacheStore(),
	])
