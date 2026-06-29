"""
skills.py — User-defined callable skills that the LLM can invoke.

Skills are ordinary Python functions decorated with ``@skill`` (or wrapped in a
``Skill`` object).  They are:

1. **Described** in the LLM prompt so the model knows they exist and what they do.
2. **Source-injected** as a preamble into the generated code so the function is
   actually in-scope at execution time.

Usage::

    import pychartai as pai

    @pai.skill
    def top_products(df, n: int = 5):
        '''Return the top-N products by revenue.'''
        return df.nlargest(n, 'revenue')

    sdf = pai.read_csv('sales.csv')
    sdf.add_skill(top_products)
    sdf.chat('Show the top 3 products')

The LLM will see the function signature and docstring, and the generated code
can call ``top_products(df, n=3)`` safely.
"""

from __future__ import annotations

import inspect
import textwrap
from dataclasses import dataclass, field
from typing import Callable, List, Optional


@dataclass
class Skill:
	"""A named, documented Python callable registered for LLM use.

	Attributes:
		func:        The underlying Python callable.
		name:        Override name (defaults to ``func.__name__``).
		description: Override description (defaults to the docstring).
	"""

	func: Callable
	name: str = field(default='')
	description: str = field(default='')

	def __post_init__(self) -> None:
		if not self.name:
			self.name = self.func.__name__
		if not self.description:
			self.description = (inspect.getdoc(self.func) or f'Function {self.name}').strip()

	# ------------------------------------------------------------------
	# Source & prompt helpers
	# ------------------------------------------------------------------

	def to_source(self) -> str:
		"""Return the dedented source code of the wrapped function, without decorators."""
		import re as _re
		try:
			src = inspect.getsource(self.func)
			src = textwrap.dedent(src)
			# Strip decorator lines (lines starting with @) that appear before `def`
			lines = src.splitlines(keepends=True)
			stripped = []
			in_decorators = True
			for line in lines:
				if in_decorators and _re.match(r'\s*@', line):
					continue
				in_decorators = False
				stripped.append(line)
			return ''.join(stripped)
		except (OSError, TypeError):
			return f'# skill source unavailable: {self.name}'

	def to_prompt_fragment(self) -> str:
		"""One-line description suitable for LLM prompt injection."""
		try:
			sig = str(inspect.signature(self.func))
		except (ValueError, TypeError):
			sig = '(...)'
		first_line = self.description.splitlines()[0] if self.description else ''
		return f'  - {self.name}{sig}: {first_line}'

	@property
	def __name__(self) -> str:
		return self.name

	def __call__(self, *args, **kwargs):
		return self.func(*args, **kwargs)


def skill(
	func: Optional[Callable] = None,
	*,
	name: str = '',
	description: str = '',
):
	"""Decorator / factory that wraps a callable as a :class:`Skill`.

	Usages::

		@skill
		def my_func(df): ...

		@skill(name='custom_name', description='Does X')
		def my_func(df): ...

		wrapped = skill(my_func)
	"""
	if func is not None:
		# Called as @skill without arguments: skill(func)
		return Skill(func=func, name=name, description=description)

	# Called as @skill(...) factory
	def decorator(f: Callable) -> Skill:
		return Skill(func=f, name=name, description=description)

	return decorator


# ---------------------------------------------------------------------------
# Prompt / code helpers used by CustomLLM
# ---------------------------------------------------------------------------

def build_skills_prompt(skills: List[Skill]) -> str:
	"""Return the block injected before the SQL directive in the LLM prompt.

	Example output::

		# IMPORTANT — Pre-defined skill functions (call these instead of writing SQL):
		  - top_products(df, n=5): Return the top-N products by revenue.
		  - flag_outliers(df, col, sigma=3): Flag rows where col exceeds sigma stddevs.
		# INSTRUCTION: Call these Python functions directly in your generated code.
		# Example: result = top_products(df, n=3)
	"""
	if not skills:
		return ''
	lines = ['# IMPORTANT — Pre-defined skill functions (call these instead of writing custom SQL):']
	for s in skills:
		lines.append(s.to_prompt_fragment())
	lines.append('# INSTRUCTION: Call these Python functions directly in your generated code.')
	# Show a concrete call example using the first skill
	ex = skills[0]
	try:
		import inspect
		sig = inspect.signature(ex.func)
		params = list(sig.parameters.values())
		# Build a minimal example call string
		example_args = []
		for p in params:
			if p.name == 'df':
				example_args.append('df')
			elif p.default is not inspect.Parameter.empty:
				example_args.append(f'{p.name}={p.default!r}')
			else:
				example_args.append(p.name)
		example_call = f'{ex.name}({", ".join(example_args[:3])})'
	except Exception:
		example_call = f'{ex.name}(df)'
	lines.append(f'# Example: result = {example_call}')
	return '\n'.join(lines)


def build_skills_preamble(skills: List[Skill]) -> str:
	"""Return Python source code to prepend to LLM-generated code.

	Ensures every skill function is defined in-scope at execution time.
	"""
	if not skills:
		return ''
	blocks = ['# --- pychartai injected skills ---']
	for s in skills:
		blocks.append(s.to_source())
	blocks.append('# --- end skills ---')
	return '\n'.join(blocks) + '\n'
