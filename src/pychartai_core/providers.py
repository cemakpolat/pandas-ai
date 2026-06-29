"""
providers.py — LLM wrapper classes for the pychartai public API.

Design (SOLID)
--------------
S — PyChartLLM configures a litellm model string and resolves its API key.
    Named subclasses (OllamaLLM etc.) set per-provider defaults only.
O — New providers are subclasses; PyChartLLM and _REGISTRY are not changed.
L — Every named class is a PyChartLLM; all are interchangeable.
I — Public interface: model, generate(), generate_stream(), __repr__().
D — High-level code (agent.py, smart_df.py) depends on PyChartLLM, not litellm.

API-key security
----------------
Keys are NEVER stored in __repr__, logs, or positional args.
Resolution order:
  1. Explicit  api_key=  argument
  2. Environment variable  (e.g. OPENAI_API_KEY)
  3. litellm reads its own env vars as a final fallback

Create a .env file at the project root (never commit it):
    OPENAI_API_KEY=sk-...
    GITHUB_TOKEN=ghp_...
    GEMINI_API_KEY=AIza...
    ANTHROPIC_API_KEY=sk-ant-...
    DEEPSEEK_API_KEY=...
    DASHSCOPE_API_KEY=...

Usage::

    import pychartai as pai

    # Universal — any provider
    llm = pai.PyChartLLM(model='ollama/llama3.2')
    llm = pai.PyChartLLM(model='openai/gpt-4o')
    llm = pai.PyChartLLM(model='github/gpt-4.1')

    # Named convenience classes
    llm = pai.OllamaLLM(model='llama3.2')
    llm = pai.OpenAILLM(model='gpt-4o')
    llm = pai.GitHubLLM(model='gpt-4.1')
"""

from __future__ import annotations

import os
from typing import Iterator, Optional

from .model_manager import LLMProvider, LiteLLMProvider, OllamaAvailabilityChecker
from .config import OLLAMA_BASE_URL

# Registry: pychartai prefix -> (litellm_prefix, env_var)
_REGISTRY: dict[str, tuple[str, str]] = {
	'openai':    ('',          'OPENAI_API_KEY'),    # litellm uses bare model names for openai
	'github':    ('openai',    'GITHUB_TOKEN'),       # github endpoint speaks openai protocol
	'gemini':    ('gemini',    'GEMINI_API_KEY'),
	'anthropic': ('anthropic', 'ANTHROPIC_API_KEY'),
	'deepseek':  ('deepseek',  'DEEPSEEK_API_KEY'),
	'qwen':      ('',          'DASHSCOPE_API_KEY'),  # DashScope is openai-compat; bare name
	'ollama':    ('ollama',    ''),
	'generic':   ('',          'OPENAI_API_KEY'),
}

# Provider-specific litellm api_base overrides
_BASE_URLS: dict[str, str] = {
	'github': 'https://models.github.ai/inference',
	'qwen':   'https://dashscope.aliyuncs.com/compatible-mode/v1',
}

# Models that must not receive a temperature parameter
_NO_TEMP_PATTERNS = ('gpt-5',)




class PyChartLLM:
	"""Universal LLM provider — the single public API class.

	Wraps any supported model with one consistent interface. Named subclasses
	(OllamaLLM, OpenAILLM, …) are convenience shortcuts that pre-fill defaults.

	Args:
		model:       'provider/model_name'  e.g. 'ollama/llama3.2', 'openai/gpt-4o',
		             'github/gpt-4.1', 'gemini/gemini-2.0-flash'.
		api_key:     Optional explicit key.  Omit to use the env variable.  Never
		             stored in repr or logs (name-mangled).
		base_url:    Override the provider default endpoint URL.
		temperature: Sampling temperature (default 0.1).
		max_tokens:  Token budget per response (default 2048).
	"""

	def __init__(
		self,
		model: str,
		api_key: Optional[str] = None,
		base_url: Optional[str] = None,
		temperature: float = 0.1,
		max_tokens: int = 2048,
	) -> None:
		if '/' not in model:
			raise ValueError(
				f'model must be "provider/model_name", e.g. "ollama/llama3.2". Got: {model!r}'
			)
		self.model = model
		self._temperature = temperature
		self._max_tokens = max_tokens
		self.__api_key = api_key        # name-mangled — not in repr
		self._base_url = base_url
		self._provider: Optional[LLMProvider] = None

	@property
	def provider_name(self) -> str:
		return self.model.split('/')[0]

	@property
	def model_name(self) -> str:
		return self.model.split('/', 1)[1]

	def _resolve_api_key(self, env_var: str) -> Optional[str]:
		"""Explicit key → env var → None (litellm handles its own env fallback)."""
		return self.__api_key or (os.environ.get(env_var) if env_var else None)

	def _get_provider(self) -> LLMProvider:
		"""Lazy-build the underlying LiteLLMProvider (cached after first call)."""
		if self._provider is not None:
			return self._provider

		pname = self.provider_name
		mname = self.model_name

		if pname not in _REGISTRY and not self._base_url:
			raise ValueError(
				f'Unknown provider {pname!r}. Known: {sorted(_REGISTRY)}. '
				'For custom endpoints pass base_url=.',
			)

		litellm_prefix, env_var = _REGISTRY.get(pname, ('', 'OPENAI_API_KEY'))
		api_key = self._resolve_api_key(env_var)

		# Build the litellm model string
		if litellm_prefix:
			litellm_model = f'{litellm_prefix}/{mname}'
		else:
			litellm_model = mname   # openai, qwen, generic use bare model names

		body_url = self._base_url or _BASE_URLS.get(pname)

		self._provider = LiteLLMProvider(
			model=litellm_model,
			api_key=api_key,
			base_url=body_url,
			temperature=self._temperature,
			max_tokens=self._max_tokens,
		)
		# Drop temperature for models that don't accept it
		if any(p in mname for p in _NO_TEMP_PATTERNS):
			self._provider._call_kwargs.pop('temperature', None)

		return self._provider

	# --- Delegation to provider -----------------------------------------------

	def generate(self, prompt: str, **kwargs) -> str:
		return self._get_provider().generate(prompt, **kwargs)

	def generate_stream(self, prompt: str, **kwargs) -> Iterator[str]:
		yield from self._get_provider().generate_stream(prompt, **kwargs)

	@property
	def last_usage(self) -> dict:
		"""Token usage from the most recent generate() call.

		Returns a dict with keys ``prompt_tokens``, ``completion_tokens``,
		``total_tokens``, and ``model``.  Empty dict for local models (Ollama)
		that do not report usage.
		"""
		if self._provider is None:
			return {}
		return getattr(self._provider, 'last_usage', {})

	def __repr__(self) -> str:
		return f'PyChartLLM(model={self.model!r})'


# ---------------------------------------------------------------------------
# Named convenience subclasses (Open/Closed: extend without modifying above)
# ---------------------------------------------------------------------------

class OllamaLLM(PyChartLLM):
	"""Local Ollama model.

	Args:
		model:    Ollama model name, e.g. 'llama3.2' or 'mistral'.
		base_url: Server URL. Defaults to $OLLAMA_HOST or localhost:11434.
		validate: Raise RuntimeError on init if the model is missing on the server.
	"""

	def __init__(
		self,
		model: str = 'llama3.2',
		base_url: Optional[str] = None,
		validate: bool = False,
	) -> None:
		super().__init__(f'ollama/{model}', base_url=base_url)
		if validate:
			checker = OllamaAvailabilityChecker(model, base_url=base_url)
			if not checker.model_exists():
				raise RuntimeError(
					f'Model "{model}" not found on Ollama at {base_url or OLLAMA_BASE_URL}. '
					f'Pull it with: ollama pull {model}'
				)

	def __repr__(self) -> str:
		return f'OllamaLLM(model={self.model_name!r})'


class GitHubLLM(PyChartLLM):
	"""GitHub AI Models (reads GITHUB_TOKEN from env)."""

	def __init__(self, model: str = 'gpt-4.1', api_key: Optional[str] = None) -> None:
		super().__init__(f'github/{model}', api_key=api_key)

	def __repr__(self) -> str:
		return f'GitHubLLM(model={self.model_name!r})'


class OpenAILLM(PyChartLLM):
	"""OpenAI API (reads OPENAI_API_KEY from env)."""

	def __init__(self, model: str = 'gpt-4o', api_key: Optional[str] = None) -> None:
		super().__init__(f'openai/{model}', api_key=api_key)

	def __repr__(self) -> str:
		return f'OpenAILLM(model={self.model_name!r})'


class GeminiLLM(PyChartLLM):
	"""Google Gemini via OpenAI-compatible endpoint (reads GEMINI_API_KEY from env)."""

	def __init__(self, model: str = 'gemini-2.0-flash', api_key: Optional[str] = None) -> None:
		super().__init__(f'gemini/{model}', api_key=api_key)

	def __repr__(self) -> str:
		return f'GeminiLLM(model={self.model_name!r})'


class AnthropicLLM(PyChartLLM):
	"""Anthropic Claude via OpenAI-compatible endpoint (reads ANTHROPIC_API_KEY from env)."""

	def __init__(self, model: str = 'claude-3-5-sonnet-20241022', api_key: Optional[str] = None) -> None:
		super().__init__(f'anthropic/{model}', api_key=api_key)

	def __repr__(self) -> str:
		return f'AnthropicLLM(model={self.model_name!r})'


class QwenLLM(PyChartLLM):
	"""Alibaba Qwen via DashScope (reads DASHSCOPE_API_KEY from env)."""

	def __init__(self, model: str = 'qwen-plus', api_key: Optional[str] = None) -> None:
		super().__init__(f'qwen/{model}', api_key=api_key)

	def __repr__(self) -> str:
		return f'QwenLLM(model={self.model_name!r})'


class DeepSeekLLM(PyChartLLM):
	"""DeepSeek AI (reads DEEPSEEK_API_KEY from env)."""

	def __init__(self, model: str = 'deepseek-chat', api_key: Optional[str] = None) -> None:
		super().__init__(f'deepseek/{model}', api_key=api_key)

	def __repr__(self) -> str:
		return f'DeepSeekLLM(model={self.model_name!r})'


class GenericLLM(PyChartLLM):
	"""Any OpenAI-compatible endpoint.

	Args:
		model:       Model name as the endpoint expects it.
		base_url:    Full base URL of the service.
		api_key:     API key. Falls back to api_key_env env var.
		api_key_env: Environment variable to read when api_key is None.
		temperature: Sampling temperature.
		max_tokens:  Token budget.

	Example::

		llm = pai.GenericLLM(
			model='mistral-large-latest',
			base_url='https://api.mistral.ai/v1',
			api_key_env='MISTRAL_API_KEY',
		)
	"""

	def __init__(
		self,
		model: str,
		base_url: str,
		api_key: Optional[str] = None,
		api_key_env: str = 'OPENAI_API_KEY',
		temperature: float = 0.1,
		max_tokens: int = 2048,
	) -> None:
		resolved_key = api_key or os.environ.get(api_key_env)
		super().__init__(
			f'generic/{model}',
			api_key=resolved_key,
			base_url=base_url,
			temperature=temperature,
			max_tokens=max_tokens,
		)

	def __repr__(self) -> str:
		return f'GenericLLM(model={self.model_name!r}, base_url={self._base_url!r})'


class PandasAILLM:
	"""Pass-through wrapper for any native pandasai-compatible LLM.

	Note: the PyChartAgent path (default) and streaming are NOT available via
	this wrapper — pandasai's own Agent handles execution.
	"""

	def __init__(self, pandasai_llm) -> None:
		self._llm = pandasai_llm

	def get_inner(self):
		return self._llm

	def __repr__(self) -> str:
		return f'PandasAILLM({self._llm!r})'
