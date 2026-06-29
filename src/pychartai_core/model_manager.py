"""
model_manager.py — LLM provider abstraction layer.

SOLID design
------------
S — LLMProvider defines one responsibility: text generation.
O — New providers are added as subclasses; nothing here is modified.
L — Every subclass is a valid LLMProvider substitute.
I — Interface is minimal: generate() + generate_stream().
D — Upper layers depend on LLMProvider, not on litellm or requests directly.
"""

from __future__ import annotations

import os
import random
import time
from abc import ABC, abstractmethod
from typing import Iterator, Optional


class LLMProvider(ABC):
	"""Abstract base for all LLM providers.

	Every concrete provider must implement :meth:`generate`.
	Streaming falls back to :meth:`generate` by default.
	"""

	@abstractmethod
	def generate(self, prompt: str, **kwargs) -> str:
		"""Return a complete text response for *prompt*."""

	def generate_stream(self, prompt: str, **kwargs) -> Iterator[str]:
		"""Yield response tokens. Default: yield full response as one token."""
		yield self.generate(prompt, **kwargs)


class LiteLLMProvider(LLMProvider):
	"""Concrete LLM provider backed by the litellm library.

	litellm unifies 100+ providers behind one interface. API keys are read
	from environment variables automatically — pass api_key only to override.

	Model string examples:
	  'ollama/llama3.2'              local Ollama
	  'gpt-4o'                       OpenAI           (OPENAI_API_KEY)
	  'github/gpt-4.1'               GitHub Models    (GITHUB_TOKEN)
	  'gemini/gemini-2.0-flash'       Google Gemini    (GEMINI_API_KEY)
	  'anthropic/claude-3-5-sonnet'   Anthropic        (ANTHROPIC_API_KEY)
	  'deepseek/deepseek-chat'        DeepSeek         (DEEPSEEK_API_KEY)
	"""

	def __init__(
		self,
		model: str,
		api_key: Optional[str] = None,
		base_url: Optional[str] = None,
		temperature: float = 0.1,
		max_tokens: int = 2048,
		num_retries: int = 3,
		**extra_kwargs,
	) -> None:
		self._model = model
		self.last_usage: dict = {}   # populated after each generate() call
		self._call_kwargs: dict = {
			'temperature': temperature,
			'max_tokens': max_tokens,
			'num_retries': num_retries,
			**extra_kwargs,
		}
		if api_key:
			self._call_kwargs['api_key'] = api_key
		if base_url:
			self._call_kwargs['api_base'] = base_url

	_MAX_RATE_RETRIES = 4          # 1 + 3 back-off attempts
	_RATE_BASE_DELAY  = 2.0        # seconds before first retry
	_RATE_MAX_DELAY   = 60.0       # cap

	def generate(self, prompt: str, **kwargs) -> str:
		import litellm
		merged = {**self._call_kwargs, **kwargs}
		last_exc: Optional[Exception] = None

		for attempt in range(self._MAX_RATE_RETRIES):
			try:
				resp = litellm.completion(
					model=self._model,
					messages=[{'role': 'user', 'content': prompt}],
					**merged,
				)
				# Capture token usage for cost tracking (may be None for local models)
				usage = getattr(resp, 'usage', None)
				if usage is not None:
					self.last_usage = {
						'prompt_tokens': getattr(usage, 'prompt_tokens', 0) or 0,
						'completion_tokens': getattr(usage, 'completion_tokens', 0) or 0,
						'total_tokens': getattr(usage, 'total_tokens', 0) or 0,
						'model': self._model,
					}
				return resp.choices[0].message.content.strip()

			except Exception as exc:
				last_exc = exc
				err_str = str(exc).lower()
				is_rate_limit = (
					'ratelimit' in err_str
					or 'rate_limit' in err_str
					or 'rate limit' in err_str
					or '429' in err_str
					or getattr(exc, 'status_code', None) == 429
				)
				if is_rate_limit and attempt < self._MAX_RATE_RETRIES - 1:
					# Exponential backoff with full jitter (Decorrelated Jitter)
					delay = min(
						self._RATE_BASE_DELAY * (2 ** attempt) * random.uniform(0.5, 1.5),
						self._RATE_MAX_DELAY,
					)
					time.sleep(delay)
					continue
				# Not a rate-limit error, or exhausted retries
				raise RuntimeError(f'LLM error ({self._model}): {exc}') from exc

		raise RuntimeError(
			f'Rate limit: {self._model} returned 429 after {self._MAX_RATE_RETRIES} attempts. '
			'Consider reducing request frequency or upgrading your API tier.'
		) from last_exc

	def generate_stream(self, prompt: str, **kwargs) -> Iterator[str]:
		import litellm
		merged = {**self._call_kwargs, **kwargs, 'stream': True}
		try:
			for chunk in litellm.completion(
				model=self._model,
				messages=[{'role': 'user', 'content': prompt}],
				**merged,
			):
				delta = chunk.choices[0].delta.content
				if delta:
					yield delta
		except Exception as exc:
			raise RuntimeError(f'Stream error ({self._model}): {exc}') from exc


class OllamaAvailabilityChecker:
	"""Health-check helper for local Ollama server/model presence.

	Kept separate (SRP) so LiteLLMProvider remains free of health-check logic.
	"""

	def __init__(self, model_name: str, base_url: Optional[str] = None) -> None:
		self.model_name = model_name
		self.base_url = base_url or os.environ.get('OLLAMA_HOST', 'http://localhost:11434')

	def is_available(self) -> bool:
		import requests
		try:
			return requests.get(f'{self.base_url}/api/tags', timeout=5).status_code == 200
		except Exception:
			return False

	def model_exists(self) -> bool:
		import requests
		for ep in ('/api/models', '/api/list', '/api/tags'):
			try:
				r = requests.get(f'{self.base_url}{ep}', timeout=5)
				if r.status_code == 200 and self.model_name in r.text:
					return True
			except Exception:
				continue
		return False


# Backward-compat alias — external code importing OllamaProvider still works
OllamaProvider = LiteLLMProvider


