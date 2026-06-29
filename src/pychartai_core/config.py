"""
Global configuration for pychartai.

Provider-specific settings (base URLs, env-var names) live in providers.py.
Only universal settings shared across the whole package are kept here.
"""

import os
import threading
from typing import Any, Dict

# ---------------------------------------------------------------------------
# Ollama runtime constants (needed by OllamaProvider in model_manager.py)
# ---------------------------------------------------------------------------

OLLAMA_BASE_URL: str = os.environ.get('OLLAMA_HOST', 'http://localhost:11434')
OLLAMA_TIMEOUT: int = int(os.environ.get('OLLAMA_TIMEOUT', '300'))


# ---------------------------------------------------------------------------
# Global configuration singleton
# ---------------------------------------------------------------------------

class GlobalConfig:
	"""Thread-safe singleton config store for the pychartai public API.

	Usage::

		import pychartai as pai
		pai.config.set({'llm': pai.OllamaLLM(model='llama3.2')})
		pai.config.get('llm')
	"""

	_instance: 'GlobalConfig | None' = None
	_init_lock: threading.Lock = threading.Lock()
	_valid_backends = frozenset({'seaborn', 'matplotlib', 'plotly'})

	_defaults: Dict[str, Any] = {
		'llm': None,
		'chart_backend': 'seaborn',
		'charts_output_dir': 'exports/charts',
		'memory_size': 100,
		'verbose': False,
		'max_retries': 3,
		'llm_timeout': 60,
	}

	def __new__(cls) -> 'GlobalConfig':
		if cls._instance is None:
			with cls._init_lock:
				# Double-checked locking
				if cls._instance is None:
					instance = super().__new__(cls)
					instance._store = dict(cls._defaults)
					instance._rlock = threading.RLock()
					cls._instance = instance
		return cls._instance

	def set(self, options: Dict[str, Any]) -> None:
		"""Bulk-update configuration keys with validation.

		Thread-safe: safe to call from multiple threads simultaneously.
		"""
		self._validate(options)
		with self._rlock:
			self._store.update(options)

	def _validate(self, options: Dict[str, Any]) -> None:
		if 'chart_backend' in options:
			val = options['chart_backend']
			if val not in self._valid_backends:
				raise ValueError(
					f'Invalid chart_backend {val!r}. '
					f'Choose from: {sorted(self._valid_backends)}'
				)
		if 'max_retries' in options:
			val = options['max_retries']
			if not isinstance(val, int) or val < 0:
				raise ValueError(f'max_retries must be a non-negative integer, got {val!r}')
		if 'llm_timeout' in options:
			val = options['llm_timeout']
			if not isinstance(val, (int, float)) or val <= 0:
				raise ValueError(f'llm_timeout must be a positive number, got {val!r}')
		if 'memory_size' in options:
			val = options['memory_size']
			if not isinstance(val, int) or val < 0:
				raise ValueError(f'memory_size must be a non-negative integer, got {val!r}')

	def get(self, key: str, default: Any = None) -> Any:
		"""Retrieve a config value by key.

		Thread-safe: safe to call from multiple threads simultaneously.
		"""
		with self._rlock:
			return self._store.get(key, default)

	def reset(self) -> None:
		"""Reset all settings to defaults (useful in tests)."""
		with self._rlock:
			self._store = dict(self._defaults)

	def __repr__(self) -> str:
		with self._rlock:
			# Hide the LLM object repr for brevity; show everything else
			store = {k: v for k, v in self._store.items()}
		return f'GlobalConfig({store!r})'


# Module-level singleton
config = GlobalConfig()
