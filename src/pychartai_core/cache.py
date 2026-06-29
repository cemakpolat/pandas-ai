"""
cache.py — File-based response cache for LLM query results.

Avoids redundant LLM calls for identical (query, DataFrame) pairs.
The cache key is derived from the raw query and a lightweight DataFrame
fingerprint (column names + dtypes + row count).

Usage::

    import pychartai as pai

    # Enable with the default cache directory (.pychartai_cache/)
    pai.config.set({'cache': True})

    # Or bring your own cache instance with a custom directory
    pai.config.set({'cache': pai.ResponseCache('.my_cache')})

    sdf = pai.read_csv('sales.csv')
    sdf.chat('What is the average revenue?')   # calls LLM
    sdf.chat('What is the average revenue?')   # served from cache

Cache entries are stored as UTF-8 JSON files.  Each filename is the first
16 hex characters of the SHA-256 of (query + fingerprint).
"""

from __future__ import annotations

import hashlib
import json
import os
from typing import Optional

import pandas as pd


class ResponseCache:
	"""Lightweight file-based JSON response cache.

	Args:
		cache_dir: Directory where ``.json`` cache entries are stored.
		           Created on demand if it does not exist.
	"""

	def __init__(self, cache_dir: str = '.pychartai_cache') -> None:
		self._dir = cache_dir
		os.makedirs(cache_dir, exist_ok=True)

	# ------------------------------------------------------------------
	# Public API
	# ------------------------------------------------------------------

	def get(self, query: str, df_fingerprint: str) -> Optional[str]:
		"""Return the cached response, or *None* on a cache miss."""
		path = self._entry_path(query, df_fingerprint)
		if os.path.isfile(path):
			try:
				with open(path, encoding='utf-8') as fh:
					return json.load(fh)['response']
			except (json.JSONDecodeError, KeyError, OSError):
				return None
		return None

	def put(self, query: str, df_fingerprint: str, response: str) -> None:
		"""Store *response* under the (query, fingerprint) key."""
		os.makedirs(self._dir, exist_ok=True)
		path = self._entry_path(query, df_fingerprint)
		entry = {
			'query': query,
			'fingerprint': df_fingerprint,
			'response': response,
		}
		with open(path, 'w', encoding='utf-8') as fh:
			json.dump(entry, fh, ensure_ascii=False, indent=2)

	def clear(self) -> int:
		"""Delete all cache entries.  Returns count of files removed."""
		removed = 0
		for fname in os.listdir(self._dir):
			if fname.endswith('.json'):
				try:
					os.remove(os.path.join(self._dir, fname))
					removed += 1
				except OSError:
					pass
		return removed

	def store(self, query: str, df: 'pd.DataFrame', response: str) -> None:
		"""Alias for put() using a DataFrame directly."""
		self.put(query, self.fingerprint(df), response)

	def lookup(self, query: str, df: 'pd.DataFrame') -> Optional[str]:
		"""Alias for get() using a DataFrame directly."""
		return self.get(query, self.fingerprint(df))

	def size(self) -> int:
		"""Return the number of cached entries currently on disk."""
		try:
			return sum(1 for f in os.listdir(self._dir) if f.endswith('.json'))
		except OSError:
			return 0

	# ------------------------------------------------------------------
	# Helpers
	# ------------------------------------------------------------------

	def _entry_path(self, query: str, df_fingerprint: str) -> str:
		raw = (query + df_fingerprint).encode('utf-8')
		key = hashlib.sha256(raw).hexdigest()[:16]
		return os.path.join(self._dir, f'{key}.json')

	# ------------------------------------------------------------------
	# Static helpers
	# ------------------------------------------------------------------

	@staticmethod
	def fingerprint(df: pd.DataFrame) -> str:
		"""Lightweight DataFrame fingerprint: columns + dtypes + shape."""
		col_info = ','.join(f'{c}:{t}' for c, t in zip(df.columns, df.dtypes))
		return f'{df.shape[0]}x{df.shape[1]}|{col_info}'
