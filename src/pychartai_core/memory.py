"""
memory.py — Conversation memory for multi-turn context.

Stores the last N query-result pairs so follow-up queries
(e.g. 'now show that as a percentage', 'filter to Q3 only')
can reference prior context.

Usage::

    from pychartai_core.memory import ConversationMemory

    mem = ConversationMemory(window_size=10)
    mem.add('What is the average revenue?', '42000')
    context = mem.get_context()   # formatted string for LLM injection
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, List, Optional

# Hard cap per stored result to prevent memory leaks in long conversations.
# Full results are always returned by .chat(); this cap only affects the
# copy kept in memory for context injection.
_MAX_RESULT_CHARS: int = 2000


@dataclass(frozen=True)
class Turn:
	"""A single query-result pair.

	``result`` is capped at :data:`_MAX_RESULT_CHARS` characters at storage
	time to prevent long-running conversations from accumulating hundreds of
	megabytes of data.
	"""

	query: str
	result: str
	chart_path: Optional[str] = None


class ConversationMemory:
	"""Fixed-size sliding window of conversation turns.

	Args:
		window_size:      Maximum number of turns to retain.  Must be >= 1.
		max_result_chars: Maximum characters stored per result value.
		                  Defaults to 2000.  Set higher for richer context,
		                  lower to reduce memory consumption.
	"""

	def __init__(
		self,
		window_size: int = 10,
		max_result_chars: int = _MAX_RESULT_CHARS,
		*,
		max_turns: Optional[int] = None,
	) -> None:
		if max_turns is not None:
			window_size = max_turns
		if window_size < 1:
			raise ValueError('window_size must be >= 1')
		if max_result_chars < 1:
			raise ValueError('max_result_chars must be >= 1')
		self._window_size = window_size
		self._max_result_chars = max_result_chars
		self._turns: Deque[Turn] = deque(maxlen=window_size)

	# ------------------------------------------------------------------
	# Public API
	# ------------------------------------------------------------------

	def add(self, query: str, result: str, chart_path: Optional[str] = None) -> None:
		"""Record a completed turn.

		``result`` is automatically truncated to ``max_result_chars``
		before storage.
		"""
		result_str = str(result)
		if len(result_str) > self._max_result_chars:
			result_str = result_str[:self._max_result_chars] + ' [truncated]'
		self._turns.append(Turn(query=query, result=result_str, chart_path=chart_path))

	def get_context(self, max_turns: Optional[int] = None) -> str:
		"""Return a formatted string suitable for LLM prompt injection.

		Args:
			max_turns: Limit the number of recent turns included.
			           Defaults to all turns in the window.
		"""
		turns = list(self._turns)
		if max_turns is not None:
			turns = turns[-max_turns:]
		if not turns:
			return ''

		lines: list[str] = ['Conversation history (most recent last):']
		for i, turn in enumerate(turns, 1):
			lines.append(f'  Q{i}: {turn.query}')
			lines.append(f'  A{i}: {turn.result}')
		return '\n'.join(lines)

	@property
	def last_query(self) -> Optional[str]:
		"""Return the most recent query, or None."""
		return self._turns[-1].query if self._turns else None

	@property
	def last_result(self) -> Optional[str]:
		"""Return the most recent result, or None."""
		return self._turns[-1].result if self._turns else None

	@property
	def turns(self) -> List[Turn]:
		"""Return a copy of all stored turns."""
		return list(self._turns)

	def clear(self) -> None:
		"""Remove all stored turns."""
		self._turns.clear()

	def __len__(self) -> int:
		return len(self._turns)

	def __bool__(self) -> bool:
		return len(self._turns) > 0

	def __repr__(self) -> str:
		return (
			f'ConversationMemory(window_size={self._window_size}, '
			f'turns={len(self._turns)}, '
			f'max_result_chars={self._max_result_chars})'
		)
