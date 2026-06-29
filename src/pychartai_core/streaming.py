"""streaming.py — Stream types for pychartai async/streaming responses.

Usage::

    import pychartai as pai

    for event in df.chat_stream("What is the average revenue by region?"):
        if event.type == 'token':
            print(event.text, end='', flush=True)
        elif event.type == 'result':
            print('\\n\\nFinal answer:', event.value)
        elif event.type == 'error':
            print('\\nError:', event.error)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class StreamEvent:
	"""An event emitted during a streaming analysis.

	Attributes:
		type:  Event kind — one of ``'token'``, ``'result'``, or ``'error'``.
		text:  The raw LLM token (only populated when *type* is ``'token'``).
		value: The final analysis answer (only populated when *type* is
		       ``'result'``).
		error: The error message (only populated when *type* is ``'error'``).

	Typical usage pattern::

		for event in df.chat_stream("Summarise the data"):
		    if event.type == 'token':
		        print(event.text, end='', flush=True)
		    elif event.type == 'result':
		        final_answer = event.value
	"""

	type: str		# 'token' | 'result' | 'error'
	text: str = ''		# populated for type='token'
	value: Any = None	# populated for type='result'
	error: str = ''		# populated for type='error'

	def __str__(self) -> str:
		if self.type == 'token':
			return self.text
		if self.type == 'result':
			return str(self.value) if self.value is not None else ''
		return f'Error: {self.error}'

	def is_final(self) -> bool:
		"""Return *True* if this is the last event in the stream."""
		return self.type in ('result', 'error')
