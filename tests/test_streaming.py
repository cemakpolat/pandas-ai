"""Tests for streaming.py (StreamEvent) and SmartDataFrame.chat_stream() / achat()."""

import asyncio
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pandas as pd
import pytest

from pychartai_core.streaming import StreamEvent
from pychartai_core.smart_df import SmartDataFrame
from pychartai_core.config import config


# ---------------------------------------------------------------------------
# StreamEvent dataclass
# ---------------------------------------------------------------------------

class TestStreamEvent:

	def test_token_event_fields(self):
		ev = StreamEvent(type='token', text='hello')
		assert ev.type == 'token'
		assert ev.text == 'hello'
		assert ev.value is None
		assert ev.error == ''

	def test_result_event_fields(self):
		ev = StreamEvent(type='result', value='42')
		assert ev.type == 'result'
		assert ev.value == '42'
		assert ev.text == ''
		assert ev.error == ''

	def test_error_event_fields(self):
		ev = StreamEvent(type='error', error='something went wrong')
		assert ev.type == 'error'
		assert ev.error == 'something went wrong'
		assert ev.text == ''
		assert ev.value is None

	def test_default_text_is_empty_string(self):
		ev = StreamEvent(type='result', value=100)
		assert ev.text == ''

	def test_default_value_is_none(self):
		ev = StreamEvent(type='token', text='x')
		assert ev.value is None

	def test_default_error_is_empty_string(self):
		ev = StreamEvent(type='result', value='ok')
		assert ev.error == ''

	# __str__

	def test_str_token_returns_text(self):
		ev = StreamEvent(type='token', text='piece ')
		assert str(ev) == 'piece '

	def test_str_result_returns_value_string(self):
		ev = StreamEvent(type='result', value='The answer is 42')
		assert str(ev) == 'The answer is 42'

	def test_str_result_none_value_returns_empty(self):
		ev = StreamEvent(type='result', value=None)
		assert str(ev) == ''

	def test_str_result_non_string_value(self):
		ev = StreamEvent(type='result', value=3.14)
		assert str(ev) == '3.14'

	def test_str_error_returns_error_prefix(self):
		ev = StreamEvent(type='error', error='timeout')
		assert str(ev) == 'Error: timeout'

	# is_final

	def test_is_final_token_is_false(self):
		ev = StreamEvent(type='token', text='hi')
		assert ev.is_final() is False

	def test_is_final_result_is_true(self):
		ev = StreamEvent(type='result', value='done')
		assert ev.is_final() is True

	def test_is_final_error_is_true(self):
		ev = StreamEvent(type='error', error='oops')
		assert ev.is_final() is True

	def test_is_final_unknown_type_is_false(self):
		ev = StreamEvent(type='progress')
		assert ev.is_final() is False


# ---------------------------------------------------------------------------
# SmartDataFrame.chat_stream() — error paths (no real LLM needed)
# ---------------------------------------------------------------------------

class TestChatStreamErrorPaths:

	def setup_method(self):
		config.reset()
		self._df = SmartDataFrame(pd.DataFrame({'x': [1, 2, 3], 'y': [4, 5, 6]}))

	def test_yields_error_event_when_no_llm_configured(self):
		events = list(self._df.chat_stream('What is the average x?'))
		assert len(events) >= 1
		last = events[-1]
		assert last.type == 'error'
		assert 'No LLM' in last.error or 'llm' in last.error.lower()

	def test_yields_error_for_pandasai_llm(self, monkeypatch):
		"""PandasAILLM pass-through does not support streaming."""
		from pychartai_core import PandasAILLM

		# Minimal fake inner LLM for PandasAILLM (positional arg)
		class _FakeInner:
			pass

		fake_pai_llm = PandasAILLM(_FakeInner())
		config.set({'llm': fake_pai_llm})

		events = list(self._df.chat_stream('Summarise the data'))
		assert any(e.type == 'error' for e in events)
		error_events = [e for e in events if e.type == 'error']
		assert any('PandasAILLM' in e.error or 'streaming' in e.error.lower()
		           for e in error_events)

	def test_last_event_is_final(self):
		"""Regardless of outcome, the last event must be final (result or error)."""
		events = list(self._df.chat_stream('test'))
		assert len(events) >= 1
		assert events[-1].is_final()

	def test_yields_at_least_one_event(self):
		events = list(self._df.chat_stream('test'))
		assert len(events) >= 1


# ---------------------------------------------------------------------------
# SmartDataFrame.chat_stream() — happy path with a mock provider
# ---------------------------------------------------------------------------

class TestChatStreamMockProvider:

	def setup_method(self):
		config.reset()

	def _make_mock_llm(self, tokens, final_result='Answer: 42'):
		"""Build a mock OllamaLLM whose provider streams *tokens* then completes."""
		from pychartai_core.providers import OllamaLLM

		full_response = ''.join(tokens) + f'\nresult = "{final_result}"\n'
		all_tokens = tokens + [f'\nresult = "{final_result}"\n']

		class _MockProvider:
			def generate_stream(self, prompt):
				yield from all_tokens

			def generate(self, prompt, **kw):
				return full_response

			def is_available(self):
				return True

			def model_exists(self):
				return True

		mock_llm = OllamaLLM.__new__(OllamaLLM)
		mock_llm.model = 'ollama/test-model'  # full 'provider/model' string
		mock_llm._PyChartLLM__api_key = None
		mock_llm._base_url = None
		mock_llm._temperature = 0.1
		mock_llm._max_tokens = 2048
		mock_llm._provider = _MockProvider()
		return mock_llm

	def test_yields_token_events_then_result(self):
		tokens = ['The ', 'answer ', 'is ']
		mock_llm = self._make_mock_llm(tokens)
		config.set({'llm': mock_llm})

		df = SmartDataFrame(pd.DataFrame({'val': [1, 2, 3]}))
		events = list(df.chat_stream('What is the sum?'))

		token_events = [e for e in events if e.type == 'token']
		final_events = [e for e in events if e.is_final()]

		assert len(token_events) >= 1
		assert len(final_events) >= 1

	def test_all_tokens_concatenate_to_response(self):
		tokens = ['Hello', ', ', 'world', '!']
		mock_llm = self._make_mock_llm(tokens, final_result='done')
		config.set({'llm': mock_llm})

		df = SmartDataFrame(pd.DataFrame({'a': [1]}))
		events = list(df.chat_stream('test'))

		assembled = ''.join(e.text for e in events if e.type == 'token')
		assert 'Hello' in assembled

	def test_last_event_is_result_or_error(self):
		mock_llm = self._make_mock_llm(['token1'])
		config.set({'llm': mock_llm})

		df = SmartDataFrame(pd.DataFrame({'a': [1]}))
		events = list(df.chat_stream('test'))

		assert events[-1].is_final()

	def test_stream_with_custom_sandbox(self):
		from pychartai_core.sandbox import RestrictedSandbox
		mock_llm = self._make_mock_llm(['result = 99\n'])
		config.set({'llm': mock_llm})

		df = SmartDataFrame(pd.DataFrame({'n': [10, 20, 30]}))
		sandbox = RestrictedSandbox()
		events = list(df.chat_stream('Sum the values', sandbox=sandbox))

		assert len(events) >= 1
		assert events[-1].is_final()


# ---------------------------------------------------------------------------
# SmartDataFrame.achat() — async wrapper
# ---------------------------------------------------------------------------

class TestAchat:

	def setup_method(self):
		config.reset()

	def test_achat_raises_without_llm(self):
		df = SmartDataFrame(pd.DataFrame({'x': [1, 2]}))

		async def _run():
			return await df.achat('What is x?')

		with pytest.raises(RuntimeError, match='No LLM'):
			asyncio.run(_run())

	def test_achat_returns_same_as_chat(self, monkeypatch):
		"""achat() should return the same result as the underlying chat()."""
		df = SmartDataFrame(pd.DataFrame({'x': [1, 2]}))

		# Patch at class level since SmartDataFrame delegates instance attributes to _df
		monkeypatch.setattr(SmartDataFrame, 'chat', lambda self, *a, **kw: 'mocked result')

		async def _run():
			return await df.achat('query')

		result = asyncio.run(_run())
		assert result == 'mocked result'

	def test_achat_passes_chart_library_through(self, monkeypatch):
		df = SmartDataFrame(pd.DataFrame({'x': [1, 2]}))
		captured = {}

		def _fake_chat(self_inner, query, chart_library=None, **kw):
			captured['chart_library'] = chart_library
			return 'ok'

		monkeypatch.setattr(SmartDataFrame, 'chat', _fake_chat)

		async def _run():
			return await df.achat('plot it', chart_library='plotly')

		asyncio.run(_run())
		assert captured.get('chart_library') == 'plotly'

	def test_achat_passes_sandbox_through(self, monkeypatch):
		from pychartai_core.sandbox import RestrictedSandbox
		df = SmartDataFrame(pd.DataFrame({'x': [1]}))
		captured = {}

		def _fake_chat(self_inner, query, chart_library=None, sandbox=None, **kw):
			captured['sandbox'] = sandbox
			return 'ok'

		monkeypatch.setattr(SmartDataFrame, 'chat', _fake_chat)

		sb = RestrictedSandbox()

		async def _run():
			return await df.achat('test', sandbox=sb)

		asyncio.run(_run())
		assert captured.get('sandbox') is sb

	def test_achat_passes_plot_type_through(self, monkeypatch):
		df = SmartDataFrame(pd.DataFrame({'x': [1]}))
		captured = {}

		def _fake_chat(self_inner, query, chart_library=None, plot_type=None, **kw):
			captured['plot_type'] = plot_type
			return 'ok'

		monkeypatch.setattr(SmartDataFrame, 'chat', _fake_chat)

		async def _run():
			return await df.achat('plot x', plot_type='bar')

		asyncio.run(_run())
		assert captured.get('plot_type') == 'bar'

	def test_achat_is_awaitable(self):
		df = SmartDataFrame(pd.DataFrame({'x': [1]}))

		import inspect
		# Confirm achat returns a coroutine when called on an object
		coro = df.achat('test')
		assert inspect.isawaitable(coro)
		# Clean up without running
		coro.close()

	def test_achat_can_run_concurrently(self, monkeypatch):
		"""Two concurrent achat() calls should both complete."""
		df1 = SmartDataFrame(pd.DataFrame({'a': [1]}))
		df2 = SmartDataFrame(pd.DataFrame({'b': [2]}))

		monkeypatch.setattr(SmartDataFrame, 'chat', lambda self, *a, **kw: 'r')

		async def _run():
			r1, r2 = await asyncio.gather(df1.achat('q1'), df2.achat('q2'))
			return r1, r2

		r1, r2 = asyncio.run(_run())
		assert r1 == 'r'
		assert r2 == 'r'


# ---------------------------------------------------------------------------
# StreamEvent — edge cases
# ---------------------------------------------------------------------------

class TestStreamEventEdgeCases:

	def test_integer_result_value_str(self):
		ev = StreamEvent(type='result', value=0)
		assert str(ev) == '0'

	def test_dataframe_result_value_str(self):
		df = pd.DataFrame({'a': [1]})
		ev = StreamEvent(type='result', value=df)
		# Should call str() on the dataframe
		assert 'a' in str(ev)

	def test_empty_token_text(self):
		ev = StreamEvent(type='token', text='')
		assert str(ev) == ''

	def test_multiple_tokens_concat(self):
		events = [
			StreamEvent(type='token', text='The '),
			StreamEvent(type='token', text='answer '),
			StreamEvent(type='token', text='is 42'),
			StreamEvent(type='result', value='42'),
		]
		assembled = ''.join(str(e) for e in events if e.type == 'token')
		assert assembled == 'The answer is 42'
		assert events[-1].is_final()
