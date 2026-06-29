"""Tests for ConversationMemory."""
import pytest

from pychartai_core.memory import ConversationMemory, Turn


class TestConversationMemory:
	def test_add_and_retrieve(self):
		mem = ConversationMemory(window_size=5)
		mem.add('What is the mean?', '42')
		assert len(mem) == 1
		assert mem.last_query == 'What is the mean?'
		assert mem.last_result == '42'

	def test_window_eviction(self):
		mem = ConversationMemory(window_size=2)
		mem.add('q1', 'r1')
		mem.add('q2', 'r2')
		mem.add('q3', 'r3')
		assert len(mem) == 2
		assert mem.turns[0].query == 'q2'
		assert mem.turns[1].query == 'q3'

	def test_get_context_empty(self):
		mem = ConversationMemory()
		assert mem.get_context() == ''

	def test_get_context_format(self):
		mem = ConversationMemory()
		mem.add('hello', 'world')
		ctx = mem.get_context()
		assert 'Q1: hello' in ctx
		assert 'A1: world' in ctx

	def test_get_context_max_turns(self):
		mem = ConversationMemory(window_size=10)
		for i in range(5):
			mem.add(f'q{i}', f'r{i}')
		ctx = mem.get_context(max_turns=2)
		assert 'q3' in ctx
		assert 'q4' in ctx
		assert 'q0' not in ctx

	def test_clear(self):
		mem = ConversationMemory()
		mem.add('q', 'r')
		mem.clear()
		assert len(mem) == 0
		assert not mem

	def test_bool(self):
		mem = ConversationMemory()
		assert not mem
		mem.add('q', 'r')
		assert mem

	def test_invalid_window_size(self):
		with pytest.raises(ValueError, match='window_size must be >= 1'):
			ConversationMemory(window_size=0)

	def test_chart_path(self):
		mem = ConversationMemory()
		mem.add('plot', '/tmp/chart.png', chart_path='/tmp/chart.png')
		assert mem.turns[0].chart_path == '/tmp/chart.png'

	def test_last_query_empty(self):
		mem = ConversationMemory()
		assert mem.last_query is None
		assert mem.last_result is None
