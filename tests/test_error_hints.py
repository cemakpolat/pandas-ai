"""Tests for error_hints module."""
import pytest

from pychartai_core.error_hints import get_hint, format_error_with_hint


class TestErrorHints:
	def test_ollama_not_running(self):
		hint = get_hint('Connection refused to Ollama')
		assert hint is not None
		assert 'ollama serve' in hint

	def test_api_key(self):
		hint = get_hint('Invalid API key provided')
		assert hint is not None
		assert 'API key' in hint or 'environment variable' in hint

	def test_rate_limit(self):
		hint = get_hint('Rate limit exceeded')
		assert hint is not None
		assert 'rate' in hint.lower()

	def test_timeout(self):
		hint = get_hint('Request timed out after 60s')
		assert hint is not None
		assert 'timeout' in hint.lower() or 'llm_timeout' in hint

	def test_empty_dataframe(self):
		hint = get_hint('Error: empty DataFrame')
		assert hint is not None

	def test_no_match_returns_none(self):
		hint = get_hint('Some completely unrelated error xyz123')
		assert hint is None

	def test_format_with_hint(self):
		msg = format_error_with_hint('Connection refused to server')
		assert 'Suggestion:' in msg

	def test_format_without_hint(self):
		msg = format_error_with_hint('Random unique error xyz')
		assert 'Suggestion:' not in msg
		assert msg == 'Random unique error xyz'

	def test_docker_hint(self):
		hint = get_hint('Docker daemon not running')
		assert hint is not None
		assert 'docker' in hint.lower()

	def test_model_not_found(self):
		hint = get_hint('Model not found: llama4')
		assert hint is not None
		assert 'model' in hint.lower()

	def test_max_retries(self):
		hint = get_hint('Max retries exceeded')
		assert hint is not None
		assert 'retry' in hint.lower() or 'rephras' in hint.lower()
