"""
error_hints.py — Map common error patterns to helpful suggestions.

Used by the agent and SmartDataFrame to provide actionable error messages
instead of raw tracebacks.
"""

from __future__ import annotations

from typing import Optional


_HINTS: list[tuple[str, str]] = [
	# LLM connectivity
	('connection refused', 'Ollama is not running. Start it with: ollama serve'),
	('connect to ollama', 'Ollama is not running. Start it with: ollama serve'),
	('could not connect', 'The LLM service is unreachable. Check the URL and that the server is running.'),

	# API keys
	('api key', 'Set your API key via environment variable (e.g. OPENAI_API_KEY) or pai.config.set().'),
	('api_key', 'Set your API key via environment variable (e.g. OPENAI_API_KEY) or pai.config.set().'),
	('authentication', 'Check your API key. It may be invalid or expired.'),
	('401', 'Authentication failed. Verify your API key is correct.'),
	('403', 'Access forbidden. Your API key may lack the required permissions.'),

	# Rate limits
	('rate limit', 'You are being rate-limited. Wait a moment and try again.'),
	('429', 'Too many requests. The API is rate-limiting you. Wait and retry.'),
	('quota', 'API quota exceeded. Check your billing/plan limits.'),

	# Model issues
	('model not found', 'The requested model is not available. Check the model name and that it is pulled/deployed.'),
	('does not exist', 'The requested model does not exist. Run "ollama pull <model>" or check the model name.'),

	# Timeout
	('timed out', 'LLM generation timed out. Increase llm_timeout in config or use a faster model.'),
	('timeout', 'Request timed out. Increase llm_timeout or simplify the query.'),

	# Sandbox
	('restrictedpython', 'Code was blocked by the sandbox. The LLM may have generated unsafe code.'),
	('not allowed', 'The sandbox blocked an operation. Try rephrasing the query.'),
	('import', 'An import was blocked by the sandbox. Only whitelisted modules are allowed.'),

	# Docker
	('docker', 'Docker is not running or the container failed to start. Run: docker info'),
	('container', 'The Docker sandbox container may not be running. Call sandbox.start() first.'),

	# Data
	('empty dataframe', 'The DataFrame is empty. Load data before running a query.'),
	('keyerror', 'A column name was not found. Check that column names match exactly (case-sensitive).'),
	('no numeric', 'No numeric columns found. The operation requires at least one numeric column.'),

	# pandasai
	('pandasai', 'If pandasai is not installed, use agent=\'own\' or the default sandbox mode.'),

	# Code generation errors
	("'nonetype' object is not callable", 'The LLM generated code that calls None as a function. Try rephrasing as a simpler aggregation (e.g., "What is the total X?" instead of "How does X affect Y?").'),
	('nonetype', 'The LLM generated code that produced an unexpected None. Rephrase the question more specifically.'),
	('object is not callable', 'The LLM generated code that called a non-callable value. Try a more specific question.'),

	# General
	('max retries', 'All retry attempts failed. The LLM may be struggling with this query — try rephrasing.'),
]


def get_hint(error_message: str) -> Optional[str]:
	"""Return a helpful suggestion for *error_message*, or None."""
	lowered = error_message.lower()
	for pattern, hint in _HINTS:
		if pattern in lowered:
			return hint
	return None


def format_error_with_hint(error_message: str) -> str:
	"""Return the error message with an appended hint if one matches."""
	hint = get_hint(error_message)
	if hint:
		return f'{error_message}\n\nSuggestion: {hint}'
	return error_message
