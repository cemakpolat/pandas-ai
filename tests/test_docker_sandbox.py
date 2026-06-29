"""Integration tests for DockerSandbox — requires Docker to be running.

These tests actually spin up Docker containers and execute code inside them.
They are skipped automatically when Docker is not available.

Run subset:
    pytest tests/test_docker_sandbox.py -v

Run with output:
    pytest tests/test_docker_sandbox.py -v -s
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pandas as pd
import pytest

from pychartai_core.sandbox import DockerSandbox


# ---------------------------------------------------------------------------
# Pytest fixture / skip guard
# ---------------------------------------------------------------------------

def _docker_available() -> bool:
	import subprocess
	try:
		r = subprocess.run(
			['docker', 'info'],
			capture_output=True, timeout=10,
		)
		return r.returncode == 0
	except Exception:
		return False


requires_docker = pytest.mark.skipif(
	not _docker_available(),
	reason='Docker daemon is not available',
)


@pytest.fixture(scope='module')
def docker_sandbox():
	"""Module-scoped sandbox — started once, shared across all tests in this file."""
	sb = DockerSandbox(
		container_name='pychartai-test-sandbox',
		timeout=60,
		memory='256m',
		network='none',
	)
	sb.start()
	yield sb
	sb.stop()


# ---------------------------------------------------------------------------
# Basic execution inside Docker
# ---------------------------------------------------------------------------

@requires_docker
class TestDockerSandboxExecution:

	def test_scalar_result(self, docker_sandbox):
		result = docker_sandbox.execute('result = 42', {})
		assert result == 42

	def test_string_result(self, docker_sandbox):
		result = docker_sandbox.execute('result = "hello from docker"', {})
		assert result == 'hello from docker'

	def test_float_result(self, docker_sandbox):
		result = docker_sandbox.execute('result = 3.14', {})
		assert abs(float(result) - 3.14) < 0.001

	def test_arithmetic(self, docker_sandbox):
		result = docker_sandbox.execute('result = (10 + 5) * 2', {})
		assert result == 30

	def test_multiline_code(self, docker_sandbox):
		code = (
			'total = 0\n'
			'for i in range(1, 6):\n'
			'    total += i\n'
			'result = total\n'
		)
		result = docker_sandbox.execute(code, {})
		assert result == 15

	def test_no_result_variable_returns_fallback(self, docker_sandbox):
		result = docker_sandbox.execute('x = 1 + 1', {})
		# Container returns 'No result' string when result is not set
		assert result == 'No result' or result is None or result == ''

	def test_list_comprehension(self, docker_sandbox):
		code = 'result = str(sum(x**2 for x in range(1, 6)))'
		result = docker_sandbox.execute(code, {})
		assert str(result) == '55'


# ---------------------------------------------------------------------------
# DataFrame passed through Docker (JSON serialization round-trip)
# ---------------------------------------------------------------------------

@requires_docker
class TestDockerSandboxDataFrame:

	def test_dataframe_len(self, docker_sandbox):
		df = pd.DataFrame({'a': [1, 2, 3, 4, 5]})
		result = docker_sandbox.execute('result = str(len(df))', {'df': df})
		assert str(result) == '5'

	def test_dataframe_column_sum(self, docker_sandbox):
		df = pd.DataFrame({'sales': [100, 200, 300]})
		result = docker_sandbox.execute('result = str(int(df["sales"].sum()))', {'df': df})
		assert str(result) == '600'

	def test_dataframe_column_mean(self, docker_sandbox):
		df = pd.DataFrame({'revenue': [10.0, 20.0, 30.0]})
		result = docker_sandbox.execute('result = str(df["revenue"].mean())', {'df': df})
		assert abs(float(result) - 20.0) < 0.001

	def test_dataframe_filter_count(self, docker_sandbox):
		df = pd.DataFrame({'score': [10, 20, 30, 40, 50]})
		result = docker_sandbox.execute(
			'result = str(len(df[df["score"] > 25]))', {'df': df}
		)
		assert str(result) == '3'

	def test_dataframe_groupby(self, docker_sandbox):
		df = pd.DataFrame({
			'region': ['A', 'A', 'B', 'B'],
			'revenue': [100, 200, 150, 250],
		})
		code = (
			'agg = df.groupby("region")["revenue"].sum()\n'
			'result = str(int(agg["A"]))\n'
		)
		result = docker_sandbox.execute(code, {'df': df})
		assert str(result) == '300'

	def test_dataframe_max_value(self, docker_sandbox):
		df = pd.DataFrame({'val': [5, 3, 9, 1, 7]})
		result = docker_sandbox.execute('result = str(int(df["val"].max()))', {'df': df})
		assert str(result) == '9'

	def test_dataframe_result_returned_as_dataframe(self, docker_sandbox):
		"""When result is a DataFrame the container serializes it back via JSON."""
		df = pd.DataFrame({'x': [1, 2, 3]})
		code = 'result = df[df["x"] > 1]'
		result = docker_sandbox.execute(code, {'df': df})
		# Container returns a DataFrame deserialized from JSON
		assert isinstance(result, (pd.DataFrame, str))
		if isinstance(result, pd.DataFrame):
			assert len(result) == 2

	def test_multiple_dataframes(self, docker_sandbox):
		df1 = pd.DataFrame({'a': [1, 2]})
		df2 = pd.DataFrame({'b': [10, 20]})
		code = 'result = str(int(df["a"].sum() + df1["b"].sum()))'
		result = docker_sandbox.execute(code, {'df': df1, 'df1': df2})
		assert str(result) == '33'


# ---------------------------------------------------------------------------
# pandas / numpy inside Docker
# ---------------------------------------------------------------------------

@requires_docker
class TestDockerSandboxPandasNumpy:

	def test_pandas_describe(self, docker_sandbox):
		df = pd.DataFrame({'x': [1, 2, 3, 4, 5]})
		result = docker_sandbox.execute(
			'result = str(round(df["x"].mean(), 1))', {'df': df}
		)
		assert str(result) == '3.0'

	def test_numpy_available(self, docker_sandbox):
		code = 'import numpy as np; result = str(float(np.sqrt(144.0)))'
		result = docker_sandbox.execute(code, {})
		assert str(result) == '12.0'

	def test_pandas_available(self, docker_sandbox):
		code = (
			'import pandas as pd\n'
			'df2 = pd.DataFrame({"v": [1, 2, 3]})\n'
			'result = str(int(df2["v"].sum()))\n'
		)
		result = docker_sandbox.execute(code, {})
		assert str(result) == '6'

	def test_string_operations(self, docker_sandbox):
		df = pd.DataFrame({'name': ['Alice', 'Bob', 'Charlie']})
		code = 'result = str(df["name"].str.upper().tolist())'
		result = docker_sandbox.execute(code, {'df': df})
		assert "'ALICE'" in str(result)
		assert "'BOB'" in str(result)


# ---------------------------------------------------------------------------
# Error handling inside Docker
# ---------------------------------------------------------------------------

@requires_docker
class TestDockerSandboxErrors:

	def test_zero_division_error_raises(self, docker_sandbox):
		with pytest.raises(RuntimeError, match='[Ss]andbox execution error|ZeroDivisionError'):
			docker_sandbox.execute('result = 1 / 0', {})

	def test_name_error_raises(self, docker_sandbox):
		with pytest.raises(RuntimeError, match='[Ss]andbox execution error|NameError'):
			docker_sandbox.execute('result = undefined_variable_xyz_abc', {})

	def test_type_error_raises(self, docker_sandbox):
		with pytest.raises(RuntimeError, match='[Ss]andbox execution error|TypeError'):
			docker_sandbox.execute('result = "string" + 123', {})

	def test_error_does_not_poison_sandbox(self, docker_sandbox):
		"""After an error the sandbox container should still be usable."""
		try:
			docker_sandbox.execute('result = 1 / 0', {})
		except RuntimeError:
			pass
		# Subsequent call should still work
		result = docker_sandbox.execute('result = 99', {})
		assert result == 99


# ---------------------------------------------------------------------------
# Network isolation
# ---------------------------------------------------------------------------

@requires_docker
class TestDockerSandboxNetworkIsolation:

	def test_outbound_network_blocked(self, docker_sandbox):
		"""Container runs with --network=none, so socket connections fail."""
		code = (
			'import socket\n'
			'try:\n'
			'    s = socket.create_connection(("8.8.8.8", 53), timeout=2)\n'
			'    result = "connected"\n'
			'except (OSError, socket.error):\n'
			'    result = "blocked"\n'
		)
		result = docker_sandbox.execute(code, {})
		assert result == 'blocked'


# ---------------------------------------------------------------------------
# Context-manager lifecycle
# ---------------------------------------------------------------------------

@requires_docker
class TestDockerSandboxLifecycle:

	def test_context_manager_starts_and_stops(self):
		import uuid
		name = f'pychartai-ctxmgr-{uuid.uuid4().hex[:6]}'
		with DockerSandbox(container_name=name, timeout=30) as sb:
			assert sb._started is True
			result = sb.execute('result = "context manager works"', {})
			assert result == 'context manager works'
		assert sb._started is False

	def test_manual_start_stop(self):
		import uuid
		name = f'pychartai-manual-{uuid.uuid4().hex[:6]}'
		sb = DockerSandbox(container_name=name, timeout=30)
		assert sb._started is False
		sb.start()
		assert sb._started is True
		result = sb.execute('result = 2 + 2', {})
		assert result == 4
		sb.stop()
		assert sb._started is False

	def test_execute_raises_before_start(self):
		sb = DockerSandbox()
		with pytest.raises(RuntimeError, match='not running'):
			sb.execute('result = 1', {})

	def test_repr_shows_state(self):
		sb = DockerSandbox()
		assert 'stopped' in repr(sb)


# ---------------------------------------------------------------------------
# SmartDataFrame.chat() with DockerSandbox (requires LLM → skip if not set)
# ---------------------------------------------------------------------------

@requires_docker
class TestDockerSandboxWithSmartDataFrame:

	def test_chat_passes_sandbox_to_analyzer(self, docker_sandbox, monkeypatch):
		"""SmartDataFrame.chat(sandbox=docker_sandbox) reaches the analyzer."""
		from pychartai_core.smart_df import SmartDataFrame
		from pychartai_core.config import config
		import pychartai_core.analyzer as az

		captured = {}

		class _FakeAnalyzer:
			def __init__(self, *a, **kw):
				pass

			def analyze(self, df, query, **kw):
				captured['sandbox'] = kw.get('sandbox')
				return 'ok'

		monkeypatch.setattr(az, 'DataAnalyzer', _FakeAnalyzer)
		config.reset()
		config.set({'llm': object()})

		df = SmartDataFrame(pd.DataFrame({'x': [1, 2]}))
		result = df.chat('summarise', sandbox=docker_sandbox, agent='sandbox')

		assert result == 'ok'
		assert captured.get('sandbox') is docker_sandbox

	def test_module_chat_passes_sandbox(self, docker_sandbox, monkeypatch):
		"""Module-level pai.chat(query, df, sandbox=...) forwards the sandbox."""
		import pychartai as pai
		import pychartai_core.analyzer as az
		from pychartai_core.config import config

		captured = {}

		class _FakeAnalyzer:
			def __init__(self, *a, **kw):
				pass

			def analyze(self, df, query, **kw):
				captured['sandbox'] = kw.get('sandbox')
				return 'ok'

			def _analyze_with_sandbox_multi(self, query, frames, backend, sandbox):
				captured['sandbox'] = sandbox
				return 'ok'

		monkeypatch.setattr(az, 'DataAnalyzer', _FakeAnalyzer)
		config.reset()
		config.set({'llm': object()})

		df = pd.DataFrame({'x': [1]})
		pai.chat('test', df, sandbox=docker_sandbox)

		assert captured.get('sandbox') is docker_sandbox
