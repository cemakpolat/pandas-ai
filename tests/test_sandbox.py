"""Tests for sandbox.py — RestrictedSandbox and DockerSandbox."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pandas as pd
import numpy as np
import pytest

from pychartai_core.sandbox import RestrictedSandbox, DockerSandbox


# ---------------------------------------------------------------------------
# RestrictedSandbox — basic execution
# ---------------------------------------------------------------------------

class TestRestrictedSandboxBasic:

	def setup_method(self):
		self.sb = RestrictedSandbox()

	def test_returns_none_when_no_result_set(self):
		result = self.sb.execute('x = 1', {})
		assert result is None

	def test_returns_scalar_result(self):
		result = self.sb.execute('result = 42', {})
		assert result == 42

	def test_returns_string_result(self):
		result = self.sb.execute('result = "hello"', {})
		assert result == 'hello'

	def test_returns_list_result(self):
		result = self.sb.execute('result = [1, 2, 3]', {})
		assert result == [1, 2, 3]

	def test_context_variables_accessible(self):
		code = 'result = x + y'
		result = self.sb.execute(code, {'x': 10, 'y': 5})
		assert result == 15

	def test_arithmetic_expressions(self):
		code = 'result = (a * b) + c'
		result = self.sb.execute(code, {'a': 3, 'b': 4, 'c': 2})
		assert result == 14

	def test_multiline_code_executes(self):
		code = (
			'total = 0\n'
			'for i in [1, 2, 3, 4, 5]:\n'
			'    total = total + i\n'
			'result = total\n'
		)
		result = self.sb.execute(code, {})
		assert result == 15

	def test_dataframe_passed_in_context(self):
		df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
		code = 'result = len(df)'
		result = self.sb.execute(code, {'df': df})
		assert result == 3

	def test_dataframe_operation_in_sandbox(self):
		df = pd.DataFrame({'val': [10, 20, 30]})
		code = 'result = df["val"].sum()'
		result = self.sb.execute(code, {'df': df})
		assert result == 60

	def test_numpy_array_operation(self):
		arr = np.array([1.0, 2.0, 3.0])
		code = 'result = float(np.mean(arr))'
		result = self.sb.execute(code, {'arr': arr, 'np': np})
		assert abs(result - 2.0) < 1e-9

	def test_inplace_add_operator(self):
		# Verify _simple_inplacevar helper handles iadd correctly.
		from pychartai_core.sandbox import _simple_inplacevar
		assert _simple_inplacevar('iadd', 10, 5) == 15

	def test_inplace_subtract_operator(self):
		from pychartai_core.sandbox import _simple_inplacevar
		assert _simple_inplacevar('isub', 10, 3) == 7

	def test_inplace_multiply_operator(self):
		from pychartai_core.sandbox import _simple_inplacevar
		assert _simple_inplacevar('imul', 4, 3) == 12


# ---------------------------------------------------------------------------
# RestrictedSandbox — blocked operations (security)
# ---------------------------------------------------------------------------

class TestRestrictedSandboxSecurity:

	def setup_method(self):
		self.sb = RestrictedSandbox()

	def test_blocks_os_import(self):
		with pytest.raises((ImportError, Exception)):
			self.sb.execute('import os; result = os.getcwd()', {})

	def test_blocks_subprocess_import(self):
		with pytest.raises((ImportError, Exception)):
			self.sb.execute('import subprocess; result = subprocess.check_output("ls")', {})

	def test_blocks_socket_import(self):
		with pytest.raises((ImportError, Exception)):
			self.sb.execute('import socket; result = socket.gethostname()', {})

	def test_blocks_sys_import(self):
		with pytest.raises((ImportError, Exception)):
			self.sb.execute('import sys; result = sys.version', {})

	def test_blocks_shutil_import(self):
		with pytest.raises((ImportError, Exception)):
			self.sb.execute('import shutil; result = "ok"', {})

	def test_blocks_glob_import(self):
		with pytest.raises((ImportError, Exception)):
			self.sb.execute('import glob; result = "ok"', {})

	def test_code_error_propagates(self):
		with pytest.raises(Exception):
			self.sb.execute('result = 1 / 0', {})

	def test_undefined_name_raises(self):
		with pytest.raises((NameError, Exception)):
			self.sb.execute('result = undefined_variable_xyz', {})

	def test_syntax_error_raises(self):
		with pytest.raises(Exception):
			self.sb.execute('result = (1 + ', {})


# ---------------------------------------------------------------------------
# RestrictedSandbox — allowed imports
# ---------------------------------------------------------------------------

class TestRestrictedSandboxAllowedImports:

	def setup_method(self):
		self.sb = RestrictedSandbox()

	def test_math_import_allowed(self):
		code = 'import math; result = math.sqrt(16)'
		result = self.sb.execute(code, {})
		assert abs(result - 4.0) < 1e-9

	def test_statistics_import_allowed(self):
		code = 'import statistics; result = statistics.mean([1, 2, 3, 4, 5])'
		result = self.sb.execute(code, {})
		assert result == 3.0

	def test_datetime_import_allowed(self):
		code = 'import datetime; result = str(datetime.date(2024, 1, 1))'
		result = self.sb.execute(code, {})
		assert result == '2024-01-01'

	def test_json_import_allowed(self):
		code = 'import json; result = json.dumps({"key": "value"})'
		result = self.sb.execute(code, {})
		import json
		assert json.loads(result) == {'key': 'value'}

	def test_re_import_allowed(self):
		code = 'import re; result = bool(re.match(r"\\d+", "123"))'
		result = self.sb.execute(code, {})
		assert result is True

	def test_collections_import_allowed(self):
		code = 'from collections import Counter; result = dict(Counter([1,1,2,3]))'
		result = self.sb.execute(code, {})
		assert result == {1: 2, 2: 1, 3: 1}

	def test_itertools_import_allowed(self):
		code = 'import itertools; result = list(itertools.islice(itertools.count(1), 3))'
		result = self.sb.execute(code, {})
		assert result == [1, 2, 3]


# ---------------------------------------------------------------------------
# RestrictedSandbox — custom allow_imports
# ---------------------------------------------------------------------------

class TestRestrictedSandboxCustomImports:

	def test_custom_allow_list_permits_extra_module(self):
		sb = RestrictedSandbox(allow_imports=('math', 'statistics'))
		result = sb.execute('import math; result = math.pi', {})
		assert abs(result - 3.14159) < 0.001

	def test_custom_allow_list_blocks_default_allowed_when_not_listed(self):
		"""If 're' is not in the custom allow list, it should be blocked."""
		sb = RestrictedSandbox(allow_imports=('math',))
		with pytest.raises((ImportError, Exception)):
			sb.execute('import re; result = "bad"', {})

	def test_empty_allow_list_blocks_everything(self):
		sb = RestrictedSandbox(allow_imports=())
		with pytest.raises((ImportError, Exception)):
			sb.execute('import math; result = math.pi', {})

	def test_repr_includes_allow_imports(self):
		sb = RestrictedSandbox(allow_imports=('math',))
		assert 'RestrictedSandbox' in repr(sb)
		assert 'math' in repr(sb)


# ---------------------------------------------------------------------------
# RestrictedSandbox — pandas / numpy integration (data science use case)
# ---------------------------------------------------------------------------

class TestRestrictedSandboxDataScience:

	def setup_method(self):
		self.sb = RestrictedSandbox()

	def test_pandas_groupby_aggregation(self):
		df = pd.DataFrame({
			'region': ['A', 'A', 'B', 'B'],
			'revenue': [100, 200, 150, 250],
		})
		code = (
			'agg = df.groupby("region")["revenue"].sum().reset_index()\n'
			'result = agg["revenue"].sum()\n'
		)
		result = self.sb.execute(code, {'df': df})
		assert result == 700

	def test_pandas_filtering(self):
		df = pd.DataFrame({'score': [10, 20, 30, 40, 50]})
		code = 'result = len(df[df["score"] > 25])'
		result = self.sb.execute(code, {'df': df})
		assert result == 3

	def test_pandas_describe_returns_dataframe(self):
		df = pd.DataFrame({'x': [1.0, 2.0, 3.0, 4.0, 5.0]})
		code = 'result = df.describe()'
		result = self.sb.execute(code, {'df': df})
		assert isinstance(result, pd.DataFrame)
		assert 'x' in result.columns

	def test_numpy_operations(self):
		code = (
			'arr = np.array([1, 4, 9, 16, 25])\n'
			'result = float(np.sqrt(arr).mean())\n'
		)
		result = self.sb.execute(code, {'np': np})
		assert abs(result - 3.0) < 0.001

	def test_dict_result_passthrough(self):
		code = 'result = {"type": "plot", "value": "/tmp/chart.png"}'
		result = self.sb.execute(code, {})
		assert isinstance(result, dict)
		assert result['type'] == 'plot'
		assert result['value'] == '/tmp/chart.png'


# ---------------------------------------------------------------------------
# DockerSandbox — initialization (no Docker required)
# ---------------------------------------------------------------------------

class TestDockerSandboxInit:

	def test_default_image(self):
		sb = DockerSandbox()
		assert sb.image == DockerSandbox.DEFAULT_IMAGE

	def test_custom_image(self):
		sb = DockerSandbox(image='python:3.12-slim')
		assert sb.image == 'python:3.12-slim'

	def test_container_name_generated_when_none(self):
		sb = DockerSandbox()
		assert sb.container_name.startswith('pychartai-sandbox-')

	def test_custom_container_name(self):
		sb = DockerSandbox(container_name='my-container')
		assert sb.container_name == 'my-container'

	def test_default_timeout(self):
		sb = DockerSandbox()
		assert sb.timeout == 30

	def test_custom_timeout(self):
		sb = DockerSandbox(timeout=60)
		assert sb.timeout == 60

	def test_default_memory(self):
		sb = DockerSandbox()
		assert sb.memory == '512m'

	def test_default_network(self):
		sb = DockerSandbox()
		assert sb.network == 'none'

	def test_not_started_initially(self):
		sb = DockerSandbox()
		assert sb._started is False

	def test_start_raises_when_docker_unavailable(self, monkeypatch):
		"""When Docker is not installed, start() raises RuntimeError."""
		import subprocess as _sp

		def _raise(*args, **kwargs):
			raise FileNotFoundError('docker not found')

		monkeypatch.setattr(_sp, 'run', _raise)
		sb = DockerSandbox()
		with pytest.raises(RuntimeError, match='Docker is not available'):
			sb.start()

	def test_execute_raises_when_not_started(self):
		sb = DockerSandbox()
		with pytest.raises(RuntimeError, match='not running'):
			sb.execute('result = 1', {'df': pd.DataFrame()})
