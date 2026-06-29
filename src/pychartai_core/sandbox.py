"""sandbox.py — Sandboxed Python code execution backends for pychartai.

Two sandbox implementations are provided:

RestrictedSandbox
    In-process execution via RestrictedPython.  Prevents arbitrary imports,
    file I/O, shell commands, and network calls from LLM-generated code.
    Chart generation is fully supported (chart helpers are injected as trusted
    functions).

DockerSandbox
    Isolated execution inside a Docker container (no host filesystem access,
    no network by default, memory limited).  Chart generation is not supported
    (no display server in the container); output is text / DataFrame only.

Usage::

    import pychartai as pai

    # --- RestrictedPython (no Docker required) ---
    sandbox = pai.RestrictedSandbox()
    df.chat("What is the average salary?", sandbox=sandbox)

    # --- Docker ---
    with pai.DockerSandbox() as sandbox:
        result = pai.chat("Who gets paid the most?", employees_df, salaries_df, sandbox=sandbox)

    # --- Docker (explicit lifecycle) ---
    sandbox = pai.DockerSandbox()
    sandbox.start()
    result = pai.chat("...", df, sandbox=sandbox)
    sandbox.stop()
"""

from __future__ import annotations

import builtins as _builtins
import io
import json
import operator as _operator
import os
import subprocess
import tempfile
import textwrap
import uuid
from typing import Any, Dict, Iterator, Optional

import pandas as pd

# ---------------------------------------------------------------------------
# Allowed imports for RestrictedSandbox
# ---------------------------------------------------------------------------

_DEFAULT_ALLOWED_IMPORTS: frozenset = frozenset({
	# Core data-science packages
	'pandas', 'numpy', 'math', 'statistics',
	# Standard library — safe, no shell/network access
	'datetime', 'json', 're', 'collections',
	'itertools', 'functools', 'operator',
	# Visualization backends (matplotlib/seaborn/plotly all write files via
	# the chart helpers which are injected as trusted callables; direct import
	# is still allowed so that LLM-generated helper code can reference them)
	'matplotlib', 'matplotlib.pyplot',
	'seaborn', 'plotly', 'plotly.express',
	# NOTE: scipy is intentionally excluded — its advanced modules expose
	# subprocess-level capabilities (e.g. scipy.io, scipy.integrate internals)
	# that could be used to escape the sandbox.  Add it back per-instance via
	# RestrictedSandbox(allow_imports=(..., 'scipy')) only when you trust the
	# LLM and accept the reduced isolation.
})

_real_import = _builtins.__import__


def _make_safe_importer(allowed: frozenset):
	"""Return a guarded __import__ that only permits top-level names in *allowed*."""

	def _guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
		top = name.split('.')[0]
		if top not in allowed:
			raise ImportError(
				f"Import of '{name}' is blocked in sandbox mode. "
				f"Allowed: {sorted(allowed)}"
			)
		return _real_import(name, globals, locals, fromlist, level)

	return _guarded_import


def _simple_inplacevar(op: str, x: Any, y: Any) -> Any:
	"""Handle in-place operations (+=, -=, …) inside restricted code."""
	_ops = {
		'iadd': _operator.iadd,
		'isub': _operator.isub,
		'imul': _operator.imul,
		'itruediv': _operator.truediv,
		'ifloordiv': _operator.floordiv,
		'imod': _operator.mod,
		'ipow': _operator.pow,
		'iand': _operator.and_,
		'ior': _operator.or_,
		'ixor': _operator.xor,
		'ilshift': _operator.lshift,
		'irshift': _operator.rshift,
	}
	fn = _ops.get(op)
	return fn(x, y) if fn else x


# ---------------------------------------------------------------------------
# RestrictedSandbox
# ---------------------------------------------------------------------------

class RestrictedSandbox:
	"""Execute LLM-generated pandas code via RestrictedPython.

	RestrictedPython compiles generated code with restrictions that block:

	- Arbitrary module imports (only ``allow_imports`` names are permitted)
	- ``eval``, ``exec``, ``compile``, raw ``__import__``
	- File I/O (``open`` is not in the safe builtins)
	- Shell commands (``os.system``, ``subprocess``, …)

	Trusted functions injected into the context (chart helpers, pandas,
	numpy) run at full Python privilege — only the *generated* code is
	restricted.

	Args:
		allow_imports: Tuple of module top-level names that generated code may
		               import.  Defaults to common data-science packages.

	Requires::

		pip install RestrictedPython

	Usage::

		sandbox = RestrictedSandbox()
		# context is built automatically by pychartai before calling execute()
		result = sandbox.execute(my_code, {'df': df, 'pd': pd, 'np': np})
	"""

	def __init__(
		self,
		allow_imports: tuple[str, ...] = tuple(_DEFAULT_ALLOWED_IMPORTS),
	) -> None:
		self.allow_imports = frozenset(allow_imports)

	def execute(self, code: str, context: dict[str, Any]) -> Any:
		"""Execute *code* in a RestrictedPython environment.

		Args:
			code:    Clean Python code string (no markdown fences).
			context: Variables injected as globals (``df``, ``pd``, ``np``,
			         chart helpers, ``chart_path``, etc.).

		Returns:
			The value of the ``result`` variable set by the code, or *None*.

		Raises:
			ImportError: If RestrictedPython is not installed.
			Any exception raised by the generated code.
		"""
		try:
			from RestrictedPython import compile_restricted, safe_globals
		except ImportError:
			raise ImportError(
				'RestrictedPython is required for sandbox execution. '
				'Install with: pip install RestrictedPython'
			)

		# Support both RestrictedPython 7.x and 8.x guard APIs
		try:
			# 7.x API
			from RestrictedPython.Guards import (
				guarded_getattr, guarded_getitem, guarded_getiter,
			)
			_getattr = guarded_getattr
			_getitem = guarded_getitem
			_getiter = guarded_getiter
		except ImportError:
			# 8.x API
			from RestrictedPython.Guards import safer_getattr
			_getattr = safer_getattr
			_getitem = lambda obj, key: obj[key]
			_getiter = iter
		# Write guard: allow assignments to known safe types only.
		# This blocks writes to protected objects while allowing normal
		# pandas attribute assignments (e.g. df.columns = [...]).
		def _write(obj):
			_blocked = (type, type(lambda: None), type(_make_safe_importer))
			if isinstance(obj, _blocked):
				raise AttributeError('Write access to this object is restricted in sandbox mode.')
			return obj

		byte_code = compile_restricted(code, '<pychartai-sandbox>', 'exec')

		globs: dict[str, Any] = dict(safe_globals)
		globs['_getattr_'] = _getattr
		globs['_getitem_'] = _getitem
		globs['_getiter_'] = _getiter
		globs['_write_'] = _write
		globs['_inplacevar_'] = _simple_inplacevar
		# RestrictedPython transforms `print(x)` → `_print_(getattr_)(x)` at compile
		# time.  Supply the PrintCollector *class* (not an instance) as _print_ so
		# that each print call creates a collector that has `_call_print`.
		try:
			from RestrictedPython import PrintCollector
			globs['_print_'] = PrintCollector
		except ImportError:
			globs['_print_'] = lambda *a, **kw: None
		# Inject the guarded importer into __builtins__ (where Python actually
		# looks for __import__) as well as into globals for safety.
		safe_importer = _make_safe_importer(self.allow_imports)
		if isinstance(globs.get('__builtins__'), dict):
			globs['__builtins__']['__import__'] = safe_importer
			# Also inject common safe built-ins that safe_globals omits but
			# data-analysis code routinely needs.
			for _name, _obj in (
				('list', list), ('dict', dict), ('tuple', tuple), ('set', set),
				('frozenset', frozenset), ('range', range), ('zip', zip),
				('map', map), ('filter', filter), ('sorted', sorted),
				('reversed', reversed), ('enumerate', enumerate),
				('type', type), ('isinstance', isinstance),
				('issubclass', issubclass), ('hasattr', hasattr),
				('getattr', getattr), ('setattr', setattr),
				('print', print), ('repr', repr),
				('abs', abs), ('pow', pow), ('divmod', divmod),
				('bool', bool), ('bytes', bytes), ('bytearray', bytearray),
				('iter', iter), ('next', next), ('any', any), ('all', all),
				('vars', vars), ('id', id),
			):
				globs['__builtins__'].setdefault(_name, _obj)
		globs['__import__'] = safe_importer
		globs.update(context)

		local_ns: dict[str, Any] = {}
		exec(byte_code, globs, local_ns)
		return local_ns.get('result')

	def __repr__(self) -> str:
		return f'RestrictedSandbox(allow_imports={sorted(self.allow_imports)!r})'


# ---------------------------------------------------------------------------
# DockerSandbox
# ---------------------------------------------------------------------------

_RUNNER_TEMPLATE = '''\
import io
import json
import sys
import pandas as pd
import numpy as np

# --- Deserialize dataframes ---
with open('__FRAMES_PATH__', 'r') as _f:
    _frames_data = json.load(_f)
for _k, _v in _frames_data.items():
    globals()[_k] = pd.read_json(io.StringIO(_v), orient='records')

# --- Execute generated code ---
try:
__CODE__
except Exception as _exc:
    print(json.dumps({"type": "error", "value": str(_exc)}), flush=True)
    sys.exit(0)  # always exit 0; caller inspects the JSON payload for errors

# --- Serialize result ---
_result = globals().get('result')
if isinstance(_result, pd.DataFrame):
    print(json.dumps({"type": "dataframe", "value": _result.to_json(orient='records')}), flush=True)
elif isinstance(_result, dict) and 'type' in _result and 'value' in _result:
    _v = _result['value']
    if isinstance(_v, pd.DataFrame):
        _result['value'] = _v.to_json(orient='records')
    print(json.dumps({"type": "string", "value": str(_v)}), flush=True)
elif isinstance(_result, (int, float)):
    print(json.dumps({"type": "number", "value": _result}), flush=True)
elif _result is not None:
    print(json.dumps({"type": "string", "value": str(_result)}), flush=True)
else:
    print(json.dumps({"type": "string", "value": "No result"}), flush=True)
'''


class DockerSandbox:
	"""Execute LLM-generated code in an isolated Docker container.

	The container runs with restricted network access, a memory limit,
	and no access to the host filesystem.  DataFrames are serialized to
	JSON, copied into the container, and execution results are read back
	via ``docker exec`` stdout.

	**Limitations**

	- Chart generation is not supported (no display server inside the
	  container).  Queries that produce charts will raise or return an
	  empty result.
	- The ``pandas`` and ``numpy`` packages are installed automatically
	  on :meth:`start` if the chosen image does not include them (takes
	  ~30 s on first run with ``python:3.11-slim``).  Use a pre-built
	  image to avoid this delay.

	Args:
		image:          Docker image to use.  Default is ``python:3.11-slim``.
		container_name: Container name.  Auto-generated when *None*.
		timeout:        Per-call execution timeout in seconds.
		memory:         Docker ``--memory`` limit (e.g. ``"512m"``).
		network:        Docker ``--network`` value.  ``"none"`` (default)
		                blocks all outbound network traffic.

	Usage::

		# Context-manager style (recommended)
		with pai.DockerSandbox() as sandbox:
		    result = pai.chat("Who gets paid the most?", df1, df2, sandbox=sandbox)

		# Manual lifecycle
		sandbox = pai.DockerSandbox(memory="1g", timeout=60)
		sandbox.start()
		result = df.chat("Summarise the data", sandbox=sandbox)
		sandbox.stop()
	"""

	DEFAULT_IMAGE = 'python:3.11-slim'
	MAX_OUTPUT_BYTES = 10 * 1024 * 1024  # 10 MB stdout limit

	def __init__(
		self,
		image: str = DEFAULT_IMAGE,
		container_name: Optional[str] = None,
		timeout: int = 30,
		memory: str = '512m',
		network: str = 'none',
	) -> None:
		self.image = image
		self.container_name = container_name or f'pychartai-sandbox-{uuid.uuid4().hex[:8]}'
		self.timeout = timeout
		self.memory = memory
		self.network = network
		self._started = False

	# ------------------------------------------------------------------
	# Lifecycle
	# ------------------------------------------------------------------

	def start(self) -> 'DockerSandbox':
		"""Pull the image (if needed) and start the sandbox container.

		Installs ``pandas`` and ``numpy`` inside the container unless they
		are already present in the chosen image.

		Returns:
			*self* for chaining / use as a context manager.

		Raises:
			RuntimeError: If Docker is not installed or not running.
		"""
		try:
			subprocess.run(
				['docker', 'info'],
				capture_output=True, check=True, timeout=10,
			)
		except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired) as exc:
			raise RuntimeError(
				'Docker is not available or not running. '
				'See https://docs.docker.com/get-docker/'
			) from exc

		# Start the container WITHOUT network isolation first so that pip can
		# download packages if they are not already in the image.
		# Security hardening (defense-in-depth — Docker namespace isolation
		# is the primary boundary; these reduce blast radius if escape is
		# attempted):
		#   --security-opt no-new-privileges  — block setuid/privilege escalation
		#   --cap-drop NET_ADMIN,NET_RAW,SYS_ADMIN,SYS_PTRACE,SYS_MODULE
		#                                     — drop the most dangerous Linux capabilities
		#   --network=none (applied below)    — block outbound network traffic
		#   memory limit                      — bound resource consumption
		# We keep DAC_OVERRIDE / CHOWN / FOWNER so `docker cp` of the runner
		# script remains readable inside the container.  A user wanting tighter
		# isolation should use a pre-built image that already has pandas/numpy
		# and switch to --user/--read-only manually.
		self._docker([
			'run', '-d', '--rm',
			'--name', self.container_name,
			f'--memory={self.memory}',
			'--security-opt', 'no-new-privileges',
			'--cap-drop', 'NET_ADMIN',
			'--cap-drop', 'NET_RAW',
			'--cap-drop', 'SYS_ADMIN',
			'--cap-drop', 'SYS_PTRACE',
			'--cap-drop', 'SYS_MODULE',
			self.image,
			'tail', '-f', '/dev/null',
		])

		# Install deps (fast if already cached in image layers).
		# This must happen before network isolation is applied.
		try:
			self._docker([
				'exec', self.container_name,
				'pip', 'install', 'pandas', 'numpy',
				'--quiet', '--no-cache-dir',
			], timeout=120)
		except RuntimeError as exc:
			# Verify pandas is actually importable; if not, the sandbox is useless.
			try:
				self._docker([
					'exec', self.container_name,
					'python3', '-c', 'import pandas, numpy',
				], timeout=10)
			except RuntimeError:
				self.stop()
				raise RuntimeError(
					'Failed to install pandas/numpy in the Docker sandbox and '
					'they are not available in the image. Use a pre-built image '
					'that includes pandas and numpy, or check your network.'
				) from exc

		# Apply network isolation after packages are installed.
		# docker network disconnect removes the container from the default
		# bridge network, giving equivalent isolation to --network=none.
		if self.network == 'none':
			try:
				self._docker(['network', 'disconnect', 'bridge', self.container_name])
			except RuntimeError:
				pass	# container may already be on a non-bridge network

		self._started = True
		return self

	def stop(self) -> None:
		"""Stop and remove the sandbox container."""
		if self._started:
			try:
				subprocess.run(
					['docker', 'stop', self.container_name],
					capture_output=True, timeout=15,
				)
			except Exception as exc:  # noqa: BLE001
				# Best-effort cleanup — container may have already exited
				import logging as _logging
				_logging.getLogger('pychartai').debug(
					'DockerSandbox.stop(): could not stop container %r: %s',
					self.container_name, exc,
				)
			self._started = False

	# ------------------------------------------------------------------
	# Execution
	# ------------------------------------------------------------------

	def execute(self, code: str, context: dict[str, Any]) -> Any:
		"""Run *code* inside the container.

		Only :class:`pandas.DataFrame` values in *context* are serialized
		and made available to the code.  Non-DataFrame entries (``pd``,
		``np``, chart helpers, etc.) are ignored because the container has
		its own Python environment.

		Args:
			code:    Python code string.
			context: Mapping that must include the DataFrame(s) the code
			         operates on (key = variable name used in the code,
			         e.g. ``{'df': employees_df, 'df1': salaries_df}``).

		Returns:
			Deserialized result value (string, number, or DataFrame).

		Raises:
			RuntimeError: If ``start()`` has not been called, or if Docker
			              command fails.
		"""
		if not self._started:
			raise RuntimeError(
				'DockerSandbox is not running. Call sandbox.start() first.'
			)

		# Extract only DataFrames; Docker provides its own pd/np environment
		dataframes = {k: v for k, v in context.items() if isinstance(v, pd.DataFrame)}
		frames_json = {k: df.to_json(orient='records') for k, df in dataframes.items()}

		run_id = uuid.uuid4().hex[:8]
		container_runner = f'/tmp/runner_{run_id}.py'
		container_frames = f'/tmp/frames_{run_id}.json'

		# Build runner script: replace placeholders without f-string to avoid
		# curly-brace conflicts with the embedded code.
		indented_code = textwrap.indent(code, '    ')
		runner_src = (
			_RUNNER_TEMPLATE
			.replace('__FRAMES_PATH__', container_frames)
			.replace('__CODE__', indented_code)
		)

		tmp_runner = tmp_frames = None
		try:
			with tempfile.NamedTemporaryFile(
				mode='w', suffix='.py', delete=False, encoding='utf-8'
			) as f:
				f.write(runner_src)
				tmp_runner = f.name

			with tempfile.NamedTemporaryFile(
				mode='w', suffix='.json', delete=False, encoding='utf-8'
			) as f:
				json.dump(frames_json, f)
				tmp_frames = f.name

			self._docker(['cp', tmp_frames, f'{self.container_name}:{container_frames}'])
			self._docker(['cp', tmp_runner, f'{self.container_name}:{container_runner}'])

			output = self._docker(
				['exec', self.container_name, 'python3', container_runner],
				timeout=self.timeout,
			)
		finally:
			for p in [tmp_runner, tmp_frames]:
				if p:
					try:
						os.unlink(p)
					except OSError:
						pass

		# Parse result from (last non-empty) JSON line
		if len(output.encode('utf-8', errors='replace')) > self.MAX_OUTPUT_BYTES:
			raise RuntimeError(
				f'Docker sandbox output exceeded {self.MAX_OUTPUT_BYTES // (1024 * 1024)} MB limit. '
				'The generated code may be producing excessive output.'
			)
		for line in reversed(output.strip().splitlines()):
			line = line.strip()
			if not line:
				continue
			try:
				data = json.loads(line)
				rtype = data.get('type', 'string')
				rval = data.get('value', '')
				if rtype == 'error':
					raise RuntimeError(f'Sandbox execution error: {rval}')
				if rtype == 'dataframe':
					return pd.read_json(io.StringIO(rval), orient='records')
				return rval
			except json.JSONDecodeError:
				continue

		return output.strip() or 'No result'

	# ------------------------------------------------------------------
	# Context-manager support
	# ------------------------------------------------------------------

	def __enter__(self) -> 'DockerSandbox':
		return self.start()

	def __exit__(self, *args: Any) -> None:
		self.stop()

	def __repr__(self) -> str:
		state = 'running' if self._started else 'stopped'
		return f'DockerSandbox(image={self.image!r}, state={state!r})'

	# ------------------------------------------------------------------
	# Internal helpers
	# ------------------------------------------------------------------

	def _docker(self, args: list, timeout: Optional[int] = 30) -> str:
		result = subprocess.run(
			['docker'] + args,
			capture_output=True,
			text=True,
			timeout=timeout,
		)
		if result.returncode != 0:
			raise RuntimeError(
				f"Docker command failed: docker {' '.join(str(a) for a in args)}\n"
				f'stderr: {result.stderr.strip()}'
			)
		return result.stdout.strip()
