"""
connections.py — Data source connectors.

A :class:`BaseConnection` knows how to load data from an external source and
produce a ``pandas.DataFrame``.  Use :func:`connect` (or ``pai.connect()``)
to load data and wrap it in a :class:`~pychartai_core.smart_df.SmartDataFrame`.

Provided connectors:

===================  ======================================================
Connector            Source
===================  ======================================================
CSVConnection        Local CSV file (``pd.read_csv``)
ExcelConnection      Local Excel file (``pd.read_excel``)
JSONConnection       Local JSON file (``pd.read_json``)
ParquetConnection    Local Parquet file (``pd.read_parquet``)
SQLConnection        Any SQLAlchemy-compatible database (requires sqlalchemy)
===================  ======================================================

Usage::

    import pychartai as pai

    # CSV
    sdf = pai.connect(pai.CSVConnection('data/sales.csv'))

    # Excel
    sdf = pai.connect(pai.ExcelConnection('data/sales.xlsx', sheet_name='Q1'))

    # SQLite (requires sqlalchemy)
    sdf = pai.connect(
        pai.SQLConnection('sqlite:///mydb.sqlite', table='sales')
    )

    # Attach a schema to a connection
    conn = pai.CSVConnection('data/sales.csv')
    schema = pai.Schema({'revenue': pai.Column(unit='USD')})
    sdf = pai.connect(conn, schema=schema)
    sdf.chat('which region has the highest revenue?')
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional, TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
	from .smart_df import SmartDataFrame
	from .schema import Schema


class BaseConnection(ABC):
	"""Abstract base for all data source connectors.

	Subclass and implement :meth:`load` to return a ``pd.DataFrame``.
	"""

	@abstractmethod
	def load(self) -> pd.DataFrame:
		"""Load and return the data as a pandas DataFrame."""

	def __repr__(self) -> str:
		return f'{type(self).__name__}()'


class CSVConnection(BaseConnection):
	"""Load data from a CSV file.

	Args:
		path:    Path to the CSV file.
		**kwargs: Forwarded to ``pd.read_csv``.
	"""

	def __init__(self, path: str, **kwargs: Any) -> None:
		self._path = path
		self._kwargs = kwargs

	def load(self) -> pd.DataFrame:
		return pd.read_csv(self._path, **self._kwargs)

	def __repr__(self) -> str:
		return f'CSVConnection({self._path!r})'


class ExcelConnection(BaseConnection):
	"""Load data from an Excel workbook.

	Args:
		path:       Path to the ``.xlsx`` / ``.xls`` file.
		sheet_name: Sheet name or index (default ``0``).
		**kwargs:   Forwarded to ``pd.read_excel``.
	"""

	def __init__(self, path: str, sheet_name: Any = 0, **kwargs: Any) -> None:
		self._path = path
		self._sheet = sheet_name
		self._kwargs = kwargs

	def load(self) -> pd.DataFrame:
		return pd.read_excel(self._path, sheet_name=self._sheet, **self._kwargs)

	def __repr__(self) -> str:
		return f'ExcelConnection({self._path!r}, sheet_name={self._sheet!r})'


class JSONConnection(BaseConnection):
	"""Load data from a JSON file.

	Args:
		path:    Path to the JSON file.
		**kwargs: Forwarded to ``pd.read_json``.
	"""

	def __init__(self, path: str, **kwargs: Any) -> None:
		self._path = path
		self._kwargs = kwargs

	def load(self) -> pd.DataFrame:
		return pd.read_json(self._path, **self._kwargs)

	def __repr__(self) -> str:
		return f'JSONConnection({self._path!r})'


class ParquetConnection(BaseConnection):
	"""Load data from a Parquet file.

	Args:
		path:    Path to the Parquet file.
		**kwargs: Forwarded to ``pd.read_parquet``.
	"""

	def __init__(self, path: str, **kwargs: Any) -> None:
		self._path = path
		self._kwargs = kwargs

	def load(self) -> pd.DataFrame:
		return pd.read_parquet(self._path, **self._kwargs)

	def __repr__(self) -> str:
		return f'ParquetConnection({self._path!r})'


class SQLConnection(BaseConnection):
	"""Load data from an SQL table/query via SQLAlchemy.

	Requires ``sqlalchemy`` to be installed::

		pip install sqlalchemy

	For SQLite no additional driver is needed.  For other databases install
	the appropriate dialect (e.g. ``psycopg2`` for PostgreSQL).

	Args:
		connection_string: SQLAlchemy connection string
		                   (e.g. ``'sqlite:///mydb.sqlite'``,
		                   ``'postgresql://user:pw@host/db'``).
		query:             Raw SQL query.  Mutually exclusive with *table*.
		table:             Table name; generates ``SELECT * FROM <table>``.
		**kwargs:          Forwarded to ``pd.read_sql``.

	Raises:
		ImportError:  If ``sqlalchemy`` is not installed.
		ValueError:   If neither *query* nor *table* is provided.
	"""

	def __init__(
		self,
		connection_string: str,
		*,
		query: Optional[str] = None,
		table: Optional[str] = None,
		**kwargs: Any,
	) -> None:
		if query is None and table is None:
			raise ValueError('Provide either query= or table= for SQLConnection.')
		self._conn_str = connection_string
		self._query = query if query is not None else f'SELECT * FROM "{table}"'
		self._kwargs = kwargs

	def load(self) -> pd.DataFrame:
		try:
			from sqlalchemy import create_engine  # type: ignore[import]
		except ImportError as exc:
			raise ImportError(
				'SQLConnection requires sqlalchemy. Install it with: pip install sqlalchemy'
			) from exc
		engine = create_engine(self._conn_str)
		return pd.read_sql(self._query, engine, **self._kwargs)

	def __repr__(self) -> str:
		return f'SQLConnection({self._conn_str!r}, query={self._query!r})'


# ---------------------------------------------------------------------------
# connect() helper
# ---------------------------------------------------------------------------

def connect(
	connection: BaseConnection,
	*,
	schema: Optional['Schema'] = None,
	chart_library: Optional[str] = None,
	config: Optional[dict] = None,
) -> 'SmartDataFrame':
	"""Load data from *connection* and wrap it in a :class:`SmartDataFrame`.

	Args:
		connection:   A :class:`BaseConnection` instance.
		schema:       Optional :class:`~pychartai_core.schema.Schema` to
		              attach for semantic context.
		chart_library: Default chart backend (``'seaborn'``, ``'matplotlib'``,
		               or ``'plotly'``).
		config:       Optional config overrides (same as :func:`pai.config.set`).

	Returns:
		A :class:`~pychartai_core.smart_df.SmartDataFrame` ready for
		``.chat()`` calls.
	"""
	from .smart_df import SmartDataFrame  # avoid circular import at module level

	df = connection.load()
	sdf = SmartDataFrame(df)

	if schema is not None:
		sdf.set_schema(schema)

	if chart_library is not None or config is not None:
		from .config import config as global_config
		merged: dict = {}
		if chart_library is not None:
			merged['chart_backend'] = chart_library
		if config:
			merged.update(config)
		if merged:
			global_config.set(merged)

	return sdf
