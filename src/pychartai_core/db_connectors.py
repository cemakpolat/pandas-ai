"""db_connectors.py — Database-specific connection adapters.

Provides drop-in connectors for PostgreSQL, MySQL, Snowflake, BigQuery, Redshift.
Each handles authentication, connection string construction, and basic schema inspection.

Usage::

    import pychartai as pai

    # PostgreSQL
    conn = pai.PostgreSQLConnection(
        host='localhost', database='analytics',
        user='postgres', password='secret'
    )
    df = conn.load()
    sdf = pai.SmartDataFrame(df)

    # BigQuery
    conn = pai.BigQueryConnection(
        project_id='my-project', dataset_id='analytics',
        credentials_path='/path/to/service-account.json'
    )
    df = conn.load()
"""

from __future__ import annotations

import os
from typing import List, Optional, Any

import pandas as pd

from .connections import SQLConnection


# ---------------------------------------------------------------------------
# PostgreSQL
# ---------------------------------------------------------------------------

class PostgreSQLConnection(SQLConnection):
    """PostgreSQL connector via psycopg2.

    Args:
        host:       Database server hostname.
        port:       Server port (default 5432).
        database:   Database name.
        user:       Username.
        password:   Password.
        schema:     Schema name (default 'public').
        table:      Table name (mutually exclusive with query=).
        query:      SQL query (mutually exclusive with table=).
        **kwargs:   Forwarded to pd.read_sql().

    Requires: ``pip install pychartai[db-postgres]``
    """

    def __init__(
        self,
        host: str,
        database: str,
        user: str,
        password: str,
        port: int = 5432,
        schema: str = 'public',
        table: Optional[str] = None,
        query: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        connection_string = (
            f'postgresql://{user}:{password}@{host}:{port}/{database}'
        )
        self.schema = schema
        super().__init__(
            connection_string,
            table=table,
            query=query,
            **kwargs,
        )

    def list_tables(self) -> List[str]:
        """List all tables in the schema."""
        query = (
            f"SELECT table_name FROM information_schema.tables "
            f"WHERE table_schema = '{self.schema}' AND table_type = 'BASE TABLE'"
        )
        result = pd.read_sql(query, f'postgresql://{self._conn_str.split("://")[1]}')
        return result['table_name'].tolist() if not result.empty else []


# ---------------------------------------------------------------------------
# MySQL / MariaDB
# ---------------------------------------------------------------------------

class MySQLConnection(SQLConnection):
    """MySQL / MariaDB connector via pymysql.

    Args:
        host:       Database server hostname.
        port:       Server port (default 3306).
        database:   Database name.
        user:       Username.
        password:   Password.
        table:      Table name (mutually exclusive with query=).
        query:      SQL query (mutually exclusive with table=).
        **kwargs:   Forwarded to pd.read_sql().

    Requires: ``pip install pychartai[db-mysql]``
    """

    def __init__(
        self,
        host: str,
        database: str,
        user: str,
        password: str,
        port: int = 3306,
        table: Optional[str] = None,
        query: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        connection_string = (
            f'mysql+pymysql://{user}:{password}@{host}:{port}/{database}'
        )
        super().__init__(
            connection_string,
            table=table,
            query=query,
            **kwargs,
        )

    def list_tables(self) -> List[str]:
        """List all tables in the database."""
        query = (
            f"SELECT TABLE_NAME FROM INFORMATION_SCHEMA.TABLES "
            f"WHERE TABLE_SCHEMA = DATABASE()"
        )
        result = pd.read_sql(query, self._conn_str)
        return result['TABLE_NAME'].tolist() if not result.empty else []


# ---------------------------------------------------------------------------
# Snowflake
# ---------------------------------------------------------------------------

class SnowflakeConnection(SQLConnection):
    """Snowflake connector via snowflake-sqlalchemy.

    Args:
        account:    Snowflake account identifier (e.g. 'xy12345.us-east-1').
        user:       Username.
        password:   Password.
        warehouse:  Warehouse name.
        database:   Database name.
        schema:     Schema name (default 'PUBLIC').
        table:      Table name (mutually exclusive with query=).
        query:      SQL query (mutually exclusive with table=).
        **kwargs:   Forwarded to pd.read_sql().

    Requires: ``pip install pychartai[db-snowflake]``
    """

    def __init__(
        self,
        account: str,
        user: str,
        password: str,
        warehouse: str,
        database: str,
        schema: str = 'PUBLIC',
        table: Optional[str] = None,
        query: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        connection_string = (
            f'snowflake://{user}:{password}@{account}/'
            f'{database}/{schema}?warehouse={warehouse}'
        )
        self.database = database
        self.schema = schema
        super().__init__(
            connection_string,
            table=table,
            query=query,
            **kwargs,
        )

    def list_tables(self) -> List[str]:
        """List all tables in the schema."""
        query = (
            f"SELECT TABLE_NAME FROM INFORMATION_SCHEMA.TABLES "
            f"WHERE TABLE_SCHEMA = '{self.schema}'"
        )
        result = pd.read_sql(query, self._conn_str)
        return result['TABLE_NAME'].tolist() if not result.empty else []


# ---------------------------------------------------------------------------
# Google BigQuery
# ---------------------------------------------------------------------------

class BigQueryConnection(SQLConnection):
    """Google BigQuery connector.

    Args:
        project_id:         GCP project ID.
        dataset_id:         BigQuery dataset ID.
        credentials_path:   Path to service account JSON (optional;
                           uses Application Default Credentials if not provided).
        table:              Table name (mutually exclusive with query=).
        query:              SQL query (mutually exclusive with table=).
        **kwargs:           Forwarded to pd.read_sql().

    Requires: ``pip install pychartai[db-bigquery]``
    """

    def __init__(
        self,
        project_id: str,
        dataset_id: str,
        credentials_path: Optional[str] = None,
        table: Optional[str] = None,
        query: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        if credentials_path and os.path.isfile(credentials_path):
            os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = credentials_path

        connection_string = f'bigquery://{project_id}/{dataset_id}'
        self.project_id = project_id
        self.dataset_id = dataset_id
        super().__init__(
            connection_string,
            table=table,
            query=query,
            **kwargs,
        )

    def list_tables(self) -> List[str]:
        """List all tables in the dataset."""
        query = (
            f"SELECT table_name FROM `{self.project_id}.{self.dataset_id}.__TABLES__`"
        )
        result = pd.read_sql(query, self._conn_str)
        return result['table_name'].tolist() if not result.empty else []


# ---------------------------------------------------------------------------
# Amazon Redshift
# ---------------------------------------------------------------------------

class RedshiftConnection(SQLConnection):
    """Amazon Redshift connector via redshift-connector or psycopg2.

    Args:
        host:       Redshift cluster endpoint hostname.
        port:       Server port (default 5439).
        database:   Database name.
        user:       Username.
        password:   Password.
        cluster:    Cluster identifier (optional, for reference).
        schema:     Schema name (default 'public').
        table:      Table name (mutually exclusive with query=).
        query:      SQL query (mutually exclusive with table=).
        **kwargs:   Forwarded to pd.read_sql().

    Requires: ``pip install pychartai[db-redshift]``
    """

    def __init__(
        self,
        host: str,
        database: str,
        user: str,
        password: str,
        port: int = 5439,
        cluster: Optional[str] = None,
        schema: str = 'public',
        table: Optional[str] = None,
        query: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        # Redshift is Postgres-compatible, so we use psycopg2
        connection_string = (
            f'postgresql://{user}:{password}@{host}:{port}/{database}'
        )
        self.schema = schema
        self.cluster = cluster
        super().__init__(
            connection_string,
            table=table,
            query=query,
            **kwargs,
        )

    def list_tables(self) -> List[str]:
        """List all tables in the schema."""
        query = (
            f"SELECT table_name FROM information_schema.tables "
            f"WHERE table_schema = '{self.schema}' AND table_type = 'BASE TABLE'"
        )
        result = pd.read_sql(query, self._conn_str)
        return result['table_name'].tolist() if not result.empty else []
