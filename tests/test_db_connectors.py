"""tests/test_db_connectors.py — Tests for database connector classes."""

import pytest
import pandas as pd
from unittest.mock import patch, MagicMock

from pychartai_core.db_connectors import (
    PostgreSQLConnection,
    MySQLConnection,
    SnowflakeConnection,
    BigQueryConnection,
    RedshiftConnection,
)


class TestPostgreSQLConnection:

    def test_connection_string_construction(self):
        conn = PostgreSQLConnection(
            host='localhost',
            database='mydb',
            user='postgres',
            password='secret',
            schema='public',
            table='users',
        )
        assert 'postgresql://' in conn._conn_str
        assert 'postgres:secret@localhost' in conn._conn_str
        assert 'mydb' in conn._conn_str
        assert conn.schema == 'public'

    def test_with_custom_port(self):
        conn = PostgreSQLConnection(
            host='db.example.com',
            database='analytics',
            user='admin',
            password='pwd',
            port=5433,
            table='sales',
        )
        assert ':5433' in conn._conn_str

    def test_table_parameter(self):
        conn = PostgreSQLConnection(
            host='localhost',
            database='mydb',
            user='postgres',
            password='secret',
            table='sales',
        )
        assert 'sales' in conn._query.lower()

    def test_query_parameter(self):
        conn = PostgreSQLConnection(
            host='localhost',
            database='mydb',
            user='postgres',
            password='secret',
            query='SELECT * FROM sales WHERE year = 2024',
        )
        assert 'WHERE year = 2024' in conn._query


class TestMySQLConnection:

    def test_connection_string_construction(self):
        conn = MySQLConnection(
            host='localhost',
            database='mydb',
            user='root',
            password='secret',
            table='users',
        )
        assert 'mysql+pymysql://' in conn._conn_str
        assert 'root:secret@localhost' in conn._conn_str
        assert 'mydb' in conn._conn_str

    def test_custom_port(self):
        conn = MySQLConnection(
            host='db.example.com',
            database='analytics',
            user='admin',
            password='pwd',
            port=3307,
            table='sales',
        )
        assert ':3307' in conn._conn_str

    def test_table_parameter(self):
        conn = MySQLConnection(
            host='localhost',
            database='mydb',
            user='root',
            password='secret',
            table='users',
        )
        assert 'users' in conn._query.lower()


class TestSnowflakeConnection:

    def test_connection_string_construction(self):
        conn = SnowflakeConnection(
            account='xy12345.us-east-1',
            user='admin',
            password='secret',
            warehouse='COMPUTE_WH',
            database='ANALYTICS',
            schema='PUBLIC',
            table='SALES',
        )
        assert 'snowflake://' in conn._conn_str
        assert 'admin:secret@xy12345.us-east-1' in conn._conn_str
        assert 'ANALYTICS' in conn._conn_str
        assert 'COMPUTE_WH' in conn._conn_str

    def test_custom_schema(self):
        conn = SnowflakeConnection(
            account='xy12345.us-east-1',
            user='admin',
            password='secret',
            warehouse='COMPUTE_WH',
            database='ANALYTICS',
            schema='STAGING',
            table='SALES',
        )
        assert conn.schema == 'STAGING'

    def test_table_parameter(self):
        conn = SnowflakeConnection(
            account='xy12345.us-east-1',
            user='admin',
            password='secret',
            warehouse='COMPUTE_WH',
            database='ANALYTICS',
            table='SALES',
        )
        assert 'SALES' in conn._query.upper()


class TestBigQueryConnection:

    def test_connection_string_construction(self):
        conn = BigQueryConnection(
            project_id='my-gcp-project',
            dataset_id='analytics',
            table='sales',
        )
        assert 'bigquery://' in conn._conn_str
        assert 'my-gcp-project' in conn._conn_str
        assert 'analytics' in conn._conn_str
        assert conn.project_id == 'my-gcp-project'
        assert conn.dataset_id == 'analytics'

    def test_credentials_path_not_required(self):
        # Should not raise even without credentials_path
        # (assumes Application Default Credentials)
        conn = BigQueryConnection(
            project_id='my-project',
            dataset_id='analytics',
            table='sales',
        )
        assert conn.project_id == 'my-project'

    def test_table_parameter(self):
        conn = BigQueryConnection(
            project_id='my-project',
            dataset_id='analytics',
            table='sales',
        )
        assert 'sales' in conn._query.lower()

    def test_credentials_path_sets_env(self):
        import os as _os
        with patch.dict('os.environ', {}, clear=False):
            with patch('pychartai_core.db_connectors.os.path.isfile', return_value=True):
                BigQueryConnection(
                    project_id='my-project',
                    dataset_id='analytics',
                    credentials_path='/path/to/creds.json',
                    table='sales',
                )
                assert _os.environ.get('GOOGLE_APPLICATION_CREDENTIALS') == '/path/to/creds.json'


class TestRedshiftConnection:

    def test_connection_string_construction(self):
        conn = RedshiftConnection(
            host='redshift.us-east-1.redshift.amazonaws.com',
            database='mydb',
            user='admin',
            password='secret',
            table='sales',
        )
        assert 'postgresql://' in conn._conn_str  # Redshift uses psycopg2
        assert 'admin:secret@' in conn._conn_str
        assert 'redshift.us-east-1.redshift.amazonaws.com' in conn._conn_str
        assert ':5439' in conn._conn_str  # default port

    def test_custom_port(self):
        conn = RedshiftConnection(
            host='redshift.us-east-1.redshift.amazonaws.com',
            database='analytics',
            user='admin',
            password='pwd',
            port=5440,
            table='sales',
        )
        assert ':5440' in conn._conn_str

    def test_cluster_parameter(self):
        conn = RedshiftConnection(
            host='redshift.us-east-1.redshift.amazonaws.com',
            database='mydb',
            user='admin',
            password='secret',
            cluster='my-cluster-1',
            table='sales',
        )
        assert conn.cluster == 'my-cluster-1'

    def test_table_parameter(self):
        conn = RedshiftConnection(
            host='redshift.us-east-1.redshift.amazonaws.com',
            database='mydb',
            user='admin',
            password='secret',
            table='sales',
        )
        assert 'sales' in conn._query.lower()


class TestConnectionValidation:

    def test_postgres_requires_table_or_query(self):
        with pytest.raises(ValueError, match='query|table'):
            PostgreSQLConnection(
                host='localhost',
                database='mydb',
                user='postgres',
                password='secret',
            )

    def test_mysql_requires_table_or_query(self):
        with pytest.raises(ValueError, match='query|table'):
            MySQLConnection(
                host='localhost',
                database='mydb',
                user='root',
                password='secret',
            )

    def test_snowflake_requires_table_or_query(self):
        with pytest.raises(ValueError, match='query|table'):
            SnowflakeConnection(
                account='xy12345.us-east-1',
                user='admin',
                password='secret',
                warehouse='COMPUTE_WH',
                database='ANALYTICS',
            )

    def test_bigquery_requires_table_or_query(self):
        with pytest.raises(ValueError, match='query|table'):
            BigQueryConnection(
                project_id='my-project',
                dataset_id='analytics',
            )

    def test_redshift_requires_table_or_query(self):
        with pytest.raises(ValueError, match='query|table'):
            RedshiftConnection(
                host='redshift.us-east-1.redshift.amazonaws.com',
                database='mydb',
                user='admin',
                password='secret',
            )
