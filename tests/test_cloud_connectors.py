"""tests/test_cloud_connectors.py — Tests for cloud storage / SaaS connectors."""

import io
import pytest
import pandas as pd

from pychartai_core.cloud_connectors import (
    S3Connection,
    GCSConnection,
    AzureBlobConnection,
    GoogleSheetsConnection,
    _infer_format,
    _read_bytes,
)


# ---------------------------------------------------------------------------
# Format inference helpers
# ---------------------------------------------------------------------------

class TestInferFormat:

    def test_csv_extension(self):
        assert _infer_format('data/sales.csv', None) == 'csv'

    def test_parquet_extension(self):
        assert _infer_format('events.parquet', None) == 'parquet'

    def test_json_extension(self):
        assert _infer_format('records.json', None) == 'json'

    def test_xlsx_extension(self):
        assert _infer_format('book.xlsx', None) == 'xlsx'

    def test_override_takes_precedence(self):
        assert _infer_format('data.bin', 'csv') == 'csv'

    def test_override_with_leading_dot(self):
        assert _infer_format('data', '.parquet') == 'parquet'

    def test_unknown_format_raises(self):
        with pytest.raises(ValueError, match='[Uu]nsupported|undetected'):
            _infer_format('data.bin', None)

    def test_no_extension_raises(self):
        with pytest.raises(ValueError):
            _infer_format('datafile', None)


class TestReadBytes:

    def test_read_csv_bytes(self):
        data = b'a,b\n1,2\n3,4\n'
        df = _read_bytes(data, 'csv')
        assert list(df.columns) == ['a', 'b']
        assert len(df) == 2

    def test_read_json_bytes(self):
        data = b'[{"x": 1, "y": 2}, {"x": 3, "y": 4}]'
        df = _read_bytes(data, 'json')
        assert list(df.columns) == ['x', 'y']
        assert len(df) == 2


# ---------------------------------------------------------------------------
# S3
# ---------------------------------------------------------------------------

class TestS3Connection:

    def test_uri_parsing(self):
        conn = S3Connection('s3://my-bucket/data/sales.csv')
        assert conn._bucket == 'my-bucket'
        assert conn._key == 'data/sales.csv'
        assert conn._fmt == 'csv'

    def test_bucket_key_separately(self):
        conn = S3Connection(bucket='my-bucket', key='events.parquet')
        assert conn._bucket == 'my-bucket'
        assert conn._key == 'events.parquet'
        assert conn._fmt == 'parquet'

    def test_invalid_uri_scheme_raises(self):
        with pytest.raises(ValueError, match='s3://'):
            S3Connection('http://my-bucket/data.csv')

    def test_missing_bucket_raises(self):
        with pytest.raises(ValueError, match='uri|bucket'):
            S3Connection(key='data.csv')

    def test_file_format_override(self):
        conn = S3Connection(bucket='b', key='data', file_format='csv')
        assert conn._fmt == 'csv'

    def test_repr(self):
        conn = S3Connection('s3://b/k.csv')
        assert 's3://b/k.csv' in repr(conn)

    def test_endpoint_url_stored(self):
        # endpoint_url enables S3-compatible stores (MinIO, R2, Wasabi)
        conn = S3Connection(
            's3://b/k.csv',
            endpoint_url='http://localhost:9000',
        )
        assert conn._creds['endpoint_url'] == 'http://localhost:9000'

    def test_endpoint_url_passed_to_boto3(self):
        import sys
        from unittest.mock import MagicMock

        mock_boto3 = MagicMock()
        mock_body = MagicMock()
        mock_body.read.return_value = b'a,b\n1,2\n'
        mock_boto3.client.return_value.get_object.return_value = {'Body': mock_body}
        sys.modules['boto3'] = mock_boto3
        try:
            conn = S3Connection(
                's3://bucket/data.csv',
                endpoint_url='http://localhost:9000',
                aws_access_key_id='k', aws_secret_access_key='s',
            )
            conn.load()
            _, kwargs = mock_boto3.client.call_args
            assert kwargs['endpoint_url'] == 'http://localhost:9000'
        finally:
            del sys.modules['boto3']

    def test_load_uses_boto3(self):
        # Mock boto3 so we don't hit AWS
        import sys
        from unittest.mock import MagicMock

        mock_boto3 = MagicMock()
        mock_body = MagicMock()
        mock_body.read.return_value = b'a,b\n1,2\n'
        mock_boto3.client.return_value.get_object.return_value = {'Body': mock_body}
        sys.modules['boto3'] = mock_boto3
        try:
            conn = S3Connection('s3://bucket/data.csv')
            df = conn.load()
            assert list(df.columns) == ['a', 'b']
        finally:
            del sys.modules['boto3']


# ---------------------------------------------------------------------------
# GCS
# ---------------------------------------------------------------------------

class TestGCSConnection:

    def test_uri_parsing(self):
        conn = GCSConnection('gs://my-bucket/data/events.parquet')
        assert conn._bucket == 'my-bucket'
        assert conn._blob == 'data/events.parquet'
        assert conn._fmt == 'parquet'

    def test_bucket_blob_separately(self):
        conn = GCSConnection(bucket='b', blob='sales.csv')
        assert conn._bucket == 'b'
        assert conn._blob == 'sales.csv'

    def test_invalid_uri_scheme_raises(self):
        with pytest.raises(ValueError, match='gs://'):
            GCSConnection('s3://my-bucket/data.csv')

    def test_missing_blob_raises(self):
        with pytest.raises(ValueError, match='uri|bucket'):
            GCSConnection(bucket='b')

    def test_repr(self):
        conn = GCSConnection('gs://b/k.json')
        assert 'gs://b/k.json' in repr(conn)


# ---------------------------------------------------------------------------
# Azure Blob
# ---------------------------------------------------------------------------

class TestAzureBlobConnection:

    def test_account_url_construction(self):
        conn = AzureBlobConnection(
            container='data', blob='sales.csv',
            account_url='https://acct.blob.core.windows.net',
        )
        assert conn._container == 'data'
        assert conn._blob == 'sales.csv'
        assert conn._fmt == 'csv'

    def test_connection_string_alternative(self):
        conn = AzureBlobConnection(
            container='data', blob='events.parquet',
            connection_string='DefaultEndpointsProtocol=https;AccountName=...',
        )
        assert conn._fmt == 'parquet'

    def test_missing_auth_raises(self):
        with pytest.raises(ValueError, match='account_url|connection_string'):
            AzureBlobConnection(container='data', blob='sales.csv')

    def test_file_format_override(self):
        conn = AzureBlobConnection(
            container='c', blob='data', file_format='json',
            account_url='https://acct.blob.core.windows.net',
        )
        assert conn._fmt == 'json'

    def test_repr(self):
        conn = AzureBlobConnection(
            container='data', blob='sales.csv',
            account_url='https://acct.blob.core.windows.net',
        )
        assert 'data/sales.csv' in repr(conn)


# ---------------------------------------------------------------------------
# Google Sheets
# ---------------------------------------------------------------------------

class TestGoogleSheetsConnection:

    def test_construction(self):
        conn = GoogleSheetsConnection(
            spreadsheet_id='1AbC123',
            sheet_name='Sheet1',
            credentials_path='/path/to/creds.json',
        )
        assert conn._spreadsheet_id == '1AbC123'
        assert conn._sheet_name == 'Sheet1'

    def test_missing_spreadsheet_id_raises(self):
        with pytest.raises(ValueError, match='spreadsheet_id'):
            GoogleSheetsConnection(spreadsheet_id='')

    def test_load_requires_credentials_file(self):
        conn = GoogleSheetsConnection(
            spreadsheet_id='1AbC123',
            credentials_path='/nonexistent/creds.json',
        )
        # google libs may not be installed; either ImportError or ValueError is fine
        with pytest.raises((ValueError, ImportError)):
            conn.load()

    def test_repr(self):
        conn = GoogleSheetsConnection(spreadsheet_id='1AbC123', sheet_name='Tab')
        assert '1AbC123' in repr(conn)
