"""cloud_connectors.py — Cloud storage and SaaS data source connectors.

Load data files directly from cloud object storage (S3, GCS, Azure Blob) and
SaaS sources (Google Sheets). Each connector auto-detects the file format
(CSV / Parquet / JSON / Excel) from the object key/extension, or you can force
it with ``file_format=``.

Usage::

    import pychartai as pai

    # AWS S3 — pip install pychartai[cloud-s3]
    conn = pai.S3Connection('s3://my-bucket/data/sales.csv')
    sdf = pai.SmartDataFrame(conn.load())

    # Google Cloud Storage — pip install pychartai[cloud-gcs]
    conn = pai.GCSConnection('gs://my-bucket/data/events.parquet')

    # Azure Blob Storage — pip install pychartai[cloud-azure]
    conn = pai.AzureBlobConnection(
        account_url='https://acct.blob.core.windows.net',
        container='data', blob='sales.csv',
    )

    # Google Sheets — pip install pychartai[cloud-gsheets]
    conn = pai.GoogleSheetsConnection(
        spreadsheet_id='1AbC...', sheet_name='Sheet1',
        credentials_path='/path/to/service-account.json',
    )
"""

from __future__ import annotations

import io
import os
from typing import Any, Optional

import pandas as pd

from .connections import BaseConnection


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_FORMAT_READERS = {
    'csv': pd.read_csv,
    'parquet': pd.read_parquet,
    'json': pd.read_json,
    'xlsx': pd.read_excel,
    'xls': pd.read_excel,
}


def _infer_format(key: str, override: Optional[str]) -> str:
    """Determine the file format from an extension or explicit override."""
    if override:
        fmt = override.lower().lstrip('.')
    else:
        fmt = key.rsplit('.', 1)[-1].lower() if '.' in key else ''
    if fmt not in _FORMAT_READERS:
        raise ValueError(
            f'Unsupported or undetected file format {fmt!r} for {key!r}. '
            f'Pass file_format= explicitly. Supported: {sorted(_FORMAT_READERS)}'
        )
    return fmt


def _read_bytes(data: bytes, fmt: str, **kwargs: Any) -> pd.DataFrame:
    """Read raw bytes into a DataFrame using the format-appropriate reader."""
    reader = _FORMAT_READERS[fmt]
    if fmt in ('xlsx', 'xls'):
        return reader(io.BytesIO(data), **kwargs)
    if fmt == 'parquet':
        return reader(io.BytesIO(data), **kwargs)
    # csv / json accept a text or bytes buffer
    return reader(io.BytesIO(data), **kwargs)


# ---------------------------------------------------------------------------
# AWS S3
# ---------------------------------------------------------------------------

class S3Connection(BaseConnection):
    """Load a data file from an AWS S3 bucket.

    Args:
        uri:            ``s3://bucket/key`` URI, or pass *bucket* and *key* separately.
        bucket:         Bucket name (if not using a full URI).
        key:            Object key / path within the bucket.
        file_format:    Force the file format ('csv', 'parquet', 'json', 'xlsx').
        aws_access_key_id:     Optional explicit credentials (else env / IAM role).
        aws_secret_access_key: Optional explicit secret.
        region_name:    AWS region.
        endpoint_url:   Override the S3 endpoint — point at an S3-compatible store
                        (MinIO, Cloudflare R2, Wasabi, …) or a local emulator.
        **kwargs:       Forwarded to the pandas reader.

    Requires: ``pip install pychartai[cloud-s3]``
    """

    def __init__(
        self,
        uri: Optional[str] = None,
        *,
        bucket: Optional[str] = None,
        key: Optional[str] = None,
        file_format: Optional[str] = None,
        aws_access_key_id: Optional[str] = None,
        aws_secret_access_key: Optional[str] = None,
        region_name: Optional[str] = None,
        endpoint_url: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        if uri:
            if not uri.startswith('s3://'):
                raise ValueError(f"S3 URI must start with 's3://'. Got: {uri!r}")
            path = uri[len('s3://'):]
            bucket, _, key = path.partition('/')
        if not bucket or not key:
            raise ValueError('Provide either uri="s3://bucket/key" or bucket= and key=.')
        self._bucket = bucket
        self._key = key
        self._fmt = _infer_format(key, file_format)
        self._creds = {
            'aws_access_key_id': aws_access_key_id,
            'aws_secret_access_key': aws_secret_access_key,
            'region_name': region_name,
            'endpoint_url': endpoint_url,
        }
        self._kwargs = kwargs

    def load(self) -> pd.DataFrame:
        try:
            import boto3  # type: ignore[import]
        except ImportError as exc:
            raise ImportError(
                'S3Connection requires boto3. Install: pip install pychartai[cloud-s3]'
            ) from exc
        creds = {k: v for k, v in self._creds.items() if v is not None}
        client = boto3.client('s3', **creds)
        obj = client.get_object(Bucket=self._bucket, Key=self._key)
        data = obj['Body'].read()
        return _read_bytes(data, self._fmt, **self._kwargs)

    def __repr__(self) -> str:
        return f'S3Connection(s3://{self._bucket}/{self._key})'


# ---------------------------------------------------------------------------
# Google Cloud Storage
# ---------------------------------------------------------------------------

class GCSConnection(BaseConnection):
    """Load a data file from Google Cloud Storage.

    Args:
        uri:                ``gs://bucket/blob`` URI, or pass *bucket* and *blob*.
        bucket:             Bucket name (if not using a full URI).
        blob:               Object path within the bucket.
        file_format:        Force the file format.
        credentials_path:   Path to a service-account JSON (else ADC).
        project:            GCP project id (optional).
        **kwargs:           Forwarded to the pandas reader.

    Requires: ``pip install pychartai[cloud-gcs]``
    """

    def __init__(
        self,
        uri: Optional[str] = None,
        *,
        bucket: Optional[str] = None,
        blob: Optional[str] = None,
        file_format: Optional[str] = None,
        credentials_path: Optional[str] = None,
        project: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        if uri:
            if not uri.startswith('gs://'):
                raise ValueError(f"GCS URI must start with 'gs://'. Got: {uri!r}")
            path = uri[len('gs://'):]
            bucket, _, blob = path.partition('/')
        if not bucket or not blob:
            raise ValueError('Provide either uri="gs://bucket/blob" or bucket= and blob=.')
        self._bucket = bucket
        self._blob = blob
        self._fmt = _infer_format(blob, file_format)
        self._credentials_path = credentials_path
        self._project = project
        self._kwargs = kwargs

    def load(self) -> pd.DataFrame:
        try:
            from google.cloud import storage  # type: ignore[import]
        except ImportError as exc:
            raise ImportError(
                'GCSConnection requires google-cloud-storage. '
                'Install: pip install pychartai[cloud-gcs]'
            ) from exc

        emulator_host = os.environ.get('STORAGE_EMULATOR_HOST')
        if emulator_host:
            # Local emulator (e.g. fake-gcs-server) — use anonymous credentials
            # and point the client at the emulator endpoint.
            from google.auth.credentials import AnonymousCredentials  # type: ignore[import]
            client = storage.Client(
                project=self._project or 'test',
                credentials=AnonymousCredentials(),
                client_options={'api_endpoint': emulator_host},
            )
        elif self._credentials_path and os.path.isfile(self._credentials_path):
            client = storage.Client.from_service_account_json(
                self._credentials_path, project=self._project
            )
        else:
            client = storage.Client(project=self._project)
        bucket = client.bucket(self._bucket)
        blob = bucket.blob(self._blob)
        data = blob.download_as_bytes()
        return _read_bytes(data, self._fmt, **self._kwargs)

    def __repr__(self) -> str:
        return f'GCSConnection(gs://{self._bucket}/{self._blob})'


# ---------------------------------------------------------------------------
# Azure Blob Storage
# ---------------------------------------------------------------------------

class AzureBlobConnection(BaseConnection):
    """Load a data file from Azure Blob Storage.

    Args:
        account_url:      Storage account URL (e.g. ``https://acct.blob.core.windows.net``).
        container:        Container name.
        blob:             Blob path within the container.
        file_format:      Force the file format.
        connection_string: Full Azure connection string (alternative to account_url + credential).
        credential:       Account key or SAS token (else DefaultAzureCredential).
        **kwargs:         Forwarded to the pandas reader.

    Requires: ``pip install pychartai[cloud-azure]``
    """

    def __init__(
        self,
        container: str,
        blob: str,
        *,
        account_url: Optional[str] = None,
        connection_string: Optional[str] = None,
        credential: Optional[str] = None,
        file_format: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        if not account_url and not connection_string:
            raise ValueError('Provide either account_url= or connection_string=.')
        self._container = container
        self._blob = blob
        self._fmt = _infer_format(blob, file_format)
        self._account_url = account_url
        self._connection_string = connection_string
        self._credential = credential
        self._kwargs = kwargs

    def load(self) -> pd.DataFrame:
        try:
            from azure.storage.blob import BlobServiceClient  # type: ignore[import]
        except ImportError as exc:
            raise ImportError(
                'AzureBlobConnection requires azure-storage-blob. '
                'Install: pip install pychartai[cloud-azure]'
            ) from exc
        if self._connection_string:
            service = BlobServiceClient.from_connection_string(self._connection_string)
        else:
            service = BlobServiceClient(
                account_url=self._account_url,
                credential=self._credential,
            )
        blob_client = service.get_blob_client(container=self._container, blob=self._blob)
        data = blob_client.download_blob().readall()
        return _read_bytes(data, self._fmt, **self._kwargs)

    def __repr__(self) -> str:
        return f'AzureBlobConnection({self._container}/{self._blob})'


# ---------------------------------------------------------------------------
# Google Sheets
# ---------------------------------------------------------------------------

class GoogleSheetsConnection(BaseConnection):
    """Load a worksheet from a Google Sheets spreadsheet.

    Args:
        spreadsheet_id:    The spreadsheet ID (from its URL).
        sheet_name:        Worksheet/tab name (default: first sheet).
        credentials_path:  Path to a service-account JSON with Sheets access.
        header_row:        Row index (0-based) to use as column headers (default 0).
        **kwargs:          Forwarded to ``pd.DataFrame`` construction (ignored for now).

    Requires: ``pip install pychartai[cloud-gsheets]``
    """

    _SCOPES = ['https://www.googleapis.com/auth/spreadsheets.readonly']

    def __init__(
        self,
        spreadsheet_id: str,
        *,
        sheet_name: Optional[str] = None,
        credentials_path: Optional[str] = None,
        header_row: int = 0,
        **kwargs: Any,
    ) -> None:
        if not spreadsheet_id:
            raise ValueError('spreadsheet_id is required.')
        self._spreadsheet_id = spreadsheet_id
        self._sheet_name = sheet_name
        self._credentials_path = credentials_path
        self._header_row = header_row
        self._kwargs = kwargs

    def load(self) -> pd.DataFrame:
        try:
            from google.oauth2.service_account import Credentials  # type: ignore[import]
            from googleapiclient.discovery import build  # type: ignore[import]
        except ImportError as exc:
            raise ImportError(
                'GoogleSheetsConnection requires google-api-python-client and '
                'google-auth. Install: pip install pychartai[cloud-gsheets]'
            ) from exc

        if not (self._credentials_path and os.path.isfile(self._credentials_path)):
            raise ValueError(
                'GoogleSheetsConnection requires credentials_path to a valid '
                'service-account JSON file.'
            )
        creds = Credentials.from_service_account_file(
            self._credentials_path, scopes=self._SCOPES
        )
        service = build('sheets', 'v4', credentials=creds)
        rng = self._sheet_name if self._sheet_name else 'A:ZZ'
        result = (
            service.spreadsheets()
            .values()
            .get(spreadsheetId=self._spreadsheet_id, range=rng)
            .execute()
        )
        values = result.get('values', [])
        if not values:
            return pd.DataFrame()
        header = values[self._header_row]
        rows = values[self._header_row + 1:]
        # Pad short rows so every row has len(header) columns
        padded = [r + [None] * (len(header) - len(r)) for r in rows]
        return pd.DataFrame(padded, columns=header)

    def __repr__(self) -> str:
        return f'GoogleSheetsConnection({self._spreadsheet_id!r}, sheet={self._sheet_name!r})'
