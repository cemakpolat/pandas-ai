"""tests/integration/test_cloud_integration.py

Integration tests for cloud connectors against local emulators.

These tests are SKIPPED by default. To run them:

    docker compose -f docker-compose.cloud-test.yml up -d
    PYCHARTAI_CLOUD_IT=1 pytest tests/integration/test_cloud_integration.py -v
    docker compose -f docker-compose.cloud-test.yml down -v

Emulators:
    MinIO     (S3-compatible)   http://localhost:9000
    Azurite   (Azure Blob)      http://localhost:10000
    fake-gcs  (Google Storage)  http://localhost:4443
"""

import io
import os

import pandas as pd
import pytest

from pychartai_core.cloud_connectors import (
    S3Connection,
    GCSConnection,
    AzureBlobConnection,
)

# Skip the entire module unless explicitly enabled
pytestmark = pytest.mark.skipif(
    os.environ.get('PYCHARTAI_CLOUD_IT') != '1',
    reason='Cloud integration tests disabled. Set PYCHARTAI_CLOUD_IT=1 and start emulators.',
)

# Shared sample data — written to each emulator, then read back and compared
SAMPLE_CSV = b'region,revenue,units\nNorth,1000,50\nSouth,800,40\nEast,1200,60\n'
EXPECTED_ROWS = 3
EXPECTED_COLS = ['region', 'revenue', 'units']


# ---------------------------------------------------------------------------
# MinIO (S3)
# ---------------------------------------------------------------------------

S3_ENDPOINT = os.environ.get('PYCHARTAI_S3_ENDPOINT', 'http://localhost:9000')
S3_KEY_ID = os.environ.get('PYCHARTAI_S3_KEY', 'minioadmin')
S3_SECRET = os.environ.get('PYCHARTAI_S3_SECRET', 'minioadmin')


@pytest.fixture(scope='module')
def s3_seeded():
    """Create a bucket and upload the sample CSV to MinIO."""
    boto3 = pytest.importorskip('boto3')
    bucket, key = 'pychartai-it', 'data/sales.csv'
    client = boto3.client(
        's3',
        endpoint_url=S3_ENDPOINT,
        aws_access_key_id=S3_KEY_ID,
        aws_secret_access_key=S3_SECRET,
        region_name='us-east-1',
    )
    try:
        client.create_bucket(Bucket=bucket)
    except Exception:
        pass  # bucket may already exist
    client.put_object(Bucket=bucket, Key=key, Body=SAMPLE_CSV)
    return bucket, key


class TestS3Integration:

    def test_load_csv_from_minio(self, s3_seeded):
        bucket, key = s3_seeded
        conn = S3Connection(
            bucket=bucket, key=key,
            endpoint_url=S3_ENDPOINT,
            aws_access_key_id=S3_KEY_ID,
            aws_secret_access_key=S3_SECRET,
            region_name='us-east-1',
        )
        df = conn.load()
        assert list(df.columns) == EXPECTED_COLS
        assert len(df) == EXPECTED_ROWS
        assert df['revenue'].sum() == 3000

    def test_load_via_uri(self, s3_seeded):
        bucket, key = s3_seeded
        conn = S3Connection(
            f's3://{bucket}/{key}',
            endpoint_url=S3_ENDPOINT,
            aws_access_key_id=S3_KEY_ID,
            aws_secret_access_key=S3_SECRET,
            region_name='us-east-1',
        )
        df = conn.load()
        assert len(df) == EXPECTED_ROWS


# ---------------------------------------------------------------------------
# Azurite (Azure Blob)
# ---------------------------------------------------------------------------

# Azurite well-known development connection string
AZURITE_CONN_STR = os.environ.get(
    'PYCHARTAI_AZURITE_CONN_STR',
    'DefaultEndpointsProtocol=http;'
    'AccountName=devstoreaccount1;'
    'AccountKey=Eby8vdM02xNOcqFlqUwJPLlmEtlCDXJ1OUzFT50uSRZ6IFsuFq2UVErCz4I6tq/'
    'K1SZFPTOtr/KBHBeksoGMGw==;'
    'BlobEndpoint=http://localhost:10000/devstoreaccount1;',
)


@pytest.fixture(scope='module')
def azurite_seeded():
    """Create a container and upload the sample CSV to Azurite."""
    azure_blob = pytest.importorskip('azure.storage.blob')
    from azure.storage.blob import BlobServiceClient

    container, blob = 'pychartai-it', 'sales.csv'
    service = BlobServiceClient.from_connection_string(AZURITE_CONN_STR)
    try:
        service.create_container(container)
    except Exception:
        pass
    blob_client = service.get_blob_client(container=container, blob=blob)
    blob_client.upload_blob(SAMPLE_CSV, overwrite=True)
    return container, blob


class TestAzureIntegration:

    def test_load_csv_from_azurite(self, azurite_seeded):
        container, blob = azurite_seeded
        conn = AzureBlobConnection(
            container=container, blob=blob,
            connection_string=AZURITE_CONN_STR,
        )
        df = conn.load()
        assert list(df.columns) == EXPECTED_COLS
        assert len(df) == EXPECTED_ROWS
        assert df['units'].sum() == 150


# ---------------------------------------------------------------------------
# fake-gcs-server (Google Cloud Storage)
# ---------------------------------------------------------------------------

GCS_ENDPOINT = os.environ.get('PYCHARTAI_GCS_ENDPOINT', 'http://localhost:4443')


@pytest.fixture(scope='module')
def gcs_seeded():
    """Create a bucket and upload the sample CSV to fake-gcs-server."""
    storage = pytest.importorskip('google.cloud.storage')
    # Point the client at the emulator
    os.environ['STORAGE_EMULATOR_HOST'] = GCS_ENDPOINT
    from google.cloud import storage as gcs
    from google.auth.credentials import AnonymousCredentials

    bucket_name, blob_name = 'pychartai-it', 'sales.csv'
    client = gcs.Client(
        project='test',
        credentials=AnonymousCredentials(),
        client_options={'api_endpoint': GCS_ENDPOINT},
    )
    try:
        bucket = client.create_bucket(bucket_name)
    except Exception:
        bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    blob.upload_from_string(SAMPLE_CSV, content_type='text/csv')
    return bucket_name, blob_name


class TestGCSIntegration:

    def test_load_csv_from_fake_gcs(self, gcs_seeded, monkeypatch):
        bucket_name, blob_name = gcs_seeded
        monkeypatch.setenv('STORAGE_EMULATOR_HOST', GCS_ENDPOINT)
        conn = GCSConnection(bucket=bucket_name, blob=blob_name, project='test')
        df = conn.load()
        assert list(df.columns) == EXPECTED_COLS
        assert len(df) == EXPECTED_ROWS
