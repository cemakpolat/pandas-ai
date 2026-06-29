"""tests/test_redactor.py — Tests for DataRedactor PII detection and masking."""

import hashlib

import pandas as pd
import pytest

from pychartai_core.redactor import DataRedactor, _hash_value


@pytest.fixture()
def pii_df():
    return pd.DataFrame({
        'name':     ['Alice', 'Bob', 'Carol'],
        'email':    ['a@x.com', 'b@x.com', 'c@x.com'],
        'phone':    ['555-0001', '555-0002', '555-0003'],
        'revenue':  [1000.0, 800.0, 1200.0],
        'region':   ['North', 'South', 'East'],
    })


class TestAutoDetect:

    def test_detects_email_and_phone(self, pii_df):
        r = DataRedactor()
        detected = r.detect_pii_columns(pii_df)
        assert 'email' in detected
        assert 'phone' in detected

    def test_detects_name(self, pii_df):
        r = DataRedactor()
        detected = r.detect_pii_columns(pii_df)
        assert 'name' in detected

    def test_non_pii_columns_not_detected(self, pii_df):
        r = DataRedactor()
        detected = r.detect_pii_columns(pii_df)
        assert 'revenue' not in detected
        assert 'region' not in detected

    def test_extra_columns_always_included(self, pii_df):
        r = DataRedactor(extra_columns=['revenue'])
        detected = r.detect_pii_columns(pii_df)
        assert 'revenue' in detected


class TestHashStrategy:

    def test_hash_replaces_values(self, pii_df):
        r = DataRedactor(strategy='hash')
        result = r.redact(pii_df)
        assert result['email'].iloc[0] != 'a@x.com'
        assert len(result['email'].iloc[0]) == 12   # SHA-256 hex[:12]

    def test_hash_preserves_groupability(self, pii_df):
        # Same input → same hash → group-by still works
        r = DataRedactor(strategy='hash')
        result = r.redact(pii_df)
        h1 = result['email'].iloc[0]
        assert h1 == _hash_value('a@x.com')

    def test_non_pii_cols_unchanged(self, pii_df):
        r = DataRedactor(strategy='hash')
        result = r.redact(pii_df)
        assert list(result['revenue']) == [1000.0, 800.0, 1200.0]
        assert list(result['region']) == ['North', 'South', 'East']

    def test_original_df_not_mutated(self, pii_df):
        r = DataRedactor(strategy='hash')
        r.redact(pii_df)
        assert pii_df['email'].iloc[0] == 'a@x.com'


class TestMaskStrategy:

    def test_mask_replaces_with_asterisks(self, pii_df):
        r = DataRedactor(strategy='mask')
        result = r.redact(pii_df)
        assert all(v == '***' for v in result['email'])

    def test_mask_preserves_nan(self):
        import numpy as np
        df = pd.DataFrame({'email': ['a@b.com', None, 'c@d.com']})
        r = DataRedactor(strategy='mask')
        result = r.redact(df)
        assert pd.isna(result['email'].iloc[1])
        assert result['email'].iloc[0] == '***'


class TestDropStrategy:

    def test_drop_removes_column(self, pii_df):
        r = DataRedactor(strategy='drop')
        result = r.redact(pii_df)
        assert 'email' not in result.columns
        assert 'phone' not in result.columns
        assert 'name' not in result.columns

    def test_drop_keeps_non_pii(self, pii_df):
        r = DataRedactor(strategy='drop')
        result = r.redact(pii_df)
        assert 'revenue' in result.columns
        assert 'region' in result.columns


class TestExplicitColumns:

    def test_explicit_columns_override_autodetect(self, pii_df):
        r = DataRedactor(strategy='mask')
        result = r.redact(pii_df, columns=['revenue'])
        # revenue should be masked, email should NOT be (not in explicit list)
        assert result['revenue'].iloc[0] == '***'
        assert result['email'].iloc[0] == 'a@x.com'  # not redacted

    def test_nonexistent_column_is_skipped(self, pii_df):
        r = DataRedactor(strategy='mask')
        result = r.redact(pii_df, columns=['nonexistent_col'])
        # no error raised, df unchanged
        assert list(result.columns) == list(pii_df.columns)


class TestReportMode:

    def test_report_returns_tuple(self, pii_df):
        r = DataRedactor(strategy='hash')
        result = r.redact(pii_df, report=True)
        assert isinstance(result, tuple)
        redacted_df, stats = result
        assert isinstance(stats, dict)
        assert 'email' in stats
        assert stats['email'] == 3   # 3 non-null values changed

    def test_report_counts_match_non_null(self):
        df = pd.DataFrame({'email': ['a@b.com', None, 'c@d.com']})
        r = DataRedactor(strategy='mask')
        _, stats = r.redact(df, report=True)
        assert stats['email'] == 2   # only 2 non-null values


class TestValidation:

    def test_invalid_strategy_raises(self):
        with pytest.raises(ValueError, match='strategy'):
            DataRedactor(strategy='invalid')

    def test_repr_shows_strategy(self):
        r = DataRedactor(strategy='mask')
        assert 'mask' in repr(r)
