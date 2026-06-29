"""Tests for DataProfiler."""
import pytest
import pandas as pd
import numpy as np

from pychartai_core.profiler import DataProfiler, ProfileReport


@pytest.fixture
def sample_df():
	return pd.DataFrame({
		'name': ['Alice', 'Bob', 'Charlie', 'Alice', 'Bob'],
		'age': [25, 30, 35, 25, 30],
		'salary': [50000.0, 60000.0, 70000.0, 50000.0, 60000.0],
		'city': ['NY', 'SF', 'LA', 'NY', 'SF'],
	})


@pytest.fixture
def missing_df():
	return pd.DataFrame({
		'a': [1, 2, None, 4],
		'b': [None, None, 3, 4],
		'c': ['x', 'y', 'z', 'w'],
	})


class TestDataProfiler:
	def test_profile_basic(self, sample_df):
		report = DataProfiler.profile(sample_df)
		assert isinstance(report, ProfileReport)
		assert report.n_rows == 5
		assert report.n_columns == 4

	def test_numeric_stats(self, sample_df):
		report = DataProfiler.profile(sample_df)
		assert report.numeric_stats is not None
		assert 'age' in report.numeric_stats.index
		assert 'salary' in report.numeric_stats.index
		assert 'skew' in report.numeric_stats.columns

	def test_categorical_stats(self, sample_df):
		report = DataProfiler.profile(sample_df)
		assert report.categorical_stats is not None
		assert 'name' in report.categorical_stats.index
		assert 'city' in report.categorical_stats.index

	def test_missing_values(self, missing_df):
		report = DataProfiler.profile(missing_df)
		assert report.missing['missing_count'].sum() > 0
		# column 'b' has 2 missing values
		b_row = report.missing[report.missing['column'] == 'b']
		assert int(b_row['missing_count'].iloc[0]) == 2

	def test_duplicates(self, sample_df):
		report = DataProfiler.profile(sample_df)
		assert report.duplicates >= 0

	def test_correlations(self, sample_df):
		report = DataProfiler.profile(sample_df)
		assert report.correlations is not None
		assert report.correlations.shape[0] >= 2

	def test_constant_columns(self):
		df = pd.DataFrame({'a': [1, 1, 1], 'b': [2, 3, 4]})
		report = DataProfiler.profile(df)
		assert 'a' in report.constant_columns

	def test_summary_string(self, sample_df):
		report = DataProfiler.profile(sample_df)
		summary = report.summary
		assert 'rows' in summary
		assert 'columns' in summary

	def test_to_dict(self, sample_df):
		report = DataProfiler.profile(sample_df)
		d = report.to_dict()
		assert 'n_rows' in d
		assert d['n_rows'] == 5

	def test_empty_df(self):
		df = pd.DataFrame()
		report = DataProfiler.profile(df)
		assert report.n_rows == 0
		assert report.n_columns == 0
