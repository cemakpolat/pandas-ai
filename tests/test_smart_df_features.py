"""Tests for SmartDataFrame new features (memory, profile, repr_html)."""
import pytest
import pandas as pd

from pychartai_core.smart_df import SmartDataFrame


@pytest.fixture
def sdf():
	df = pd.DataFrame({
		'name': ['Alice', 'Bob', 'Charlie'],
		'age': [25, 30, 35],
		'salary': [50000, 60000, 70000],
	})
	return SmartDataFrame(df)


class TestSmartDataFrameMemory:
	def test_memory_disabled_by_default(self, sdf):
		assert sdf.memory is None

	def test_enable_memory(self, sdf):
		sdf.enable_memory(window_size=5)
		assert sdf.memory is not None
		assert len(sdf.memory) == 0

	def test_disable_memory(self, sdf):
		sdf.enable_memory()
		sdf.disable_memory()
		assert sdf.memory is None

	def test_enable_memory_returns_self(self, sdf):
		result = sdf.enable_memory()
		assert result is sdf


class TestSmartDataFrameProfile:
	def test_profile(self, sdf):
		report = sdf.profile()
		assert report.n_rows == 3
		assert report.n_columns == 3

	def test_profile_has_numeric_stats(self, sdf):
		report = sdf.profile()
		assert report.numeric_stats is not None
		assert 'age' in report.numeric_stats.index

	def test_profile_summary(self, sdf):
		report = sdf.profile()
		assert '3' in report.summary


class TestSmartDataFrameReprHtml:
	def test_repr_html(self, sdf):
		html = sdf._repr_html_()
		assert 'SmartDataFrame' in html
		assert '3' in html  # 3 rows
		assert '<table' in html
