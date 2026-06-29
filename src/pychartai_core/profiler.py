"""
profiler.py — Lightweight DataFrame profiling / auto-EDA.

Generates rich summary statistics, distribution info, missing-value analysis,
and correlations — all without an LLM call (pure pandas).

Usage::

    from pychartai_core.profiler import DataProfiler

    report = DataProfiler.profile(df)
    print(report.summary)          # overview text
    print(report.missing)          # missing value counts
    print(report.correlations)     # correlation matrix
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import pandas as pd
import numpy as np


@dataclass
class ProfileReport:
	"""Result of a DataFrame profiling run."""

	n_rows: int
	n_columns: int
	memory_usage_mb: float
	dtypes: Dict[str, int]
	numeric_stats: Optional[pd.DataFrame]
	categorical_stats: Optional[pd.DataFrame]
	missing: pd.DataFrame
	duplicates: int
	correlations: Optional[pd.DataFrame]
	constant_columns: List[str]
	high_cardinality: List[str]

	@property
	def summary(self) -> str:
		"""Human-readable summary string."""
		lines = [
			f'DataFrame Profile: {self.n_rows:,} rows × {self.n_columns} columns',
			f'Memory usage: {self.memory_usage_mb:.2f} MB',
			f'Dtypes: {self.dtypes}',
			f'Duplicate rows: {self.duplicates:,}',
		]
		if self.constant_columns:
			lines.append(f'Constant columns (single value): {self.constant_columns}')
		if self.high_cardinality:
			lines.append(f'High-cardinality columns (>50 unique): {self.high_cardinality}')

		total_missing = int(self.missing['missing_count'].sum())
		if total_missing > 0:
			lines.append(f'Total missing values: {total_missing:,}')
			worst = self.missing[self.missing['missing_count'] > 0].head(5)
			for _, row in worst.iterrows():
				lines.append(f'  {row["column"]}: {int(row["missing_count"])} ({row["missing_pct"]:.1f}%)')
		else:
			lines.append('No missing values.')

		return '\n'.join(lines)

	def to_dict(self) -> Dict[str, Any]:
		"""Serialize to a JSON-friendly dict."""
		d: Dict[str, Any] = {
			'n_rows': self.n_rows,
			'n_columns': self.n_columns,
			'memory_usage_mb': self.memory_usage_mb,
			'dtypes': self.dtypes,
			'duplicates': self.duplicates,
			'constant_columns': self.constant_columns,
			'high_cardinality': self.high_cardinality,
		}
		if self.numeric_stats is not None:
			d['numeric_stats'] = self.numeric_stats.to_dict()
		if self.categorical_stats is not None:
			d['categorical_stats'] = self.categorical_stats.to_dict()
		d['missing'] = self.missing.to_dict(orient='records')
		if self.correlations is not None:
			d['correlations'] = self.correlations.to_dict()
		return d


class DataProfiler:
	"""Profiler — no LLM needed. Use as DataProfiler(df).profile() or DataProfiler.profile(df)."""

	def __init__(self, df: pd.DataFrame = None) -> None:
		self._df = df

	def profile(self, df: pd.DataFrame = None) -> ProfileReport:
		"""Generate a comprehensive profile of the DataFrame."""
		if isinstance(self, pd.DataFrame):
			# Called as unbound static: DataProfiler.profile(df)
			target = self
		elif df is not None:
			target = df
		else:
			target = self._df
		if target is None:
			raise ValueError('Pass a DataFrame to DataProfiler(df) or DataProfiler.profile(df)')
		return DataProfiler._do_profile(target)

	@staticmethod
	def _do_profile(df: pd.DataFrame) -> ProfileReport:
		"""Generate a comprehensive profile of the DataFrame."""
		n_rows, n_cols = df.shape
		mem_mb = df.memory_usage(deep=True).sum() / (1024 * 1024)
		dtype_counts = df.dtypes.astype(str).value_counts().to_dict()

		# Numeric stats
		numeric_cols = df.select_dtypes(include=[np.number])
		numeric_stats = None
		if not numeric_cols.empty:
			numeric_stats = numeric_cols.describe().T
			numeric_stats['skew'] = numeric_cols.skew()
			numeric_stats['kurtosis'] = numeric_cols.kurtosis()

		# Categorical stats
		cat_cols = df.select_dtypes(include=['object', 'category', 'string'])
		cat_stats = None
		if not cat_cols.empty:
			records = []
			for col in cat_cols.columns:
				series = df[col]
				records.append({
					'column': col,
					'unique': series.nunique(),
					'top': series.mode().iloc[0] if not series.mode().empty else None,
					'top_freq': series.value_counts().iloc[0] if not series.value_counts().empty else 0,
				})
			cat_stats = pd.DataFrame(records).set_index('column')

		# Missing values
		missing_count = df.isnull().sum()
		missing_df = pd.DataFrame({
			'column': missing_count.index,
			'missing_count': missing_count.values,
			'missing_pct': (missing_count.values / max(n_rows, 1)) * 100,
		}).sort_values('missing_count', ascending=False).reset_index(drop=True)

		# Duplicates
		duplicates = int(df.duplicated().sum())

		# Correlations (numeric only)
		correlations = None
		if numeric_cols.shape[1] >= 2:
			correlations = numeric_cols.corr()

		# Constant columns
		constant_cols = [col for col in df.columns if df[col].nunique() <= 1]

		# High cardinality
		high_card = [col for col in cat_cols.columns if df[col].nunique() > 50]

		return ProfileReport(
			n_rows=n_rows,
			n_columns=n_cols,
			memory_usage_mb=mem_mb,
			dtypes=dtype_counts,
			numeric_stats=numeric_stats,
			categorical_stats=cat_stats,
			missing=missing_df,
			duplicates=duplicates,
			correlations=correlations,
			constant_columns=constant_cols,
			high_cardinality=high_card,
		)
