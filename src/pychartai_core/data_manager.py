"""
Data management module for handling pandas DataFrames.

Sample data generators live in data/generators/sample_data.py (outside src/)
and are loaded on first use via importlib so this package has no hard dependency
on that demo utility module.
"""

import importlib.util
import pandas as pd
from pathlib import Path
from typing import Any, Dict, Optional

# Resolve generators path relative to this file:
#   src/pychartai_core/data_manager.py  -> ../../..  -> project root
_GEN_PATH = Path(__file__).resolve().parent.parent.parent / 'data' / 'generators' / 'sample_data.py'

_generators_cache: Dict[str, Any] = {}


def _get_generators() -> Dict[str, Any]:
	"""Load sample-data generator functions on first call."""
	if _generators_cache:
		return _generators_cache
	if not _GEN_PATH.exists():
		raise ImportError(
			f'Sample data generators not found at {_GEN_PATH}. '
			'Ensure data/generators/sample_data.py is present at the project root.'
		)
	spec = importlib.util.spec_from_file_location('_pychartai_sample_data', _GEN_PATH)
	mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
	spec.loader.exec_module(mod)  # type: ignore[union-attr]
	_generators_cache.update({
		'sales':          mod.create_sales_data,
		'sales_extended': mod.create_sales_data,
		'weather':        mod.create_weather_data,
		'stocks':         mod.create_stocks_data,
		'analytics':      mod.create_analytics_data,
		'ecommerce':      mod.create_ecommerce_data,
		'health':         mod.create_health_data,
		'energy':         mod.create_energy_data,
	})
	return _generators_cache


class DataManager:
	"""Manages data loading and preprocessing."""

	def __init__(self):
		self.dataframes: Dict[str, pd.DataFrame] = {}

	def create_sample_data(self, name: str, data_type: str = 'sales') -> pd.DataFrame:
		"""
		Create sample data for testing.

		Args:
			name: Name to store the dataset under.
			data_type: One of sales, weather, stocks, analytics, ecommerce, health, energy.

		Returns:
			Generated DataFrame.
		"""
		generators = _get_generators()
		generator = generators.get(data_type)
		if generator is None:
			raise ValueError(f'Unknown data type: {data_type}')
		df = generator()
		self.dataframes[name] = df
		return df

	def load_dataframe(self, name: str) -> Optional[pd.DataFrame]:
		"""Return a stored DataFrame by name."""
		return self.dataframes.get(name)

	def sample_dataset_path(self, data_type: str, data_dir: str = 'data/use_cases') -> str:
		"""Return canonical CSV path for a sample dataset type."""
		return str(Path(data_dir) / f'{data_type}.csv')

	def ensure_sample_dataset(self, data_type: str, data_dir: str = 'data/use_cases') -> str:
		"""Ensure a sample dataset CSV exists on disk and return its path."""
		path = Path(self.sample_dataset_path(data_type, data_dir))
		path.parent.mkdir(parents=True, exist_ok=True)
		if not path.exists():
			df = self.create_sample_data(f'{data_type}_file', data_type)
			df.to_csv(path, index=False)
		return str(path)

	def load_or_create_sample_data(
		self,
		name: str,
		data_type: str,
		data_dir: str = 'data/use_cases',
	) -> pd.DataFrame:
		"""Load sample data from disk, generating it first if missing."""
		path = self.ensure_sample_dataset(data_type, data_dir=data_dir)
		df = pd.read_csv(path)
		self.dataframes[name] = df
		return df

	def list_dataframes(self) -> list:
		"""List all stored DataFrame names."""
		return list(self.dataframes.keys())

	def get_dataframe_info(self, name: str) -> Dict[str, Any]:
		"""Get shape, columns, dtypes, and memory info for a stored DataFrame."""
		df = self.dataframes.get(name)
		if df is None:
			return {}
		return {
			'name': name,
			'shape': df.shape,
			'columns': df.columns.tolist(),
			'dtypes': df.dtypes.to_dict(),
			'memory_usage': df.memory_usage(deep=True).sum(),
		}

