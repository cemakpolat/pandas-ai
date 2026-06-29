"""
schema.py — Semantic Layer: describe columns, tables, and domain context.

The semantic layer closes the gap between raw column names and the LLM's
understanding of your data.  Attach a :class:`Schema` to a
:class:`~pychartai_core.smart_df.SmartDataFrame` and its metadata is
automatically injected into every LLM prompt.

Usage::

    import pychartai_core as pai

    schema = pai.Schema(
        name='Monthly Sales',
        description='Aggregated sales records from the ERP system.',
        columns={
            'revenue':  pai.Column(description='Monthly revenue', unit='USD'),
            'region':   pai.Column(description='Geographic region',
                                   values=['North', 'South', 'East', 'West']),
            'category': pai.Column(description='Product category', dtype='str'),
        },
    )

    sdf = pai.read_csv('sales.csv')
    sdf.set_schema(schema)
    sdf.chat('Which region has the highest revenue in Q3?')
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class Column:
	"""Metadata for a single DataFrame column.

	Attributes:
		description: Human-readable description of the column.
		dtype:       Expected data type (e.g. ``'float'``, ``'str'``, ``'datetime'``).
		unit:        Physical unit (e.g. ``'USD'``, ``'kg'``, ``'°C'``).
		values:      Enumerable set of expected values (categorical columns).
	"""

	description: str = ''
	dtype: str = ''
	unit: str = ''
	values: Optional[List[Any]] = None

	def to_prompt_fragment(self, col_name: str) -> str:
		"""One-line description for LLM prompt injection."""
		parts: list[str] = [f'  - {col_name}']
		if self.dtype:
			parts.append(f'[{self.dtype}]')
		if self.description:
			parts.append(f': {self.description}')
		extras: list[str] = []
		if self.unit:
			extras.append(f'unit={self.unit}')
		if self.values is not None:
			sample = self.values[:6]
			suffix = '...' if len(self.values) > 6 else ''
			extras.append(f'values={sample}{suffix}')
		if extras:
			parts.append(f' ({", ".join(extras)})')
		return ''.join(parts)


@dataclass
class Schema:
	"""Semantic description of a DataFrame.

	Attributes:
		columns:     Mapping of column name → :class:`Column` metadata.
		             A plain ``str`` value is treated as a short description.
		name:        Dataset/table name.
		description: High-level description injected into the LLM context.
	"""

	columns: Dict[str, Any] = field(default_factory=dict)
	name: str = ''
	description: str = ''

	def to_prompt_fragment(self) -> str:
		"""Multi-line context block injected before the user query."""
		lines: list[str] = ['# Dataset context (semantic layer):']
		if self.name:
			lines.append(f'# Dataset: {self.name}')
		if self.description:
			lines.append(f'# {self.description}')
		if self.columns:
			lines.append('# Column metadata:')
			for col_name, col_meta in self.columns.items():
				if isinstance(col_meta, Column):
					lines.append(col_meta.to_prompt_fragment(col_name))
				else:
					# Accept plain string as short description
					lines.append(f'  - {col_name}: {col_meta}')
		return '\n'.join(lines)
