"""
reporter.py — Auto-insight HTML report generator.

Generates a self-contained HTML report from a DataFrame, including:
  - Summary statistics (via DataProfiler)
  - Auto-selected charts (bar, distribution, correlation heatmap, time-series)
  - Optional LLM-generated narrative per section

Usage::

    from pychartai_core.reporter import InsightReporter
    InsightReporter(sdf).generate('report.html')
"""
from __future__ import annotations

import base64
import html
import io
import os
import textwrap
from typing import Optional

import pandas as pd


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _fig_to_b64(fig) -> str:
	"""Convert a matplotlib / seaborn Figure to an inline base-64 PNG."""
	buf = io.BytesIO()
	fig.savefig(buf, format='png', bbox_inches='tight', dpi=100)
	buf.seek(0)
	return base64.b64encode(buf.read()).decode('ascii')


def _df_to_html_table(df: pd.DataFrame, max_rows: int = 10) -> str:
	"""Render up to *max_rows* of *df* as an HTML table."""
	return (
		df.head(max_rows)
		.to_html(index=True, border=0, classes='stats-table', escape=True)
	)


# ---------------------------------------------------------------------------
# Chart generation helpers
# ---------------------------------------------------------------------------

def _chart_top_category(df: pd.DataFrame, cat_col: str, num_col: str):
	"""Bar chart of top-10 values in *cat_col* summed by *num_col*."""
	try:
		import matplotlib
		matplotlib.use('Agg')
		import matplotlib.pyplot as plt

		data = (
			df.groupby(cat_col)[num_col]
			.sum()
			.nlargest(10)
			.sort_values()
		)
		fig, ax = plt.subplots(figsize=(7, 4))
		data.plot.barh(ax=ax, color='steelblue')
		ax.set_title(f'Top {cat_col} by {num_col}')
		ax.set_xlabel(num_col)
		ax.set_ylabel(cat_col)
		fig.tight_layout()
		return fig
	except Exception:
		return None


def _chart_time_series(df: pd.DataFrame, date_col: str, num_col: str):
	"""Line chart of *num_col* over *date_col*."""
	try:
		import matplotlib
		matplotlib.use('Agg')
		import matplotlib.pyplot as plt

		tmp = df[[date_col, num_col]].copy()
		tmp[date_col] = pd.to_datetime(tmp[date_col], errors='coerce')
		tmp = tmp.dropna(subset=[date_col]).sort_values(date_col)
		tmp = tmp.groupby(date_col)[num_col].sum().reset_index()

		fig, ax = plt.subplots(figsize=(8, 4))
		ax.plot(tmp[date_col], tmp[num_col], linewidth=1.5)
		ax.set_title(f'{num_col} over time ({date_col})')
		ax.set_xlabel(date_col)
		ax.set_ylabel(num_col)
		fig.autofmt_xdate()
		fig.tight_layout()
		return fig
	except Exception:
		return None


def _chart_distribution(df: pd.DataFrame, num_col: str):
	"""Histogram + KDE for a numeric column."""
	try:
		import matplotlib
		matplotlib.use('Agg')
		import matplotlib.pyplot as plt

		series = df[num_col].dropna()
		if len(series) < 2:
			return None
		fig, ax = plt.subplots(figsize=(6, 4))
		ax.hist(series, bins=30, color='steelblue', edgecolor='white', alpha=0.8)
		ax.set_title(f'Distribution of {num_col}')
		ax.set_xlabel(num_col)
		ax.set_ylabel('Count')
		fig.tight_layout()
		return fig
	except Exception:
		return None


def _chart_correlation(df: pd.DataFrame):
	"""Correlation heatmap for numeric columns."""
	try:
		import matplotlib
		matplotlib.use('Agg')
		import matplotlib.pyplot as plt

		num_df = df.select_dtypes(include='number')
		if num_df.shape[1] < 2:
			return None
		corr = num_df.corr()

		fig, ax = plt.subplots(figsize=(min(10, corr.shape[1] + 2), min(8, corr.shape[0] + 1)))
		cax = ax.matshow(corr, cmap='RdBu_r', vmin=-1, vmax=1)
		fig.colorbar(cax)
		ax.set_xticks(range(len(corr.columns)))
		ax.set_yticks(range(len(corr.columns)))
		ax.set_xticklabels(corr.columns, rotation=45, ha='left', fontsize=8)
		ax.set_yticklabels(corr.columns, fontsize=8)
		ax.set_title('Correlation Matrix', pad=20)
		fig.tight_layout()
		return fig
	except Exception:
		return None


# ---------------------------------------------------------------------------
# LLM narrative
# ---------------------------------------------------------------------------

def _llm_narrative(llm, context: str, max_sentences: int = 3) -> Optional[str]:
	"""Ask the LLM for a short narrative about *context*."""
	if llm is None:
		return None
	try:
		prompt = (
			f'You are a data analyst.  In {max_sentences} sentences, describe the '
			f'key insight from the following data summary:\n\n{context}\n\n'
			'Reply with plain text only — no bullet points, no markdown.'
		)
		response = llm.call(prompt)
		return response.strip() if isinstance(response, str) else None
	except Exception:
		return None


# ---------------------------------------------------------------------------
# HTML template
# ---------------------------------------------------------------------------

_HTML_TEMPLATE = '''\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>PyChartAI Insight Report</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: 'Segoe UI', Arial, sans-serif; background: #f5f7fa;
         color: #333; padding: 20px; }}
  h1 {{ font-size: 1.8rem; margin-bottom: 4px; }}
  h2 {{ font-size: 1.2rem; margin: 28px 0 10px; color: #1a56db; border-bottom: 2px solid #1a56db; padding-bottom: 4px; }}
  h3 {{ font-size: 1rem; margin: 16px 0 6px; color: #555; }}
  .subtitle {{ color: #666; font-size: 0.9rem; margin-bottom: 20px; }}
  .card {{ background: #fff; border-radius: 8px; padding: 20px;
           box-shadow: 0 1px 4px rgba(0,0,0,.1); margin-bottom: 20px; }}
  .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 16px; }}
  .metric {{ background: #fff; border-radius: 8px; padding: 16px;
             box-shadow: 0 1px 3px rgba(0,0,0,.08); text-align: center; }}
  .metric .value {{ font-size: 2rem; font-weight: 700; color: #1a56db; }}
  .metric .label {{ font-size: 0.8rem; color: #888; margin-top: 4px; }}
  pre.summary {{ background: #f0f4ff; border-left: 4px solid #1a56db;
                 padding: 12px 16px; font-size: 0.85rem; border-radius: 0 6px 6px 0;
                 white-space: pre-wrap; }}
  .narrative {{ background: #fffbe6; border-left: 4px solid #f59e0b;
                padding: 10px 14px; border-radius: 0 6px 6px 0;
                font-style: italic; color: #555; margin: 12px 0; }}
  img.chart {{ max-width: 100%; border-radius: 6px; margin-top: 8px; }}
  table.stats-table {{ border-collapse: collapse; width: 100%; font-size: 0.8rem; }}
  table.stats-table th {{ background: #1a56db; color: #fff; padding: 6px 10px; text-align: left; }}
  table.stats-table td {{ padding: 5px 10px; border-bottom: 1px solid #e8eaf0; }}
  table.stats-table tr:nth-child(even) td {{ background: #f8f9ff; }}
  footer {{ text-align: center; color: #aaa; font-size: 0.75rem; margin-top: 32px; }}
</style>
</head>
<body>
<h1>PyChartAI Insight Report</h1>
<p class="subtitle">Generated automatically from a {n_rows:,}-row × {n_cols}-column DataFrame</p>

{body}

<footer>Generated by PyChartAI &mdash; pychartai_core.reporter</footer>
</body>
</html>
'''

_CHART_SECTION = '''\
<div class="card">
  <h2>{title}</h2>
  {narrative}
  <img class="chart" src="data:image/png;base64,{b64}" alt="{title}">
</div>
'''

_NARRATIVE_BLOCK = '<div class="narrative">{text}</div>'


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class InsightReporter:
	"""Generate a self-contained HTML insight report from a SmartDataFrame.

	Args:
	    sdf:             A :class:`~pychartai_core.smart_df.SmartDataFrame`.
	    llm:             Optional LLM instance for narrative text.  If not
	                     provided, the global config LLM is used when available.
	"""

	def __init__(self, sdf, llm=None) -> None:
		if isinstance(sdf, pd.DataFrame):
			from .smart_df import SmartDataFrame
			sdf = SmartDataFrame(sdf)
		self._sdf = sdf
		self._llm = llm

	# ------------------------------------------------------------------
	# Public API
	# ------------------------------------------------------------------

	def generate(
		self,
		output_file: str = 'report.html',
		*,
		llm_narrative: bool = True,
	) -> str:
		"""Build the report and write it to *output_file*.

		Args:
		    output_file:    Destination HTML path.
		    llm_narrative:  If True, call the LLM for a brief narrative per
		                    chart section (requires a configured LLM).

		Returns:
		    The absolute path to the generated file.
		"""
		from .profiler import DataProfiler

		df: pd.DataFrame = object.__getattribute__(self._sdf, '_df')
		profile = DataProfiler.profile(df)

		# Resolve LLM
		llm = self._llm
		if llm is None and llm_narrative:
			try:
				from .config import config as global_config
				llm = global_config.get('llm')
			except Exception:
				pass

		sections: list[str] = []

		# --- Overview metrics ---
		sections.append(self._overview_section(profile))

		# --- Numeric stats table ---
		if profile.numeric_stats is not None and not profile.numeric_stats.empty:
			sections.append(self._stats_section(profile, llm if llm_narrative else None))

		# --- Auto charts ---
		chart_sections = self._build_chart_sections(df, profile, llm if llm_narrative else None)
		sections.extend(chart_sections)

		body = '\n'.join(sections)
		full_html = _HTML_TEMPLATE.format(
			n_rows=profile.n_rows,
			n_cols=profile.n_columns,
			body=body,
		)

		output_file = os.path.abspath(output_file)
		os.makedirs(os.path.dirname(output_file) or '.', exist_ok=True)
		with open(output_file, 'w', encoding='utf-8') as fh:
			fh.write(full_html)

		return output_file

	# ------------------------------------------------------------------
	# Section builders
	# ------------------------------------------------------------------

	@staticmethod
	def _overview_section(profile) -> str:
		metrics = [
			(f'{profile.n_rows:,}', 'Rows'),
			(str(profile.n_columns), 'Columns'),
			(f'{profile.memory_usage_mb:.2f} MB', 'Memory'),
			(str(profile.duplicates), 'Duplicate Rows'),
		]
		metric_html = ''.join(
			f'<div class="metric"><div class="value">{v}</div><div class="label">{l}</div></div>'
			for v, l in metrics
		)
		summary_html = f'<pre class="summary">{html.escape(profile.summary)}</pre>'
		return (
			'<div class="card"><h2>Overview</h2>'
			f'<div class="grid" style="margin-bottom:16px">{metric_html}</div>'
			f'{summary_html}</div>'
		)

	@staticmethod
	def _stats_section(profile, llm=None) -> str:
		table = _df_to_html_table(profile.numeric_stats.T if profile.numeric_stats is not None else pd.DataFrame())
		narrative = ''
		if llm:
			ctx = f'Numeric stats for {profile.n_columns} columns:\n{str(profile.numeric_stats)}'
			text = _llm_narrative(llm, ctx)
			if text:
				narrative = _NARRATIVE_BLOCK.format(text=html.escape(text))
		return (
			f'<div class="card"><h2>Numeric Statistics</h2>'
			f'{narrative}{table}</div>'
		)

	@staticmethod
	def _build_chart_sections(df: pd.DataFrame, profile, llm=None) -> list:
		sections = []

		numeric_cols = df.select_dtypes(include='number').columns.tolist()
		cat_cols = df.select_dtypes(include=['object', 'category', 'string']).columns.tolist()
		date_cols = [
			c for c in df.columns
			if 'date' in c.lower() or 'time' in c.lower() or pd.api.types.is_datetime64_any_dtype(df[c])
		]

		charts_added = 0
		MAX_CHARTS = 5

		# 1. Top-category bar chart
		if cat_cols and numeric_cols and charts_added < MAX_CHARTS:
			cat_col = cat_cols[0]
			num_col = numeric_cols[0]
			fig = _chart_top_category(df, cat_col, num_col)
			if fig is not None:
				title = f'Top {cat_col} by {num_col}'
				sections.append(InsightReporter._chart_html(fig, title, llm, df, cat_col, num_col))
				charts_added += 1

		# 2. Time-series
		if date_cols and numeric_cols and charts_added < MAX_CHARTS:
			fig = _chart_time_series(df, date_cols[0], numeric_cols[0])
			if fig is not None:
				title = f'{numeric_cols[0]} over time'
				ctx = f'Time-series of {numeric_cols[0]} along {date_cols[0]}.'
				narrative = ''
				if llm:
					text = _llm_narrative(llm, ctx)
					if text:
						narrative = _NARRATIVE_BLOCK.format(text=html.escape(text))
				b64 = _fig_to_b64(fig)
				import matplotlib.pyplot as plt
				plt.close(fig)
				sections.append(_CHART_SECTION.format(title=title, narrative=narrative, b64=b64))
				charts_added += 1

		# 3. Distribution of first numeric col
		if numeric_cols and charts_added < MAX_CHARTS:
			fig = _chart_distribution(df, numeric_cols[0])
			if fig is not None:
				title = f'Distribution of {numeric_cols[0]}'
				ctx = f'Distribution of {numeric_cols[0]}: {df[numeric_cols[0]].describe().to_string()}'
				narrative = ''
				if llm:
					text = _llm_narrative(llm, ctx)
					if text:
						narrative = _NARRATIVE_BLOCK.format(text=html.escape(text))
				b64 = _fig_to_b64(fig)
				import matplotlib.pyplot as plt
				plt.close(fig)
				sections.append(_CHART_SECTION.format(title=title, narrative=narrative, b64=b64))
				charts_added += 1

		# 4. Correlation heatmap
		if len(numeric_cols) >= 2 and charts_added < MAX_CHARTS:
			fig = _chart_correlation(df)
			if fig is not None:
				corr = df.select_dtypes(include='number').corr()
				ctx = f'Correlation matrix:\n{corr.to_string()}'
				narrative = ''
				if llm:
					text = _llm_narrative(llm, ctx)
					if text:
						narrative = _NARRATIVE_BLOCK.format(text=html.escape(text))
				b64 = _fig_to_b64(fig)
				import matplotlib.pyplot as plt
				plt.close(fig)
				sections.append(_CHART_SECTION.format(title='Correlation Matrix', narrative=narrative, b64=b64))
				charts_added += 1

		# 5. Second numeric distribution
		if len(numeric_cols) >= 2 and charts_added < MAX_CHARTS:
			fig = _chart_distribution(df, numeric_cols[1])
			if fig is not None:
				title = f'Distribution of {numeric_cols[1]}'
				b64 = _fig_to_b64(fig)
				import matplotlib.pyplot as plt
				plt.close(fig)
				sections.append(_CHART_SECTION.format(title=title, narrative='', b64=b64))
				charts_added += 1

		return sections

	@staticmethod
	def _chart_html(fig, title: str, llm, df: pd.DataFrame, cat_col: str, num_col: str) -> str:
		ctx = (
			f'Bar chart showing top values of {cat_col} by {num_col}.\n'
			f'Top values:\n'
			+ str(df.groupby(cat_col)[num_col].sum().nlargest(5))
		)
		narrative = ''
		if llm:
			text = _llm_narrative(llm, ctx)
			if text:
				narrative = _NARRATIVE_BLOCK.format(text=html.escape(text))
		b64 = _fig_to_b64(fig)
		import matplotlib.pyplot as plt
		plt.close(fig)
		return _CHART_SECTION.format(title=title, narrative=narrative, b64=b64)
