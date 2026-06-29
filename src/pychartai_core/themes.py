"""
themes.py — Chart theme/style customization.

Provides built-in themes (light, dark, corporate, minimal) and lets users
create custom themes with color palettes, font sizes, and background colors.

Usage::

    import pychartai as pai

    pai.config.set({'chart_theme': 'dark'})
    # or per-call:
    df.chat('bar chart of revenue by region', chart_options={'theme': 'dark'})

    # Custom theme:
    from pychartai_core.themes import ChartTheme
    my_theme = ChartTheme(
        name='brand',
        palette=['#1f77b4', '#ff7f0e', '#2ca02c'],
        background='#fafafa',
        text_color='#333333',
    )
    pai.config.set({'chart_theme': my_theme})
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union


@dataclass
class ChartTheme:
	"""Encapsulates visual styling for all chart backends."""

	name: str = 'light'
	palette: List[str] = field(default_factory=lambda: [
		'#4C72B0', '#DD8452', '#55A868', '#C44E52',
		'#8172B3', '#937860', '#DA8BC3', '#8C8C8C',
	])
	background: str = '#ffffff'
	plot_background: str = '#ffffff'
	text_color: str = '#333333'
	grid_color: str = '#e0e0e0'
	grid_alpha: float = 0.5
	font_family: str = 'sans-serif'
	title_size: int = 14
	label_size: int = 11
	tick_size: int = 10
	figure_dpi: int = 150
	grid_style: str = '--'
	show_grid: bool = True

	def apply_matplotlib(self) -> None:
		"""Apply this theme to matplotlib's rcParams."""
		import matplotlib.pyplot as plt

		plt.rcParams.update({
			'figure.facecolor': self.background,
			'axes.facecolor': self.plot_background,
			'axes.edgecolor': self.grid_color,
			'axes.labelcolor': self.text_color,
			'axes.titlesize': self.title_size,
			'axes.labelsize': self.label_size,
			'xtick.color': self.text_color,
			'ytick.color': self.text_color,
			'xtick.labelsize': self.tick_size,
			'ytick.labelsize': self.tick_size,
			'text.color': self.text_color,
			'font.family': self.font_family,
			'figure.dpi': self.figure_dpi,
			'axes.grid': self.show_grid,
			'grid.color': self.grid_color,
			'grid.alpha': self.grid_alpha,
			'grid.linestyle': self.grid_style,
		})

	def apply_seaborn(self) -> None:
		"""Apply this theme's palette to seaborn."""
		import seaborn as sns

		style = 'darkgrid' if self.name == 'dark' else 'whitegrid'
		sns.set_theme(style=style, palette=self.palette, font_scale=1.1)
		self.apply_matplotlib()

	def to_plotly_template(self) -> dict:
		"""Return a Plotly layout dict compatible with ``fig.update_layout()``."""
		return {
			'paper_bgcolor': self.background,
			'plot_bgcolor': self.plot_background,
			'font': {
				'family': self.font_family,
				'size': self.label_size,
				'color': self.text_color,
			},
			'title': {'font': {'size': self.title_size}},
			'colorway': self.palette,
			'xaxis': {
				'gridcolor': self.grid_color,
				'showgrid': self.show_grid,
			},
			'yaxis': {
				'gridcolor': self.grid_color,
				'showgrid': self.show_grid,
			},
		}


# ---- Built-in themes ----

LIGHT_THEME = ChartTheme(name='light')

DARK_THEME = ChartTheme(
	name='dark',
	palette=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f'],
	background='#1e1e1e',
	plot_background='#2d2d2d',
	text_color='#e0e0e0',
	grid_color='#444444',
	grid_alpha=0.4,
)

CORPORATE_THEME = ChartTheme(
	name='corporate',
	palette=['#003f5c', '#2f4b7c', '#665191', '#a05195', '#d45087', '#f95d6a', '#ff7c43', '#ffa600'],
	background='#fafafa',
	plot_background='#ffffff',
	text_color='#2d2d2d',
	grid_color='#d4d4d4',
	title_size=16,
	font_family='serif',
)

MINIMAL_THEME = ChartTheme(
	name='minimal',
	palette=['#333333', '#666666', '#999999', '#cccccc'],
	background='#ffffff',
	plot_background='#ffffff',
	text_color='#333333',
	grid_color='#eeeeee',
	grid_alpha=0.3,
	show_grid=False,
)

BUILTIN_THEMES: Dict[str, ChartTheme] = {
	'light': LIGHT_THEME,
	'dark': DARK_THEME,
	'corporate': CORPORATE_THEME,
	'minimal': MINIMAL_THEME,
}


def resolve_theme(theme: Union[str, ChartTheme, None]) -> ChartTheme:
	"""Resolve a theme name or instance to a ``ChartTheme``.

	Args:
		theme: Name of a built-in theme, a ``ChartTheme`` instance, or None
		       (returns the light theme).
	"""
	if theme is None:
		return LIGHT_THEME
	if isinstance(theme, ChartTheme):
		return theme
	if isinstance(theme, str):
		t = BUILTIN_THEMES.get(theme.lower())
		if t is None:
			available = ', '.join(BUILTIN_THEMES)
			raise ValueError(f'Unknown theme {theme!r}. Available: {available}')
		return t
	raise TypeError(f'Expected str or ChartTheme, got {type(theme).__name__}')
