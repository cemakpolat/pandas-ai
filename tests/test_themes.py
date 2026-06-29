"""Tests for chart themes."""
import pytest

from pychartai_core.themes import (
	ChartTheme, LIGHT_THEME, DARK_THEME, CORPORATE_THEME, MINIMAL_THEME,
	BUILTIN_THEMES, resolve_theme,
)


class TestChartTheme:
	def test_builtin_themes_exist(self):
		assert 'light' in BUILTIN_THEMES
		assert 'dark' in BUILTIN_THEMES
		assert 'corporate' in BUILTIN_THEMES
		assert 'minimal' in BUILTIN_THEMES

	def test_resolve_none(self):
		theme = resolve_theme(None)
		assert theme.name == 'light'

	def test_resolve_string(self):
		theme = resolve_theme('dark')
		assert theme.name == 'dark'
		assert theme.background == '#1e1e1e'

	def test_resolve_instance(self):
		custom = ChartTheme(name='custom', palette=['#ff0000'])
		assert resolve_theme(custom) is custom

	def test_resolve_invalid(self):
		with pytest.raises(ValueError, match='Unknown theme'):
			resolve_theme('nonexistent')

	def test_resolve_wrong_type(self):
		with pytest.raises(TypeError):
			resolve_theme(42)

	def test_dark_theme_colors(self):
		assert DARK_THEME.text_color == '#e0e0e0'
		assert DARK_THEME.plot_background == '#2d2d2d'

	def test_corporate_theme(self):
		assert CORPORATE_THEME.font_family == 'serif'
		assert CORPORATE_THEME.title_size == 16

	def test_apply_matplotlib(self):
		"""Just ensure it doesn't raise."""
		import matplotlib
		matplotlib.use('Agg')
		LIGHT_THEME.apply_matplotlib()

	def test_apply_seaborn(self):
		"""Just ensure it doesn't raise."""
		import matplotlib
		matplotlib.use('Agg')
		DARK_THEME.apply_seaborn()

	def test_plotly_template(self):
		t = DARK_THEME.to_plotly_template()
		assert t['paper_bgcolor'] == '#1e1e1e'
		assert 'colorway' in t
		assert t['font']['color'] == '#e0e0e0'

	def test_custom_theme(self):
		custom = ChartTheme(
			name='brand',
			palette=['#112233', '#445566'],
			background='#fafafa',
		)
		assert len(custom.palette) == 2
		assert custom.background == '#fafafa'
