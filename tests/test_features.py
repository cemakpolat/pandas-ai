"""
test_features.py — Unit tests for Skills, Schema, Cache, Pipeline, and Connections.

All tests run without a live LLM connection.

Usage:
    python tests/test_features.py
    python -m pytest tests/test_features.py -v
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import types
import unittest
import io as _io

import pandas as pd

# Ensure src/ is on the path when running directly
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pychartai_core as pai


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sales_df() -> pd.DataFrame:
	return pd.DataFrame({
		'product': ['A', 'B', 'C', 'D'],
		'region':  ['North', 'South', 'North', 'East'],
		'revenue': [1000.0, 2000.0, 1500.0, 3000.0],
		'units':   [10, 20, 15, 30],
	})


# ===========================================================================
# Skills
# ===========================================================================

class TestSkillDataclass(unittest.TestCase):

	def test_auto_name_from_function(self):
		def top_products(df, n=5):
			'''Return top N products by revenue.'''
			return df.nlargest(n, 'revenue')

		s = pai.Skill(func=top_products)
		self.assertEqual(s.name, 'top_products')

	def test_auto_description_from_docstring(self):
		def top_products(df, n=5):
			'''Return top N products by revenue.'''
			return df
		s = pai.Skill(func=top_products)
		self.assertIn('Return top N', s.description)

	def test_override_name_and_description(self):
		def fn(df): return df
		s = pai.Skill(func=fn, name='my_skill', description='Does X')
		self.assertEqual(s.name, 'my_skill')
		self.assertEqual(s.description, 'Does X')

	def test_to_source_contains_def(self):
		def simple(df):
			'''Simple.'''
			return df
		s = pai.Skill(func=simple)
		src = s.to_source()
		self.assertIn('def simple', src)

	def test_to_prompt_fragment(self):
		def top_products(df, n=5):
			'''Return top N products by revenue.'''
			return df
		s = pai.Skill(func=top_products)
		fragment = s.to_prompt_fragment()
		self.assertIn('top_products', fragment)
		self.assertIn('Return top N', fragment)

	def test_skill_is_callable(self):
		def double(x):
			return x * 2
		s = pai.Skill(func=double)
		self.assertEqual(s(5), 10)


class TestSkillDecorator(unittest.TestCase):

	def test_decorator_no_args(self):
		@pai.skill
		def my_fn(df):
			'''My fn.'''
			return df

		self.assertIsInstance(my_fn, pai.Skill)
		self.assertEqual(my_fn.name, 'my_fn')

	def test_decorator_with_args(self):
		@pai.skill(name='custom', description='Custom desc')
		def my_fn(df):
			return df

		self.assertIsInstance(my_fn, pai.Skill)
		self.assertEqual(my_fn.name, 'custom')
		self.assertEqual(my_fn.description, 'Custom desc')

	def test_plain_function_wrap(self):
		def fn(df): return df
		s = pai.skill(fn)
		self.assertIsInstance(s, pai.Skill)


class TestSkillsPromptHelpers(unittest.TestCase):

	def _make_skill(self, name: str, doc: str = 'Does X'):
		# Create a genuinely-named function so inspect.getsource reflects the name
		fn_code = compile(f'def {name}(df): return df', '<generated>', 'exec')
		ns: dict = {}
		exec(fn_code, ns)  # noqa: S102
		fn = ns[name]
		fn.__doc__ = doc
		return pai.Skill(func=fn)

	def test_build_skills_prompt_empty(self):
		from pychartai_core.skills import build_skills_prompt
		result = build_skills_prompt([])
		self.assertEqual(result, '')

	def test_build_skills_prompt_content(self):
		from pychartai_core.skills import build_skills_prompt
		s = self._make_skill('top_products', 'Return top products.')
		result = build_skills_prompt([s])
		self.assertIn('top_products', result)
		self.assertIn('Return top products', result)

	def test_build_skills_preamble_empty(self):
		from pychartai_core.skills import build_skills_preamble
		result = build_skills_preamble([])
		self.assertEqual(result, '')

	def test_build_skills_preamble_content(self):
		from pychartai_core.skills import build_skills_preamble

		# Use a real def so inspect.getsource can find the source code
		@pai.skill
		def top_products(df, n=5):
			'''Return top N products by revenue.'''
			return df.nlargest(n, 'revenue')

		result = build_skills_preamble([top_products])
		self.assertIn('top_products', result)
		self.assertIn('injected skills', result)


class TestSmartDataFrameSkills(unittest.TestCase):

	def setUp(self):
		self.sdf = pai.SmartDataFrame(_sales_df())

	def test_add_skill_with_skill_instance(self):
		@pai.skill
		def top_products(df, n=3):
			'''Return top N.'''
			return df.nlargest(n, 'revenue')

		self.sdf.add_skill(top_products)
		self.assertEqual(len(self.sdf.skills), 1)
		self.assertEqual(self.sdf.skills[0].name, 'top_products')

	def test_add_skill_with_callable(self):
		def flag_outliers(df, col):
			return df[df[col] > df[col].mean() + 2 * df[col].std()]

		self.sdf.add_skill(flag_outliers)
		self.assertEqual(len(self.sdf.skills), 1)

	def test_add_skill_chaining(self):
		def fn1(df): return df
		def fn2(df): return df
		self.sdf.add_skill(fn1).add_skill(fn2)
		self.assertEqual(len(self.sdf.skills), 2)

	def test_remove_skill(self):
		def fn(df): return df
		self.sdf.add_skill(fn)
		self.sdf.remove_skill('fn')
		self.assertEqual(len(self.sdf.skills), 0)

	def test_add_skill_invalid_type(self):
		with self.assertRaises(TypeError):
			self.sdf.add_skill('not_a_callable')


# ===========================================================================
# Schema (Semantic Layer)
# ===========================================================================

class TestColumn(unittest.TestCase):

	def test_basic_fragment(self):
		col = pai.Column(description='Monthly revenue', unit='USD')
		fragment = col.to_prompt_fragment('revenue')
		self.assertIn('revenue', fragment)
		self.assertIn('Monthly revenue', fragment)
		self.assertIn('USD', fragment)

	def test_values_in_fragment(self):
		col = pai.Column(description='Region', values=['North', 'South', 'East'])
		fragment = col.to_prompt_fragment('region')
		self.assertIn('North', fragment)

	def test_dtype_in_fragment(self):
		col = pai.Column(dtype='float', description='Price')
		fragment = col.to_prompt_fragment('price')
		self.assertIn('float', fragment)

	def test_empty_column(self):
		col = pai.Column()
		fragment = col.to_prompt_fragment('col')
		self.assertIn('col', fragment)


class TestSchema(unittest.TestCase):

	def _make_schema(self):
		return pai.Schema(
			name='Sales Data',
			description='Monthly sales from ERP.',
			columns={
				'revenue': pai.Column(description='Monthly revenue', unit='USD'),
				'region':  pai.Column(description='Geographic region',
				                      values=['North', 'South', 'East']),
			},
		)

	def test_to_prompt_fragment_name(self):
		schema = self._make_schema()
		fragment = schema.to_prompt_fragment()
		self.assertIn('Sales Data', fragment)

	def test_to_prompt_fragment_description(self):
		schema = self._make_schema()
		fragment = schema.to_prompt_fragment()
		self.assertIn('Monthly sales from ERP', fragment)

	def test_to_prompt_fragment_columns(self):
		schema = self._make_schema()
		fragment = schema.to_prompt_fragment()
		self.assertIn('revenue', fragment)
		self.assertIn('region', fragment)
		self.assertIn('USD', fragment)
		self.assertIn('North', fragment)

	def test_plain_string_column(self):
		schema = pai.Schema(columns={'price': 'Unit price in EUR'})
		fragment = schema.to_prompt_fragment()
		self.assertIn('Unit price in EUR', fragment)

	def test_empty_schema(self):
		schema = pai.Schema()
		fragment = schema.to_prompt_fragment()
		self.assertIsInstance(fragment, str)


class TestSmartDataFrameSchema(unittest.TestCase):

	def test_set_schema_and_retrieve(self):
		sdf = pai.SmartDataFrame(_sales_df())
		schema = pai.Schema(name='Test')
		sdf.set_schema(schema)
		self.assertIs(sdf.schema, schema)

	def test_set_schema_chaining(self):
		sdf = pai.SmartDataFrame(_sales_df())
		schema = pai.Schema(name='Test')
		returned = sdf.set_schema(schema)
		self.assertIs(returned, sdf)

	def test_schema_default_none(self):
		sdf = pai.SmartDataFrame(_sales_df())
		self.assertIsNone(sdf.schema)


# ===========================================================================
# Cache
# ===========================================================================

class TestResponseCache(unittest.TestCase):

	def setUp(self):
		self.tmp_dir = tempfile.mkdtemp(prefix='pychartai_test_cache_')
		self.cache = pai.ResponseCache(cache_dir=self.tmp_dir)

	def tearDown(self):
		import shutil
		shutil.rmtree(self.tmp_dir, ignore_errors=True)

	def test_miss_returns_none(self):
		result = self.cache.get('unknown query', 'fp')
		self.assertIsNone(result)

	def test_put_and_get(self):
		self.cache.put('what is avg revenue', 'fp123', 'avg revenue is 1875.0')
		result = self.cache.get('what is avg revenue', 'fp123')
		self.assertEqual(result, 'avg revenue is 1875.0')

	def test_different_fingerprint_is_miss(self):
		self.cache.put('query', 'fp1', 'result1')
		result = self.cache.get('query', 'fp2')
		self.assertIsNone(result)

	def test_different_query_is_miss(self):
		self.cache.put('query1', 'fp', 'result1')
		result = self.cache.get('query2', 'fp')
		self.assertIsNone(result)

	def test_clear(self):
		self.cache.put('q1', 'fp', 'r1')
		self.cache.put('q2', 'fp', 'r2')
		removed = self.cache.clear()
		self.assertEqual(removed, 2)
		self.assertEqual(self.cache.size(), 0)

	def test_size(self):
		self.assertEqual(self.cache.size(), 0)
		self.cache.put('q', 'fp', 'r')
		self.assertEqual(self.cache.size(), 1)

	def test_fingerprint_includes_shape(self):
		df = _sales_df()
		fp = pai.ResponseCache.fingerprint(df)
		self.assertIn('4x4', fp)

	def test_fingerprint_includes_columns(self):
		df = _sales_df()
		fp = pai.ResponseCache.fingerprint(df)
		self.assertIn('revenue', fp)


# ===========================================================================
# Pipeline
# ===========================================================================

class TestPipelineContext(unittest.TestCase):

	def test_is_dict_subclass(self):
		from pychartai_core.pipeline import PipelineContext
		ctx = PipelineContext({'a': 1, 'b': 2})
		self.assertIsInstance(ctx, dict)
		self.assertEqual(ctx['a'], 1)


class TestPipelineStep(unittest.TestCase):

	def _make_step(self, side_effect=None):
		from pychartai_core.pipeline import PipelineStep, PipelineContext
		class DoubleStep(PipelineStep):
			def run(self, ctx):
				ctx['count'] = ctx.get('count', 0) + 1
				if side_effect:
					side_effect(ctx)
				return ctx
		return DoubleStep()

	def test_step_enabled_default(self):
		step = self._make_step()
		self.assertTrue(step.enabled)

	def test_step_skip(self):
		step = self._make_step()
		step.skip()
		self.assertFalse(step.enabled)

	def test_step_enable_after_skip(self):
		step = self._make_step()
		step.skip().enable()
		self.assertTrue(step.enabled)


class TestPipeline(unittest.TestCase):

	def _count_step(self):
		from pychartai_core.pipeline import PipelineStep, PipelineContext
		class CountStep(PipelineStep):
			def run(self, ctx):
				ctx['count'] = ctx.get('count', 0) + 1
				return ctx
		return CountStep()

	def test_steps_run_in_order(self):
		from pychartai_core.pipeline import Pipeline, PipelineStep, PipelineContext

		order = []

		class Step(PipelineStep):
			def __init__(self, n):
				self.n = n
			def run(self, ctx):
				order.append(self.n)
				return ctx

		p = Pipeline([Step(1), Step(2), Step(3)])
		p.run({})
		self.assertEqual(order, [1, 2, 3])

	def test_disabled_step_is_skipped(self):
		step = self._count_step()
		step.skip()
		from pychartai_core.pipeline import Pipeline
		p = Pipeline([step])
		ctx = p.run({'count': 0})
		self.assertEqual(ctx['count'], 0)

	def test_add_step_appends(self):
		from pychartai_core.pipeline import Pipeline
		p = Pipeline([])
		p.add_step(self._count_step())
		self.assertEqual(len(p), 1)

	def test_add_step_at_index(self):
		from pychartai_core.pipeline import Pipeline, PipelineStep, PipelineContext

		order = []

		class Tag(PipelineStep):
			def __init__(self, n):
				self.n = n
			def run(self, ctx):
				order.append(self.n)
				return ctx

		p = Pipeline([Tag(1), Tag(3)])
		p.add_step(Tag(2), index=1)
		p.run({})
		self.assertEqual(order, [1, 2, 3])

	def test_remove_step_by_type(self):
		from pychartai_core.pipeline import Pipeline, PipelineStep, PipelineContext

		class SpecialStep(PipelineStep):
			def run(self, ctx):
				return ctx

		p = Pipeline([self._count_step(), SpecialStep(), self._count_step()])
		p.remove_step(SpecialStep)
		self.assertEqual(len(p), 2)

	def test_repr(self):
		from pychartai_core.pipeline import Pipeline
		p = Pipeline([self._count_step()])
		self.assertIn('CountStep', repr(p))


class TestBuiltinSteps(unittest.TestCase):

	def test_validate_input_passes_valid(self):
		from pychartai_core.pipeline import ValidateInput, PipelineContext
		step = ValidateInput()
		ctx = PipelineContext({'df': _sales_df(), 'query': 'What is total revenue?'})
		result = step.run(ctx)
		self.assertIn('original_query', result)

	def test_validate_input_rejects_empty_df(self):
		from pychartai_core.pipeline import ValidateInput, PipelineContext
		step = ValidateInput()
		ctx = PipelineContext({'df': pd.DataFrame(), 'query': 'query'})
		with self.assertRaises(ValueError):
			step.run(ctx)

	def test_validate_input_rejects_empty_query(self):
		from pychartai_core.pipeline import ValidateInput, PipelineContext
		step = ValidateInput()
		ctx = PipelineContext({'df': _sales_df(), 'query': '   '})
		with self.assertRaises(ValueError):
			step.run(ctx)

	def test_inject_schema_prepends_fragment(self):
		from pychartai_core.pipeline import InjectSchema, PipelineContext
		schema = pai.Schema(name='TestSchema', description='A test dataset.')
		step = InjectSchema()
		ctx = PipelineContext({'query': 'What is the average?', 'schema': schema})
		result = step.run(ctx)
		self.assertIn('TestSchema', result['query'])
		self.assertIn('What is the average?', result['query'])

	def test_inject_schema_no_schema_unchanged(self):
		from pychartai_core.pipeline import InjectSchema, PipelineContext
		step = InjectSchema()
		ctx = PipelineContext({'query': 'original', 'schema': None})
		result = step.run(ctx)
		self.assertEqual(result['query'], 'original')

	def test_inject_skills_appends_prompt(self):
		from pychartai_core.pipeline import InjectSkills, PipelineContext

		@pai.skill
		def top_products(df, n=5):
			'''Return top N products.'''
			return df.nlargest(n, 'revenue')

		step = InjectSkills()
		ctx = PipelineContext({'query': 'who is best?', 'skills': [top_products]})
		result = step.run(ctx)
		self.assertIn('top_products', result['query'])
		self.assertIn('top_products', result.get('skill_sources', {}))

	def test_cache_lookup_hit(self):
		tmp = tempfile.mkdtemp()
		try:
			cache = pai.ResponseCache(cache_dir=tmp)
			df = _sales_df()
			# Use the extended fingerprint that CacheLookup now produces
			fp = pai.ResponseCache.fingerprint(df) + '||'
			cache.put('test query', fp, 'cached answer')

			from pychartai_core.pipeline import CacheLookup, PipelineContext
			step = CacheLookup()
			ctx = PipelineContext({
				'df': df,
				'query': 'test query',
				'original_query': 'test query',
				'cache': cache,
			})
			result = step.run(ctx)
			self.assertTrue(result['cache_hit'])
			self.assertEqual(result['result'], 'cached answer')
		finally:
			import shutil
			shutil.rmtree(tmp, ignore_errors=True)

	def test_cache_lookup_miss(self):
		tmp = tempfile.mkdtemp()
		try:
			cache = pai.ResponseCache(cache_dir=tmp)
			from pychartai_core.pipeline import CacheLookup, PipelineContext
			step = CacheLookup()
			ctx = PipelineContext({
				'df': _sales_df(),
				'query': 'unknown query',
				'original_query': 'unknown query',
				'cache': cache,
			})
			result = step.run(ctx)
			self.assertFalse(result.get('cache_hit', False))
			self.assertNotIn('result', result)
		finally:
			import shutil
			shutil.rmtree(tmp, ignore_errors=True)

	def test_cache_store_persists(self):
		tmp = tempfile.mkdtemp()
		try:
			cache = pai.ResponseCache(cache_dir=tmp)
			from pychartai_core.pipeline import CacheStore, PipelineContext
			step = CacheStore()
			df = _sales_df()
			ctx = PipelineContext({
				'df': df,
				'original_query': 'store test',
				'result': '42',
				'cache': cache,
				'cache_hit': False,
			})
			step.run(ctx)
			# CacheStore uses the extended fingerprint (df + backend + schema)
			fp = pai.ResponseCache.fingerprint(df) + '||'
			self.assertEqual(cache.get('store test', fp), '42')
		finally:
			import shutil
			shutil.rmtree(tmp, ignore_errors=True)


class TestDefaultPipeline(unittest.TestCase):

	def test_has_six_steps(self):
		from pychartai_core.pipeline import default_pipeline
		p = default_pipeline()
		self.assertEqual(len(p), 6)

	def test_step_types(self):
		from pychartai_core.pipeline import (
			default_pipeline, ValidateInput, InjectSchema,
			InjectSkills, CacheLookup, CallAnalyzer, CacheStore,
		)
		p = default_pipeline()
		step_types = [type(s) for s in p._steps]
		self.assertIn(ValidateInput, step_types)
		self.assertIn(InjectSchema, step_types)
		self.assertIn(InjectSkills, step_types)
		self.assertIn(CacheLookup, step_types)
		self.assertIn(CallAnalyzer, step_types)
		self.assertIn(CacheStore, step_types)


class TestSmartDataFramePipeline(unittest.TestCase):

	def test_pipeline_property_lazy_creation(self):
		sdf = pai.SmartDataFrame(_sales_df())
		p = sdf.pipeline
		self.assertIsNotNone(p)
		# Second access returns same instance
		self.assertIs(sdf.pipeline, p)

	def test_pipeline_can_be_replaced(self):
		from pychartai_core.pipeline import Pipeline
		sdf = pai.SmartDataFrame(_sales_df())
		custom = Pipeline([])
		sdf.pipeline = custom
		self.assertIs(sdf.pipeline, custom)


# ===========================================================================
# Connections
# ===========================================================================

class TestCSVConnection(unittest.TestCase):

	def setUp(self):
		self.tmp = tempfile.NamedTemporaryFile(
			mode='w', suffix='.csv', delete=False
		)
		_sales_df().to_csv(self.tmp.name, index=False)
		self.tmp.close()

	def tearDown(self):
		os.unlink(self.tmp.name)

	def test_load_returns_dataframe(self):
		conn = pai.CSVConnection(self.tmp.name)
		df = conn.load()
		self.assertIsInstance(df, pd.DataFrame)
		self.assertEqual(len(df), 4)

	def test_repr(self):
		conn = pai.CSVConnection('/path/to/file.csv')
		self.assertIn('CSVConnection', repr(conn))
		self.assertIn('file.csv', repr(conn))


class TestExcelConnection(unittest.TestCase):

	def setUp(self):
		self.tmp = tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False)
		self.tmp.close()
		_sales_df().to_excel(self.tmp.name, index=False)

	def tearDown(self):
		os.unlink(self.tmp.name)

	def test_load_returns_dataframe(self):
		conn = pai.ExcelConnection(self.tmp.name)
		df = conn.load()
		self.assertIsInstance(df, pd.DataFrame)
		self.assertEqual(len(df), 4)


class TestJSONConnection(unittest.TestCase):

	def setUp(self):
		self.tmp = tempfile.NamedTemporaryFile(
			mode='w', suffix='.json', delete=False
		)
		_sales_df().to_json(self.tmp.name, orient='records', indent=2)
		self.tmp.close()

	def tearDown(self):
		os.unlink(self.tmp.name)

	def test_load_returns_dataframe(self):
		conn = pai.JSONConnection(self.tmp.name)
		df = conn.load()
		self.assertIsInstance(df, pd.DataFrame)
		self.assertEqual(len(df), 4)


class TestParquetConnection(unittest.TestCase):

	def setUp(self):
		self.tmp = tempfile.NamedTemporaryFile(suffix='.parquet', delete=False)
		self.tmp.close()
		_sales_df().to_parquet(self.tmp.name, index=False)

	def tearDown(self):
		os.unlink(self.tmp.name)

	def test_load_returns_dataframe(self):
		conn = pai.ParquetConnection(self.tmp.name)
		df = conn.load()
		self.assertIsInstance(df, pd.DataFrame)
		self.assertEqual(len(df), 4)


class TestSQLConnection(unittest.TestCase):

	def setUp(self):
		self.tmp_db = tempfile.NamedTemporaryFile(suffix='.sqlite', delete=False)
		self.tmp_db.close()
		try:
			import sqlalchemy
			engine = sqlalchemy.create_engine(f'sqlite:///{self.tmp_db.name}')
			_sales_df().to_sql('sales', engine, index=False, if_exists='replace')
			self.have_sqlalchemy = True
		except ImportError:
			self.have_sqlalchemy = False

	def tearDown(self):
		os.unlink(self.tmp_db.name)

	def test_requires_query_or_table(self):
		with self.assertRaises(ValueError):
			pai.SQLConnection('sqlite://')

	def test_load_from_table(self):
		if not self.have_sqlalchemy:
			self.skipTest('sqlalchemy not installed')
		conn = pai.SQLConnection(f'sqlite:///{self.tmp_db.name}', table='sales')
		df = conn.load()
		self.assertIsInstance(df, pd.DataFrame)
		self.assertEqual(len(df), 4)

	def test_load_from_query(self):
		if not self.have_sqlalchemy:
			self.skipTest('sqlalchemy not installed')
		conn = pai.SQLConnection(
			f'sqlite:///{self.tmp_db.name}',
			query='SELECT product, revenue FROM sales',
		)
		df = conn.load()
		self.assertIsInstance(df, pd.DataFrame)
		self.assertIn('revenue', df.columns)


class TestConnectHelper(unittest.TestCase):

	def setUp(self):
		self.tmp = tempfile.NamedTemporaryFile(
			mode='w', suffix='.csv', delete=False
		)
		_sales_df().to_csv(self.tmp.name, index=False)
		self.tmp.close()

	def tearDown(self):
		os.unlink(self.tmp.name)

	def test_connect_returns_smart_dataframe(self):
		conn = pai.CSVConnection(self.tmp.name)
		sdf = pai.connect(conn)
		self.assertIsInstance(sdf, pai.SmartDataFrame)

	def test_connect_with_schema(self):
		schema = pai.Schema(name='Sales', columns={'revenue': 'Monthly revenue USD'})
		conn = pai.CSVConnection(self.tmp.name)
		sdf = pai.connect(conn, schema=schema)
		self.assertIsNotNone(sdf.schema)
		self.assertEqual(sdf.schema.name, 'Sales')

	def test_sdf_data_correct(self):
		conn = pai.CSVConnection(self.tmp.name)
		sdf = pai.connect(conn)
		self.assertEqual(sdf.shape, (4, 4))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _count_and_run() -> tuple[int, int]:
	loader = unittest.TestLoader()
	suite = loader.loadTestsFromModule(sys.modules[__name__])
	runner = unittest.TextTestRunner(verbosity=2)
	result = runner.run(suite)
	total = result.testsRun
	failed = len(result.failures) + len(result.errors)
	return total, failed


if __name__ == '__main__':
	total, failed = _count_and_run()
	passed = total - failed
	print(f'\n{"=" * 60}')
	print(f'  Results: {passed} passed, {failed} failed out of {total}')
	print(f'{"=" * 60}')
	sys.exit(0 if failed == 0 else 1)
