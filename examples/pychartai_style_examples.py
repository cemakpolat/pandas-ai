"""
pychartai-style examples — tests the public API end-to-end with Ollama.

Scenarios
---------
  single    SmartDataFrame(df).chat()
  read-csv  pai.read_csv(path).chat()
  multi     SmartDataFrame with extra_dfs
  all       Runs all three scenarios

Usage::

    python examples/pychartai_style_examples.py --model llama3.2 --backend seaborn --scenario all
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pandas as pd
import pychartai as pai


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sep(title: str) -> None:
    print(f'\n{"=" * 60}')
    print(f'  {title}')
    print(f'{"=" * 60}')


def _ok(label: str, result) -> None:
    preview = str(result)[:120].replace('\n', ' ')
    print(f'  [OK] {label}: {preview}')


def _configure(model: str, backend: str) -> None:
    llm = pai.OllamaLLM(model=model)
    pai.config.set({'llm': llm, 'chart_backend': backend})
    print(f'  LLM  : {model}  (Ollama)')
    print(f'  Chart: {backend}')


def _save_or_discard(output_dir: str, filename: str, keep: bool) -> None:
    path = os.path.join(output_dir, filename)
    if not keep and os.path.isfile(path):
        os.remove(path)


# ---------------------------------------------------------------------------
# Scenario: single DataFrame — SmartDataFrame(df).chat()
# ---------------------------------------------------------------------------

def scenario_single(model: str, backend: str, output_dir: str, keep: bool) -> None:
    _sep('Scenario: single  (SmartDataFrame.chat)')
    _configure(model, backend)

    df = pd.DataFrame({
        'month':   ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun'],
        'revenue': [12000, 15000, 13500, 17000, 16000, 19500],
        'costs':   [8000,  9500,  8800,  10200, 9800,  11000],
        'region':  ['North', 'South', 'East', 'West', 'North', 'South'],
    })

    sdf = pai.SmartDataFrame(df, chart_library=backend)

    queries = [
        ('text',  'What is the total revenue across all months?'),
        ('text',  'Which month had the highest profit (revenue minus costs)?'),
        ('chart', f'Show a bar chart of revenue by month'),
    ]

    os.makedirs(output_dir, exist_ok=True)

    for kind, q in queries:
        print(f'\n  Q: {q}')
        try:
            result = sdf.chat(q)
            _ok(kind, result)
            if kind == 'chart' and isinstance(result, str) and os.path.isfile(result):
                dest = os.path.join(output_dir, 'single_revenue_chart.' + result.rsplit('.', 1)[-1])
                if keep:
                    import shutil
                    shutil.copy(result, dest)
                    print(f'  Saved -> {dest}')
        except Exception as exc:
            print(f'  [FAIL] {exc}')

    print('\n  single scenario DONE')


# ---------------------------------------------------------------------------
# Scenario: read-csv — pai.read_csv(path).chat()
# ---------------------------------------------------------------------------

def scenario_read_csv(model: str, backend: str, output_dir: str, keep: bool) -> None:
    _sep('Scenario: read-csv  (pai.read_csv → SmartDataFrame.chat)')
    _configure(model, backend)

    csv_path = os.path.join(
        os.path.dirname(__file__), '..', 'data', 'use_cases', 'sales.csv'
    )
    csv_path = os.path.normpath(csv_path)

    if not os.path.isfile(csv_path):
        print(f'  [SKIP] CSV not found: {csv_path}')
        print('  Run "make prepare-data" first to generate datasets.')
        return

    print(f'  Loading: {csv_path}')
    sdf = pai.read_csv(csv_path, chart_library=backend)
    print(f'  Rows: {len(sdf)}, Columns: {list(sdf.columns)}')

    queries = [
        'What are the top 3 best-selling products?',
        'What is the total revenue by region?',
        f'Show a bar chart of total sales by product',
    ]

    os.makedirs(output_dir, exist_ok=True)

    for q in queries:
        print(f'\n  Q: {q}')
        try:
            result = sdf.chat(q)
            _ok('result', result)
        except Exception as exc:
            print(f'  [FAIL] {exc}')

    print('\n  read-csv scenario DONE')


# ---------------------------------------------------------------------------
# Scenario: multi — multiple DataFrames via extra_dfs
# ---------------------------------------------------------------------------

def scenario_multi(model: str, backend: str, output_dir: str, keep: bool) -> None:
    _sep('Scenario: multi  (SmartDataFrame + extra_dfs)')
    _configure(model, backend)

    sales_df = pd.DataFrame({
        'product':  ['Widget A', 'Widget B', 'Gadget X', 'Gadget Y'],
        'units':    [320, 180, 450, 90],
        'revenue':  [9600, 7200, 22500, 5400],
        'category': ['Widget', 'Widget', 'Gadget', 'Gadget'],
    })

    costs_df = pd.DataFrame({
        'product':       ['Widget A', 'Widget B', 'Gadget X', 'Gadget Y'],
        'cogs':          [4000, 3500, 10000, 2800],
        'marketing':     [800, 600, 2000, 400],
    })

    sdf = pai.SmartDataFrame(sales_df, chart_library=backend)

    queries = [
        'What is the total revenue for each category in the sales data?',
        'Using the sales data, which product has the highest revenue per unit?',
        'Join sales and costs to calculate profit (revenue - cogs) per product.',
    ]

    os.makedirs(output_dir, exist_ok=True)

    for q in queries:
        print(f'\n  Q: {q}')
        try:
            result = sdf.chat(q, extra_dfs={'costs': costs_df})
            _ok('result', result)
        except Exception as exc:
            print(f'  [FAIL] {exc}')

    print('\n  multi scenario DONE')


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description='pychartai-style examples')
    parser.add_argument('--model',      default='llama3.2')
    parser.add_argument('--backend',    default='seaborn',
                        choices=['seaborn', 'matplotlib', 'plotly'])
    parser.add_argument('--scenario',   default='all',
                        choices=['all', 'single', 'read-csv', 'multi'])
    parser.add_argument('--output-dir', default='exports/charts/pychartai_style')
    parser.add_argument('--keep',       action='store_true',
                        help='Keep generated chart files after the run')
    args = parser.parse_args()

    print(f'pychartai v{pai.__version__}  |  model={args.model}  backend={args.backend}  scenario={args.scenario}')

    runners = {
        'single':   scenario_single,
        'read-csv': scenario_read_csv,
        'multi':    scenario_multi,
    }

    to_run = list(runners.keys()) if args.scenario == 'all' else [args.scenario]

    for name in to_run:
        try:
            runners[name](args.model, args.backend, args.output_dir, args.keep)
        except KeyboardInterrupt:
            print('\nInterrupted.')
            sys.exit(1)
        except Exception as exc:
            print(f'\n[ERROR] scenario {name!r} failed: {exc}')

    print('\nDone.')


if __name__ == '__main__':
    main()
