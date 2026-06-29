"""
llm_chart_examples.py — LLM-powered chart and text-query examples.

Runs natural-language queries against each dataset using pychartai's
SmartDataFrame and generates charts with the requested backend.

Usage::

    python examples/llm_chart_examples.py --model llama3.2 --backend seaborn --dataset sales
    python examples/llm_chart_examples.py --model llama3.2 --backend all --dataset all
    python examples/llm_chart_examples.py --model llama3.2 --dataset sales   # text only
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pandas as pd
import pychartai as pai


# ---------------------------------------------------------------------------
# Dataset definitions
# ---------------------------------------------------------------------------

DATASETS = {
    'sales': {
        'file': 'sales.csv',
        'text_queries': [
            'What is the total revenue (quantity * price) by region?',
            'Which salesperson had the highest total sales value?',
            'What are the top 3 best-selling products by total quantity?',
        ],
        'chart_queries': [
            ('bar',  'Show a bar chart of total revenue by region'),
            ('bar',  'Show a bar chart of total quantity sold per product'),
            ('line', 'Show a line chart of daily total sales over time'),
        ],
    },
    'weather': {
        'file': 'weather.csv',
        'text_queries': [
            'What is the average temperature across all cities?',
            'Which city has the highest average humidity?',
        ],
        'chart_queries': [
            ('bar', 'Show a bar chart of average temperature by city'),
        ],
    },
    'ecommerce': {
        'file': 'ecommerce.csv',
        'text_queries': [
            'What is the total revenue by category?',
            'Which country has the highest number of orders?',
        ],
        'chart_queries': [
            ('bar', 'Show a bar chart of total revenue by category'),
        ],
    },
    'health': {
        'file': 'health.csv',
        'text_queries': [
            'What is the average age of patients?',
            'What is the most common diagnosis?',
        ],
        'chart_queries': [
            ('bar', 'Show a bar chart of patient count by diagnosis'),
        ],
    },
    'energy': {
        'file': 'energy.csv',
        'text_queries': [
            'What is the total energy consumption by source?',
        ],
        'chart_queries': [
            ('bar', 'Show a bar chart of energy consumption by source'),
        ],
    },
    'stocks': {
        'file': 'stocks.csv',
        'text_queries': [
            'Which stock had the highest average closing price?',
        ],
        'chart_queries': [
            ('line', 'Show a line chart of closing prices over time'),
        ],
    },
    'analytics': {
        'file': 'analytics.csv',
        'text_queries': [
            'What is the total number of sessions by channel?',
        ],
        'chart_queries': [
            ('bar', 'Show a bar chart of sessions by channel'),
        ],
    },
}

BACKENDS = ['seaborn', 'matplotlib', 'plotly']


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sep(title: str) -> None:
    print(f'\n{"=" * 60}')
    print(f'  {title}')
    print(f'{"=" * 60}')


def _result_preview(result) -> str:
    s = str(result).replace('\n', ' ')
    return s[:120] + ('...' if len(s) > 120 else '')


def _load_dataset(name: str, data_dir: str) -> pd.DataFrame | None:
    info = DATASETS.get(name)
    if info is None:
        print(f'  [SKIP] Unknown dataset: {name}')
        return None
    path = os.path.join(data_dir, info['file'])
    if not os.path.isfile(path):
        print(f'  [SKIP] File not found: {path}  (run "make prepare-data")')
        return None
    df = pd.read_csv(path)
    print(f'  Loaded {name}: {df.shape[0]} rows × {df.shape[1]} cols — {list(df.columns)}')
    return df


# ---------------------------------------------------------------------------
# Core runner
# ---------------------------------------------------------------------------

def run_dataset(
    name: str,
    df: pd.DataFrame,
    backends: list[str],
    output_dir: str,
    keep: bool,
    text_only: bool = False,
) -> None:
    info = DATASETS[name]

    # --- Text queries ---
    sdf = pai.SmartDataFrame(df)
    print(f'\n  -- Text queries --')
    for q in info['text_queries']:
        print(f'\n  Q: {q}')
        try:
            result = sdf.chat(q)
            print(f'  [OK] {_result_preview(result)}')
        except Exception as exc:
            print(f'  [FAIL] {exc}')

    if text_only:
        return

    # --- Chart queries ---
    for backend in backends:
        print(f'\n  -- Charts ({backend}) --')
        sdf_b = pai.SmartDataFrame(df, chart_library=backend)
        for chart_type, q in info['chart_queries']:
            print(f'\n  Q [{backend}]: {q}')
            try:
                result = sdf_b.chat(q)
                if isinstance(result, str) and os.path.isfile(result):
                    if keep:
                        ext = result.rsplit('.', 1)[-1]
                        dest = os.path.join(
                            output_dir,
                            f'{name}_{chart_type}_{backend}.{ext}',
                        )
                        shutil.copy(result, dest)
                        print(f'  [OK] chart saved -> {dest}')
                    else:
                        print(f'  [OK] chart generated: {os.path.basename(result)}')
                else:
                    print(f'  [OK] {_result_preview(result)}')
            except Exception as exc:
                print(f'  [FAIL] {exc}')


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description='LLM chart examples')
    parser.add_argument('--model',      default='llama3.2')
    parser.add_argument('--backend',    default='all',
                        help='seaborn | matplotlib | plotly | all')
    parser.add_argument('--dataset',    default='sales',
                        help='Dataset name or "all"')
    parser.add_argument('--data-dir',   default='data/use_cases')
    parser.add_argument('--output-dir', default='exports/charts/manual')
    parser.add_argument('--keep',       action='store_true')
    args = parser.parse_args()

    # Resolve backends
    if args.backend == 'all':
        backends: list[str] = BACKENDS
        text_only = False
    elif args.backend in BACKENDS:
        backends = [args.backend]
        text_only = False
    else:
        # No backend arg supplied → text only
        backends = []
        text_only = True

    # Resolve datasets
    dataset_names = list(DATASETS.keys()) if args.dataset == 'all' else [args.dataset]

    print(f'pychartai v{pai.__version__}')
    print(f'Model  : {args.model}  (Ollama)')
    print(f'Backend: {args.backend}')
    print(f'Dataset: {args.dataset}')

    llm = pai.OllamaLLM(model=args.model)
    pai.config.set({'llm': llm})
    os.makedirs(args.output_dir, exist_ok=True)

    for name in dataset_names:
        _sep(f'Dataset: {name}')
        df = _load_dataset(name, args.data_dir)
        if df is None:
            continue
        try:
            run_dataset(
                name, df, backends,
                args.output_dir, args.keep,
                text_only=(text_only or not backends),
            )
        except KeyboardInterrupt:
            print('\nInterrupted.')
            sys.exit(1)
        except Exception as exc:
            print(f'\n[ERROR] dataset {name!r}: {exc}')

    print('\nDone.')


if __name__ == '__main__':
    main()
