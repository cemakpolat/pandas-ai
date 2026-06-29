"""compare_agents_extensive.py — NL analytics benchmark: pychartai vs pandasai.

Run with:
    python compare_agents_extensive.py --model llama3.2 [--pandasai]

Requirements:
    pip install pychartai[dev]    # includes pandasai
    ollama pull llama3.2 && ollama serve

This script benchmarks 12 natural-language analytics queries against the
built-in sales dataset.  Pandasai comparison requires pychartai[pandasai].
"""

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description='NL analytics benchmark')
parser.add_argument('--model', default='llama3.2', help='Ollama model name')
parser.add_argument('--pandasai', action='store_true', help='Also run pandasai comparison')
parser.add_argument('--csv', default='data/use_cases/sales.csv', help='Dataset path')
args = parser.parse_args()

# ---------------------------------------------------------------------------
# Load dataset
# ---------------------------------------------------------------------------
csv_path = Path(args.csv)
if not csv_path.exists():
    print(f'ERROR: dataset not found at {csv_path}')
    print('Run: make prepare-data  (or specify --csv <path>)')
    sys.exit(1)

df = pd.read_csv(csv_path)
print(f'Dataset: {len(df)} rows × {len(df.columns)} columns — {list(df.columns)}')

# ---------------------------------------------------------------------------
# Queries
# ---------------------------------------------------------------------------
QUERIES = [
    'What is the total revenue?',
    'What is the average revenue by region?',
    'Which region has the highest revenue?',
    'How many unique products are there?',
    'What is the total revenue per product?',
    'What is the average units sold?',
    'List the top 3 regions by revenue',
    'What is the revenue for the North region?',
    'What percentage of total revenue comes from each region?',
    'Which product generated the most revenue?',
    'What is the median revenue?',
    'How many rows are in the dataset?',
]

# ---------------------------------------------------------------------------
# pychartai benchmark
# ---------------------------------------------------------------------------
import pychartai as pai

llm = pai.OllamaLLM(model=args.model)
pai.config.set({'llm': llm, 'verbose': False})
sdf = pai.SmartDataFrame(df)

print(f'\n{"="*60}')
print(f'pychartai  ({args.model})')
print('='*60)
pc_results = []
for q in QUERIES:
    t0 = time.monotonic()
    ok = False
    try:
        result = sdf.chat(q)
        ok = bool(result and len(str(result)) > 0)
        latency = time.monotonic() - t0
        status = '✅' if ok else '⚠️ (empty)'
        print(f'  {status}  [{latency:.2f}s]  {q[:55]}')
        print(f'         → {str(result)[:80]}')
    except Exception as exc:
        latency = time.monotonic() - t0
        print(f'  ❌  [{latency:.2f}s]  {q[:55]}')
        print(f'         Error: {exc}')
    pc_results.append({'query': q, 'ok': ok, 'latency': latency})

pc_correct = sum(1 for r in pc_results if r['ok'])
pc_avg_lat = sum(r['latency'] for r in pc_results) / len(pc_results)
print(f'\nScore: {pc_correct}/{len(QUERIES)}  avg latency: {pc_avg_lat:.2f}s')

# ---------------------------------------------------------------------------
# pandasai comparison (optional)
# ---------------------------------------------------------------------------
if args.pandasai:
    try:
        from pandasai import SmartDataframe as PandasAISmartDf
        from pandasai.llm.langchain import LangChain
    except ImportError:
        print('\nERROR: pandasai not installed. Run: pip install pychartai[pandasai]')
        sys.exit(1)

    print(f'\n{"="*60}')
    print(f'pandasai   ({args.model})')
    print('='*60)
    pai_agent = PandasAISmartDf(df, config={'llm': llm})
    pai_results = []
    for q in QUERIES:
        t0 = time.monotonic()
        ok = False
        try:
            result = pai_agent.chat(q)
            ok = bool(result and len(str(result)) > 0)
            latency = time.monotonic() - t0
            status = '✅' if ok else '⚠️ (empty)'
            print(f'  {status}  [{latency:.2f}s]  {q[:55]}')
        except Exception as exc:
            latency = time.monotonic() - t0
            print(f'  ❌  [{latency:.2f}s]  {q[:55]}')
        pai_results.append({'query': q, 'ok': ok, 'latency': latency})

    pai_correct = sum(1 for r in pai_results if r['ok'])
    pai_avg_lat = sum(r['latency'] for r in pai_results) / len(pai_results)
    print(f'\nScore: {pai_correct}/{len(QUERIES)}  avg latency: {pai_avg_lat:.2f}s')

    # Summary table
    print(f'\n{"="*60}')
    print('SUMMARY')
    print(f'{"="*60}')
    print(f'{"Metric":<30} {"pychartai":>12} {"pandasai":>12}')
    print(f'{"-"*54}')
    print(f'{"Correctness":<30} {pc_correct:>10}/{len(QUERIES)} {pai_correct:>10}/{len(QUERIES)}')
    print(f'{"Avg latency (s)":<30} {pc_avg_lat:>12.2f} {pai_avg_lat:>12.2f}')
