"""compare_agents_charts.py — Chart generation benchmark: pychartai vs pandasai.

Run with:
    python compare_agents_charts.py --model llama3.2 [--pandasai]

Requirements:
    pip install pychartai[dev]
    ollama pull llama3.2 && ollama serve
"""

import argparse
import os
import sys
import time
from pathlib import Path

import pandas as pd

parser = argparse.ArgumentParser(description='Chart generation benchmark')
parser.add_argument('--model', default='llama3.2', help='Ollama model name')
parser.add_argument('--pandasai', action='store_true', help='Also run pandasai comparison')
parser.add_argument('--csv', default='data/use_cases/sales.csv', help='Dataset path')
parser.add_argument('--backend', default='seaborn', choices=['seaborn', 'matplotlib', 'plotly'])
args = parser.parse_args()

csv_path = Path(args.csv)
if not csv_path.exists():
    print(f'ERROR: dataset not found at {csv_path}')
    sys.exit(1)

df = pd.read_csv(csv_path)
print(f'Dataset: {len(df)} rows × {len(df.columns)} columns')

CHART_QUERIES = [
    f'Plot a bar chart of revenue by region',
    f'Plot a histogram of revenue',
    f'Plot a line chart of revenue over time',
    f'Plot a pie chart of revenue by region',
    f'Plot a scatter chart of revenue vs units',
]

import pychartai as pai

llm = pai.OllamaLLM(model=args.model)
pai.config.set({'llm': llm, 'verbose': False, 'chart_backend': args.backend})
sdf = pai.SmartDataFrame(df)

print(f'\n{"="*60}')
print(f'pychartai  ({args.model}, {args.backend})')
print('='*60)
pc_results = []
for q in CHART_QUERIES:
    t0 = time.monotonic()
    ok = False
    try:
        path = sdf.chat(q)
        latency = time.monotonic() - t0
        ok = bool(path and os.path.isfile(str(path)))
        status = '✅' if ok else '⚠️ (no file)'
        print(f'  {status}  [{latency:.2f}s]  {q[:55]}')
        if ok:
            print(f'         → {path}')
    except Exception as exc:
        latency = time.monotonic() - t0
        print(f'  ❌  [{latency:.2f}s]  {q[:55]}')
        print(f'         Error: {exc}')
    pc_results.append({'query': q, 'ok': ok, 'latency': latency})

pc_pass = sum(1 for r in pc_results if r['ok'])
pc_avg = sum(r['latency'] for r in pc_results) / len(pc_results)
print(f'\nPass rate: {pc_pass}/{len(CHART_QUERIES)}  avg latency: {pc_avg:.2f}s')

if args.pandasai:
    try:
        from pandasai import SmartDataframe as PandasAISmartDf
    except ImportError:
        print('\nERROR: pandasai not installed. Run: pip install pychartai[pandasai]')
        sys.exit(1)

    print(f'\n{"="*60}')
    print(f'pandasai   ({args.model})')
    print('='*60)
    pai_sdf = PandasAISmartDf(df, config={'llm': llm})
    pai_results = []
    for q in CHART_QUERIES:
        t0 = time.monotonic()
        ok = False
        try:
            result = pai_sdf.chat(q)
            latency = time.monotonic() - t0
            ok = bool(result)
            status = '✅' if ok else '⚠️'
            print(f'  {status}  [{latency:.2f}s]  {q[:55]}')
        except Exception as exc:
            latency = time.monotonic() - t0
            print(f'  ❌  [{latency:.2f}s]  {q[:55]}')
        pai_results.append({'query': q, 'ok': ok, 'latency': latency})

    pai_pass = sum(1 for r in pai_results if r['ok'])
    pai_avg = sum(r['latency'] for r in pai_results) / len(pai_results)
    print(f'\nPass rate: {pai_pass}/{len(CHART_QUERIES)}  avg latency: {pai_avg:.2f}s')

    print(f'\n{"="*60}')
    print('SUMMARY')
    print(f'{"="*60}')
    print(f'{"Metric":<30} {"pychartai":>12} {"pandasai":>12}')
    print(f'{"-"*54}')
    print(f'{"Pass rate":<30} {pc_pass:>9}/{len(CHART_QUERIES)} {pai_pass:>9}/{len(CHART_QUERIES)}')
    print(f'{"Avg latency (s)":<30} {pc_avg:>12.2f} {pai_avg:>12.2f}')
