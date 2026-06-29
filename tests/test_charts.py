"""
tests/test_charts.py — Direct visualization unit tests (no LLM required).

Tests all 9 chart functions across all 3 backends:
  - seaborn   (seaborn + matplotlib, static PNG)
  - plotly    (interactive HTML/PNG via kaleido)
  - matplotlib (raw matplotlib, static PNG)

Run with:
    python tests/test_charts.py                   # all backends
    python tests/test_charts.py --backend seaborn
    python tests/test_charts.py --backend plotly
    python tests/test_charts.py --backend matplotlib
    python tests/test_charts.py --keep            # don't delete output files
"""

import sys
import os
import argparse
import tempfile
import traceback

# Ensure the package is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pandas as pd
import numpy as np

from pychartai_core.visualization import (
    bar_chart,
    line_chart,
    scatter_chart,
    histogram,
    heatmap,
    box_chart,
    violin_chart,
    pie_chart,
    area_chart,
    count_chart,
    kde_chart,
    strip_chart,
    regression_chart,
    pairplot_chart,
    stacked_bar_chart,
    bubble_chart,
    step_chart,
    swarm_chart,
    ecdf_chart,
    funnel_chart,
)

# ---------------------------------------------------------------------------
# Sample data
# ---------------------------------------------------------------------------

def _make_sales_df() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    regions = ["North", "South", "East", "West"]
    n = 200
    return pd.DataFrame({
        "region":   rng.choice(regions, n),
        "channel":  rng.choice(["Online", "Store"], n),
        "revenue":  rng.uniform(500, 5000, n).round(2),
        "units":    rng.integers(1, 50, n),
        "month":    rng.integers(1, 13, n),
    })


def _make_numeric_df() -> pd.DataFrame:
    rng = np.random.default_rng(7)
    n = 300
    return pd.DataFrame({
        "a": rng.normal(0, 1, n),
        "b": rng.normal(2, 1.5, n),
        "c": rng.uniform(0, 10, n),
        "d": rng.normal(1, 0.5, n),
    })


def _make_category_totals() -> pd.DataFrame:
    return pd.DataFrame({
        "category": ["Electronics", "Clothing", "Food", "Books", "Sports"],
        "total":    [4500, 3200, 2800, 1500, 2100],
    })


def _make_funnel_df() -> pd.DataFrame:
    return pd.DataFrame({
        "stage":  ["Awareness", "Interest", "Consideration", "Purchase", "Retention"],
        "count":  [10000, 6200, 3400, 1800, 950],
    })


# ---------------------------------------------------------------------------
# Test runner helpers
# ---------------------------------------------------------------------------

PASS  = "PASS"
FAIL  = "FAIL"
SKIP  = "SKIP"

_results: list[tuple[str, str, str, str]] = []  # (status, backend, chart, detail)


def _run(label: str, backend: str, fn, *args, output_file: str, **kwargs) -> bool:
    """
    Call *fn* and record pass/fail.  Returns True on success.
    """
    try:
        path = fn(*args, output_file=output_file, backend=backend, **kwargs)
        exists = os.path.isfile(path)
        file_size = os.path.getsize(path) if exists else 0
        if exists and file_size > 0:
            _results.append((PASS, backend, label, path))
            print(f"  {PASS}  [{backend:12s}]  {label}")
            return True
        elif exists:
            _results.append((FAIL, backend, label, f"empty file: {path}"))
            print(f"  {FAIL}  [{backend:12s}]  {label}  — empty file: {path}")
            return False
        else:
            _results.append((FAIL, backend, label, f"file not found: {path}"))
            print(f"  {FAIL}  [{backend:12s}]  {label}  — file not found: {path}")
            return False
    except Exception as exc:
        detail = f"{type(exc).__name__}: {exc}"
        _results.append((FAIL, backend, label, detail))
        print(f"  {FAIL}  [{backend:12s}]  {label}  — {detail}")
        if os.getenv("VERBOSE"):
            traceback.print_exc()
        return False


# ---------------------------------------------------------------------------
# Chart tests
# ---------------------------------------------------------------------------

def run_tests(backend: str, outdir: str) -> int:
    """Run all chart tests for *backend*.  Returns number of failures."""
    sales  = _make_sales_df()
    nums   = _make_numeric_df()
    totals = _make_category_totals()
    funnel = _make_funnel_df()

    def out(name: str) -> str:
        ext = ".html" if backend == "plotly" else ".png"
        return os.path.join(outdir, f"{backend}_{name}{ext}")

    print(f"\n{'─'*60}")
    print(f"  Backend: {backend.upper()}")
    print(f"{'─'*60}")

    tests = [
        # (label,  fn,           positional-args,  kwargs)
        ("bar_chart",
         bar_chart,
         (sales, "region", "revenue"),
         {"title": "Revenue by Region", "output_file": out("bar")}),

        ("bar_chart_hue",
         bar_chart,
         (sales, "region", "revenue"),
         {"hue": "channel", "title": "Revenue by Region & Channel",
          "output_file": out("bar_hue")}),

        ("line_chart",
         line_chart,
         (sales, "month", "revenue"),
         {"title": "Revenue over Months", "output_file": out("line")}),

        ("line_chart_hue",
         line_chart,
         (sales, "month", "revenue"),
         {"hue": "channel", "title": "Revenue by Channel", "output_file": out("line_hue")}),

        ("scatter_chart",
         scatter_chart,
         (nums, "a", "b"),
         {"title": "A vs B", "output_file": out("scatter")}),

        ("scatter_chart_hue",
         scatter_chart,
         (sales, "units", "revenue"),
         {"hue": "channel", "title": "Units vs Revenue", "output_file": out("scatter_hue")}),

        ("histogram",
         histogram,
         (sales, "revenue"),
         {"title": "Revenue Distribution", "bins": 25, "output_file": out("hist")}),

        ("histogram_hue",
         histogram,
         (sales, "revenue"),
         {"hue": "channel", "title": "Revenue by Channel", "output_file": out("hist_hue")}),

        ("heatmap",
         heatmap,
         (nums,),
         {"title": "Correlation Heatmap", "output_file": out("heatmap")}),

        ("box_chart",
         box_chart,
         (sales, "region", "revenue"),
         {"title": "Revenue by Region", "output_file": out("box")}),

        ("violin_chart",
         violin_chart,
         (sales, "region", "revenue"),
         {"title": "Revenue Distribution", "output_file": out("violin")}),

        ("pie_chart",
         pie_chart,
         (totals, "category", "total"),
         {"title": "Sales by Category", "output_file": out("pie")}),

        ("area_chart",
         area_chart,
         (sales, "month", "revenue"),
         {"title": "Revenue Area", "output_file": out("area")}),

        ("count_chart",
         count_chart,
         (sales, "region"),
         {"title": "Transactions by Region", "output_file": out("count")}),

        ("count_chart_hue",
         count_chart,
         (sales, "region"),
         {"hue": "channel", "title": "Transactions by Region & Channel",
          "output_file": out("count_hue")}),

        ("kde_chart",
         kde_chart,
         (sales, "revenue"),
         {"title": "Revenue Density", "output_file": out("kde")}),

        ("kde_chart_hue",
         kde_chart,
         (sales, "revenue"),
         {"hue": "channel", "title": "Revenue Density by Channel",
          "output_file": out("kde_hue")}),

        ("strip_chart",
         strip_chart,
         (sales, "region", "revenue"),
         {"title": "Revenue by Region (strip)", "output_file": out("strip")}),

        ("regression_chart",
         regression_chart,
         (sales, "units", "revenue"),
         {"title": "Units vs Revenue (regression)", "output_file": out("regression")}),

        ("regression_chart_hue",
         regression_chart,
         (sales, "units", "revenue"),
         {"hue": "channel", "title": "Units vs Revenue by Channel",
          "output_file": out("regression_hue")}),

        ("pairplot_chart",
         pairplot_chart,
         (nums,),
         {"title": "Pairplot of Numeric Features", "output_file": out("pairplot")}),

        ("pairplot_chart_hue",
         pairplot_chart,
         (sales,),
         {"hue": "channel", "title": "Pairplot by Channel",
          "output_file": out("pairplot_hue")}),

        ("stacked_bar_chart",
         stacked_bar_chart,
         (sales, "region", "revenue", "channel"),
         {"title": "Revenue by Region & Channel (stacked)",
          "output_file": out("stacked_bar")}),

        ("stacked_bar_chart_normalize",
         stacked_bar_chart,
         (sales, "region", "revenue", "channel"),
         {"normalize": True, "title": "Revenue Share by Region (100%)",
          "output_file": out("stacked_bar_norm")}),

        ("bubble_chart",
         bubble_chart,
         (sales, "units", "revenue", "units"),
         {"title": "Units vs Revenue (bubble)",
          "output_file": out("bubble")}),

        ("bubble_chart_hue",
         bubble_chart,
         (sales, "units", "revenue", "units"),
         {"hue": "channel", "title": "Units vs Revenue by Channel (bubble)",
          "output_file": out("bubble_hue")}),

        ("step_chart",
         step_chart,
         (sales, "month", "revenue"),
         {"title": "Revenue Step", "output_file": out("step")}),

        ("step_chart_hue",
         step_chart,
         (sales, "month", "revenue"),
         {"hue": "channel", "title": "Revenue Step by Channel",
          "output_file": out("step_hue")}),

        ("swarm_chart",
         swarm_chart,
         (sales, "region", "revenue"),
         {"title": "Revenue by Region (swarm)",
          "output_file": out("swarm")}),

        ("ecdf_chart",
         ecdf_chart,
         (sales, "revenue"),
         {"title": "Revenue ECDF", "output_file": out("ecdf")}),

        ("ecdf_chart_hue",
         ecdf_chart,
         (sales, "revenue"),
         {"hue": "channel", "title": "Revenue ECDF by Channel",
          "output_file": out("ecdf_hue")}),

        ("funnel_chart",
         funnel_chart,
         (funnel, "stage", "count"),
         {"title": "Conversion Funnel", "output_file": out("funnel")}),
    ]

    failures = 0
    for label, fn, args, kwargs in tests:
        output_file = kwargs.pop("output_file")
        ok = _run(label, backend, fn, *args, output_file=output_file, **kwargs)
        if not ok:
            failures += 1

    if backend == "seaborn":
        for chart_name in ("pie", "area"):
            chart_path = out(chart_name)
            exists = os.path.isfile(chart_path)
            file_size = os.path.getsize(chart_path) if exists else 0
            check_label = f"seaborn_parity_{chart_name}"
            if exists and file_size > 0:
                _results.append((PASS, backend, check_label, f"{chart_path} ({file_size} bytes)"))
                print(f"  {PASS}  [{backend:12s}]  {check_label}")
            else:
                failures += 1
                detail = "file not found" if not exists else f"empty file: {chart_path}"
                _results.append((FAIL, backend, check_label, detail))
                print(f"  {FAIL}  [{backend:12s}]  {check_label}  — {detail}")

    return failures


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Test pychartai visualization functions across backends."
    )
    parser.add_argument(
        "--backend",
        choices=["seaborn", "plotly", "matplotlib", "all"],
        default="all",
        help="Backend to test (default: all)",
    )
    parser.add_argument(
        "--keep",
        action="store_true",
        help="Keep generated chart files after the test run",
    )
    parser.add_argument(
        "--outdir",
        default="exports/charts/test",
        help="Output directory for test charts (default: exports/charts/test)",
    )
    args = parser.parse_args()

    backends = (
        ["seaborn", "plotly", "matplotlib"]
        if args.backend == "all"
        else [args.backend]
    )

    outdir = args.outdir
    os.makedirs(outdir, exist_ok=True)

    print(f"\nChart Backend Test Suite")
    print(f"Output dir  : {outdir}")
    print(f"Backends    : {', '.join(backends)}")

    total_failures = 0
    for backend in backends:
        total_failures += run_tests(backend, outdir)

    # Summary
    total   = len(_results)
    passed  = sum(1 for r in _results if r[0] == PASS)
    failed  = sum(1 for r in _results if r[0] == FAIL)

    print(f"\n{'='*60}")
    print(f"  Results: {passed} passed, {failed} failed out of {total}")
    print(f"{'='*60}")

    if failed:
        print("\nFailed tests:")
        for status, backend, chart, detail in _results:
            if status == FAIL:
                print(f"  [{backend}] {chart}: {detail}")

    if not args.keep:
        # Clean up generated test files
        for _, _, _, detail in _results:
            if os.path.isfile(detail):
                try:
                    os.remove(detail)
                except OSError:
                    pass
        # Remove test outdir if empty
        try:
            os.rmdir(outdir)
        except OSError:
            pass

    sys.exit(0 if total_failures == 0 else 1)


if __name__ == "__main__":
    main()
