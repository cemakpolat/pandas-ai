"""
io.py — Data-loading helpers that return SmartDataFrame instances.

These mirror the pandas top-level reader functions but return
``SmartDataFrame`` instead of ``pd.DataFrame``, enabling ``.chat()``
immediately after loading.
"""

from __future__ import annotations

from typing import Any
import pandas as pd

from .smart_df import SmartDataFrame


def read_csv(filepath_or_buffer: Any, **kwargs) -> SmartDataFrame:
    """Read a CSV file into a SmartDataFrame.

    Accepts all keyword arguments supported by :func:`pandas.read_csv`.

    Example::

        df = pai.read_csv("data/sales.csv")
        df = pai.read_csv("data/sales.csv", parse_dates=["date"])
        print(df.chat("What is the average revenue by region?"))
    """
    return SmartDataFrame(pd.read_csv(filepath_or_buffer, **kwargs))


def read_excel(filepath_or_buffer: Any, **kwargs) -> SmartDataFrame:
    """Read an Excel file into a SmartDataFrame.

    Accepts all keyword arguments supported by :func:`pandas.read_excel`.

    Example::

        df = pai.read_excel("data/report.xlsx", sheet_name="Q1")
    """
    return SmartDataFrame(pd.read_excel(filepath_or_buffer, **kwargs))


def read_json(filepath_or_buffer: Any, **kwargs) -> SmartDataFrame:
    """Read a JSON file or string into a SmartDataFrame.

    Accepts all keyword arguments supported by :func:`pandas.read_json`.

    Example::

        df = pai.read_json("data/records.json", orient="records")
    """
    return SmartDataFrame(pd.read_json(filepath_or_buffer, **kwargs))


def read_parquet(filepath_or_buffer: Any, **kwargs) -> SmartDataFrame:
    """Read a Parquet file into a SmartDataFrame.

    Accepts all keyword arguments supported by :func:`pandas.read_parquet`.

    Example::

        df = pai.read_parquet("data/warehouse.parquet")
    """
    return SmartDataFrame(pd.read_parquet(filepath_or_buffer, **kwargs))


def DataFrame(data: Any = None, **kwargs) -> SmartDataFrame:
    """Wrap an existing pandas DataFrame (or any DataFrame-constructible data).

    If *data* is already a ``pd.DataFrame``, it is wrapped directly.
    Otherwise, it is passed to ``pd.DataFrame(data, **kwargs)``.

    Example::

        import pandas as pd
        raw = pd.read_csv("data.csv")
        df = pai.DataFrame(raw)
        df.chat("Summarise the data")

        # Also accepts dicts, lists, etc.
        df = pai.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        df.chat("What is the sum of a?")
    """
    if isinstance(data, pd.DataFrame):
        return SmartDataFrame(data)
    return SmartDataFrame(pd.DataFrame(data, **kwargs))
