"""redactor.py — Optional PII / sensitive-data redaction before LLM calls.

Usage::

    import pychartai as pai

    redactor = pai.DataRedactor()
    safe_df = redactor.redact(df)             # auto-detect PII columns
    safe_df = redactor.redact(df, columns=['email', 'phone'])  # explicit

    # Wire into SmartDataFrame so every .chat() call redacts automatically:
    sdf = pai.SmartDataFrame(df, redactor=redactor)

    # Or globally:
    pai.config.set({'redactor': redactor})

Strategies
----------
- ``'hash'``  (default) — replace values with SHA-256 hex[:12]; preserves groupability
- ``'mask'``  — replace values with ``'***'``
- ``'drop'``  — remove the column entirely from the copy sent to the LLM

The original DataFrame is **never mutated**.
"""

from __future__ import annotations

import hashlib
import re
from typing import Dict, List, Optional

import pandas as pd


# Patterns matched against column names (case-insensitive).
# A column whose name matches any pattern is treated as PII.
_DEFAULT_PII_PATTERNS: List[str] = [
    r'\bemail\b',
    r'\bphone\b',
    r'\bmobile\b',
    r'\bssn\b',
    r'\bsocial.?security\b',
    r'\bcredit.?card\b',
    r'\bcard.?number\b',
    r'\biban\b',
    r'\bpassword\b',
    r'\bpasswd\b',
    r'\bpin\b',
    r'\bname\b',
    r'\bfirst.?name\b',
    r'\blast.?name\b',
    r'\bfull.?name\b',
    r'\baddress\b',
    r'\bstreet\b',
    r'\bpostcode\b',
    r'\bzip.?code\b',
    r'\bdate.?of.?birth\b',
    r'\bdob\b',
    r'\bbirthday\b',
    r'\bip.?addr\b',
    r'\blocation\b',
    r'\bgps\b',
    r'\blatitude\b',
    r'\blongitude\b',
    r'\bpassport\b',
    r'\bnational.?id\b',
    r'\bdriving.?licen[cs]e\b',
    r'\bmedical.?record\b',
    r'\bdiagnosis\b',
    r'\bprescription\b',
]


class DataRedactor:
    """Detect and redact PII columns in a DataFrame before sending to the LLM.

    Args:
        strategy:    How to redact detected columns: ``'hash'``, ``'mask'``, or ``'drop'``.
        patterns:    List of regex patterns matched against column names.  Defaults to
                     the built-in list of 30+ PII patterns.  Pass ``[]`` to disable
                     auto-detection (rely on explicit ``columns=`` in :meth:`redact`).
        extra_columns: Additional column names to always redact regardless of name.
    """

    def __init__(
        self,
        strategy: str = 'hash',
        patterns: Optional[List[str]] = None,
        extra_columns: Optional[List[str]] = None,
    ) -> None:
        if strategy not in ('hash', 'mask', 'drop'):
            raise ValueError(f"strategy must be 'hash', 'mask', or 'drop'. Got: {strategy!r}")
        self.strategy = strategy
        self._patterns = [re.compile(p, re.IGNORECASE) for p in (
            _DEFAULT_PII_PATTERNS if patterns is None else patterns
        )]
        self._extra_columns: List[str] = list(extra_columns or [])

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect_pii_columns(self, df: pd.DataFrame) -> List[str]:
        """Return column names that match PII patterns."""
        detected = []
        for col in df.columns:
            col_str = str(col)
            if any(p.search(col_str) for p in self._patterns):
                detected.append(col)
        for col in self._extra_columns:
            if col in df.columns and col not in detected:
                detected.append(col)
        return detected

    def redact(
        self,
        df: pd.DataFrame,
        columns: Optional[List[str]] = None,
        report: bool = False,
    ) -> 'pd.DataFrame | tuple[pd.DataFrame, Dict[str, int]]':
        """Return a copy of *df* with PII columns redacted.

        Args:
            df:       Source DataFrame.  Never mutated.
            columns:  Explicit list of columns to redact.  If None, auto-detect
                      using the configured patterns.
            report:   If True, return a ``(redacted_df, stats)`` tuple where *stats*
                      maps each redacted column name to the count of values changed.

        Returns:
            Redacted DataFrame, or ``(DataFrame, stats_dict)`` when *report=True*.
        """
        target_cols = columns if columns is not None else self.detect_pii_columns(df)
        result = df.copy()
        stats: Dict[str, int] = {}

        for col in target_cols:
            if col not in result.columns:
                continue
            changed = int(result[col].notna().sum())
            stats[col] = changed

            if self.strategy == 'drop':
                result = result.drop(columns=[col])
            elif self.strategy == 'mask':
                result[col] = result[col].where(result[col].isna(), other='***')
            else:  # 'hash'
                result[col] = result[col].apply(_hash_value)

        if report:
            return result, stats
        return result

    def __repr__(self) -> str:
        return f'DataRedactor(strategy={self.strategy!r}, patterns={len(self._patterns)})'


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _hash_value(v: object) -> object:
    """SHA-256 hex[:12] of str(v), preserving None/NaN."""
    if v is None:
        return None
    import math
    try:
        if isinstance(v, float) and math.isnan(v):
            return v
    except (TypeError, ValueError):
        pass
    return hashlib.sha256(str(v).encode()).hexdigest()[:12]
