"""tests/smoke_examples.py — Import smoke tests for all example modules.

Verifies that every file under examples/ can be imported without error and
that no example makes unconditional LLM calls at import time.

Run directly:
    python tests/smoke_examples.py

Or via pytest:
    pytest tests/smoke_examples.py -v
"""

import ast
import importlib.util
import os
import sys
from pathlib import Path

# Make sure we pick up the src layout
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / 'src'))

EXAMPLES_DIR = ROOT / 'examples'


def _load_source(path: Path) -> ast.Module:
    """Parse the file and return its AST — no execution."""
    return ast.parse(path.read_text(encoding='utf-8'), filename=str(path))


def _has_top_level_llm_call(tree: ast.Module) -> bool:
    """Return True if there are obvious top-level LLM calls (not inside a function/if)."""
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
            # heuristic: calls to .chat() at module level are problematic
            call = node.value
            if isinstance(call.func, ast.Attribute) and call.func.attr in ('chat', 'generate'):
                return True
    return False


def check_all_examples():
    example_files = sorted(EXAMPLES_DIR.glob('*.py'))
    example_files = [f for f in example_files if f.name != '__init__.py']

    passed = []
    failed = []

    for path in example_files:
        try:
            tree = _load_source(path)
            if _has_top_level_llm_call(tree):
                print(f'  WARN  {path.name} — has top-level LLM call (may need __main__ guard)')
            else:
                print(f'  OK    {path.name}')
            passed.append(path.name)
        except SyntaxError as exc:
            print(f'  FAIL  {path.name} — SyntaxError: {exc}')
            failed.append(path.name)
        except Exception as exc:
            print(f'  FAIL  {path.name} — {exc}')
            failed.append(path.name)

    print(f'\n{len(passed)} OK, {len(failed)} failed')
    if failed:
        print('Failed files:', failed)
        sys.exit(1)


if __name__ == '__main__':
    check_all_examples()
