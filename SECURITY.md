# Security Policy

## Supported versions

| Version | Supported |
|---------|-----------|
| 0.4.x   | ✅ Current |
| < 0.4   | ❌ End of life |

---

## Reporting a vulnerability

**Please do not report security vulnerabilities through public GitHub Issues.**

Send a report by email to **cem.akpolat@eficode.com** with:

1. A description of the vulnerability
2. Steps to reproduce (proof-of-concept code is helpful)
3. Potential impact / severity assessment
4. Any suggested mitigation or fix (optional)

You will receive an acknowledgement within **48 hours** and a detailed response within **7 days**.

If the vulnerability is confirmed we will:

- Work with you to understand the full impact
- Prepare and release a patch
- Credit you in the release notes (unless you prefer to remain anonymous)

---

## Security model

### Sandbox execution

pychartai executes LLM-generated Python code in one of two sandboxes:

| Sandbox | How it works | When to use |
|---|---|---|
| `RestrictedSandbox` (default) | [RestrictedPython](https://restrictedpython.readthedocs.io/) — blocks dangerous builtins at the AST level before execution | Production in-process use |
| `DockerSandbox` | Isolated Docker container with `--cap-drop`, `--security-opt no-new-privileges`, `--network=none` | High-security or multi-tenant deployments |

#### Known limitations of RestrictedSandbox

- RestrictedPython blocks common escape vectors but **cannot guarantee** that a sufficiently crafty LLM-generated payload cannot escape.
- `scipy` is **not** in the default allowed-imports whitelist.  Users who opt in via `RestrictedSandbox(allow_imports=(..., 'scipy'))` should be aware that advanced scipy submodules can expose subprocess capabilities.
- For untrusted inputs or multi-tenant deployments use `DockerSandbox`.

### LLM prompt injection

User query text is passed to the LLM without sanitization.  A crafted query could attempt to override the system prompt.  Mitigations:

- The LLM is asked only to generate Python code; it is not given system-level instructions or credentials.
- All generated code runs inside the sandbox; even a successful prompt injection cannot escape the sandbox boundary.

### API keys

- API keys are never stored in `__repr__`, log output, or positional arguments.
- The `.env` file is excluded from version control via `.gitignore`.
- The `.env.example` template contains only placeholder values.

---

## Dependency security

Dependencies are pinned with upper-version bounds (`litellm>=1.35.0,<2`, `RestrictedPython>=7.0,<9`) to prevent silent major-version upgrades from introducing breaking security changes.

We recommend running [pip-audit](https://pypi.org/project/pip-audit/) against your installed environment:

```bash
pip install pip-audit
pip-audit
```
