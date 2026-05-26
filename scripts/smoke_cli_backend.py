#!/usr/bin/env python3
"""Smoke test for the CLI backend (aide/backend/backend_cli.py).

Exercises three things end-to-end against a real CLI (default: claude):
    1. Plain text round-trip via backend_cli.query directly.
    2. Function-calling via the review_func_spec schema (returns dict).
    3. Top-level aide.backend.query dispatch with AIDE_USE_CLI=1.

Usage:
    # Default (claude CLI; user must be authenticated):
    python scripts/smoke_cli_backend.py

    # Test a different CLI (must be installed + authed):
    AIDE_CLI_AGENT=gemini python scripts/smoke_cli_backend.py

    # Pick a specific model alias the CLI accepts:
    AIDE_SMOKE_MODEL=haiku python scripts/smoke_cli_backend.py

Halts on the first failure; prints what failed and why.
Cost: roughly 1–2 cents on a Claude Pro/Max subscription per full run.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import sys

# Make the test runnable from anywhere within the repo.
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from aide.backend import backend_cli
from aide.backend import query as dispatch_query
from aide.backend.utils import FunctionSpec

# Mirrors the live review_func_spec at aide/agent.py:19-44 (the only func_spec
# AIDE actually uses). Kept locally so this script doesn't pull in the full
# agent module (and its heavy deps) just for the schema.
REVIEW_SCHEMA = {
    "type": "object",
    "properties": {
        "is_bug": {
            "type": "boolean",
            "description": "true if the script execution contained a non-trivial error.",
        },
        "summary": {
            "type": "string",
            "description": "Short summary of the script execution (1-2 sentences).",
        },
        "metric": {
            "type": "number",
            "description": "Validation metric value (use null if absent).",
        },
        "lower_is_better": {
            "type": "boolean",
            "description": "true if a lower metric is better.",
        },
    },
    "required": ["is_bug", "summary", "metric", "lower_is_better"],
}


def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(name)s %(levelname)s] %(message)s",
    )


def _check_cli_installed(agent: str) -> None:
    bin_name = backend_cli.ADAPTERS[agent].bin_name
    if shutil.which(bin_name) is None:
        sys.exit(
            f"FAIL: `{bin_name}` CLI not found on PATH. Install it and "
            f"authenticate (e.g. `{bin_name} /login`) before running this smoke test."
        )


def test_plain_text(model: str) -> None:
    print("\n=== Test 1: plain text round-trip ===")
    out, req_time, in_tok, out_tok, info = backend_cli.query(
        system_message="You are concise.",
        user_message="Reply with exactly: smoke-ok",
        func_spec=None,
        model=model,
    )
    print(f"  output:   {out!r}")
    print(f"  time:     {req_time:.2f}s")
    print(f"  tokens:   in={in_tok} out={out_tok}")
    print(f"  cost USD: {info.get('total_cost_usd', 0):.4f}")
    assert isinstance(out, str), f"expected str, got {type(out).__name__}"
    assert "smoke-ok" in out.lower(), f"expected 'smoke-ok' in output, got: {out!r}"
    print("  PASS")


def test_function_calling(model: str) -> None:
    print("\n=== Test 2: function-calling via JSON schema ===")
    spec = FunctionSpec(
        name="submit_review",
        json_schema=REVIEW_SCHEMA,
        description="Submit a review of the experiment.",
    )
    out, req_time, in_tok, out_tok, info = backend_cli.query(
        system_message=(
            "You evaluate ML experiment runs. Call submit_review with a fair "
            "assessment of the run described below."
        ),
        user_message=(
            "The experiment trained a linear model for 3 epochs. Final "
            "validation MSE was 0.42 (lower is better). No errors in logs."
        ),
        func_spec=spec,
        model=model,
    )
    print(f"  output:   {json.dumps(out, indent=2) if isinstance(out, dict) else out!r}")
    print(f"  time:     {req_time:.2f}s")
    print(f"  tokens:   in={in_tok} out={out_tok}")
    assert isinstance(out, dict), f"expected dict (func_spec), got {type(out).__name__}"
    for key in ("is_bug", "summary", "metric", "lower_is_better"):
        assert key in out, f"missing required key {key!r} in {out}"
    assert isinstance(out["is_bug"], bool), f"is_bug not bool: {out['is_bug']!r}"
    assert isinstance(out["summary"], str), f"summary not str: {out['summary']!r}"
    # metric: float OR None (schema allows null via type "number" interpreted loosely;
    # the int-to-float coercion in backend_cli should make int values floats)
    assert out["metric"] is None or isinstance(out["metric"], float), (
        f"metric not float-or-none after coercion: {out['metric']!r} "
        f"(type={type(out['metric']).__name__})"
    )
    assert isinstance(out["lower_is_better"], bool), (
        f"lower_is_better not bool: {out['lower_is_better']!r}"
    )
    print("  PASS")


def test_dispatcher(model: str) -> None:
    print("\n=== Test 3: top-level dispatch via AIDE_USE_CLI ===")
    # AIDE_USE_CLI is already set if the user invoked with it; we rely on it here.
    if not os.environ.get("AIDE_USE_CLI"):
        print("  AIDE_USE_CLI not set in this process — setting it temporarily")
        os.environ["AIDE_USE_CLI"] = "1"
    out = dispatch_query(
        system_message="Be concise.",
        user_message="Reply with exactly: dispatched",
        model=model,
    )
    print(f"  output: {out!r}")
    assert isinstance(out, str), f"expected str, got {type(out).__name__}"
    assert "dispatched" in out.lower(), f"expected 'dispatched', got: {out!r}"
    print("  PASS")


def main() -> int:
    _setup_logging()
    agent = os.environ.get("AIDE_CLI_AGENT", "claude").lower()
    if agent not in backend_cli.ADAPTERS:
        sys.exit(
            f"FAIL: AIDE_CLI_AGENT={agent!r} unknown. "
            f"Available: {sorted(backend_cli.ADAPTERS.keys())}"
        )

    adapter = backend_cli.ADAPTERS[agent]
    print(f"Smoke test target: agent={agent}, bin={adapter.bin_name}")
    print(f"Adapter test status: {adapter.tested}")
    _check_cli_installed(agent)

    # Sensible cheap default per known-CLI; user can override.
    default_model = {
        "claude": "haiku",
        "codex": None,  # let codex pick its default
        "gemini": "gemini-2.5-flash",
    }.get(agent)
    model = os.environ.get("AIDE_SMOKE_MODEL", default_model)
    print(f"Model: {model or '(adapter default)'}")

    try:
        test_plain_text(model)
        test_function_calling(model)
        test_dispatcher(model)
    except AssertionError as e:
        print(f"\nFAIL: {e}")
        return 1
    except Exception as e:
        print(f"\nERROR ({type(e).__name__}): {e}")
        return 2

    print("\nAll smoke tests passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
