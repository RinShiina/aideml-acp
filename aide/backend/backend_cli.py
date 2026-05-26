"""Backend for driving coding-agent CLI tools (claude, codex, gemini) as an LLM backend.

This lets subscription users (Claude Pro/Max, ChatGPT Plus/Pro, Gemini Pro) run AIDE
without API keys by subprocess-calling the locally-installed agent CLI in one-shot
(--print / exec) mode.

Activation: set ``AIDE_USE_CLI=1`` in the environment. The dispatcher in
``aide/backend/__init__.py`` then routes calls here regardless of model name.

Adapter selection: ``AIDE_CLI_AGENT`` env var picks which CLI to drive
(``claude`` / ``codex`` / ``gemini``; default ``claude``).

Adapter status (READ before relying on each):
    - claude  : tested locally with ``claude --version`` 2.1.150
    - codex   : UNTESTED — based on ``codex exec`` docs as of 2026-05; flags marked
                with ``# TODO(codex):`` may need adjustment when actually run
    - gemini  : UNTESTED — based on ``gemini -p`` docs as of 2026-05; tool-disable
                mechanism is uncertain (see ``# TODO(gemini):`` markers)

Design notes:
    - Process-wide tempfile.mkdtemp() reserved as subprocess ``cwd`` so the agent
      cannot auto-discover the user's real ``CLAUDE.md`` / ``.claude/`` and pollute
      its context.
    - Subscription auth: NEVER pass ``--bare`` to claude (it disables OAuth/keychain
      reads, defeating the whole subscription-friendly point).
    - ``temperature`` / ``max_tokens`` kwargs from AIDE are dropped at debug level —
      CLI tools have no flags for these; defaults apply.
    - Token counts are returned when CLI exposes them (claude/gemini do; codex
      doesn't reliably). The wrapper at ``__init__.py:67-72`` discards them anyway.
"""

from __future__ import annotations

import atexit
import json
import logging
import os
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass
from typing import Callable

from .utils import FunctionSpec, OutputType

logger = logging.getLogger("aide")


# Per-process scratch directory for subprocess ``cwd``.
# Created lazily; cleaned at interpreter exit.
_TEMP_CWD: str | None = None


def _get_temp_cwd() -> str:
    global _TEMP_CWD
    if _TEMP_CWD is None:
        _TEMP_CWD = tempfile.mkdtemp(prefix="aide-cli-")
        atexit.register(lambda: shutil.rmtree(_TEMP_CWD, ignore_errors=True))
    return _TEMP_CWD


# Internal shape returned by an adapter's ``parse_output``:
#   (text_or_dict, in_tokens, out_tokens, info_dict)
ParsedReply = tuple[OutputType, int, int, dict]


@dataclass
class CLIAdapter:
    """Per-CLI plug for the shared subprocess runner.

    Attributes:
        bin_name: CLI binary to look up on PATH (e.g. ``"claude"``).
        build_args: ``(system_message, user_message, model, func_spec, model_kwargs)``
            → ``argv`` list to pass to ``subprocess.run``.
        prompt_via_stdin: If True, send ``user_message`` as stdin bytes and exclude
            it from argv. If False, the adapter is responsible for placing the user
            prompt in argv and stdin will be ``DEVNULL`` (avoids the well-known
            codex stdin-hanging issue).
        parse_output: ``(stdout, stderr, returncode, func_spec)`` → ``ParsedReply``.
            Raises ``RuntimeError`` on fatal CLI errors; raises with the literal
            substring ``"non-JSON"`` to trigger the schema-retry path in ``query``.
        supports_json_schema: True if the CLI can enforce a JSON Schema on its
            output natively (claude ``--json-schema``, codex ``--output-schema``).
        tested: human-readable test status, surfaced in errors so users know which
            adapters are battle-tested vs. only-on-paper.
    """

    bin_name: str
    build_args: Callable[..., list[str]]
    prompt_via_stdin: bool
    parse_output: Callable[..., ParsedReply]
    supports_json_schema: bool
    tested: str


def _coerce_metric_to_float(d: dict) -> dict:
    """AIDE's ``parse_exec_result`` (agent.py:324) expects ``metric`` to be a
    float; JSON parsing turns whole-number floats into ints. Coerce to keep
    the downstream ``isinstance(x, float)`` check happy."""
    if isinstance(d, dict) and "metric" in d and isinstance(d["metric"], int):
        d["metric"] = float(d["metric"])
    return d


def _strip_fences(text: str) -> str:
    """Strip ```json ... ``` fences and surrounding whitespace if present."""
    s = text.strip()
    if s.startswith("```"):
        # Remove opening fence (possibly with language tag) and closing fence
        lines = s.split("\n")
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        s = "\n".join(lines).strip()
    return s


# ─────────────────────────────────────────────────────────────────────────────
# Adapter: claude (Anthropic Claude Code CLI) — TESTED
# ─────────────────────────────────────────────────────────────────────────────

def _claude_build_args(
    system_message: str | None,
    user_message: str | None,
    model: str | None,
    func_spec: FunctionSpec | None,
    model_kwargs: dict,
) -> list[str]:
    # NEVER pass --bare: it forces ANTHROPIC_API_KEY auth and disables OAuth/keychain.
    args = [
        "claude",
        "-p",
        "--output-format", "json",
        "--tools", "",                  # don't let the model execute tools
        "--no-session-persistence",     # each call is fresh
    ]
    if model:
        args.extend(["--model", model])
    if system_message:
        args.extend(["--system-prompt", system_message])
    if func_spec is not None:
        args.extend(["--json-schema", json.dumps(func_spec.json_schema)])
    if (mb := model_kwargs.get("max_budget_usd")) is not None:
        args.extend(["--max-budget-usd", str(mb)])
    if (fm := model_kwargs.get("fallback_model")) is not None:
        args.extend(["--fallback-model", str(fm)])
    return args


def _claude_parse_output(
    stdout: bytes, stderr: bytes, returncode: int, func_spec: FunctionSpec | None
) -> ParsedReply:
    raw = stdout.decode("utf-8", errors="replace")
    # claude -p --output-format json always emits a JSON envelope, even on
    # auth errors. Try to parse before treating non-zero exit as fatal.
    try:
        envelope = json.loads(raw)
    except json.JSONDecodeError:
        err = stderr.decode("utf-8", errors="replace")[:2000]
        raise RuntimeError(
            f"claude CLI: non-JSON output (exit {returncode}). stderr: {err}"
        )

    if envelope.get("is_error"):
        msg = envelope.get("result") or envelope.get("api_error_status") or "unknown"
        # Auth/quota errors are fatal; transient errors should be retried by caller.
        raise RuntimeError(f"claude CLI error: {msg}")

    text = envelope.get("result", "")
    usage = envelope.get("usage", {}) or {}
    in_tokens = int(usage.get("input_tokens", 0) or 0)
    out_tokens = int(usage.get("output_tokens", 0) or 0)

    model_used = None
    if mu := envelope.get("modelUsage"):
        model_used = next(iter(mu.keys()), None)
    info = {
        "stop_reason": envelope.get("stop_reason"),
        "model": model_used,
        "total_cost_usd": envelope.get("total_cost_usd", 0.0),
    }

    if func_spec is not None:
        # When --json-schema is used, claude returns the parsed structured object
        # in envelope["structured_output"], and envelope["result"] is empty. Prefer
        # the structured field; fall back to parsing "result" text if it's missing
        # (some claude versions / paths may differ).
        structured = envelope.get("structured_output")
        if isinstance(structured, dict):
            return _coerce_metric_to_float(structured), in_tokens, out_tokens, info
        candidate = _strip_fences(text)
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError as e:
            raise RuntimeError(
                f"claude CLI: non-JSON for func_spec={func_spec.name}: {text[:500]}"
            ) from e
        return _coerce_metric_to_float(parsed), in_tokens, out_tokens, info

    return text, in_tokens, out_tokens, info


# ─────────────────────────────────────────────────────────────────────────────
# Adapter: codex (OpenAI Codex CLI) — UNTESTED
# ─────────────────────────────────────────────────────────────────────────────
# Based on `codex exec` docs as of 2026-05. Friend with ChatGPT Pro to verify.
# Sources:
#   https://github.com/openai/codex
#   https://developers.openai.com/codex/noninteractive

def _codex_build_args(
    system_message: str | None,
    user_message: str | None,
    model: str | None,
    func_spec: FunctionSpec | None,
    model_kwargs: dict,
) -> list[str]:
    args = ["codex", "exec"]
    if model:
        args.extend(["--model", model])
    # TODO(codex): codex has --sandbox modes but no clean "--tools ''" equivalent.
    # For pure text generation we rely on the prompt itself + the fact that the
    # model has no instruction to invoke tools.
    args.extend(["--sandbox", "read-only"])
    args.extend(["--cwd", _get_temp_cwd()])

    if func_spec is not None:
        # TODO(codex): --output-schema takes a file path, not inline JSON.
        # Write the schema to a temp file under the scratch cwd.
        schema_path = os.path.join(
            _get_temp_cwd(), f"_schema_{func_spec.name}.json"
        )
        with open(schema_path, "w") as f:
            json.dump(func_spec.json_schema, f)
        args.extend(["--output-schema", schema_path])
        args.append("--json")  # JSONL event stream

    # Codex has no documented --system-prompt flag for `exec`.
    # Workaround: prepend the system text to the user prompt as a SYSTEM block.
    prompt = (user_message or "").strip()
    if system_message:
        prompt = f"SYSTEM:\n{system_message.strip()}\n\nUSER:\n{prompt}"
    if func_spec is not None:
        prompt += (
            "\n\nRespond ONLY with a JSON object matching the requested schema. "
            "No prose, no markdown fences."
        )
    args.append(prompt)
    return args


def _codex_parse_output(
    stdout: bytes, stderr: bytes, returncode: int, func_spec: FunctionSpec | None
) -> ParsedReply:
    if returncode != 0:
        err = stderr.decode("utf-8", errors="replace")[:2000]
        raise RuntimeError(f"codex CLI failed (exit {returncode}): {err}")

    raw = stdout.decode("utf-8", errors="replace").strip()

    # If --json was passed, output is JSONL events. Final text is in the last
    # event with a textual payload — schema has churned across versions, so we
    # accept several common shapes.
    final_text = raw
    events = []
    try:
        events = [json.loads(line) for line in raw.split("\n") if line.strip()]
    except (json.JSONDecodeError, ValueError):
        events = []

    if events:
        for ev in events:
            if not isinstance(ev, dict):
                continue
            # TODO(codex): the exact field with the final assistant text varies by
            # release. Probe common keys until something sticks.
            for key in ("text", "content", "output", "final_message", "message"):
                if isinstance(ev.get(key), str) and ev[key]:
                    final_text = ev[key]
                    break

    info = {"model": "codex", "stop_reason": "unknown", "total_cost_usd": 0.0}

    if func_spec is not None:
        candidate = _strip_fences(final_text)
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError as e:
            raise RuntimeError(
                f"codex CLI: non-JSON for func_spec={func_spec.name}: {final_text[:500]}"
            ) from e
        return _coerce_metric_to_float(parsed), 0, 0, info

    return final_text, 0, 0, info


# ─────────────────────────────────────────────────────────────────────────────
# Adapter: gemini (Google Gemini CLI) — UNTESTED
# ─────────────────────────────────────────────────────────────────────────────
# Based on `gemini -p` docs as of 2026-05. User has Gemini Pro subscription
# and plans to verify locally.
# Sources:
#   https://github.com/google-gemini/gemini-cli
#   https://geminicli.com/docs/cli/headless/

def _gemini_build_args(
    system_message: str | None,
    user_message: str | None,
    model: str | None,
    func_spec: FunctionSpec | None,
    model_kwargs: dict,
) -> list[str]:
    args = ["gemini", "--output-format", "json"]
    if model:
        args.extend(["-m", model])

    # TODO(gemini): system prompt is normally injected via GEMINI_SYSTEM_MD env
    # var pointing to a .md file. We side-step that by prepending the system text
    # to the user prompt (simpler, no env management). Acceptable but not ideal —
    # revisit if model accuracy suffers.
    prompt = (user_message or "").strip()
    if system_message:
        prompt = f"SYSTEM:\n{system_message.strip()}\n\nUSER:\n{prompt}"
    if func_spec is not None:
        # TODO(gemini): the CLI has no --json-schema equivalent (feature request
        # #13388). Until that lands, we rely on prompt-engineering the model to
        # output JSON and parse it ourselves.
        schema_text = json.dumps(func_spec.json_schema, indent=2)
        prompt += (
            f"\n\nRespond ONLY with a JSON object matching this schema "
            f"(no prose, no markdown fences):\n{schema_text}"
        )

    args.extend(["-p", prompt])
    return args


def _gemini_parse_output(
    stdout: bytes, stderr: bytes, returncode: int, func_spec: FunctionSpec | None
) -> ParsedReply:
    if returncode != 0:
        err = stderr.decode("utf-8", errors="replace")[:2000]
        raise RuntimeError(f"gemini CLI failed (exit {returncode}): {err}")

    raw = stdout.decode("utf-8", errors="replace")

    # gemini --output-format json envelope:
    # {"response": "...", "stats": {"inputTokens": ..., "outputTokens": ...}, "error": null}
    try:
        envelope = json.loads(raw)
    except json.JSONDecodeError:
        # Fall back to treating stdout as plain text (some versions don't honor --output-format).
        text = raw.strip()
        info = {"model": "gemini", "stop_reason": "unknown", "total_cost_usd": 0.0}
        if func_spec is not None:
            candidate = _strip_fences(text)
            try:
                return _coerce_metric_to_float(json.loads(candidate)), 0, 0, info
            except json.JSONDecodeError as e:
                raise RuntimeError(
                    f"gemini CLI: non-JSON for func_spec={func_spec.name}: {text[:500]}"
                ) from e
        return text, 0, 0, info

    if envelope.get("error"):
        raise RuntimeError(f"gemini CLI error: {envelope['error']}")

    text = envelope.get("response", "")
    stats = envelope.get("stats", {}) or {}
    in_tokens = int(stats.get("inputTokens", 0) or 0)
    out_tokens = int(stats.get("outputTokens", 0) or 0)
    info = {
        "model": "gemini",
        "stop_reason": "unknown",
        "total_cost_usd": 0.0,
    }

    if func_spec is not None:
        candidate = _strip_fences(text)
        try:
            return _coerce_metric_to_float(json.loads(candidate)), in_tokens, out_tokens, info
        except json.JSONDecodeError as e:
            raise RuntimeError(
                f"gemini CLI: non-JSON for func_spec={func_spec.name}: {text[:500]}"
            ) from e

    return text, in_tokens, out_tokens, info


# ─────────────────────────────────────────────────────────────────────────────
# Registry
# ─────────────────────────────────────────────────────────────────────────────

ADAPTERS: dict[str, CLIAdapter] = {
    "claude": CLIAdapter(
        bin_name="claude",
        build_args=_claude_build_args,
        prompt_via_stdin=True,
        parse_output=_claude_parse_output,
        supports_json_schema=True,
        tested="claude v2.1.150 (Claude Code) — primary target",
    ),
    "codex": CLIAdapter(
        bin_name="codex",
        build_args=_codex_build_args,
        prompt_via_stdin=False,  # codex exec hangs if stdin is piped; use DEVNULL
        parse_output=_codex_parse_output,
        supports_json_schema=True,  # via --output-schema
        tested="UNTESTED — based on `codex exec` docs as of 2026-05",
    ),
    "gemini": CLIAdapter(
        bin_name="gemini",
        build_args=_gemini_build_args,
        prompt_via_stdin=False,  # gemini uses -p flag for the prompt
        parse_output=_gemini_parse_output,
        supports_json_schema=False,  # gemini CLI has no custom-schema flag yet
        tested="UNTESTED — based on `gemini -p` docs as of 2026-05",
    ),
}


# ─────────────────────────────────────────────────────────────────────────────
# Public entry point — matches the contract of other backends
# (see aide/backend/__init__.py wrapper at __init__.py:67-72)
# ─────────────────────────────────────────────────────────────────────────────

def query(
    system_message: str | None,
    user_message: str | None,
    func_spec: FunctionSpec | None = None,
    **model_kwargs,
) -> tuple[OutputType, float, int, int, dict]:
    """Drive a coding-agent CLI to produce a response.

    Returns the same 5-tuple shape as other backends:
        ``(output, req_time_sec, in_tokens, out_tokens, info_dict)``

    ``output`` is ``str`` for plain text and ``dict`` when ``func_spec`` requests
    structured output. ``info_dict`` includes ``stop_reason``, ``model``, and
    ``total_cost_usd`` when the CLI exposes them.
    """
    agent_name = os.environ.get("AIDE_CLI_AGENT", "claude").lower()
    if agent_name not in ADAPTERS:
        raise RuntimeError(
            f"AIDE_CLI_AGENT={agent_name!r} unknown. "
            f"Available adapters: {sorted(ADAPTERS.keys())}"
        )

    adapter = ADAPTERS[agent_name]

    if shutil.which(adapter.bin_name) is None:
        raise RuntimeError(
            f"`{adapter.bin_name}` CLI not found on PATH. Install it and ensure it is "
            f"authenticated for your subscription. Adapter status: {adapter.tested}"
        )

    # Pop CLI-specific kwargs before passing the rest to the adapter.
    cli_timeout = int(model_kwargs.pop("cli_timeout", 600))

    # Drop kwargs with no CLI flag equivalent (api-only).
    for stripped in ("temperature", "max_tokens"):
        if stripped in model_kwargs:
            logger.debug(
                "CLI backend (%s): dropping kwarg %s=%r — no CLI flag equivalent",
                agent_name, stripped, model_kwargs[stripped],
            )
            model_kwargs.pop(stripped)

    model = model_kwargs.pop("model", None)

    t0 = time.time()
    output: OutputType = ""
    in_tokens = 0
    out_tokens = 0
    info: dict = {}

    # We retry ONCE when func_spec parsing fails — most often the model returned
    # JSON with stray prose; an explicit reminder usually fixes it.
    max_attempts = 2 if func_spec is not None else 1
    last_error: Exception | None = None
    success = False
    effective_user = user_message

    for attempt in range(1, max_attempts + 1):
        argv = adapter.build_args(system_message, effective_user, model, func_spec, model_kwargs)
        logger.debug("CLI backend argv (attempt %d): %s", attempt, " ".join(repr(a) for a in argv))

        try:
            if adapter.prompt_via_stdin:
                stdin_bytes: bytes | None = (effective_user or "").encode("utf-8")
                stdin_kw: dict = {"input": stdin_bytes}
            else:
                stdin_kw = {"stdin": subprocess.DEVNULL}

            result = subprocess.run(
                argv,
                capture_output=True,
                cwd=_get_temp_cwd(),
                timeout=cli_timeout,
                **stdin_kw,
            )
        except subprocess.TimeoutExpired as e:
            last_error = e
            logger.warning(
                "CLI %s timed out after %ds (attempt %d/%d)",
                adapter.bin_name, cli_timeout, attempt, max_attempts,
            )
            continue

        try:
            output, in_tokens, out_tokens, info = adapter.parse_output(
                result.stdout, result.stderr, result.returncode, func_spec
            )
            success = True
            break
        except RuntimeError as e:
            last_error = e
            # Retry only schema-parse failures (and only when we have a func_spec);
            # fatal errors (auth, network) re-raise immediately.
            if func_spec is not None and "non-JSON" in str(e) and attempt < max_attempts:
                logger.warning(
                    "CLI %s: schema parse failed (%s); retrying with reminder",
                    adapter.bin_name, str(e)[:200],
                )
                reminder = (
                    "\n\nIMPORTANT: Respond ONLY with a single JSON object matching the "
                    "requested schema. No prose, no explanation, no markdown fences."
                )
                effective_user = (user_message or "") + reminder
                continue
            raise

    if not success:
        raise RuntimeError(
            f"CLI {adapter.bin_name} failed after {max_attempts} attempt(s): {last_error}"
        ) from last_error

    req_time = time.time() - t0

    logger.info(
        "CLI %s (%s) - %.2fs - %d tokens (in:%d, out:%d) - $%.4f",
        adapter.bin_name,
        info.get("model") or "?",
        req_time,
        in_tokens + out_tokens,
        in_tokens, out_tokens,
        info.get("total_cost_usd") or 0.0,
    )

    return output, req_time, in_tokens, out_tokens, info
