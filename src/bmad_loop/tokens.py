"""Token accounting from coding-CLI transcript files.

One parser per CLI family, selected by the profile's `usage_parser` key. All
parsers are tolerant — unknown lines and shapes are skipped, never fatal; a
transcript that yields nothing reads as None (untracked), not zero.

- claude-jsonl:  ~/.claude/projects/<munged-cwd>/<session-id>.jsonl;
                 assistant entries carry an API `usage` block, summed.
- codex-rollout: ~/.codex/sessions/YYYY/MM/DD/rollout-*.jsonl; `token_count`
                 event payloads carry CUMULATIVE totals, so the last one is
                 the session usage (sessions here are fresh per task).
- gemini-chat:   ~/.gemini/tmp/<project>/chats/session-*.jsonl — a JSONL
                 patch stream: bare message entries (and messages inside
                 "$set" patches) carry a per-API-call `tokens` block
                 {input, output, cached, thoughts, tool, total}; the same
                 message id is re-emitted as it accretes content, so the
                 last occurrence per id wins, then unique messages are
                 summed. `input` includes the cached portion.
- copilot-events: ~/.copilot/session-state/<session>/events.jsonl; some
                 entries carry `data.modelMetrics.<model>.usage`
                 {inputTokens, outputTokens, cacheReadTokens, cacheWriteTokens,
                 reasoningTokens} that is CUMULATIVE per model, so the last
                 entry bearing modelMetrics holds the session totals (summed
                 across models). reasoningTokens fold into output.
"""

from __future__ import annotations

import glob
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .model import TokenUsage

# Per-parser transcript-location conventions. `probe.py` imports this same
# dict for its own (unrelated-signature) discovery helper used by
# `bmad-loop probe-adapter`; keep both in lockstep by importing rather than
# duplicating.
TRANSCRIPT_GLOBS = {
    "claude-jsonl": "~/.claude/projects/*/*.jsonl",
    "codex-rollout": "~/.codex/sessions/*/*/*/rollout-*.jsonl",
    "gemini-chat": "~/.gemini/tmp/*/chats/session-*.jsonl",
    "copilot-events": "~/.copilot/session-state/*/events.jsonl",
}


def read_usage(parser: str, transcript_path: Path) -> TokenUsage | None:
    if parser == "claude-jsonl":
        return tally(transcript_path)
    if parser == "codex-rollout":
        return tally_codex_rollout(transcript_path)
    if parser == "gemini-chat":
        return tally_gemini_chat(transcript_path)
    if parser == "copilot-events":
        return tally_copilot_events(transcript_path)
    return None


def _jsonl_entries(transcript_path: Path):
    if not transcript_path.is_file():
        return
    with transcript_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(entry, dict):
                yield entry


def _int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


# ------------------------------------------------------------- claude-jsonl


def _usage_block(entry: dict[str, Any]) -> dict[str, Any] | None:
    message = entry.get("message")
    if isinstance(message, dict) and isinstance(message.get("usage"), dict):
        return message["usage"]
    if isinstance(entry.get("usage"), dict):
        return entry["usage"]
    return None


def tally(transcript_path: Path) -> TokenUsage:
    total = TokenUsage()
    for entry in _jsonl_entries(transcript_path):
        usage = _usage_block(entry)
        if not usage:
            continue
        total.add(
            TokenUsage(
                input_tokens=_int(usage.get("input_tokens")),
                output_tokens=_int(usage.get("output_tokens")),
                cache_read_tokens=_int(usage.get("cache_read_input_tokens")),
                cache_creation_tokens=_int(usage.get("cache_creation_input_tokens")),
            )
        )
    return total


# ----------------------------------------------------------- codex-rollout


def tally_codex_rollout(transcript_path: Path) -> TokenUsage | None:
    last: dict[str, Any] | None = None
    for entry in _jsonl_entries(transcript_path):
        payload = entry.get("payload")
        if not isinstance(payload, dict) or payload.get("type") != "token_count":
            continue
        info = payload.get("info")
        if not isinstance(info, dict):
            continue
        totals = info.get("total_token_usage")
        last = totals if isinstance(totals, dict) else info
    if last is None:
        return None
    cached = _int(last.get("cached_input_tokens"))
    return TokenUsage(
        # Codex's input_tokens includes the cached portion; split it out.
        input_tokens=max(0, _int(last.get("input_tokens")) - cached),
        output_tokens=_int(last.get("output_tokens")),
        cache_read_tokens=cached,
    )


# ------------------------------------------------------------- gemini-chat


def _gemini_messages(entry: dict[str, Any]):
    if "id" in entry:
        yield entry
    set_patch = entry.get("$set")
    if isinstance(set_patch, dict):
        messages = set_patch.get("messages")
        if isinstance(messages, list):
            for message in messages:
                if isinstance(message, dict):
                    yield message


def tally_gemini_chat(transcript_path: Path) -> TokenUsage | None:
    by_id: dict[str, dict[str, Any]] = {}
    for entry in _jsonl_entries(transcript_path):
        for message in _gemini_messages(entry):
            tokens = message.get("tokens")
            message_id = message.get("id")
            if isinstance(tokens, dict) and isinstance(message_id, str):
                by_id[message_id] = tokens
    if not by_id:
        return None
    total = TokenUsage()
    for tokens in by_id.values():
        cached = _int(tokens.get("cached"))
        total.add(
            TokenUsage(
                input_tokens=max(0, _int(tokens.get("input")) - cached),
                output_tokens=_int(tokens.get("output")) + _int(tokens.get("thoughts")),
                cache_read_tokens=cached,
            )
        )
    return total


# --------------------------------------------------------- copilot-events


def tally_copilot_events(transcript_path: Path) -> TokenUsage | None:
    # data.modelMetrics.<model>.usage is cumulative per model, so the LAST entry
    # carrying modelMetrics holds the session totals; sum across its models.
    # (Schema verified against a single Copilot CLI 1.0.63 events.jsonl — revisit
    # the cumulative/multi-model assumption if a newer build disagrees.)
    last: dict[str, Any] | None = None
    for entry in _jsonl_entries(transcript_path):
        data = entry.get("data")
        if not isinstance(data, dict):
            continue
        metrics = data.get("modelMetrics")
        if isinstance(metrics, dict) and metrics:
            last = metrics
    if last is None:
        return None
    total = TokenUsage()
    for model_metrics in last.values():
        if not isinstance(model_metrics, dict):
            continue
        usage = model_metrics.get("usage")
        if not isinstance(usage, dict):
            continue
        total.add(
            TokenUsage(
                input_tokens=_int(usage.get("inputTokens")),
                output_tokens=_int(usage.get("outputTokens")) + _int(usage.get("reasoningTokens")),
                cache_read_tokens=_int(usage.get("cacheReadTokens")),
                cache_creation_tokens=_int(usage.get("cacheWriteTokens")),
            )
        )
    return total


# ------------------------------------------------------- transcript discovery


def _candidate_is_file(path: Path) -> bool:
    try:
        return path.is_file()
    except OSError:
        return False


def _candidate_mtime(path: Path) -> float:
    try:
        return path.stat().st_mtime
    except OSError:
        return 0.0


def _expand_cwd(value: str, home: Path | None) -> str:
    """Expand a leading ``~`` the same way the discovery glob does: substitute
    the test ``home`` override when given, else the real home directory
    (#775 review finding 5) — so a tilde-form cwd anchor still matches the
    absolute paths a real ``session_meta.payload.cwd`` carries."""
    if value.startswith("~"):
        prefix = str(home) if home is not None else os.path.expanduser("~")
        return prefix + value[1:]
    return value


def _strip_extended_windows_prefix(value: str) -> str:
    """Normalize a Windows extended-length path prefix to its ordinary form:
    ``\\\\?\\UNC\\server\\share`` -> ``\\\\server\\share``, ``\\\\?\\C:\\repo``
    -> ``C:\\repo`` (#775 review pass 3 finding 1). ``os.path.normpath``/
    ``normcase`` treat these as distinct strings even though they name the
    same location, so a cwd recorded with one spelling would never match a
    ``payload.cwd`` recorded with the other's. Pure string manipulation —
    a no-op (and safe) on POSIX, where a real cwd never starts this way."""
    if value.startswith("\\\\?\\UNC\\"):
        return "\\\\" + value[len("\\\\?\\UNC\\") :]
    if value.startswith("\\\\?\\"):
        return value[len("\\\\?\\") :]
    return value


def _normalize_cwd(value: str | Path, home: Path | None = None) -> str:
    expanded = _expand_cwd(str(value), home)
    expanded = _strip_extended_windows_prefix(expanded)
    return os.path.normcase(os.path.normpath(expanded))


def _read_first_json_line(path: Path) -> Any:
    try:
        # errors="strict" (the open() default, spelled out here) is
        # deliberate, not an oversight: "replace" turns an invalid byte
        # inside a JSON string into a VALID replacement character, so a
        # corrupt candidate would decode "successfully" and win discovery —
        # only for the later strict tally to find nothing (#775 review pass 2
        # finding 2). Skip the whole candidate on any decode/parse failure
        # instead, exactly like an unreadable or malformed one.
        with path.open(encoding="utf-8", errors="strict") as f:
            line = f.readline()
    except (OSError, UnicodeDecodeError, ValueError):
        return None
    line = line.strip()
    if not line:
        return None
    try:
        return json.loads(line)
    except (json.JSONDecodeError, ValueError, RecursionError, MemoryError):
        # RecursionError: the stdlib json decoder is recursive-descent, so a
        # deeply nested (possibly truncated/partial-write) candidate can blow
        # the interpreter's recursion limit; MemoryError: an implausibly huge
        # structure. Neither is proof the CANDIDATE is ours or isn't — skip
        # it like any other malformed file, never let it abort discovery
        # (#775 review pass 3 finding 2).
        return None


def _newest_candidate(paths: list[Path]) -> Path:
    return max(paths, key=_candidate_mtime)


def _parse_session_start(payload: dict[str, Any]) -> float | None:
    """Epoch seconds from a codex ``session_meta`` payload's ISO-8601
    ``timestamp`` (e.g. ``2026-09-09T14:46:06.000Z``), or None if missing/
    unparseable. This is the session's LAUNCH time — unlike the rollout
    file's mtime, which keeps advancing as the session is appended to and so
    cannot tell a session apart from one that merely ran longer or started
    later in the same shared worktree cwd (#775 follow-up).

    A timestamp with no timezone designator is interpreted as UTC EXPLICITLY
    (never the host's local zone, per #775 review finding 3) — the same
    convention every real codex timestamp on disk already uses via its ``Z``
    suffix; a naive value must not silently pick up a multi-hour host offset.
    """
    ts = payload.get("timestamp")
    if not isinstance(ts, str):
        return None
    v = ts.strip()
    if v.endswith("Z"):
        v = v[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(v)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.timestamp()


def discover_transcript(
    parser: str,
    *,
    session_id: str | None,
    cwd: str | Path | None,
    not_before: float | None,
    home: Path | None = None,
) -> Path | None:
    """Locate ``parser``'s transcript from (session_id, cwd, launch time)
    alone — for a Stop payload that carried no ``transcript_path``, or a
    session rescued post-kill where no Stop arrived at all (#775).

    Pure and tolerant: a bad glob, an unreadable file, a decode failure, or
    malformed JSON is skipped per-candidate, never raised or logged. Never
    guesses a "newest anything" — with neither ``session_id`` nor ``cwd`` to
    anchor on, this returns None rather than attribute an unrelated session's
    transcript.

    Among several same-cwd codex candidates (no filename match), this
    REFUSES AMBIGUITY: a null usage beats a wrong one. Only a candidate whose
    own launch falls within the plausibility window
    ``[not_before-120, not_before+300]`` is even considered (a rollout is
    created at launch, so 120s of clock skew either way plus 5 minutes for a
    slow start is the whole plausible range — an unrelated session sharing
    the cwd 20 minutes later is dropped outright, never merely outscored).
    A single survivor wins unconditionally; among several, the earliest
    start at-or-after ``not_before`` wins, and only when none qualifies does
    the latest start still before it win instead — but either way, a TIE for
    that winning delta/start returns None rather than pick one arbitrarily.
    File mtime is NEVER a decider anywhere in this selection (#775 follow-up
    + review pass 1 finding 1 + review pass 2 finding 1): it tracks how long
    a rollout kept being appended to, not when the session started, so it
    carries no identity evidence a later concurrent session in a shared
    worktree cwd couldn't equally claim.

    ``home``, when given, substitutes for ``~`` both in the convention glob
    (escaped with ``glob.escape`` first, so a literal home path containing
    glob metacharacters like ``[`` still matches — #775 review pass 2 finding
    3) and in a tilde-form ``cwd`` anchor, so tests never touch the real home
    directory and a ``cwd="~/..."`` caller still matches the absolute paths a
    real ``session_meta.payload.cwd`` carries (#775 review pass 1 finding 5).
    """
    pattern = TRANSCRIPT_GLOBS.get(parser)
    if not pattern:
        return None
    # glob.escape the resolved home prefix — a literal directory component
    # containing glob metacharacters (`[home]`, `?`, …) must not be read as
    # glob syntax (#775 review pass 2 finding 3); only the convention
    # suffix's own `*` wildcards are meant to be wildcards.
    home_dir = str(home) if home is not None else os.path.expanduser("~")
    glob_pattern = pattern.replace("~", glob.escape(home_dir), 1)
    try:
        candidates = [Path(p) for p in glob.glob(glob_pattern)]
    except OSError:
        candidates = []
    candidates = [p for p in candidates if _candidate_is_file(p)]

    if not_before is not None:
        threshold = not_before - 120
        candidates = [p for p in candidates if _candidate_mtime(p) >= threshold]

    if session_id:
        name_matches = [p for p in candidates if session_id.lower() in p.name.lower()]
        if name_matches:
            return _newest_candidate(name_matches)

    if parser != "codex-rollout":
        # Other parsers (claude-jsonl, gemini-chat, copilot-events): filename/
        # session_id match only, already tried above; no meta parsing.
        return None

    if cwd is None and session_id is None:
        return None

    cwd_norm = _normalize_cwd(cwd, home) if cwd is not None else None
    meta_matches: list[tuple[Path, float | None]] = []
    for path in candidates:
        meta = _read_first_json_line(path)
        if not isinstance(meta, dict) or meta.get("type") != "session_meta":
            continue
        payload = meta.get("payload")
        if not isinstance(payload, dict):
            continue
        if session_id is not None and payload.get("id") != session_id:
            continue
        if cwd_norm is not None:
            payload_cwd = payload.get("cwd")
            if not isinstance(payload_cwd, str) or _normalize_cwd(payload_cwd, home) != cwd_norm:
                continue
        meta_matches.append((path, _parse_session_start(payload)))
    if not meta_matches:
        return None

    if not_before is None:
        # Weak path: with no launch time to anchor on, the newest matching
        # transcript by mtime is the best available guess — same fallback the
        # pre-#775-follow-up code used unconditionally.
        return _newest_candidate([path for path, _ in meta_matches])

    # Plausibility window (#775 review pass 2 finding 1): a codex rollout file
    # is created right at launch, so ANY candidate is implausible unless its
    # own start falls within [not_before-120, not_before+300] (120s of clock
    # skew either recorded time might lag the other by; up to 5 minutes for a
    # slow start). Candidates outside it — e.g. an unrelated session that
    # merely happens to share the cwd and started 20 minutes later — are
    # dropped before any tie-breaking, so they can never win by default.
    window_lo = not_before - 120
    window_hi = not_before + 300
    windowed = [
        (path, start)
        for path, start in meta_matches
        if start is not None and window_lo <= start <= window_hi
    ]
    if not windowed:
        return None
    if len(windowed) == 1:
        return windowed[0][0]

    # Several plausible candidates: REFUSE ambiguity rather than guess — a
    # null usage beats a wrong one. Never mtime as a decider (a rollout's
    # mtime tracks how long it ran, not when it started, so it carries no
    # identity evidence). Prefer a start at-or-after our launch, earliest
    # first; a tie for that smallest delta is ambiguous and returns None. Only
    # when nothing started at/after not_before does the latest pre-window
    # start win — itself refused (None) on a tie.
    on_or_after = [(path, start) for path, start in windowed if start >= not_before]
    if on_or_after:
        best_delta = min(start - not_before for _, start in on_or_after)
        tied = [path for path, start in on_or_after if start - not_before == best_delta]
        return None if len(tied) > 1 else tied[0]

    best_start = max(start for _, start in windowed)
    tied = [path for path, start in windowed if start == best_start]
    return None if len(tied) > 1 else tied[0]
