import calendar
import json
import os
import sys
import time as time_module
from datetime import datetime

import pytest

from bmad_loop.model import TokenUsage
from bmad_loop.tokens import (
    discover_transcript,
    read_usage,
    tally,
    tally_codex_rollout,
    tally_copilot_events,
    tally_gemini_chat,
)


def test_weighted_total():
    usage = TokenUsage(
        input_tokens=100,
        output_tokens=50,
        cache_read_tokens=1000,
        cache_creation_tokens=10,
    )
    assert usage.weighted_total(0.1) == 100 + 50 + 10 + 100
    assert usage.weighted_total(1.0) == usage.total
    assert usage.weighted_total(0.0) == 160


def test_tally_mixed_shapes(tmp_path):
    lines = [
        # Claude Code shape: usage nested in message
        {
            "type": "assistant",
            "message": {"usage": {"input_tokens": 100, "output_tokens": 50}},
        },
        # cache fields
        {
            "type": "assistant",
            "message": {
                "usage": {
                    "input_tokens": 10,
                    "output_tokens": 5,
                    "cache_read_input_tokens": 2000,
                    "cache_creation_input_tokens": 300,
                }
            },
        },
        # top-level usage shape
        {"type": "message", "usage": {"input_tokens": 1, "output_tokens": 1}},
        # noise: no usage, malformed values tolerated
        {"type": "user", "message": {"content": "hi"}},
        {"type": "summary"},
    ]
    path = tmp_path / "t.jsonl"
    with path.open("w") as f:
        for line in lines:
            f.write(json.dumps(line) + "\n")
        f.write("not json at all\n")
        f.write("\n")

    usage = tally(path)
    assert usage.input_tokens == 111
    assert usage.output_tokens == 56
    assert usage.cache_read_tokens == 2000
    assert usage.cache_creation_tokens == 300
    assert usage.total == 111 + 56 + 2000 + 300


def test_tally_missing_file(tmp_path):
    assert tally(tmp_path / "nope.jsonl").total == 0


def test_codex_rollout_last_cumulative_wins(tmp_path):
    lines = [
        {"type": "session_meta", "payload": {"id": "abc"}},
        # token_count payloads are cumulative; only the last one counts
        {
            "type": "event_msg",
            "payload": {
                "type": "token_count",
                "info": {
                    "total_token_usage": {
                        "input_tokens": 100,
                        "cached_input_tokens": 40,
                        "output_tokens": 10,
                    }
                },
            },
        },
        {"type": "event_msg", "payload": {"type": "agent_message"}},
        {
            "type": "event_msg",
            "payload": {
                "type": "token_count",
                "info": {
                    "total_token_usage": {
                        "input_tokens": 500,
                        "cached_input_tokens": 200,
                        "output_tokens": 60,
                    }
                },
            },
        },
    ]
    path = tmp_path / "rollout.jsonl"
    path.write_text("\n".join(json.dumps(line) for line in lines) + "\nnot json\n")

    usage = tally_codex_rollout(path)
    assert usage.input_tokens == 300  # cached portion split out of input
    assert usage.cache_read_tokens == 200
    assert usage.output_tokens == 60


def test_codex_rollout_without_token_counts_is_none(tmp_path):
    path = tmp_path / "rollout.jsonl"
    path.write_text(json.dumps({"type": "event_msg", "payload": {"type": "agent_message"}}) + "\n")
    assert tally_codex_rollout(path) is None
    assert tally_codex_rollout(tmp_path / "nope.jsonl") is None


def test_gemini_chat_dedupes_reemitted_messages(tmp_path):
    # shape captured from a real ~/.gemini/tmp/<project>/chats/session-*.jsonl
    # (2026-06-11): a JSONL patch stream where the same message id is
    # re-emitted as it accretes content, and `input` includes `cached`.
    lines = [
        {"sessionId": "s1", "projectHash": "x", "kind": "main"},
        {"$set": {"messages": [{"id": "u1", "type": "user", "content": []}]}},
        {
            "id": "g1",
            "type": "gemini",
            "tokens": {
                "input": 12273,
                "output": 45,
                "cached": 0,
                "thoughts": 87,
                "tool": 0,
            },
        },
        {"$set": {"lastUpdated": "..."}},
        # same message re-emitted with toolCalls added: must not double-count
        {
            "id": "g1",
            "type": "gemini",
            "toolCalls": [{}],
            "tokens": {
                "input": 12273,
                "output": 45,
                "cached": 0,
                "thoughts": 87,
                "tool": 0,
            },
        },
        {
            "id": "g2",
            "type": "gemini",
            "tokens": {
                "input": 12429,
                "output": 2,
                "cached": 11367,
                "thoughts": 16,
                "tool": 0,
            },
        },
    ]
    path = tmp_path / "session.jsonl"
    path.write_text("\n".join(json.dumps(line) for line in lines) + "\nnot json\n")

    usage = tally_gemini_chat(path)
    assert usage.input_tokens == 12273 + (12429 - 11367)  # cached split out of input
    assert usage.cache_read_tokens == 11367
    assert usage.output_tokens == (45 + 87) + (2 + 16)  # output + thoughts


def test_gemini_chat_without_tokens_is_none(tmp_path):
    path = tmp_path / "session.jsonl"
    path.write_text(json.dumps({"id": "u1", "type": "user", "content": []}) + "\n")
    assert tally_gemini_chat(path) is None
    assert tally_gemini_chat(tmp_path / "nope.jsonl") is None


def test_copilot_events_last_cumulative_across_models(tmp_path):
    # shape from ~/.copilot/session-state/<session>/events.jsonl: per line
    # {id, type, data:{...}}; data.modelMetrics.<model>.usage is cumulative.
    lines = [
        {"id": "e0", "type": "session_start", "data": {"sessionId": "s1"}},
        # an earlier, smaller cumulative snapshot — superseded by the last one
        {
            "id": "e1",
            "type": "metrics",
            "data": {"modelMetrics": {"gpt-5-mini": {"usage": {"inputTokens": 100}}}},
        },
        {"id": "e2", "type": "message", "data": {"content": "noise"}},
        # final cumulative snapshot, two models — totals come from here
        {
            "id": "e3",
            "type": "metrics",
            "data": {
                "modelMetrics": {
                    "gpt-5-mini": {
                        "usage": {
                            "inputTokens": 500,
                            "outputTokens": 60,
                            "cacheReadTokens": 200,
                            "cacheWriteTokens": 30,
                            "reasoningTokens": 5,
                        }
                    },
                    "gpt-5": {
                        "usage": {
                            "inputTokens": 40,
                            "outputTokens": 8,
                            "reasoningTokens": 2,
                        }
                    },
                }
            },
        },
    ]
    path = tmp_path / "events.jsonl"
    path.write_text("\n".join(json.dumps(line) for line in lines) + "\nnot json\n")

    usage = tally_copilot_events(path)
    assert usage.input_tokens == 540  # 500 + 40
    assert usage.output_tokens == 75  # (60 + 5) + (8 + 2), reasoning folded in
    assert usage.cache_read_tokens == 200
    assert usage.cache_creation_tokens == 30


def test_copilot_events_without_metrics_is_none(tmp_path):
    path = tmp_path / "events.jsonl"
    path.write_text(json.dumps({"id": "e0", "type": "message", "data": {"content": "hi"}}) + "\n")
    assert tally_copilot_events(path) is None
    assert tally_copilot_events(tmp_path / "nope.jsonl") is None


def test_read_usage_dispatch(tmp_path):
    path = tmp_path / "t.jsonl"
    path.write_text(json.dumps({"usage": {"input_tokens": 1, "output_tokens": 2}}) + "\n")
    assert read_usage("claude-jsonl", path).total == 3
    assert read_usage("none", path) is None

    cop = tmp_path / "events.jsonl"
    cop.write_text(
        json.dumps({"data": {"modelMetrics": {"m": {"usage": {"inputTokens": 7}}}}}) + "\n"
    )
    assert read_usage("copilot-events", cop).input_tokens == 7


# ------------------------------------------------------- discover_transcript


def _iso(epoch):
    from datetime import datetime, timezone

    return datetime.fromtimestamp(epoch, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.000Z")


def _write_rollout(path, *, session_id, cwd, mtime, start=None):
    """A fake ``~/.codex/sessions/.../rollout-*.jsonl`` whose first line is a
    real session_meta payload shape (#775 evidence). ``start`` is the
    session's own launch timestamp (``payload.timestamp``, ISO-8601 ``Z``);
    defaults to an unparseable placeholder for tests that don't care about it
    (the weak newest-mtime path, or a filename/session_id short-circuit)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    meta = {
        "timestamp": "2026-09-09T14:46:06.000Z",
        "type": "session_meta",
        "payload": {
            "session_id": session_id,
            "id": session_id,
            "timestamp": start if start is not None else "...",
            "cwd": cwd,
        },
    }
    path.write_text(json.dumps(meta) + "\n")
    os.utime(path, (mtime, mtime))


def test_discover_transcript_session_id_filename_match_wins(tmp_path):
    home = tmp_path / "home"
    base = home / ".codex" / "sessions" / "2026" / "09" / "09"
    target_id = "01a08521-99b8-72a0-b98b-156a48d74ca2"
    target = base / f"rollout-2026-09-09T14-46-06-{target_id}.jsonl"
    _write_rollout(target, session_id=target_id, cwd="/some/project", mtime=1000)
    # A newer, unrelated rollout: must lose to the filename match even though
    # it is the more recent file.
    other_id = "11111111-2222-3333-4444-555555555555"
    other = base / f"rollout-2026-09-09T15-00-00-{other_id}.jsonl"
    _write_rollout(other, session_id=other_id, cwd="/other/project", mtime=5000)

    found = discover_transcript(
        "codex-rollout", session_id=target_id, cwd=None, not_before=None, home=home
    )
    assert found == target


def test_discover_transcript_cwd_match_picks_newest_for_cwd(tmp_path):
    home = tmp_path / "home"
    base = home / ".codex" / "sessions" / "2026" / "09" / "09"
    older_same_cwd = base / "rollout-2026-09-09T14-00-00-aaaaaaaa.jsonl"
    _write_rollout(older_same_cwd, session_id="aaaaaaaa", cwd="/proj/a", mtime=1000)
    newer_same_cwd = base / "rollout-2026-09-09T14-10-00-bbbbbbbb.jsonl"
    _write_rollout(newer_same_cwd, session_id="bbbbbbbb", cwd="/proj/a", mtime=2000)
    newer_other_cwd = base / "rollout-2026-09-09T14-20-00-cccccccc.jsonl"
    _write_rollout(newer_other_cwd, session_id="cccccccc", cwd="/proj/other", mtime=5000)

    found = discover_transcript(
        "codex-rollout", session_id=None, cwd="/proj/a", not_before=None, home=home
    )
    assert found == newer_same_cwd


def test_discover_transcript_picks_earliest_after_launch_not_newest_mtime(tmp_path):
    """#775 follow-up + review pass 2 finding 1: worktrees are shared, so a
    concurrent LATER Codex session in the same cwd must not win the
    read-right-after-end race just because its rollout file has a newer
    mtime (it keeps growing as it runs longer) — the 5-seconds-later session
    is preferred by the "smallest delta among on-or-after candidates" tier.

    Ablation note (corrected, review pass 3 finding 3): this test does NOT
    by itself pin the plausibility window's upper bound. Setting
    ``window_hi = float("inf")`` does not fail it — with both candidates
    still in play, the smallest-delta tier picks the 5-seconds-later one
    anyway (5 < 1200), independent of any window. What actually isolates
    ``window_hi`` is a SOLE distant candidate with nothing competing:
    test_upper_plausibility_bound_refuses_sole_distant_candidate below."""
    home = tmp_path / "home"
    base = home / ".codex" / "sessions" / "2026" / "09" / "09"
    not_before = 1_700_000_000.0

    ours = base / "rollout-2026-09-09T14-46-06-ourssession.jsonl"
    _write_rollout(
        ours,
        session_id="ourssession",
        cwd="/proj/a",
        mtime=not_before + 5,  # deliberately the OLDER of the two mtimes
        start=_iso(not_before + 5),
    )
    later_concurrent = base / "rollout-2026-09-09T15-06-06-laterrun.jsonl"
    _write_rollout(
        later_concurrent,
        session_id="laterrun",
        cwd="/proj/a",
        mtime=not_before + 20 * 60 + 500,  # much newer mtime: ran longer
        start=_iso(not_before + 20 * 60),
    )

    found = discover_transcript(
        "codex-rollout", session_id=None, cwd="/proj/a", not_before=not_before, home=home
    )
    assert found == ours


def test_discover_transcript_excludes_a_session_started_before_launch(tmp_path):
    """A rollout whose OWN launch timestamp is well before ``not_before``
    (here 10 minutes, past the plausibility window's ``-120s`` lower bound)
    is excluded even though its file mtime alone would clear the coarse
    pre-filter — proving the start-based window, not just the pre-existing
    mtime floor, does the excluding."""
    home = tmp_path / "home"
    base = home / ".codex" / "sessions" / "2026" / "09" / "09"
    not_before = 1_700_000_000.0

    too_early = base / "rollout-2026-09-09T14-30-06-tooearly.jsonl"
    _write_rollout(
        too_early,
        session_id="tooearly",
        cwd="/proj/a",
        mtime=not_before + 100,  # clears the coarse mtime pre-filter easily
        start=_iso(not_before - 10 * 60),  # but started 10 minutes too early
    )

    found = discover_transcript(
        "codex-rollout", session_id=None, cwd="/proj/a", not_before=not_before, home=home
    )
    assert found is None


def test_discover_transcript_not_before_excludes_old_match(tmp_path):
    """Ablation: delete the ``if not_before is not None:`` mtime-floor block
    in ``discover_transcript`` and this test fails alone — the excluded
    session_id filename match would be returned instead of None."""
    home = tmp_path / "home"
    base = home / ".codex" / "sessions" / "2026" / "09" / "09"
    session_id = "ffff0000-0000-0000-0000-000000000000"
    old = base / f"rollout-2026-09-09T10-00-00-{session_id}.jsonl"
    _write_rollout(old, session_id=session_id, cwd="/proj/a", mtime=1000)

    # not_before is far past old's mtime + the 120s tolerance, so the only
    # candidate (which matches by filename) must be excluded before matching.
    found = discover_transcript(
        "codex-rollout", session_id=session_id, cwd=None, not_before=4900, home=home
    )
    assert found is None


def test_discover_transcript_no_anchor_returns_none(tmp_path):
    """Never guesses a "newest anything": with neither session_id nor cwd,
    a real candidate on disk must still not be returned.

    Ablation: delete the ``if cwd is None and session_id is None: return
    None`` guard and this test fails alone — the lone candidate on disk
    would be returned instead of None."""
    home = tmp_path / "home"
    base = home / ".codex" / "sessions" / "2026" / "09" / "09"
    _write_rollout(
        base / "rollout-2026-09-09T14-00-00-11112222.jsonl",
        session_id="11112222",
        cwd="/proj/a",
        mtime=1000,
    )

    found = discover_transcript(
        "codex-rollout", session_id=None, cwd=None, not_before=None, home=home
    )
    assert found is None


def test_discover_transcript_skips_corrupt_first_line(tmp_path):
    home = tmp_path / "home"
    base = home / ".codex" / "sessions" / "2026" / "09" / "09"
    base.mkdir(parents=True, exist_ok=True)
    corrupt = base / "rollout-2026-09-09T14-00-00-corrupt.jsonl"
    corrupt.write_text("not json at all\n")
    os.utime(corrupt, (5000, 5000))
    good = base / "rollout-2026-09-09T14-10-00-33334444.jsonl"
    _write_rollout(good, session_id="33334444", cwd="/proj/a", mtime=6000)

    found = discover_transcript(
        "codex-rollout", session_id=None, cwd="/proj/a", not_before=None, home=home
    )
    assert found == good


def test_discover_transcript_unknown_parser_returns_none(tmp_path):
    assert (
        discover_transcript("none", session_id="x", cwd=None, not_before=None, home=tmp_path)
        is None
    )


def test_probe_transcript_globs_imports_from_tokens():
    """probe.py imports TRANSCRIPT_GLOBS from tokens.py rather than
    duplicating it (#775) — same dict object, one source of truth."""
    from bmad_loop import probe
    from bmad_loop.tokens import TRANSCRIPT_GLOBS

    assert probe.TRANSCRIPT_GLOBS is TRANSCRIPT_GLOBS


# --------------------------------------- #775 Codex review findings (P2 x5)


def test_previous_session_cannot_win(tmp_path):
    """Review finding 1: the naive "smallest signed delta" rule let a session
    that started BEFORE not_before (inside the -120s skew window) outscore
    one that started at/after it, because a negative delta can be smaller
    than a positive one. A prior session must never win over the real one.

    Ablation: replace the two-tier selection (on-or-after first, skew window
    only as a fallback) with a single `min(start - not_before)` over every
    start in `[not_before - 120, inf)` and this test fails alone — "previous"
    (60s before) beats "ours" (5s after)."""
    base = tmp_path / ".codex" / "sessions" / "2026" / "09" / "10"
    for name, start in [("previous", 1940), ("ours", 2005)]:
        _write_rollout(
            base / f"rollout-{name}.jsonl",
            session_id=name,
            cwd="/repo",
            mtime=2100,
            start=_iso(start),
        )
    found = discover_transcript(
        "codex-rollout", session_id=None, cwd="/repo", not_before=2000, home=tmp_path
    )
    assert found.name == "rollout-ours.jsonl"


def test_skew_fallback_picks_latest_start_before_launch(tmp_path):
    """When NO candidate started at/after not_before, the bounded 120s
    clock-skew fallback still applies — picking the latest (closest to
    not_before) of the eligible earlier starts, never the earliest."""
    base = tmp_path / ".codex" / "sessions" / "2026" / "09" / "10"
    _write_rollout(
        base / "rollout-far.jsonl",
        session_id="far",
        cwd="/repo",
        mtime=2100,
        start=_iso(1900),  # 100s before not_before: inside the skew window
    )
    _write_rollout(
        base / "rollout-close.jsonl",
        session_id="close",
        cwd="/repo",
        mtime=2100,
        start=_iso(1995),  # 5s before not_before: closer, should win
    )
    found = discover_transcript(
        "codex-rollout", session_id=None, cwd="/repo", not_before=2000, home=tmp_path
    )
    assert found.name == "rollout-close.jsonl"


@pytest.mark.parametrize("starts", [(2005, 2005), (1990, 1990)])
def test_ambiguous_sessions_are_refused(tmp_path, starts):
    """Review pass 2 finding 1: attribution must REFUSE ambiguity — a null
    usage beats a wrong one. Two candidates tied at the same start (whether
    both at/after launch or both before it, within the plausibility window)
    must return None, never break the tie by file mtime — proven here by
    giving "other" the newer mtime in both cases, which must NOT win.

    Adapted from the reviewer's repro: their second case, ``(1995, 3200)``,
    is no longer ambiguous now that the plausibility window (see
    test_discover_transcript_picks_earliest_after_launch_not_newest_mtime)
    drops anything outside ``[not_before-120, not_before+300]`` outright —
    3200 is 1200s after not_before=2000, well past the +300s bound, so it is
    excluded before scoring and "ours" (1995) is the sole, unambiguous
    survivor rather than a tie. Replaced here with a genuine PRE-window tie
    (1990, 1990) so both refusal paths — a tie among on-or-after starts, and
    a tie among pre-window starts — are exercised.

    Ablation: change either ``return None if len(tied) > 1`` to
    ``return _newest_candidate(tied) if len(tied) > 1`` (i.e. restore an
    mtime tie-break) and both parametrizations fail alone — "other" (the
    newer mtime in both cases) would win instead of None."""
    base = tmp_path / ".codex" / "sessions" / "2026" / "09" / "10"
    for name, start, mtime in zip(("ours", "other"), starts, (3300, 3400)):
        _write_rollout(
            base / f"rollout-{name}.jsonl",
            session_id=name,
            cwd="/repo",
            start=_iso(start),
            mtime=mtime,
        )
    assert (
        discover_transcript(
            "codex-rollout", session_id=None, cwd="/repo", not_before=2000, home=tmp_path
        )
        is None
    )


def test_bad_encoding_candidate_is_skipped(tmp_path):
    """Review finding 2: a single unrelated non-UTF-8 rollout must not abort
    discovery — it is skipped like any other unreadable/malformed candidate,
    never raised."""
    p = tmp_path / ".codex" / "sessions" / "2026" / "09" / "10" / "rollout-bad.jsonl"
    p.parent.mkdir(parents=True)
    p.write_bytes(b"\xff\n")
    assert (
        discover_transcript(
            "codex-rollout", session_id=None, cwd="/repo", not_before=None, home=tmp_path
        )
        is None
    )


def test_bad_encoding_candidate_skipped_alongside_a_real_match(tmp_path):
    """The same non-UTF-8 file, but with a real matching candidate also on
    disk: discovery must still find the real one rather than abort."""
    base = tmp_path / ".codex" / "sessions" / "2026" / "09" / "10"
    bad = base / "rollout-bad.jsonl"
    bad.parent.mkdir(parents=True)
    bad.write_bytes(b"\xff\xfe\x00\xff\n")
    good = base / "rollout-good.jsonl"
    _write_rollout(good, session_id="good", cwd="/repo", mtime=2000, start=_iso(2000))

    found = discover_transcript(
        "codex-rollout", session_id=None, cwd="/repo", not_before=2000, home=tmp_path
    )
    assert found == good


def test_corrupt_string_candidate_is_skipped(tmp_path):
    """Review pass 2 finding 2: ``errors="replace"`` (the pass-1 fix) turns
    an invalid byte INSIDE a JSON string into a valid replacement character,
    so a corrupt candidate decodes "successfully" and wins discovery over a
    real one — only for the later strict tally to find nothing. Decoding
    must be strict; a decode failure skips the whole candidate, exactly like
    an unreadable file.

    Ablation: put ``errors="replace"`` back in ``_read_first_json_line`` and
    this test fails alone — "bad" (a real session_meta whose ``id`` field
    holds an invalid UTF-8 byte, corrupted after writing valid JSON) would be
    returned instead of "good"."""
    base = tmp_path / ".codex" / "sessions" / "2026" / "09" / "10"
    bad = base / "rollout-bad.jsonl"
    good = base / "rollout-good.jsonl"
    _write_rollout(bad, session_id="bad", cwd="/repo", mtime=2100, start=_iso(2001))
    _write_rollout(good, session_id="good", cwd="/repo", mtime=2100, start=_iso(2005))
    bad.write_bytes(bad.read_bytes().replace(b'"bad"', b'"\xff"'))

    found = discover_transcript(
        "codex-rollout", session_id=None, cwd="/repo", not_before=2000, home=tmp_path
    )
    assert found == good


class _FakeHostOffsetDatetime(datetime):
    """``datetime`` subclass simulating a host PERMANENTLY at UTC+7 for any
    NAIVE ``.timestamp()`` call, regardless of the REAL host's actual
    timezone (#775 review pass 3 finding 5). A tz-AWARE instance still
    computes the true absolute epoch via the real ``timestamp()`` — only the
    naive branch is faked. Installed by monkeypatching ``tokens.datetime``
    to this class: ``_parse_session_start``'s FIXED implementation always
    attaches ``tzinfo=timezone.utc`` before calling ``.timestamp()`` (so the
    fake offset never actually applies to it), while the OLD host-local-naive
    implementation calls ``.timestamp()`` on a still-naive value and gets
    shifted by the fake 25200s — deterministically, on every platform and
    every real host timezone, including a CI runner that happens to be UTC."""

    _FAKE_HOST_OFFSET_S = 7 * 3600

    def timestamp(self) -> float:
        if self.tzinfo is not None:
            return super().timestamp()
        return calendar.timegm(self.timetuple()) - self._FAKE_HOST_OFFSET_S


def test_parse_session_start_treats_naive_timestamp_as_utc(monkeypatch):
    """Review pass-1 finding 3 + pass-2 P3 + pass-3 P3: a timestamp with no
    timezone designator must be interpreted as UTC explicitly, never the
    host's local zone.

    The original pass-1 form of this test (``with_z == without_z``) is
    VACUOUS on a UTC host: the old host-local-naive implementation also
    passes it there, because "host-local" and "UTC" coincide numerically —
    reviewed and reproduced (pass 2 P3), and STILL vacuous after pass 2's
    ``+07:00``-offset strengthening on a genuinely UTC-hosted platform (pass
    3 P3, since that leg is unconditionally tz-aware and never exercises the
    naive branch at all). Layered four ways, weakest to strongest:

    1. The parsed naive epoch is pinned against ``calendar.timegm`` — which
       interprets a ``struct_time`` as UTC by construction, regardless of the
       host's zone — rather than against another call into the same function
       under test.
    2. On POSIX (skipped on win32, which has no per-process ``TZ``/
       ``time.tzset()``), the process timezone is flipped to a REAL non-UTC
       zone via a NESTED ``pytest.MonkeyPatch.context()`` and the SAME
       assertion is re-run; the nested context's own ``__exit__`` restores
       ``TZ`` in ``os.environ`` before the outer ``finally`` calls
       ``time.tzset()`` again, so the C runtime is re-synced to the ALREADY-
       restored value (#775 review pass 3 finding 4 — calling ``tzset()``
       while ``TZ`` is still the test's overridden value is a no-op that
       leaves the process timezone changed after the test returns).
    3. Host-independent on every platform, but still real-host-dependent in
       principle: the naive value, its ``Z`` form, and an explicit
       ``+07:00`` offset on the same clock digits must relate exactly as UTC
       arithmetic demands. Every leg here is either UTC or explicitly
       offset, so — per pass 3 P3 — this alone cannot catch a naive-branch
       regression on a genuinely UTC-hosted platform.
    4. Fully host-independent (pass 3 P3's fix): install
       ``_FakeHostOffsetDatetime`` (a permanent, deterministic UTC+7 double
       for naive ``.timestamp()`` calls) and re-run the naive-vs-``Z``
       equality. This is the only leg that exercises the naive branch under
       conditions the REAL host cannot coincidentally satisfy either way —
       see test_original_timezone_bug_still_passes, which proves this leg
       (and only this leg, unconditionally) catches the old implementation.

    Ablation: drop the ``if dt.tzinfo is None: dt = dt.replace(tzinfo=
    timezone.utc)`` line in ``_parse_session_start`` and every assertion
    below fails on a non-UTC host; on a UTC host, only leg 4 (and, on POSIX,
    leg 2) still catches it — see test_original_timezone_bug_still_passes,
    which makes that unconditional rather than host-dependent."""
    import bmad_loop.tokens as tokens_module

    naive = "2026-09-10T00:00:00"
    expected_utc_epoch = calendar.timegm(time_module.strptime(naive, "%Y-%m-%dT%H:%M:%S"))
    assert tokens_module._parse_session_start({"timestamp": naive}) == expected_utc_epoch

    if sys.platform != "win32":
        try:
            with pytest.MonkeyPatch.context() as tz_mp:
                tz_mp.setenv("TZ", "America/Los_Angeles")
                time_module.tzset()
                assert (
                    tokens_module._parse_session_start({"timestamp": naive}) == expected_utc_epoch
                )
            # tz_mp.__exit__ (end of the `with`) already restored/deleted TZ
            # in os.environ; tzset() below re-syncs the C runtime to THAT
            # now-restored value, not to the LA value still in effect a
            # moment ago.
        finally:
            time_module.tzset()

    with_z = tokens_module._parse_session_start({"timestamp": naive + "Z"})
    without_z = tokens_module._parse_session_start({"timestamp": naive})
    with_offset = tokens_module._parse_session_start({"timestamp": naive + "+07:00"})
    assert with_z == without_z
    assert with_z - with_offset == 7 * 3600

    # Fully host-independent (#775 review pass 3 finding 5): a fake host
    # permanently at UTC+7, regardless of what the real host actually is.
    monkeypatch.setattr(tokens_module, "datetime", _FakeHostOffsetDatetime)
    with_z_faked = tokens_module._parse_session_start({"timestamp": naive + "Z"})
    without_z_faked = tokens_module._parse_session_start({"timestamp": naive})
    assert with_z_faked == without_z_faked


def test_original_timezone_bug_still_passes(monkeypatch):
    """Review pass 3 finding 5, ablation-as-test: swap ``_parse_session_start``
    for the ORIGINAL host-local-naive implementation (``datetime.fromisoformat
    (...).timestamp()``, no explicit UTC attachment) and confirm the
    regression test above now catches it, UNCONDITIONALLY — on every
    platform, regardless of the real host's timezone. Before the fake-host
    double existed, this exact substitution (the reviewer's own repro) made
    the regression test PASS despite the bug being restored, whenever the
    real host happened to be at UTC+00 (pass 3 P3); with the double in
    place, it fails there too, because the double's naive branch fires
    regardless of what the real host's zone actually is."""
    import bmad_loop.tokens as tokens_module

    def old_buggy(payload: dict) -> float:
        ts = payload["timestamp"]
        v = ts.strip()
        if v.endswith("Z"):
            v = v[:-1] + "+00:00"
        # Looked up dynamically off the module (not a captured import) so
        # this old implementation is ALSO subject to the main test's
        # `_FakeHostOffsetDatetime` installation partway through its body.
        return tokens_module.datetime.fromisoformat(v).timestamp()

    monkeypatch.setattr(tokens_module, "_parse_session_start", old_buggy)
    with pytest.raises(AssertionError):
        test_parse_session_start_treats_naive_timestamp_as_utc(monkeypatch)


def test_cwd_expansion_uses_home_override(tmp_path):
    """Review finding 5: a tilde-form cwd anchor must expand against the same
    ``home`` override the glob itself uses, so it still matches the absolute
    paths real session_meta.payload.cwd values carry.

    Ablation: drop the ``home`` argument from the ``_normalize_cwd(cwd,
    home)`` call for the ``cwd`` anchor (leaving the payload side alone) and
    this test fails — "~/repo" no longer normalizes to the same string as
    the absolute ``tmp_path / "repo"``, so the candidate is filtered out and
    None comes back instead."""
    p = tmp_path / ".codex" / "sessions" / "2026" / "09" / "10" / "rollout-a.jsonl"
    _write_rollout(p, session_id="a", cwd=str(tmp_path / "repo"), mtime=2000, start=_iso(2000))

    found = discover_transcript(
        "codex-rollout", session_id=None, cwd="~/repo", not_before=2000, home=tmp_path
    )
    assert found == p


def test_literal_home_directory(tmp_path):
    """Review pass 2 finding 3: a home directory containing glob
    metacharacters (``[home]`` here) must be treated as a LITERAL path
    component, not glob syntax — even with an exact session_id match, which
    has nothing to do with the cwd/meta matching this finding is otherwise
    about.

    Ablation: drop the ``glob.escape(home_dir)`` call (passing the raw
    ``home_dir`` straight into the pattern) and this test fails alone —
    ``[home]`` is read as a character class by ``glob.glob``, matches
    nothing, and discovery returns None instead of the real file."""
    home = tmp_path / "[home]"
    target = home / ".codex" / "sessions" / "2026" / "09" / "10" / "rollout-ours.jsonl"
    _write_rollout(target, session_id="ours", cwd="/repo", mtime=2100, start=_iso(2005))

    found = discover_transcript(
        "codex-rollout", session_id="ours", cwd="/repo", not_before=2000, home=home
    )
    assert found == target


def test_strip_extended_windows_prefix():
    """Review pass 3 finding 1: the normaliser helper tested DIRECTLY, so
    this runs on every platform regardless of host OS — pure string
    manipulation, no dependency on actual Windows path semantics."""
    from bmad_loop.tokens import _strip_extended_windows_prefix

    assert _strip_extended_windows_prefix(r"\\?\C:\repo") == r"C:\repo"
    assert _strip_extended_windows_prefix(r"\\?\UNC\server\share") == r"\\server\share"
    assert _strip_extended_windows_prefix(r"C:\repo") == r"C:\repo"  # unaffected
    assert _strip_extended_windows_prefix("/repo") == "/repo"  # unaffected (POSIX)


@pytest.mark.skipif(sys.platform != "win32", reason="Windows path semantics only")
def test_extended_windows_cwd(tmp_path):
    """Review pass 3 finding 1, end-to-end: a rollout recording the
    EXTENDED-prefix spelling of a Windows cwd (some Codex builds record this
    form in session_meta.payload.cwd) must still match the ORDINARY spelling
    `spec.cwd` actually carries.

    Ablation: drop the `_strip_extended_windows_prefix` call from
    `_normalize_cwd` and this test fails alone on Windows — `\\\\?\\C:\\repo`
    and `C:/repo` normalize to different strings and the real candidate is
    filtered out."""
    p = tmp_path / ".codex" / "sessions" / "2026" / "09" / "10" / "rollout-ours.jsonl"
    _write_rollout(p, session_id="ours", cwd=r"\\?\C:\repo", mtime=2100, start=_iso(2005))

    found = discover_transcript(
        "codex-rollout", session_id=None, cwd="C:/repo", not_before=2000, home=tmp_path
    )
    assert found == p


def test_read_first_json_line_tolerates_recursion_error(tmp_path):
    """Review pass 3 finding 2, isolated at the lowest layer that can catch
    it: a deeply nested (or truncated mid-write) JSON header must not raise
    RecursionError out of `_read_first_json_line` — skip it like any other
    malformed candidate. A real file reproduces the recursive-descent
    decoder's RecursionError just as reliably as the reviewer's monkeypatched
    Path.open, without needing the monkeypatch.

    Ablation: drop `RecursionError` from the except clause around
    `json.loads` and this test fails alone — RecursionError escapes."""
    from bmad_loop.tokens import _read_first_json_line

    p = tmp_path / "rollout-bad.jsonl"
    p.write_text('{"type":"session_meta","payload":' + "[" * 10000)
    assert _read_first_json_line(p) is None


def test_partial_header_is_tolerated(tmp_path):
    """Review pass 3 finding 2, end-to-end: the same pathological candidate
    sitting ALONGSIDE a real, valid rollout — discovery must still find the
    valid one rather than let the malformed neighbor's RecursionError abort
    the whole scan."""
    base = tmp_path / ".codex" / "sessions" / "2026" / "09" / "10"
    good = base / "rollout-good.jsonl"
    _write_rollout(good, session_id="good", cwd="/repo", mtime=2100, start=_iso(2005))
    bad = base / "rollout-bad.jsonl"
    bad.write_text('{"type":"session_meta","payload":' + "[" * 10000)

    found = discover_transcript(
        "codex-rollout", session_id=None, cwd="/repo", not_before=2000, home=tmp_path
    )
    assert found == good


def test_upper_plausibility_bound_refuses_sole_distant_candidate(tmp_path):
    """Review pass 3 finding 3: the OLDER two-candidate test
    (test_discover_transcript_picks_earliest_after_launch_not_newest_mtime)
    does NOT actually pin the window's upper bound — with the bound removed,
    the "smallest delta among on-or-after candidates" tier still
    independently prefers the closer of the two competing candidates, so
    that test passes with or without `window_hi`. This test isolates the
    upper bound itself with a SOLE, unopposed candidate: 20 minutes after
    launch with nothing else competing must still be refused (None) —
    accepting it just because nothing else is in the running would be
    exactly the implausible-attribution bug the window exists to prevent.

    Ablation: set `window_hi = float("inf")` in `discover_transcript` and
    this test fails alone (the two-candidate test above does NOT — confirmed
    pass 3 finding 3) — the sole distant candidate would be returned instead
    of None."""
    base = tmp_path / ".codex" / "sessions" / "2026" / "09" / "10"
    _write_rollout(
        base / "rollout-toolate.jsonl",
        session_id="toolate",
        cwd="/repo",
        mtime=2100,
        start=_iso(2000 + 1200),  # 20 minutes after launch, unopposed
    )
    found = discover_transcript(
        "codex-rollout", session_id=None, cwd="/repo", not_before=2000, home=tmp_path
    )
    assert found is None
