import pytest

from bmad_loop.adapters import profile as profile_mod
from bmad_loop.adapters.profile import (
    CLIProfile,
    HookSpec,
    ProfileError,
    get_profile,
    load_profiles,
)

MINIMAL_PROFILE = """
name = "mycli"
binary = "mycli"
bypass_args = ["--yes"]

[hooks]
dialect = "claude-settings-json"
config_path = ".mycli/settings.json"
events = { SessionStart = "SessionStart", Stop = "Stop" }
"""


HOOKLESS_PROFILE = """
name = "mycli-http"
binary = "mycli"

[hooks]
dialect = "none"
"""


def test_builtin_profiles_load():
    profiles = load_profiles()
    assert {"claude", "codex", "gemini", "opencode-http"} <= set(profiles)
    assert profiles["claude"].usage_parser == "claude-jsonl"
    assert profiles["codex"].hooks.dialect == "codex-hooks-json"
    assert "SessionEnd" not in profiles["codex"].hooks.events  # codex has no such hook
    assert profiles["gemini"].hooks.events["AfterAgent"] == "Stop"
    assert profiles["gemini"].launch_args == ("-i",)
    # claude reads .claude/skills; codex and gemini read .agents/skills
    assert profiles["claude"].skill_tree == ".claude/skills"
    assert profiles["codex"].skill_tree == ".agents/skills"
    assert profiles["gemini"].skill_tree == ".agents/skills"
    # each profile carries the gitignored configs a worktree checkout omits
    assert ".mcp.json" in profiles["claude"].seed_files
    assert ".claude/settings.json" in profiles["claude"].seed_files
    assert profiles["codex"].seed_files == (".codex/config.toml",)
    assert profiles["gemini"].seed_files == (".gemini/settings.json",)
    # copilot: turn-end is agentStop (Copilot 1.0.63 never fires PascalCase Stop),
    # no PreCompact equivalent, and its events.jsonl parser is wired up
    assert profiles["copilot"].hooks.events == {
        "agentStop": "Stop",
        "sessionStart": "SessionStart",
        "sessionEnd": "SessionEnd",
    }
    assert profiles["copilot"].usage_parser == "copilot-events"
    # copilot writes token totals only on shutdown (poll grace) and fires
    # agentStop per turn (multi-turn reviews need more nudges)
    assert profiles["copilot"].usage_grace_s == 8.0
    assert profiles["copilot"].stop_without_result_nudges == 5
    # copilot also fires agentStop for subagent turns (empty transcriptPath) — those
    # are ignored so the main session's turn-end drives completion
    assert profiles["copilot"].subagent_stop_without_transcript is True
    # other built-ins keep the defaults: read usage once, inherit the global nudge
    # limit, and treat every Stop as the main turn-end (no subagent filtering)
    for name in ("claude", "codex", "gemini"):
        assert profiles[name].usage_grace_s == 0.0
        assert profiles[name].stop_without_result_nudges is None
        assert profiles[name].subagent_stop_without_transcript is False
    # claude forces its classic (inline/scrollback) renderer so a pane capture is
    # not collapsed to the final frame by the fullscreen alt-screen TUI, and
    # disables background tasks so a dev session cannot background its
    # implementation sub-agent and strand it at turn end (#109); other profiles
    # add no such env overrides
    assert profiles["claude"].env.get("CLAUDE_CODE_DISABLE_ALTERNATE_SCREEN") == "1"
    assert profiles["claude"].env.get("CLAUDE_CODE_DISABLE_BACKGROUND_TASKS") == "1"
    for name in sorted(set(profiles) - {"claude"}):
        assert "CLAUDE_CODE_DISABLE_ALTERNATE_SCREEN" not in profiles[name].env
        assert "CLAUDE_CODE_DISABLE_BACKGROUND_TASKS" not in profiles[name].env
    # opencode-http is hookless (HTTP/SSE transport): no hook dialect surfaces,
    # skills read from the claude tree, usage comes over HTTP (no transcript parser)
    opencode = profiles["opencode-http"]
    assert opencode.hookless is True
    assert opencode.hooks.dialect == "none"
    assert opencode.hooks.config_path == ""
    assert opencode.hooks.events == {}
    assert opencode.skill_tree == ".claude/skills"
    assert opencode.usage_parser == "none"
    assert opencode.binary == "opencode"
    # every hook-driven built-in stays non-hookless
    for name in sorted(set(profiles) - {"opencode-http"}):
        assert profiles[name].hookless is False


def test_usage_grace_and_nudges_default_when_unset(tmp_path):
    # MINIMAL_PROFILE omits both -> 0.0 / None
    profiles_dir = tmp_path / ".bmad-loop" / "profiles"
    profiles_dir.mkdir(parents=True)
    (profiles_dir / "mycli.toml").write_text(MINIMAL_PROFILE)
    prof = load_profiles(tmp_path)["mycli"]
    assert prof.usage_grace_s == 0.0
    assert prof.stop_without_result_nudges is None
    assert prof.subagent_stop_without_transcript is False


def test_seed_files_default_empty_when_unset(tmp_path):
    # MINIMAL_PROFILE omits seed_files -> defaults to ()
    profiles_dir = tmp_path / ".bmad-loop" / "profiles"
    profiles_dir.mkdir(parents=True)
    (profiles_dir / "mycli.toml").write_text(MINIMAL_PROFILE)
    assert load_profiles(tmp_path)["mycli"].seed_files == ()


def test_skill_tree_defaults_when_unset():
    # MINIMAL_PROFILE omits skill_tree -> defaults to .claude/skills
    assert get_profile("claude").skill_tree == ".claude/skills"


def test_legacy_alias_resolves():
    assert get_profile("claude-code-tmux").name == "claude"


def test_opencode_alias_resolves():
    assert get_profile("opencode").name == "opencode-http"


def test_hookless_user_profile_parses(tmp_path):
    profiles_dir = tmp_path / ".bmad-loop" / "profiles"
    profiles_dir.mkdir(parents=True)
    (profiles_dir / "mycli-http.toml").write_text(HOOKLESS_PROFILE)
    prof = load_profiles(tmp_path)["mycli-http"]
    assert prof.hookless is True
    assert prof.hooks.dialect == "none"
    assert prof.hooks.config_path == ""
    assert prof.hooks.events == {}


def test_unknown_profile_raises():
    with pytest.raises(ProfileError, match="unknown CLI profile"):
        get_profile("acme-cli")


def test_adapter_field_defaults_to_generic_and_parses():
    """The adapter kind is read (not validated) at parse time: unset defaults to
    the bundled tmux generic; opencode-http declares its HTTP adapter kind."""
    assert get_profile("claude").adapter == "generic"
    assert get_profile("opencode-http").adapter == "opencode-http"


def test_adapter_field_not_validated_at_parse_time(tmp_path):
    """A profile naming an unregistered adapter kind still PARSES — validity is
    enforced later against the live registry (at construction / by `validate`),
    never a hardcoded set here."""
    profiles_dir = tmp_path / ".bmad-loop" / "profiles"
    profiles_dir.mkdir(parents=True)
    (profiles_dir / "future.toml").write_text(
        MINIMAL_PROFILE.replace("[hooks]", 'adapter = "not-a-real-kind-yet"\n[hooks]')
    )
    assert load_profiles(tmp_path)["mycli"].adapter == "not-a-real-kind-yet"


# --------------------------------------------------------------------------- #
# bmad_loop.profiles entry-point discovery — the companion to the adapter
# registry so an out-of-tree package ships both its class and the selecting
# profile with zero project config.


@pytest.fixture
def profile_scan(monkeypatch):
    """Isolate + re-arm the profile entry-point scan: snapshot/clear the module's
    external-scan state, then hand back a hook to install fake entry points."""
    saved_loaded = profile_mod._EXTERNALS_LOADED
    saved_profiles = dict(profile_mod._EXTERNAL_PROFILES)
    saved_errors = dict(profile_mod._PROFILE_LOAD_ERRORS)

    def arm(*eps, scan_error=None):
        def fake_entry_points(*, group):
            assert group == profile_mod.PROFILES_GROUP
            if scan_error is not None:
                raise scan_error
            return list(eps)

        monkeypatch.setattr(profile_mod.importlib.metadata, "entry_points", fake_entry_points)
        profile_mod._EXTERNALS_LOADED = False
        profile_mod._EXTERNAL_PROFILES.clear()
        profile_mod._PROFILE_LOAD_ERRORS.clear()

    yield arm

    profile_mod._EXTERNALS_LOADED = saved_loaded
    profile_mod._EXTERNAL_PROFILES.clear()
    profile_mod._EXTERNAL_PROFILES.update(saved_profiles)
    profile_mod._PROFILE_LOAD_ERRORS.clear()
    profile_mod._PROFILE_LOAD_ERRORS.update(saved_errors)


class _FakeEntryPoint:
    def __init__(self, name, load):
        self.name = name
        self._load = load

    def load(self):
        return self._load()


def _plugin_profile(name="acme", adapter="acme"):
    return CLIProfile(name=name, binary=name, adapter=adapter, hooks=HookSpec("none", "", {}))


def test_entry_point_profile_is_discovered(profile_scan):
    """A pip-installed profile provider (a callable returning CLIProfiles) makes
    its profile resolvable with no project TOML — the zero-config selection path."""
    profile_scan(_FakeEntryPoint("acme", lambda: (lambda: [_plugin_profile()])))
    prof = get_profile("acme")
    assert prof.name == "acme" and prof.adapter == "acme"
    assert profile_mod.external_profile_errors() == {}


def test_entry_point_profile_provider_may_be_iterable(profile_scan):
    """The provider may be an iterable directly, not only a callable returning
    one — both shapes are accepted."""
    profile_scan(_FakeEntryPoint("acme", lambda: [_plugin_profile()]))
    assert "acme" in load_profiles()


def test_project_profile_overrides_entry_point(profile_scan, tmp_path):
    """Precedence packaged < entry-point < project: a project-local TOML of the
    same name wins over an entry-point profile."""
    profile_scan(_FakeEntryPoint("acme", lambda: [_plugin_profile(adapter="acme")]))
    profiles_dir = tmp_path / ".bmad-loop" / "profiles"
    profiles_dir.mkdir(parents=True)
    (profiles_dir / "acme.toml").write_text(
        MINIMAL_PROFILE.replace('name = "mycli"', 'name = "acme"')
    )
    prof = load_profiles(tmp_path)["acme"]
    assert prof.binary == "mycli"  # the project TOML, not the entry-point profile


def test_entry_point_profile_can_override_packaged(profile_scan):
    """Entry-point profiles overlay the packaged built-ins (packaged <
    entry-point), so a plugin may re-point a bundled name."""
    profile_scan(_FakeEntryPoint("acme", lambda: [_plugin_profile(name="claude", adapter="acme")]))
    assert load_profiles()["claude"].adapter == "acme"


def test_broken_profile_provider_degrades_and_is_recorded(profile_scan):
    """A provider that blows up must not break profile loading: the built-ins
    still load, and the failure is recorded for diagnostics."""

    def boom():
        raise RuntimeError("half-installed plugin")

    profile_scan(_FakeEntryPoint("broken", boom))
    profiles = load_profiles()
    assert "claude" in profiles  # built-ins unaffected
    assert list(profile_mod.external_profile_errors()) == ["broken"]
    assert "half-installed" in profile_mod.external_profile_errors()["broken"]


def test_one_broken_profile_package_does_not_hide_the_rest(profile_scan):
    """Per-entry isolation: a good provider still registers alongside a broken one."""

    def boom():
        raise RuntimeError("broke")

    profile_scan(
        _FakeEntryPoint("broken", boom),
        _FakeEntryPoint("acme", lambda: [_plugin_profile()]),
    )
    profiles = load_profiles()
    assert "acme" in profiles
    assert list(profile_mod.external_profile_errors()) == ["broken"]


def test_profile_provider_returning_junk_is_rejected(profile_scan):
    """A provider that yields a non-CLIProfile is the package's bug — recorded,
    never trusted into the profile map."""
    profile_scan(_FakeEntryPoint("acme", lambda: [object()]))
    profiles = load_profiles()
    assert "acme" not in profiles
    assert "not CLIProfile" in profile_mod.external_profile_errors()["acme"]


def test_profile_scan_failure_degrades(profile_scan):
    """The enumeration itself blowing up leaves built-in loading working, with the
    scan failure recorded."""
    profile_scan(scan_error=RuntimeError("metadata index corrupt"))
    assert "claude" in load_profiles()
    assert "<entry-point scan>" in profile_mod.external_profile_errors()


def test_render_prompt_passthrough_and_template():
    claude = get_profile("claude")
    assert claude.render_prompt("/bmad-dev-auto 1-1-a") == "/bmad-dev-auto 1-1-a"
    codex = get_profile("codex")
    assert codex.render_prompt("/bmad-dev-auto 1-1-a") == (
        "Use the $bmad-dev-auto skill now, and use subagents as needed: 1-1-a"
    )
    opencode = get_profile("opencode-http")
    assert opencode.render_prompt("/bmad-dev-auto 1-1-a") == (
        "Use the bmad-dev-auto skill now: 1-1-a"
    )
    # non-slash prompts pass through {prompt}; {skill}/{args} degrade gracefully
    assert claude.render_prompt("just do it") == "just do it"


def test_user_profile_overlay(tmp_path):
    profiles_dir = tmp_path / ".bmad-loop" / "profiles"
    profiles_dir.mkdir(parents=True)
    (profiles_dir / "mycli.toml").write_text(MINIMAL_PROFILE)
    # override a built-in by reusing its name
    (profiles_dir / "claude-override.toml").write_text(
        MINIMAL_PROFILE.replace('name = "mycli"', 'name = "claude"')
    )
    profiles = load_profiles(tmp_path)
    assert "mycli" in profiles
    assert profiles["mycli"].bypass_args == ("--yes",)
    assert profiles["claude"].binary == "mycli"  # overridden
    assert "codex" in profiles  # built-ins still present


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        ('name = "mycli"\nbinary = "mycli"', "missing"),  # no [hooks]
        (
            MINIMAL_PROFILE.replace('dialect = "claude-settings-json"', 'dialect = "nope"'),
            "dialect",
        ),
        (MINIMAL_PROFILE.replace('Stop = "Stop"', 'Stop = "TurnDone"'), "canonical"),
        (
            MINIMAL_PROFILE.replace("[hooks]", 'usage_parser = "magic"\n[hooks]'),
            "usage_parser",
        ),
        (
            MINIMAL_PROFILE.replace(
                'config_path = ".mycli/settings.json"',
                'config_path = "/abs/settings.json"',
            ),
            "relative",
        ),
        (
            MINIMAL_PROFILE.replace(
                "[hooks]",
                'skill_tree = "/abs/skills"\n[hooks]',
            ),
            "skill_tree",
        ),
        (
            MINIMAL_PROFILE.replace(
                "[hooks]",
                'seed_files = ["/etc/passwd"]\n[hooks]',
            ),
            "seed_files",
        ),
        (
            MINIMAL_PROFILE.replace("[hooks]", "usage_grace_s = -1\n[hooks]"),
            "usage_grace_s",
        ),
        (
            MINIMAL_PROFILE.replace("[hooks]", "stop_without_result_nudges = -2\n[hooks]"),
            "stop_without_result_nudges",
        ),
        # real dialects still hard-require config_path and events
        (
            MINIMAL_PROFILE.replace('config_path = ".mycli/settings.json"', ""),
            "config_path",
        ),
        (
            MINIMAL_PROFILE.replace(
                'events = { SessionStart = "SessionStart", Stop = "Stop" }', ""
            ),
            "events",
        ),
        # hookless must not carry hook plumbing (config_path/events)
        (
            MINIMAL_PROFILE.replace('dialect = "claude-settings-json"', 'dialect = "none"'),
            "hookless",
        ),
    ],
)
def test_invalid_profiles_rejected(tmp_path, mutation, match):
    profiles_dir = tmp_path / ".bmad-loop" / "profiles"
    profiles_dir.mkdir(parents=True)
    (profiles_dir / "bad.toml").write_text(mutation)
    with pytest.raises(ProfileError, match=match):
        load_profiles(tmp_path)
