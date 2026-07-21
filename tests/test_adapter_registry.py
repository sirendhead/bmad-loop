"""Coding-CLI adapter-registry selection + discovery proof.

The CLI axis selects its adapter *class* through a registry
(:func:`~bmad_loop.adapters.registry.register_adapter`) keyed on the
``profile.adapter`` field, rather than name-branching in ``cli._make_adapters``.
So a new adapter family is a registration — not a core edit — exactly as a new
transport backend is (:mod:`bmad_loop.adapters.multiplexer`). These tests pin the
registry (builtin + external registration, builtins-first-wins, unknown-kind
fail-loud), the out-of-tree entry-point discovery and its degrade-not-crash
contract, and the ``cli._make_adapters`` dispatch: a registered kind builds, the
``(cfg, synthesizes)`` cache shares instances, ``needs_mux`` gates the transport
resolution, a construction failure becomes a clean ``SystemExit``, and — the
regression pin — ``opencode-http`` still dispatches to the HTTP adapters unchanged.

The registry is deliberately simpler than the multiplexer seam: adapters are built
per-run (``cli._make_adapters`` owns the cache), so there is no ``lru_cache`` /
``cache_clear`` and no ``configure_*`` / platform-default machinery — the
``fresh_adapter_registry`` fixture snapshots only ``_ADAPTERS`` / the two loaded
flags / ``_EXTERNAL_ERRORS``.

Entry points are faked by monkeypatching ``importlib.metadata.entry_points``
through the ``registry`` module's own binding; one test builds a real
``*.dist-info`` on ``sys.path`` to prove the scan works against genuine packaging
metadata.
"""

from __future__ import annotations

import pytest
from conftest import install_bmad_config

from bmad_loop import cli
from bmad_loop import policy as policy_mod
from bmad_loop.adapters import multiplexer as mux_mod
from bmad_loop.adapters import profile as profile_mod
from bmad_loop.adapters import registry as m
from bmad_loop.adapters.registry import AdapterBuilder, AdapterError

# --------------------------------------------------------------------------- #
# Stubs + fixtures


class _StubAdapter:
    """Plain-variant double: records the kwargs cli._make_adapters passes so a
    test can assert the mux was (or was not) threaded in."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs


class _StubDevAdapter(_StubAdapter):
    """Dev-variant double: mirrors the real ``(*args, paths, **kwargs)`` dev
    __init__ contract every registered family's dev class honors."""

    def __init__(self, *, paths, **kwargs):
        super().__init__(**kwargs)
        self.paths = paths


def _stub_builder(*, needs_mux_construct_error=()):
    return AdapterBuilder(
        plain=_StubAdapter, dev=_StubDevAdapter, construct_error=needs_mux_construct_error
    )


class _FakeEntryPoint:
    """Duck-typed importlib.metadata.EntryPoint: the loader only touches
    ``.name`` and ``.load()``."""

    def __init__(self, name, load):
        self.name = name
        self._load = load

    def load(self):
        return self._load()


@pytest.fixture
def fresh_adapter_registry(monkeypatch):
    """Isolate the global adapter registry: snapshot, clear, restore. No
    lru_cache / configured-choice to reset — the deliberate asymmetry vs. the mux
    seam. The externals scan is parked as already-loaded so whatever adapters are
    installed on the dev box can't leak into builtin tests (discovery tests re-arm
    it via :func:`scan_adapter_registry`). The companion profile-module external
    scan is suppressed too, so an installed ``bmad_loop.profiles`` package can't
    leak a profile into the ``_make_adapters`` integration tests."""
    saved_adapters = dict(m._ADAPTERS)
    saved_loaded = m._BUILTINS_LOADED
    saved_ext_loaded = m._EXTERNALS_LOADED
    saved_ext_errors = dict(m._EXTERNAL_ERRORS)
    m._ADAPTERS.clear()
    m._BUILTINS_LOADED = False
    m._EXTERNALS_LOADED = True  # externals OFF by default; discovery tests opt back in
    m._EXTERNAL_ERRORS.clear()

    saved_prof_loaded = profile_mod._EXTERNALS_LOADED
    saved_prof_profiles = dict(profile_mod._EXTERNAL_PROFILES)
    saved_prof_errors = dict(profile_mod._PROFILE_LOAD_ERRORS)
    profile_mod._EXTERNALS_LOADED = True
    profile_mod._EXTERNAL_PROFILES.clear()
    profile_mod._PROFILE_LOAD_ERRORS.clear()

    yield m

    m._ADAPTERS.clear()
    m._ADAPTERS.update(saved_adapters)
    m._BUILTINS_LOADED = saved_loaded
    m._EXTERNALS_LOADED = saved_ext_loaded
    m._EXTERNAL_ERRORS.clear()
    m._EXTERNAL_ERRORS.update(saved_ext_errors)
    profile_mod._EXTERNALS_LOADED = saved_prof_loaded
    profile_mod._EXTERNAL_PROFILES.clear()
    profile_mod._EXTERNAL_PROFILES.update(saved_prof_profiles)
    profile_mod._PROFILE_LOAD_ERRORS.clear()
    profile_mod._PROFILE_LOAD_ERRORS.update(saved_prof_errors)


@pytest.fixture
def scan_adapter_registry(fresh_adapter_registry, monkeypatch):
    """fresh_adapter_registry with the externals scan re-armed. Yields a hook:
    call it with fake entry points (or a ``scan_error`` to raise from the scan
    itself) and the next resolution performs that scan."""

    def arm(*eps, scan_error=None):
        def fake_entry_points(*, group):
            assert group == m.ADAPTERS_GROUP
            if scan_error is not None:
                raise scan_error
            return list(eps)

        monkeypatch.setattr(m.importlib.metadata, "entry_points", fake_entry_points)
        m._EXTERNALS_LOADED = False
        m._EXTERNAL_ERRORS.clear()

    yield fresh_adapter_registry, arm


def _write_policy(project, text) -> None:
    d = project / ".bmad-loop"
    d.mkdir(parents=True, exist_ok=True)
    (d / "policy.toml").write_text(text, encoding="utf-8")


def _write_profile(project, name, *, adapter, hookless=True) -> None:
    d = project / ".bmad-loop" / "profiles"
    d.mkdir(parents=True, exist_ok=True)
    if hookless:
        hooks = '[hooks]\ndialect = "none"\n'
    else:
        hooks = '[hooks]\ndialect = "claude-settings-json"\nconfig_path = ".hermes/settings.json"\n[hooks.events]\nStop = "Stop"\n'
    (d / f"{name}.toml").write_text(
        f'name = "{name}"\nbinary = "{name}"\nadapter = "{adapter}"\n{hooks}', encoding="utf-8"
    )


def _run_dir(project):
    return project / ".bmad-loop" / "runs" / "r"


# --------------------------------------------------------------------------- #
# Registry — builtin registration + fail-loud


def test_builtin_kinds_registered_with_needs_mux(fresh_adapter_registry):
    """The two bundled kinds register with the correct transport requirement:
    generic drives tmux (needs_mux), opencode-http is hookless HTTP/SSE (does not)."""
    generic = fresh_adapter_registry.get_adapter_kind("generic")
    http = fresh_adapter_registry.get_adapter_kind("opencode-http")
    assert generic.needs_mux is True
    assert http.needs_mux is False
    assert fresh_adapter_registry.known_adapter_kinds() == ["generic", "opencode-http"]


def test_load_thunk_returns_real_builder(fresh_adapter_registry):
    """The lazy thunk resolves to the real adapter classes — imported only now,
    never at registration/listing time."""
    from bmad_loop.adapters.generic import GenericAdapter, GenericDevAdapter

    builder = fresh_adapter_registry.get_adapter_kind("generic").load()
    assert builder.plain is GenericAdapter
    assert builder.dev is GenericDevAdapter
    assert builder.construct_error == ()


def test_unknown_kind_fails_loud_naming_known(fresh_adapter_registry):
    """An explicit but unregistered kind is a misconfiguration: fail loud, listing
    the registered kinds — never silently fall back."""
    with pytest.raises(AdapterError, match=r"nonesuch.*generic.*opencode-http"):
        fresh_adapter_registry.get_adapter_kind("nonesuch")


def test_register_adapter_first_wins(fresh_adapter_registry):
    """A duplicate registration of a name is ignored (first wins) — the mechanism
    that lets builtins load first and shrug off a same-named external."""
    registry = fresh_adapter_registry
    registry.register_adapter("dup", needs_mux=True, load=lambda: _stub_builder())
    registry.register_adapter("dup", needs_mux=False, load=lambda: _stub_builder())
    assert registry.get_adapter_kind("dup").needs_mux is True


def test_detect_adapters_labels_builtin(fresh_adapter_registry):
    """detect_adapters lists every kind, sorted, labelled builtin-vs-external,
    without invoking any load thunk (no heavy import)."""
    registry = fresh_adapter_registry
    registry.register_adapter("extra", needs_mux=False, load=lambda: _stub_builder())
    rows = {r.name: r for r in registry.detect_adapters()}
    assert set(rows) == {"generic", "opencode-http", "extra"}
    assert rows["generic"].builtin is True and rows["generic"].needs_mux is True
    assert rows["opencode-http"].builtin is True and rows["opencode-http"].needs_mux is False
    assert rows["extra"].builtin is False
    assert [r.name for r in registry.detect_adapters()] == sorted(rows)


# --------------------------------------------------------------------------- #
# External discovery (the bmad_loop.adapters entry-point scan)


def test_entry_point_adapter_registers_and_is_selectable(scan_adapter_registry):
    """The pip-install-and-go path: the entry point's module import registers the
    kind; it resolves and lists like a builtin, with no load error."""
    registry, arm = scan_adapter_registry

    def load():
        registry.register_adapter("extadapter", needs_mux=False, load=lambda: _stub_builder())

    arm(_FakeEntryPoint("extadapter", load))
    kind = registry.get_adapter_kind("extadapter")
    assert kind.needs_mux is False
    assert "extadapter" in {r.name for r in registry.detect_adapters()}
    assert registry.external_adapter_errors() == {}


def test_builtins_first_wins_over_external(scan_adapter_registry):
    """Ordering guarantee: builtins register before the scan, so an external that
    tries to register the bundled name ``generic`` cannot shadow it."""
    registry, arm = scan_adapter_registry

    def load():
        # a hostile/clumsy external claiming the builtin name with wrong needs_mux
        registry.register_adapter("generic", needs_mux=False, load=lambda: _stub_builder())

    arm(_FakeEntryPoint("shadow", load))
    kind = registry.get_adapter_kind("generic")
    assert kind.needs_mux is True  # the builtin, not the external
    assert kind.load().plain.__name__ == "GenericAdapter"


def test_broken_entry_point_degrades_and_is_recorded(scan_adapter_registry):
    """A distribution whose import blows up must not break selection: the builtins
    still resolve, and the failure is recorded for adapters/validate to show."""
    registry, arm = scan_adapter_registry

    def boom():
        raise ImportError("No module named 'ghost_dependency'")

    arm(_FakeEntryPoint("brokenadapter", boom))
    assert registry.get_adapter_kind("generic").needs_mux is True  # selection still works
    errors = registry.external_adapter_errors()
    assert list(errors) == ["brokenadapter"]
    assert "ghost_dependency" in errors["brokenadapter"]


def test_one_broken_package_does_not_hide_the_rest(scan_adapter_registry):
    """Per-entry isolation: the loader keeps importing after a failure, so a
    working adapter still registers alongside a broken one."""
    registry, arm = scan_adapter_registry

    def boom():
        raise RuntimeError("half-installed")

    def load():
        registry.register_adapter("goodadapter", needs_mux=True, load=lambda: _stub_builder())

    arm(_FakeEntryPoint("brokenadapter", boom), _FakeEntryPoint("goodadapter", load))
    assert registry.get_adapter_kind("goodadapter").needs_mux is True
    assert list(registry.external_adapter_errors()) == ["brokenadapter"]


def test_scan_failure_degrades(scan_adapter_registry):
    """Even the entry-point enumeration itself blowing up leaves selection working,
    with the scan failure recorded."""
    registry, arm = scan_adapter_registry
    arm(scan_error=RuntimeError("metadata index corrupt"))
    assert registry.get_adapter_kind("generic").needs_mux is True
    assert "<entry-point scan>" in registry.external_adapter_errors()


def test_scan_runs_once_per_process(scan_adapter_registry):
    """The loaded-flag is set up front: a second resolution does not re-scan (a
    third-party import failure is not transient; re-importing would re-fail)."""
    registry, arm = scan_adapter_registry
    calls = []

    def load():
        calls.append(1)

    arm(_FakeEntryPoint("extadapter", load))
    registry.known_adapter_kinds()
    registry.known_adapter_kinds()
    assert len(calls) == 1


def test_real_dist_info_metadata_is_discovered(fresh_adapter_registry, monkeypatch, tmp_path):
    """End-to-end against genuine packaging metadata: a real ``*.dist-info`` +
    module on sys.path is found by the unpatched importlib scan and its import
    registers the kind — proving the group name works outside our fakes."""
    site = tmp_path / "site"
    site.mkdir()
    (site / "extadapter_pkg.py").write_text(
        "from bmad_loop.adapters.registry import register_adapter\n"
        "def _load():\n"
        "    raise AssertionError('load thunk must stay lazy')\n"
        "register_adapter('extadapter-real', False, _load)\n",
        encoding="utf-8",
    )
    dist = site / "extadapter-0.1.dist-info"
    dist.mkdir()
    (dist / "METADATA").write_text("Metadata-Version: 2.1\nName: extadapter\nVersion: 0.1\n")
    (dist / "entry_points.txt").write_text(
        "[bmad_loop.adapters]\nextadapter = extadapter_pkg\n", encoding="utf-8"
    )
    monkeypatch.syspath_prepend(str(site))
    fresh_adapter_registry._EXTERNALS_LOADED = False  # re-arm the (real) scan
    assert "extadapter-real" in fresh_adapter_registry.known_adapter_kinds()
    assert fresh_adapter_registry.external_adapter_errors() == {}


# --------------------------------------------------------------------------- #
# cli._make_adapters dispatch through the registry


def test_make_adapters_dispatches_registered_kind(fresh_adapter_registry, project, monkeypatch):
    """A registered out-of-tree kind (the migrated Hermes case) dispatches with no
    core branching: the synthesizing dev/review roles build the dev variant with
    ``paths`` + the shared mux, triage builds the plain variant, and the
    ``(cfg, synthesizes)`` cache shares the dev instance across dev+review."""
    registry = fresh_adapter_registry
    registry.register_adapter("hermes", needs_mux=True, load=lambda: _stub_builder())
    monkeypatch.setattr(mux_mod, "_usable", lambda mux: True)
    install_bmad_config(project)
    _write_profile(project.project, "hermes", adapter="hermes", hookless=False)
    _write_policy(project.project, '[adapter]\nname = "hermes"\n')
    pol = policy_mod.load(project.project / ".bmad-loop" / "policy.toml")

    adapters = cli._make_adapters(project.project, _run_dir(project.project), pol)

    assert isinstance(adapters["dev"], _StubDevAdapter)
    assert adapters["dev"] is adapters["review"]  # (cfg, synthesizes) sharing
    assert adapters["dev"].paths.project == project.project
    assert "mux" in adapters["dev"].kwargs  # needs_mux=True threaded the transport
    assert isinstance(adapters["triage"], _StubAdapter)
    assert not isinstance(adapters["triage"], _StubDevAdapter)
    assert adapters["triage"] is not adapters["dev"]
    assert "mux" in adapters["triage"].kwargs


def test_make_adapters_skips_mux_for_needs_mux_false_kind(
    fresh_adapter_registry, project, monkeypatch
):
    """A ``needs_mux=False`` kind must never resolve the multiplexer — the same
    guarantee the hookless HTTP adapter relies on."""
    registry = fresh_adapter_registry
    registry.register_adapter("noxport", needs_mux=False, load=lambda: _stub_builder())

    def no_mux():
        raise AssertionError("a needs_mux=False kind must not resolve a multiplexer")

    monkeypatch.setattr(mux_mod, "get_multiplexer", no_mux)
    install_bmad_config(project)
    _write_profile(project.project, "noxport", adapter="noxport")
    _write_policy(project.project, '[adapter]\nname = "noxport"\n')
    pol = policy_mod.load(project.project / ".bmad-loop" / "policy.toml")

    adapters = cli._make_adapters(project.project, _run_dir(project.project), pol)
    assert isinstance(adapters["dev"], _StubDevAdapter)
    assert "mux" not in adapters["dev"].kwargs


def test_make_adapters_construct_error_becomes_systemexit(
    fresh_adapter_registry, project, monkeypatch
):
    """A family-declared construction failure raised in __init__ is converted to a
    clean SystemExit — a run aborts with a message, not a traceback."""

    class _Boom(Exception):
        pass

    class _Exploding:
        def __init__(self, **kwargs):
            raise _Boom("server would not start")

    registry = fresh_adapter_registry
    registry.register_adapter(
        "boomer",
        needs_mux=False,
        load=lambda: AdapterBuilder(plain=_Exploding, dev=_Exploding, construct_error=(_Boom,)),
    )
    install_bmad_config(project)
    _write_profile(project.project, "boomer", adapter="boomer")
    _write_policy(project.project, '[adapter]\nname = "boomer"\n')
    pol = policy_mod.load(project.project / ".bmad-loop" / "policy.toml")

    with pytest.raises(SystemExit, match="server would not start"):
        cli._make_adapters(project.project, _run_dir(project.project), pol)


def test_make_adapters_unknown_kind_systemexit_names_profile(fresh_adapter_registry, project):
    """A profile whose ``adapter`` names no registered kind aborts the run with a
    SystemExit that names both the profile and the known kinds."""
    install_bmad_config(project)
    _write_profile(project.project, "weird", adapter="ghostkind")
    _write_policy(project.project, '[adapter]\nname = "weird"\n')
    pol = policy_mod.load(project.project / ".bmad-loop" / "policy.toml")

    with pytest.raises(SystemExit, match=r"weird.*ghostkind.*generic"):
        cli._make_adapters(project.project, _run_dir(project.project), pol)


def test_make_adapters_generic_shares_synthesizing_but_not_triage(
    fresh_adapter_registry, project, monkeypatch
):
    """The ``(cfg, synthesizes)`` cache with the real builtin generic kind: dev and
    review (same cfg, both bmad-dev-auto) share one GenericDevAdapter, triage on the
    same cfg gets a separate plain GenericAdapter."""
    from bmad_loop.adapters.generic import GenericAdapter, GenericDevAdapter

    monkeypatch.setattr(mux_mod, "_usable", lambda mux: True)
    install_bmad_config(project)
    adapters = cli._make_adapters(project.project, _run_dir(project.project), policy_mod.load(None))
    assert isinstance(adapters["dev"], GenericDevAdapter)
    assert adapters["dev"] is adapters["review"]
    assert isinstance(adapters["triage"], GenericAdapter)
    assert not isinstance(adapters["triage"], GenericDevAdapter)
    assert adapters["triage"] is not adapters["dev"]


def test_adapters_command_lists_builtins_and_surfaces_failures(
    scan_adapter_registry, capsys, tmp_path
):
    """`bmad-loop adapters` renders the kind table and names a failed out-of-tree
    package — the one place an operator looks when an installed adapter is missing."""
    import argparse

    registry, arm = scan_adapter_registry

    def boom():
        raise ImportError("No module named 'ghost_dependency'")

    arm(_FakeEntryPoint("brokenadapter", boom))
    args = argparse.Namespace(project=tmp_path)
    assert cli.cmd_adapters(args) == 0
    captured = capsys.readouterr()
    assert "generic" in captured.out  # the table renders
    assert "opencode-http" in captured.out
    assert "brokenadapter" in captured.err
    assert "ghost_dependency" in captured.err


def test_adapters_command_flags_dangling_kind_reference(fresh_adapter_registry, capsys, tmp_path):
    """A project profile whose adapter kind never registered is named as a warning:
    the table can't show a kind that isn't there, so the dangling reference is
    surfaced explicitly."""
    import argparse

    _write_profile(tmp_path, "weird", adapter="ghostkind")
    args = argparse.Namespace(project=tmp_path)
    assert cli.cmd_adapters(args) == 0
    captured = capsys.readouterr()
    assert "ghostkind" in captured.err
    assert "weird" in captured.err


def test_make_adapters_opencode_http_dispatch_unchanged(
    fresh_adapter_registry, project, monkeypatch
):
    """Regression pin: routing the ``opencode-http`` profile through the registry
    yields exactly the pre-registry adapters — OpencodeDevAdapter for the
    synthesizing roles (never resolving a mux), OpencodeHttpAdapter for triage."""
    from bmad_loop.adapters import opencode_http
    from bmad_loop.adapters.opencode_http import OpencodeDevAdapter, OpencodeHttpAdapter

    def no_mux():
        raise AssertionError("hookless opencode-http must not resolve a multiplexer")

    monkeypatch.setattr(opencode_http, "_require_httpx", lambda: object())
    monkeypatch.setattr(mux_mod, "get_multiplexer", no_mux)
    install_bmad_config(project)
    _write_policy(project.project, '[adapter]\nname = "opencode"\n')  # alias → opencode-http
    pol = policy_mod.load(project.project / ".bmad-loop" / "policy.toml")

    adapters = cli._make_adapters(project.project, _run_dir(project.project), pol)
    assert isinstance(adapters["dev"], OpencodeDevAdapter)
    assert adapters["dev"] is adapters["review"]
    assert adapters["dev"].profile.adapter == "opencode-http"
    assert isinstance(adapters["triage"], OpencodeHttpAdapter)
    assert not isinstance(adapters["triage"], OpencodeDevAdapter)
