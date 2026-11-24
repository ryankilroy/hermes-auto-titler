"""store.SessionDB adapter: title provenance rules on the Hermes storage interface.

The Hermes repositories are replaced by an in-memory fake that enforces the same
contract (``set_auto_title`` only upgrades strictly lower provenance; ``set_title``
writes user provenance; a held title raises a ``ValueError``). The real store is
exercised by ``scripts/live_store_check.py``.
"""

import enum
import sys
import types

import pytest

from hermes_auto_titler import store as store_mod


class TitleSource(enum.Enum):
    DERIVED = "derived"
    LLM = "llm"
    USER = "user"


_RANK = {None: -1, TitleSource.DERIVED: 0, TitleSource.LLM: 1, TitleSource.USER: 2}


class FakeSessions:
    def __init__(self):
        self.rows = {}  # sid -> [title, source]
        self.before_set_source = None  # hook(sid, source) -> simulate a racing writer
        self.before_set_auto = None

    def add(self, sid, title=None, source=None):
        self.rows[sid] = [title, source]

    def get_title(self, sid):
        return self.rows[sid][0]

    def get_title_source(self, sid):
        return self.rows[sid][1]

    def _check_unique(self, sid, title):
        for other, (t, _s) in self.rows.items():
            if other != sid and t == title:
                raise ValueError(f"Title '{title}' is already in use by session {other}")

    def set_title(self, sid, title):
        self._check_unique(sid, title)
        self.rows[sid] = [title, TitleSource.USER]
        return True

    def set_title_source(self, sid, source):
        if self.before_set_source:
            self.before_set_source(sid, source)
        if self.rows[sid][0] is None:
            return False
        self.rows[sid][1] = source
        return True

    def set_auto_title(self, sid, title, *, source):
        if self.before_set_auto:
            self.before_set_auto(sid, title)
        if source is TitleSource.USER:
            raise ValueError("invalid automatic title source")
        cur_t, cur_s = self.rows[sid]
        if cur_t is not None and _RANK[cur_s] >= _RANK[source]:
            return False
        self._check_unique(sid, title)
        self.rows[sid] = [title, source]
        return True


@pytest.fixture
def adapter(monkeypatch):
    mod = types.ModuleType("persistence.conversation.sessions")
    mod.TitleSource = TitleSource
    for name in ("persistence", "persistence.conversation"):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    monkeypatch.setitem(sys.modules, "persistence.conversation.sessions", mod)

    sessions = FakeSessions()
    db = store_mod.SessionDB.__new__(store_mod.SessionDB)
    db._store = types.SimpleNamespace(sessions=sessions)
    return db, sessions


def test_getters_and_constants(adapter):
    db, sessions = adapter
    sessions.add("s1", "T", TitleSource.LLM)
    sessions.add("s2")
    assert db.get_session_title("s1") == "T"
    assert db.get_session_title_source("s1") == store_mod.SessionDB.TITLE_SOURCE_LLM == "llm"
    assert db.get_session_title_source("s2") is None
    assert store_mod.SessionDB.TITLE_SOURCE_USER == "user"
    assert store_mod.SessionDB.TITLE_SOURCE_DERIVED == "derived"


def test_set_auto_title_translates_source_string(adapter):
    db, sessions = adapter
    sessions.add("s1")
    assert db.set_auto_title("s1", "A", source="derived") is True
    assert db.set_auto_title("s1", "B", source="llm") is True
    assert sessions.rows["s1"] == ["B", TitleSource.LLM]
    # llm -> llm is the store's no-op; rewriting needs rewrite_llm_title.
    assert db.set_auto_title("s1", "C", source="llm") is False


def test_rewrite_replaces_llm_title_and_keeps_llm_provenance(adapter):
    db, sessions = adapter
    sessions.add("s1", "Old", TitleSource.LLM)
    assert db.rewrite_llm_title("s1", "New", "Old") is True
    assert sessions.rows["s1"] == ["New", TitleSource.LLM]


def test_rewrite_refuses_stale_expected_title(adapter):
    db, sessions = adapter
    sessions.add("s1", "Changed", TitleSource.LLM)
    assert db.rewrite_llm_title("s1", "New", "Old") is False
    assert sessions.rows["s1"] == ["Changed", TitleSource.LLM]


@pytest.mark.parametrize("source", [TitleSource.USER, TitleSource.DERIVED, None])
def test_rewrite_only_applies_to_llm_titles(adapter, source):
    db, sessions = adapter
    sessions.add("s1", "Old", source)
    assert db.rewrite_llm_title("s1", "New", "Old") is False
    assert sessions.rows["s1"] == ["Old", source]


def test_rewrite_conflict_raises_and_restores_llm_provenance(adapter):
    db, sessions = adapter
    sessions.add("s1", "Old", TitleSource.LLM)
    sessions.add("s2", "Taken", TitleSource.USER)
    with pytest.raises(ValueError, match="already in use"):
        db.rewrite_llm_title("s1", "Taken", "Old")
    assert sessions.rows["s1"] == ["Old", TitleSource.LLM]


def test_user_title_landing_before_demotion_is_restored_not_overwritten(adapter):
    db, sessions = adapter
    sessions.add("s1", "Old", TitleSource.LLM)

    def user_retitles_first(sid, source):
        if source is TitleSource.DERIVED:
            sessions.rows[sid] = ["User pick", TitleSource.USER]

    sessions.before_set_source = user_retitles_first
    assert db.rewrite_llm_title("s1", "New", "Old") is False
    # the demotion briefly touched the user's row; provenance must be back to user
    assert sessions.rows["s1"] == ["User pick", TitleSource.USER]


def test_user_title_landing_after_demotion_wins(adapter):
    db, sessions = adapter
    sessions.add("s1", "Old", TitleSource.LLM)

    def user_retitles_mid_write(sid, title):
        sessions.rows[sid] = ["User pick", TitleSource.USER]

    sessions.before_set_auto = user_retitles_mid_write
    assert db.rewrite_llm_title("s1", "New", "Old") is False
    assert sessions.rows["s1"] == ["User pick", TitleSource.USER]


def test_open_session_db_falls_back_to_legacy_on_old_hosts(monkeypatch):
    monkeypatch.setitem(sys.modules, "persistence_assembly", None)  # ImportError
    db = store_mod.open_session_db()
    assert not isinstance(db, store_mod.SessionDB)


# -- retry-sweep secret scope -------------------------------------------------

def _fake_secret_scope(monkeypatch, bound):
    calls = []
    mod = types.ModuleType("agent.secret_scope")
    mod.build_profile_secret_scope = lambda home: {"home": str(home)}
    mod.current_secret_scope = lambda: bound[0]
    mod.set_secret_scope = lambda secrets, profile_home=None: calls.append(("set", profile_home)) or bound.__setitem__(0, secrets) or "tok"
    mod.reset_secret_scope = lambda token: calls.append(("reset", token)) or bound.__setitem__(0, None)
    monkeypatch.setitem(sys.modules, "agent.secret_scope", mod)
    return calls


def test_profile_secret_scope_binds_and_releases(monkeypatch, tmp_path):
    from hermes_auto_titler.titler import profile_secret_scope

    bound = [None]
    calls = _fake_secret_scope(monkeypatch, bound)
    with profile_secret_scope(tmp_path):
        assert bound[0] == {"home": str(tmp_path)}
    assert bound[0] is None
    assert calls == [("set", str(tmp_path)), ("reset", "tok")]


def test_profile_secret_scope_releases_on_error(monkeypatch, tmp_path):
    from hermes_auto_titler.titler import profile_secret_scope

    bound = [None]
    _fake_secret_scope(monkeypatch, bound)
    with pytest.raises(RuntimeError):
        with profile_secret_scope(tmp_path):
            raise RuntimeError("boom")
    assert bound[0] is None


def test_profile_secret_scope_keeps_an_existing_scope(monkeypatch, tmp_path):
    from hermes_auto_titler.titler import profile_secret_scope

    bound = [{"turn": "scope"}]
    calls = _fake_secret_scope(monkeypatch, bound)
    with profile_secret_scope(tmp_path):
        assert bound[0] == {"turn": "scope"}
    assert calls == []


def test_profile_secret_scope_is_a_noop_on_hosts_without_scopes(monkeypatch, tmp_path):
    from hermes_auto_titler.titler import profile_secret_scope

    monkeypatch.setitem(sys.modules, "agent.secret_scope", None)  # ImportError
    with profile_secret_scope(tmp_path):
        pass
