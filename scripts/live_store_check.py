"""Live check of hermes_auto_titler.store against the REAL Hermes storage interface
in an isolated HERMES_HOME. Run with the Hermes venv python + PYTHONPATH=<hermes-agent>."""
import os
import sys
import tempfile
from pathlib import Path

home = Path(tempfile.mkdtemp(prefix="hat-live-"))
os.environ["HERMES_HOME"] = str(home)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from persistence.conversation.sessions import NewSession  # noqa: E402
from persistence_assembly import open_persistence, profile_scope  # noqa: E402

from hermes_auto_titler.store import SessionDB, open_session_db  # noqa: E402

db = open_session_db()
assert isinstance(db, SessionDB), type(db)

raw = open_persistence(profile_scope(), eager=True)
for sid in ("s1", "s2", "s3"):
    raw.sessions.create(NewSession(session_id=sid, source="cli"))

def src(s):
    return db.get_session_title_source(s)

# 1. untitled -> llm
assert db.get_session_title_source("s1") is None
assert db.set_auto_title("s1", "First title", source="llm") is True
assert (db.get_session_title("s1"), src("s1")) == ("First title", "llm")

# 2. llm -> llm rewrite (the case the plugin exists for)
assert db.rewrite_llm_title("s1", "Second title", "First title") is True
assert (db.get_session_title("s1"), src("s1")) == ("Second title", "llm")

# 3. stale expected title -> refused, untouched
assert db.rewrite_llm_title("s1", "Nope", "First title") is False
assert (db.get_session_title("s1"), src("s1")) == ("Second title", "llm")

# 4. user title is never overwritten, by either path
assert db.set_session_title("s2", "Mine") is True
assert src("s2") == "user"
assert db.rewrite_llm_title("s2", "Auto", "Mine") is False
assert db.set_auto_title("s2", "Auto", source="llm") is False
assert (db.get_session_title("s2"), src("s2")) == ("Mine", "user")

# 5. uniqueness conflict raises ValueError and leaves s1 as it was (llm provenance restored)
try:
    db.rewrite_llm_title("s1", "Mine", "Second title")
except ValueError as exc:
    assert "already in use" in str(exc), exc
else:
    raise AssertionError("expected ValueError on title conflict")
assert (db.get_session_title("s1"), src("s1")) == ("Second title", "llm")

# 6. derived -> llm upgrade through set_auto_title
assert db.set_auto_title("s3", "Draft", source="derived") is True
assert src("s3") == "derived"
assert db.set_auto_title("s3", "Better", source="llm") is True
assert (db.get_session_title("s3"), src("s3")) == ("Better", "llm")

# 7. read paths + accounting don't raise
rows = db.list_sessions_rich(limit=10, include_children=False, order_by_last_active=True)
assert isinstance(rows, list)
assert db.list_sessions_rich(limit=1000, min_message_count=0, include_children=False) is not None
raw.messages.append("s1", __import__("persistence.conversation.messages", fromlist=["NewMessage"]).NewMessage.from_live(
    dict(role="user", content="hello world")))
conv = db.get_messages_as_conversation("s1", include_ancestors=True, include_compacted=True)
assert conv and conv[0]["role"] == "user", conv
db.record_auxiliary_usage("s1", "hermes_auto_titler", model="m", billing_provider="p",
                          input_tokens=10, output_tokens=2, estimated_cost_usd=None)

print("LIVE OK", home)
