"""Live check: the retry-sweep thread can read secrets under a multiplexing gateway.

Run with the Hermes venv python and PYTHONPATH=<hermes-agent>. Uses a throwaway HERMES_HOME.
"""
import os
import sys
import tempfile
import threading
from pathlib import Path

home = Path(tempfile.mkdtemp(prefix="hat-scope-"))
(home / ".env").write_text("ANTHROPIC_API_KEY=sk-live-check\n", encoding="utf-8")
os.environ["HERMES_HOME"] = str(home)
os.environ.pop("ANTHROPIC_API_KEY", None)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent.secret_scope import UnscopedSecretError, get_secret, set_multiplex_active  # noqa: E402

from hermes_auto_titler.titler import profile_secret_scope  # noqa: E402

set_multiplex_active(True)  # what the gateway does at startup
out = {}


def bare():
    try:
        get_secret("ANTHROPIC_API_KEY")
        out["bare"] = "read ok"
    except UnscopedSecretError:
        out["bare"] = "UnscopedSecretError"


def scoped():
    with profile_secret_scope(home):
        out["scoped"] = get_secret("ANTHROPIC_API_KEY")
        with profile_secret_scope(home):  # nested: no-op, no double bind
            out["nested"] = get_secret("ANTHROPIC_API_KEY")
    try:
        get_secret("ANTHROPIC_API_KEY")
        out["after"] = "leaked"
    except UnscopedSecretError:
        out["after"] = "released"


for fn in (bare, scoped):
    t = threading.Thread(target=fn)
    t.start()
    t.join()
print(out)
assert out == {"bare": "UnscopedSecretError", "scoped": "sk-live-check",
               "nested": "sk-live-check", "after": "released"}, out
print("SCOPE LIVE OK")
