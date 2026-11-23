"""Session-store access that works on both sides of the Hermes storage refactor.

Hermes removed the old ``hermes_state.SessionDB`` (mixins, ``TITLE_SOURCE_*`` class
constants, ``get_session_title*``, ``set_auto_title``, ``_execute_write`` ...) in favour of
a storage interface (``persistence_assembly`` + typed repositories). ``hermes_state`` now
ships only a deprecated stand-in whose ``SessionDB`` raises ``AttributeError`` for every
title getter/setter, which is what broke background title evaluation.

``SessionDB`` below is a thin adapter exposing exactly the legacy surface the titler uses,
implemented on the storage interface. ``open_session_db()`` returns it on current hosts and
falls back to the legacy ``hermes_state.SessionDB`` on older ones.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

log = logging.getLogger(__name__)

TITLE_SOURCE_DERIVED = "derived"
TITLE_SOURCE_LLM = "llm"
TITLE_SOURCE_USER = "user"
MAX_TITLE_LENGTH = 100


class SessionDB:
    """Legacy-shaped facade over the Hermes storage interface (current hosts)."""

    TITLE_SOURCE_DERIVED = TITLE_SOURCE_DERIVED
    TITLE_SOURCE_LLM = TITLE_SOURCE_LLM
    TITLE_SOURCE_USER = TITLE_SOURCE_USER
    MAX_TITLE_LENGTH = MAX_TITLE_LENGTH

    def __init__(self) -> None:
        from persistence_assembly import open_persistence, profile_scope

        # Handles on one profile share a single backend; keep ours open for the process.
        self._store = open_persistence(profile_scope(), eager=True)

    def close(self) -> None:
        self._store.close()

    # -- titles ------------------------------------------------------------

    def get_session_title(self, session_id: str) -> Optional[str]:
        return self._store.sessions.get_title(session_id)

    def get_session_title_source(self, session_id: str) -> Optional[str]:
        src = self._store.sessions.get_title_source(session_id)
        return None if src is None else src.value

    def set_session_title(self, session_id: str, title: str) -> bool:
        """User-provenance write (raises ValueError on a title conflict)."""
        return bool(self._store.sessions.set_title(session_id, title))

    def set_session_title_source(self, session_id: str, source: str) -> bool:
        from persistence.conversation.sessions import TitleSource

        return bool(self._store.sessions.set_title_source(session_id, TitleSource(source)))

    def set_auto_title(self, session_id: str, title: str, source: str = TITLE_SOURCE_LLM) -> bool:
        """Write an automatic title over an untitled row or strictly lower provenance.

        Raises ValueError (``TitleConflict``) when another session holds the title.
        """
        from persistence.conversation.sessions import TitleSource

        return bool(self._store.sessions.set_auto_title(session_id, title, source=TitleSource(source)))

    def rewrite_llm_title(self, session_id: str, title: str, expected_title: Optional[str]) -> bool:
        """Replace an existing ``llm`` title without ever overwriting a user title.

        The store has no llm->llm verb (``set_auto_title`` only upgrades strictly lower
        provenance), so demote the row to ``derived`` and let ``set_auto_title`` - itself a
        compare-and-swap that refuses once a ``user`` title holds the row - write the new one.
        Not a single transaction, so every step re-checks and undoes itself:

        * the title/source must still be ``expected_title`` / ``llm`` before demoting;
        * if the title changed between that check and the demotion (a user ``/title``
          landed in the gap), the user's provenance is restored and nothing is written;
        * if the write is refused or conflicts, the previous ``llm`` provenance is restored.

        Returns True when the new title was written; raises ValueError on a title conflict.
        """
        sessions = self._store.sessions
        from persistence.conversation.sessions import TitleSource

        if sessions.get_title(session_id) != expected_title:
            return False
        if sessions.get_title_source(session_id) != TitleSource.LLM:
            return False
        if not sessions.set_title_source(session_id, TitleSource.DERIVED):
            return False

        if sessions.get_title(session_id) != expected_title:
            # Someone retitled us inside the gap; that write was theirs, not ours to demote.
            sessions.set_title_source(session_id, TitleSource.USER)
            return False

        try:
            written = bool(sessions.set_auto_title(session_id, title, source=TitleSource.LLM))
        except BaseException:
            self._restore_llm(session_id, expected_title)
            raise
        if not written:
            self._restore_llm(session_id, expected_title)
        return written

    def _restore_llm(self, session_id: str, expected_title: Optional[str]) -> None:
        sessions = self._store.sessions
        try:
            if (
                sessions.get_title(session_id) == expected_title
                and sessions.get_title_source(session_id) is not None
                and sessions.get_title_source(session_id).value == TITLE_SOURCE_DERIVED
            ):
                sessions.set_title_source(session_id, _llm())
        except Exception as exc:  # pragma: no cover - best effort undo
            log.warning("auto-titler %s: failed to restore llm provenance: %s", session_id[:12], exc)

    # -- reads -------------------------------------------------------------

    def list_sessions_rich(
        self,
        limit: int = 20,
        min_message_count: int = 0,
        include_children: bool = False,
        order_by_last_active: bool = False,
        **_ignored: Any,
    ) -> List[Dict[str, Any]]:
        from persistence.conversation.sessions import SessionOrder, SessionQuery, SessionWindow

        query = SessionQuery(
            include_children=include_children,
            min_message_count=min_message_count,
            order=SessionOrder.LAST_ACTIVE if order_by_last_active else SessionOrder.STARTED_AT,
        )
        window = SessionWindow(limit=None if limit is not None and limit < 0 else limit, offset=0)
        return [e.as_list_row() for e in self._store.sessions.list_sessions(query, window)]

    def get_messages_as_conversation(
        self,
        session_id: str,
        include_ancestors: bool = False,
        include_inactive: bool = False,
        include_compacted: bool = False,
    ) -> List[Dict[str, Any]]:
        from persistence.conversation.messages import ConversationRequest, TranscriptView

        if include_inactive:
            view = TranscriptView.ALL
        else:
            view = TranscriptView.DISPLAY if include_compacted else TranscriptView.ACTIVE
        request = ConversationRequest(view=view, include_lineage=include_ancestors)
        return [m.as_openai_message() for m in self._store.messages.read_conversation(session_id, request)]

    # -- accounting --------------------------------------------------------

    def record_auxiliary_usage(
        self,
        session_id: str,
        task: str,
        *,
        model: Optional[str] = None,
        billing_provider: Optional[str] = None,
        input_tokens: int = 0,
        output_tokens: int = 0,
        cache_read_tokens: int = 0,
        cache_write_tokens: int = 0,
        estimated_cost_usd: Optional[float] = None,
        **_ignored: Any,
    ) -> None:
        from persistence.conversation.usage import TokenCounts

        self._store.usage.record_auxiliary_usage(
            session_id,
            task,
            model=model,
            billing_provider=billing_provider,
            tokens=TokenCounts(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cache_read_tokens=cache_read_tokens,
                cache_write_tokens=cache_write_tokens,
            ),
            estimated_cost_usd=estimated_cost_usd,
        )


def _llm():
    from persistence.conversation.sessions import TitleSource

    return TitleSource.LLM


def open_session_db():
    """Open the profile's session store: the adapter on current hosts, legacy ``SessionDB`` otherwise."""
    try:
        import persistence_assembly  # noqa: F401
    except ImportError:
        from hermes_state import SessionDB as LegacySessionDB

        return LegacySessionDB()
    return SessionDB()
