"""
One game Session per visitor.

A visitor is identified by a random "sid" cookie set by the server. Each visitor gets their own Session
(own game, AI settings and, in demo mode, own in-memory statistics). Sessions nobody has touched for a while
are closed, which stops their game thread, and the number of simultaneous sessions is capped.
"""

import secrets
import threading
import time

from webapp.game_session import Session


class Busy(Exception):
    """Raised when every session slot is taken."""


class SessionManager:
    def __init__(self, persist_stats=True, max_sessions=10, idle_seconds=30 * 60):
        """
        Args:
            persist_stats (bool): True keeps the shared stats files (local use); False gives every visitor
                private in-memory statistics (demo)
            max_sessions (int): Most simultaneous visitors; further ones get Busy
            idle_seconds (float): A session with no request for this long is closed
        """
        self.persist_stats = persist_stats
        self.max_sessions = max_sessions
        self.idle_seconds = idle_seconds
        self.lock = threading.Lock()
        self.sessions = {}   # sid -> [Session, last_used]
        threading.Thread(target=self._reap_forever, daemon=True).start()

    def get(self, sid):
        """
        Find the visitor's session, creating one when the cookie is missing or unknown.

        Returns:
            (Session, str): The session, and a new sid the browser must store (None if its sid was valid)

        Raises:
            Busy: A new session is needed but max_sessions are already running
        """
        with self.lock:
            entry = self.sessions.get(sid)
            if entry:
                entry[1] = time.time()
                return entry[0], None
            self._reap()
            if len(self.sessions) >= self.max_sessions:
                raise Busy()
            new_sid = secrets.token_urlsafe(16)
            session = Session(persist_stats=self.persist_stats)
            self.sessions[new_sid] = [session, time.time()]
            return session, new_sid

    def _reap(self):
        """Close sessions idle for too long. The caller holds the lock."""
        now = time.time()
        for sid in [s for s, (_, last) in self.sessions.items() if now - last > self.idle_seconds]:
            self.sessions.pop(sid)[0].close()

    def _reap_forever(self):
        while True:
            time.sleep(60)
            with self.lock:
                self._reap()
