"""
Web server for playing Trump against the trained networks.

Every browser (identified by a cookie) gets its own game.

Usage (from the repository root):
    python -m webapp.server            # then open http://127.0.0.1:8000
    python -m webapp.server --port 9000
    python -m webapp.server --demo --host 0.0.0.0     # for friends: private in-memory stats, capped sessions
"""

import argparse
import json
import os
import threading
import webbrowser
from http.cookies import SimpleCookie
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from webapp.sessions import Busy, SessionManager

STATIC_DIR = os.path.join(os.path.dirname(__file__), 'static')
manager = None
COOKIE_MAX_AGE = 30 * 24 * 3600


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def _session(self):
        """This visitor's Session (a new cookie is queued if they are new); None if the server is full."""
        cookie = SimpleCookie(self.headers.get('Cookie', ''))
        sid = cookie['sid'].value if 'sid' in cookie else None
        try:
            session, new_sid = manager.get(sid)
        except Busy:
            return None
        if new_sid:
            self.new_sid = new_sid
        return session

    def _send(self, code, body, content_type='application/json'):
        data = body if isinstance(body, bytes) else json.dumps(body).encode()
        self.send_response(code)
        if getattr(self, 'new_sid', None):
            self.send_header('Set-Cookie', 'sid=%s; Path=/; HttpOnly; SameSite=Lax; Max-Age=%d' % (self.new_sid, COOKIE_MAX_AGE))
        self.send_header('Content-Type', content_type)
        self.send_header('Content-Length', str(len(data)))
        self.send_header('Cache-Control', 'no-store')
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path.startswith('/api/'):
            session = self._session()
            if session is None:
                self._send(503, {'busy': True})
            elif self.path.startswith('/api/state'):
                self._send(200, session.snapshot())
            elif self.path == '/api/stats':
                self._send(200, session.stats_report())
            else:
                self._send(404, {'error': 'not found'})
        elif self.path in ('/', '/index.html'):
            with open(os.path.join(STATIC_DIR, 'index.html'), 'rb') as f:
                self._send(200, f.read(), 'text/html; charset=utf-8')
        else:
            self._send(404, {'error': 'not found'})

    def do_POST(self):
        length = int(self.headers.get('Content-Length', 0))
        body = json.loads(self.rfile.read(length) or b'{}')
        session = self._session()
        if session is None:
            self._send(503, {'ok': False, 'busy': True, 'error': 'The table is full'})
        elif self.path == '/api/new_game':
            session.restart()
            self._send(200, {'ok': True})
        elif self.path == '/api/settings':
            session.update_settings(body.get('randomness'), body.get('reveal'))
            self._send(200, {'ok': True})
        elif self.path == '/api/model':
            ok, error = session.set_model(body.get('model'))
            self._send(200 if ok else 400, {'ok': ok, 'error': error})
        elif self.path == '/api/reset_stats':
            session.stats.reset()
            self._send(200, {'ok': True})
        elif self.path in ('/api/bid', '/api/card', '/api/ack'):
            kind = self.path.rsplit('/', 1)[1]
            ok, error = session.submit(kind, body.get('value'))
            self._send(200 if ok else 400, {'ok': ok, 'error': error})
        else:
            self._send(404, {'error': 'not found'})


def main():
    global manager
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=8000)
    parser.add_argument('--host', default='127.0.0.1', help='0.0.0.0 makes the server reachable from other machines')
    parser.add_argument('--demo', action='store_true', help='private in-memory stats per visitor instead of the shared stats files')
    parser.add_argument('--max-sessions', type=int, default=10, help='most simultaneous visitors')
    parser.add_argument('--idle-minutes', type=float, default=30, help='close an idle visitor game after this many minutes')
    parser.add_argument('--no-browser', action='store_true')
    args = parser.parse_args()

    manager = SessionManager(persist_stats=not args.demo, max_sessions=args.max_sessions,
                             idle_seconds=args.idle_minutes * 60)
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    url = 'http://%s:%d' % ('127.0.0.1' if args.host == '0.0.0.0' else args.host, args.port)
    print('Trump table ready at', url)
    if not args.no_browser and not args.demo:
        threading.Timer(0.8, webbrowser.open, args=(url,)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()
