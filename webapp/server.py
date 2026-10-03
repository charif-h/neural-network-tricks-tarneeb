"""
Local web server for playing Trump against the trained networks.

Usage (from the repository root):
    python -m webapp.server            # then open http://127.0.0.1:8000
    python -m webapp.server --port 9000
"""

import argparse
import json
import os
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from webapp.game_session import Session

STATIC_DIR = os.path.join(os.path.dirname(__file__), 'static')
session = None


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def _send(self, code, body, content_type='application/json'):
        data = body if isinstance(body, bytes) else json.dumps(body).encode()
        self.send_response(code)
        self.send_header('Content-Type', content_type)
        self.send_header('Content-Length', str(len(data)))
        self.send_header('Cache-Control', 'no-store')
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path.startswith('/api/state'):
            self._send(200, session.snapshot())
        elif self.path == '/api/stats':
            self._send(200, session.stats_report())
        elif self.path in ('/', '/index.html'):
            with open(os.path.join(STATIC_DIR, 'index.html'), 'rb') as f:
                self._send(200, f.read(), 'text/html; charset=utf-8')
        else:
            self._send(404, {'error': 'not found'})

    def do_POST(self):
        length = int(self.headers.get('Content-Length', 0))
        body = json.loads(self.rfile.read(length) or b'{}')
        if self.path == '/api/new_game':
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
    global session
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=8000)
    parser.add_argument('--no-browser', action='store_true')
    args = parser.parse_args()

    session = Session()
    server = ThreadingHTTPServer(('127.0.0.1', args.port), Handler)
    url = 'http://127.0.0.1:%d' % args.port
    print('Trump table ready at', url)
    if not args.no_browser:
        threading.Timer(0.8, webbrowser.open, args=(url,)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()
