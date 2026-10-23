import json
from http.server import BaseHTTPRequestHandler


class LocalHandler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def send(self, status, value, content_type='application/json'):
        raw = json.dumps(value, allow_nan=False).encode() if content_type == 'application/json' else value
        self.send_response(status)
        self.send_header('Content-Type', content_type)
        self.send_header('Content-Length', str(len(raw)))
        self.send_header('Cache-Control', 'no-store')
        self.send_header('X-Content-Type-Options', 'nosniff')
        self.send_header('Content-Security-Policy', "default-src 'self'; style-src 'self' 'unsafe-inline'; script-src 'self' 'unsafe-inline'; img-src 'self' blob:; frame-ancestors 'none'")
        self.end_headers()
        self.wfile.write(raw)

    def read_request(self):
        if self.headers.get('Origin') != f'http://{self.headers.get("Host")}':
            raise ValueError('same_origin_required')
        size = int(self.headers.get('Content-Length', '0'))
        if not 0 < size <= 8192:
            raise ValueError('invalid_request_size')
        return json.loads(self.rfile.read(size), parse_constant=lambda _: (_ for _ in ()).throw(ValueError('nonfinite request')))
