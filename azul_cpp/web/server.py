from http.server import HTTPServer, SimpleHTTPRequestHandler

class MyHandler(SimpleHTTPRequestHandler):
    def end_headers(self):
        # 开启多线程 Wasm 必须的两个 Header
        self.send_header("Cross-Origin-Opener-Policy", "same-origin")
        self.send_header("Cross-Origin-Embedder-Policy", "require-corp")
        super().end_headers()

print("Server started at http://localhost:8080")
HTTPServer(('localhost', 8080), MyHandler).serve_forever()