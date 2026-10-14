"""Real HTTP transport against a local deterministic Chat Completions server."""
import json
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from coding_agent.model import OpenAICompatibleModel


class ModelTests(unittest.TestCase):
    def test_real_http_chat_completions_and_wire_tool_call_id(self):
        requests = []
        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_):
                pass
            def do_POST(self):
                self.server.testcase.assertEqual(self.path, '/v1/chat/completions')
                self.server.testcase.assertEqual(self.headers['Authorization'], 'Bearer fake-test-key')
                data = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                requests.append(data)
                if len(requests) == 1:
                    message = {'role': 'assistant', 'content': None, 'tool_calls': [{'id': 'call-test', 'type': 'function', 'function': {'name': 'read_file', 'arguments': '{"path":"order.py"}'}}]}
                else:
                    self.server.testcase.assertEqual(data['messages'][-1]['tool_call_id'], 'call-test')
                    message = {'role': 'assistant', 'content': 'Done'}
                body = json.dumps({'choices': [{'message': message}]}).encode()
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Content-Length', str(len(body)))
                self.end_headers()
                self.wfile.write(body)
        server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        server.testcase = self
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            model = OpenAICompatibleModel(
                f'http://127.0.0.1:{server.server_port}/v1',
                'fake-test-key',
                'test-model',
                reasoning_effort='none',
                timeout_seconds=5,
            )
            first = model.complete([{'role': 'user', 'content': 'Inspect order.py'}], [])
            self.assertEqual(first.tool_calls[0]['arguments'], {'path': 'order.py'})
            second = model.complete([{'role': 'tool', 'tool_call_id': first.tool_calls[0]['id'], 'content': 'code'}], [])
            self.assertEqual(second.content, 'Done')
            self.assertEqual(requests[0]['model'], 'test-model')
            self.assertEqual(requests[0]['reasoning_effort'], 'none')
        finally:
            server.shutdown()
            thread.join(timeout=5)
            server.server_close()

    def test_model_timeout_must_be_positive_number(self):
        with self.assertRaisesRegex(ValueError, 'positive number'):
            OpenAICompatibleModel('http://localhost/v1', 'key', 'model', timeout_seconds=0)
        with self.assertRaisesRegex(ValueError, 'positive number'):
            OpenAICompatibleModel('http://localhost/v1', 'key', 'model', timeout_seconds='slow')
