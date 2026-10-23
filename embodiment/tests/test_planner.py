import json
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from pathlib import Path
import tempfile
import threading
import unittest
from embodiment.benchmark import BIN,COLORS,CUBE,layout
from embodiment.planner import Planner,PlannerError,scene_description,validate_instruction
from interactive.actions import DEFAULT_MODEL


def cubes():
    return {color:(*pose,1,0,0,0) for color,pose in layout(0).items()}


class CubeInstructionValidation(unittest.TestCase):
    def test_known_target_and_completion(self):
        scene = scene_description(cubes())
        content = json.dumps({'action':'pick_and_place','target_color':'red','destination':'bin'})
        self.assertEqual(validate_instruction(content,scene)['instruction'],'Pick up the red cube and place it in the bin.')
        done = json.dumps({'action':'done','target_color':None,'destination':None})
        with self.assertRaisesRegex(PlannerError,'premature'):
            validate_instruction(done,scene)
        placed = {color:(*BIN[:2],BIN[2]+CUBE/2,1,0,0,0) for color in COLORS}
        self.assertIsNone(validate_instruction(done,scene_description(placed))['instruction'])

    def test_untrusted_or_unsupported_output(self):
        scene = scene_description(cubes())
        invalid = ('{"action":"pick_and_place","target_color":"red","target_color":"blue","destination":"bin"}',
                   '{"action":"pick_and_place","target_color":"yellow","destination":"bin"}',
                   '{"action":"pick_and_place","target_color":"red","destination":"shelf"}',
                   '{"action":"execute_code","target_color":"red","destination":"bin"}',
                   '{"action":"pick_and_place","target_color":[],"destination":"bin"}',
                   '```json\n{}\n```','null','{"action":NaN}')
        for content in invalid:
            with self.subTest(content=content),self.assertRaises(PlannerError):
                validate_instruction(content,scene)

    def test_missing_and_unreachable_objects(self):
        with self.assertRaises(PlannerError):
            scene_description({'red':(0,)*7})
        data = cubes();data['red']=(.8,0.,.0125,1.,0.,0.,0.)
        with self.assertRaisesRegex(PlannerError,'unavailable'):
            validate_instruction('{"action":"pick_and_place","target_color":"red","destination":"bin"}',scene_description(data))


class PlannerMockHTTP(unittest.TestCase):
    def test_reuses_authenticated_http_and_redacted_audit(self):
        received = []
        class Handler(BaseHTTPRequestHandler):
            def log_message(self,*args): pass
            def reply(self,data):
                body = json.dumps(data).encode();self.send_response(200);self.send_header('Content-Length',str(len(body)));self.end_headers();self.wfile.write(body)
            def do_GET(self):
                self.reply({'data':[{'id':DEFAULT_MODEL}]})
            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                received.append((self.headers.get('Authorization'),body))
                output = json.dumps({'action':'pick_and_place','target_color':'red','destination':'bin'})
                self.reply({'id':'mock-completion','choices':[{'finish_reason':'stop','message':{'content':output}}],
                            'usage':{'total_tokens':20}})
        server = ThreadingHTTPServer(('127.0.0.1',0),Handler)
        thread = threading.Thread(target=server.serve_forever,daemon=True);thread.start()
        try:
            with tempfile.TemporaryDirectory() as temp:
                client = Planner(base=f'http://127.0.0.1:{server.server_port}/v1/',key='test-only',timeout=2)
                path = Path(temp)/'planner.json'
                result = client.plan('clear the table',cubes(),path)
                self.assertEqual(result['target_color'],'red')
                audit = json.loads(path.read_text())
                self.assertEqual(audit['mode'],'mock')
                self.assertTrue(audit['completion_executed'])
                self.assertNotIn('test-only',path.read_text())
                self.assertNotIn('clear the table',path.read_text())
                with self.assertRaises(FileExistsError):
                    client.plan('clear the table',cubes(),path)
                self.assertEqual(len(received),1)
                self.assertEqual(received[0][0],'Bearer test-only')
                self.assertEqual(received[0][1]['response_format']['type'],'json_schema')
                with self.assertRaisesRegex(PlannerError,'failure_context'):
                    client.plan('clear the table',cubes(),Path(temp)/'failure.json',{'category':'arbitrary'})
        finally:
            server.shutdown();server.server_close();thread.join(timeout=2)


if __name__ == '__main__':
    unittest.main()
