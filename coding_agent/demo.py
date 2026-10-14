"""Offline reproducible demonstration. Model is scripted; Bash and files are real."""
import argparse
import json
import shutil
import uuid
from pathlib import Path
from .model import ModelResponse, ScriptedModel
from .runtime import AgentRuntime, RunPaused
from .storage import Storage
from .journal import unified_diff, rollback


FIX = 'def calculate_total(prices: list[float], discount_percent: float) -> float:\n    if not 0 <= discount_percent <= 100:\n        raise ValueError("Invalid discount")\n    return sum(prices) * (1 - discount_percent / 100)\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='.demo-runs')
    args = parser.parse_args()
    source = Path(__file__).resolve().parents[1]
    root = Path(args.output).resolve() / uuid.uuid4().hex[:12]
    workspace = root / 'workspace'
    workspace.mkdir(parents=True)
    for name in ('order.py', 'test_order.py'):
        shutil.copyfile(source / 'workspace' / name, workspace / name)
    shutil.copyfile(source / 'policy.json', root / 'policy.json')
    original = (workspace / 'order.py').read_bytes()
    test_before = (workspace / 'test_order.py').read_bytes()
    plan = {'goal': 'Fix order total', 'steps': [
        {'id': 'inspect', 'description': 'Read order.py', 'tools': ['read_file'], 'dependencies': [], 'verification': {'kind': 'file_read', 'path': 'order.py'}},
        {'id': 'fix', 'description': 'Write order.py', 'tools': ['write_file'], 'dependencies': ['inspect'], 'verification': {'kind': 'file_written', 'path': 'order.py'}},
        {'id': 'test', 'description': 'Run original tests through Bash', 'tools': ['Bash'], 'dependencies': ['fix'], 'verification': {'kind': 'tests_pass', 'command': 'python3 -m unittest -v', 'cwd': '.'}},
    ], 'risks': ['Never modify tests'], 'assumptions': ['Discount is a percentage'], 'done_when': ['Original tests pass']}
    def call(cid, name, **arguments):
        return ModelResponse(None, [{'id': cid, 'name': name, 'arguments': arguments}])
    model = ScriptedModel([
        call('planning-read', 'read_file', path='test_order.py'),
        ModelResponse(json.dumps(plan), []),
        call('read', 'read_file', path='order.py'),
        call('write', 'write_file', path='order.py', content=FIX),
        call('test', 'Bash', command='python3 -m unittest -v', cwd='.'),
        ModelResponse('Demo task complete.', []),
    ])
    database = root / 'state.sqlite3'
    runtime = AgentRuntime(Storage(database), model)
    rid = runtime.create_run('Fix order total, do not change tests', workspace, root / 'policy.json', mode='safe', plan=True)
    print('DEMO: scripted model; real files, real Bash, real tests.')
    print('Artifacts:', root)
    print('run_id:', rid)
    while True:
        try:
            print(runtime.run_until_pause(rid))
            break
        except RunPaused as pause:
            print('PAUSE:', pause.reason)
            print(runtime.task_checklist(rid))
            if pause.reason == 'plan_approval':
                assert (workspace / 'order.py').read_bytes() == original
                print('Demo driver explicitly approves the saved plan.')
                runtime.approve_plan(rid)
            elif pause.reason == 'action_approval':
                for cid in pause.call_ids:
                    print('Demo driver explicitly approves action:', cid)
                    runtime.approve_action(rid, cid)
            else:
                raise
            # Reopen persistent storage on every pause, without repeating completed actions.
            runtime = AgentRuntime(Storage(database), model)
    result = runtime.storage.get_call(rid, 'test')['result']
    if result.get('exit_code') != 0:
        raise RuntimeError('Actual Bash tests failed: ' + str(result))
    print(result.get('stderr', ''))
    print(runtime.task_checklist(rid))
    print(unified_diff(runtime.storage, rid))
    assert (workspace / 'test_order.py').read_bytes() == test_before
    print('Budget:', runtime.storage.get_run(rid)['used'])
    rollback(Storage(database), rid)
    assert (workspace / 'order.py').read_bytes() == original
    print('ROLLBACK after reopening database: original order.py restored; tests unchanged.')


if __name__ == '__main__':
    main()
