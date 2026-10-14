"""Regression tests exercise observable invariants, real Bash and process restart."""
from __future__ import annotations
import copy
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch
from coding_agent.cli import main
from coding_agent.journal import rollback, unified_diff, WorkspaceConflict, snapshot
from coding_agent.model import ModelResponse, ScriptedModel, OpenAICompatibleModel
from coding_agent.planning import validate_plan
from coding_agent.runtime import AgentRuntime, RunPaused, SimulatedCrash
from coding_agent.security import Policy, PolicyError, resolve_workspace_path
from coding_agent.storage import Storage, StorageError, BudgetExceeded
from coding_agent.tools import ToolRegistry, bounded, size

PROJECT = Path(__file__).resolve().parents[1]
FIX = 'def calculate_total(prices: list[float], discount_percent: float) -> float:\n    if not 0 <= discount_percent <= 100:\n        raise ValueError("Invalid discount")\n    return sum(prices) * (1 - discount_percent / 100)\n'


def response(cid, name, **args):
    return ModelResponse(None, [{'id': cid, 'name': name, 'arguments': args}])


def plan():
    return {'goal': 'Fix order calculation', 'steps': [
        {'id': 'read', 'description': 'Read implementation', 'tools': ['read_file'], 'dependencies': [], 'verification': {'kind': 'file_read', 'path': 'order.py'}},
        {'id': 'fix', 'description': 'Write corrected implementation', 'tools': ['write_file'], 'dependencies': ['read'], 'verification': {'kind': 'file_written', 'path': 'order.py'}},
        {'id': 'test', 'description': 'Verify real tests', 'tools': ['Bash'], 'dependencies': ['fix'], 'verification': {'kind': 'tests_pass', 'command': 'python3 -m unittest -v', 'cwd': '.'}},
    ], 'risks': ['Do not change tests'], 'assumptions': ['Discount is a percentage'], 'done_when': ['Original tests pass']}


class AgentTests(unittest.TestCase):
    def setUp(self):
        parent = Path.cwd() / '.test-tmp'
        parent.mkdir(exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=parent)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.ws = self.root / 'workspace'
        self.ws.mkdir()
        for name in ('order.py', 'test_order.py'):
            (self.ws / name).write_bytes((PROJECT / 'workspace' / name).read_bytes())
        self.original = (self.ws / 'order.py').read_bytes()
        self.tests = (self.ws / 'test_order.py').read_bytes()
        self.config = json.loads((PROJECT / 'policy.json').read_text())
        self.config['bash']['max_output_bytes'] = 2048
        self.config['budgets']['tool_output_bytes'] = 131072
        self.policy_file = self.root / 'policy.json'
        self.save_policy()
        self.storage = Storage(self.root / 'state.sqlite3')

    def save_policy(self):
        self.policy_file.write_text(json.dumps(self.config), encoding='utf-8')

    def make(self, responses=(), *, mode='auto', planning=False, hook=None):
        runtime = AgentRuntime(self.storage, ScriptedModel(list(responses)), hook)
        rid = runtime.create_run('Fix calculation, do not change tests', self.ws, self.policy_file, mode=mode, plan=planning)
        return runtime, rid

    def resume(self, responses=(), hook=None):
        return AgentRuntime(Storage(self.storage.path), ScriptedModel(list(responses)), hook)

    def policy(self):
        return Policy.load(self.policy_file)

    def crash(self, point):
        def hook(actual):
            if point == actual:
                raise SimulatedCrash(point)
        return hook

    def test_real_bash_fixes_order_and_original_tests_pass(self):
        runtime, rid = self.make([
            ModelResponse(None, [response('r1', 'read_file', path='order.py').tool_calls[0], response('r2', 'read_file', path='test_order.py').tool_calls[0]]),
            response('w', 'write_file', path='order.py', content=FIX),
            response('b', 'Bash', command='python3 -m unittest -v', cwd='.'),
            ModelResponse('Tests passed', []),
        ])
        self.assertEqual(runtime.run_until_pause(rid), 'Tests passed')
        result = self.storage.get_call(rid, 'b')['result']
        self.assertEqual(result.get('exit_code'), 0, result)
        self.assertIn('Ran 2 tests', result['stderr'])
        self.assertEqual((self.ws / 'test_order.py').read_bytes(), self.tests)
        self.assertEqual([m['tool_call_id'] for m in self.storage.messages(rid) if m['role'] == 'tool'], ['r1', 'r2', 'w', 'b'])

    def test_real_bash_returns_nonzero_for_original_bug(self):
        result = ToolRegistry(self.ws, self.policy()).execute('Bash', {'command': 'python3 -m unittest -v', 'cwd': '.'}, 2048)
        self.assertNotEqual(result['exit_code'], 0)
        self.assertIn('FAILED', result['stderr'])

    def test_real_bash_timeout(self):
        (self.ws / 'test_slow.py').write_text('import time\ntime.sleep(10)\n')
        self.config['bash']['timeout_seconds'] = 1
        self.save_policy()
        result = ToolRegistry(self.ws, self.policy()).execute('Bash', {'command': 'python3 -m unittest -v', 'cwd': '.'}, 2048)
        self.assertTrue(result['timed_out'])
        self.assertFalse(result['ok'])

    def test_real_bash_large_output_is_bounded(self):
        (self.ws / 'test_loud.py').write_text('print("x" * 200000)\n')
        result = ToolRegistry(self.ws, self.policy()).execute('Bash', {'command': 'python3 -m unittest -v', 'cwd': '.'}, 1024)
        self.assertTrue(result['truncated'])
        self.assertLessEqual(len(result['stdout'].encode()) + len(result['stderr'].encode()), 1024)
        final = bounded(result, 512)
        self.assertLessEqual(size(final), 512)
        self.assertIn('exit_code', final)
        self.assertIn('stdout', final)
        self.assertIn('stderr', final)

    def test_validation_error_is_tool_result_and_loop_continues(self):
        runtime, rid = self.make([response('bad', 'not_a_tool'), response('bad2', 'read_file', path=42), response('missing', 'read_file', path='missing.py'), response('ok', 'read_file', path='order.py'), ModelResponse('done', [])])
        self.assertEqual(runtime.run_until_pause(rid), 'done')
        for cid in ('bad', 'bad2', 'missing'):
            self.assertFalse(self.storage.get_call(rid, cid)['result']['ok'])
        self.assertTrue(self.storage.get_call(rid, 'ok')['result']['ok'])

    def test_safe_write_approval_persists_across_new_runtime(self):
        runtime, rid = self.make([response('w', 'write_file', path='order.py', content=FIX)], mode='safe')
        with self.assertRaises(RunPaused):
            runtime.run_until_pause(rid)
        self.assertEqual((self.ws / 'order.py').read_bytes(), self.original)
        second = self.resume([ModelResponse('done', [])])
        second.approve_action(rid, 'w')
        second.run_until_pause(rid)
        self.assertEqual((self.ws / 'order.py').read_text(), FIX)

    def test_rejection_reaches_model(self):
        runtime, rid = self.make([response('w', 'write_file', path='order.py', content=FIX)], mode='safe')
        with self.assertRaises(RunPaused):
            runtime.run_until_pause(rid)
        second = self.resume([ModelResponse('Understood', [])])
        second.reject_action(rid, 'w')
        second.run_until_pause(rid)
        self.assertIn('user_rejected', second.model.requests[0][0][-1]['content'])
        self.assertEqual((self.ws / 'order.py').read_bytes(), self.original)

    def test_delete_always_requires_approval(self):
        runtime, rid = self.make([response('d', 'delete_file', path='order.py')])
        with self.assertRaises(RunPaused):
            runtime.run_until_pause(rid)
        self.assertEqual(self.storage.get_call(rid, 'd')['status'], 'awaiting_approval')

    def test_policy_is_rechecked_after_approval(self):
        runtime, rid = self.make([response('w', 'write_file', path='order.py', content=FIX)], mode='safe')
        with self.assertRaises(RunPaused):
            runtime.run_until_pause(rid)
        self.config['protected_files'].append('order.py')
        self.save_policy()
        runtime.approve_action(rid, 'w')
        second = self.resume([ModelResponse('denied', [])])
        second.run_until_pause(rid)
        self.assertFalse(self.storage.get_call(rid, 'w')['result']['ok'])
        self.assertEqual((self.ws / 'order.py').read_bytes(), self.original)

    def test_paths_and_allow_deny_precedence(self):
        for path in ('../policy.json', str(self.policy_file), 'private/x', 'C:/outside.txt', 'a/../../x'):
            with self.subTest(path=path), self.assertRaises(PolicyError):
                resolve_workspace_path(self.ws, path, self.policy())
        self.assertEqual(resolve_workspace_path(self.ws, 'order.py', self.policy()), self.ws / 'order.py')

    def test_symlink_escape(self):
        outside = self.root / 'outside'
        outside.mkdir()
        try:
            (self.ws / 'link').symlink_to(outside, target_is_directory=True)
        except OSError:
            self.skipTest('Host does not allow symlinks; Ubuntu suite is required')
        with self.assertRaises(PolicyError):
            resolve_workspace_path(self.ws, 'link/secret', self.policy())

    def test_bash_cannot_follow_test_symlink_outside_workspace(self):
        outside = self.root / 'test_out.py'
        outside.write_text('raise RuntimeError("must not execute")')
        try:
            (self.ws / 'test_out.py').symlink_to(outside)
        except OSError:
            self.skipTest('Host does not allow symlinks')
        with self.assertRaises(PolicyError):
            ToolRegistry(self.ws, self.policy()).validate('Bash', {'command': 'python3 -m unittest -v', 'cwd': '.'})

    def test_temporary_write_symlink_cannot_overwrite_outside(self):
        victim = self.root / 'victim'
        victim.write_text('secret')
        try:
            (self.ws / '.order.py.agent-tmp').symlink_to(victim)
        except OSError:
            self.skipTest('Host does not allow symlinks')
        ToolRegistry(self.ws, self.policy()).execute('write_file', {'path': 'order.py', 'content': FIX}, 1024)
        self.assertEqual(victim.read_text(), 'secret')

    def test_list_files_does_not_reveal_denied_directory(self):
        (self.ws / 'private').mkdir()
        (self.ws / 'private' / 'secret').write_text('hidden')
        result = ToolRegistry(self.ws, self.policy()).execute('list_files', {'path': '.'}, 2048)
        self.assertNotIn('private/secret', result['files'])

    def test_command_grammar_and_denied_cwd(self):
        self.policy().validate_command('python3 -m unittest -v')
        for command in ('rm -rf .', 'python3 -m unittest -v && echo hi', 'bash -c true', 'python3 -c pass', 'python3 -m unittest -v > a', 'echo ${HOME}', 'python3 -m unittest -v # comment'):
            self.config['commands']['allow'].append(command)
            self.save_policy()
            with self.subTest(command=command), self.assertRaises(PolicyError):
                self.policy().validate_command(command)
        (self.ws / 'private').mkdir()
        with self.assertRaises(PolicyError):
            ToolRegistry(self.ws, self.policy()).validate('Bash', {'command': 'python3 -m unittest -v', 'cwd': 'private'})

    def test_policy_and_state_cannot_live_inside_workspace(self):
        local = self.ws / 'policy.json'
        local.write_text(self.policy_file.read_text())
        with self.assertRaises(PolicyError):
            AgentRuntime(self.storage).create_run('x', self.ws, local)
        with self.assertRaises(ValueError):
            AgentRuntime(Storage(self.ws / 'state.db')).create_run('x', self.ws, self.policy_file)

    def test_policy_rejects_escaping_directories_and_broad_generated_patterns(self):
        self.config['directories']['allow'] = ['../']
        self.save_policy()
        with self.assertRaises(PolicyError):
            self.policy()
        self.config['directories']['allow'] = ['.']
        self.config['generated'] = ['*.py']
        self.save_policy()
        with self.assertRaises(PolicyError):
            self.policy()

    def test_protected_test_write_is_denied(self):
        runtime, rid = self.make([response('w', 'write_file', path='test_order.py', content=''), ModelResponse('done', [])])
        runtime.run_until_pause(rid)
        self.assertEqual((self.ws / 'test_order.py').read_bytes(), self.tests)
        self.assertFalse(self.storage.get_call(rid, 'w')['result']['ok'])

    def test_two_writes_in_one_batch_do_not_conflict_with_each_other(self):
        runtime, rid = self.make([ModelResponse(None, [response('a', 'write_file', path='a.txt', content='a').tool_calls[0], response('b', 'write_file', path='b.txt', content='b').tool_calls[0]]), ModelResponse('done', [])])
        runtime.run_until_pause(rid)
        self.assertEqual((self.ws / 'a.txt').read_text(), 'a')
        self.assertEqual((self.ws / 'b.txt').read_text(), 'b')

    def test_plan_forbids_bash_and_writes_even_in_auto(self):
        runtime, rid = self.make([response('w', 'write_file', path='order.py', content=FIX), response('b', 'Bash', command='python3 -m unittest -v', cwd='.'), ModelResponse(json.dumps(plan()), [])], planning=True)
        with patch.object(ToolRegistry, 'execute') as execute, self.assertRaises(RunPaused):
            runtime.run_until_pause(rid)
        execute.assert_not_called()
        self.assertEqual((self.ws / 'order.py').read_bytes(), self.original)
        self.assertEqual(self.storage.get_run(rid)['used']['side_effects'], 0)

    def test_three_tasks_actual_evidence_restart_and_checklist(self):
        runtime, rid = self.make([ModelResponse(json.dumps(plan()), [])], planning=True, mode='safe')
        with self.assertRaises(RunPaused):
            runtime.run_until_pause(rid)
        runtime.approve_plan(rid)
        second = self.resume([response('read', 'read_file', path='order.py'), response('write', 'write_file', path='order.py', content=FIX)])
        with self.assertRaises(RunPaused):
            second.run_until_pause(rid)
        self.assertEqual([t['status'] for t in self.storage.tasks(rid)], ['completed', 'running', 'pending'])
        second.approve_action(rid, 'write')
        third = self.resume([response('test', 'Bash', command='python3 -m unittest -v', cwd='.')])
        with self.assertRaises(RunPaused):
            third.run_until_pause(rid)
        third.approve_action(rid, 'test')
        fourth = self.resume([ModelResponse('done', [])])
        fourth.run_until_pause(rid)
        tasks = self.storage.tasks(rid)
        self.assertTrue(all(t['status'] == 'completed' and t['evidence'] for t in tasks))
        self.assertEqual(tasks[2]['evidence']['exit_code'], 0)
        self.assertEqual(fourth.task_checklist(rid).count('[x]'), 3)
        self.assertEqual(len(self.storage.get_call(rid, 'read')['attempts']), 1)

    def test_wrong_file_does_not_complete_task_and_later_tool_is_denied(self):
        runtime, rid = self.make([ModelResponse(json.dumps(plan()), [])], planning=True)
        with self.assertRaises(RunPaused):
            runtime.run_until_pause(rid)
        runtime.approve_plan(rid)
        runtime.model = ScriptedModel([response('wrong', 'read_file', path='test_order.py'), response('early', 'write_file', path='order.py', content=FIX)])
        with self.assertRaises(RuntimeError):  # Scripted model exhausted, not successful completion.
            runtime.run_until_pause(rid)
        self.assertNotEqual(self.storage.tasks(rid)[0]['status'], 'completed')
        self.assertFalse(self.storage.get_call(rid, 'early')['result']['ok'])
        self.assertEqual((self.ws / 'order.py').read_bytes(), self.original)

    def test_premature_final_cannot_finish_plan(self):
        runtime, rid = self.make([ModelResponse(json.dumps(plan()), [])], planning=True)
        with self.assertRaises(RunPaused):
            runtime.run_until_pause(rid)
        runtime.approve_plan(rid)
        runtime.model = ScriptedModel([ModelResponse('done', [])])
        with self.assertRaises(RuntimeError):
            runtime.run_until_pause(rid)
        self.assertNotEqual(self.storage.get_run(rid)['status'], 'completed')

    def test_invalid_dependency_graphs_and_verification_rejected(self):
        for case in ('cycle', 'unknown', 'duplicate', 'bad_rule'):
            value = plan()
            if case == 'cycle':
                value['steps'][0]['dependencies'] = ['test']
            elif case == 'unknown':
                value['steps'][0]['dependencies'] = ['missing']
            elif case == 'duplicate':
                value['steps'][1]['id'] = 'read'
            else:
                value['steps'][0]['verification'] = {'kind': 'model_said_done'}
            with self.subTest(case=case), self.assertRaises(ValueError):
                validate_plan(value, ToolRegistry(self.ws, self.policy()))

    def test_model_budget_before_request(self):
        self.config['budgets']['model_requests'] = 0
        self.save_policy()
        runtime, rid = self.make([ModelResponse('unused', [])])
        with self.assertRaises(BudgetExceeded):
            runtime.run_until_pause(rid)
        self.assertFalse(runtime.model.requests)

    def test_tool_and_side_effect_budget_reject_entire_batch(self):
        for key in ('tool_calls', 'side_effects'):
            with self.subTest(key=key):
                self.config['budgets'][key] = 1
                self.save_policy()
                runtime, rid = self.make([ModelResponse(None, [response('a', 'write_file', path='a', content='a').tool_calls[0], response('b', 'write_file', path='b', content='b').tool_calls[0]])])
                with self.assertRaises(BudgetExceeded):
                    runtime.run_until_pause(rid)
                self.assertFalse((self.ws / 'a').exists())
                self.assertEqual(self.storage.get_run(rid)['calls'], {})
                self.assertEqual(self.storage.get_run(rid)['used'][key], 0)
                self.config['budgets'][key] = 60

    def test_output_zero_prevents_side_effect(self):
        self.config['budgets']['tool_output_bytes'] = 0
        self.save_policy()
        runtime, rid = self.make([response('w', 'write_file', path='order.py', content=FIX)])
        with self.assertRaises(BudgetExceeded):
            runtime.run_until_pause(rid)
        self.assertEqual((self.ws / 'order.py').read_bytes(), self.original)

    def test_output_bound_including_unicode_json_escaping_and_envelope(self):
        for limit in (256, 257, 512, 1000):
            output = bounded({'ok': True, 'content': '"\n\\Привет🙂' * 1000}, limit)
            self.assertLessEqual(size(output), limit)
            self.assertTrue(output['truncated'])

    def test_output_budget_exactly_exhausted_and_survives_resume(self):
        self.config['budgets']['tool_output_bytes'] = 256
        self.save_policy()
        (self.ws / 'large').write_text('x' * 10000)
        runtime, rid = self.make([response('r', 'read_file', path='large'), response('w', 'write_file', path='order.py', content=FIX)])
        with self.assertRaises(BudgetExceeded):
            runtime.run_until_pause(rid)
        saved = Storage(self.storage.path).get_run(rid)
        self.assertLessEqual(saved['used']['tool_output_bytes'], 256)
        self.assertTrue(saved['calls']['r']['result']['truncated'])
        self.assertEqual((self.ws / 'order.py').read_bytes(), self.original)

    def test_atomic_batch_crash_rolls_back_counters_calls_and_message(self):
        runtime, rid = self.make([response('w', 'write_file', path='order.py', content=FIX)], hook=self.crash('inside_batch_transaction'))
        with self.assertRaises(SimulatedCrash):
            runtime.run_until_pause(rid)
        saved = self.storage.get_run(rid)
        self.assertEqual(saved['calls'], {})
        self.assertEqual(saved['used']['tool_calls'], 0)
        self.assertEqual(len(saved['messages']), 2)
        self.assertEqual(saved['used']['model_requests'], 1)

    def test_crash_after_batch_debit_keeps_saved_action(self):
        runtime, rid = self.make([response('w', 'write_file', path='order.py', content=FIX)], hook=self.crash('after_batch_commit'))
        with self.assertRaises(SimulatedCrash):
            runtime.run_until_pause(rid)
        saved = self.storage.get_run(rid)
        self.assertEqual(saved['used']['side_effects'], 1)
        self.assertEqual(saved['calls']['w']['status'], 'ready')
        second = self.resume([ModelResponse('done', [])])
        second.run_until_pause(rid)
        self.assertEqual(self.storage.get_run(rid)['used']['side_effects'], 1)

    def test_model_debit_is_not_refunded_after_crash(self):
        runtime, rid = self.make([], hook=self.crash('after_model_debit'))
        with self.assertRaises(SimulatedCrash):
            runtime.run_until_pause(rid)
        self.resume([ModelResponse('done', [])]).run_until_pause(rid)
        self.assertEqual(self.storage.get_run(rid)['used']['model_requests'], 2)

    def crashed_write(self, point='after_tool_execution'):
        runtime, rid = self.make([response('w', 'write_file', path='order.py', content=FIX)], hook=self.crash(point))
        with self.assertRaises(SimulatedCrash):
            runtime.run_until_pause(rid)
        second = self.resume()
        with self.assertRaises(RunPaused):
            second.run_until_pause(rid)
        return second, rid

    def test_uncertain_write_never_repeats_silently(self):
        runtime, rid = self.crashed_write()
        self.assertEqual(self.storage.get_call(rid, 'w')['status'], 'uncertain')
        self.assertEqual((self.ws / 'order.py').read_text(), FIX)
        self.assertIn('order.py', runtime.recovery_info(rid, 'w')['changed_paths'])
        with self.assertRaises(StorageError):
            runtime.resolve_uncertain(rid, 'w', 'retry')
        with self.assertRaises(StorageError):
            rollback(self.storage, rid)

    def test_retry_is_budgeted_attempt_same_wire_id(self):
        runtime, rid = self.crashed_write()
        runtime.resolve_uncertain(rid, 'w', 'retry', attribution='accept')
        second = self.resume([ModelResponse('done', [])])
        second.run_until_pause(rid)
        saved = self.storage.get_run(rid)
        self.assertEqual(saved['used']['side_effects'], 2)
        self.assertEqual(saved['calls']['w']['attempt'], 2)
        self.assertEqual([m['tool_call_id'] for m in saved['messages'] if m['role'] == 'tool'], ['w'])
        self.assertGreaterEqual(saved['used']['tool_output_bytes'], 2048)

    def test_retry_cannot_bypass_exhausted_side_effect_budget(self):
        self.config['budgets']['side_effects'] = 1
        self.save_policy()
        runtime, rid = self.crashed_write()
        with self.assertRaises(BudgetExceeded):
            runtime.resolve_uncertain(rid, 'w', 'retry', attribution='accept')
        self.assertEqual(self.storage.get_call(rid, 'w')['attempt'], 1)
        self.assertEqual(self.storage.get_call(rid, 'w')['status'], 'uncertain')

    def test_result_message_budget_commit_is_atomic(self):
        runtime, rid = self.crashed_write('inside_result_transaction')
        saved = self.storage.get_run(rid)
        self.assertFalse(any(m['role'] == 'tool' for m in saved['messages']))
        self.assertEqual(saved['used']['tool_output_bytes'], 2048)
        self.assertEqual(saved['files'], {})

    def test_completed_action_recovery_does_not_repeat(self):
        runtime, rid = self.make([response('w', 'write_file', path='order.py', content=FIX)], hook=self.crash('after_result_commit'))
        with self.assertRaises(SimulatedCrash):
            runtime.run_until_pause(rid)
        self.resume([ModelResponse('done', [])]).run_until_pause(rid)
        self.assertEqual(len(self.storage.get_call(rid, 'w')['attempts']), 1)
        self.assertEqual(self.storage.get_run(rid)['used']['side_effects'], 1)

    def test_duplicate_completed_call_returns_cached_result(self):
        r = response('w', 'write_file', path='order.py', content=FIX)
        runtime, rid = self.make([r, r, ModelResponse('done', [])])
        with patch.object(ToolRegistry, 'execute', autospec=True, wraps=None) as execute:
            execute.return_value = {'ok': True}
            runtime.run_until_pause(rid)
            self.assertEqual(execute.call_count, 1)
        self.assertEqual(self.storage.get_run(rid)['used']['side_effects'], 1)

    def test_changed_duplicate_is_protocol_error(self):
        runtime, rid = self.make([response('r', 'read_file', path='order.py'), response('r', 'read_file', path='test_order.py')])
        with self.assertRaises(StorageError):
            runtime.run_until_pause(rid)
        self.assertEqual(self.storage.get_run(rid)['status'], 'protocol_error')

    def test_same_batch_duplicate_id_does_not_execute_anything(self):
        call = response('w', 'write_file', path='order.py', content=FIX).tool_calls[0]
        runtime, rid = self.make([ModelResponse(None, [call, call])])
        with self.assertRaises(StorageError):
            runtime.run_until_pause(rid)
        self.assertEqual((self.ws / 'order.py').read_bytes(), self.original)

    def test_bash_changes_are_journaled_and_rollbackable(self):
        (self.ws / 'test_effect.py').write_text('from pathlib import Path\nPath("side.txt").write_text("side effect")\n')
        runtime, rid = self.make([response('b', 'Bash', command='python3 -m unittest -v', cwd='.'), ModelResponse('done', [])])
        runtime.run_until_pause(rid)
        self.assertTrue((self.ws / 'side.txt').exists())
        self.assertIn('side.txt', unified_diff(self.storage, rid))
        rollback(self.storage, rid)
        self.assertFalse((self.ws / 'side.txt').exists())

    def test_uncertain_bash_must_not_be_retried(self):
        runtime, rid = self.make([response('b', 'Bash', command='python3 -m unittest -v', cwd='.')], hook=self.crash('after_tool_execution'))
        with self.assertRaises(SimulatedCrash):
            runtime.run_until_pause(rid)
        with self.assertRaises(StorageError):
            self.resume().resolve_uncertain(rid, 'b', 'retry')

    def test_plan_rejection_never_executes_tasks(self):
        runtime, rid = self.make([ModelResponse(json.dumps(plan()), [])], planning=True)
        with self.assertRaises(RunPaused):
            runtime.run_until_pause(rid)
        runtime.reject_plan(rid)
        with self.assertRaises(StorageError):
            self.resume().run_until_pause(rid)
        self.assertEqual(self.storage.get_run(rid)['used']['side_effects'], 0)

    def test_budget_is_shared_between_planning_and_execution(self):
        runtime, rid = self.make([response('p', 'read_file', path='order.py'), ModelResponse(json.dumps(plan()), [])], planning=True)
        with self.assertRaises(RunPaused):
            runtime.run_until_pause(rid)
        before = self.storage.get_run(rid)['used'].copy()
        runtime.approve_plan(rid)
        self.assertEqual(self.storage.get_run(rid)['used'], before)
        second = self.resume([response('r', 'read_file', path='order.py')])
        with self.assertRaises(RuntimeError):
            second.run_until_pause(rid)
        self.assertEqual(self.storage.get_run(rid)['used']['tool_calls'], before['tool_calls'] + 1)

    def test_recovery_choice_atomic_when_process_crashes(self):
        runtime, rid = self.crashed_write()
        before = self.storage.get_run(rid)
        runtime.crash_hook = self.crash('inside_recovery_transaction')
        with self.assertRaises(SimulatedCrash):
            runtime.resolve_uncertain(rid, 'w', 'retry', attribution='accept')
        after = self.storage.get_run(rid)
        self.assertEqual(after['used'], before['used'])
        self.assertEqual(after['calls']['w']['attempt'], 1)
        self.assertEqual(after['files'], {})

    def test_uncertain_fail_accept_changes_then_rollback(self):
        runtime, rid = self.crashed_write()
        runtime.resolve_uncertain(rid, 'w', 'fail', attribution='accept')
        self.resume([ModelResponse('done', [])]).run_until_pause(rid)
        self.assertIn('order.py', unified_diff(self.storage, rid))
        rollback(Storage(self.storage.path), rid)
        self.assertEqual((self.ws / 'order.py').read_bytes(), self.original)

    def test_uncertain_abort_exclude_keeps_changes_unowned(self):
        runtime, rid = self.crashed_write()
        runtime.resolve_uncertain(rid, 'w', 'abort', attribution='exclude')
        rollback(self.storage, rid)
        self.assertEqual((self.ws / 'order.py').read_text(), FIX)

    def test_delete_uncertain_cannot_retry(self):
        runtime, rid = self.make([response('d', 'delete_file', path='order.py')])
        with self.assertRaises(RunPaused):
            runtime.run_until_pause(rid)
        runtime.approve_action(rid, 'd')
        runtime.crash_hook = self.crash('after_tool_execution')
        with self.assertRaises(SimulatedCrash):
            runtime.run_until_pause(rid)
        with self.assertRaises(StorageError):
            self.resume().resolve_uncertain(rid, 'd', 'retry', attribution='accept')

    def test_external_edit_stops_next_action(self):
        runtime, rid = self.make([response('w', 'write_file', path='order.py', content=FIX)], mode='safe')
        with self.assertRaises(RunPaused):
            runtime.run_until_pause(rid)
        (self.ws / 'order.py').write_text('external')
        runtime.approve_action(rid, 'w')
        with self.assertRaises(WorkspaceConflict):
            self.resume().run_until_pause(rid)
        self.assertEqual((self.ws / 'order.py').read_text(), 'external')

    def test_diff_and_rollback_after_new_storage_restore_only_owned_files(self):
        runtime, rid = self.make([response('w', 'write_file', path='order.py', content=FIX), response('new', 'write_file', path='new.txt', content='new'), ModelResponse('done', [])])
        runtime.run_until_pause(rid)
        (self.ws / 'unrelated').write_text('keep')
        self.assertIn('order.py', unified_diff(self.storage, rid))
        rollback(Storage(self.storage.path), rid)
        self.assertEqual((self.ws / 'order.py').read_bytes(), self.original)
        self.assertFalse((self.ws / 'new.txt').exists())
        self.assertEqual((self.ws / 'unrelated').read_text(), 'keep')
        self.assertEqual(rollback(self.storage, rid), [])

    def test_binary_delete_restored_byte_exactly(self):
        raw = bytes(range(256))
        (self.ws / 'binary.bin').write_bytes(raw)
        runtime, rid = self.make([response('d', 'delete_file', path='binary.bin')])
        with self.assertRaises(RunPaused):
            runtime.run_until_pause(rid)
        runtime.approve_action(rid, 'd')
        self.resume([ModelResponse('done', [])]).run_until_pause(rid)
        rollback(self.storage, rid)
        self.assertEqual((self.ws / 'binary.bin').read_bytes(), raw)

    def test_rollback_conflict_preflights_every_file(self):
        runtime, rid = self.make([response('a', 'write_file', path='a.txt', content='a'), response('z', 'write_file', path='z.txt', content='z'), ModelResponse('done', [])])
        runtime.run_until_pause(rid)
        (self.ws / 'z.txt').write_text('external')
        with self.assertRaises(WorkspaceConflict):
            rollback(self.storage, rid)
        self.assertEqual((self.ws / 'a.txt').read_text(), 'a')
        self.assertEqual((self.ws / 'z.txt').read_text(), 'external')

    def test_rollback_policy_recheck(self):
        runtime, rid = self.make([response('w', 'write_file', path='order.py', content=FIX), ModelResponse('done', [])])
        runtime.run_until_pause(rid)
        self.config['protected_files'].append('order.py')
        self.save_policy()
        with self.assertRaises(PolicyError):
            rollback(self.storage, rid)
        self.assertEqual((self.ws / 'order.py').read_text(), FIX)

    def test_rollback_interrupted_can_resume(self):
        runtime, rid = self.make([response('a', 'write_file', path='a.txt', content='a'), response('b', 'write_file', path='b.txt', content='b'), ModelResponse('done', [])])
        runtime.run_until_pause(rid)
        with self.assertRaises(SimulatedCrash):
            rollback(self.storage, rid, self.crash('during_rollback'))
        rollback(Storage(self.storage.path), rid)
        self.assertFalse((self.ws / 'a.txt').exists())
        self.assertFalse((self.ws / 'b.txt').exists())

    def test_cli_decisions_need_no_model_credentials(self):
        runtime, rid = self.make([response('w', 'write_file', path='order.py', content=FIX)], mode='safe')
        with self.assertRaises(RunPaused):
            runtime.run_until_pause(rid)
        with patch.dict(os.environ, {}, clear=True), redirect_stdout(io.StringIO()):
            self.assertEqual(main(['--state', str(self.storage.path), 'action', 'approve', rid, 'w']), 0)

    def test_os_lock_rejects_concurrent_runtime(self):
        with self.storage.exclusive(), self.assertRaises(StorageError):
            with Storage(self.storage.path).exclusive():
                pass

    def test_real_process_kill_after_write_restores_uncertain(self):
        script = '''import os,sys
from pathlib import Path
from coding_agent.storage import Storage
from coding_agent.runtime import AgentRuntime
from coding_agent.model import ScriptedModel,ModelResponse
root=Path(sys.argv[1])
def hook(point):
    if point=='after_tool_execution': os._exit(77)
r=AgentRuntime(Storage(root/'child.sqlite3'),ScriptedModel([ModelResponse(None,[{'id':'w','name':'write_file','arguments':{'path':'order.py','content':'changed'}}])]),hook)
rid=r.create_run('fix',root/'workspace',root/'policy.json',mode='auto')
print(rid,flush=True)
r.run_until_pause(rid)
'''
        env = os.environ.copy()
        env['PYTHONPATH'] = str(PROJECT)
        process = subprocess.run([sys.executable, '-B', '-c', script, str(self.root)], env=env, capture_output=True, text=True, timeout=20)
        self.assertEqual(process.returncode, 77, process.stderr)
        rid = process.stdout.strip()
        runtime = AgentRuntime(Storage(self.root / 'child.sqlite3'))
        with self.assertRaises(RunPaused):
            runtime.run_until_pause(rid)
        self.assertEqual(runtime.storage.get_call(rid, 'w')['status'], 'uncertain')


if __name__ == '__main__':
    unittest.main()
