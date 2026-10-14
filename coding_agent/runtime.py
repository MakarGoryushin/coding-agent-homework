"""Explicit state machine; the model proposes actions, this runtime authorizes them."""
from __future__ import annotations
import copy
import json
import uuid
from pathlib import Path
from .journal import snapshot, assert_expected, record_changes, changed, diff_trees
from .planning import PLAN_PROMPT, validate_plan, current_task, verify
from .security import Policy, BUDGET_KEYS
from .storage import StorageError, BudgetExceeded
from .tools import ToolRegistry, bounded, dumps, size

SYSTEM_PROMPT = '''You are a coding agent. Inspect files, make necessary changes and run tests using Bash.
Never change tests. Tools are proposals validated by the runtime. Follow the current task and its verification rule exactly.
Errors are tool results; adapt your next action. Do not claim tests passed without a successful Bash result.
Finish with a concise factual text. When a plan exists, complete every task before finishing.'''
MIN_OUTPUT = 256


class RunPaused(RuntimeError):
    def __init__(self, reason, run_id, call_ids=None):
        super().__init__(reason)
        self.reason, self.run_id, self.call_ids = reason, run_id, call_ids or []


class SimulatedCrash(BaseException):
    """Bypasses ordinary tool error handling; used only by deterministic failure tests."""


def debit(state, **amounts):
    for key, amount in amounts.items():
        if state['used'][key] + amount > state['budget'][key]:
            raise BudgetExceeded(key + ' budget exhausted; action did not start')
    for key, amount in amounts.items():
        state['used'][key] += amount


def error(kind, message):
    return {'ok': False, 'error': {'type': kind, 'message': str(message)}}


class AgentRuntime:
    def __init__(self, storage, model=None, crash_hook=None):
        self.storage, self.model, self.crash_hook = storage, model, crash_hook

    def hit(self, point):
        if self.crash_hook:
            self.crash_hook(point)

    def create_run(self, prompt, workspace, policy_path, *, mode='safe', plan=False):
        with self.storage.exclusive():
            root = Path(workspace).resolve(strict=True)
            policy = Policy.load(policy_path)
            policy.validate_workspace(root)
            if self.storage.path.is_relative_to(root):
                raise ValueError('State database must be outside workspace')
            if mode not in ('safe', 'auto'):
                raise ValueError('mode must be safe or auto')
            rid = uuid.uuid4().hex
            state = {'id': rid, 'version': 2, 'workspace': str(root), 'policy_path': str(policy.path), 'mode': mode, 'phase': 'planning' if plan else 'execution', 'status': 'running', 'messages': [{'role': 'system', 'content': PLAN_PROMPT if plan else SYSTEM_PROMPT}, {'role': 'user', 'content': prompt}], 'calls': {}, 'queue': [], 'tasks': [], 'plan': None, 'decisions': [], 'budget': policy.budgets, 'used': {k: 0 for k in BUDGET_KEYS}, 'output_reserved': 0, 'model_pending': False, 'files': {}, 'expected_tree': snapshot(root, policy), 'rollback': None}
            self.storage.create(state)
            return rid

    def registry(self, state):
        policy = Policy.load(state['policy_path'])
        policy.validate_workspace(state['workspace'])
        return ToolRegistry(state['workspace'], policy)

    def run_until_pause(self, run_id, *, max_iterations=1000):
        with self.storage.exclusive():
            self._recover_started(run_id)
            for _ in range(max_iterations):
                state = self.storage.get_run(run_id)
                if state['status'] == 'completed':
                    return state.get('final', '')
                if state['status'] in ('failed', 'rolled_back', 'rolling_back', 'budget_exhausted', 'protocol_error'):
                    raise StorageError('Run cannot resume: ' + state['status'])
                uncertain = [c['id'] for c in state['calls'].values() if c['status'] == 'uncertain']
                if uncertain:
                    raise RunPaused('uncertain_action', run_id, uncertain)
                if state['status'] == 'awaiting_plan_approval':
                    raise RunPaused('plan_approval', run_id)
                try:
                    if state['queue']:
                        cid = state['queue'][0]
                        call = state['calls'][cid]
                        if call['status'] == 'awaiting_approval':
                            raise RunPaused('action_approval', run_id, [cid])
                        self._execute(run_id, cid)
                    else:
                        self._ask_model(run_id)
                except BudgetExceeded as exc:
                    with self.storage.edit(run_id) as saved:
                        saved['status'] = 'budget_exhausted'
                        saved['reason'] = str(exc)
                    raise
            raise RunPaused('iteration_limit', run_id)

    def _ask_model(self, rid):
        state = self.storage.get_run(rid)
        registry = self.registry(state)
        task = current_task(state)
        messages = copy.deepcopy(state['messages'])
        if task:
            messages.append({'role': 'system', 'content': 'Current task (do not execute later tasks): ' + dumps(task)})
        with self.storage.edit(rid) as saved:
            debit(saved, model_requests=1)
            saved['model_pending'] = True
        self.hit('after_model_debit')
        if self.model is None:
            raise ValueError('Model is required to continue this run')
        response = self.model.complete(messages, registry.api_schemas(state['phase'] == 'planning'))
        self.hit('after_model_response')
        calls = response.tool_calls
        if not calls:
            self._text_response(rid, response.content or '', registry)
            return
        seen = set()
        prepared = []
        side = 0
        for call in calls:
            cid = call.get('id')
            if not isinstance(cid, str) or not cid or cid in seen:
                self._protocol_error(rid, 'Missing or duplicate tool_call_id in batch')
            seen.add(cid)
            name, args = call.get('name'), call.get('arguments')
            old = state['calls'].get(cid)
            if old:
                if old['name'] != name or old['arguments'] != args:
                    self._protocol_error(rid, 'Reused tool_call_id has different name or arguments')
                if old['status'] != 'completed':
                    self._protocol_error(rid, 'Reused unresolved tool_call_id')
                prepared.append((call, 'cached', False, None))
                continue
            try:
                spec = registry.validate(name, args, read_only=state['phase'] == 'planning')
                if state['tasks'] and (task is None or name not in task['tools']):
                    raise ValueError('Tool is not permitted for the current task')
                status = 'awaiting_approval' if spec.deletion or (spec.side_effect and state['mode'] == 'safe') else 'ready'
                side += int(spec.side_effect)
                prepared.append((call, status, spec.side_effect, None))
            except Exception as exc:
                prepared.append((call, 'ready', False, error(type(exc).__name__, exc)))
        api_calls = [{'id': c['id'], 'type': 'function', 'function': {'name': c.get('name', ''), 'arguments': dumps(c.get('arguments'))}} for c in calls]
        with self.storage.edit(rid) as saved:
            debit(saved, tool_calls=len(calls), side_effects=side)
            # Enough envelope capacity for the entire batch; no first call on an impossible batch.
            if saved['budget']['tool_output_bytes'] - saved['used']['tool_output_bytes'] < MIN_OUTPUT * len(calls):
                raise BudgetExceeded('tool_output_bytes cannot cover the whole batch')
            saved['model_pending'] = False
            saved['messages'].append({'role': 'assistant', 'content': response.content, 'tool_calls': api_calls})
            for c, status, effect, invalid in prepared:
                if status == 'cached':
                    saved['calls'][c['id']]['replay'] = True
                else:
                    saved['calls'][c['id']] = {'id': c['id'], 'name': c.get('name'), 'arguments': c.get('arguments'), 'status': status, 'side_effect': effect, 'invalid': invalid, 'task_id': task['id'] if task else None, 'attempt': 1, 'attempts': [], 'result': None, 'output_grant': 0, 'before_tree': None}
                saved['queue'].append(c['id'])
            if task:
                next(t for t in saved['tasks'] if t['id'] == task['id'])['status'] = 'running'
            self.hit('inside_batch_transaction')
        self.hit('after_batch_commit')

    def _protocol_error(self, rid, reason):
        with self.storage.edit(rid) as state:
            state['status'], state['reason'] = 'protocol_error', reason
        raise StorageError(reason)

    def _text_response(self, rid, content, registry):
        state = self.storage.get_run(rid)
        plan, invalid = None, None
        if state['phase'] == 'planning':
            try:
                plan = validate_plan(json.loads(content), registry)
            except Exception as exc:
                invalid = str(exc)
        with self.storage.edit(rid) as saved:
            saved['model_pending'] = False
            saved['messages'].append({'role': 'assistant', 'content': content})
            if invalid:
                saved['messages'].append({'role': 'user', 'content': 'Invalid plan. Repair it: ' + invalid})
            elif plan:
                saved['plan'] = plan
                saved['tasks'] = [dict(t, status='pending', evidence=None) for t in plan['steps']]
                saved['status'] = 'awaiting_plan_approval'
            elif any(t['status'] != 'completed' for t in saved['tasks']):
                saved['messages'].append({'role': 'user', 'content': 'Tasks are incomplete. Continue the current task and produce its required evidence.'})
            else:
                saved['status'], saved['final'] = 'completed', content

    def _execute(self, rid, cid):
        state = self.storage.get_run(rid)
        call = state['calls'][cid]
        registry = self.registry(state)
        replay = call.get('replay', False)
        if replay:
            # Cached replies have no filesystem action and therefore no uncertain window.
            with self.storage.edit(rid) as saved:
                remaining = saved['budget']['tool_output_bytes'] - saved['used']['tool_output_bytes']
                grant = min(registry.policy.max_output_bytes, remaining - MIN_OUTPUT * (len(saved['queue']) - 1))
                if grant < MIN_OUTPUT:
                    raise BudgetExceeded('tool_output_bytes exhausted')
                result = bounded(call['result'], grant)
                debit(saved, tool_output_bytes=size(result))
                saved['messages'].append({'role': 'tool', 'tool_call_id': cid, 'content': dumps(result)})
                saved['calls'][cid]['replay'] = False
                saved['queue'].pop(0)
            return
        result = call.get('invalid')
        if call['status'] == 'rejected':
            result = error('user_rejected', 'The user rejected this action')
        if replay:
            result = call['result']
        executed = result is None
        if result is None:
            try:
                spec = registry.validate(call['name'], call['arguments'], read_only=state['phase'] == 'planning')
                if (spec.deletion or (spec.side_effect and state['mode'] == 'safe')) and call['status'] != 'approved':
                    raise ValueError('Required approval missing')
                task = current_task(state)
                if state['tasks'] and (not task or task['id'] != call['task_id'] or call['name'] not in task['tools']):
                    raise ValueError('Task dependency or current-task check failed')
            except Exception as exc:
                result = error(type(exc).__name__, exc)
                executed = False
        before = snapshot(state['workspace'], registry.policy)
        if result is None and call['side_effect']:
            assert_expected(before, state['expected_tree'])
        with self.storage.edit(rid) as saved:
            remaining = saved['budget']['tool_output_bytes'] - saved['used']['tool_output_bytes']
            # Reserve envelopes for every remaining call in the batch.
            grant = min(registry.policy.max_output_bytes, remaining - MIN_OUTPUT * (len(saved['queue']) - 1))
            if grant < MIN_OUTPUT:
                raise BudgetExceeded('tool_output_bytes exhausted')
            saved['output_reserved'] = grant
            c = saved['calls'][cid]
            c['output_grant'] = grant
            c['before_tree'] = before
            c['status'] = 'started'
        self.hit('after_action_start')
        if result is None:
            try:
                result = registry.execute(call['name'], call['arguments'], grant)
            except Exception as exc:
                result = error(type(exc).__name__, exc)
        self.hit('after_tool_execution')
        after = snapshot(state['workspace'], registry.policy)
        # Bash must not alter protected tests. Its unexpected effects are still journaled.
        if call['name'] == 'Bash' and any(p in registry.policy.protected_files for p in changed(before, after)):
            result = error('protected_file_changed', 'Bash changed a protected file; inspect diff')
        output = bounded(result, grant)
        with self.storage.edit(rid) as saved:
            c = saved['calls'][cid]
            if call['side_effect'] and executed:
                record_changes(saved, before, after)
            debit(saved, tool_output_bytes=size(output))
            saved['output_reserved'] = 0
            c.update(status='completed', result=output, output_grant=0, replay=False)
            c['attempts'].append({'attempt': c['attempt'], 'status': 'completed', 'result': output})
            saved['messages'].append({'role': 'tool', 'tool_call_id': cid, 'content': dumps(output)})
            saved['queue'].pop(0)
            task = next((t for t in saved['tasks'] if t['id'] == call['task_id']), None)
            if task and task['status'] != 'completed':
                evidence = verify(task, call, result, after)
                if evidence:
                    task.update(status='completed', evidence=evidence)
            self.hit('inside_result_transaction')
        self.hit('after_result_commit')

    def _recover_started(self, rid):
        with self.storage.edit(rid) as state:
            for call in state['calls'].values():
                if call['status'] == 'started':
                    # Unknown output was potentially produced: conservatively spend the reservation.
                    debit(state, tool_output_bytes=call['output_grant'])
                    call['output_grant'] = 0
                    call['status'] = 'uncertain'
                    call['attempts'].append({'attempt': call['attempt'], 'status': 'uncertain'})
            state['output_reserved'] = 0

    def approve_action(self, rid, cid):
        self._decision(rid, cid, 'approved')

    def reject_action(self, rid, cid):
        self._decision(rid, cid, 'rejected')

    def _decision(self, rid, cid, decision):
        with self.storage.exclusive(), self.storage.edit(rid) as state:
            call = state['calls'].get(cid)
            if not call or call['status'] != 'awaiting_approval':
                raise StorageError('Call is not awaiting approval')
            call['status'] = decision
            state['decisions'].append({'kind': 'action', 'id': cid, 'attempt': call['attempt'], 'decision': decision})

    def approve_plan(self, rid):
        with self.storage.exclusive(), self.storage.edit(rid) as state:
            if state['status'] != 'awaiting_plan_approval':
                raise StorageError('Not awaiting plan approval')
            validate_plan(state['plan'], self.registry(state))
            state.update(status='running', phase='execution')
            state['decisions'].append({'kind': 'plan', 'decision': 'approved'})
            state['messages'].append({'role': 'system', 'content': SYSTEM_PROMPT + '\nThe saved plan is approved. Execute its tasks.'})

    def reject_plan(self, rid):
        with self.storage.exclusive(), self.storage.edit(rid) as state:
            if state['status'] != 'awaiting_plan_approval':
                raise StorageError('Not awaiting plan approval')
            state['status'] = 'failed'
            state['decisions'].append({'kind': 'plan', 'decision': 'rejected'})

    def recovery_info(self, rid, cid):
        state = self.storage.get_run(rid)
        call = state['calls'].get(cid)
        if not call or call['status'] not in ('started', 'uncertain'):
            raise StorageError('No uncertain action with this id')
        current = snapshot(state['workspace'], self.registry(state).policy)
        before = call['before_tree'] or state['expected_tree']
        return {'call_id': cid, 'attempt': call['attempt'], 'changed_paths': changed(before, current), 'diff': diff_trees(before, current)}

    def resolve_uncertain(self, rid, cid, decision, *, attribution=None):
        with self.storage.exclusive():
            self._recover_started(rid)
            state = self.storage.get_run(rid)
            call = state['calls'].get(cid)
            if not call or call['status'] != 'uncertain' or decision not in ('retry', 'fail', 'abort'):
                raise StorageError('Invalid uncertain action or decision')
            registry = self.registry(state)
            after = snapshot(state['workspace'], registry.policy)
            before = call['before_tree'] or state['expected_tree']
            if changed(before, after) and attribution not in ('accept', 'exclude'):
                raise StorageError('Review recover show, then explicitly choose --changes accept or exclude')
            # Test code can have arbitrary side effects: even unittest Bash is not retryable.
            if decision == 'retry':
                if call['name'] not in ('read_file', 'list_files', 'write_file'):
                    raise StorageError('This tool is not retryable; choose fail or abort')
                registry.validate(call['name'], call['arguments'], state['phase'] == 'planning')
                if changed(before, after) and attribution != 'accept':
                    raise StorageError('Retry would overwrite unowned changes; accept them or choose fail/abort')
            with self.storage.edit(rid) as saved:
                c = saved['calls'][cid]
                if decision == 'retry':
                    debit(saved, tool_calls=1, side_effects=int(c['side_effect']))
                    if saved['budget']['tool_output_bytes'] - saved['used']['tool_output_bytes'] < MIN_OUTPUT * len(saved['queue']):
                        raise BudgetExceeded('No output budget for retry')
                if attribution == 'accept':
                    record_changes(saved, before, after)
                else:
                    saved['expected_tree'] = after
                saved['decisions'].append({'kind': 'recovery', 'id': cid, 'attempt': c['attempt'], 'decision': decision, 'changes': attribution, 'paths': changed(before, after)})
                if decision == 'abort':
                    c['status'] = 'failed'
                    saved['status'] = 'failed'
                elif decision == 'fail':
                    c['invalid'] = error('uncertain_marked_failed', 'User chose not to repeat this action')
                    c['status'] = 'ready'
                else:
                    c['attempt'] += 1
                    c['status'] = 'awaiting_approval' if c['side_effect'] and saved['mode'] == 'safe' else 'ready'
                    c['invalid'] = None
                self.hit('inside_recovery_transaction')

    def task_checklist(self, rid):
        return '\n'.join(f"[{'x' if t['status'] == 'completed' else ' '}] {t['id']}: {t['description']} ({t['status']})" for t in self.storage.tasks(rid))
