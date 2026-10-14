"""Plan validation and objective evidence for each task."""
from .security import lexical


PLAN_PROMPT = '''You are in read-only Plan Mode. Inspect the project, then return only a JSON plan:
{"goal":"...","steps":[{"id":"inspect","description":"Read order.py","tools":["read_file"],"dependencies":[],"verification":{"kind":"file_read","path":"order.py"}},{"id":"fix","description":"Fix order.py","tools":["write_file"],"dependencies":["inspect"],"verification":{"kind":"file_written","path":"order.py"}},{"id":"test","description":"Run tests","tools":["Bash"],"dependencies":["fix"],"verification":{"kind":"tests_pass","command":"python3 -m unittest -v","cwd":"."}}],"risks":[],"assumptions":[],"done_when":["tests pass"]}.
Use task-specific evidence. Allowed verification kinds: file_read, files_listed, file_written, file_deleted, tests_pass. Each kind needs path, except tests_pass which needs command and cwd. No Bash, write or delete until the plan is approved.'''

KIND_TOOL = {'file_read': 'read_file', 'files_listed': 'list_files', 'file_written': 'write_file', 'file_deleted': 'delete_file', 'tests_pass': 'Bash'}


def validate_plan(plan, registry):
    if not isinstance(plan, dict) or set(plan) != {'goal', 'steps', 'risks', 'assumptions', 'done_when'}:
        raise ValueError('Plan requires goal, steps, risks, assumptions, done_when')
    if not isinstance(plan['goal'], str) or not plan['goal'].strip():
        raise ValueError('Plan goal must be nonempty')
    for field in ('risks', 'assumptions', 'done_when'):
        if not isinstance(plan[field], list) or any(not isinstance(v, str) for v in plan[field]):
            raise ValueError(field + ' must be an array of strings')
    if not plan['done_when'] or not isinstance(plan['steps'], list) or not plan['steps']:
        raise ValueError('Plan needs steps and completion criteria')
    ids = set()
    for task in plan['steps']:
        if not isinstance(task, dict) or set(task) != {'id', 'description', 'tools', 'dependencies', 'verification'}:
            raise ValueError('Every task needs id, description, tools, dependencies, verification')
        if not isinstance(task['id'], str) or not task['id'] or task['id'] in ids:
            raise ValueError('Task ids must be nonempty and unique')
        ids.add(task['id'])
        if not isinstance(task['description'], str) or not task['description']:
            raise ValueError('Task description required')
        if not isinstance(task['dependencies'], list) or any(not isinstance(d, str) for d in task['dependencies']):
            raise ValueError('Dependencies must be an array of ids')
        if not isinstance(task['tools'], list) or not task['tools'] or any(t not in registry.specs for t in task['tools']):
            raise ValueError('Unknown or missing task tools')
        rule = task['verification']
        if not isinstance(rule, dict) or rule.get('kind') not in KIND_TOOL:
            raise ValueError('Unknown verification kind')
        tool = KIND_TOOL[rule['kind']]
        if tool not in task['tools']:
            raise ValueError('Verification tool must be in task tools')
        if tool == 'Bash':
            if set(rule) != {'kind', 'command', 'cwd'}:
                raise ValueError('tests_pass requires command and cwd')
            registry.validate('Bash', {'command': rule['command'], 'cwd': rule['cwd']})
        else:
            if set(rule) != {'kind', 'path'}:
                raise ValueError('File verification requires path')
            lexical(rule['path'])
            from .security import resolve_workspace_path
            resolve_workspace_path(registry.workspace, rule['path'], registry.policy, write=tool in ('write_file', 'delete_file'))
    done = set()
    remaining = list(plan['steps'])
    while remaining:
        ready = [t for t in remaining if set(t['dependencies']).issubset(done)]
        if not ready:
            raise ValueError('Plan dependencies contain a cycle or an unknown id')
        for task in ready:
            remaining.remove(task)
            done.add(task['id'])
    return plan


def current_task(state):
    done = {t['id'] for t in state['tasks'] if t['status'] == 'completed'}
    return next((t for t in state['tasks'] if t['status'] != 'completed' and set(t['dependencies']).issubset(done)), None)


def verify(task, call, result, tree):
    if not result.get('ok'):
        return None
    rule, args = task['verification'], call['arguments']
    if KIND_TOOL[rule['kind']] != call['name']:
        return None
    if rule['kind'] == 'tests_pass':
        if args != {'command': rule['command'], 'cwd': rule['cwd']} or result.get('exit_code') != 0:
            return None
        evidence = {'exit_code': 0, 'command': args['command'], 'cwd': args['cwd']}
    else:
        from pathlib import PurePosixPath
        if PurePosixPath(args['path']) != PurePosixPath(rule['path']):
            return None
        file = tree.get(str(PurePosixPath(rule['path'])))
        if rule['kind'] == 'file_written':
            import hashlib
            if not file or file.get('hash') != hashlib.sha256(args['content'].encode()).hexdigest():
                return None
        if rule['kind'] == 'file_deleted' and file is not None:
            return None
        evidence = {'path': rule['path'], 'hash': file.get('hash') if file else None}
    return dict(evidence, tool_call_id=call['id'], attempt=call['attempt'])
