"""Byte-exact snapshots, run-owned changes, conflict checks and restartable rollback."""
from __future__ import annotations
import base64
import difflib
import hashlib
import os
import stat
from pathlib import Path
from .security import is_generated, resolve_workspace_path, Policy
from .storage import StorageError
from .tools import atomic_write


class WorkspaceConflict(StorageError):
    pass


def snapshot(workspace, policy):
    root = Path(workspace).resolve()
    tree = {}
    for directory, dirs, names in os.walk(root, followlinks=False):
        for name in sorted(names + dirs):
            path = Path(directory) / name
            rel = path.relative_to(root).as_posix()
            if path.is_symlink():
                tree[rel] = {'kind': 'symlink', 'target': os.readlink(path)}
            elif path.is_file() and not is_generated(rel, policy):
                data = path.read_bytes()
                tree[rel] = {'kind': 'file', 'data': base64.b64encode(data).decode('ascii'), 'hash': hashlib.sha256(data).hexdigest(), 'mode': stat.S_IMODE(path.stat().st_mode)}
        dirs[:] = sorted(d for d in dirs if not (Path(directory) / d).is_symlink())
    return tree


def changed(before, after):
    return sorted(p for p in before.keys() | after.keys() if before.get(p) != after.get(p))


def assert_expected(current, expected):
    paths = changed(expected, current)
    if paths:
        raise WorkspaceConflict('External workspace change: ' + ', '.join(paths))


def record_changes(state, before, after):
    for path in changed(before, after):
        if path not in state['files']:
            state['files'][path] = {'before': before.get(path), 'after': after.get(path)}
        else:
            state['files'][path]['after'] = after.get(path)
        if state['files'][path]['before'] == state['files'][path]['after']:
            del state['files'][path]
    state['expected_tree'] = after


def diff_trees(before, after):
    chunks = []
    for path in changed(before, after):
        old, new = before.get(path), after.get(path)
        def text(item):
            if item is None:
                return ''
            if item['kind'] == 'symlink':
                return 'symlink -> ' + item['target'] + '\n'
            return base64.b64decode(item['data']).decode('utf-8', errors='replace')
        lines = list(difflib.unified_diff(text(old).splitlines(True), text(new).splitlines(True), fromfile='a/' + path if old else '/dev/null', tofile='b/' + path if new else '/dev/null'))
        chunks.extend(lines or [f'{path}: metadata or empty-file change\n'])
    return ''.join(chunks)


def unified_diff(storage, run_id):
    files = storage.get_run(run_id)['files']
    return diff_trees({p: v['before'] for p, v in files.items() if v['before'] is not None}, {p: v['after'] for p, v in files.items() if v['after'] is not None})


def rollback(storage, run_id, crash_hook=None):
    with storage.exclusive():
        state = storage.get_run(run_id)
        if any(c['status'] in ('started', 'uncertain') for c in state['calls'].values()):
            raise StorageError('Resolve uncertain calls and attribute their changes before rollback')
        if state['status'] == 'rolled_back':
            return []
        policy = Policy.load(state['policy_path'])
        policy.validate_workspace(state['workspace'])
        tree = snapshot(state['workspace'], policy)
        job = state.get('rollback')
        # Validate ALL files and permissions before the first mutation.
        for path, entry in state['files'].items():
            resolve_workspace_path(state['workspace'], path, policy, write=True)
            if any(item is not None and item['kind'] != 'file' for item in entry.values()):
                raise WorkspaceConflict('Rollback of symlinks is not supported: ' + path)
            current = tree.get(path)
            if current != entry['after'] and not (job and current == entry['before']):
                raise WorkspaceConflict('Rollback conflict; no files changed: ' + path)
        with storage.edit(run_id) as saved:
            saved['rollback'] = {'status': 'started'}
            saved['status'] = 'rolling_back'
        restored = []
        for path, entry in state['files'].items():
            target = resolve_workspace_path(state['workspace'], path, policy, write=True)
            if tree.get(path) != entry['before']:
                if entry['before'] is None:
                    target.unlink()
                else:
                    atomic_write(target, base64.b64decode(entry['before']['data']))
                    os.chmod(target, entry['before']['mode'])
            restored.append(path)
            if crash_hook:
                crash_hook('during_rollback')
        after = snapshot(state['workspace'], policy)
        with storage.edit(run_id) as saved:
            saved['rollback'] = {'status': 'completed'}
            saved['status'] = 'rolled_back'
            saved['expected_tree'] = after
            saved['decisions'].append({'kind': 'rollback', 'decision': 'restore_run_files'})
        return restored
