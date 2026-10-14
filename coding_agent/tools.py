"""Schemas, bounded tool results and subprocess execution. No agent decisions here."""
from __future__ import annotations
import json
import os
import shutil
import signal
import subprocess
import tempfile
import threading
from dataclasses import dataclass
from pathlib import Path
from .security import PolicyError, resolve_workspace_path


class ToolValidationError(ValueError):
    pass


def dumps(value):
    return json.dumps(value, ensure_ascii=False, separators=(',', ':'))


def size(value):
    return len(dumps(value).encode('utf-8'))


def validate_schema(value, schema):
    types = {'object': dict, 'array': list, 'string': str, 'integer': int, 'boolean': bool}
    if 'type' in schema and type(value) is not types[schema['type']]:
        raise ToolValidationError('Expected ' + schema['type'])
    if 'enum' in schema and value not in schema['enum']:
        raise ToolValidationError('Unexpected enum value')
    if isinstance(value, dict):
        props = schema.get('properties', {})
        if set(schema.get('required', [])) - value.keys():
            raise ToolValidationError('Missing required argument')
        if schema.get('additionalProperties') is False and value.keys() - props.keys():
            raise ToolValidationError('Unexpected arguments')
        for k, v in value.items():
            if k in props:
                validate_schema(v, props[k])
    if isinstance(value, list) and 'items' in schema:
        for item in value:
            validate_schema(item, schema['items'])


def bounded(result, limit):
    """Exact UTF-8 JSON bound, including escaping and metadata; preserve Bash status."""
    full = dict(result, truncated=result.get('truncated', False))
    if size(full) <= limit:
        return full
    base = {'ok': bool(result.get('ok')), 'truncated': True}
    for key in ('exit_code', 'timed_out'):
        if key in result:
            base[key] = result[key]
    for key in ('stdout', 'stderr'):
        if key in result:
            base[key] = ''
    if size(base) > limit:
        raise ValueError('Insufficient space for result envelope')
    text_keys = [k for k in ('stdout', 'stderr', 'content', 'path') if k in result]
    if not text_keys:
        text_keys = ['preview']
        result = {'preview': dumps(result)}
    for key in text_keys:
        text = str(result[key])
        lo, hi = 0, len(text)
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if size(dict(base, **{key: text[:mid]})) <= limit:
                lo = mid
            else:
                hi = mid - 1
        if size(dict(base, **{key: text[:lo]})) <= limit:
            base[key] = text[:lo]
    return base


def atomic_write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix='.agent-', dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as output:
            output.write(data)
            output.flush()
            os.fsync(output.fileno())
        if path.exists():
            os.chmod(temporary, path.stat().st_mode)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


@dataclass(frozen=True)
class ToolSpec:
    name: str
    schema: dict
    side_effect: bool = False
    deletion: bool = False

    def api_schema(self):
        descriptions = {'Bash': 'Run an exact allowed unittest command through Bash', 'list_files': 'List permitted files', 'read_file': 'Read a UTF-8 file', 'write_file': 'Create or replace a UTF-8 file', 'delete_file': 'Delete a single file, requires approval'}
        return {'type': 'function', 'function': {'name': self.name, 'description': descriptions[self.name], 'parameters': self.schema}}


def schema(*fields):
    return {'type': 'object', 'properties': {f: {'type': 'string'} for f in fields}, 'required': list(fields), 'additionalProperties': False}


class ToolRegistry:
    def __init__(self, workspace, policy):
        self.workspace, self.policy = Path(workspace).resolve(), policy
        self.specs = {s.name: s for s in [ToolSpec('list_files', schema('path')), ToolSpec('read_file', schema('path')), ToolSpec('write_file', schema('path', 'content'), True), ToolSpec('delete_file', schema('path'), True, True), ToolSpec('Bash', schema('command', 'cwd'), True)]}

    def api_schemas(self, read_only=False):
        return [s.api_schema() for s in self.specs.values() if not read_only or not s.side_effect]

    def validate(self, name, arguments, read_only=False):
        if not isinstance(name, str) or name not in self.specs:
            raise ToolValidationError('Unknown tool')
        spec = self.specs[name]
        validate_schema(arguments, spec.schema)
        if read_only and spec.side_effect:
            raise PolicyError('Plan Mode permits only reading')
        if name == 'Bash':
            self.policy.validate_command(arguments['command'])
            target = resolve_workspace_path(self.workspace, arguments['cwd'], self.policy, must_exist=True)
            if not target.is_dir():
                raise PolicyError('Bash cwd must be a directory')
            # Discovery/import may follow symlinks; reject them before invoking unittest.
            for directory, dirs, names in os.walk(target, followlinks=False):
                if any((Path(directory) / n).is_symlink() for n in dirs + names):
                    raise PolicyError('Bash test tree must not contain symlinks')
            # unittest discovers files recursively. Never run across a denied subtree.
            for denied in self.policy.directories_deny:
                forbidden = (self.workspace / denied).resolve()
                if forbidden.exists() and forbidden.is_relative_to(target):
                    raise PolicyError('Bash cwd contains a denied directory: ' + denied)
        else:
            target = resolve_workspace_path(self.workspace, arguments['path'], self.policy, must_exist=name != 'write_file', write=spec.side_effect)
            if name == 'list_files' and not target.is_dir():
                raise NotADirectoryError(arguments['path'])
            if name != 'list_files' and target.exists() and not target.is_file():
                raise IsADirectoryError(arguments['path'])
        return spec

    def execute(self, name, arguments, output_limit):
        # Defense in depth, called again immediately before actual I/O.
        self.validate(name, arguments)
        if name == 'Bash':
            return self._bash(arguments, output_limit)
        path = resolve_workspace_path(self.workspace, arguments['path'], self.policy, write=self.specs[name].side_effect)
        relative = path.relative_to(self.workspace).as_posix()
        if name == 'read_file':
            with path.open('rb') as file:
                data = file.read(output_limit + 1)
            return {'ok': True, 'path': relative, 'content': data[:output_limit].decode('utf-8', errors='replace'), 'truncated': len(data) > output_limit}
        if name == 'list_files':
            files = []
            for directory, dirs, names in os.walk(path, followlinks=False):
                permitted = []
                for d in dirs:
                    candidate = Path(directory) / d
                    try:
                        resolve_workspace_path(self.workspace, candidate.relative_to(self.workspace).as_posix(), self.policy)
                        if not candidate.is_symlink():
                            permitted.append(d)
                    except PolicyError:
                        pass
                dirs[:] = sorted(permitted)
                for n in sorted(names):
                    candidate = Path(directory) / n
                    rel = candidate.relative_to(self.workspace).as_posix()
                    try:
                        resolve_workspace_path(self.workspace, rel, self.policy)
                    except PolicyError:
                        continue
                    files.append(rel)
                    if size(files) > output_limit:
                        return {'ok': True, 'files': files[:-1], 'truncated': True}
            return {'ok': True, 'files': files}
        if name == 'write_file':
            atomic_write(path, arguments['content'].encode('utf-8'))
            return {'ok': True, 'path': relative, 'bytes_written': len(arguments['content'].encode('utf-8'))}
        path.unlink()
        return {'ok': True, 'path': relative, 'deleted': True}

    def _bash(self, arguments, output_limit):
        executable = os.environ.get('AGENT_BASH') or shutil.which('bash')
        if not executable:
            raise RuntimeError('Bash unavailable. Use Ubuntu/WSL or set AGENT_BASH to a Bash executable')
        cwd = resolve_workspace_path(self.workspace, arguments['cwd'], self.policy, must_exist=True)
        env = os.environ.copy()
        for key in ('BASH_ENV', 'ENV', 'SHELLOPTS', 'BASHOPTS', 'CDPATH', 'PYTHONPATH', 'PYTHONSTARTUP'):
            env.pop(key, None)
        env['PYTHONDONTWRITEBYTECODE'] = '1'
        env['PYTHONIOENCODING'] = 'utf-8'
        process = subprocess.Popen([executable, '--noprofile', '--norc', '-c', arguments['command']], cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=os.name != 'nt')
        cap = min(output_limit, self.policy.max_output_bytes)
        buffers = [bytearray(), bytearray()]
        total = [0]
        truncated = [False]
        mutex = threading.Lock()
        def drain(stream, index):
            try:
                while True:
                    chunk = stream.read(4096)
                    if not chunk:
                        break
                    with mutex:
                        take = min(len(chunk), max(0, cap - total[0]))
                        buffers[index].extend(chunk[:take])
                        total[0] += take
                        truncated[0] |= take < len(chunk)
            finally:
                stream.close()
        threads = [threading.Thread(target=drain, args=(stream, i), daemon=True) for i, stream in enumerate((process.stdout, process.stderr))]
        for thread in threads:
            thread.start()
        timed_out = False
        try:
            process.wait(timeout=self.policy.timeout_seconds)
        except subprocess.TimeoutExpired:
            timed_out = True
            if os.name == 'nt':
                subprocess.run(['taskkill', '/PID', str(process.pid), '/T', '/F'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=10)
                if process.poll() is None:
                    process.kill()
            else:
                os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=10)
        for thread in threads:
            thread.join(timeout=5)
        if any(thread.is_alive() for thread in threads):
            if os.name != 'nt':
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                for thread in threads:
                    thread.join(timeout=5)
            raise RuntimeError('Child process kept output pipes open')
        return {'ok': process.returncode == 0 and not timed_out, 'stdout': buffers[0].decode('utf-8', errors='replace'), 'stderr': buffers[1].decode('utf-8', errors='replace'), 'exit_code': process.returncode, 'timed_out': timed_out, 'truncated': truncated[0]}
