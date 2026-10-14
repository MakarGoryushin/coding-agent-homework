"""Fail-closed policy independent of the agent loop. This is not an OS sandbox."""
from __future__ import annotations
import json
import re
from pathlib import Path, PureWindowsPath


class PolicyError(ValueError):
    pass


BUDGET_KEYS = ('model_requests', 'tool_calls', 'side_effects', 'tool_output_bytes')


class Policy:
    @classmethod
    def load(cls, path):
        obj = cls()
        obj.path = Path(path).resolve(strict=True)
        raw = json.loads(obj.path.read_text(encoding='utf-8'))
        obj.raw = raw
        for section in ('commands', 'directories'):
            for key in ('allow', 'deny'):
                value = raw.get(section, {}).get(key)
                if not isinstance(value, list) or any(not isinstance(v, str) for v in value):
                    raise PolicyError(f'{section}.{key} must be an explicit list of strings')
                setattr(obj, section + '_' + key, tuple(value))
        for entry in (*obj.directories_allow, *obj.directories_deny):
            lexical(entry)
        obj.budgets = raw.get('budgets', {})
        for key in BUDGET_KEYS:
            n = obj.budgets.get(key)
            if type(n) is not int or n < 0:
                raise PolicyError(f'budgets.{key} must be a nonnegative integer')
        obj.timeout_seconds = raw.get('bash', {}).get('timeout_seconds', 30)
        obj.max_output_bytes = raw.get('bash', {}).get('max_output_bytes', 8192)
        if type(obj.timeout_seconds) is not int or not 1 <= obj.timeout_seconds <= 300:
            raise PolicyError('Bash timeout must be in 1..300 seconds')
        if type(obj.max_output_bytes) is not int or not 256 <= obj.max_output_bytes <= 1048576:
            raise PolicyError('Bash output limit must be in 256..1048576 bytes')
        obj.protected_files = raw.get('protected_files', ['test_order.py'])
        if not isinstance(obj.protected_files, list):
            raise PolicyError('protected_files must be a list')
        for entry in obj.protected_files:
            lexical(entry)
        generated = raw.get('generated', ['__pycache__', '*.pyc'])
        if not isinstance(generated, list) or any(entry not in ('__pycache__', '*.pyc') for entry in generated):
            raise PolicyError('Only __pycache__ and *.pyc generated exceptions are supported')
        obj.generated = generated
        return obj

    def validate_workspace(self, workspace):
        root = Path(workspace).resolve(strict=True)
        if not root.is_dir() or self.path.is_relative_to(root):
            raise PolicyError('Policy must be outside an existing workspace directory')
        for entry in (*self.directories_allow, *self.directories_deny):
            if not (root / entry).resolve().is_relative_to(root):
                raise PolicyError('Policy directory escapes workspace: ' + entry)

    def validate_command(self, command):
        if command in self.commands_deny:
            raise PolicyError('Command explicitly denied')
        if command not in self.commands_allow:
            raise PolicyError('Command is not an exact entry in commands.allow')
        if not re.fullmatch(r'python3? -m unittest(?: test_[A-Za-z0-9_]+\.py)? -v', command):
            raise PolicyError('Only simple Python unittest commands are supported')


def lexical(requested):
    if not isinstance(requested, str) or not requested or '\x00' in requested:
        raise PolicyError('Path must be a nonempty string without NUL')
    if Path(requested).is_absolute() or PureWindowsPath(requested).drive or '\\' in requested:
        raise PolicyError('Use relative POSIX paths')
    if '..' in Path(requested).parts:
        raise PolicyError('Parent traversal is forbidden')


def resolve_workspace_path(workspace, requested, policy, *, must_exist=False, write=False):
    lexical(requested)
    root = Path(workspace).resolve(strict=True)
    target = (root / requested).resolve(strict=False)
    if not target.is_relative_to(root):
        raise PolicyError('Path escapes workspace (possibly through a symlink)')
    for entry in policy.directories_deny:
        if target.is_relative_to((root / entry).resolve()):
            raise PolicyError('Denied directory: ' + entry)
    if not any(target.is_relative_to((root / entry).resolve()) for entry in policy.directories_allow):
        raise PolicyError('Path is not explicitly allowed')
    if write:
        cur = root
        for part in Path(requested).parts:
            cur /= part
            if cur.is_symlink():
                raise PolicyError('Writes through symlinks are forbidden')
        if target == root:
            raise PolicyError('Cannot write workspace root')
        if target.relative_to(root).as_posix() in policy.protected_files:
            raise PolicyError('Protected file: ' + requested)
    if must_exist and not target.exists():
        raise FileNotFoundError(requested)
    return target


def is_generated(relative_path, policy):
    path = Path(relative_path)
    return path.suffix == '.pyc' and ('*.pyc' in policy.generated or ('__pycache__' in policy.generated and '__pycache__' in path.parts))
