"""One durable JSON document per run; every state transition is one SQLite commit."""
from __future__ import annotations
import json
import sqlite3
from contextlib import contextmanager, closing
from pathlib import Path


class StorageError(RuntimeError):
    pass


class BudgetExceeded(StorageError):
    pass


class Storage:
    def __init__(self, path):
        self.path = Path(path).resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with closing(self.connect()) as db, db:
            db.execute('CREATE TABLE IF NOT EXISTS runs_v2 (id TEXT PRIMARY KEY, data TEXT NOT NULL)')

    def connect(self):
        db = sqlite3.connect(self.path, timeout=10)
        db.execute('PRAGMA synchronous=FULL')
        return db

    def create(self, state):
        with closing(self.connect()) as db, db:
            db.execute('INSERT INTO runs_v2 VALUES (?,?)', (state['id'], json.dumps(state)))

    def get_run(self, run_id):
        with closing(self.connect()) as db:
            row = db.execute('SELECT data FROM runs_v2 WHERE id=?', (run_id,)).fetchone()
        if row is None:
            raise StorageError('Unknown run_id (v1 runs are intentionally not migrated): ' + run_id)
        return json.loads(row[0])

    @contextmanager
    def edit(self, run_id):
        db = self.connect()
        try:
            db.execute('BEGIN IMMEDIATE')
            row = db.execute('SELECT data FROM runs_v2 WHERE id=?', (run_id,)).fetchone()
            if row is None:
                raise StorageError('Unknown run_id: ' + run_id)
            state = json.loads(row[0])
            yield state
            db.execute('UPDATE runs_v2 SET data=? WHERE id=?', (json.dumps(state), run_id))
            db.commit()
        except BaseException:
            db.rollback()
            raise
        finally:
            db.close()

    @contextmanager
    def exclusive(self):
        """Nonblocking OS lock, automatically released on process death."""
        handle = self.path.with_suffix(self.path.suffix + '.lock').open('a+b')
        handle.seek(0, 2)
        if handle.tell() == 0:
            handle.write(b'0')
            handle.flush()
        handle.seek(0)
        try:
            if __import__('os').name == 'nt':
                import msvcrt
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            handle.close()
            raise StorageError('Another CLI operation is using this state database') from exc
        try:
            yield
        finally:
            handle.close()

    def messages(self, run_id):
        return self.get_run(run_id)['messages']

    def tasks(self, run_id):
        return self.get_run(run_id)['tasks']

    def get_call(self, run_id, call_id):
        return self.get_run(run_id)['calls'].get(call_id)

    def calls_with_status(self, run_id, *statuses):
        return [c for c in self.get_run(run_id)['calls'].values() if c['status'] in statuses]
