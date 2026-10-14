"""CLI decisions persist separately; resume explicitly continues model execution."""
from __future__ import annotations
import argparse
import json
import sys
from .storage import Storage
from .runtime import AgentRuntime, RunPaused
from .model import OpenAICompatibleModel
from .journal import rollback, unified_diff


def parser():
    root = argparse.ArgumentParser(prog='coding-agent')
    root.add_argument('--state', default='state/agent-v2.sqlite3')
    commands = root.add_subparsers(dest='command', required=True)
    run = commands.add_parser('run')
    run.add_argument('--workspace', required=True)
    run.add_argument('--policy', required=True)
    run.add_argument('--mode', choices=['safe', 'auto'], default='safe')
    run.add_argument('--plan', action='store_true')
    run.add_argument('--prompt', required=True)
    for name in ('resume', 'status', 'tasks', 'diff', 'rollback'):
        commands.add_parser(name).add_argument('run_id')
    for name in ('action', 'plan', 'recover'):
        cmd = commands.add_parser(name)
        cmd.add_argument('decision', choices=['approve', 'reject'] if name != 'recover' else ['show', 'retry', 'fail', 'abort'])
        cmd.add_argument('run_id')
        if name != 'plan':
            cmd.add_argument('call_id')
        if name == 'recover':
            cmd.add_argument('--changes', choices=['accept', 'exclude'])
    return root


def drive(runtime, rid):
    try:
        print(runtime.run_until_pause(rid))
    except RunPaused as paused:
        print('Paused:', paused.reason)
        if paused.reason == 'plan_approval':
            print(json.dumps(runtime.storage.get_run(rid)['plan'], indent=2, ensure_ascii=False))
            print(f'Approve: python -m coding_agent plan approve {rid}')
        for cid in paused.call_ids:
            call = runtime.storage.get_call(rid, cid)
            print(f"{cid}, attempt {call['attempt']}: {call['name']} {json.dumps(call['arguments'], ensure_ascii=False)}")
            if paused.reason == 'uncertain_action':
                print(json.dumps(runtime.recovery_info(rid, cid), indent=2, ensure_ascii=False))
    checklist = runtime.task_checklist(rid)
    if checklist:
        print(checklist)


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        storage = Storage(args.state)
        runtime = AgentRuntime(storage)
        if args.command in ('run', 'resume'):
            runtime.model = OpenAICompatibleModel()
            rid = args.run_id if args.command == 'resume' else runtime.create_run(args.prompt, args.workspace, args.policy, mode=args.mode, plan=args.plan)
            print('run_id=' + rid)
            drive(runtime, rid)
        elif args.command in ('action', 'plan'):
            method = getattr(runtime, args.decision + '_' + args.command)
            method(args.run_id, args.call_id) if args.command == 'action' else method(args.run_id)
            print('Decision saved. Continue: python -m coding_agent resume ' + args.run_id)
        elif args.command == 'recover':
            if args.decision == 'show':
                print(json.dumps(runtime.recovery_info(args.run_id, args.call_id), indent=2, ensure_ascii=False))
            else:
                runtime.resolve_uncertain(args.run_id, args.call_id, args.decision, attribution=args.changes)
                print('Recovery decision saved. Continue with resume, or inspect diff / rollback.')
        elif args.command == 'status':
            state = storage.get_run(args.run_id)
            public = {k: state[k] for k in ('id', 'phase', 'status', 'used', 'budget', 'output_reserved', 'tasks', 'decisions')}
            public['calls'] = [{k: c[k] for k in ('id', 'name', 'status', 'task_id', 'attempt')} for c in state['calls'].values()]
            print(json.dumps(public, indent=2, ensure_ascii=False))
        elif args.command == 'tasks':
            print(runtime.task_checklist(args.run_id))
        elif args.command == 'diff':
            print(unified_diff(storage, args.run_id) or 'No changes recorded.')
        elif args.command == 'rollback':
            print('Restored:', ', '.join(rollback(storage, args.run_id)) or 'nothing')
        return 0
    except Exception as exc:
        print(f'{type(exc).__name__}: {exc}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
