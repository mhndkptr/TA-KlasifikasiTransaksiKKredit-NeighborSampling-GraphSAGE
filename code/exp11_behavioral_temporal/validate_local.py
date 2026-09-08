"""Fixed smoke + three-preset pilot plan; does not tune from test results."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--rows', type=int, default=1000000)
    parser.add_argument('--epochs', type=int, default=30)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    output = root/'validation'
    output.mkdir(exist_ok=True)
    commands = [('smoke', [str(root/'run.py'), '--config', str(root/'config.smoke.yaml'), '--no-progress'])]
    for preset, name in [('config.exp10_control.yaml', 'control'),
                         ('config.features_only.yaml', 'features_only'),
                         ('config.gpu16gb.yaml', 'main')]:
        commands.append((name, [str(root/'run.py'), '--config', str(root/preset),
            '--max-rows', str(args.rows), '--epochs', str(args.epochs), '--strategy', 'uniform',
            '--seed', '42', '--name', f'exp11_pilot_{args.rows}_{name}', '--no-progress']))
    plan = {'created_utc': datetime.now(timezone.utc).isoformat(), 'commands': commands,
            'source': {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                       for p in sorted((root/'exp11').rglob('*.py'))}}
    (output/'pilot_plan.json').write_text(json.dumps(plan, indent=2), encoding='utf-8')
    for name, arguments in commands:
        print(f'START {name}', flush=True)
        with (output/f'{name}.log').open('w', encoding='utf-8') as stream:
            result = subprocess.run([sys.executable, *arguments], stdout=stream, stderr=subprocess.STDOUT)
        print(f'END {name}: exit={result.returncode}; log={output/name}.log', flush=True)
        if result.returncode:
            return result.returncode
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
