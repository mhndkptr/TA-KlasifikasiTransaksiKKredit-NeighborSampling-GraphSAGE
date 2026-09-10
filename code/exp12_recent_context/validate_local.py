"""Predeclare controls/ablations; execute without tuning from test scores."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone

from exp12.artifacts import source_manifest, write_summaries


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--rows', type=int, default=1000000)
    parser.add_argument('--epochs', type=int, default=30)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    output = root/'validation'
    output.mkdir(exist_ok=True)
    commands = [('smoke', [str(root/'run.py'), '--config', str(root/'config.smoke.yaml'), '--no-progress'])]
    for preset, name in [('config.control_gpu.yaml', 'control'), ('config.features_only.yaml', 'features_only'),
                         ('config.logic_only.yaml', 'logic_only'), ('config.gpu16gb.yaml', 'main'),
                         ('config.self_only.yaml', 'self_only')]:
        command = [str(root/'run.py'), '--config', str(root/preset),
            '--max-rows', str(args.rows), '--epochs', str(args.epochs), '--strategy', 'uniform',
            '--seed', '42', '--name', f'exp12_pilot_{args.rows}_{name}', '--no-progress']
        if name in {'logic_only', 'main', 'self_only'}:
            command += ['--selection-min-fraud', '1', '--calibration-min-fraud', '1']
        commands.append((name, command))
    plan = {'created_utc': datetime.now(timezone.utc).isoformat(), 'commands': commands,
            'source': source_manifest(), 'selection': 'No preset chosen from test; report all outcomes'}
    (output/'pilot_plan.json').write_text(json.dumps(plan, indent=2), encoding='utf-8')
    for name, arguments in commands:
        print(f'START {name}', flush=True)
        with (output/f'{name}.log').open('w', encoding='utf-8') as stream:
            result = subprocess.run([sys.executable, *arguments], stdout=stream, stderr=subprocess.STDOUT)
        print(f'END {name}: exit={result.returncode}; log={output/name}.log', flush=True)
        if result.returncode:
            return result.returncode
    results, latest = [], {}
    expected_names = {f'exp12_pilot_{args.rows}_{name}' for name in
                      ['control', 'features_only', 'logic_only', 'main', 'self_only']}
    for folder in ['exp12', 'exp12_smoke']:
        for path in (root.parents[1]/'result'/folder).glob('*/metrics.json'):
            data = json.loads(path.read_text(encoding='utf-8'))
            name = data['config']['experiment']['name']
            if (data['source_manifest'] == plan['source'] and
                    (name in expected_names or name == 'exp12_smoke')):
                key = (name, data['metrics']['strategy'], data['metrics']['seed'])
                if key not in latest or path.stat().st_mtime_ns > latest[key][0]:
                    latest[key] = (path.stat().st_mtime_ns, path, data)
    for _, path, data in latest.values():
        results.append({'run': path.parent.name, 'source': str(path), 'metrics': data['metrics'],
                        'validation_partitions': data['training']['validation_partitions']})
    (output/'pilot_summary.json').write_text(json.dumps(results, indent=2), encoding='utf-8')
    write_summaries(root.parents[1]/'result'/'exp12')
    write_summaries(root.parents[1]/'result'/'exp12_smoke')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
