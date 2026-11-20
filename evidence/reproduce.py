"""Verify a public Vesuvius fixture, compare tile selection and optionally run Villa."""

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess
import sys

import zarr

from fetch_fixture import fetch

HERE = Path(__file__).resolve().parent
VILLA_BASE = 'c8f3ba45304f0a317a49a155929e30d294f201eb'


def run_json(command, destination, environment):
    result = subprocess.run(command, env=environment, capture_output=True, text=True)
    if result.returncode:
        sys.stderr.write(result.stderr + '\n' + result.stdout)
        raise SystemExit(result.returncode)
    # Villa writes the model configuration before the final JSON report.
    output = result.stdout.strip()
    start = output.rfind('\n{') + 1
    report = json.loads(output[start:])
    destination.write_text(json.dumps(report, indent=2) + '\n')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('case', choices=('w00', 'w058'))
    parser.add_argument('--villa', required=True, type=Path, help='patched Villa checkout')
    parser.add_argument('--baseline', type=Path, help='clean Villa checkout for --native')
    parser.add_argument('--work', required=True, type=Path)
    parser.add_argument('--cache', type=Path, default=Path('scroll-data-cache'))
    parser.add_argument('--native', action='store_true')
    parser.add_argument('--runs', type=int, default=5)
    parser.add_argument('--project', type=Path, default=HERE.parent,
                        help='directory with benchmark.py and compare_native.py')
    args = parser.parse_args()
    if args.native and not args.baseline:
        parser.error('--native requires --baseline')
    if args.runs < 1:
        parser.error('runs must be positive')
    if not (args.project / 'benchmark.py').is_file():
        parser.error('put the evidence directory inside the release checkout or pass --project')
    for checkout in (args.villa, args.baseline):
        if checkout is None:
            continue
        revision = subprocess.check_output(
            ['git', '-C', str(checkout), 'rev-parse', 'HEAD'], text=True
        ).strip()
        if revision != VILLA_BASE:
            parser.error(f'{checkout} must be at pinned Villa commit {VILLA_BASE}')

    args.work.mkdir(parents=True, exist_ok=True)
    volume = args.work / f'{args.case}.zarr'
    manifest = json.loads((HERE / 'sources.json').read_text())
    case = manifest['cases'][args.case]
    environment = os.environ.copy()
    environment.update(OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
    if not volume.exists():
        subprocess.run(
            [sys.executable, str(HERE / 'fetch_fixture.py'), args.case, str(volume),
             '--cache', str(args.cache)], env=environment, check=True
        )
    group = zarr.open_group(str(volume), mode='r')
    expected = sha256(json.dumps(case, sort_keys=True).encode()).hexdigest()
    if group.attrs.get('source_manifest_sha256') != expected:
        parser.error('existing fixture uses a different manifest; choose a new work directory')
    roi = list(map(str, case['roi_yx']))
    schedule = run_json(
        [sys.executable, str(args.project / 'benchmark.py'), str(args.villa), str(volume),
         '--roi', *roi, '--runs', str(args.runs)],
        args.work / f'{args.case}-selection.json', environment
    )
    stages = schedule['runs']
    print(f"{args.case}: identical schedules across {args.runs} runs; "
          f"chunks {stages['before']['chunk_positions']} -> {stages['after']['chunk_positions']}",
          flush=True)
    if not args.native:
        return

    model = manifest['model']
    checkpoint = fetch(model['url'], model['sha256'], args.cache / 'reader-v3.pth')
    timing = run_json(
        [sys.executable, str(HERE / 'native_runs.py'), str(args.baseline), str(args.villa),
         args.case, str(volume), '--runs', str(args.runs)],
        args.work / f'{args.case}-native-timing.json', environment
    )
    comparison = run_json(
        [sys.executable, str(args.project / 'compare_native.py'), str(args.baseline),
         str(args.villa), str(volume), str(checkpoint), '--roi', *roi,
         '--direction', case['direction']],
        args.work / f'{args.case}-native-tiff.json', environment
    )
    before = timing['stages']['before']['median_seconds']
    after = timing['stages']['after']['median_seconds']
    print(f"{args.case}: native planning medians {before:.6f} -> {after:.6f} s; "
          f"different pixels {comparison['different_pixels']}/{comparison['pixels']}", flush=True)


if __name__ == '__main__':
    main()
