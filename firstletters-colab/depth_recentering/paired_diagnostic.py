"""Post-hoc paired-reference diagnostic; not a deployable recentering method.

After the frozen cross-patch benchmark failed, this asks whether known injected
offsets are recoverable when each validation patch is given its OWN unshifted
profile. That profile would not be known for an actually displaced mesh.
"""

import argparse
from hashlib import sha256
import json
from pathlib import Path

import numpy as np

from benchmark import bounded_match, summarize


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    original = json.loads(args.results.read_text())
    records = []
    for cell in original['cells']:
        if cell['split'] != 'validation':
            continue
        native = np.array(cell['native_profile'])
        reference = native[4:24]
        own_peak = int(np.argmax(reference))
        for injected in range(-4, 5):
            observed = native[4 + injected:24 + injected]
            records.append({
                'cell': cell['id'], 'injected_offset_planes': injected,
                'truth_correction_planes': -injected,
                'paired_peak': {'accepted': True, 'correction_planes':
                                float(np.argmax(observed) - own_peak)},
                'paired_bounded_template': bounded_match(
                    observed, reference, cell['anatomically_eligible']),
            })
    methods = ['paired_peak', 'paired_bounded_template']
    report = {
        'diagnostic': 'post_hoc_own_unshifted_reference',
        'pre_registered': False,
        'reason_added': 'Distinguish cross-patch profile variability from an indexing/sign failure after the frozen benchmark failed.',
        'unavailable_information_supplied': 'Each held-out patch receives its own unshifted profile as the reference. This is not available when the true mesh displacement is unknown.',
        'original_results_sha256': sha256(args.results.read_bytes()).hexdigest(),
        'source_array_bytes_sha256': original['source_array_bytes_sha256'],
        'relative_sign_convention': 'Observed source crop [4+d,24+d) has corrective offset -d; reference band is [4,24), not a known physical recto.',
        'summary': {m: summarize(records, m) for m in methods},
        'per_offset': {str(d): {m: summarize([r for r in records if
                        r['injected_offset_planes'] == d], m) for m in methods}
                       for d in range(-4, 5)},
        'cases': records,
        'not_evidence_for_deployment': True,
    }
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps({k: report[k] for k in ['diagnostic', 'pre_registered',
                    'relative_sign_convention', 'summary']}, indent=2))


if __name__ == '__main__':
    main()
