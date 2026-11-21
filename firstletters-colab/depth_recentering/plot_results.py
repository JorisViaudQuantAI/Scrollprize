"""Draw a standard scientific plot from the released-control result JSON."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    data = json.loads(args.results.read_text())
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.3), layout='constrained',
                             gridspec_kw={'width_ratios': [1.1, 1]})
    colors = {'calibration': '#d87716', 'validation': '#286ba6'}
    for cell in data['cells']:
        axes[0].plot((np.arange(28) - 13.5) * 9.366, cell['native_profile'],
                     color=colors[cell['split']], alpha=0.45, linewidth=1)
    axes[0].axvline(0, color='#555555', linestyle='--', linewidth=1)
    axes[0].set(xlabel='Relative depth from stored band center (µm)',
                ylabel='Trimmed mean CT intensity (uint8 scale)',
                title='Profiles vary across the same released surface')
    axes[0].legend(handles=[Patch(color=v, label=k.title() + ' cells')
                           for k, v in colors.items()], frameon=False)
    axes[0].grid(alpha=0.15)
    cells = [c['id'] for c in data['cells'] if c['split'] == 'validation']
    matrix = np.zeros((12, 9))
    for case in data['cases']:
        if case['split'] != 'validation':
            continue
        method = case['bounded_template']
        category = 0 if not method['accepted'] else (
            1 if abs(method['correction_planes'] - case['truth_correction_planes']) <= 1 else 2)
        matrix[cells.index(case['cell']), case['injected_offset_planes'] + 4] = category
    palette = ['#e3e6e8', '#27826b', '#b83e3e']
    axes[1].imshow(matrix, cmap=ListedColormap(palette),
                   norm=BoundaryNorm([-0.5, 0.5, 1.5, 2.5], 3), aspect='auto')
    axes[1].set_xticks(range(9), range(-4, 5))
    axes[1].set_yticks(range(12), cells)
    axes[1].set(xlabel='Injected offset in native surface planes', ylabel='Held-out cell',
                title='Frozen bounded method: 34/108 accepted')
    axes[1].legend(handles=[Patch(color=color, label=label) for color, label in zip(
        palette, ['Abstained (74)', 'Accepted, error ≤1 (25)', 'Accepted, error >1 (9)'])],
        frameon=False, loc='upper center', bbox_to_anchor=(0.5, -0.13), ncols=1)
    fig.suptitle('PHerc0841 control: template recentering failed the continuation gate',
                 fontsize=14, weight='bold')
    fig.savefig(args.output, dpi=160)
    plt.close(fig)


if __name__ == '__main__':
    main()
