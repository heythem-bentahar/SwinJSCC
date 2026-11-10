"""Summarize A / B / C results across training seeds.

Reads results/<name>.csv for names snr_aware, snr_type_aware, no_info (seed 0) and the *_s1, *_s2, ...
variants (extra seeds), then prints and writes results/seed_summary.csv.

Usage: python aggregate_seeds.py [--results results] [--snr-min 1] [--snr-max 13]
"""
import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

FAMILIES = {'snr_aware': 'A', 'snr_type_aware': 'B', 'no_info': 'C'}
NAME_RE = re.compile(r'^(snr_aware|snr_type_aware|no_info)(?:_s(\d+))?$')
CHANNELS = ['awgn', 'rayleigh', 'rician', 'impulsive']


def load_runs(results_dir):
    """Return {(model, seed): rows} for every matching CSV."""
    runs = {}
    for path in sorted(Path(results_dir).glob('*.csv')):
        m = NAME_RE.match(path.stem)
        if not m:
            continue
        seed = int(m.group(2)) if m.group(2) else 0
        with open(path, newline='') as f:
            rows = list(csv.DictReader(f))
        for r in rows:
            r['SNR_dB'] = float(r['SNR_dB'])
            r['PSNR_dB'] = float(r['PSNR_dB'])
            r['MS_SSIM'] = float(r['MS_SSIM'])
        runs[(FAMILIES[m.group(1)], seed)] = rows
    return runs


def run_mean(rows, dataset, channel, snr_min, snr_max, metric='PSNR_dB'):
    v = [r[metric] for r in rows if r['dataset'] == dataset and r['channel'] == channel
         and snr_min <= r['SNR_dB'] <= snr_max]
    return float(np.mean(v)) if v else float('nan')


def fmt(values):
    values = np.asarray(values, dtype=float)
    if len(values) < 2:
        return '{:.2f}'.format(values[0])
    return '{:.2f} +/- {:.2f}'.format(values.mean(), values.std(ddof=1))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--results', default='results')
    p.add_argument('--snr-min', type=float, default=1.0)
    p.add_argument('--snr-max', type=float, default=13.0)
    p.add_argument('--per-seed', action='store_true', help='also print each seed\'s paired difference')
    a = p.parse_args()

    runs = load_runs(a.results)
    seeds = {m: sorted(s for (mm, s) in runs if mm == m) for m in 'ABC'}
    for m in 'ABC':
        print('Model {}: seeds {}'.format(m, seeds[m] or 'none found'))
    common = sorted(set(seeds['A']) & set(seeds['B']) & set(seeds['C']))
    print('Seeds available for all three models: {}'.format(common))
    print('PSNR averaged over SNR {:g}..{:g} dB, mean +/- std (sample, ddof=1) across seeds.\n'.format(
        a.snr_min, a.snr_max))

    out_rows = []
    print('Differences are PAIRED: computed per seed, then mean +/- sample std of those per-seed differences.')
    print('(Seeds share sampler/noise/order seeds, so each seed is a matched pair.)\n')
    header = '{:8s} {:10s} {:>15s} {:>15s} {:>15s}   {:>22s} {:>22s} {:>22s}'.format(
        'dataset', 'channel', 'A', 'B', 'C', 'B-A (wins)', 'A-C (wins)', 'B-C (wins)')
    print(header)
    for dataset in ('cifar10', 'kodak'):
        for channel in CHANNELS:
            per = {m: [run_mean(runs[(m, s)], dataset, channel, a.snr_min, a.snr_max)
                       for s in common] for m in 'ABC'}
            if not common or any(np.isnan(per[m]).any() for m in 'ABC'):
                continue
            diffs = {'B-A': np.array(per['B']) - np.array(per['A']),
                     'A-C': np.array(per['A']) - np.array(per['C']),
                     'B-C': np.array(per['B']) - np.array(per['C'])}

            def d(k):
                x = diffs[k]
                sd = x.std(ddof=1) if len(x) > 1 else float('nan')
                return '{:+.2f} +/- {:.2f} ({}/{})'.format(x.mean(), sd, int((x > 0).sum()), len(x))
            print('{:8s} {:10s} {:>15s} {:>15s} {:>15s}   {:>22s} {:>22s} {:>22s}'.format(
                dataset, channel, fmt(per['A']), fmt(per['B']), fmt(per['C']),
                d('B-A'), d('A-C'), d('B-C')))
            if a.per_seed:
                for k in ('B-A', 'A-C', 'B-C'):
                    print('           per-seed {} (seeds {}): {}'.format(
                        k, common, ' '.join('{:+.2f}'.format(v) for v in diffs[k])))
            out_rows.append(dict(
                dataset=dataset, channel=channel, n_seeds=len(common),
                A_mean=np.mean(per['A']), A_std=np.std(per['A'], ddof=1) if len(common) > 1 else '',
                B_mean=np.mean(per['B']), B_std=np.std(per['B'], ddof=1) if len(common) > 1 else '',
                C_mean=np.mean(per['C']), C_std=np.std(per['C'], ddof=1) if len(common) > 1 else '',
                BminusA=diffs['B-A'].mean(), BminusA_std=diffs['B-A'].std(ddof=1) if len(common) > 1 else '',
                BminusA_wins=int((diffs['B-A'] > 0).sum()),
                AminusC=diffs['A-C'].mean(), AminusC_std=diffs['A-C'].std(ddof=1) if len(common) > 1 else '',
                AminusC_wins=int((diffs['A-C'] > 0).sum()),
                BminusC=diffs['B-C'].mean(), BminusC_std=diffs['B-C'].std(ddof=1) if len(common) > 1 else '',
                BminusC_wins=int((diffs['B-C'] > 0).sum())))

    if out_rows:
        path = Path(a.results) / 'seed_summary.csv'
        with open(path, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
            w.writeheader()
            w.writerows(out_rows)
        print('\nwrote', path)
    print('"(k/n)" = number of seeds in which the first model is ahead. The +/- is the spread of the paired '
          'per-seed differences (n is small, so it is a rough guide, not a significance test).')


if __name__ == '__main__':
    main()
