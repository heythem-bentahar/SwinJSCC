"""Plot Model A (SNR-aware) vs Model B (SNR + channel-type-aware), optionally with Model C (no side information).

Usage: python plot_ab.py --a results/A.csv --b results/B.csv [--c results/C.csv] [--labels results/B_labels.csv] --out plots_ab/
"""
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

# Validated categorical slots 1 and 2 (blue, orange) from the reference palette.
COLOR_A, COLOR_B, COLOR_C = '#2a78d6', '#eb6834', '#4f8a3c'
INK, INK_MUTED, GRID, SURFACE = '#0b0b0b', '#52514e', '#e3e2dd', '#fcfcfb'
CHANNELS = ['awgn', 'rayleigh', 'rician', 'impulsive']
TITLES = {'awgn': 'AWGN', 'rayleigh': 'Rayleigh', 'rician': 'Rician (K=5)', 'impulsive': 'Impulsive'}
LABEL_A = 'SNR-aware (A)'
LABEL_B = 'SNR + channel-type-aware (B)'
LABEL_C = 'No side information (C)'
METRICS = {'PSNR_dB': ('psnr', 'PSNR (dB)'), 'MS_SSIM': ('msssim', 'MS-SSIM')}


def load(path):
    with open(path, newline='') as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r['SNR_dB'] = float(r['SNR_dB'])
        for k in ('PSNR_dB', 'MS_SSIM'):
            if k in r:
                r[k] = float(r[k])
        if 'beyond_paper' in r:
            r['beyond_paper'] = r['beyond_paper'] in ('True', 'true', '1')
    return rows


def series(rows, dataset, channel, metric):
    pts = sorted((r['SNR_dB'], r[metric], r['beyond_paper'])
                 for r in rows if r['dataset'] == dataset and r['channel'] == channel)
    return pts


def style_axes(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        ax.spines[side].set_color(INK_MUTED)
    ax.tick_params(colors=INK_MUTED, labelsize=9)


def draw_line(ax, pts, color, marker, label):
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    ax.plot(xs, ys, color=color, linewidth=2, label=label)
    inside = [p for p in pts if not p[2]]
    beyond = [p for p in pts if p[2]]
    ax.plot([p[0] for p in inside], [p[1] for p in inside], linestyle='none', marker=marker,
            markersize=6, color=color, markeredgecolor=SURFACE, markeredgewidth=1.5)
    # Unseen SNRs (beyond the paper's training range): hollow markers.
    ax.plot([p[0] for p in beyond], [p[1] for p in beyond], linestyle='none', marker=marker,
            markersize=6, markerfacecolor=SURFACE, markeredgecolor=color, markeredgewidth=1.5)


def plot_metric(rows_a, rows_b, dataset, metric, out_dir, rows_c=None):
    prefix, ylabel = METRICS[metric]
    fig, axes = plt.subplots(2, 2, figsize=(9, 6.5), sharex=True, facecolor=SURFACE)
    for ax, chan in zip(axes.flat, CHANNELS):
        style_axes(ax)
        draw_line(ax, series(rows_a, dataset, chan, metric), COLOR_A, 'o', LABEL_A)
        draw_line(ax, series(rows_b, dataset, chan, metric), COLOR_B, 's', LABEL_B)
        if rows_c:
            draw_line(ax, series(rows_c, dataset, chan, metric), COLOR_C, '^', LABEL_C)
        ax.set_title(TITLES[chan], color=INK, fontsize=11, loc='left')
        ax.set_ylabel(ylabel, color=INK_MUTED, fontsize=9)
    for ax in axes[1]:
        ax.set_xlabel('SNR (dB)', color=INK_MUTED, fontsize=9)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=3 if rows_c else 2, frameon=False, fontsize=9,
               labelcolor=INK)
    fig.text(0.5, 0.005, 'Hollow markers: SNRs outside the paper training range (beyond paper). '
             'Single training run per model.', ha='center', fontsize=8, color=INK_MUTED)
    fig.tight_layout(rect=(0, 0.03, 1, 0.94))
    fig.savefig(out_dir / '{}_vs_snr_{}.pdf'.format(prefix, dataset))
    plt.close(fig)


def plot_confusion(label_rows, out_dir):
    """PSNR change versus the correct label, averaged over SNR (rows: actual channel)."""
    grid = np.zeros((4, 4))
    for i, actual in enumerate(CHANNELS):
        for j, label in enumerate(CHANNELS):
            def mean_psnr(l):
                v = [r['PSNR_dB'] for r in label_rows
                     if r['actual_channel'] == actual and r['label_channel'] == l]
                return float(np.mean(v))
            grid[i, j] = mean_psnr(label) - mean_psnr(actual)
    limit = max(abs(grid).max(), 1e-6)
    cmap = LinearSegmentedColormap.from_list('div', [COLOR_B, '#f0efec', COLOR_A])
    fig, ax = plt.subplots(figsize=(6, 5), facecolor=SURFACE)
    im = ax.imshow(grid, cmap=cmap, vmin=-limit, vmax=limit)
    ax.set_xticks(range(4), [TITLES[c] for c in CHANNELS], rotation=30, ha='right', color=INK_MUTED, fontsize=9)
    ax.set_yticks(range(4), [TITLES[c] for c in CHANNELS], color=INK_MUTED, fontsize=9)
    ax.set_xlabel('Label given to Model B', color=INK_MUTED)
    ax.set_ylabel('Actual channel', color=INK_MUTED)
    ax.set_title('PSNR change vs. correct label (dB, mean over SNR)', color=INK, fontsize=11, loc='left')
    for i in range(4):
        for j in range(4):
            ax.text(j, i, '{:+.2f}'.format(grid[i, j]), ha='center', va='center', color=INK, fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04).ax.tick_params(colors=INK_MUTED, labelsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / 'label_confusion.pdf')
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--a', required=True)
    p.add_argument('--b', required=True)
    p.add_argument('--c', default=None, help='optional Model C (no side information) results CSV')
    p.add_argument('--labels', default=None)
    p.add_argument('--out', required=True)
    a = p.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rows_a, rows_b = load(a.a), load(a.b)
    rows_c = load(a.c) if a.c else None
    for dataset in sorted({r['dataset'] for r in rows_a} & {r['dataset'] for r in rows_b}):
        for metric in METRICS:
            plot_metric(rows_a, rows_b, dataset, metric, out, rows_c)
    if a.labels:
        plot_confusion(load(a.labels), out)
    print('wrote figures to', out)


if __name__ == '__main__':
    main()
