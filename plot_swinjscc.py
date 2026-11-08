"""Plot the organized SwinJSCC CSV. Requires matplotlib only.
Usage: python plot_swinjscc.py [results.csv] --out plots
Default CSV: SwinJSCC_results_organized.csv beside this script.
"""
import argparse
import csv
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

COLORS = {'small': '#2563eb', 'base': '#d97706', 'large': '#059669'}
FRACTIONS = {32: '1/48', 64: '1/24', 96: '1/16', 128: '1/12', 192: '1/8'}
METRICS = [('PSNR_dB', 'PSNR (dB)'), ('MS_SSIM', 'MS-SSIM')]


def load_results(path):
    with path.open(newline='', encoding='utf-8-sig') as handle:
        reader = csv.DictReader(handle)
        required = {'checkpoint', 'model_size', 'channel', 'SNR_dB', 'C', 'CBR', 'PSNR_dB', 'MS_SSIM'}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError('CSV must contain: ' + ', '.join(sorted(required)))
        rows, seen = [], set()
        for line, row in enumerate(reader, 2):
            for key in ['SNR_dB', 'CBR', 'PSNR_dB', 'MS_SSIM']:
                row[key] = float(row[key])
            row['C'] = int(row['C'])
            key = (row['checkpoint'], row['SNR_dB'], row['C'])
            if key in seen:
                raise ValueError(f'Duplicate setting on CSV line {line}: {key}')
            seen.add(key)
            row['prefixed'] = row['checkpoint'].startswith('pretrained_')
            rows.append(row)
    if not rows:
        raise ValueError('CSV contains no measurements.')
    return rows


def select(rows, **criteria):
    return [r for r in rows if all(r[k] == v for k, v in criteria.items())]


def curve(ax, rows, xkey, metric, label, color):
    if not rows:
        return
    rows = sorted(rows, key=lambda r: r[xkey])
    xs = [r[xkey] for r in rows]
    if len(xs) != len(set(xs)):
        raise ValueError(f'Ambiguous curve: {label}. Filter to one checkpoint/condition.')
    ax.plot(xs, [r[metric] for r in rows], 'o-', label=label, color=color, linewidth=2, markersize=5)


def decorate(ax, xlabel, ylabel, rate=False, snrs=None):
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=.25)
    ax.margins(x=.08, y=.12)
    if rate:
        ax.set_xticks([c / 1536 for c in FRACTIONS], list(FRACTIONS.values()))
    elif snrs:
        ax.set_xticks(snrs)
    if ax.lines:
        ax.legend(fontsize=9)
    else:
        ax.text(.5, .5, 'No matching measurements', ha='center', transform=ax.transAxes)


def save(fig, out, name, title):
    fig.suptitle(title, fontsize=16, fontweight='bold')
    fig.text(.5, .012, 'Measured Kodak results only; no paper reference curves or error bars', ha='center', fontsize=9, color='#64748b')
    fig.tight_layout(rect=(0, .04, 1, .94))
    for ext in ['png', 'pdf']:
        path = out / f'{name}.{ext}'
        fig.savefig(path, dpi=200, bbox_inches='tight')
        print(path.resolve())
    plt.close(fig)


def plot_all(rows, out):
    # Base model: same CBR permits a controlled channel comparison.
    base = select(rows, model_size='base', prefixed=False)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, (metric, ylabel) in zip(axes, METRICS):
        for channel, color in [('AWGN', '#2563eb'), ('Rayleigh', '#d97706')]:
            curve(ax, select(base, channel=channel, C=96), 'SNR_dB', metric, channel, color)
        decorate(ax, 'SNR (dB)', ylabel, snrs=[1, 4, 7, 10, 13])
    save(fig, out, 'base_snr_channels', 'Base model: SNR sweep at CBR = 1/16')

    # Separate rate panels: AWGN and Rayleigh were tested at different SNRs.
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for col, (channel, snr, color) in enumerate([('AWGN', 10., '#2563eb'), ('Rayleigh', 3., '#d97706')]):
        for row, (metric, ylabel) in enumerate(METRICS):
            ax = axes[row, col]
            curve(ax, select(base, channel=channel, SNR_dB=snr), 'CBR', metric, channel, color)
            decorate(ax, 'CBR', ylabel, rate=True)
            ax.set_title(f'{channel}: SNR = {snr:g} dB')
    save(fig, out, 'base_rate_sweeps', 'Base model: transmission budget sweeps (different fixed SNRs)')

    final_awgn = select(rows, channel='AWGN', prefixed=False)
    # Three model sizes, at each of the three measured SNR levels.
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for col, snr in enumerate([1., 7., 13.]):
        for row, (metric, ylabel) in enumerate(METRICS):
            ax = axes[row, col]
            for size, color in COLORS.items():
                subset = [r for r in select(final_awgn, model_size=size, SNR_dB=snr) if r['C'] in [32, 96, 192]]
                curve(ax, subset, 'CBR', metric, size.title(), color)
            decorate(ax, 'CBR', ylabel)
            ax.set_xticks([c/1536 for c in [32,96,192]], ['1/48','1/16','1/8'])
            ax.set_title(f'SNR = {snr:g} dB')
    save(fig, out, 'model_sizes_vs_rate', 'Small / base / large: AWGN rate comparison')

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for col, c in enumerate([32, 96, 192]):
        for row, (metric, ylabel) in enumerate(METRICS):
            ax = axes[row, col]
            for size, color in COLORS.items():
                subset = [r for r in select(final_awgn, model_size=size, C=c) if r['SNR_dB'] in [1.,7.,13.]]
                curve(ax, subset, 'SNR_dB', metric, size.title(), color)
            decorate(ax, 'SNR (dB)', ylabel, snrs=[1,7,13])
            ax.set_title(f'CBR = {FRACTIONS[c]}')
    save(fig, out, 'model_sizes_vs_snr', 'Small / base / large: AWGN SNR comparison')

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    small = select(rows, model_size='small', channel='AWGN')
    for col in range(2):
        for row, (metric, ylabel) in enumerate(METRICS):
            ax = axes[row, col]
            for prefix, label, color in [(True,'With pretrained_ prefix','#9333ea'), (False,'Without pretrained_ prefix','#2563eb')]:
                subset = select(small, prefixed=prefix, **({'C':96} if col==0 else {'SNR_dB':10.}))
                curve(ax, subset, 'SNR_dB' if col==0 else 'CBR', metric, label, color)
            decorate(ax, 'SNR (dB)' if col==0 else 'CBR', ylabel, rate=col==1, snrs=[1,4,7,10,13] if col==0 else None)
            ax.set_title('CBR = 1/16' if col==0 else 'SNR = 10 dB')
    save(fig, out, 'small_checkpoint_comparison', 'Small AWGN: checkpoint comparison (training stages unverified)')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('csv', nargs='?', type=Path, default=Path(__file__).resolve().with_name('SwinJSCC_results.csv'))
    parser.add_argument('--out', type=Path, default=Path('plots'))
    args = parser.parse_args()
    rows = load_results(args.csv)
    args.out.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    plot_all(rows, args.out)
    print(f'Done: {len(rows)} measurements; 5 figures in PNG and PDF.')


if __name__ == '__main__':
    main()
