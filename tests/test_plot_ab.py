import csv
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TYPES = ['awgn', 'rayleigh', 'rician', 'impulsive']
SNRS = [-2, 1, 4, 7, 10, 13, 16]


def write(path, rows):
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def eval_rows(model, offset):
    return [dict(model=model, dataset=d, channel=c, SNR_dB=s, beyond_paper=s in (-2, 16),
                 PSNR_dB=20 + s * 0.3 + offset, MS_SSIM=0.7 + s * 0.01)
            for d in ('cifar10', 'kodak') for c in TYPES for s in SNRS]


def test_plot_ab_writes_all_figures(tmp_path):
    write(tmp_path / 'a.csv', eval_rows('A', 0.0))
    write(tmp_path / 'b.csv', eval_rows('B', 0.5))
    labels = [dict(actual_channel=a, label_channel=l, SNR_dB=s, PSNR_dB=25 - (a != l) * 2, MS_SSIM=0.9)
              for a in TYPES for l in TYPES for s in (1, 4, 7, 10, 13)]
    write(tmp_path / 'labels.csv', labels)
    out = tmp_path / 'out'
    r = subprocess.run([sys.executable, 'plot_ab.py', '--a', str(tmp_path / 'a.csv'),
                        '--b', str(tmp_path / 'b.csv'), '--labels', str(tmp_path / 'labels.csv'),
                        '--out', str(out)], cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-1500:]
    for name in ('psnr_vs_snr_cifar10.pdf', 'psnr_vs_snr_kodak.pdf', 'msssim_vs_snr_cifar10.pdf',
                 'msssim_vs_snr_kodak.pdf', 'label_confusion.pdf'):
        assert (out / name).exists() and (out / name).stat().st_size > 1000, name


def test_plot_ab_without_labels(tmp_path):
    write(tmp_path / 'a.csv', eval_rows('A', 0.0))
    write(tmp_path / 'b.csv', eval_rows('B', 0.5))
    out = tmp_path / 'out'
    r = subprocess.run([sys.executable, 'plot_ab.py', '--a', str(tmp_path / 'a.csv'),
                        '--b', str(tmp_path / 'b.csv'), '--out', str(out)],
                       cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-1500:]
    assert not (out / 'label_confusion.pdf').exists()
