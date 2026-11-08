import csv
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
BASE = [sys.executable, 'main.py', '--training', '--trainset', 'CIFAR10',
        '--model', 'SwinJSCC_w/_SA', '--channel-type', 'mixed', '--C', '32',
        '--multiple-snr', '1,4,7,10,13', '--epochs', '1',
        '--train-subset', '256', '--val-subset', '64']


@pytest.fixture(scope='module')
def runs():
    names = {'A': 'smoke_snr_aware', 'B': 'smoke_snr_type_aware'}
    for n in names.values():
        shutil.rmtree(ROOT / 'history' / n, ignore_errors=True)
    for key, name in names.items():
        extra = ['--type-aware'] if key == 'B' else []
        r = subprocess.run(BASE + extra + ['--run-name', name], cwd=ROOT,
                           capture_output=True, text=True)
        assert r.returncode == 0, r.stderr[-2000:]
    yield {k: ROOT / 'history' / v for k, v in names.items()}
    for v in names.values():
        shutil.rmtree(ROOT / 'history' / v, ignore_errors=True)


def read(p):
    with open(p, newline='') as f:
        return list(csv.reader(f))


def test_sampler_logs_identical(runs):
    a = read(runs['A'] / 'sampler_log.csv')
    b = read(runs['B'] / 'sampler_log.csv')
    assert a == b
    assert len(a) == 1 + 2  # header + 256/128 training steps; validation drew nothing


def test_checkpoints_exist(runs):
    for r in runs.values():
        assert (r / 'models' / 'best.model').exists()
        assert (r / 'models' / 'last.model').exists()


def test_val_log_has_per_channel_columns(runs):
    rows = read(runs['A'] / 'val_log.csv')
    assert rows[0] == ['epoch', 'mean_psnr', 'psnr_awgn', 'psnr_rayleigh', 'psnr_rician', 'psnr_impulsive']
    assert len(rows) == 2
