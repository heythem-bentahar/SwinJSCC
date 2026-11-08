import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent
BASE = [sys.executable, 'main.py', '--training', '--trainset', 'CIFAR10',
        '--model', 'SwinJSCC_w/_SA', '--channel-type', 'mixed', '--C', '32',
        '--multiple-snr', '1,4,7,10,13', '--train-subset', '256', '--val-subset', '64',
        '--type-aware']
FULL, SPLIT, EMPTY = 'smoke_resume_full', 'smoke_resume_split', 'smoke_resume_none'


def run(name, epochs, resume=False):
    cmd = BASE + ['--run-name', name, '--epochs', str(epochs)] + (['--resume'] if resume else [])
    return subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)


@pytest.fixture(scope='module')
def runs():
    for n in (FULL, SPLIT, EMPTY):
        shutil.rmtree(ROOT / 'history' / n, ignore_errors=True)
    assert run(FULL, 2).returncode == 0
    assert run(SPLIT, 1).returncode == 0
    r = run(SPLIT, 2, resume=True)
    assert r.returncode == 0, r.stderr[-2000:]
    yield
    for n in (FULL, SPLIT, EMPTY):
        shutil.rmtree(ROOT / 'history' / n, ignore_errors=True)


def read(name, f):
    return (ROOT / 'history' / name / f).read_text()


def test_resume_sampler_log_matches_uninterrupted(runs):
    assert read(FULL, 'sampler_log.csv') == read(SPLIT, 'sampler_log.csv')


def test_resume_val_log_matches_uninterrupted(runs):
    assert read(FULL, 'val_log.csv') == read(SPLIT, 'val_log.csv')


def test_resume_final_weights_match_uninterrupted(runs):
    a = torch.load(ROOT / 'history' / FULL / 'models' / 'last.model')
    b = torch.load(ROOT / 'history' / SPLIT / 'models' / 'last.model')
    assert a.keys() == b.keys()
    for k in a:
        assert torch.equal(a[k], b[k]), k


def test_resume_without_checkpoint_fails_clearly():
    shutil.rmtree(ROOT / 'history' / EMPTY, ignore_errors=True)
    r = run(EMPTY, 1, resume=True)
    assert r.returncode != 0
    assert 'resume' in (r.stderr + r.stdout).lower()
    shutil.rmtree(ROOT / 'history' / EMPTY, ignore_errors=True)
