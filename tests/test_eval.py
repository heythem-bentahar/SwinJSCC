import numpy as np
import pytest
import torch
from PIL import Image
from torch.utils.data import DataLoader, TensorDataset

import eval_channels as ev
from tests.helpers import make_net, images

TYPES = ['awgn', 'rayleigh', 'rician', 'impulsive']


def tiny_loader(n=8):
    x = images(n, seed=3)
    return DataLoader(TensorDataset(x.cpu(), torch.zeros(n)), batch_size=4)


def test_evaluate_returns_psnr_and_msssim():
    net = make_net(type_aware=True).eval()
    psnr, msssim = ev.evaluate(net, tiny_loader(), 'awgn', 7)
    assert 5 < psnr < 60
    assert 0 < msssim <= 1


def test_rows_cover_grid():
    net = make_net(type_aware=False).eval()
    rows = ev.run_eval(net, 'A', tiny_loader(), None)
    assert len(rows) == 4 * 7
    assert {(r['channel'], r['SNR_dB']) for r in rows} == {(c, s) for c in TYPES for s in ev.EVAL_SNRS}
    assert {r['SNR_dB'] for r in rows if r['beyond_paper']} == {-2, 16}
    assert all(r['dataset'] == 'cifar10' and r['model'] == 'A' for r in rows)


def test_kodak_crops_to_valid_size(tmp_path):
    Image.fromarray(np.random.randint(0, 255, (375, 500, 3), dtype=np.uint8)).save(tmp_path / 'k1.png')
    ds = ev.KodakFolder(str(tmp_path))
    img, name = ds[0]
    assert name == 'k1.png'
    assert img.shape == (3, 368, 496)
    net = make_net(type_aware=False).eval()
    psnr, _ = ev.evaluate(net, DataLoader(ds, batch_size=1), 'awgn', 7)
    assert psnr > 0


def test_eval_deterministic_with_noise_seed():
    net = make_net(type_aware=True).eval()
    a = ev.evaluate(net, tiny_loader(), 'rayleigh', 4)
    b = ev.evaluate(net, tiny_loader(), 'rayleigh', 4)
    assert a == b


def test_empty_kodak_dir_raises(tmp_path):
    with pytest.raises(ValueError):
        ev.KodakFolder(str(tmp_path))


def test_label_sweep_has_16_pairs_per_snr():
    net = make_net(type_aware=True).eval()
    rows = ev.label_sweep(net, tiny_loader())
    assert len(rows) == 16 * 5
    for snr in ev.PAPER_SNRS:
        pairs = {(r['actual_channel'], r['label_channel']) for r in rows if r['SNR_dB'] == snr}
        assert pairs == {(a, l) for a in TYPES for l in TYPES}


def test_diagonal_equals_normal_eval():
    net = make_net(type_aware=True).eval()
    rows = ev.label_sweep(net, tiny_loader())
    for r in rows:
        if r['actual_channel'] == r['label_channel'] and r['SNR_dB'] == 7:
            psnr, msssim = ev.evaluate(net, tiny_loader(), r['actual_channel'], 7)
            assert abs(r['PSNR_dB'] - psnr) < 1e-6
            assert abs(r['MS_SSIM'] - msssim) < 1e-6


def test_label_sweep_refused_for_model_a():
    net = make_net(type_aware=False).eval()
    with pytest.raises(ValueError, match='type-aware'):
        ev.label_sweep(net, tiny_loader())
