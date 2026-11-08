"""Evaluate trained SwinJSCC models (CIFAR10 test and optional Kodak) per channel and SNR.

Usage:
  python eval_channels.py --checkpoint history/snr_aware/models/best.model --run-name snr_aware \
      --type-aware 0 --out results/snr_aware.csv [--kodak-dir DIR] [--label-sweep]
"""
import argparse
import csv
import math
import types
from glob import glob
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

from loss.distortion import MS_SSIM
from net.network import SwinJSCC

CHAN_TYPES = ['awgn', 'rayleigh', 'rician', 'impulsive']
PAPER_SNRS = [1, 4, 7, 10, 13]
EVAL_SNRS = [-2, 1, 4, 7, 10, 13, 16]
BEYOND_PAPER = {-2, 16}
NOISE_SEED = 2024


def build_net(type_aware, C=32, model='SwinJSCC_w/_SA'):
    """CIFAR10-configuration SwinJSCC on CUDA (same hyper-parameters as main.py)."""
    common = dict(model=model, img_size=(32, 32), C=C, window_size=2, mlp_ratio=4.,
                  qkv_bias=True, qk_scale=None, norm_layer=nn.LayerNorm, patch_norm=True)
    config = types.SimpleNamespace(
        encoder_kwargs=dict(common, patch_size=2, in_chans=3, embed_dims=[128, 256],
                            depths=[2, 4], num_heads=[4, 8]),
        decoder_kwargs=dict(common, embed_dims=[256, 128], depths=[4, 2], num_heads=[8, 4]),
        logger=None, pass_channel=True, norm=False, downsample=2,
        device=torch.device('cuda:0'), CUDA=True)
    args = types.SimpleNamespace(
        channel_type='mixed', type_aware=type_aware, multiple_snr='1,4,7,10,13',
        C=str(C), model=model, distortion_metric='MSE', trainset='CIFAR10')
    return SwinJSCC(args, config).cuda()


class KodakFolder(Dataset):
    """Images from a folder, center-cropped so both sides are multiples of 8."""

    def __init__(self, directory, multiple=8):
        self.paths = sorted(glob(str(Path(directory) / '*.png')) + glob(str(Path(directory) / '*.jpg')))
        if not self.paths:
            raise ValueError('No PNG/JPG images found in {}'.format(directory))
        self.multiple = multiple

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = transforms.functional.to_tensor(Image.open(self.paths[i]).convert('RGB'))
        _, h, w = img.shape
        m = self.multiple
        nh, nw = h // m * m, w // m * m
        top, left = (h - nh) // 2, (w - nw) // 2
        return img[:, top:top + nh, left:left + nw], Path(self.paths[i]).name


def cifar_test_loader(root='./CIFAR10/', batch_size=1024):
    ds = datasets.CIFAR10(root=root, train=False, transform=transforms.ToTensor(), download=False)
    return DataLoader(ds, batch_size=batch_size, shuffle=False)


def evaluate(net, loader, channel, snr, label=None, noise_seed=NOISE_SEED):
    """Mean per-image PSNR (0-255 scale) and MS-SSIM for one (applied channel, SNR).
    `label` is the channel label fed to the network (default: the applied channel)."""
    net.eval()
    net.channel.set_noise_seed(noise_seed)
    first = next(iter(loader))[0]
    small = first.shape[-1] <= 64
    ms = (MS_SSIM(window_size=3, data_range=1., levels=4, channel=3) if small
          else MS_SSIM(data_range=1., levels=4, channel=3)).cuda()
    psnrs, msssims = [], []
    with torch.no_grad():
        for batch in loader:
            x = batch[0].cuda()
            recon = net(x, snr, int(net.channel_number[0]), channel, label)[0].clamp(0., 1.)
            mse = ((x * 255. - recon * 255.) ** 2).mean(dim=(1, 2, 3))
            psnrs.extend((10 * torch.log10(255. ** 2 / mse)).tolist())
            msssims.extend((1 - ms(x, recon)).tolist())
    return sum(psnrs) / len(psnrs), sum(msssims) / len(msssims)


def run_eval(net, model_name, cifar_loader, kodak_loader):
    rows = []
    for dataset, loader in (('cifar10', cifar_loader), ('kodak', kodak_loader)):
        if loader is None:
            continue
        for chan in CHAN_TYPES:
            for snr in EVAL_SNRS:
                psnr, msssim = evaluate(net, loader, chan, snr)
                rows.append(dict(model=model_name, dataset=dataset, channel=chan, SNR_dB=snr,
                                 beyond_paper=snr in BEYOND_PAPER, PSNR_dB=psnr, MS_SSIM=msssim))
    return rows


def write_csv(path, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--run-name', required=True)
    p.add_argument('--type-aware', type=int, choices=[0, 1], required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--kodak-dir', default=None)
    p.add_argument('--cifar-root', default='./CIFAR10/')
    p.add_argument('--label-sweep', action='store_true')
    a = p.parse_args()
    net = build_net(bool(a.type_aware))
    net.load_state_dict(torch.load(a.checkpoint), strict=True)
    cifar = cifar_test_loader(a.cifar_root)
    rows = run_eval(net, a.run_name, cifar, None)
    if a.label_sweep:
        label_rows = label_sweep(net, cifar)
        write_csv(str(Path(a.out).with_name(a.run_name + '_labels.csv')), label_rows)
    if a.kodak_dir:
        kodak = DataLoader(KodakFolder(a.kodak_dir), batch_size=1, shuffle=False)
        rows += [r for r in run_eval(net, a.run_name, None, kodak)]
    write_csv(a.out, rows)
    print('wrote', a.out)


def label_sweep(net, loader):
    """PSNR/MS-SSIM for every (actual channel, supplied label) pair at the paper SNRs.
    The applied channel stays `actual`; only the label fed to the network changes."""
    if not net.type_aware:
        raise ValueError('label sweep needs a type-aware model (Model B)')
    rows = []
    for actual in CHAN_TYPES:
        for label in CHAN_TYPES:
            for snr in PAPER_SNRS:
                psnr, msssim = evaluate(net, loader, actual, snr, label)
                rows.append(dict(actual_channel=actual, label_channel=label, SNR_dB=snr,
                                 PSNR_dB=psnr, MS_SSIM=msssim))
    return rows


if __name__ == '__main__':
    main()
