"""Sanity-check the impulsive channel against AWGN (CPU only, a few seconds).

Prints, per SNR, the measured noise variance and excess kurtosis of each channel and the fraction of
samples beyond 4 standard deviations. The impulsive channel keeps AWGN's total noise power, but it should
be heavy-tailed (kurtosis well above 3). If it is not, the implementation or its constants need a look.

Usage: python check_impulsive.py
"""
import types

import numpy as np
import torch

from net.channel import Channel, IMPULSE_P, IMPULSE_RATIO


def stats(chan_type, snr, n=2_000_000, seed=0):
    args = types.SimpleNamespace(channel_type=chan_type, multiple_snr=str(snr))
    config = types.SimpleNamespace(device=torch.device('cpu'), logger=None)
    ch = Channel(args, config)
    ch.set_noise_seed(seed)
    x = torch.zeros(n, dtype=torch.complex64)
    noise = ch.complex_forward(x, snr)  # input is zero, so the output is the noise itself
    z = torch.cat([noise.real, noise.imag]).double().numpy()
    var = z.var()
    kurt = np.mean((z - z.mean()) ** 4) / var ** 2
    tail = np.mean(np.abs(z) > 4 * np.sqrt(var))
    return var, kurt, tail


if __name__ == '__main__':
    print('IMPULSE_P={}, IMPULSE_RATIO={}'.format(IMPULSE_P, IMPULSE_RATIO))
    print('{:>6s} {:>10s} {:>12s} {:>12s} {:>10s} {:>12s}'.format(
        'SNR', 'channel', 'variance', 'expected', 'kurtosis', 'P(|z|>4sd)'))
    for snr in (-2, 1, 10, 16):
        expected = 1.0 / (2 * 10 ** (snr / 10))  # per real component
        for chan in ('awgn', 'impulsive'):
            var, kurt, tail = stats(chan, snr)
            print('{:>6d} {:>10s} {:>12.5f} {:>12.5f} {:>10.2f} {:>12.6f}'.format(
                snr, chan, var, expected, kurt, tail))
    print('\nGaussian kurtosis is 3.0. From the code constants, the impulsive mixture should give about {:.2f}.'.format(
        3 * ((1 - IMPULSE_P) + IMPULSE_P * (1 + IMPULSE_RATIO) ** 2) / (1 + IMPULSE_P * IMPULSE_RATIO) ** 2))
