import torch.nn as nn
import numpy as np
import os
import torch
import time

RICIAN_K = 5.0        # Rician K-factor (LOS power / scattered power)
IMPULSE_P = 0.1       # probability that a symbol is hit by an impulse
IMPULSE_RATIO = 10.0  # impulse power / background power


class Channel(nn.Module):
    """
    Currently the channel model is either error free, erasure channel,
    rayleigh channel or the AWGN channel.
    """

    def __init__(self, args, config):
        super(Channel, self).__init__()
        self.config = config
        self.chan_type = args.channel_type
        self.device = config.device
        self.h = torch.sqrt(torch.randn(1) ** 2
                            + torch.randn(1) ** 2) / 1.414
        if config.logger:
            config.logger.info('【Channel】: Built {} channel, SNR {} dB.'.format(
                args.channel_type, args.multiple_snr))

    def gaussian_noise_layer(self, input_layer, std, name=None):
        device = input_layer.get_device()
        noise_real = torch.normal(mean=0.0, std=std, size=np.shape(input_layer), device=device)
        noise_imag = torch.normal(mean=0.0, std=std, size=np.shape(input_layer), device=device)
        noise = noise_real + 1j * noise_imag
        return input_layer + noise

    def rayleigh_noise_layer(self, input_layer, std, name=None):
        noise_real = torch.normal(mean=0.0, std=std, size=np.shape(input_layer))
        noise_imag = torch.normal(mean=0.0, std=std, size=np.shape(input_layer))
        noise = noise_real + 1j * noise_imag
        h = torch.sqrt(torch.normal(mean=0.0, std=1, size=np.shape(input_layer)) ** 2
                       + torch.normal(mean=0.0, std=1, size=np.shape(input_layer)) ** 2) / np.sqrt(2)
        if self.config.CUDA:
            noise = noise.to(input_layer.get_device())
            h = h.to(input_layer.get_device())
        return input_layer * h + noise


    def rician_noise_layer(self, input_layer, std, name=None):
        device = input_layer.get_device()
        shape = np.shape(input_layer)
        noise = torch.normal(0.0, std, size=shape, device=device) + 1j * torch.normal(0.0, std, size=shape, device=device)
        k = RICIAN_K
        real = np.sqrt(k / (k + 1)) + np.sqrt(1 / (2 * (k + 1))) * torch.randn(shape, device=device)
        imag = np.sqrt(1 / (2 * (k + 1))) * torch.randn(shape, device=device)
        h = torch.sqrt(real ** 2 + imag ** 2)  # E[h^2] = 1
        return input_layer * h + noise

    def impulsive_noise_layer(self, input_layer, std, name=None):
        # Bernoulli-Gaussian noise; total noise power equals the Gaussian case, so SNR is comparable
        device = input_layer.get_device()
        shape = np.shape(input_layer)
        std_bg = std / np.sqrt(1 + IMPULSE_P * IMPULSE_RATIO)
        std_imp = std_bg * np.sqrt(IMPULSE_RATIO)
        hit = (torch.rand(shape, device=device) < IMPULSE_P).float()
        noise_real = torch.normal(0.0, std_bg, size=shape, device=device) + hit * torch.normal(0.0, std_imp, size=shape, device=device)
        noise_imag = torch.normal(0.0, std_bg, size=shape, device=device) + hit * torch.normal(0.0, std_imp, size=shape, device=device)
        return input_layer + (noise_real + 1j * noise_imag)

    def complex_normalize(self, x, power):
        pwr = torch.mean(x ** 2) * 2
        out = np.sqrt(power) * x / torch.sqrt(pwr)
        return out, pwr


    def forward(self, input, chan_param, avg_pwr=False):
        if avg_pwr:
            power = 1
            channel_tx = np.sqrt(power) * input / torch.sqrt(avg_pwr * 2)
        else:
            channel_tx, pwr = self.complex_normalize(input, power=1)
        input_shape = channel_tx.shape
        channel_in = channel_tx.reshape(-1)
        L = channel_in.shape[0]
        channel_in = channel_in[:L // 2] + channel_in[L // 2:] * 1j
        channel_output = self.complex_forward(channel_in, chan_param)
        channel_output = torch.cat([torch.real(channel_output), torch.imag(channel_output)])
        channel_output = channel_output.reshape(input_shape)
        if self.chan_type == 1 or self.chan_type == 'awgn' or self.chan_type == 'impulsive':
            noise = (channel_output - channel_tx).detach()
            noise.requires_grad = False
            channel_tx = channel_tx + noise
            if avg_pwr:
                return channel_tx * torch.sqrt(avg_pwr * 2)
            else:
                return channel_tx * torch.sqrt(pwr)
        elif self.chan_type == 2 or self.chan_type == 'rayleigh' or self.chan_type == 'rician':
            if avg_pwr:
                return channel_output * torch.sqrt(avg_pwr * 2)
            else:
                return channel_output * torch.sqrt(pwr)

    def complex_forward(self, channel_in, chan_param):
        if self.chan_type == 0 or self.chan_type == 'none':
            return channel_in

        elif self.chan_type == 1 or self.chan_type == 'awgn':
            channel_tx = channel_in
            sigma = np.sqrt(1.0 / (2 * 10 ** (chan_param / 10)))
            chan_output = self.gaussian_noise_layer(channel_tx,
                                                    std=sigma,
                                                    name="awgn_chan_noise")
            return chan_output

        elif self.chan_type == 2 or self.chan_type == 'rayleigh':
            channel_tx = channel_in
            sigma = np.sqrt(1.0 / (2 * 10 ** (chan_param / 10)))
            chan_output = self.rayleigh_noise_layer(channel_tx,
                                                    std=sigma,
                                                    name="rayleigh_chan_noise")
            return chan_output

        elif self.chan_type == 'rician':
            sigma = np.sqrt(1.0 / (2 * 10 ** (chan_param / 10)))
            return self.rician_noise_layer(channel_in, std=sigma, name="rician_chan_noise")

        elif self.chan_type == 'impulsive':
            sigma = np.sqrt(1.0 / (2 * 10 ** (chan_param / 10)))
            return self.impulsive_noise_layer(channel_in, std=sigma, name="impulsive_chan_noise")


    def noiseless_forward(self, channel_in):
        channel_tx = self.normalize(channel_in, power=1)
        return channel_tx

