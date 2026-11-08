from net.decoder import *
from net.encoder import *
from loss.distortion import Distortion
from net.channel import Channel
from net.sampling import ConditionSampler
from random import choice
import torch.nn as nn


class SwinJSCC(nn.Module):
    def __init__(self, args, config, sampler=None):
        super(SwinJSCC, self).__init__()
        self.config = config
        self.chan_types = ['awgn', 'rayleigh', 'rician', 'impulsive']
        self.mixed = (args.channel_type == 'mixed')
        self.type_aware = getattr(args, 'type_aware', False)
        cond_dim = 1 + len(self.chan_types) if self.type_aware else 1
        encoder_kwargs = dict(config.encoder_kwargs, cond_dim=cond_dim)
        decoder_kwargs = dict(config.decoder_kwargs, cond_dim=cond_dim)
        self.encoder = create_encoder(**encoder_kwargs)
        self.decoder = create_decoder(**decoder_kwargs)
        if config.logger is not None:
            config.logger.info("Network config: ")
            config.logger.info("Encoder: ")
            config.logger.info(encoder_kwargs)
            config.logger.info("Decoder: ")
            config.logger.info(decoder_kwargs)
        self.distortion_loss = Distortion(args)
        self.channel = Channel(args, config)
        self.pass_channel = config.pass_channel
        self.squared_difference = torch.nn.MSELoss(reduction='none')
        self.H = self.W = 0
        self.multiple_snr = args.multiple_snr.split(",")
        for i in range(len(self.multiple_snr)):
            self.multiple_snr[i] = int(self.multiple_snr[i])
        self.channel_number = args.C.split(",")
        for i in range(len(self.channel_number)):
            self.channel_number[i] = int(self.channel_number[i])
        self.sampler = sampler if sampler is not None else ConditionSampler(
            0, self.multiple_snr, self.chan_types)
        self.downsample = config.downsample
        self.model = args.model

    def distortion_loss_wrapper(self, x_gen, x_real):
        distortion_loss = self.distortion_loss.forward(x_gen, x_real, normalization=self.config.norm)
        return distortion_loss

    def feature_pass_channel(self, feature, chan_param, avg_pwr=False):
        noisy_feature = self.channel.forward(feature, chan_param, avg_pwr)
        return noisy_feature

    def forward(self, input_image, given_SNR=None, given_rate=None, given_chan=None, label_chan=None):
        """given_chan: channel actually applied (eval override). label_chan: channel
        label fed to encoder/decoder when type_aware (defaults to the applied channel)."""
        for name in (given_chan, label_chan):
            if name is not None and name not in self.chan_types:
                raise ValueError("Unknown channel name {!r}; expected one of {}".format(name, self.chan_types))
        previous_chan = self.channel.chan_type
        try:
            return self._forward(input_image, given_SNR, given_rate, given_chan, label_chan)
        finally:
            if given_chan is not None:
                self.channel.chan_type = previous_chan

    def _forward(self, input_image, given_SNR, given_rate, given_chan, label_chan):
        B, _, H, W = input_image.shape

        if H != self.H or W != self.W:
            self.encoder.update_resolution(H, W)
            self.decoder.update_resolution(H // (2 ** self.downsample), W // (2 ** self.downsample))
            self.H = H
            self.W = W

        sampled_chan = None
        if given_SNR is None:
            sampled_chan, chan_param = self.sampler.next()
        else:
            chan_param = given_SNR

        if given_chan is not None:
            self.channel.chan_type = given_chan
        elif self.mixed and sampled_chan is not None:
            self.channel.chan_type = sampled_chan
        cond = chan_param
        if self.type_aware:
            label = label_chan if label_chan is not None else self.channel.chan_type
            if label not in self.chan_types:
                raise ValueError("Cannot build a channel label from {!r}".format(label))
            onehot = [float(label == t) for t in self.chan_types]
            cond = [chan_param] + onehot

        if given_rate is None:
            channel_number = choice(self.channel_number)
        else:
            channel_number = given_rate

        if self.model == 'SwinJSCC_w/o_SAandRA' or self.model == 'SwinJSCC_w/_SA':
            feature = self.encoder(input_image, cond, channel_number, self.model)
            CBR = feature.numel() / 2 / input_image.numel()
            if self.pass_channel:
                noisy_feature = self.feature_pass_channel(feature, chan_param)
            else:
                noisy_feature = feature

        elif self.model == 'SwinJSCC_w/_RA' or self.model == 'SwinJSCC_w/_SAandRA':
            feature, mask = self.encoder(input_image, cond, channel_number, self.model)
            CBR = channel_number / (2 * 3 * 2 ** (self.downsample * 2))
            avg_pwr = torch.sum(feature ** 2) / mask.sum()
            if self.pass_channel:
                noisy_feature = self.feature_pass_channel(feature, chan_param, avg_pwr)
            else:
                noisy_feature = feature
            noisy_feature = noisy_feature * mask

        recon_image = self.decoder(noisy_feature, cond, self.model)
        mse = self.squared_difference(input_image * 255., recon_image.clamp(0., 1.) * 255.)
        loss_G = self.distortion_loss.forward(input_image, recon_image.clamp(0., 1.))
        return recon_image, CBR, chan_param, mse.mean(), loss_G.mean()

