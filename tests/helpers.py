import types
import torch
import torch.nn as nn
from net.network import SwinJSCC
from net.sampling import ConditionSampler

TYPES = ['awgn', 'rayleigh', 'rician', 'impulsive']


def make_net(type_aware=False, channel_type='mixed', C=32, model='SwinJSCC_w/_SA',
             init_seed=0, sampler_seed=0, noise_seed=0):
    """Small CIFAR10-configuration SwinJSCC on CUDA for tests."""
    common = dict(model=model, img_size=(32, 32), C=C, window_size=2, mlp_ratio=4.,
                  qkv_bias=True, qk_scale=None, norm_layer=nn.LayerNorm, patch_norm=True)
    config = types.SimpleNamespace(
        encoder_kwargs=dict(common, patch_size=2, in_chans=3, embed_dims=[128, 256],
                            depths=[2, 4], num_heads=[4, 8]),
        decoder_kwargs=dict(common, embed_dims=[256, 128], depths=[4, 2], num_heads=[8, 4]),
        logger=None, pass_channel=True, norm=False, downsample=2,
        device=torch.device('cuda:0'), CUDA=True)
    args = types.SimpleNamespace(
        channel_type=channel_type, type_aware=type_aware, multiple_snr='1,4,7,10,13',
        C=str(C), model=model, distortion_metric='MSE', trainset='CIFAR10')
    sampler = ConditionSampler(sampler_seed, [1, 4, 7, 10, 13], TYPES)
    state = torch.random.get_rng_state()
    cuda_state = torch.cuda.get_rng_state()
    torch.manual_seed(init_seed)
    net = SwinJSCC(args, config, sampler=sampler).cuda()
    torch.random.set_rng_state(state)
    torch.cuda.set_rng_state(cuda_state)
    net.channel.set_noise_seed(noise_seed)
    return net


def images(n=4, seed=0):
    g = torch.Generator(device='cuda').manual_seed(seed)
    return torch.rand(n, 3, 32, 32, device='cuda', generator=g)
