import types
import pytest
import torch
from net.channel import Channel

TYPES = ['awgn', 'rayleigh', 'rician', 'impulsive']
DEV = torch.device('cuda:0')


def make_channel(chan_type):
    args = types.SimpleNamespace(channel_type=chan_type, multiple_snr='10')
    config = types.SimpleNamespace(device=DEV, CUDA=True, logger=None)
    return Channel(args, config)


def x():
    g = torch.Generator(device='cuda').manual_seed(99)
    return torch.randn(2, 16, 8, 8, device=DEV, generator=g)


@pytest.mark.parametrize('t', TYPES)
def test_same_noise_seed_same_output(t):
    a, b = make_channel(t), make_channel(t)
    a.set_noise_seed(5)
    b.set_noise_seed(5)
    assert torch.equal(a(x(), 5), b(x(), 5))


@pytest.mark.parametrize('t', TYPES)
def test_noise_independent_of_global_rng(t):
    a, b = make_channel(t), make_channel(t)
    a.set_noise_seed(5)
    b.set_noise_seed(5)
    out_a = a(x(), 5)
    torch.rand(100, device=DEV)
    torch.manual_seed(123)
    out_b = b(x(), 5)
    assert torch.equal(out_a, out_b)


@pytest.mark.parametrize('t', TYPES)
def test_unseeded_behaviour_unchanged(t):
    a = make_channel(t)
    assert not torch.equal(a(x(), 5), a(x(), 5))


def test_different_seed_different_output():
    a = make_channel('awgn')
    a.set_noise_seed(1)
    o1 = a(x(), 5)
    a.set_noise_seed(2)
    assert not torch.equal(o1, a(x(), 5))
