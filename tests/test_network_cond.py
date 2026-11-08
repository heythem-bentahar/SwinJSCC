import pytest
import torch
from tests.helpers import make_net, images


def run(net, **kw):
    net.channel.set_noise_seed(11)
    with torch.no_grad():
        return net(images(), 7, **kw)[0]


def test_label_override_changes_output_when_type_aware():
    net = make_net(type_aware=True).eval()
    right = run(net, given_chan='awgn', label_chan='awgn')
    wrong = run(net, given_chan='awgn', label_chan='impulsive')
    assert not torch.equal(right, wrong)


def test_label_override_ignored_when_not_type_aware():
    net = make_net(type_aware=False).eval()
    a = run(net, given_chan='awgn', label_chan='awgn')
    b = run(net, given_chan='awgn', label_chan='impulsive')
    assert torch.equal(a, b)


def test_given_chan_changes_applied_channel():
    net = make_net(type_aware=True).eval()
    seen = {}
    orig = net.channel.forward

    def spy(*a, **k):
        seen['t'] = net.channel.chan_type
        return orig(*a, **k)
    net.channel.forward = spy
    run(net, given_chan='rician')
    assert seen['t'] == 'rician'
    assert net.channel.chan_type == 'mixed'  # restored after the call


def test_unknown_label_raises():
    net = make_net(type_aware=True).eval()
    with pytest.raises(ValueError):
        run(net, given_chan='awgn', label_chan='mixed')
    with pytest.raises(ValueError):
        run(net, given_chan='bogus')


def test_given_chan_does_not_advance_sampler():
    net = make_net(type_aware=True).eval()
    run(net, given_chan='awgn')
    assert net.sampler.log == []


def test_a_and_b_differ_only_in_cond_dim():
    a = make_net(type_aware=False).state_dict()
    b = make_net(type_aware=True).state_dict()
    assert a.keys() == b.keys()
    diff = [k for k in a if a[k].shape != b[k].shape]
    assert diff, 'B must have a wider ModNet input'
    assert all('bm_list' in k and k.endswith('fc.0.weight') for k in diff)
