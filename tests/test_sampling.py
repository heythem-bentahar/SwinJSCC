import random
import torch
from net.sampling import ConditionSampler

TYPES = ['awgn', 'rayleigh', 'rician', 'impulsive']
SNRS = [1, 4, 7, 10, 13]


def test_same_seed_same_sequence():
    a = ConditionSampler(7, SNRS, TYPES)
    b = ConditionSampler(7, SNRS, TYPES)
    assert [a.next() for _ in range(200)] == [b.next() for _ in range(200)]


def test_independent_of_global_rng():
    a = ConditionSampler(7, SNRS, TYPES)
    b = ConditionSampler(7, SNRS, TYPES)
    out_a = []
    for _ in range(50):
        random.random()
        torch.rand(3)
        out_a.append(a.next())
    out_b = [b.next() for _ in range(50)]
    assert out_a == out_b


def test_uniform_over_channels():
    s = ConditionSampler(0, SNRS, TYPES)
    draws = [s.next()[0] for _ in range(4000)]
    for t in TYPES:
        assert 900 <= draws.count(t) <= 1100


def test_only_listed_snrs():
    s = ConditionSampler(0, SNRS, TYPES)
    assert {s.next()[1] for _ in range(500)} <= set(SNRS)


def test_log_records_every_draw():
    s = ConditionSampler(3, SNRS, TYPES)
    drawn = [s.next() for _ in range(10)]
    assert s.log == drawn
