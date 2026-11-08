import torch
from data.datasets import get_cifar_loaders

ROOT = './CIFAR10/'


def loaders(**kw):
    return get_cifar_loaders(ROOT, batch_size=128, **kw)


def test_split_sizes():
    train, val, test = loaders()
    assert len(train.dataset) == 45000
    assert len(val.dataset) == 5000
    assert len(test.dataset) == 10000


def test_split_disjoint_and_fixed():
    t1, v1, _ = loaders(split_seed=3)
    t2, v2, _ = loaders(split_seed=3)
    assert list(v1.dataset.indices) == list(v2.dataset.indices)
    assert not set(t1.dataset.indices) & set(v1.dataset.indices)
    _, v3, _ = loaders(split_seed=4)
    assert list(v1.dataset.indices) != list(v3.dataset.indices)


def test_same_order_seed_same_batches():
    a, _, _ = loaders(order_seed=5)
    b, _, _ = loaders(order_seed=5)
    for i, ((xa, _), (xb, _)) in zip(range(3), zip(a, b)):
        assert torch.equal(xa, xb)


def test_val_comes_from_train_split():
    _, val, test = loaders()
    assert val.dataset.dataset.train is True
    assert test.dataset.train is False
    assert max(val.dataset.indices) < 50000


def test_val_and_test_unshuffled_no_flip():
    _, v1, _ = loaders()
    _, v2, _ = loaders()
    x1 = next(iter(v1))[0]
    x2 = next(iter(v2))[0]
    assert torch.equal(x1, x2)
