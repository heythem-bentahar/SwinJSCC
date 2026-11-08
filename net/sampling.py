import random


class ConditionSampler:
    """Seeded, self-contained draws of (channel type, SNR).

    Uses its own random.Random so the sequence is independent of the global
    RNG state (model init, data order, channel noise)."""

    def __init__(self, seed, snrs, chan_types):
        self.rng = random.Random(seed)
        self.snrs = list(snrs)
        self.chan_types = list(chan_types)
        self.log = []

    def next(self):
        chan = self.rng.choice(self.chan_types)
        snr = self.rng.choice(self.snrs)
        self.log.append((chan, snr))
        return chan, snr
