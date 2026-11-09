# Run notes

## Timing (Task 5, Step 5)
- Model A (SNR-aware), CIFAR10, C=32, batch 128, mixed channels, RTX 5060: **130 s per epoch** (351 training steps plus full validation: 5,000 images x 4 channels x 5 SNRs).
- After 1 epoch: mean validation PSNR 23.06 dB (AWGN 23.48, Rayleigh 22.22, Rician 23.04, impulsive 23.49).
- 20 epochs is about 43 minutes per model; 100 epochs about 3.6 hours per model.
- Console note: the logger prints a harmless UnicodeEncodeError for the character in "Channel" on Windows cp1252 consoles; set `PYTHONIOENCODING=utf-8` to silence it.

## Epoch choice (Task 9, Step 1)
- 60 epochs per model (about 2.2 h each, about 4.4 h total at 130 s/epoch), sequential on one GPU. This is below the plan's 12-hour threshold, so the user was not asked. Best checkpoint chosen by validation PSNR, so overshooting epochs does not hurt.
- Seeds for both runs: init 0, sampler 0, noise 0, order 0 (identical). Single seed per model.

## A vs B results (Task 9, Steps 4-7)
- Epochs actually used: the planned 60 epochs were not reached. A (`snr_aware`) was stopped after epoch 22 (best checkpoint: epoch 21, mean val PSNR 28.56 dB). B (`snr_type_aware`) stopped after epoch 20 (best: epoch 20, 28.88 dB). Both models are undertrained and the stopping points differ by 1-2 epochs, so the comparison is approximately matched, not exactly.
- Seeds: init 0, sampler 0, noise 0, order 0 for both; single seed per model. The first 7,021 rows of both `sampler_log.csv` files are identical (A's log is longer).
- Evaluated with `best.model` of each run (`eval_channels.py`), channels awgn/rayleigh/rician/impulsive, SNR -2..16 dB. Outputs: `results/snr_aware.csv`, `results/snr_type_aware.csv`, `results/snr_type_aware_labels.csv`; figures in `plots_ab/`.
- CIFAR10 test: B is higher than A in PSNR at every channel and SNR, by about 0.2-0.7 dB. On AWGN and impulsive the gap grows with SNR (+0.17 dB at -2 dB, +0.66 dB at 16 dB); on Rayleigh it stays near +0.3 dB.
- Kodak (full-resolution images, models trained on 32x32 CIFAR10 and applied at test time): B is higher by about 0.5-1.1 dB. Report this as a train-on-CIFAR10, test-on-Kodak result.
- Impulsive vs AWGN: nearly identical results for both models, because the impulsive channel keeps the same total noise power. It is a weak test of channel-type awareness.
- Wrong-label test (B, CIFAR10, SNR 10 dB): a wrong channel label costs at most about 0.45 dB (e.g. Rayleigh true channel, AWGN label: 28.62 -> 28.25 dB; AWGN true channel, Rayleigh label: 32.24 -> 31.78 dB). B uses the label, but only weakly. Part of B's advantage over A may come from other effects (single seed, training noise).
- Caveats: undertrained models, one seed, gaps of a few tenths of a dB may be within seed noise, Kodak is out-of-distribution. Not compared with the paper's Fig. 10a/10d.
