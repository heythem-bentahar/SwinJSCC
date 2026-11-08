# Run notes

## Timing (Task 5, Step 5)
- Model A (SNR-aware), CIFAR10, C=32, batch 128, mixed channels, RTX 5060: **130 s per epoch** (351 training steps plus full validation: 5,000 images x 4 channels x 5 SNRs).
- After 1 epoch: mean validation PSNR 23.06 dB (AWGN 23.48, Rayleigh 22.22, Rician 23.04, impulsive 23.49).
- 20 epochs is about 43 minutes per model; 100 epochs about 3.6 hours per model.
- Console note: the logger prints a harmless UnicodeEncodeError for the character in "Channel" on Windows cp1252 consoles; set `PYTHONIOENCODING=utf-8` to silence it.

## Epoch choice (Task 9, Step 1)
- 60 epochs per model (about 2.2 h each, about 4.4 h total at 130 s/epoch), sequential on one GPU. This is below the plan's 12-hour threshold, so the user was not asked. Best checkpoint chosen by validation PSNR, so overshooting epochs does not hurt.
- Seeds for both runs: init 0, sampler 0, noise 0, order 0 (identical). Single seed per model.
