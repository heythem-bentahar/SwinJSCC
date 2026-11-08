# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

SwinJSCC (Swin Transformer deep joint source-channel coding, IEEE TCCN 2024). It is being used for a COE 540 course project. The fork evaluates the authors' pretrained checkpoints and extends the channel models. `SwinJSCC_handoff.md` has the current progress, measured results, and next steps. Read it before running experiments so you don't repeat sweeps that are already done.

## Environment

- Windows, Conda env `torch`, RTX 5060 (8 GB), PyTorch 2.10 + cu128. Upstream recommends torch ≤ 1.12 because SA/RA inference can differ on newer versions. Old torch GPU builds don't run on this GPU, so keep the current env and treat version drift as a possible, unproven cause of gaps from the paper.
- The code is CUDA-only. It hard-codes `.cuda()` in `main.py`, `net/encoder.py`, and `net/channel.py`, plus `config.device = cuda:0`.
- There is no test suite, linter, or build step. Run everything from the repo root with `main.py`.

## Commands

Evaluation is the default mode. Training needs `--training`.

```bash
# Evaluate SAandRA base model on Kodak, AWGN SNR sweep at C=96 (CBR 1/16)
python main.py --trainset DIV2K --testset kodak --distortion-metric MSE --model SwinJSCC_w/_SAandRA --channel-type awgn --C 96 --multiple-snr 1,4,7,10,13 --model_size base

# Rate sweep: --C 32,64,96,128,192 --multiple-snr 10
# Choose a checkpoint without editing code: --model-path path/to/x.model
# Save originals/reconstructions/side-by-sides + per-image metrics.csv: --save-images
```

- In test mode, `--C` and `--multiple-snr` are comma lists. `test()` evaluates the full SNR × C grid and prints 2-D arrays of SNR/CBR/PSNR/MS-SSIM. In training mode, each batch randomly samples one value from each list.
- `--trainset` selects the **architecture** as well as the data. `DIV2K` means the HR model (256×256 crops, 4 downsamples, window 8). `CIFAR10` means the 32×32 model (2 downsamples, window 2). Use `--trainset DIV2K` to evaluate HR checkpoints. In eval mode, `get_loader` returns `(None, test_loader)` and never touches training data.
- `--model_size` must match the checkpoint, because `load_state_dict` is `strict=True`.
- Checkpoint is resolved in this order: `--model-path`, then `config.model_path` in `main.py` (test only). Training starts from scratch unless `--model-path` is given.
- Dataset paths are hard-coded in `main.py`'s `config` class. These are Kodak `test_data_dir`, the DIV2K/CLIC `base_path` (upstream Linux paths that don't exist here), and `./CIFAR10/`.
- Each run creates `./history/<timestamp>/`. Training saves `models/<timestamp>_EP<n>.model`, and `--save-images` writes `visuals/<channel>/SNR_<s>_C_<c>/`.
- Plots: `python plot_swinjscc.py SwinJSCC_results.csv --out plots`. It expects columns `checkpoint,model_size,channel,SNR_dB,C,CBR,PSNR_dB,MS_SSIM` and rejects duplicate (checkpoint, SNR, C) rows.

## Architecture

The pipeline in `net/network.py` `SwinJSCC.forward` runs **encoder → power normalization + channel → (mask) → decoder**. It returns `(recon, CBR, SNR, mse, loss)`.

- **Encoder/decoder** (`net/encoder.py`, `net/decoder.py`): these are Swin stages with patch merging/expanding. When the input H/W changes, `update_resolution` is called, which rebuilds the attention masks. That is how full-resolution Kodak (768×512) works with a model trained on 256 crops.
- **Model variants** branch on the `model` string inside the encoder and decoder forward:
  - `w/o_SAandRA` uses a fixed linear head projecting to `C` channels.
  - `w/_SA` adds the Channel ModNet, an `AdaptiveModulator` MLP (sigmoid output) that scales features by SNR in both the encoder and decoder.
  - `w/_RA` and `w/_SAandRA` add the Rate ModNet. The encoder outputs a wide latent, scores channels, keeps the top-`rate` channels via sort, and returns `(feature, mask)`. The network computes power over the masked elements only, sends the full tensor through the channel, then re-applies the mask. For these models `channel_number` is `None` in config, and CBR is computed analytically as `C / (2·3·4^downsample)`. That gives `C/1536` for HR, so C=32…192 maps to CBR 1/48…1/8.
- **Channel** (`net/channel.py`): normalizes to unit power, packs the real tensor into complex symbols (first half real, second half imaginary), adds noise, then unpacks and rescales. AWGN and impulsive use a straight-through noise (`detach`) trick. Rayleigh and Rician return the faded output directly. The fork added `rician` (K=5), `impulsive` (Bernoulli-Gaussian with the same total noise power, so SNR is comparable), and `mixed`, which picks a random channel type per forward pass.
- **`--type-aware`** (fork addition) makes the ModNet condition `[SNR, one-hot(channel type)]`, so `cond_dim = 5` instead of 1. This changes ModNet layer shapes, so pretrained upstream checkpoints only load without this flag.
- **Loss/metrics** (`loss/distortion.py`): the training loss is `Distortion` (MSE or MS-SSIM per `--distortion-metric`). PSNR is always computed on the 0–255 scale. Logged MS-SSIM uses a separate `MS_SSIM` instance (window 3 for CIFAR10).

## Pretrained checkpoints and results

- Checkpoints come from the authors' Google Drive (link in README.md). The files currently in use are `SwinJSCC_w_SAandRA_{AWGN,Rayleigh}_HRimage_cbr_psnr_snr.model` (base). When you switch `--channel-type`, switch the checkpoint too.
- Prefer files without the `pretrained_` prefix. The mapping from filename to training stage is unverified.
- Measured results are in `SwinJSCC_results.csv` and `SwinJSCC_handoff.md`. Measured AWGN SNR-sweep PSNR is below the paper's Fig. 10b. Paper values read off figures are approximate. Don't claim exact reproduction or a confirmed root cause.

## Working preferences (from the handoff notes)

Keep explanations short and concrete. Avoid re-running completed sweeps, unnecessary downloads, speculative fixes, and long debugging detours.
