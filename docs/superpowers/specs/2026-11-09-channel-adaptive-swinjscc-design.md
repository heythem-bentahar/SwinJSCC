# Channel-Adaptive SwinJSCC: Design

## Intent
Course project (COE 540, Project 8: Channel-Adaptive Semantic Communication). Answer the template's question: does an encoder/decoder that is told the channel type perform better than one that is not? Reproduce only the parts of the SwinJSCC paper (IEEE TCCN 2024) needed as a baseline for that question.

## Goals
1. **Baseline.** Reproduce Fig. 10a (AWGN) and 10d (fast Rayleigh): PSNR versus SNR on CIFAR10 for the paper's SNR-adaptive model ("w/ SA"), trained from scratch.
2. **Channels.** AWGN, Rayleigh, Rician (K=5) and impulsive (Bernoulli-Gaussian, p=0.1, power ratio 10). These already exist in `net/channel.py`. Each needs a short literature reference in the report.
3. **Model A (SNR-aware):** `SwinJSCC_w/_SA`. It receives SNR but not channel type. The template calls this "channel-unaware"; the report uses "SNR-aware" because the paper's w/_SA already adapts to SNR, and explains the mapping to the template's wording.
4. **Model B (SNR + channel-type-aware):** the same model with `--type-aware`. It receives SNR plus a one-hot channel type in both encoder and decoder.
5. **Experiment.** Train A and B on the same mix of channels (`--channel-type mixed`). Evaluate each on every channel and plot PSNR (and MS-SSIM) against SNR, with A and B as two curves per channel.
6. **Secondary test.** Evaluate the CIFAR10-trained models on Kodak as a generalization check, labeled as such. Do not compare it with the paper's Kodak numbers.
7. **Deliverable.** Plots, tables and a short report following the project template.

## Non-goals
- Rate adaptation (`w/_RA`, `w/_SAandRA`), Figs. 9, 11, 12, BPG+LDPC comparison, DIV2K training.
- A domain-specific channel (underwater or satellite).
- Matching the paper's absolute numbers on Kodak.

## Anti-goals
- Claiming exact reproduction, or a confirmed cause for gaps from the paper.
- A comparison where A and B differ in anything other than the channel-type input.

## Decided by the user
- GPU: RTX 5060 8 GB. Train on CIFAR10, test on CIFAR10 and Kodak.
- Four channels, no fifth.
- Only parts aligned with the template are reproduced.
- Previously trained checkpoints `history/2026-11-07_02-13-44/models/*_EP1..7` were trained locally with modified training parameters. Their configuration is unknown.

## Constraints
- Run from the repo root; CUDA only; PyTorch 2.10 (the authors warn SA/RA inference may differ above torch 1.12; unproven here).
- CIFAR10 CBR = C/96. Use C=32 (CBR 1/3), as in the paper's Fig. 10.
- Use the paper's settings: Adam, lr 1e-4, batch 128, training SNR drawn uniformly from {1, 4, 7, 10, 13} dB (the code samples from this list).
- Test SNRs: 1, 4, 7, 10, 13 dB, plus unseen -2 and 16 dB (marked as beyond the paper).
- SNR is defined identically across channels (impulsive keeps total noise power equal to AWGN; fading channels score lower at equal SNR).

## Required code changes (known)
- Let `--testset kodak` work with `--trainset CIFAR10`: the test dataset must be decoupled from the architecture choice in `main.py` and `data/datasets.py`.
- `--type-aware` changes ModNet input shape, so A and B are separate trained models and A must be trained without the flag.

## Left to the builder
- Number of epochs and time budget (paper trains CIFAR10 for far fewer epochs than the repo default of `tot_epoch = 20`; pick what fits).
- Checkpoint naming, result CSV format (reuse `SwinJSCC_results.csv` columns plus a model-type column), and plot layout (extend `plot_swinjscc.py`).
- Whether to inspect the old EP1–EP7 checkpoints to identify them (not required).

## Open items
- Literature references for the four channels (to be gathered for the report).

## Clarifications (from builder check)

Answered by the existing code:
- **Where channel type enters:** `net/network.py` builds `cond = [SNR] + one-hot(channel type)` and passes it to both encoder and decoder wherever SNR is passed today. `cond_dim` goes from 1 to 5 in the ModNet input layer. Model A uses `cond_dim = 1`. No other difference.
- **Channel type at test time:** assumed known to both ends (like SNR), as side information.
- **Fading:** Rayleigh and Rician are per-symbol i.i.d. fast fading, applied raw. No equalization and no CSI beyond SNR and channel type.

Builder's calls (marked as calls, not user decisions):
- **Mixed training:** the channel type is drawn uniformly per batch (`random.choice`). Seed both A and B runs identically so they see the same channel sequence where possible.
- **Baseline acceptance:** no hard tolerance. If a curve falls short of the paper, report the gap without claiming a cause.
- **Old checkpoints:** EP1–EP7 are never loaded. Use fresh output directories.

## Final decisions (user)
- Train from scratch on CIFAR10. No high-resolution training set is available, so fine-tuning the authors' pretrained models is out.
- Two training runs only: Model A and Model B, both on mixed channels. The AWGN and Rayleigh curves from Model A serve as the Fig. 10a/10d baseline. No single-channel baseline runs.
- One seed per model. The report states the results are from a single run.
- Deadline and GPU-time budget were not given. Size epochs to what fits on the RTX 5060 and report the epoch count used.

## Tightened requirements (user, before training)
1. **Naming.** A = SNR-aware; B = SNR + channel-type-aware. Use these names in code, filenames, plots and the report. Never call A simply "unaware".
2. **Matched training.** A and B use the same images and image order, C, optimizer, learning rate, number of training steps, and the same channel/SNR sampling sequence. Use separate random generators for (a) model initialization, (b) channel-type and SNR sampling, (c) data order, and (d) channel noise. A shared global seed is not enough. Required code change: `net/network.py` currently samples with global `random.choice`, so switch to dedicated `random.Random` / `torch.Generator` objects. Verify by logging the sampled (channel, SNR) sequence for both runs and checking that they match.
3. **Validation-based checkpoint selection.** Hold out part of the CIFAR10 training set (proposed: 5,000 of 50,000 images, fixed split, my call) as validation. Both models get the same training budget. Pick each model's checkpoint by validation performance on the same mixed-channel, SNR-sampled validation protocol. The official 10,000-image test set is used only for final evaluation. Required code change: a train/validation split in `data/datasets.py` and a validation pass in the training loop.
4. **Wrong-label test for B.** After training, evaluate B on the test set with (i) correct channel labels and (ii) deliberately wrong labels, with the actual channel unchanged. Report the drop. No extra training. Required code change: separate the channel label passed to the network from the channel actually applied (an eval-only override). Wrong-label scheme: for each actual channel, test every other label (full 4×4 confusion table), my call.
