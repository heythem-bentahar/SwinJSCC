# Channel-Adaptive SwinJSCC Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train and compare Model A (SNR-aware) and Model B (SNR + channel-type-aware) SwinJSCC on CIFAR10 over four channels, with matched training, validation-based checkpointing, a Kodak generalization test and a wrong-label test for B.

**Architecture:** Keep the existing `main.py` / `net/` code. Add a seeded condition sampler and a seeded channel-noise generator so A and B see identical (channel, SNR) sequences and noise. Add a CIFAR10 train/validation split and best-checkpoint saving. Put all evaluation (per channel, per SNR, Kodak, wrong labels) in a new `eval_channels.py`, and plotting in a new `plot_ab.py`.

**Tech Stack:** Python, PyTorch 2.10 (CUDA, RTX 5060 8 GB), pytest, matplotlib. Run from the repo root. The repo is not a git repository, so there are no commit steps; each task ends with its tests passing.

**Spec:** `docs/superpowers/specs/2026-11-09-channel-adaptive-swinjscc-design.md`

## Global Constraints

- Model A = `SwinJSCC_w/_SA` without `--type-aware` (SNR-aware). Model B = same with `--type-aware` (SNR + channel-type-aware). Use "SNR-aware" and "SNR + channel-type-aware" in code, filenames, plots and report. Never "unaware".
- CIFAR10, C=32 (CBR 1/3 = C/96). Adam, lr 1e-4, batch 128, training SNR sampled from {1, 4, 7, 10, 13} dB.
- Channels: awgn, rayleigh, rician (K=5), impulsive (p=0.1, power ratio 10). Mixed training draws the channel uniformly per batch.
- A and B: same images and order, C, optimizer, lr, number of steps, and the same (channel, SNR) sequence. Separate generators for model init, condition sampling, data order and channel noise.
- Validation: 5,000 of the 50,000 CIFAR10 training images, fixed split. Official 10,000-image test set is used only for final evaluation.
- Test SNRs: 1, 4, 7, 10, 13 dB, plus unseen -2 and 16 dB (marked "beyond paper").
- Never load `history/2026-11-07_02-13-44` checkpoints; use fresh output directories.
- Kodak is a secondary generalization test only. Do not compare with the paper's Kodak numbers.
- Report states results are from a single run per model.

## Review Focus

- Channel label passed as a name not in the four types (e.g. `"mixed"` at eval) must raise a clear error, not silently use zeros.
- Evaluation with `--type-aware` off must ignore any label override and give identical output.
- Re-running the sampler after the global torch RNG was consumed must give the same sequence.
- Kodak images whose sides are not divisible by the window grid must be cropped, not crash.
- A validation pass must not change the training sampler/noise generator state (otherwise A and B drift apart).

## File Structure

- Create `net/sampling.py`: `ConditionSampler` (seeded channel/SNR draws).
- Modify `net/channel.py`: optional seeded noise generator.
- Modify `net/network.py`: use sampler, add eval-only `given_chan` / `label_chan`.
- Modify `data/datasets.py`: CIFAR10 train/val/test loaders with seeded order.
- Modify `main.py`: new flags, validation loop, best-checkpoint saving (CIFAR10 path only).
- Create `eval_channels.py`: per-channel/SNR evaluation, Kodak, wrong-label table.
- Create `plot_ab.py`: A vs B curves and label-confusion heatmap.
- Create `tests/`: `test_sampling.py`, `test_channel_noise.py`, `test_network_cond.py`, `test_data_split.py`, `test_eval.py`.

---

### Task 1: Seeded condition sampler

**Files:**
- Create: `net/sampling.py`
- Modify: `net/network.py` (forward, `__init__`)
- Test: `tests/test_sampling.py`

**Interfaces:**
- Produces: `ConditionSampler(seed: int, snrs: list[int], chan_types: list[str])` with `next() -> tuple[str, int]` and `log: list[tuple[str, int]]` (all draws so far). `SwinJSCC(args, config, sampler=None)`; `SwinJSCC.sampler` is used in `forward` when `given_SNR is None` / channel type is `mixed`.

- [ ] **Step 1: Write failing tests** in `tests/test_sampling.py`: `test_same_seed_same_sequence` (two samplers, seed 7, 200 draws equal); `test_independent_of_global_rng` (consume `random.random()` and `torch.rand` between draws, sequence unchanged); `test_uniform_over_channels` (4000 draws, each of 4 channels between 900 and 1100); `test_only_listed_snrs`.
- [ ] **Step 2: Run** `python -m pytest tests/test_sampling.py -v`. Expected: FAIL, module not found. Install pytest if missing.
- [ ] **Step 3: Implement** `ConditionSampler` in `net/sampling.py` using its own `random.Random(seed)`; append every draw to `log`.
- [ ] **Step 4: Wire into `SwinJSCC`**: replace the global `choice(self.multiple_snr)` and `choice(self.chan_types)` in `forward` with `self.sampler.next()` when training/no given values. A given SNR or given channel bypasses the sampler and does not advance it.
- [ ] **Step 5: Run tests.** Expected: PASS.

### Task 2: Seeded channel noise

**Files:**
- Modify: `net/channel.py` (all four noise layers, `__init__`)
- Test: `tests/test_channel_noise.py`

**Interfaces:**
- Produces: `Channel.set_noise_seed(seed: int) -> None`. After it, all noise draws use a dedicated `torch.Generator(device='cuda')`. Without it, behavior is unchanged.

- [ ] **Step 1: Write failing tests:** `test_same_noise_seed_same_output` (two Channel objects, same seed, same input, same output for each of the 4 types); `test_noise_independent_of_global_rng` (output unchanged after `torch.rand` calls); `test_unseeded_behaviour_unchanged` (output differs between calls).
- [ ] **Step 2: Run, expect FAIL.**
- [ ] **Step 3: Implement** by passing `generator=` to `torch.normal`, `torch.randn`, `torch.rand` in `gaussian_noise_layer`, `rayleigh_noise_layer`, `rician_noise_layer`, `impulsive_noise_layer`. In `rayleigh_noise_layer` create tensors on the generator's device directly instead of CPU-then-move.
- [ ] **Step 4: Run tests, expect PASS.**

### Task 3: Eval-only channel override and label override

**Files:**
- Modify: `net/network.py` (`forward`)
- Test: `tests/test_network_cond.py`

**Interfaces:**
- Consumes: `ConditionSampler` (Task 1), `Channel` (Task 2).
- Produces: `SwinJSCC.forward(input_image, given_SNR=None, given_rate=None, given_chan=None, label_chan=None)`. `given_chan` sets the channel actually applied (one of the four names). `label_chan` sets the one-hot fed to encoder and decoder; default is `given_chan`. When `type_aware` is False, `label_chan` is ignored. Unknown names raise `ValueError`.

- [ ] **Step 1: Write failing tests** (small CIFAR-config net on CUDA, 4 images): `test_label_override_changes_output_when_type_aware` (same input and seeds, `label_chan` differing from `given_chan` gives a different reconstruction); `test_label_override_ignored_when_not_type_aware` (outputs identical); `test_given_chan_changes_applied_channel` (`net.channel.chan_type` matches during the call); `test_unknown_label_raises` (`label_chan="mixed"` raises ValueError); `test_a_and_b_differ_only_in_cond_dim` (parameter shapes equal except the ModNet input layers).
- [ ] **Step 2: Run** `python -m pytest tests/test_network_cond.py -v`. Expected: FAIL.
- [ ] **Step 3: Implement** in `forward`; build `cond = [chan_param] + one_hot(label)` only when `type_aware`. Restore `channel.chan_type` after the call if `given_chan` was used, so evaluation does not leak state.
- [ ] **Step 4: Run tests.** Expected: PASS.

### Task 4: CIFAR10 train / validation / test loaders

**Files:**
- Modify: `data/datasets.py`
- Test: `tests/test_data_split.py`

**Interfaces:**
- Produces: `get_cifar_loaders(root: str, batch_size: int, val_size: int = 5000, split_seed: int = 0, order_seed: int = 0) -> tuple[DataLoader, DataLoader, DataLoader]` returning `(train, val, test)`. Train uses `drop_last=True` and a shuffle driven by a `torch.Generator` seeded with `order_seed`. Val and test are unshuffled; test batch size 1024.

- [ ] **Step 1: Write failing tests:** `test_split_sizes` (45,000 / 5,000 / 10,000); `test_split_disjoint_and_fixed` (same `split_seed` gives the same val indices; train and val disjoint); `test_same_order_seed_same_batches` (first 3 batches equal across two loaders); `test_val_comes_from_train_split` (val images come from the 50,000 training images, not the official test set).
- [ ] **Step 2: Run, expect FAIL.**
- [ ] **Step 3: Implement** with `torch.utils.data.Subset` on the CIFAR10 training set, using a permutation from `split_seed`. Horizontal flip on train only. Keep the old `get_loader` for DIV2K.
- [ ] **Step 4: Run tests.** Expected: PASS.

### Task 5: Matched training with validation-based checkpoints

**Files:**
- Modify: `main.py` (CIFAR10 path only)
- Test: `tests/test_train_smoke.py`

**Interfaces:**
- Consumes: `get_cifar_loaders` (Task 4), `ConditionSampler` (Task 1), `Channel.set_noise_seed` (Task 2).
- Produces: flags `--run-name` (required with `--training`; output dir `history/<run-name>/`), `--epochs` (default 20), `--init-seed`, `--sampler-seed`, `--noise-seed`, `--order-seed` (all default 0). Training writes `models/best.model` (best mean validation PSNR), `models/last.model`, `sampler_log.csv` (step, channel, snr) and `val_log.csv` (epoch, mean PSNR, per-channel PSNR).

- [ ] **Step 1: Write a failing smoke test** that runs a short training for A and B with the same seeds on a 256-image subset (`--epochs 1`) and asserts: both `sampler_log.csv` files are identical; `best.model` exists for both; the log length equals the number of training steps (validation did not advance the training sampler).
- [ ] **Step 2: Run, expect FAIL.**
- [ ] **Step 3: Implement.** Seed `torch.manual_seed(init_seed)` only around model construction, then restore the RNG state. Build the sampler and noise generator from their own seeds. Validation runs after each epoch under `torch.no_grad()` and `net.eval()` with its own fixed `ConditionSampler(seed=12345)` over all four channels and the five SNRs and a separate noise seed, so it never touches training state. Select `best.model` by mean validation PSNR. Remove the `config.model_path` fallback when training.
- [ ] **Step 4: Run test.** Expected: PASS.
- [ ] **Step 5: Time one epoch** (`python main.py --training --trainset CIFAR10 --model SwinJSCC_w/_SA --channel-type mixed --C 32 --multiple-snr 1,4,7,10,13 --run-name timing_A --epochs 1`) and record seconds per epoch in `docs/superpowers/plans/notes.md`. This sizes the epoch count for Task 9.

### Task 6: Evaluation script

**Files:**
- Create: `eval_channels.py`
- Test: `tests/test_eval.py`

**Interfaces:**
- Produces: `python eval_channels.py --checkpoint PATH --run-name NAME --type-aware {0,1} --out results/NAME.csv [--kodak-dir DIR]`. CSV columns: `model,dataset,channel,SNR_dB,beyond_paper,PSNR_dB,MS_SSIM`. Function `evaluate(net, loader, channel: str, snr: int, label: str | None = None) -> tuple[float, float]` returning mean PSNR and MS-SSIM over the loader.

- [ ] **Step 1: Write failing tests:** `test_evaluate_returns_psnr_and_msssim` (tiny net, 8 images); `test_rows_cover_grid` (4 channels × 7 SNRs on CIFAR10 test; `beyond_paper` true only for -2 and 16); `test_kodak_crops_to_valid_size` (a 500×375 image does not crash and is center-cropped to multiples of 8); `test_eval_deterministic_with_noise_seed`.
- [ ] **Step 2: Run** `python -m pytest tests/test_eval.py -v`. Expected: FAIL.
- [ ] **Step 3: Implement.** Evaluate each (channel, SNR) with `given_chan` and `given_SNR`, label equal to the true channel. Seed the noise generator identically for A and B. Kodak: load PNGs from `--kodak-dir`, center-crop, batch size 1.
- [ ] **Step 4: Run tests.** Expected: PASS.

### Task 7: Wrong-label test for B

**Files:**
- Modify: `eval_channels.py`
- Test: `tests/test_eval.py`

**Interfaces:**
- Consumes: `evaluate(..., label)` (Task 6) and `label_chan` (Task 3).
- Produces: `--label-sweep` flag; writes `results/<run-name>_labels.csv` with columns `actual_channel,label_channel,SNR_dB,PSNR_dB,MS_SSIM` (4×4 channel pairs × the five paper SNRs, CIFAR10 test set).

- [ ] **Step 1: Write failing tests:** `test_label_sweep_has_16_pairs_per_snr`; `test_diagonal_equals_normal_eval` (matching label gives the same PSNR as Task 6 within 1e-6); `test_label_sweep_refused_for_model_a` (clear error).
- [ ] **Step 2: Run, expect FAIL.**
- [ ] **Step 3: Implement** the sweep loop, reusing `evaluate`.
- [ ] **Step 4: Run tests.** Expected: PASS.

### Task 8: Plots

**Files:**
- Create: `plot_ab.py`
- Test: `tests/test_plot_ab.py`

**Interfaces:**
- Produces: `python plot_ab.py --a results/A.csv --b results/B.csv --labels results/B_labels.csv --out plots_ab/`. It writes `psnr_vs_snr_<dataset>.pdf` (2×2 panels, one per channel, curves labeled "SNR-aware (A)" and "SNR + channel-type-aware (B)", unseen SNRs marked as beyond the paper), the same for MS-SSIM, and `label_confusion.pdf` (4×4 heatmap of PSNR drop versus the correct label, averaged over SNR).

- [ ] **Step 1: Write a failing test** that feeds tiny synthetic CSVs and asserts all expected PDFs exist.
- [ ] **Step 2: Run, expect FAIL.**
- [ ] **Step 3: Implement** in the matplotlib-only style of `plot_swinjscc.py`; load the `dataviz` skill for color choices.
- [ ] **Step 4: Run test.** Expected: PASS.

### Task 9: Train, evaluate, report

**Files:**
- Create: `results/` outputs; additions to `docs/superpowers/plans/notes.md`.

- [ ] **Step 1: Choose epochs** from Task 5's timing. If the total exceeds about 12 hours, ask the user for their budget first. Write the choice in `notes.md`.
- [ ] **Step 2: Train A:** `python main.py --training --trainset CIFAR10 --model SwinJSCC_w/_SA --channel-type mixed --C 32 --multiple-snr 1,4,7,10,13 --run-name snr_aware --epochs N`.
- [ ] **Step 3: Train B:** the same command plus `--type-aware` and `--run-name snr_type_aware`.
- [ ] **Step 4: Verify matching:** `cmp history/snr_aware/sampler_log.csv history/snr_type_aware/sampler_log.csv` prints nothing.
- [ ] **Step 5: Evaluate** both on CIFAR10 test and Kodak with `eval_channels.py`; run `--label-sweep` for B.
- [ ] **Step 6: Plot** with `plot_ab.py`. Compare A's AWGN and Rayleigh curves with the paper's Fig. 10a/10d qualitatively only; report any gap without claiming a cause.
- [ ] **Step 7: Summarize** in `notes.md`: epochs used, single seed, A vs B findings, wrong-label drop, Kodak caveat.

## Self-Review

- **Spec coverage:** naming (Global Constraints), matched training and separate generators (Tasks 1, 2, 5), validation (Tasks 4, 5), wrong-label test (Tasks 3, 7), Kodak and unseen SNRs (Task 6), plots and report (Tasks 8, 9). Channel literature references are the user's report work, not code, and are not in this plan.
- **Known risk:** Task 2 depends on `torch.normal(..., generator=)` working on CUDA with complex tensors being built afterward; if a layer resists, generate real and imaginary parts separately.
