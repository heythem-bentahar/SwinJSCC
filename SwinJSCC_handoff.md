# SwinJSCC: progress and next steps

## Goal
Evaluate the paper's pretrained image-transmission models and compare with its figures. No training has been done or is needed for these tests.
Repo: https://github.com/semcomm/SwinJSCC

## Setup
- Windows, Conda environment `torch`, RTX 5060 (8 GB).
- PyTorch: `2.10.0+cu128`.
- Test data: Kodak, 24 RGB PNGs, 768x512 or 512x768. Keep original resolution.
- Run commands from the project root.
- `main.py`: set `model_path` to the downloaded checkpoint; set Kodak `test_data_dir` to the actual image folder.
- Modified `data/datasets.py`: evaluation returns `(None, test_loader)` without accessing training data.
- Modified `main.py`: calculate `steps_epoch` only if `args.training`; otherwise use 0.
- Replaced `np.math.log10` with `math.log10` and added `import math`.
- Omit `--training` for evaluation. `--trainset DIV2K` selects the high-resolution architecture; a training dataset is not required after our loader fix.

## Terms
- SA: adapt to SNR using Channel ModNet.
- RA: adapt to transmission rate using Rate ModNet.
- SAandRA: both adaptations in one model.
- PSNR (dB), MS-SSIM (0–1): higher is better.
- CBR: complex channel symbols / source RGB scalar values.
- High-resolution model: CBR = C / 1536.
- C values 32,64,96,128,192 give CBR 1/48,1/24,1/16,1/12,1/8.

## Checkpoints
Current base models:
- AWGN: `SwinJSCC_w_SAandRA_AWGN_HRimage_cbr_psnr_snr.model`
- Rayleigh: `SwinJSCC_w_SAandRA_Rayleigh_HRimage_cbr_psnr_snr.model`
Set `--model_size base` for these; switch `model_path` when switching channels.
Earlier small-model tests used filenames containing `small`, with `--model_size small`.
We tested both `pretrained_...snr13.model` and files without that prefix. The paper describes staged training, but the exact filename-to-stage mapping is NOT verified. Prefer the files without `pretrained_` for the current evaluation.

## Completed: base model, 24 Kodak images
Each array contains averages across images. All four sweeps finished successfully.

### 1. AWGN SNR sweep (Figure 10b conditions)
CBR = 1/16.
SNR = [1,4,7,10,13]
PSNR = [26.311806,26.861212,27.242287,27.520257,27.684740]
MS-SSIM = [0.858059,0.883713,0.899087,0.908783,0.913959]

### 2. AWGN rate sweep (Figure 11b conditions)
SNR = 10 dB.
C = [32,64,96,128,192]
PSNR = [21.305786,27.055234,27.512667,30.010083,36.042261]
MS-SSIM = [0.705942,0.897780,0.908643,0.945739,0.985980]

### 3. Rayleigh SNR sweep (Figure 10e conditions)
CBR = 1/16.
SNR = [1,4,7,10,13]
PSNR = [25.608818,26.511711,26.960683,27.177093,27.303569]
MS-SSIM = [0.823707,0.864923,0.887036,0.897185,0.902425]

### 4. Rayleigh rate sweep (Figure 11e conditions)
SNR = 3 dB.
C = [32,64,96,128,192]
PSNR = [20.196693,25.204475,26.277582,28.347617,30.925951]
MS-SSIM = [0.591559,0.815030,0.854343,0.908346,0.951715]

## Commands
AWGN SNR sweep:
```bash
python main.py --trainset DIV2K --testset kodak --distortion-metric MSE --model SwinJSCC_w/_SAandRA --channel-type awgn --C 96 --multiple-snr 1,4,7,10,13 --model_size base
```
AWGN rate sweep: same command, but `--C 32,64,96,128,192 --multiple-snr 10`.
Rayleigh SNR sweep: same SNR command, but `--channel-type rayleigh` and load Rayleigh checkpoint.
Rayleigh rate sweep: `--channel-type rayleigh --C 32,64,96,128,192 --multiple-snr 3`, with Rayleigh checkpoint.

## Findings and limits
- Quality increases with SNR and CBR. Base AWGN PSNR is slightly higher than small-model PSNR in our SNR sweep.
- AWGN CBR=1/16, SNR=10 agrees between separate sweeps: 27.520 vs 27.513 dB.
- Our AWGN base SNR curve is BELOW the paper. Approximate visual readings of Figure 10b's SAandRA curve: [29.5,31.1,32.3,33.2,33.8] dB at [1,4,7,10,13]. These are estimates, not exact source data.
- At CBR=1/8, AWGN SNR=10, our 36.04 dB is close to Figure 11b; lower-rate points fall short.
- Authors warn SAandRA inference results may differ on PyTorch >1.12. Our PyTorch is newer. This is a possible cause, NOT proven.
- Old PyTorch GPU builds cannot run on RTX 5060. Keep current environment.
- Inspected encoder.py, decoder.py, network.py, channel.py: no confirmed cause found. Encoder scores and selects C features; channel uses selected-feature power and masks received features.
- A sorting-ties hypothesis was suggested but NOT tested. User rejected speculative debugging and asked to continue experiments.
- Do NOT claim exact reproduction, retraining, or a confirmed compatibility fix.

## Deliverables already created
- `SwinJSCC_results.png`: earlier small-model plots.
- `SwinJSCC_base_four_sweeps.png`: all four base sweeps, PSNR and MS-SSIM. Contains measured data only, no paper curves.

## Next steps
1. Compare all four completed PSNR sweeps against Figures 10b/10e/11b/11e. Label any manually read paper points as approximate.
2. Write a short results discussion: trends, gaps, checkpoint names and PyTorch limitation.
3. Optional later: save reconstructed images for visual examples, or test another dataset/model size. These are not done.
4. Only if exact reproduction is required: validate using the authors' older environment on CPU (requires device-portability edits) or a compatible older GPU. User currently prefers continuing on existing setup.

## Working preference
Keep explanations short and concrete. Avoid repeated sweeps already completed, unnecessary downloads, speculative fixes, and long debugging detours.
