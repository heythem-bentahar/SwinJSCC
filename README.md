# SwinJSCC: Taming Swin Transformer for Joint Source-Channel Coding

Official Pytorch implementation for "[SwinJSCC: Taming Swin Transformer for Deep Joint Source-Channel Coding](https://arxiv.org/abs/2308.09361)".

## Introduction

In this paper, we establish a new neural JSCC backbone that can also adapt flexibly to diverse channel conditions and transmission rates within a single model, our open-source project aims to promote the research in this field. Specifically, we show that with elaborate design, neural JSCC codec built on the emerging Swin Transformer backbone achieves superior performance than conventional neural JSCC codecs built upon CNN, while also requiring lower end-to-end processing latency. Paired with two spatial modulation modules that scale latent representations based on the channel state information and target transmission rate, our baseline SwinJSCC can further upgrade to a versatile version, which increases its capability to adapt to diverse channel conditions and rate configurations. Extensive experimental results show that our SwinJSCC achieves better or comparable performance versus the state-of-the-art engineered BPG + 5G LDPC coded transmission system with much faster end-to-end coding speed, especially for high-resolution images, in which case traditional CNN-based JSCC yet falls behind due to its limited model capacity. 

## Fork additions

This fork evaluates the authors' pretrained checkpoints and extends the channel models. The original code and results are credited to the authors (see Citation below). Everything in this section is an addition to the upstream repository.

### Channel models
`--channel-type` accepts `awgn`, `rayleigh`, `rician`, `impulsive` and `mixed`.
* `rician`: Rician fading with K = 5.
* `impulsive`: Bernoulli-Gaussian noise with the same total noise power as AWGN, so SNR values stay comparable.
* `mixed`: picks a random channel type for each forward pass (used for training one model across channels).

### Channel-type-aware ModNet
`--type-aware` feeds the Channel ModNet `[SNR, one-hot(channel type)]` instead of SNR alone. This changes the ModNet layer shapes, so the authors' pretrained checkpoints only load without this flag.

### Three-model comparison (A, B, C)
The CIFAR10 experiments compare three models trained under identical conditions (`--channel-type mixed`, `--C 32`, same seeds):

| Model | `--model` | Extra flag | Told about |
|---|---|---|---|
| A (`snr_aware`) | `SwinJSCC_w/_SA` | none | SNR |
| B (`snr_type_aware`) | `SwinJSCC_w/_SA` | `--type-aware` | SNR and channel type |
| C (`no_info`) | `SwinJSCC_w/o_SAandRA` | none | nothing (channel still adds noise at the sampled SNR) |

A and B differ only in the ModNet input size (1 vs 5). C has no ModNet layers at all, so a gap between C and A mixes the missing SNR information with the missing ModNet capacity.

### CIFAR10 training
With `--training --trainset CIFAR10`, the script runs a seeded training loop with validation:
* 45,000 / 5,000 / 10,000 train / validation / test split with a fixed split seed.
* Seeds: `--init-seed`, `--sampler-seed` (channel and SNR sampling), `--noise-seed` (channel noise) and `--order-seed` (data order and flips). Using the same seeds for two models gives them identical training conditions.
* `--run-name NAME` is required. Outputs go to `history/NAME/`: `models/best.model` (best mean validation PSNR), `models/last.model`, `val_log.csv` and `sampler_log.csv`.
* `--epochs N` sets the number of epochs (default 20).
* `--train-subset N` and `--val-subset N` use only the first N images, for quick smoke tests.

```
python main.py --training --trainset CIFAR10 --model SwinJSCC_w/_SA --channel-type mixed --C 32 --multiple-snr 1,4,7,10,13 --run-name snr_aware --epochs 60
python main.py --training --trainset CIFAR10 --model SwinJSCC_w/_SA --channel-type mixed --C 32 --multiple-snr 1,4,7,10,13 --type-aware --run-name snr_type_aware --epochs 60
python main.py --training --trainset CIFAR10 --model SwinJSCC_w/o_SAandRA --channel-type mixed --C 32 --multiple-snr 1,4,7,10,13 --run-name no_info --epochs 60
```

All reported runs were trained for 22 epochs (the original plan was 60). Seed-0 model B was first stopped at epoch 20 and then resumed to 22; its best checkpoint stayed at epoch 20.

### Extra seeds
`run_seeds.py` trains, evaluates and summarizes A, B and C for additional seeds (default 1 and 2), using the same value for all four seed flags. It skips finished runs, resumes interrupted ones, and refuses seed 0 so the existing runs are not overwritten.
```
python run_seeds.py
python run_seeds.py --seeds 1 --epochs 22
```
Runs are named `snr_aware_s1`, `snr_type_aware_s1`, `no_info_s1` and so on. At the end it runs `aggregate_seeds.py`, which can also be run alone. It prints each model's PSNR (averaged over SNR 1 to 13 dB) as mean ± standard deviation across seeds, plus the B−A, A−C and B−C differences with the number of seeds in which the first model is ahead, and writes `results/seed_summary.csv`.

### Results: A vs B vs C (3 seeds)
**Question:** does conditioning the Channel ModNet (used in both the encoder and the decoder) on the SNR, or on the SNR and the channel type, improve reconstruction over a model with no such conditioning?

**Headline:** B (SNR + channel type) is the best of the three, ahead of A and C in nearly every seed and channel, by about 0.2 to 0.8 dB PSNR. A (SNR-conditioned ModNet) shows no clear gain over C (no ModNet): its gap to C is small and changes sign with the seed and dataset. Each model was trained with 3 seeds for 22 epochs. The direction is consistent across seeds, but the seed-to-seed spread of the gaps is as large as the gaps themselves, so 3 seeds do not establish the differences (see the paired analysis below).

| Question | Main finding | Main limitation |
|---|---|---|
| Does channel-type information help? (B vs A) | B ahead in 3/3 seeds on 7 of 8 rows: +0.19 to +0.37 dB on CIFAR10, +0.35 to +0.76 dB on Kodak | B's ModNet input is larger (5 vs 1), and no uninformative-label control has been run |
| Does an SNR-conditioned ModNet beat no ModNet? (A vs C) | No clear difference: A is ahead in 2 of 3 seeds on most rows, but the mean gap is small (+0.06 to +0.10 dB on CIFAR10, −0.03 to −0.24 dB on Kodak) | C also lacks the ModNet layers, so this does not isolate SNR information. Paired per-seed analysis: run `aggregate_seeds.py` |
| Does B actually use the label? | A wrong label cost at most about 0.45 dB (seed 0 only) | One seed; it does not separate label information from extra capacity |
| Does it carry over to full-size images? | B ahead of C in 3/3 seeds on Kodak (+0.33 to +0.55 dB) | Out-of-distribution test; Kodak seed noise is 0.2 to 0.4 dB |

<details>
<summary>Full per-channel table and caveats</summary>

PSNR in dB, averaged over SNR 1 to 13 dB, as mean ± sample standard deviation across seeds 0, 1 and 2. "Wins" is the number of seeds in which the first model is ahead. Models were trained on CIFAR10 (32×32); the Kodak rows apply them to full-size images, so they are out-of-distribution.

| Dataset, channel | A (SNR) | B (SNR + type) | C (nothing) | B−A (wins) | A−C (wins) | B−C (wins) |
|---|---|---|---|---|---|---|
| CIFAR10, AWGN | 29.89 ± 0.13 | 30.27 ± 0.10 | 29.80 ± 0.14 | +0.37 (3/3) | +0.10 (2/3) | +0.47 (3/3) |
| CIFAR10, Rayleigh | 27.11 ± 0.11 | 27.29 ± 0.05 | 27.01 ± 0.05 | +0.19 (3/3) | +0.10 (2/3) | +0.28 (3/3) |
| CIFAR10, Rician | 28.81 ± 0.11 | 29.03 ± 0.07 | 28.70 ± 0.10 | +0.22 (3/3) | +0.10 (2/3) | +0.33 (3/3) |
| CIFAR10, impulsive | 29.90 ± 0.13 | 30.27 ± 0.10 | 29.83 ± 0.14 | +0.37 (3/3) | +0.06 (2/3) | +0.43 (3/3) |
| Kodak, AWGN | 33.01 ± 0.31 | 33.77 ± 0.25 | 33.22 ± 0.21 | +0.76 (3/3) | −0.21 (2/3) | +0.55 (3/3) |
| Kodak, Rayleigh | 30.58 ± 0.38 | 30.93 ± 0.14 | 30.60 ± 0.22 | +0.35 (2/3) | −0.03 (2/3) | +0.33 (3/3) |
| Kodak, Rician | 32.11 ± 0.34 | 32.61 ± 0.20 | 32.22 ± 0.23 | +0.50 (3/3) | −0.11 (2/3) | +0.39 (3/3) |
| Kodak, impulsive | 33.02 ± 0.31 | 33.76 ± 0.26 | 33.26 ± 0.21 | +0.75 (3/3) | −0.24 (0/3) | +0.51 (3/3) |

* B is ahead of C in all 3 seeds on every row (+0.28 to +0.55 dB), and ahead of A in all 3 seeds on 7 of 8 rows (Kodak Rayleigh: 2 of 3).
* A vs C: the mean gaps are small (+0.06 to +0.10 dB on CIFAR10, −0.03 to −0.24 dB on Kodak) and A leads in only 2 of 3 seeds on most rows. The table above compares each gap with the per-model standard deviations, which is not a significance test. `aggregate_seeds.py` now reports the paired per-seed differences (mean ± std, and `--per-seed` for each seed).
* Per-model standard deviations across seeds are about 0.05 to 0.14 dB on CIFAR10 and 0.2 to 0.4 dB on Kodak.
* Paired per-seed differences (`results/seed_summary.csv`): the spread of B−A is 0.13 to 0.22 dB on CIFAR10 and 0.43 to 0.46 dB on Kodak, comparable to the mean gaps (0.19 to 0.37 dB and 0.35 to 0.76 dB). With 3 seeds (2 degrees of freedom) a paired t-test needs |t| above about 4.3 at the 5% level; by rough arithmetic from that CSV only CIFAR10 Rayleigh B−C (t about 4.9) clears it. B ahead in all 3 seeds is consistent and suggestive, but it is not established with this many seeds.

Caveats:
* Only 3 seeds per model, so the standard deviations are rough. No significance test was run.
* All runs had the same 22-epoch budget (the original plan was 60). Validation PSNR was still rising near the end (for example model C, seed 0: 28.03 dB at epoch 15, 28.58 dB at epoch 21), but it also swings by up to about 1 dB between adjacent epochs (model A, seed 0: 28.56 dB at epoch 21, 27.52 dB at epoch 22), so `best.model` is a noisy choice. The validation curves are in `history/<run>/val_log.csv`.
* B's ModNet takes 5 inputs instead of 1, so part of B's gain may come from the extra input size rather than channel-type information. The wrong-label test on seed 0 (a wrong label costs at most about 0.45 dB) shows B uses the label but does not separate the two. An uninformative-label control has not been run.
* C has no ModNet layers, so an A vs C gap mixes missing SNR information with missing ModNet capacity.
* Averaging over SNR 1 to 13 dB hides behavior at −2 and 16 dB, which are outside the training SNRs. PSNR only; MS-SSIM is not aggregated across seeds.
* The impulsive channel is Bernoulli-Gaussian (impulse probability 0.1, impulse-to-background power ratio 10). It has the same total noise power as AWGN but a heavier-tailed distribution (per-component kurtosis about 9.75 from the code's constants, against 3 for a Gaussian). Its PSNR is nearly identical to AWGN for all three models, and this is not explained. `check_impulsive.py` measures the generated noise's variance and kurtosis, and a heavier setting (lower `IMPULSE_P`, higher `IMPULSE_RATIO` in `net/channel.py`) has not been tried. Until then, impulsive adds little as a test of channel-type information.
* None of this is compared with the paper's Fig. 10a/10d.

</details>

### Resuming training
Each epoch writes `history/NAME/resume.pt` (model, optimizer, all random-number states and data-loader state). To continue an interrupted run, repeat the same command with the same `--run-name` and add `--resume`:

```
python main.py --training --trainset CIFAR10 --model SwinJSCC_w/_SA --channel-type mixed --C 32 --multiple-snr 1,4,7,10,13 --run-name snr_aware --epochs 60 --resume
```

`--epochs` is the final epoch count, not the number of extra epochs. A resumed run matches an uninterrupted one exactly (same logs, identical weights), which `tests/test_resume.py` checks. Runs started before resume support existed have no `resume.pt` and cannot be resumed.

### Evaluation and plots
* `eval_channels.py` evaluates a checkpoint per channel and SNR on the CIFAR10 test set and optionally Kodak, and writes a CSV. `--label-sweep` also tests every (true channel, wrong label) pair for a type-aware model. `--model` selects the architecture (default `SwinJSCC_w/_SA`; use `SwinJSCC_w/o_SAandRA` for model C).
```
python eval_channels.py --checkpoint history/snr_type_aware/models/best.model --run-name snr_type_aware --type-aware 1 --out results/snr_type_aware.csv --kodak-dir PATH_TO_KODAK --label-sweep
python eval_channels.py --checkpoint history/no_info/models/best.model --run-name no_info --type-aware 0 --model SwinJSCC_w/o_SAandRA --out results/no_info.csv --kodak-dir PATH_TO_KODAK
```
* `plot_ab.py` plots two or three models' curves against each other and the label-confusion heatmap.
```
python plot_ab.py --a results/snr_aware.csv --b results/snr_type_aware.csv --c results/no_info.csv --labels results/snr_type_aware_labels.csv --out plots
```
`--c` (model C) and `--labels` (the CSV written by `--label-sweep`) are optional.
* `aggregate_seeds.py` summarizes results across seeds (see Extra seeds above).
* `plot_swinjscc.py` plots the pretrained-checkpoint sweeps from `SwinJSCC_results.csv`.
* `--save-images` saves originals, reconstructions, side-by-side images and per-image metrics when evaluating.
* `--model-path path/to/x.model` selects a checkpoint without editing code.

### Evaluating the pretrained checkpoints
Use `--trainset DIV2K` to select the high-resolution architecture, even in evaluation mode. No training data is needed. `--model_size` must match the checkpoint. In evaluation mode `--C` and `--multiple-snr` are comma lists, and the full SNR × C grid is evaluated.

### Tests
```
python -m pytest tests -q
```
The tests need a CUDA GPU. The training smoke and resume tests take a minute or two.

### Environment notes
The code is CUDA-only. This fork was run with PyTorch 2.10 on an RTX 5060. The upstream note below about torch > 1.12 still applies, and results from the pretrained SA and RA checkpoints may differ from the paper in newer environments. On Windows consoles, set `PYTHONIOENCODING=utf-8` to avoid a harmless logging UnicodeEncodeError.

## Installation
We implement SwinJSCC under Python 3.8 and PyTorch 1.9.

Note: We found that when using w/ SA & RA for inference in a torch > 1.12 environment, the results are inconsistent with those reported in the paper. Therefore, it is recommended to use the same version of PyTorch as the training environment (torch <= 1.12) for inference.


## Usage

All pretrained models are in [Google Drive](https://drive.google.com/drive/folders/1_EouRY4yYvMCtamX2ReBzEd5YBQbyesc?usp=sharing).

* cbr = C/(2^(2i)*3*2), i denotes the downsample number. For CIFAR10, i=2; for HR_image, i=4.
* SwinJSCC_w/o_SAandRA model is the SwinJSCC model without Channel ModNet module and Rate ModNet which is trained on a fixed SNR  and rate. SwinJSCC_w/_SA model is the SwinJSCC model with Channel ModNet module which is trained on various SNRs and a fixed rate. SwinJSCC_w/_RA model is the SwinJSCC model with Rate ModNet module which is trained on various rates and a fixed SNR. SwinJSCC_w/_SAandRA model is the SwinJSCC model with Rate ModNet module and Channel ModNet module which is trained on various rates and SNRs.
* 'multiple-snr' decides use either fixed or random SNR to train the model. For models which without Channel ModNet module, 'multiple-snr' is set as a fixed SNR. For models which with Channel ModNet module, 'muliple-snr' can be set as both fixed or random SNRs.
* 'C' decides use either fixed or random rate to train the model. For models which without Rate ModNet module, 'C' is set as a fixed rate. For models which with Rate ModNet module, 'C' can be set as both fixed or random rates.
* 'model_size' decides model params size, we set three model sizes, e.g. small, base, large.
* for high-resolution images, we can firstly train the SwinJSCC_W/O model. Then, the SwinJSCC_W/O model is used as a pre-training model to train the whole SwinJSCC model.
* You can apply our method on your own images.
```
python main.py --training --trainset {CIFAR10/DIV2K} --testset {CIFAR10/kodak/CLIC21} -- distortion-metric {MSE/MS-SSIM} --model {'SwinJSCC_w/o_SAandRA'/'SwinJSCC_w/_SA'/'SwinJSCC_w/_RA'/'SwinJSCC_w/_SAandRA'} --channel-type {awgn/rayleigh} --C {bottleneck dimension} --multiple-snr {random or fixed snr} --model_size {SwinJSCC model size}
```

### For SwinJSCC_w/o_SAandRA model 

*e.g. cbr = 0.0625, snr = 10, metric = PSNR, channel = AWGN

```
e.g.
python main.py --trainset DIV2K --testset kodak -- distortion-metric MSE --model SwinJSCC_w/o_SAandRA model --channel-type awgn --C 96 -- multiple-snr 10 --model_size base
```

You can apply our method on your own images.

### For SwinJSCC_w/_SA model 

*e.g. cbr = 0.0625, snr = 1,4,7,10,13, metric = PSNR, channel = AWGN

```
e.g.
python main.py --trainset DIV2K --testset kodak --distortion-metric MSE --model SwinJSCC_w/_SA --channel-type awgn --C 96 --multiple-snr 1,4,7,10,13 --model_size base
```
### For SwinJSCC_w/_RA model 
*e.g. cbr = 0.0208,0.0416,0.0625,0.0833,0.125, snr = 10, metric = PSNR, channel = AWGN

```
e.g.
python main.py --trainset DIV2K --testset kodak --distortion-metric MSE --model SwinJSCC_w/_RA --channel-type awgn --C 32,64,96,128,192 --multiple-snr 10 --model_size base
```

### For SwinJSCC_w/_SAandRA model 
*e.g. cbr = 0.0208,0.0416,0.0625,0.0833,0.125, snr = 1,4,7,10,13, metric = PSNR, channel = AWGN

```
e.g.
python main.py --trainset DIV2K --testset kodak --distortion-metric MSE --model SwinJSCC_w/_SAandRA --channel-type awgn --C 32,64,96,128,192 --multiple-snr 1,4,7,10,13 --model_size base
```

>If you want to train this model, please add '--training'. 


## Citation

If you find this work useful for your research, please cite:

```
@ARTICLE{10589474,
  author={Yang, Ke and Wang, Sixian and Dai, Jincheng and Qin, Xiaoqi and Niu, Kai and Zhang, Ping},
  journal={IEEE Transactions on Cognitive Communications and Networking}, 
  title={SwinJSCC: Taming Swin Transformer for Deep Joint Source-Channel Coding}, 
  year={2024},
  volume={},
  number={},
  pages={1-1},
  keywords={Transformers;Adaptation models;Signal to noise ratio;Convolutional neural networks;Wireless communication;Vectors;Image coding;Joint source-channel coding;Swin Transformer;attention mechanism;image communications},
  doi={10.1109/TCCN.2024.3424842}
}
```

## Acknowledgement
The implementation is based on [Swin Transformer](https://github.com/microsoft/Swin-Transformer).

## Related links
* BPG image format by _Fabrice Bellard_: https://bellard.org/bpg
* Sionna An Open-Source Library for Next-Generation Physical Layer Research: https://github.com/NVlabs/sionna
* DIV2K image dataset: https://data.vision.ee.ethz.ch/cvl/DIV2K/
* Kodak image dataset: http://r0k.us/graphics/kodak/
* CLIC image dataset:  http://compression.cc
