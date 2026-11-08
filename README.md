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
```

### Resuming training
Each epoch writes `history/NAME/resume.pt` (model, optimizer, all random-number states and data-loader state). To continue an interrupted run, repeat the same command with the same `--run-name` and add `--resume`:

```
python main.py --training --trainset CIFAR10 --model SwinJSCC_w/_SA --channel-type mixed --C 32 --multiple-snr 1,4,7,10,13 --run-name snr_aware --epochs 60 --resume
```

`--epochs` is the final epoch count, not the number of extra epochs. A resumed run matches an uninterrupted one exactly (same logs, identical weights), which `tests/test_resume.py` checks. Runs started before resume support existed have no `resume.pt` and cannot be resumed.

### Evaluation and plots
* `eval_channels.py` evaluates a checkpoint per channel and SNR on the CIFAR10 test set and optionally Kodak, and writes a CSV. `--label-sweep` also tests every (true channel, wrong label) pair for a type-aware model.
```
python eval_channels.py --checkpoint history/snr_type_aware/models/best.model --run-name snr_type_aware --type-aware 1 --out results/snr_type_aware.csv --kodak-dir PATH_TO_KODAK --label-sweep
```
* `plot_ab.py` plots two models' curves against each other and the label-confusion heatmap.
```
python plot_ab.py --a results/snr_aware.csv --b results/snr_type_aware.csv --labels results/snr_type_aware_labels.csv --out plots
```
`--labels` is optional and takes the CSV written by `--label-sweep`.
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
