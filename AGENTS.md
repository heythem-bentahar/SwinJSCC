# SwinJSCC Repository Agent Guidelines

## Setup & Commands

### Installation
- Python 3.8, PyTorch 1.9 (critical: torch ≤ 1.12 for SA/RA inference compatibility)
- Install dependencies via requirements file or pip freeze

### Running Experiments
```bash
# Basic training/inference
python main.py --trainset {CIFAR10/DIV2K} --testset {CIFAR10/kodak/CLIC21} \
  --distortion-metric {MSE/MS-SSIM} --model {model_type} \
  --channel-type {awgn/rayleigh} --C {bottleneck_dimension}

# Model variants and hyperparameters:
# --multiple-snr {fixed or random SNRs for training}
# --model_size {small/base/large}
```

### Test a single experiment
- Train on DIV2K, test on Kodak with MSE metric using base model at 96 bottleneck dimension with AWGN channel and SNR=10: `python main.py --trainset DIV2K --testset kodak --distortion-metric MSE --model SwinJSCC_w/o_SAandRA --channel-type awgn --C 96 --multiple-snr 10 --model_size base`

### Model variants
- **SwinJSCC_w/o_SAandRA**: No Channel/Rate ModNet, fixed SNR/rate training
- **SwinJSCC_w/_SA**: Includes Channel ModNet for various SNRs, fixed rate  
- **SwinJSCC_w/_RA**: Includes Rate ModNet for various rates, fixed SNR
- **SwinJSCC_w/_SAandRA**: Both modules, varied SNR and rate training

## Architecture

### Package Structure
- `net/`: Encoder, decoder, channel models (awgn.py, rayleigh.py), modulation network (modnet.py)
- `data/datasets.py`: Data loading utilities for CIFAR10, DIV2K, Kodak, CLIC21 datasets
- `utils.py`: Utility functions for training/inference pipelines
- `loss/distortion.py`: MSE and MS-SSIM distortion metrics

### Model sizes
- **small**: Lower parameter count
- **base**: Default model size  
- **large**: Higher capacity (recommended for high-res images)

## Known Issues / Quirks

1. **PyTorch version matters**: Using torch > 1.12 causes inconsistency with SA/RA modules. Stick to ≤1.12 for inference.
2. **High-resolution images**: For DIV2K/HR_image, train SwinJSCC_W/O first as pre-training model before fine-tuning full SwinJSCC

## Datasets
- CIFAR10: Standard classification dataset (i=2 downsampling)
- DIV2K: High-res image training set
- Kodak: Test images (http://r0k.us/graphics/kodak/)
- CLIC21: Compression test set (http://compression.cc)

## Citation
@ARTICLE{10589474, author={Yang, Ke and Wang, Sixian and Dai, Jincheng and Qin, Xiaoqi and Niu, Kai and Zhang, Ping}, journal={IEEE Transactions on Cognitive Communications and Networking}, title={SwinJSCC: Taming Swin Transformer for Deep Joint Source-Channel Coding}, year={2024} }