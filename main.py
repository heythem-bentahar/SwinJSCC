import torch.optim as optim
from net.network import SwinJSCC
from data.datasets import get_loader
from utils import *

# torch.backends.cudnn.benchmark = True
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
from datetime import datetime
import torch.nn as nn
import argparse
from loss.distortion import *
import time
import torchvision
import csv
import sys
import math
from pathlib import Path
from PIL import Image, ImageDraw

parser = argparse.ArgumentParser(description='SwinJSCC')
parser.add_argument('--training', action='store_true', help='training or testing')
parser.add_argument('--trainset', type=str, default='DIV2K', choices=['CIFAR10', 'DIV2K'], help='train dataset name')
parser.add_argument('--testset', type=str, default='ffhq', choices=['kodak', 'CLIC21', 'ffhq'], help='specify the testset for HR models')
parser.add_argument('--distortion-metric', type=str, default='MSE', choices=['MSE', 'MS-SSIM'], help='evaluation metrics')
parser.add_argument('--model',
                    type=str,
                    default='SwinJSCC_w/_SAandRA',
                    choices=['SwinJSCC_w/o_SAandRA', 'SwinJSCC_w/_SA', 'SwinJSCC_w/_RA', 'SwinJSCC_w/_SAandRA'],
                    help='SwinJSCC model or SwinJSCC without channel ModNet or rate ModNet')
parser.add_argument('--channel-type', type=str, default='awgn', choices=['awgn', 'rayleigh', 'rician', 'impulsive', 'mixed'], help='wireless channel model; mixed = random type per batch')
parser.add_argument('--type-aware', action='store_true', help='feed channel type (one-hot) to the Channel ModNet together with SNR')
parser.add_argument('--model-path', type=str, default=None, help='checkpoint to load (testing or fine-tuning); leave empty to train from scratch')
parser.add_argument('--C', type=str, default='96', help='bottleneck dimension')
parser.add_argument('--multiple-snr', type=str, default='10', help='random or fixed snr')
parser.add_argument('--model_size', type=str, default='base', choices=['small', 'base', 'large'], help='SwinJSCC model size')
parser.add_argument('--save-images', action='store_true', help='save HR originals, reconstructions, comparisons and scores')
parser.add_argument('--run-name', type=str, default=None, help='output dir name under history/ (required with --training on CIFAR10)')
parser.add_argument('--epochs', type=int, default=20, help='training epochs')
parser.add_argument('--init-seed', type=int, default=0, help='seed for model initialization')
parser.add_argument('--sampler-seed', type=int, default=0, help='seed for (channel, SNR) sampling')
parser.add_argument('--noise-seed', type=int, default=0, help='seed for channel noise')
parser.add_argument('--order-seed', type=int, default=0, help='seed for training data order and flips')
parser.add_argument('--resume', action='store_true', help='continue a CIFAR10 training run from history/<run-name>/resume.pt')
parser.add_argument('--train-subset', type=int, default=None, help='use only the first N training images (smoke tests)')
parser.add_argument('--val-subset', type=int, default=None, help='use only the first N validation images (smoke tests)')
args = parser.parse_args()
if args.training and args.trainset == 'CIFAR10' and not args.run_name:
    parser.error('--run-name is required when training on CIFAR10')


class config():
    # Checkpoint for evaluation. Change this path inside the script.
    # Training starts from scratch unless --model-path is explicitly supplied.
    model_path = r".\history\2026-11-07_02-13-44\models\2026-11-07_02-13-44_EP7.model"
    seed = 42
    pass_channel = True
    CUDA = True
    device = torch.device("cuda:0")
    norm = False
    # logger
    print_step = 100
    plot_step = 10000
    filename = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    workdir = './history/{}'.format(args.run_name if args.run_name else filename)
    log = workdir + '/Log_{}.log'.format(filename)
    samples = workdir + '/samples'
    models = workdir + '/models'
    logger = None

    # training details
    normalize = False
    learning_rate = 0.0001
    tot_epoch = args.epochs

    if args.trainset == 'CIFAR10':
        save_model_freq = 1
        image_dims = (3, 32, 32)
        train_data_dir = "./CIFAR10/"
        test_data_dir = "./CIFAR10/"
        batch_size = 128
        downsample = 2
        channel_number = int(args.C)
        encoder_kwargs = dict(
            model=args.model,
            img_size=(image_dims[1], image_dims[2]),
            patch_size=2,
            in_chans=3,
            embed_dims=[128, 256],
            depths=[2, 4],
            num_heads=[4, 8],
            C=channel_number,
            window_size=2,
            mlp_ratio=4.,
            qkv_bias=True,
            qk_scale=None,
            norm_layer=nn.LayerNorm,
            patch_norm=True,
        )
        decoder_kwargs = dict(
            model=args.model,
            img_size=(image_dims[1], image_dims[2]),
            embed_dims=[256, 128],
            depths=[4, 2],
            num_heads=[8, 4],
            C=channel_number,
            window_size=2,
            mlp_ratio=4.,
            qkv_bias=True,
            qk_scale=None,
            norm_layer=nn.LayerNorm,
            patch_norm=True,
        )
    elif args.trainset == 'DIV2K':
        save_model_freq = 100
        image_dims = (3, 256, 256)
        base_path = "/media/D/Dataset/DIV2K/"
        if args.testset == 'kodak':
            test_data_dir = ["D:/KFUPM/261/COE 540 - Computer Networks/Project/CODE/SwinJSCC-main/Kodak"]
        elif args.testset == 'CLIC21':
            test_data_dir = ["/media/D/Dataset/HR_Image_dataset/clic2021/test/"]
        elif args.testset == 'ffhq':
            test_data_dir = ["/media/D/yangke/SwinJSCC/data/ffhq/"]

        train_data_dir = [
            base_path + '/clic2020/**', base_path + '/clic2021/train', base_path + '/clic2021/valid', base_path + '/clic2022/val', base_path + '/DIV2K_train_HR', base_path + '/DIV2K_valid_HR'
        ]
        batch_size = 16
        downsample = 4
        if args.model == 'SwinJSCC_w/o_SAandRA' or args.model == 'SwinJSCC_w/_SA':
            channel_number = int(args.C)
        else:
            channel_number = None

        if args.model_size == 'small':
            encoder_kwargs = dict(
                model=args.model,
                img_size=(image_dims[1], image_dims[2]),
                patch_size=2,
                in_chans=3,
                embed_dims=[128, 192, 256, 320],
                depths=[2, 2, 2, 2],
                num_heads=[4, 6, 8, 10],
                C=channel_number,
                window_size=8,
                mlp_ratio=4.,
                qkv_bias=True,
                qk_scale=None,
                norm_layer=nn.LayerNorm,
                patch_norm=True,
            )
            decoder_kwargs = dict(
                model=args.model,
                img_size=(image_dims[1], image_dims[2]),
                embed_dims=[320, 256, 192, 128],
                depths=[2, 2, 2, 2],
                num_heads=[10, 8, 6, 4],
                C=channel_number,
                window_size=8,
                mlp_ratio=4.,
                qkv_bias=True,
                qk_scale=None,
                norm_layer=nn.LayerNorm,
                patch_norm=True,
            )
        elif args.model_size == 'base':
            encoder_kwargs = dict(
                model=args.model,
                img_size=(image_dims[1], image_dims[2]),
                patch_size=2,
                in_chans=3,
                embed_dims=[128, 192, 256, 320],
                depths=[2, 2, 6, 2],
                num_heads=[4, 6, 8, 10],
                C=channel_number,
                window_size=8,
                mlp_ratio=4.,
                qkv_bias=True,
                qk_scale=None,
                norm_layer=nn.LayerNorm,
                patch_norm=True,
            )
            decoder_kwargs = dict(
                model=args.model,
                img_size=(image_dims[1], image_dims[2]),
                embed_dims=[320, 256, 192, 128],
                depths=[2, 6, 2, 2],
                num_heads=[10, 8, 6, 4],
                C=channel_number,
                window_size=8,
                mlp_ratio=4.,
                qkv_bias=True,
                qk_scale=None,
                norm_layer=nn.LayerNorm,
                patch_norm=True,
            )
        elif args.model_size == 'large':
            encoder_kwargs = dict(
                model=args.model,
                img_size=(image_dims[1], image_dims[2]),
                patch_size=2,
                in_chans=3,
                embed_dims=[128, 192, 256, 320],
                depths=[2, 2, 18, 2],
                num_heads=[4, 6, 8, 10],
                C=channel_number,
                window_size=8,
                mlp_ratio=4.,
                qkv_bias=True,
                qk_scale=None,
                norm_layer=nn.LayerNorm,
                patch_norm=True,
            )
            decoder_kwargs = dict(
                model=args.model,
                img_size=(image_dims[1], image_dims[2]),
                embed_dims=[320, 256, 192, 128],
                depths=[2, 18, 2, 2],
                num_heads=[10, 8, 6, 4],
                C=channel_number,
                window_size=8,
                mlp_ratio=4.,
                qkv_bias=True,
                qk_scale=None,
                norm_layer=nn.LayerNorm,
                patch_norm=True,
            )


if args.trainset == 'CIFAR10':
    CalcuSSIM = MS_SSIM(window_size=3, data_range=1., levels=4, channel=3).cuda()
else:
    CalcuSSIM = MS_SSIM(data_range=1., levels=4, channel=3).cuda()


def load_weights(model_path):
    pretrained = torch.load(model_path)
    net.load_state_dict(pretrained, strict=True)
    del pretrained


def train_one_epoch(args):
    net.train()
    elapsed, losses, psnrs, msssims, cbrs, snrs = [AverageMeter() for _ in range(6)]
    metrics = [elapsed, losses, psnrs, msssims, cbrs, snrs]
    global global_step
    if args.trainset == 'CIFAR10':
        for batch_idx, (input, label) in enumerate(train_loader):
            start_time = time.time()
            global_step += 1
            input = input.cuda()
            recon_image, CBR, SNR, mse, loss_G = net(input)
            loss = loss_G
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            elapsed.update(time.time() - start_time)
            losses.update(loss.item())
            cbrs.update(CBR)
            snrs.update(SNR)
            if mse.item() > 0:
                psnr = 10 * (torch.log(255. * 255. / mse) / np.log(10))
                psnrs.update(psnr.item())
                msssim = 1 - CalcuSSIM(input, recon_image.clamp(0., 1.)).mean().item()
                msssims.update(msssim)

            if (global_step % config.print_step) == 0:
                process = (global_step % train_loader.__len__()) / (train_loader.__len__()) * 100.0
                log = (' | '.join([
                    f'Epoch {epoch}',
                    f'Step [{global_step % train_loader.__len__()}/{train_loader.__len__()}={process:.2f}%]',
                    f'Time {elapsed.val:.3f}',
                    f'Loss {losses.val:.3f} ({losses.avg:.3f})',
                    f'CBR {cbrs.val:.4f} ({cbrs.avg:.4f})',
                    f'SNR {snrs.val:.1f} ({snrs.avg:.1f})',
                    f'PSNR {psnrs.val:.3f} ({psnrs.avg:.3f})',
                    f'MSSSIM {msssims.val:.3f} ({msssims.avg:.3f})',
                    f'Lr {cur_lr}',
                ]))
                logger.info(log)
                for i in metrics:
                    i.clear()
    else:
        for batch_idx, input in enumerate(train_loader):
            start_time = time.time()
            global_step += 1
            input = input.cuda()
            recon_image, CBR, SNR, mse, loss_G = net(input)
            loss = loss_G
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            elapsed.update(time.time() - start_time)
            losses.update(loss.item())
            cbrs.update(CBR)
            snrs.update(SNR)
            if mse.item() > 0:
                psnr = 10 * (torch.log(255. * 255. / mse) / np.log(10))
                psnrs.update(psnr.item())
                msssim = 1 - CalcuSSIM(input, recon_image.clamp(0., 1.)).mean().item()
                msssims.update(msssim)

            if (global_step % config.print_step) == 0:
                process = (global_step % train_loader.__len__()) / (train_loader.__len__()) * 100.0
                log = (' | '.join([
                    f'Epoch {epoch}',
                    f'Step [{global_step % train_loader.__len__()}/{train_loader.__len__()}={process:.2f}%]',
                    f'Time {elapsed.val:.3f}',
                    f'Loss {losses.val:.3f} ({losses.avg:.3f})',
                    f'CBR {cbrs.val:.4f} ({cbrs.avg:.4f})',
                    f'SNR {snrs.val:.1f} ({snrs.avg:.1f})',
                    f'PSNR {psnrs.val:.3f} ({psnrs.avg:.3f})',
                    f'MSSSIM {msssims.val:.3f} ({msssims.avg:.3f})',
                    f'Lr {cur_lr}',
                ]))
                logger.info(log)
                for i in metrics:
                    i.clear()
    for i in metrics:
        i.clear()


def save_visual_result(original, reconstructed, name, snr, rate, cbr, psnr, msssim):
    root = Path(config.workdir) / 'visuals' / args.channel_type
    originals = root / 'originals'
    condition = root / f'SNR_{snr}_C_{rate}'
    originals.mkdir(parents=True, exist_ok=True)
    condition.mkdir(parents=True, exist_ok=True)
    stem = Path(name).stem
    original_pil = torchvision.transforms.functional.to_pil_image(original[0].detach().cpu().clamp(0, 1))
    recon_pil = torchvision.transforms.functional.to_pil_image(reconstructed[0].detach().cpu().clamp(0, 1))
    original_pil.save(originals / f'{stem}.png')
    recon_pil.save(condition / f'{stem}_reconstructed.png')
    width, height = original_pil.size
    comparison = Image.new('RGB', (width * 2, height + 40), 'white')
    comparison.paste(original_pil, (0, 40))
    comparison.paste(recon_pil, (width, 40))
    draw = ImageDraw.Draw(comparison)
    draw.text((10, 12), 'Original', fill='black')
    draw.text((width + 10, 5), f'SNR={snr} dB | CBR={float(cbr):.5f}', fill='black')
    draw.text((width + 10, 22), f'PSNR={psnr:.2f} dB | MS-SSIM={msssim:.4f}', fill='black')
    comparison.save(condition / f'{stem}_comparison.png')
    csv_path = condition / 'metrics.csv'
    new_file = not csv_path.exists()
    with csv_path.open('a', newline='', encoding='utf-8') as handle:
        writer = csv.writer(handle)
        if new_file:
            writer.writerow(['image', 'SNR_dB', 'C', 'CBR', 'PSNR_dB', 'MS_SSIM'])
        writer.writerow([name, snr, rate, float(cbr), psnr, msssim])


def test():
    config.isTrain = False
    net.eval()
    elapsed, psnrs, msssims, snrs, cbrs = [AverageMeter() for _ in range(5)]
    metrics = [elapsed, psnrs, msssims, snrs, cbrs]
    multiple_snr = args.multiple_snr.split(",")
    for i in range(len(multiple_snr)):
        multiple_snr[i] = int(multiple_snr[i])
    channel_number = args.C.split(",")
    for i in range(len(channel_number)):
        channel_number[i] = int(channel_number[i])
    results_snr = np.zeros((len(multiple_snr), len(channel_number)))
    results_cbr = np.zeros((len(multiple_snr), len(channel_number)))
    results_psnr = np.zeros((len(multiple_snr), len(channel_number)))
    results_msssim = np.zeros((len(multiple_snr), len(channel_number)))
    for i, SNR in enumerate(multiple_snr):
        for j, rate in enumerate(channel_number):
            with torch.no_grad():
                if args.trainset == 'CIFAR10':
                    for batch_idx, (input, label) in enumerate(test_loader):
                        start_time = time.time()
                        input = input.cuda()
                        recon_image, CBR, SNR, mse, loss_G = net(input, SNR, rate)

                        elapsed.update(time.time() - start_time)
                        cbrs.update(CBR)
                        snrs.update(SNR)
                        if mse.item() > 0:
                            psnr = 10 * (torch.log(255. * 255. / mse) / np.log(10))
                            psnrs.update(psnr.item())
                            msssim = 1 - CalcuSSIM(input, recon_image.clamp(0., 1.)).mean().item()
                            msssims.update(msssim)

                        log = (' | '.join([
                            f'Time {elapsed.val:.3f}',
                            f'CBR {cbrs.val:.4f} ({cbrs.avg:.4f})',
                            f'SNR {snrs.val:.1f}',
                            f'PSNR {psnrs.val:.3f} ({psnrs.avg:.3f})',
                            f'MSSSIM {msssims.val:.3f} ({msssims.avg:.3f})',
                            f'Lr {cur_lr}',
                        ]))
                        logger.info(log)
                else:
                    for batch_idx, batch in enumerate(test_loader):
                        input, names = batch
                        start_time = time.time()
                        input = input.cuda()
                        recon_image, CBR, SNR, mse, loss_G = net(input, SNR, rate)
                        if args.save_images:
                            image_psnr = float('inf') if mse.item() == 0 else 10 * math.log10(255. ** 2 / mse.item())
                            image_msssim = CalcuSSIM(input, recon_image.clamp(0., 1.)).mean().item()
                            save_visual_result(input, recon_image, names[0], SNR, rate, CBR, image_psnr, image_msssim)
                        elapsed.update(time.time() - start_time)
                        cbrs.update(CBR)
                        snrs.update(SNR)
                        if mse.item() > 0:
                            psnr = 10 * (torch.log(255. * 255. / mse) / np.log(10))
                            psnrs.update(psnr.item())
                            msssim = 1 - CalcuSSIM(input, recon_image.clamp(0., 1.)).mean().item()
                            msssims.update(msssim)
                            MSSSIM = -10 * math.log10(1 - msssim)
                        log = (' | '.join([
                            f'Time {elapsed.val:.3f}',
                            f'CBR {cbrs.val:.4f} ({cbrs.avg:.4f})',
                            f'SNR {snrs.val:.1f}',
                            f'PSNR {psnrs.val:.3f} ({psnrs.avg:.3f})',
                            f'MSSSIM {msssims.val:.3f} ({msssims.avg:.3f})',
                            f'Lr {cur_lr}',
                        ]))
                        logger.info(log)
            results_snr[i, j] = snrs.avg
            results_cbr[i, j] = cbrs.avg
            results_psnr[i, j] = psnrs.avg
            results_msssim[i, j] = msssims.avg
            for t in metrics:
                t.clear()

    print("SNR: {}".format(results_snr.tolist()))
    print("CBR: {}".format(results_cbr.tolist()))
    print("PSNR: {}".format(results_psnr.tolist()))
    print("MS-SSIM: {}".format(results_msssim.tolist()))
    if args.save_images:
        print("Saved images to:", (Path(config.workdir) / "visuals").resolve())
    print("Finish Test!")


def evaluate_val(net, val_loader, channel, snr, noise_seed=12345):
    """Mean MSE-based PSNR on the validation loader for one (channel, SNR).
    Uses a dedicated noise generator and restores the training one afterwards."""
    saved_gen = net.channel.gen
    net.channel.set_noise_seed(noise_seed)
    total_mse, total_n = 0.0, 0
    try:
        with torch.no_grad():
            for images, _ in val_loader:
                images = images.cuda()
                _, _, _, mse, _ = net(images, snr, int(args.C), channel)
                total_mse += mse.item() * images.shape[0]
                total_n += images.shape[0]
    finally:
        net.channel.gen = saved_gen
    return 10 * math.log10(255. ** 2 / (total_mse / total_n))


def validate(net, val_loader):
    net.eval()
    per_channel = {}
    for chan in net.chan_types:
        per_channel[chan] = float(np.mean([evaluate_val(net, val_loader, chan, snr)
                                           for snr in [1, 4, 7, 10, 13]]))
    net.train()
    return float(np.mean(list(per_channel.values()))), per_channel


def run_cifar_training():
    from net.sampling import ConditionSampler
    from data.datasets import get_cifar_loaders, get_loader_state, set_loader_state
    import random
    os.makedirs(config.models, exist_ok=True)
    logger = logger_configuration(config, save_log=True)
    logger.info(config.__dict__)
    seed_torch()
    snrs = [int(x) for x in args.multiple_snr.split(',')]
    sampler = ConditionSampler(args.sampler_seed, snrs, ['awgn', 'rayleigh', 'rician', 'impulsive'])
    torch.manual_seed(args.init_seed)
    torch.cuda.manual_seed_all(args.init_seed)
    net = SwinJSCC(args, config, sampler=sampler)
    seed_torch()  # identical global state after construction for every run
    net.channel.set_noise_seed(args.noise_seed)
    net = net.cuda()
    train_loader, val_loader, _ = get_cifar_loaders(
        config.train_data_dir, config.batch_size, order_seed=args.order_seed,
        train_subset=args.train_subset, val_subset=args.val_subset)
    optimizer = optim.Adam(net.parameters(), lr=config.learning_rate)
    workdir = Path(config.workdir)
    resume_path = workdir / 'resume.pt'
    best, step, start_epoch = -1.0, 0, 0
    if args.resume:
        if not resume_path.exists():
            raise SystemExit('--resume: no checkpoint found at {}'.format(resume_path))
        ck = torch.load(resume_path, weights_only=False)
        net.load_state_dict(ck['model'])
        optimizer.load_state_dict(ck['optimizer'])
        start_epoch, step, best = ck['epoch'], ck['step'], ck['best']
        sampler.rng.setstate(ck['sampler_rng'])
        sampler.log = ck['sampler_log']
        net.channel.gen.set_state(ck['noise_gen'])
        set_loader_state(train_loader, ck['loader'])
        torch.set_rng_state(ck['torch_rng'])
        torch.cuda.set_rng_state(ck['cuda_rng'])
        random.setstate(ck['py_rng'])
        logger.info('Resumed from epoch {} (step {}, best val PSNR {:.3f})'.format(start_epoch, step, best))
    else:
        with open(workdir / 'val_log.csv', 'w', newline='') as f:
            csv.writer(f).writerow(['epoch', 'mean_psnr'] + ['psnr_' + c for c in net.chan_types])
    for epoch in range(start_epoch, args.epochs):
        net.train()
        for images, _ in train_loader:
            step += 1
            images = images.cuda()
            _, cbr, snr, mse, loss = net(images, None, int(args.C))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if step % config.print_step == 0:
                logger.info('Epoch {} | step {} | loss {:.4f} | CBR {:.4f} | SNR {}'.format(
                    epoch + 1, step, loss.item(), cbr, snr))
        mean_psnr, per_channel = validate(net, val_loader)
        with open(workdir / 'val_log.csv', 'a', newline='') as f:
            csv.writer(f).writerow([epoch + 1, mean_psnr] + [per_channel[c] for c in net.chan_types])
        with open(workdir / 'sampler_log.csv', 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['step', 'channel', 'snr'])
            w.writerows([(i + 1, c, s) for i, (c, s) in enumerate(sampler.log)])
        save_model(net, config.models + '/last.model')
        if mean_psnr > best:
            best = mean_psnr
            save_model(net, config.models + '/best.model')
        logger.info('Epoch {} | val PSNR {:.3f} (best {:.3f})'.format(epoch + 1, mean_psnr, best))
        torch.save({'model': net.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch + 1,
                    'step': step, 'best': best, 'sampler_rng': sampler.rng.getstate(),
                    'sampler_log': sampler.log, 'noise_gen': net.channel.gen.get_state(),
                    'loader': get_loader_state(train_loader), 'torch_rng': torch.get_rng_state(),
                    'cuda_rng': torch.cuda.get_rng_state(), 'py_rng': random.getstate()},
                   str(resume_path) + '.tmp')
        os.replace(str(resume_path) + '.tmp', resume_path)
    print('Finish Training! best val PSNR {:.3f}'.format(best))


if __name__ == '__main__':
    if args.training and args.trainset == 'CIFAR10':
        run_cifar_training()
        sys.exit(0)
    if args.training:
        os.makedirs(config.models, exist_ok=True)
    seed_torch()
    logger = logger_configuration(config, save_log=False)
    logger.info(config.__dict__)
    torch.manual_seed(seed=config.seed)
    net = SwinJSCC(args, config)
    # Use the internal checkpoint path for testing; preserve optional CLI override.
    model_path = args.model_path
    if model_path is None and not args.training:
        model_path = config.model_path
    if not args.training and not model_path:
        raise ValueError("Set config.model_path to a checkpoint before testing.")
    if model_path:
        load_weights(model_path)
    net = net.cuda()
    model_params = [{'params': net.parameters(), 'lr': 0.0001}]
    train_loader, test_loader = get_loader(args, config)
    cur_lr = config.learning_rate
    optimizer = optim.Adam(model_params, lr=cur_lr)
    global_step = 0
    steps_epoch = global_step // len(train_loader) if args.training else 0
    if args.training:
        for epoch in range(steps_epoch, config.tot_epoch):
            train_one_epoch(args)
            if (epoch + 1) % config.save_model_freq == 0:
                save_model(net, save_path=config.models + '/{}_EP{}.model'.format(config.filename, epoch + 1))
                test()
    else:
        test()
