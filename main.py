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
args = parser.parse_args()


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
    workdir = './history/{}'.format(filename)
    log = workdir + '/Log_{}.log'.format(filename)
    samples = workdir + '/samples'
    models = workdir + '/models'
    logger = None

    # training details
    normalize = False
    learning_rate = 0.0001
    tot_epoch = 20

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


if __name__ == '__main__':
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
