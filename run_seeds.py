"""Train, evaluate and summarize models A, B and C for extra seeds (default 1 and 2).

Run from the `torch` conda environment:
    python run_seeds.py
    python run_seeds.py --seeds 1 2 --epochs 22

Safe to re-run: a finished training run (val_log.csv has `epochs` rows) is skipped, an interrupted one is
resumed with --resume, and an existing results/<run>.csv skips that evaluation. Stops at the first failure.
"""
import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# name prefix -> (main.py/eval model string, extra training flags, eval --type-aware value)
MODELS = {
    'snr_aware': ('SwinJSCC_w/_SA', [], '0'),
    'snr_type_aware': ('SwinJSCC_w/_SA', ['--type-aware'], '1'),
    'no_info': ('SwinJSCC_w/o_SAandRA', [], '0'),
}


def run(cmd):
    print('\n>>> ' + ' '.join(cmd), flush=True)
    if subprocess.run(cmd, cwd=ROOT).returncode != 0:
        sys.exit('FAILED: ' + ' '.join(cmd))


def epochs_done(run_name):
    log = ROOT / 'history' / run_name / 'val_log.csv'
    if not log.exists():
        return 0
    return max(0, len(log.read_text().strip().splitlines()) - 1)


def train(name, seed, epochs):
    model, extra, _ = MODELS[name.rsplit('_s', 1)[0]]
    done = epochs_done(name)
    if done >= epochs:
        print('[skip] training {} already has {} epochs'.format(name, done))
        return
    cmd = [sys.executable, 'main.py', '--training', '--trainset', 'CIFAR10', '--model', model, *extra,
           '--channel-type', 'mixed', '--C', '32', '--multiple-snr', '1,4,7,10,13',
           '--run-name', name, '--epochs', str(epochs),
           '--init-seed', str(seed), '--sampler-seed', str(seed),
           '--noise-seed', str(seed), '--order-seed', str(seed)]
    if (ROOT / 'history' / name / 'resume.pt').exists():
        print('[resume] {} from epoch {}'.format(name, done))
        cmd.append('--resume')
    run(cmd)


def evaluate(name, kodak_dir):
    model, _, type_aware = MODELS[name.rsplit('_s', 1)[0]]
    out = ROOT / 'results' / (name + '.csv')
    if out.exists():
        print('[skip] evaluation {} already exists'.format(out.name))
        return
    cmd = [sys.executable, 'eval_channels.py', '--checkpoint', 'history/{}/models/best.model'.format(name),
           '--run-name', name, '--type-aware', type_aware, '--model', model, '--out', 'results/{}.csv'.format(name)]
    if kodak_dir:
        cmd += ['--kodak-dir', kodak_dir]
    run(cmd)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--seeds', type=int, nargs='+', default=[1, 2])
    p.add_argument('--epochs', type=int, default=22)
    p.add_argument('--kodak-dir', default='Kodak')
    a = p.parse_args()
    if 0 in a.seeds:
        sys.exit('Seed 0 is the existing run set (snr_aware, snr_type_aware, no_info); pass only new seeds.')
    names = [(prefix + '_s' + str(s), s) for s in a.seeds for prefix in MODELS]
    for name, seed in names:
        train(name, seed, a.epochs)
    for name, _ in names:
        evaluate(name, a.kodak_dir)
    run([sys.executable, 'aggregate_seeds.py'])


if __name__ == '__main__':
    main()
