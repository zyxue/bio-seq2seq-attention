import argparse


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('-i', '--input', type=str)
    parser.add_argument('-e', '--embedding-size', type=int)
    parser.add_argument('-d', '--hidden-size', type=int)
    parser.add_argument('-b', '--batch-size', type=int, default=1)
    parser.add_argument('-l', '--num-layers', type=int, default=1)
    parser.add_argument('-r', '--bidirectional', action='store_true')
    parser.add_argument('--learning-rate', type=float, default=0.01)
    parser.add_argument('-o', '--outdir', type=str)

    parser.add_argument('-t', '--num-iters', type=int, default=5000)
    parser.add_argument('-p', '--print-every', type=int, default=100)

    parser.add_argument(
        '--plot-every', type=int, default=0,
        help='if 0, no plotting willl be done'
    )

    args = parser.parse_args()
    return args
