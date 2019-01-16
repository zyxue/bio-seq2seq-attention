import argparse


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '-f', '--input-file', type=str, required=True,
        help='input file with training data'
    )
    parser.add_argument(
        '-f', '--lang-config', type=str, required=True,
        help='config file in json for input (lang0) and output (lang1) languages'
    )

    parser.add_argument(
        '-e', '--embedding-dim', type=int, default=10,
        help='the dimension of embeding vector'
    )
    parser.add_argument(
        '-d', '--hidden-size', type=int, default=50,
        help='hidden layer size'
    )
    parser.add_argument(
        '-b', '--batch-size', type=int, default=16,
        help='batch size'
    )
    parser.add_argument(
        '-l', '--num-layers', type=int, default=1,
        help='number of hidden layers'
    )
    parser.add_argument(
        '-r', '--bidirectional', action='store_true',
        help='use bidirectional RNN if set'
    )
    parser.add_argument(
        '--learning-rate', type=float, default=0.01,
        help='learning rate'
    )
    parser.add_argument(
        '-o', '--outdir', type=str,
        help='output directory'
    )
    parser.add_argument(
        '-t', '--num-iters', type=int, default=5000,
        help='number of iterations'
    )
    parser.add_argument(
        '-p', '--print-every', type=int, default=100,
        help='print interval'
    )

    parser.add_argument(
        '--plot-every', type=int, default=0,
        help='if 0, no plots willl be generated'
    )

    parser.add_argument(
        '--device', default=None,
        help='force use of CPU or GPU. e.g. cpu or gpu:0'
    )

    return parser.parse_args()
