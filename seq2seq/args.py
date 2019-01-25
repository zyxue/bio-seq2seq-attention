import argparse


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '-f', '--data-file', type=str, required=True,
        help='input file with training data'
    )
    parser.add_argument(
        '-c', '--config', type=str, required=True,
        help='config file in json for input (lang0) and output (lang1) languages'
    )

    parser.add_argument(
        '-e', '--embedding-dim', type=int, required=True,
        help='the dimension of embeding vector'
    )
    parser.add_argument(
        '-d', '--hidden-size', type=int, default=50,
        help='hidden layer size'
    )
    parser.add_argument(
        '-b', '--batch-size', type=int, default=2,
        help='batch size'
    )
    parser.add_argument(
        '-l', '--num-hidden-layers', type=int, default=2,
        help='number of hidden layers'
    )
    parser.add_argument(
        '--learning-rate', type=float, default=0.01,
        help='learning rate'
    )
    parser.add_argument(
        '-o', '--outdir', type=str, default='.',
        help='output directory'
    )
    parser.add_argument(
        '-t', '--num-iters', type=int, default=5000,
        help='number of iterations'
    )
    parser.add_argument(
        '--teacher-forcing-ratio', type=float, default=1,
        help='how often teacher enforcing should be used for a batch'
    )
    parser.add_argument(
        '--print-loss-interval', type=int, default=30,
        help='print interval in number of training steps'
    )
    parser.add_argument(
        '--device', default=None,
        help='force use of CPU or GPU. e.g. cpu or gpu:0'
    )

    return parser.parse_args()
