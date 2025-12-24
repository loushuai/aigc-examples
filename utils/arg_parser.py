import argparse


def get_args_parser():
    parser = argparse.ArgumentParser("AIGC training", add_help=False)
    parser.add_argument("--seed", default=0, type=int)

    return parser
