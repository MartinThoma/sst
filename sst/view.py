#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Show a network."""

import logging
import sys
import pprint

# Custom modules
from . import utils

logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                    level=logging.INFO,
                    stream=sys.stdout)


def main(hypes_file, verbose=True):
    """Orchestrate."""
    hypes = utils.load_hypes(hypes_file)
    # model = utils.deserialize_model(hypes)
    pp = pprint.PrettyPrinter(indent=4)
    print("# Model: %s" % hypes['segmenter']['network_path'])
    if verbose:
        print("## Hypes")
        print(hypes)
    print("## Other")
    pp.pprint(hypes)


def get_parser():
    """Get parser for view script."""
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
    parser = ArgumentParser(description=__doc__,
                            formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("--hypes",
                        dest="hypes_file",
                        type=str,
                        required=True,
                        help=("path to a JSON file with "
                              "contains 'data' (with 'train' and 'test') as "
                              "well as 'classes' (with 'colors' for each)"))
    return parser


if __name__ == '__main__':
    args = get_parser().parse_args()
    main(args.hypes_file)
