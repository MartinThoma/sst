#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Evaluate the error (tp, tn, fp, fp) on all images."""

import logging
import sys
import os

logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                    level=logging.INFO,
                    stream=sys.stdout)

# Custom modules
from . import utils
from . import eval_image
from . import view


def main(hypes_file):
    """Evaluate a trained model."""
    hypes = utils.load_hypes(hypes_file)
    test_images_json = hypes['data']['test']
    stride = hypes['segmenter']['stride']
    model_path_trained = hypes['segmenter']['serialized_model_path']
    if not os.path.isfile(model_path_trained):
        logging.warning("No model found at '%s'.", model_path_trained)
    trained, paramters = utils.deserialize_model(model_path_trained)
    view.main(model_path_trained, verbose=False)
    eval_image.eval_pickle(trained, paramters, test_images_json, stride=stride)


def get_parser():
    """Get the parser object."""
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
