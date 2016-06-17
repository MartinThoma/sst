#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Create a video."""

import logging
import sys
import os

# Custom modules
from . import utils
from . import eval_image

logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                    level=logging.INFO,
                    stream=sys.stdout)


def make_video(hypes, files_data, video_dir, stride):
    """
    Create a video.

    Parameters
    ----------
    hypes : dict
    files_data : list
        List of paths to images
    video_dir : str
        Write overlayed images in this folder.
    stride : int
    """
    if not os.path.exists(video_dir):
        os.mkdir(video_dir)
    for i, image_path in enumerate(files_data):
        logging.info("Processing image %s / %s", (i + 1), len(files_data))
        nn = utils.deserialize_model(hypes)
        result = eval_image.eval_net(hypes,
                                     nn,
                                     image_path,
                                     stride=stride)
        # output_path = os.path.join(video_dir, image_name)
        # scipy.misc.imsave(output_path, result)
        utils.overlay_images(hypes,
                             image_path,
                             result,
                             os.path.join(video_dir, "%04d.png" % i))
    cmd = "avconv -f image2 -i %s/%%04d.png avconv_out.avi" % video_dir
    logging.info(cmd)
    os.system(cmd)


def main(hypes_file, images_file_path, video_dir, stride):
    """Orchestrate."""
    hypes = utils.load_hypes(hypes_file)
    with open(images_file_path) as f:
        images_list = f.read()
    images_list = images_list.split("\n")
    make_video(hypes, images_list, video_dir, stride)


def get_parser():
    """Get parser for video script."""
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
    parser.add_argument("--video-dir",
                        dest="video_dir",
                        default=('video'),
                        help="folder with images")
    parser.add_argument("--images",
                        dest="images",
                        required=True,
                        help=("text file with list of images "
                              "(one per line; "
                              "line path is taken until first space)"))
    parser.add_argument("--stride",
                        dest="stride",
                        default=10,
                        type=int,
                        help=("the higher this value, the longer the "
                              "evaluation takes, but the more accurate it is"))
    return parser


if __name__ == '__main__':
    args = get_parser().parse_args()
    main(args.hypes_file,
         args.images,
         args.video_dir,
         args.stride)
