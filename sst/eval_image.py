#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Segment pixel-wise street/not street for a single image with a model."""
import logging
import sys
import time
import os

import scipy
import numpy as np
import PIL

# sst modules
from . import utils

logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                    level=logging.DEBUG,
                    stream=sys.stdout)


def main(hypes_file, image_path, output_path, stride,
         hard_classification=True):
    """Evaluate a model."""
    hypes = utils.load_hypes(hypes_file)
    with Timer() as t:
        model_pickle = hypes['segmenter']['serialized_model_path']
        nn = utils.deserialize_model(model_pickle)
    logging.info("Patch size: %i", hypes['segmenter']['patch_size'])
    logging.info("Fully: %s", str(hypes['segmenter']['fully']))
    logging.info("Stride: %i", stride)
    logging.info("=> elasped deserialize model: %s s", t.secs)
    with Timer() as t:
        result = eval_net(hypes=hypes,
                          trained=nn,
                          photo_path=image_path,
                          stride=stride,
                          hard_classification=hard_classification)
    logging.info("=> elasped evaluating model: %s s", t.secs)
    scipy.misc.imsave(output_path, result)
    utils.overlay_images(hypes,
                         image_path, result, output_path,
                         hard_classification=hard_classification)


def eval_net(hypes,
             trained,
             photo_path,
             stride=10,
             hard_classification=True,
             verbose=False):
    """
    Eval a model.

    Parameters
    ----------
    hypes : dict
        Parameters relevant for the model such as patch_size
    trained : theano expression
        A trained neural network
    photo_path : string
        Path to the photo which will get classified
    stride : int
    hard_classification : bool
        If True, the image will only show either street or no street.
        If False, the image will show probabilities.
    verbose : bool

    Returns
    -------
    numpy array
        Segmented image
    """
    patch_size = hypes['segmenter']['patch_size']
    fully = hypes['segmenter']['fully']

    # read images
    feats = utils.load_color_image_features(photo_path)
    orig_dimensions = feats.shape

    patches = []
    px_left_patchcenter = (patch_size - 1) / 2

    height, width = feats.shape[0], feats.shape[1]
    if fully:
        to_pad_width = (patch_size - width) % stride
        to_pad_height = (patch_size - height) % stride

        # Order of to_pad_height / to_pad_width tested with scipy.misc.imsave
        feats = np.pad(feats,
                       [(to_pad_height, 0),
                        (to_pad_width / 2, to_pad_width - (to_pad_width / 2)),
                        (0, 0)],
                       mode='edge')
    else:
        feats = np.pad(feats,
                       [(px_left_patchcenter, px_left_patchcenter),
                        (px_left_patchcenter, px_left_patchcenter),
                        (0, 0)],
                       mode='edge')
    start_x = px_left_patchcenter
    end_x = feats.shape[0] - px_left_patchcenter
    start_y = start_x
    end_y = feats.shape[1] - px_left_patchcenter
    new_height, new_width = 0, 0
    for patch_center_x in range(start_x, end_x, stride):
        new_height += 1
        for patch_center_y in range(start_y, end_y, stride):
            if new_height == 1:
                new_width += 1
            # Get patch from original image
            new_patch = feats[patch_center_x - px_left_patchcenter:
                              patch_center_x + px_left_patchcenter + 1,
                              patch_center_y - px_left_patchcenter:
                              patch_center_y + px_left_patchcenter + 1,
                              :]
            if hypes['segmenter']['flatten']:
                new_patch = new_patch.flatten()
            patches.append(new_patch)

    if verbose:
        logging.info("stride: %s", stride)
        logging.info("patch_size: %i", patch_size)
        logging.info("fully: %s", str(fully))
        logging.info("Generated %i patches for evaluation", len(patches))
    to_classify = np.array(patches, dtype=np.float32)

    if not hypes['segmenter']['flatten']:
        x_new = []
        for ac in to_classify:
            c = []
            c.append(ac[:, :, 0])
            c.append(ac[:, :, 1])
            c.append(ac[:, :, 2])
            x_new.append(c)
        to_classify = np.array(x_new, dtype=np.float32)

    if hard_classification:
        result = trained.predict(to_classify)
    else:
        result = trained.predict_proba(to_classify)
        if not fully:
            result_vec = np.zeros(result.shape[0])
            for i, el in enumerate(result):
                result_vec[i] = el[1]
            result = result_vec

    # Compute combined segmentation of image
    if fully:
        result = result.reshape(result.shape[0], patch_size, patch_size)
        result = result.reshape(new_height, new_width, patch_size, patch_size)

        # Merge patch classifications into a single image (result2)
        result2 = np.zeros((height, width))

        left_px = (patch_size - stride) / 2
        right_px = left_px + stride  # avoid rounding problems with even stride

        offset = {'h': 0, 'w': 0}

        if verbose:
            logging.info("new_height=%i, new_width=%i", new_height, new_width)
            logging.info("result.shape = %s", str(result.shape))
        for j in range(0, new_height):
            for i in range(0, new_width):
                if i == 0:
                    left_margin_px = to_pad_width / 2
                    right_margin_px = right_px
                elif i == new_width - 1:
                    left_margin_px = left_px
                    # TODO (TOTHINK): -1: it's a kind of magic magic...
                    # seems to do the right thing...
                    right_margin_px = patch_size - (to_pad_width -
                                                    (to_pad_width / 2)) - 1
                else:
                    left_margin_px = left_px
                    right_margin_px = right_px
                if j == 0:
                    top_px = to_pad_height
                    bottom_px = right_px
                elif j == new_height - 1:
                    top_px = left_px
                    bottom_px = patch_size
                else:
                    top_px = left_px
                    bottom_px = right_px

                # TOTHINK: no +1?
                to_write = result[j, i,
                                  top_px:(bottom_px),
                                  left_margin_px:(right_margin_px)]

                if i == 0 and j == 0:
                    offset['h'] = to_write.shape[0]
                    offset['w'] = to_write.shape[1]

                start_h = (offset['h'] + (j - 1) * stride) * (j != 0)
                start_w = (offset['w'] + (i - 1) * stride) * (i != 0)
                result2[start_h:start_h + to_write.shape[0],
                        start_w:start_w + to_write.shape[1]] = to_write

        if hard_classification:
            result2 = np.round((result2 - np.amin(result2)) /
                               (np.amax(result2) - np.amin(result2)))
        return result2
    else:
        if hypes["training"]["one_hot_encoding"]:
            result = np.argmax(result, axis=1)
        result = result.reshape((new_height, new_width))

        # Scale image to correct size
        result = scale_output(result, orig_dimensions)
        return result


def eval_pickle(hypes,
                trained,
                images_json_path,
                out_path,
                stride=1):
    """
    Eval a model.

    Parameters
    ----------
    hypes : dict
        Parameters relevant for the model (e.g. patch size)
    trained : theano expression
        A trained neural network
    images_json_path : str
        Path to a JSON file
    out_path : str
    stride : int
    """
    train_filelist = utils.get_labeled_filelist(images_json_path)
    list_tuples = [(el['raw'], el['mask']) for el in train_filelist]

    total_results = {}
    elements = [0, 1]
    for i in elements:
        total_results[i] = {}
        for j in elements:
            total_results[i][j] = 0
    for i, (data_image_path, gt_image_path) in enumerate(list_tuples):
        logging.info("Processing image: %s of %s (%s)",
                     i + 1,
                     len(list_tuples),
                     total_results)
        segmentation = eval_net(hypes,
                                trained,
                                photo_path=data_image_path,
                                stride=stride)
        seg_path = os.path.join(out_path, "seg-%i.png" % i)
        overlay_path = os.path.join(out_path, "overlay-%i.png" % i)
        scipy.misc.imsave(seg_path, segmentation * 255)
        utils.overlay_images(hypes,
                             data_image_path, segmentation, overlay_path,
                             hard_classification=True)
        conf = get_error_matrix(hypes, segmentation, gt_image_path)
        total_results = merge_cms(total_results, conf)

    logging.info("Eval results: %s", total_results)
    logging.info("Accurity: %s ", get_accuracy(total_results))
    logging.info("%i images evaluated.", len(list_tuples))


def get_error_matrix(hypes, result, gt_image_path):
    """
    Get true positive, false positive, true negative, false negative.

    Parameters
    ----------
    result : numpy array
    gt_image_path : str
        Path to an image file with the labeled data.

    Returns
    -------
    dict
        with keys tp, tn, fp, fn
    """
    img = scipy.misc.imread(gt_image_path, mode='RGB')

    conf_dict = {}  # map colors to classes
    default = 0
    for i, cl in enumerate(hypes["classes"]):
        for color in cl["colors"]:
            conf_dict[color] = i
            if color == "default":
                default = i

    # Create gt image which is a matrix of classes
    gt = np.zeros(result.shape)
    for i, row in enumerate(img):
        for j, pixel in enumerate(row):
            pixel = tuple(pixel)
            if pixel in conf_dict:
                gt[i][j] = conf_dict[pixel]
            else:
                logging.debug("Didn't find %s", str(pixel))
                gt[i][j] = default
    return get_confusion_matrix(gt, result)


def scale_output(classify_image, new_shape):
    """
    Scale `classify_image` to `new_shape`.

    Parameters
    ----------
    classify_image : numpy array
    new_shape : tuple

    Returns
    -------
    numpy array
    """
    im = scipy.misc.toimage(classify_image,
                            low=np.amin(classify_image),
                            high=np.amax(classify_image))
    im = im.resize((new_shape[1], new_shape[0]),
                   resample=PIL.Image.NEAREST)
    return scipy.misc.fromimage(im)


def get_parser():
    """Get parser object."""
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
    parser = ArgumentParser(description=__doc__,
                            formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('-i', '--input',
                        dest='image_path',
                        type=lambda x: utils.is_valid_file(parser, x),
                        help='load IMAGE for pixel-wise street segmenation',
                        default=utils.get_default_data_image_path(),
                        metavar='IMAGE')
    parser.add_argument('-o', '--output',
                        dest='output_path',
                        help='store semantic segmentation here',
                        default="out.png",
                        metavar='IMAGE')
    parser.add_argument("--stride",
                        dest="stride",
                        default=10,
                        type=int,
                        help=("the higher this value, the longer the "
                              "evaluation takes, but the more accurate it is"))
    parser.add_argument("--hypes",
                        dest="hypes_file",
                        type=str,
                        required=True,
                        help=("path to a JSON file with "
                              "contains 'data' (with 'train' and 'test') as "
                              "well as 'classes' (with 'colors' for each)"))
    return parser


class Timer(object):
    """
    Timer.

    Attributes
    ----------
    verbose : boolean
    """

    def __init__(self, verbose=False):
        self.verbose = verbose

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args):
        self.end = time.time()
        self.secs = self.end - self.start
        self.msecs = self.secs * 1000  # millisecs
        if self.verbose:
            print('elapsed time: %f ms' % self.msecs)


def get_accuracy(n):
    r"""
    Get the accuracy from a confusion matrix n.

    The mean accuracy is calculated as
    .. math::
        t_i &= \sum_{j=1}^k n_{ij}\\
        acc(n) &= \frac{\sum_{i=1}^k n_{ii}}{\sum_{i=1}^k n_{ii}}
    Parameters
    ----------
    n : dict
        Confusion matrix which has integer keys 0, ..., nb_classes - 1;
        an entry n[i][j] is the count how often class i was classified as
        class j.
    Returns
    -------
    float
        accuracy (in [0, 1])
    References
    ----------
    .. [1] Martin Thoma (2016): A Survey of Semantic Segmentation,
       http://arxiv.org/abs/1602.06541
    Examples
    --------
    >>> n = {0: {0: 10, 1: 2}, 1: {0: 5, 1: 83}}
    >>> get_accuracy(n)
    0.93
    """
    return (float(n[0][0] + n[1][1]) /
            (n[0][0] + n[1][1] + n[0][1] + n[1][0]))


def merge_cms(cm1, cm2):
    """
    Merge two confusion matrices.

    Parameters
    ----------
    cm1 : dict
        Confusion matrix which has integer keys 0, ..., nb_classes - 1;
        an entry cm1[i][j] is the count how often class i was classified as
        class j.
    cm2 : dict
        Another confusion matrix.
    Returns
    -------
    dict
        merged confusion matrix
    Examples
    --------
    >>> cm1 = {0: {0: 1, 1: 2}, 1: {0: 3, 1: 4}}
    >>> cm2 = {0: {0: 5, 1: 6}, 1: {0: 7, 1: 8}}
    >>> merge_cms(cm1, cm2)
    {0: {0: 6, 1: 8}, 1: {0: 10, 1: 12}}
    """
    assert 0 in cm1
    assert len(cm1[0]) == len(cm2[0])

    cm = {}
    k = len(cm1[0])
    for i in range(k):
        cm[i] = {}
        for j in range(k):
            cm[i][j] = cm1[i][j] + cm2[i][j]

    return cm


def get_confusion_matrix(correct_seg, segmentation, elements=None):
    """
    Get the confuscation matrix of a segmentation image and its ground truth.

    The confuscation matrix is a detailed count of which classes i were
    classifed as classes j, where i and j take all (elements) names.
    Parameters
    ----------
    correct_seg : numpy array
        Representing the ground truth.
    segmentation : numpy array
        Predicted segmentation
    elements : iterable
        A list / set or another iterable which contains the possible
        segmentation classes (commonly 0 and 1).
    Returns
    -------
    dict
        A confusion matrix m[correct][classified] = number of pixels in this
        category.
    """
    assert len(correct_seg.shape) == 2, \
        "len(correct_seg.shape) = %i" % len(correct_seg.shape)
    assert correct_seg.shape == segmentation.shape, \
        "correct_seg = %s != %s = segmentation" % (correct_seg.shape,
                                                   segmentation.shape)
    height, width = correct_seg.shape

    # Get classes
    if elements is None:
        elements = set(np.unique(correct_seg))
        elements = elements.union(set(np.unique(segmentation)))
        logging.debug("elements parameter not given to get_confusion_matrix")
        logging.debug("  assume '%s'", elements)

    # Initialize confusion matrix
    confusion_matrix = {}
    for i in elements:
        confusion_matrix[i] = {}
        for j in elements:
            confusion_matrix[i][j] = 0

    for x in range(width):
        for y in range(height):
            confusion_matrix[correct_seg[y][x]][segmentation[y][x]] += 1
    return confusion_matrix


if __name__ == '__main__':
    args = get_parser().parse_args()
    main(hypes_file=args.hypes_file,
         image_path=args.image_path,
         output_path=args.output_path,
         stride=args.stride)
