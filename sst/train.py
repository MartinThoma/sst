#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Train a neural network to classify image patches as street / no street.

Take (patch size)x(patch size) (3 color) patches and classify the center pixel.

If loss doesn't change after the first iterations, you have to re-run the
training.
"""

from __future__ import print_function

# import inspect
import imp
import sys
import os
import logging

import scipy
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder


from . import utils


logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                    level=logging.DEBUG,
                    stream=sys.stdout)


def main(hypes_file):
    """
    Train a neural network with patches of patch_size x patch_size.

    (As given via the module network_path).

    Parameters
    ----------
    hypes_file : str
        Path to a JSON
    test_images_json : str
        Path to a JSON which is a list of dicts {'raw': path, 'mask': path}
    image_batch_size : int
    stride : int
    """
    hypes = utils.load_hypes(hypes_file)
    print(hypes)
    network_path = hypes['segmenter']['network_path']
    train_images_json = hypes['data']['train']
    image_batch_size = hypes['training']['batchsize']
    assert image_batch_size >= 1
    assert hypes['training']['stride'] >= 1
    t = load_data_raw_images(hypes,
                             serialization_path=hypes['data']['serialization'],
                             images_json_path=train_images_json)
    features, labels = t
    logging.info("len(features)=%i", len(features))
    logging.info("features.shape=%s", features[0].shape)
    logging.info("labels.shape=%s", labels[0].shape)
    assert len(features) > 0
    mem_size = (sys.getsizeof(42) * len(features) * features[0].size +
                sys.getsizeof(42) * len(labels) * labels[0].size)
    logging.info("Loaded %i data images with their labels (approx %s)",
                 len(features),
                 utils.sizeof_fmt(mem_size))
    class_dict = utils.count_classes(labels)
    logging.info("Classes (abs): %s", class_dict)
    class_dict_rel = {}
    total = sum(count for _, count in class_dict.items())
    for item, count in class_dict.items():
        class_dict_rel[item] = float(count) / total

    fig = plt.figure()
    ax1 = fig.add_subplot(2, 2, 1)
    ax2 = fig.add_subplot(2, 2, 2)
    ax3 = fig.add_subplot(2, 2, 3)
    ax1.imshow(scipy.misc.toimage(features[0]))
    ax2.imshow(scipy.misc.toimage(labels[0]))
    ax3.imshow(scipy.misc.toimage(features[0]))
    ax3.imshow(labels[0], cmap='jet', alpha=0.5)
    plt.show()

    logging.info("Classes (rel): %s", class_dict_rel)
    logging.info("## Network: %s", network_path)
    network = imp.load_source('sst.network', network_path)
    logging.info("Fully network: %s", str(hypes['segmenter']['fully']))
    # nn_params['code'] = inspect.getsource(network)

    # get_features is only called so that the network can be properly generated
    # it is not used for training here
    if hypes["training"]["one_hot_encoding"]:
        label_enc = OneHotEncoder(sparse=False)
        label_enc.fit([[i] for i in range(2)])  # len(hypes['classes'])
    labeled_patches = get_patches(hypes,
                                  features[:1],
                                  labels[:1],
                                  stride=hypes['training']['stride'])
    feats, _ = get_features(hypes, labeled_patches)
    net1 = network.generate_nnet(feats)

    # Generate training data and run training
    for from_img in range(0, len(features), image_batch_size):
        to_img = from_img + image_batch_size
        logging.info("Training on batch %i - %i of %i total",
                     from_img,
                     to_img,
                     len(features))
        labeled_patches = get_patches(hypes,
                                      features[from_img:to_img],
                                      labels[from_img:to_img],
                                      stride=hypes['training']['stride'])
        if hypes["training"]["one_hot_encoding"]:
            labeled_patches[1] = np.reshape(labeled_patches[1], (-1, 1))
            labeled_patches[1] = label_enc.transform(labeled_patches[1])
        if hypes['segmenter']['flatten']:
            new_l = []
            for el in labeled_patches[0]:
                new_l.append(el.flatten())
            new_l = np.array(new_l)
            labeled_patches = (new_l, labeled_patches[1])

        logging.info(("labeled_patches[0].shape: %s , "
                      "labeled_patches[1].shape: %s"),
                     labeled_patches[0].shape,
                     labeled_patches[1].shape)
        net1 = train_nnet(hypes, labeled_patches, net1)

    network.serialize_model(hypes, net1)


def load_data_raw_images(hypes,
                         serialization_path='data.pickle',
                         images_json_path='data.json'):
    """
    Load color images (3 channels) and labels (as images).

    Returns
    -------
    tuple : (featurevector list, label list)
    """
    logging.info("Start loading data...")
    data_source = serialization_path + ".npz"

    if not os.path.exists(data_source):
        # build lists of files which will be read
        train_filelist = utils.get_labeled_filelist(images_json_path)
        files_data, files_gt = [], []
        for train_el in train_filelist:
            file_data, file_gt = train_el['raw'], train_el['mask']
            files_data.append(file_data)
            files_gt.append(file_gt)

        # read files (data first)
        print("Start reading images: ", end='')
        colored_image_features = []
        for img_path in files_data:
            print('.', end='')
            ac = utils.load_color_image_features(img_path)
            # if(ac.shape[0] == 188):  # TODO: Why is this skipped?
            colored_image_features.append(ac)
        print('')
        xs_colored = np.array(colored_image_features, copy=False)

        logging.info("Get dictionary to translate colors to classes...")
        col_to_class = {}
        default_class = 0
        for i, cl in enumerate(hypes['classes']):
            for color in cl['colors']:
                if color == 'default':
                    default_class = i
                else:
                    if isinstance(color, list):
                        color = tuple(color)
                    col_to_class[color] = i

        # read grayscale groundtruth
        logging.info("Read groundtruth...")
        defaulted_colors = set()
        yl = []
        for f in files_gt:
            img = scipy.misc.imread(f, mode='RGB')
            new_img = np.zeros((img.shape[0], img.shape[1]), dtype=int)
            for i, row in enumerate(img):
                for j, pixel in enumerate(row):
                    pixel = tuple(pixel)
                    if pixel in col_to_class:
                        new_img[i][j] = col_to_class[pixel]
                    else:
                        defaulted_colors.add(pixel)
                        new_img[i][j] = default_class
            print("    %s" % f)
            yl.append(new_img)
        # yl = np.array(yl, dtype=int)  # Images can have different dimensions
        logging.info("Those colors were defaulted: %s", defaulted_colors)

        assert len(xs_colored) == len(yl), "len(xs_colored) != len(yl)"
        for i, (X, y) in enumerate(zip(xs_colored, yl), start=1):
            logging.info("Get labels (%i/%i)...", i, len(yl))
            # scipy.misc.imshow(X)
            # scipy.misc.imshow(y)
            assert X.shape[:2] == y.shape, \
                ("X.shape[1:]=%s and y.shape=%s" %
                 (X.shape[:2], y.shape))
            assert min(y.flatten()) == 0.0, \
                ("min(y)=%s" % str(min(y.flatten())))
            assert max(y.flatten()) == 1.0, \
                ("max(y)=%s" % str(max(y.flatten())))
        np.savez(serialization_path, xs_colored, yl)
    else:
        logging.info("!! Loaded pickled data" + "!" * 80)
        logging.info("Data source: %s", data_source)
        logging.info("This implies same test / training split as before.")
        npzfile = np.load(data_source)
        xs_colored = npzfile['arr_0']
        yl = npzfile['arr_1']
    return (xs_colored, yl)


def get_patches(hypes, xs, ys, stride):
    """
    Get a list of tuples (patch, label).

    Where label is int (1=street, 0=no street) and patch is a 2D-array of
    floats.

    Parameters
    ----------
    hypes : dict
        All relevant parameters of the model (e.g. patch_size and fully)
    xs : list
        Each element is an image with 3 channels (RGB), but normalized to
        [-1, 1]
    ys : list
        Each element is either 0 or 1
    stride : int
        The smaller this value, the more patches will be created.

    Returns
    -------
    tuple : (patches, labels)
        Two lists of same length. Patches is
    """
    patch_size = hypes['segmenter']['patch_size']
    fully = hypes['segmenter']['fully']
    assert stride >= 1, "Stride must be at least 1"
    assert (patch_size) >= 1, "Patch size has to be >= 1"
    assert patch_size % 2 == 1, "Patch size should be odd"
    assert xs[0].shape[0] >= patch_size and xs[0].shape[1] >= patch_size, \
        ("Patch is too big for this image: img.shape = %s" % str(xs[0].shape))
    logging.info("Get patches of size: %i", patch_size)
    patches, labels = [], []
    for X, y in zip(xs, ys):
        px_left_patchcenter = (patch_size - 1) / 2
        start_x = px_left_patchcenter
        end_x = X.shape[0] - px_left_patchcenter
        start_y = start_x
        end_y = X.shape[1] - px_left_patchcenter
        for patch_center_x in range(start_x, end_x + 1, stride):
            for patch_center_y in range(start_y, end_y + 1, stride):
                # Get patch from original image
                x_new = X[patch_center_x - px_left_patchcenter:
                          patch_center_x + px_left_patchcenter + 1,
                          patch_center_y - px_left_patchcenter:
                          patch_center_y + px_left_patchcenter + 1,
                          :]
                if x_new.shape != (patch_size, patch_size, 3):
                    # Patch was at the right / bottom border
                    print("Skip patch of shape %s" % str(x_new.shape))
                    continue
                if fully:
                    # Get Labels of the patch and flatt it to 1D
                    # x1 = patch_center_x - px_left_patchcenter
                    # x2 = patch_center_x + px_left_patchcenter + 1
                    # y1 = patch_center_y - px_left_patchcenter
                    # y2 = patch_center_y + px_left_patchcenter + 1
                    l = y[patch_center_x - px_left_patchcenter:
                          patch_center_x + px_left_patchcenter + 1,
                          patch_center_y - px_left_patchcenter:
                          patch_center_y + px_left_patchcenter + 1]
                    labels.append(l.flatten())
                    patches.append(x_new)
                else:
                    labels.append(y[patch_center_x][patch_center_y])
                    patches.append(x_new)
    assert len(patches) == len(labels), "len(patches) != len(labels)"
    logging.info("%i patches were generated.", len(patches))
    logging.info("Data before make_equal: %i", len(labels))
    if 'make_equal' in hypes['training'] and hypes['training']['make_equal']:
        patches, labels = utils.make_equal(patches, labels)
    # logging.info(labels.shape)
    logging.info("Data after make_equal: %i", len(labels))
    if fully:
        return [np.array(patches, dtype=np.float32),
                np.array(labels, dtype=np.float32)]  # fully needs float labels
    else:
        return [np.array(patches, dtype=np.float32),
                np.array(labels, dtype=np.int32)]


def get_features(hypes, labeled_patches):
    """
    Get ready-to-use features from labeled patches.

    Parameters
    ----------
    hypes : dict
    labeled_patches : tuple (patches, labels)

    Returns
    -------
    tuple (feats, y)
        list of feature vectors and list of labels
    """
    feats = labeled_patches[0]
    y = labeled_patches[1]

    if not hypes['segmenter']['fully']:
        counter = {}
        for label in y:
            if not isinstance(label, int) and not isinstance(label, np.int32):
                label = tuple([int(el) for el in list(label)])
            if label in counter:
                counter[label] += 1
            else:
                counter[label] = 1
        logging.info("Label distribution: %s", counter)
    logging.info("Feature vectors: %i", len(y))

    if not hypes['segmenter']['flatten']:
        # original shape: (25, 25, 3)
        # desired shape: (3, 25, 25)
        feats_new = []
        for ac in feats:
            c = []
            c.append(ac[:, :, 0])
            c.append(ac[:, :, 1])
            c.append(ac[:, :, 2])
            feats_new.append(c)
        feats = np.array(feats_new, dtype=np.float32)
    return (feats, y)


def train_nnet(hypes, labeled_patches, net1):
    """
    Train a neural network classifier on the patches.

    Parameters
    ----------
    hypes : dict
    labeled_patches : tuple (patches, labels)
    net1 : model object

    Returns
    -------
    trained classifier
    """
    feats, y = get_features(hypes, labeled_patches)
    print("##### y.shape: %s" % str(y.shape))
    print("##### feats type: %s" % type(feats))
    print("##### feats.shape: %s" % str(feats.shape))
    net1.fit(feats, y)
    return net1


def get_parser():
    """Get parser object."""
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
    main(hypes_file=args.hypes_file)
