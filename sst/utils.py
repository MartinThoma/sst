#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Utility functions."""

import numpy as np
import pickle
from PIL import Image
import os
import logging
import scipy
import scipy.misc
import sys
import pkg_resources
import json
import imp


def make_equal(features, ys):
    """Make sure the classes are equally distributed.

    Parameters
    ----------
    features : ndarray
    ys : ndarray

    Returns
    -------
    tuple of two ndarray
    """
    min_count = min(len(ys) - sum(ys), sum(ys))
    assert min_count > 0, (("min_count=%i (should be much bigger "
                            "than 0") % min_count)
    is_street = 0
    no_street = 0
    X_new, y_new = [], []
    for X, y in zip(features, ys):
        if y == 0 and no_street < min_count:
            no_street += 1
            X_new.append(X)
            y_new.append(y)
        elif y == 1 and is_street < min_count:
            is_street += 1
            X_new.append(X)
            y_new.append(y)
    return (np.array(X_new), np.array(y_new))


def serialize_model(model, filename):
    """Save a model.

    Parameters
    ----------
    model : theano expression
    filename : string
        Path where the file should be stored - should end with .pickle
    parameters : dict
        values which are relevant for the model, e.g. patch size
    """
    logging.info("Start serializing...")
    with open(filename, 'wb') as handle:
        pickle.dump(model, handle, protocol=pickle.HIGHEST_PROTOCOL)


def deserialize_model_pickle(filename):
    """Deserialize a model from a file to a Theano expression.

    Parameters
    ----------
    filename : str
        Path to a .pickle model file

    Returns
    -------
    object
        A model
    """
    from sst import shape
    sys.modules['shape'] = shape
    with open(filename, 'rb') as handle:
        to_pickle = pickle.load(handle)
    return to_pickle


def deserialize_model(hypes):
    """Deserialize a model from a file.

    Parameters
    ----------
    hypes : dict

    Returns
    -------
    object
        A model
    """
    model_path_trained = hypes['segmenter']['serialized_model_path']
    if not os.path.isfile(model_path_trained):
        logging.warning("No model found at '%s'.", model_path_trained)
    network_path = hypes['segmenter']['network_path']
    network = imp.load_source('sst.network', network_path)
    trained = network.load_model(hypes)
    return trained


def load_image_patch(filename):
    """Load image as normalized flat numpy array."""
    im = Image.open(filename)  # .convert('L')
    width, height = im.size
    pixels = list(im.getdata())
    features = [pixels[i * width:(i + 1) * width] for i in range(height)]
    features = np.asarray(im, dtype=np.float32).flatten()
    features /= 255.0
    return features


def load_image(filename):
    """Load image as normalized numpy array."""
    im = Image.open(filename)  # .convert('L')
    width, height = im.size
    pixels = list(im.getdata())
    features = [pixels[i * width:(i + 1) * width] for i in range(height)]
    features = np.asarray(im, dtype=np.float32)
    features /= 255.0
    return features


def overlay_images(hypes,
                   original_image,
                   overlay_arr,
                   output_path,
                   hard_classification=True,):
    """
    Overlay original_image with segmentation_image.

    Store the result with the same name as segmentation_image, but with
    `-overlay`.

    Parameters
    ----------
    hypes : dict
    original_image : string
        Path to an image file
    overlay_arr : numpy array
    segmentation_image : string
        Path to the an image file of the same size as original_image
    hard_classification : bool
        If True, the image will only show either street or no street.
        If False, the image will show probabilities.

    Returns
    -------
    str : Path of overlay image
    """
    background = Image.open(original_image)
    overlay = scipy.misc.toimage(overlay_arr)
    overlay = overlay.convert('RGBA')

    # Replace colors of segmentation to make it easier to see
    width, height = overlay.size
    pix = overlay.load()
    colors = set()
    for x in range(0, width):
        for y in range(0, height):
            if not hard_classification:
                overlay.putpixel((x, y), (0, pix[x, y][0], 0))
            else:
                cl = int(overlay_arr[y][x])
                overlay.putpixel((x, y),
                                 tuple(hypes["classes"][cl]["output"]))
            colors.add(overlay_arr[y][x])
    logging.info("%i colors: %s", len(colors), str(colors))
    background = background.convert('RGB')
    background.paste(overlay, (0, 0), mask=overlay)
    background.save(output_path, 'PNG')
    return output_path


def get_overlay_name(segmentation_name):
    """Get the appropriate name for an overlay."""
    splitted = segmentation_name.split('.')
    splitted[-2] = splitted[-2] + '-overlay'
    output_path = '.'.join(splitted)
    return output_path


def is_valid_file(parser, arg):
    """
    Check if arg is a valid file that already exists on the file system.

    Parameters
    ----------
    parser : object
    arg : str
    """
    arg = os.path.abspath(arg)
    if not os.path.exists(arg):
        parser.error("The file %s does not exist!" % arg)
    else:
        return arg


def load_color_image_features(img_path):
    """Load a colored (8-bit, RGB) image as a normalized feature vector.

    Parameters
    ----------
    img_path : string

    Returns
    -------
    numpy array
    """
    ac = scipy.misc.imread(img_path, mode='RGB')
    ac = ac / (255.0 / 2) - 1.0
    return np.array(ac)


def sizeof_fmt(num, suffix='B'):
    """Format `num` bytes to human readable format."""
    for unit in ['', 'Ki', 'Mi', 'Gi', 'Ti', 'Pi', 'Ei', 'Zi']:
        if abs(num) < 1024.0:
            return "%3.1f%s%s" % (num, unit, suffix)
        num /= 1024.0
    return "%.1f%s%s" % (num, 'Yi', suffix)


def get_model_path():
    """
    Get the path of a model pickle file.

    Returns
    -------
    str
        Path to a pickle file which contains a trained model.
    """
    misc_path = pkg_resources.resource_filename('sst', 'misc/')
    return os.path.abspath(os.path.join(misc_path, 'model.pickle'))


def get_default_data_image_path():
    """Get the default image path."""
    misc_path = pkg_resources.resource_filename('sst', 'misc/')
    return os.path.abspath(os.path.join(misc_path, "um_000000.png"))


def get_labeled_filelist(json_file_list_path):
    """
    Get a list of raw image files and mask files.

    Parameters
    ----------
    json_file_list_path : str
        Path to a json file which contains a list of dictionaries with the
        keys 'raw' and 'mask'

    Returns
    -------
    list
        List of dicts {'raw': Absolute path to file,
                       'mask': Absolute path to file}
    """
    with open(json_file_list_path) as data_file:
        data = json.load(data_file)
    for i, training_data_item in enumerate(data):
        if 'raw' not in training_data_item:
            raise ValueError(("could not find 'raw' in item %i of JSON file "
                              "'%s'") % (i, json_file_list_path))
        elif 'mask' not in training_data_item:
            raise ValueError(("could not find 'raw' in item %i of JSON file "
                              "'%s'") % (i, json_file_list_path))
        elif not os.path.isfile(training_data_item['raw']):
            raise ValueError(("could not find file '%s' specified in item %i "
                              "as raw") % (json_file_list_path['raw'], i))
        elif not os.path.isfile(training_data_item['mask']):
            raise ValueError(("could not find file '%s' specified in item %i "
                              "as mask") % (json_file_list_path['mask'], i))
        with Image.open(training_data_item['raw']) as im:
            im = Image.open(training_data_item['raw'])
            raw_width, raw_height = im.size
        with Image.open(training_data_item['mask']) as im:
            im = Image.open(training_data_item['mask'])
            mask_width, mask_height = im.size
        if raw_width != mask_width:
            raise ValueError(("raw_width = %i != %i = mask_width of "
                              "raw(%s) and mask(%s)") %
                             (raw_width, mask_width,
                              json_file_list_path['raw'],
                              json_file_list_path['mask']))
        if raw_height != mask_height:
            raise ValueError(("raw_height = %i != %i = mask_height of "
                              "raw(%s) and mask(%s)") %
                             (raw_height, mask_height,
                              json_file_list_path['raw'],
                              json_file_list_path['mask']))

    return data


def load_hypes(hypes_file):
    """Load the dictionary of the JSON hypes_file."""
    if not os.path.isfile(hypes_file):
        logging.warning("No hypes file found at '%s'.",
                        hypes_file)
    else:
        hypes = None
        with open(hypes_file) as data_file:
            hypes = json.load(data_file)
        base = os.path.dirname(hypes_file)
        if "one_hot_encoding" not in hypes["training"]:
            hypes["training"]["one_hot_encoding"] = False
        hypes['data']['train'] = os.path.join(base, hypes['data']['train'])
        hypes['data']['train'] = os.path.abspath(hypes['data']['train'])
        hypes['data']['test'] = os.path.join(base, hypes['data']['test'])
        hypes['data']['test'] = os.path.abspath(hypes['data']['test'])
        tmp = os.path.join(base, hypes['segmenter']['network_path'])
        hypes['segmenter']['network_path'] = os.path.abspath(tmp)
        tmp = os.path.join(base,
                           hypes['segmenter']['serialized_model_path'])
        hypes['segmenter']['serialized_model_path'] = os.path.abspath(tmp)
        for i in range(len(hypes['classes'])):
            for j in range(len(hypes['classes'][i]['colors'])):
                color = hypes['classes'][i]['colors'][j]
                if color != "default":
                    hypes['classes'][i]['colors'][j] = tuple(color)
        assert hypes['segmenter']['patch_size'] > 0
        return hypes


def count_classes(labels):
    """Count how often the classes appear."""
    class_dict = {}
    for image in labels:
        for row in image:
            for label in row:
                if label not in class_dict:
                    class_dict[label] = 1
                else:
                    class_dict[label] += 1
    return class_dict
