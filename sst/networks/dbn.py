#!/usr/bin/env python
# -*- coding: utf-8 -*-


"""DBN as suggested in https://jessesw.com/Deep-Learning/."""

import logging

fully = False
patch_size = 19
training_stride = 3
flatten = True


def generate_nnet(feats):
    """Generate a neural network.

    Parameters
    ----------
    feats : list with at least one feature vector

    Returns
    -------
    Neural network object
    """
    # Load it here to prevent crash of --help when it's not present
    from nolearn.dbn import DBN

    input_shape = (None,
                   feats[0].shape[0],
                   feats[0].shape[1],
                   feats[0].shape[2])
    logging.info("input shape: %s", input_shape)
    net1 = DBN([input_shape[1] * input_shape[2] * input_shape[3],
                300,
                2],
               learn_rates=0.3,
               learn_rate_decays=0.9,
               epochs=10,
               verbose=1)
    return net1
