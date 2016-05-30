#!/usr/bin/env python

"""Create filelists to use for training and testing."""

import os
import json
from sklearn.cross_validation import train_test_split


path_data = os.path.join(os.environ['DATA_PATH'],
                         'data_road/roadC621/',
                         "image_2/")
files_data = [os.path.join(path_data, f)
              for f in sorted(os.listdir(path_data))
              if f.endswith('.png')]

path_gt = os.path.join(os.environ['DATA_PATH'],
                       'data_road/roadC621/',
                       "gt_image_2/")
files_gt = [os.path.join(path_gt, f)
            for f in sorted(os.listdir(path_gt))
            if f.endswith('.png')]

zipped = list(zip(files_data, files_gt))
train, test = train_test_split(zipped, random_state=0)

train_data = []
for el in train:
    train_data.append({'raw': os.path.abspath(el[0]),
                       'mask': os.path.abspath(el[1])})

test_data = []
for el in test:
    test_data.append({'raw': os.path.abspath(el[0]),
                      'mask': os.path.abspath(el[1])})

with open('trainfiles.json', 'w') as outfile:
    json.dump(train_data, outfile, indent=4, separators=(',', ': '))

with open('testfiles.json', 'w') as outfile:
    json.dump(test_data, outfile, indent=4, separators=(',', ': '))
