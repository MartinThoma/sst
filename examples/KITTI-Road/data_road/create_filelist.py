#!/usr/bin/env python

"""Create filelists to use for training and testing."""

import os
import json
from sklearn.cross_validation import train_test_split

# Adjust to your needs:
base = '.'

# train_data
train_data = []
test_data = []

directory = os.path.join(base, 'training/image_2')
files_data_raw = [os.path.abspath(os.path.join(directory, f))
                  for f in sorted(os.listdir(directory))
                  if f.endswith('.png')]
directory = os.path.join(base, 'training/gt_image_2')
files_data_gt = [os.path.abspath(os.path.join(directory, f))
                 for f in sorted(os.listdir(directory))
                 if f.endswith('.png') and 'lane' not in f]
for raw, gt in zip(sorted(files_data_raw), sorted(files_data_gt)):
    train_data.append({'raw': raw, 'mask': gt})

# testing
train_data, test_data = train_test_split(train_data,
                                         train_size=0.8,
                                         random_state=0)


# write data
with open('trainfiles.json', 'w') as outfile:
    json.dump(train_data, outfile, indent=4, separators=(',', ': '))

with open('testfiles.json', 'w') as outfile:
    json.dump(test_data, outfile, indent=4, separators=(',', ': '))
