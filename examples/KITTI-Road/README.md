## Steps

1. Clone this repostiory and install sst
2. Download the [Download base kit with: left color images, calibration and training labels (0.5 GB)](http://www.cvlibs.net/datasets/kitti/eval_road.php)
3. Copy the downloaded stuff in the directory `data_road`
4. cd `examples/KITTI-Road/data_road`
5. Execute `create_filelist.py`. This should create a file `trainfiles.json`
   and a file `testfiles.json`. Both files should contain a list of entries
   `{'raw': [some path], 'mask': [some path]} which maps raw images to
   segmentation masks.
6. Execute `cd training/image_2; for i in *.png; do convert $i -resize 621x188! $i; done; cd -`
7. Execute `cd training/gt_image_2; for i in *.png; do convert $i -sample 621x188! $i; done; cd -`
8. Execute `cd ..`
9. Execute `sst train --hypes fully_simple_road.json`
10. Execute `sst test --hypes fully_simple_road.json --out out`


## Details

Step 7 will first read the data and create a pickled version of it for
training. If you don't change the features (e.g. patch size), you will not have
to do this step multiple times.

The training run should give a console output similar to the following:

```bash
time sst train --hypes fully_simple_road.json
2016-06-13 11:29:53,138 INFO template_path: /home/moose/GitHub/sst/sst/templates/
{u'training': {u'stride': 20, u'batchsize': 100}, u'classes': [{u'colors': [[255, 0, 255]], u'name': u'road'}, {u'colors': [u'default', [255, 0, 0]], u'name': u'background'}], u'data': {u'test': u'/home/moose/GitHub/sst/examples/KITTI-Road/data_road/testfiles.json', u'train': u'/home/moose/GitHub/sst/examples/KITTI-Road/data_road/trainfiles.json'}, u'segmenter': {u'network_path': u'/home/moose/GitHub/sst/sst/networks/fully_simple.py', u'stride': 10, u'serialized_model_path': u'/home/moose/GitHub/sst/examples/KITTI-Road/fully_simple_road.pickle'}}
2016-06-13 11:29:53,145 INFO Start loading data...
2016-06-13 11:29:53,145 INFO !! Loaded pickled data!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
2016-06-13 11:29:53,145 INFO Data source: data.pickle.npz
2016-06-13 11:29:53,145 INFO This implies same test / training split as before.
2016-06-13 11:29:54,017 INFO len(features)=231
2016-06-13 11:29:54,017 INFO features.shape=(188, 621, 3)
2016-06-13 11:29:54,017 INFO labels.shape=(188, 621)
2016-06-13 11:29:54,017 INFO Loaded 231 data images with their labels (approx 2.4GiB)
2016-06-13 11:29:54,017 INFO ## Network: /home/moose/GitHub/sst/sst/networks/fully_simple.py
2016-06-13 11:29:54,017 INFO Fully network: True
2016-06-13 11:29:54,018 INFO Get patches of size: 51
2016-06-13 11:29:54,019 INFO 203 patches were generated.
2016-06-13 11:29:54,022 INFO Feature vectors: 203
Using gpu device 0: GeForce GTX TITAN Black (CNMeM is disabled, cuDNN 4007)
/usr/local/lib/python2.7/dist-packages/theano/tensor/signal/downsample.py:6: UserWarning: downsample module has been moved to the theano.tensor.signal.pool module.
  "downsample module has been moved to the theano.tensor.signal.pool module.")
2016-06-13 11:29:57,173 INFO input shape: (None, 3, 51, 51)
2016-06-13 11:29:57,173 INFO Training on batch 0 - 100 of 231 total
2016-06-13 11:29:57,173 INFO Get patches of size: 51
2016-06-13 11:29:57,353 INFO 20300 patches were generated.
2016-06-13 11:29:57,536 INFO labeled_patches[0].shape: (20300, 51, 51, 3) , labeled_patches[1].shape: (20300, 2601)
2016-06-13 11:29:57,536 INFO Feature vectors: 20300
/usr/local/lib/python2.7/dist-packages/lasagne/layers/conv.py:460: UserWarning: The `image_shape` keyword argument to `tensor.nnet.conv2d` is deprecated, it has been renamed to `input_shape`.
  border_mode='full')
# Neural Network with 26291 learnable parameters

## Layer information

  #  name     size
---  -------  --------
  0  input    3x51x51
  1  hidden   10x51x51
  2  hidden2  1x51x51
  3  flatten  2601

  epoch    train loss    valid loss    train/val  dur
-------  ------------  ------------  -----------  -------
      1       1.36162       0.87364      1.55857  638.52s
      2       0.87560       0.87364      1.00225  637.22s
      3       0.87560       0.87364      1.00225  638.03s
      4       0.87560       0.87364      1.00225  637.40s
      5       0.87560       0.87364      1.00225  637.69s
      6       0.87560       0.87364      1.00225  636.85s
      7       0.87560       0.87364      1.00225  636.89s
      8       0.87560       0.87364      1.00225  638.81s
      9       0.87560       0.87364      1.00225  637.90s
     10       0.87560       0.87364      1.00225  639.12s
2016-06-13 13:16:18,072 INFO Training on batch 100 - 200 of 231 total
2016-06-13 13:16:18,072 INFO Get patches of size: 51
2016-06-13 13:16:18,228 INFO 20300 patches were generated.
2016-06-13 13:16:18,416 INFO labeled_patches[0].shape: (20300, 51, 51, 3) , labeled_patches[1].shape: (20300, 2601)
2016-06-13 13:16:18,416 INFO Feature vectors: 20300
     11       0.86861       0.87218      0.99591  638.75s
TODO: continue
```