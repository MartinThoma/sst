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
7. Execute `cd training/gt_image_2; for i in *.png; do convert $i -resize 621x188! -sample $i; done; cd -`
8. Execute `cd ..`
9. Execute `sst train --hypes fully_simple_road.json`
10. Execute `sst test --hypes fully_simple_road.json --out out`


## Details

Step 7 will first read the data and create a pickled version of it for
training. If you don't change the features (e.g. patch size), you will not have
to do this step multiple times.

2016-06-12 19:33:43,096 INFOcd training/gt_image_2; for i in *.png; do convert $i -resize 621x188! $i; done; cd -