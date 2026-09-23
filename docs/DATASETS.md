# Datasets

No dataset content is redistributed. Set `DATA_ROOT` (see `setup.sh`) to the
directory holding the mirrors; the scripts derive the roots below.

## Sequences used

- **TUM RGB-D** (`freiburg1_xyz`, `freiburg1_desk`, `freiburg2_xyz`,
  `freiburg2_desk`, `freiburg3_long_office_household`)
- **EuRoC MAV** (`MH_01_easy` ... `MH_05_difficult`, `V1_01_easy` ...
  `V1_03_difficult`, `V2_01_easy` ... `V2_03_difficult`)
- **KITTI odometry** sequence 00 (first 800 frames) as the vehicle-domain control.

## Expected layout

### EuRoC - `$EUROC_ROOT/<seq>/`

```
<seq>/mav0/cam0/data/*.png          cam0 images
<seq>/mav0/cam1/data/*.png          cam1 images (stereo anchor)
<seq>/mav0/cam0/sensor.yaml         intrinsics / distortion
<seq>/mav0/state_groundtruth_estimate0/data.csv
```

### TUM RGB-D - `$TUM_ROOT/rgbd_dataset_<seq>/`

```
rgbd_dataset_<seq>/rgb.txt
rgbd_dataset_<seq>/depth.txt
rgbd_dataset_<seq>/groundtruth.txt
rgbd_dataset_<seq>/{rgb,depth}/*.png
```

### KITTI odometry

```
$KITTI_ROOT/sequences/00/calib.txt          color / grey calib (image_2)
$KITTI_ROOT/sequences/00/image_2/*.png
$KITTI_GREY_ROOT/00/poses.txt               ground-truth poses
$KITTI_GREY_ROOT/00/image_0/                grey images
```

## Download helpers

`data/download_tum.sh`, `data/download_euroc.sh`, and `data/download_kitti.sh`
fetch or document each dataset. KITTI requires registration; the helper prints
the official page. See also `data/README.md`.
