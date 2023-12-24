# CasCalib : cascaded calibration for motion capture from sparse unsynchronized cameras
## [Technical report](https://open.library.ubc.ca/soa/cIRcle/collections/ubctheses/24/items/1.0437869)
>**CasCalib**\
>[James Tang](https://www.linkedin.com/in/james-tang-279332196/?originalSubdomain=ca), [Daniel Ajisafe](https://danielajisafe.github.io/), [Shashwat Suri](https://www.linkedin.com/in/shashwat-suri-88807a13b/), , [Bastian Wandt](https://bastianwandt.de/), and [Helge Rhodin](http://helge.rhodin.de/)

Use the Functions in the following order in order to get multiview camera calibrations from human pose detections. You can see how this works in __main__.py within CasCalib.

## Installation

# Step 1
Clone repository.
```
git clone https://github.com/tangytoby/CasCalib.git
cd CasCalib
```

# Step 2
Create a conda enviroment.
```
conda create -n CasCalib python=3.8
conda activate CasCalib
```

# Step 3
Install Requirements.
```
pip install -r requirements.txt
```

# Step 4
run demo for single view calibration.
```
python run_single_view.py example_data/frames/terrace1-c0_avi/00000000.jpg example_data/detections/result_terrace1-c0_.json 0
```
The results will be in the outputs folder in the folder called "single_view_<date>_<time>"

## Single View Calibration

You can run the single view calibration by simply running the following line:
```
python run_single_view.py <path_to_frame> <path_to_detections> <int>
```
The last line determines the detector that you are using. 0 for coco mmpose and 1 for alphapose.

Use run_calibration_ransac from run_calibration_ransac.py to get the camera calibration for single views. 

The outputs are:

- ankles: The 3D ankles
- cam_matrix: The intrinsic camera matrix
- normal: The normal vector of the ground plane
- ankleWorld: The plane position
- focal: The focal length
- focal_batch: intermediate results for the focal length
- ransac_focal: intermediate results for the focal length from ransac
- datastore_filtered: The filtered detections used for the calibration

## Temporal Synchronization
Use time_all in time_align.py to get the temporal offset between the sequences. This step uses the outputs from the single view calilbration, that are passed into geometry.camera_to_plane in order to transform the ankle detections from camera coordinates to plane coordinates. 

The outputs are:

- best_shift_array: array of temporal offsets
- best_scale_array: array of temporal scales, or FPS scale
- sync_dict_array: array of frame correspondances

## Rotation Translation search and ICP
use icp from ICP.py. This part requires temporal synchronization output from earlier as input.

## Bundle Adjustment
use bundle_adjustment.match_3d_plotly_input2d_farthest_point to get the matches between views.
use bundle_adjustment.bundle_adjustment to perform the bundle adjustment.
