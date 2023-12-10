## CasCalib : cascaded calibration for motion capture from sparse unsynchronized cameras

# [Thesis](https://open.library.ubc.ca/soa/cIRcle/collections/ubctheses/24/items/1.0437869) | [Mirror-Aware Neural Humans](https://arxiv.org/abs/2309.04750)

Use the Functions in the following order in order to get multiview camera calibrations from human pose detections. You can see how this works in __main__.py within CasCalib.

# Single View Calibration

You can run the single view calibration by simply running the following line:
```
python run_single_view.py <path_to_frame> <path_to_detections>
```

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

# Temporal Synchronization
Use time_all in time_align.py to get the temporal offset between the sequences. This step uses the outputs from the single view calilbration, that are passed into geometry.camera_to_plane in order to transform the ankle detections from camera coordinates to plane coordinates. 

The outputs are:

- best_shift_array: array of temporal offsets
- best_scale_array: array of temporal scales, or FPS scale
- sync_dict_array: array of frame correspondances

# Rotation Translation search and ICP
use icp from ICP.py. This part requires temporal synchronization output from earlier as input.

# Bundle Adjustment
use bundle_adjustment.match_3d_plotly_input2d_farthest_point to get the matches between views.
use bundle_adjustment.bundle_adjustment to perform the bundle adjustment.