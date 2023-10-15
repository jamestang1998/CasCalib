import os
import sys
# Might have to add to path
# sys.path.append('/local/tangytob/Summer2023/multiview_synchronization/') 
from CalibSingleFromP2D import util,data,run_calibration_ransac, eval_human_pose

import json
from datetime import datetime
import csv
import matplotlib.image as mpimg

today = datetime.now()

metrics = eval_human_pose.Metrics()


#The name of is the current date
name = str(today.strftime('%Y%m%d_%H%M%S')) + '_EPFL_campus4_res50'

#Gets the hyperparamter from hyperparameter.json
(threshold_euc, threshold_cos, angle_filter_video, 
 confidence, termination_cond, num_points, h, iter, focal_lr, point_lr) = util.hyperparameter(
     'CalibSingleFromP2D/hyperparameter.json')
hyperparam_dict = {"threshold_euc": threshold_euc, "threshold_cos": threshold_cos, 
                   "angle_filter_video": angle_filter_video, "confidence": confidence, 
                   "termination_cond": termination_cond, "num_points": num_points, "h": h, 
                   "optimizer_iteration" :iter, "focal_lr" :focal_lr, "point_lr": point_lr}

# *************************************************************************************************************

# EXTRINSIC MATRIX DETAILS
# https://ksimek.github.io/2012/08/22/extrinsic/
# for subject in ['S1', 'S5','S6','S7','S8','S9']:
# for subject in ['S5','S6','S7','S8','S9']:
# for subject in ['S1','S5','S6','S7','S8']:

# REF    SYNC   REF          SYNC         REF        SYNC
# scale, scale, shift start, shift start, shift end, shift end
# experiments_time = [experiments_time[0]]

# experiments_time = [100, 200, 400, 600, 800, 1000]

# Making the directories, eval is the accuracy wit hthe ground truth, 
# output is the calibration saved as a pickle file, plot is the plots that are created during optimization.

# *************************************************************************************************************

if os.path.isdir('CalibSingleFromP2D/plots') == False:
    os.mkdir('CalibSingleFromP2D/plots')

if os.path.isdir('CalibSingleFromP2D/plots/time_' + name) == False:
    os.mkdir('CalibSingleFromP2D/plots/time_' + name)

with open('CalibSingleFromP2D/plots/time_' + name +  '/result_sync.csv','a') as file:
    writer1 = csv.writer(file)
    writer1.writerow(["shift gt", "shift", "subset", "camera1", "camera2"])
    file.close

with open('CalibSingleFromP2D/plots/time_' + name + '/result_average_sync.csv','a') as file:
    writer1 = csv.writer(file)
    writer1.writerow(["shift gt", "cam1", "cam2", "shift avg", "shift std"])
    file.close

with open('CalibSingleFromP2D/plots/time_' + name + '/result_average_all.csv','a') as file:
    writer1 = csv.writer(file)
    writer1.writerow(["shift gt", "shift avg", "shift std", "diff avg", "diff std"])
    file.close

with open('CalibSingleFromP2D/plots/time_' + name + '/result_bundle_sync.csv','a') as file:
    writer1 = csv.writer(file)
    writer1.writerow(["cam1", "cam2", "offset", "offset pred", "offset diff", "exp", "focal pre bundle", "focal_tsai", "angle_diff pre bundle", "error_npjpe pre bundle", "focal_error pre bundle", "results_position_diff pre bundle", "focal bundle", "focal_tsai", "angle_diff bundle", "error_npjpe bundle", "focal_error bundle", "results_position_diff bundle"])
    file.close

with open('CalibSingleFromP2D/plots/time_' + name + '/result_bundle_no_sync.csv','a') as file:
    writer1 = csv.writer(file)
    writer1.writerow(["cam1", "cam2", "offset", "offset pred", "offset diff", "exp", "focal pre bundle", "focal_tsai", "angle_diff pre bundle", "error_npjpe pre bundle", "focal_error pre bundle", "results_position_diff pre bundle", "focal bundle", "focal_tsai", "angle_diff bundle", "error_npjpe bundle", "focal_error bundle", "results_position_diff bundle"])
    file.close

#########################

# grid_step = 10
# x_2d = np.linspace(-5, 5, grid_step)
# y_2d = np.linspace(0, 10, grid_step)
# xv_2d, yv_2d = np.meshgrid(x_2d, y_2d)
# coords_xy_2d =np.array((xv_2d.ravel(), yv_2d.ravel())).T

#############

with open('CalibSingleFromP2D/camera-parameters.json', 'r') as f:
    h36m_json = json.load(f)

tsai_cal = ['CalibSingleFromP2D/campus-tsai-c0.xml', 'CalibSingleFromP2D/campus-tsai-c1.xml', 'CalibSingleFromP2D/campus-tsai-c2.xml']
campus_array_names = ['campus4-c0_avi', 'campus4-c1_avi', 'campus4-c2_avi']
#cam_comb = util.random_combination(list(range(len(tsai_cal))), 2, np.inf)

cam_comb = [(0,1), (0,2)]
print(cam_comb)

with open('CalibSingleFromP2D/configuration.json', 'r') as f:
    configuration = json.load(f)

num = 0
focal_array = []
calib_array = []
for vid in campus_array_names:

    with open('/local/tangytob/ViTPose/vis_results/res50results/result_' + vid.split('_')[0] + '_.json', 'r') as f:
        points_2d = json.load(f)
    
    datastore_cal = data.coco_mmpose_dataloader(points_2d, bound_lower = 100, bound = 2500)  

    frame_dir = 'CalibSingleFromP2D/Frames/' + vid + '/00000000.jpg'
    img = mpimg.imread(frame_dir)
    
    ankles, cam_matrix, normal, ankleWorld, focal, focal_batch, ransac_focal, datastore_filtered = run_calibration_ransac.run_calibration_ransac(datastore_cal, 'CalibSingleFromP2D/hyperparameter.json', img, img.shape[1], img.shape[0], name, num, skip_frame = configuration['skip_frame'], max_len = configuration['max_len'], min_size = configuration['min_size'])
    focal_array.append(cam_matrix[0][0])
    calib_array.append({'cam_matrix': cam_matrix, 'ground_normal': normal, 'ground_position': ankleWorld})
    print(ankles, cam_matrix, normal)
    num = num + 1