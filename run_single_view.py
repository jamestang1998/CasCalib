# Might have to add to path
# sys.path.append('CalibS') 
import sys
sys.path.append('../../CalibSingleFromP2D/dlt_calib') 
import pickle 

from util import *
from run_calibration_ransac import *
from eval_human_pose import *
import json
from datetime import datetime
#import csv
import matplotlib.image as mpimg
import os
#import time_align
import numpy as np
import geometry
#import plotting 
import multiview_utils
#import ICP
#import bundle_adjustment
#import eval_functions
#import torch
#from xml.dom import minidom
#import math 
#import plotting_multiview

today = datetime.now()

metrics = Metrics()

name = str(today.strftime('%Y%m%d_%H%M%S'))

(threshold_euc, threshold_cos, angle_filter_video, 
 confidence, termination_cond, num_points, h, iter, focal_lr, point_lr) = util.hyperparameter(
     'hyperparameter.json')
hyperparam_dict = {"threshold_euc": threshold_euc, "threshold_cos": threshold_cos, 
                   "angle_filter_video": angle_filter_video, "confidence": confidence, 
                   "termination_cond": termination_cond, "num_points": num_points, "h": h, 
                   "optimizer_iteration" :iter, "focal_lr" :focal_lr, "point_lr": point_lr}

output_path = 'outputs/plots/all_' + name

if os.path.isdir('./outputs') == False:
    os.mkdir('./outputs')

if os.path.isdir('outputs/single_view_' + name) == False:
    os.mkdir('outputs/single_view_' + name)

with open('configuration.json', 'r') as f:
    configuration = json.load(f)

num = 0
focal_array = []
calib_array = []

ankle_head_2d_array = []
pose_2d_array = []

plane_matrix_array = []    
plane_dict_array = []

save_dict_array = []


gt_rotation_array = []
gt_translation_array = []
gt_intrinsics_array = []
    
with open(sys.argv[2], 'r') as f:
    points_2d = json.load(f)

if sys.argv[3] == "0":
    datastore_cal = data.coco_mmpose_dataloader(points_2d)  
elif sys.argv[3] == "1":
    datastore_cal = data.alphapose_dataloader(points_2d)

frame_dir = sys.argv[1]
img = mpimg.imread(frame_dir)

(ankles, cam_matrix, normal, ankleWorld, focal, focal_batch, ransac_focal, datastore_filtered) = run_calibration_ransac(
        datastore_cal, 'hyperparameter.json', img, 
        img.shape[1], img.shape[0], name, num, skip_frame = configuration['skip_frame'], 
        max_len = configuration['max_len'], min_size = configuration['min_size'], save_dir = 'outputs/single_view_' + name, plotting_true = False)
focal_array.append(cam_matrix[0][0])
calib_array.append({'cam_matrix': cam_matrix, 'ground_normal': normal, 'ground_position': ankleWorld})
#print(ankles, cam_matrix, normal) 

#########################
save_dict = {"cam_matrix":cam_matrix, "ground_normal":normal, "ground_position":ankleWorld}
##################################
if sys.argv[3] == "0":
    datastore = data.coco_mmpose_dataloader(points_2d)  
elif sys.argv[3] == "1":
    datastore = data.alphapose_dataloader(points_2d)

data_2d = multiview_utils.get_ankles_heads_dictionary(datastore, cond_tol = confidence)
pose_2d = multiview_utils.get_ankles_heads_pose_dictionary(datastore, cond_tol = confidence)

ankle_head_2d_array.append(data_2d)
pose_2d_array.append(pose_2d)

plane_matrix, basis_matrix = geometry.find_plane_matrix(save_dict["ground_normal"], np.linalg.inv(save_dict['cam_matrix']),save_dict['ground_position'], img.shape[1], img.shape[0])

to_pickle_plane_matrix = {"plane_matrix": plane_matrix,'intrinsics': save_dict['cam_matrix']}

plane_data_2d, plane_list  = geometry.camera_to_plane(data_2d, cam_matrix, plane_matrix, save_dict['ground_position'], save_dict["ground_normal"], img.shape[1], img.shape[0])

#print(plane_data_2d.values(), " HIII")
##################################
#print("HIIASD")
#round_normal, cam_matrix, depth_Z, ankleworld
save_dict = {"cam_matrix":cam_matrix.tolist(), "ground_normal":normal.tolist(), "ankleworld":ankleWorld.tolist(), "ankles": plane_list }
#print("************")
#print(save_dict)
calibration_path = 'outputs/single_view_' + name
with open(calibration_path  + '/calibration.json', 'w') as json_file:
    json.dump(save_dict, json_file)

with open(calibration_path + '/calibration.pickle', 'wb') as picklefile:
    pickle.dump(save_dict, picklefile)