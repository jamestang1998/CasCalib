import sys
parent_directory = os.path.abspath(os.path.join(os.getcwd(), '..'))
sys.path.append(parent_directory)
import util
import os
import data
import json

from datetime import datetime

import csv
import pickle

import run_calibration_ransac
import matplotlib.image as mpimg

today = datetime.now()
import numpy as np
import math
import plotting
import synchronization

import bundle_adjustment
import bundle_adjustment_multi
import torch
import knn
import ICP
import geometry
import matplotlib.pyplot as plt
from xml.dom import minidom
from scipy.spatial.transform import Rotation

import eval_functions
import torch
import spline_model

from eval_human_pose import Metrics
metrics = Metrics()
import csv
import plane_peturb
import h5py
import pandas as pd
import bundle_adjustment_plotly
import bundle_adjustment_plotly_fix
import itertools
import copy
import time_align
import random
import time_align_multi_person
import time_align_inlier

#The name of is the current date
name = str(today.strftime('%Y%m%d_%H%M%S')) +'_h36m'

#Gets the hyperparamter from hyperparameter.json
threshold_euc, threshold_cos, angle_filter_video, confidence, termination_cond, num_points, h, iter, focal_lr, point_lr = util.hyperparameter('/local/tangytob/Summer2023/multiview_synchronization/scripts_pairwise_bundle_temporal/hyperparameter_h36m.json')

hyperparam_dict = {"threshold_euc": threshold_euc, "threshold_cos": threshold_cos, "angle_filter_video": angle_filter_video, "confidence": confidence, "termination_cond": termination_cond, "num_points": num_points, "h": h, "optimizer_iteration" :iter, "focal_lr" :focal_lr, "point_lr": point_lr}

#Making the directories, eval is the accuracy wit hthe ground truth, output is the calibration saved as a pickle file, plot is the plots that are created during optimization.

if os.path.isdir('./plots') == False:
    os.mkdir('./plots')

plot_scale = 1
line_amount = 50

ankles_array = []

factor = 10
start = 0
end = 0#factor
index_array = []
skip = 1

camera_list = [
'54138969',
'55011271',
'58860488',
'60457274'
]

scene_num = 0
#EXTRINSIC MATRIX DETAILS
#https://ksimek.github.io/2012/08/22/extrinsic/
#for subject in ['S1', 'S5','S6','S7','S8','S9']:
#for subject in ['S5','S6','S7','S8','S9']:
#for subject in ['S1','S5','S6','S7','S8']:

#REF    SYNC   REF          SYNC         REF        SYNC
#scale, scale, shift start, shift start, shift end, shift end
#experiments_time = [experiments_time[0]]

time_pred = {"0_S1_54138969_55011271": 9.0,
"0_S1_55011271_58860488": 12.5,
"0_S1_54138969_60457274": 11.333333333333334,
"0_S1_58860488_60457274": 11.75,
"0_S1_54138969_58860488": 9.8,
"0_S1_55011271_60457274": 8.333333333333334,
"0_S5_54138969_55011271": 19.0,
"0_S5_55011271_58860488": 18.0,
"0_S5_54138969_60457274": 16.666666666666668,
"0_S5_58860488_60457274": 15.0,
"0_S5_54138969_58860488": 12.0,
"0_S5_55011271_60457274": 10.333333333333334,
"0_S6_54138969_55011271": 22.0,
"0_S6_55011271_58860488": 18.0,
"0_S6_54138969_60457274": 15.666666666666666,
"0_S6_58860488_60457274": 16.0,
"0_S6_54138969_58860488": 18.6,
"0_S6_55011271_60457274": 20.333333333333332,
"0_S7_54138969_55011271": 14.0,
"0_S7_55011271_58860488": 15.5,
"0_S7_54138969_60457274": 13.333333333333334,
"0_S7_58860488_60457274": 13.0,
"0_S7_54138969_58860488": 10.6,
"0_S7_55011271_60457274": 9.0,
"0_S8_54138969_55011271": 18.0,
"0_S8_55011271_58860488": 16.0,
"0_S8_54138969_60457274": 17.0,
"0_S8_58860488_60457274": 15.75,
"0_S8_54138969_58860488": 12.6,
"0_S8_55011271_60457274": 10.666666666666666,
"0_S9_54138969_55011271": 12.0,
"0_S9_55011271_58860488": 14.5,
"0_S9_54138969_60457274": 11.666666666666666,
"0_S9_58860488_60457274": 11.5,
"0_S9_54138969_58860488": 10.2,
"0_S9_55011271_60457274": 8.666666666666666,
"0_S11_54138969_55011271": 12.0,
"0_S11_55011271_58860488": 12.5,
"0_S11_54138969_60457274": 11.333333333333334,
"0_S11_58860488_60457274": 12.25,
"0_S11_54138969_58860488": 10.6,
"0_S11_55011271_60457274": 9.0,
"50_S1_54138969_55011271": 59.0,
"50_S1_55011271_58860488": 46.5,
"50_S1_54138969_60457274": 50.666666666666664,
"50_S1_58860488_60457274": 53.75,
"50_S1_54138969_58860488": 52.6,
"50_S1_55011271_60457274": 52.333333333333336,
"50_S5_54138969_55011271": 69.0,
"50_S5_55011271_58860488": 51.0,
"50_S5_54138969_60457274": 55.333333333333336,
"50_S5_58860488_60457274": 56.5,
"50_S5_54138969_58860488": 55.2,
"50_S5_55011271_60457274": 54.166666666666664,
"50_S6_54138969_55011271": 72.0,
"50_S6_55011271_58860488": 54.0,
"50_S6_54138969_60457274": 56.333333333333336,
"50_S6_58860488_60457274": 50.5,
"50_S6_54138969_58860488": 56.2,
"50_S6_55011271_60457274": 50.333333333333336,
"50_S7_54138969_55011271": 64.0,
"50_S7_55011271_58860488": 48.5,
"50_S7_54138969_60457274": 52.0,
"50_S7_58860488_60457274": 54.5,
"50_S7_54138969_58860488": 53.4,
"50_S7_55011271_60457274": 52.666666666666664,
"50_S8_54138969_55011271": 68.0,
"50_S8_55011271_58860488": 52.5,
"50_S8_54138969_60457274": 58.333333333333336,
"50_S8_58860488_60457274": 59.25,
"50_S8_54138969_58860488": 57.4,
"50_S8_55011271_60457274": 56.333333333333336,
"50_S9_54138969_55011271": 62.0,
"50_S9_55011271_58860488": 48.0,
"50_S9_54138969_60457274": 50.666666666666664,
"50_S9_58860488_60457274": 53.25,
"50_S9_54138969_58860488": 51.8,
"50_S9_55011271_60457274": 51.333333333333336,
"50_S11_54138969_55011271": 62.0,
"50_S11_55011271_58860488": 49.5,
"50_S11_54138969_60457274": 52.666666666666664,
"50_S11_58860488_60457274": 56.0,
"50_S11_54138969_58860488": 54.0,
"50_S11_55011271_60457274": 53.333333333333336,
"100_S1_54138969_55011271": 109.0,
"100_S1_55011271_58860488": 96.5,
"100_S1_54138969_60457274": 100.66666666666667,
"100_S1_58860488_60457274": 103.75,
"100_S1_54138969_58860488": 102.6,
"100_S1_55011271_60457274": 102.33333333333333,
"100_S5_54138969_55011271": 119.0,
"100_S5_55011271_58860488": 101.0,
"100_S5_54138969_60457274": 105.0,
"100_S5_58860488_60457274": 106.0,
"100_S5_54138969_58860488": 104.8,
"100_S5_55011271_60457274": 103.66666666666667,
"100_S6_54138969_55011271": 122.0,
"100_S6_55011271_58860488": 104.5,
"100_S6_54138969_60457274": 106.66666666666667,
"100_S6_58860488_60457274": 100.75,
"100_S6_54138969_58860488": 106.4,
"100_S6_55011271_60457274": 100.5,
"100_S7_54138969_55011271": 114.0,
"100_S7_55011271_58860488": 98.0,
"100_S7_54138969_60457274": 101.66666666666667,
"100_S7_58860488_60457274": 104.25,
"100_S7_54138969_58860488": 103.2,
"100_S7_55011271_60457274": 102.5,
"100_S8_54138969_55011271": 118.0,
"100_S8_55011271_58860488": 102.5,
"100_S8_54138969_60457274": 108.33333333333333,
"100_S8_58860488_60457274": 109.5,
"100_S8_54138969_58860488": 107.6,
"100_S8_55011271_60457274": 106.5,
"100_S9_54138969_55011271": 112.0,
"100_S9_55011271_58860488": 97.5,
"100_S9_54138969_60457274": 100.33333333333333,
"100_S9_58860488_60457274": 103.0,
"100_S9_54138969_58860488": 101.4,
"100_S9_55011271_60457274": 101.0,
"100_S11_54138969_55011271": 112.0,
"100_S11_55011271_58860488": 98.0,
"100_S11_54138969_60457274": 101.66666666666667,
"100_S11_58860488_60457274": 105.5,
"100_S11_54138969_58860488": 103.4,
"100_S11_55011271_60457274": 102.83333333333333,
"150_S1_54138969_55011271": 160.0,
"150_S1_55011271_58860488": 147.0,
"150_S1_54138969_60457274": 151.0,
"150_S1_58860488_60457274": 154.0,
"150_S1_54138969_58860488": 152.8,
"150_S1_55011271_60457274": 152.5,
"150_S5_54138969_55011271": 169.0,
"150_S5_55011271_58860488": 150.0,
"150_S5_54138969_60457274": 154.33333333333334,
"150_S5_58860488_60457274": 155.75,
"150_S5_54138969_58860488": 154.6,
"150_S5_55011271_60457274": 153.5,
"150_S6_54138969_55011271": 173.0,
"150_S6_55011271_58860488": 155.0,
"150_S6_54138969_60457274": 157.33333333333334,
"150_S6_58860488_60457274": 151.5,
"150_S6_54138969_58860488": 157.2,
"150_S6_55011271_60457274": 151.33333333333334,
"150_S7_54138969_55011271": 164.0,
"150_S7_55011271_58860488": 148.0,
"150_S7_54138969_60457274": 152.66666666666666,
"150_S7_58860488_60457274": 155.75,
"150_S7_54138969_58860488": 154.4,
"150_S7_55011271_60457274": 154.16666666666666,
"150_S8_54138969_55011271": 163.0,
"150_S8_55011271_58860488": 150.0,
"150_S8_54138969_60457274": 156.33333333333334,
"150_S8_58860488_60457274": 157.75,
"150_S8_54138969_58860488": 156.2,
"150_S8_55011271_60457274": 155.33333333333334,
"150_S9_54138969_55011271": 163.0,
"150_S9_55011271_58860488": 147.5,
"150_S9_54138969_60457274": 150.33333333333334,
"150_S9_58860488_60457274": 153.0,
"150_S9_54138969_58860488": 151.4,
"150_S9_55011271_60457274": 151.0,
"150_S11_54138969_55011271": 162.0,
"150_S11_55011271_58860488": 148.5,
"150_S11_54138969_60457274": 152.0,
"150_S11_58860488_60457274": 155.5,
"150_S11_54138969_58860488": 153.6,
"150_S11_55011271_60457274": 153.16666666666666,
"200_S1_54138969_55011271": 209.0,
"200_S1_55011271_58860488": 197.0,
"200_S1_54138969_60457274": 201.0,
"200_S1_58860488_60457274": 204.0,
"200_S1_54138969_58860488": 202.8,
"200_S1_55011271_60457274": 202.5,
"200_S5_54138969_55011271": 220.0,
"200_S5_55011271_58860488": 200.5,
"200_S5_54138969_60457274": 205.0,
"200_S5_58860488_60457274": 206.0,
"200_S5_54138969_58860488": 205.0,
"200_S5_55011271_60457274": 203.83333333333334,
"200_S6_54138969_55011271": 228.0,
"200_S6_55011271_58860488": 207.5,
"200_S6_54138969_60457274": 209.33333333333334,
"200_S6_58860488_60457274": 203.0,
"200_S6_54138969_58860488": 208.6,
"200_S6_55011271_60457274": 202.5,
"200_S7_54138969_55011271": 214.0,
"200_S7_55011271_58860488": 198.0,
"200_S7_54138969_60457274": 202.66666666666666,
"200_S7_58860488_60457274": 205.75,
"200_S7_54138969_58860488": 204.4,
"200_S7_55011271_60457274": 204.16666666666666,
"200_S8_54138969_55011271": 219.0,
"200_S8_55011271_58860488": 202.5,
"200_S8_54138969_60457274": 208.33333333333334,
"200_S8_58860488_60457274": 209.25,
"200_S8_54138969_58860488": 207.4,
"200_S8_55011271_60457274": 206.33333333333334,
"200_S9_54138969_55011271": 221.0,
"200_S9_55011271_58860488": 138.5,
"200_S9_54138969_60457274": 164.0,
"200_S9_58860488_60457274": 176.5,
"200_S9_54138969_58860488": 182.2,
"200_S9_55011271_60457274": 185.0,
"200_S11_54138969_55011271": 10.0,
"200_S11_55011271_58860488": 97.5,
"200_S11_54138969_60457274": 68.33333333333333,
"200_S11_58860488_60457274": 54.75,
"200_S11_54138969_58860488": 83.0,
"200_S11_55011271_60457274": 102.66666666666667}
#experiments_time = [100, 200, 400, 600, 800, 1000]
experiments_time = [0, 50, 100, 150,200]
image_scale = 1.0
cam_comb = util.random_combination([0,1,2,3], 2, np.inf)
#cam_comb = [(0,1), (0,2), (0,3)]
print(cam_comb)

#for subject in ['S9', 'S11']:

if os.path.isdir('./plots/time_' + name) == False:
    os.mkdir('./plots/time_' + name)

'''
with open('./plots/time_' + name +  '/result_sync.csv','a') as file:

    writer1 = csv.writer(file)
    writer1.writerow(["exp", "size", "shift", "gt shift", "difference", "subset", "camera1", "camera2", "subject"])
    file.close
'''
with open('./plots/time_' + name + '/result_average_sync.csv','a') as file:

    writer1 = csv.writer(file)
    writer1.writerow(["size", "subject", "cam1", "cam2", "shift avg", "gt shift avg", "difference avg"])
    file.close

with open('./plots/time_' + name + '/result_average_all.csv','a') as file:

    writer1 = csv.writer(file)
    writer1.writerow(["subject", "size", "shift avg", "gt shift avg", "difference avg"])
    file.close

with open('./plots/time_' + name + '/result_bundle.csv','a') as file:

    writer1 = csv.writer(file)
    writer1.writerow(["subject", "cam1", "cam2", "offset", "offset pred", "offset diff", "exp", "mode", "focal pre bundle", "focal_tsai", "angle_diff pre bundle", "error_npjpe pre bundle", "focal_error pre bundle", "results_position_diff pre bundle", "focal bundle", "focal_tsai", "angle_diff bundle", "error_npjpe bundle", "focal_error bundle", "results_position_diff bundle"])
    file.close

subject_array = ["S1","S5", "S6", "S7", "S8", "S9", "S11"]
#subject_array = ["S5"]
#subject_array = ["S9", "S11"]
#subject_array = ["S11"]
#test_list = [0, 100, 200, 300]

for t in experiments_time:
    with open('./plots/time_' + name +  '/result_sync_' + str(t) + '_.csv','a') as file:

        writer1 = csv.writer(file)
        writer1.writerow(["exp", "size", "shift", "gt shift", "difference", "subset", "camera1", "camera2", "subject"])
        file.close
    
    with open('./plots/time_' + name +  '/result_no_sync_' + str(t) + '_.csv','a') as file:

        writer1 = csv.writer(file)
        writer1.writerow(["exp", "size", "shift", "gt shift", "difference", "subset", "camera1", "camera2", "subject"])
        file.close
#test_list = [300]
    
for subject in subject_array:
    for scene in ['Walking']:

        if subject == "S7":
            scene = 'Walking 1'
        focal_array = []
        
        
        detection_path_array = []
        frame_dir_array = []
        single_view_array = []
        data_array = []

        #hdf5_path = '/local2/tangytob/h36m/processed/processed/' + subject + '/' + scene + '/annot.h5'
        '''
        h36m_3d = None
        h36m_2d = None

        h36m_3d_camera = None
        with h5py.File(hdf5_path, "r") as f:
            # Print time root level object names (aka keys) 
            # these can be group or dataset names 
            #print(f['action'], " HELLOOOASDASDASD")
            #print("Keys: %s" % f.keys())
            #print(f.value)
            # get first object name/key; may or may NOT be a group
            a_group_key = list(f.keys())[0]

            # get the object type for a_group_key: usutimey group or dataset
            #print(type(f[a_group_key])) 

            # If a_group_key is a group name, 
            # this gets the object names in the group and returns as a list
            hdf5_data = list(f[a_group_key])

            # If a_group_key is a dataset name, 
            # this gets the dataset values and returns as a list
            hdf5_data = list(f[a_group_key])
            # preferred methods to get dataset values:
            ds_obj = f[a_group_key]      # returns as a h5py dataset object
            ds_arr = f[a_group_key][()]  # returns as a numpy array

            h36m_3d = f['pose']['3d'][:]
            h36m_2d = f['pose']['2d'][:]

            h36m_3d_camera = f['camera'][:]
        '''
        #########################
        
        scale = 1
        scale_tsai = 1

        grid_step = 10

        average_angle_array = []
        dist_error = []

        ######################################################################################

        detections_array = []

        #######################################################################################

        x_2d = np.linspace(-5, 5, grid_step)
        y_2d = np.linspace(0, 10, grid_step)
        xv_2d, yv_2d = np.meshgrid(x_2d, y_2d)
        coords_xy_2d =np.array((xv_2d.ravel(), yv_2d.ravel())).T

        #############
        plane_points_3d = []

        file_num = 0

        pose_track = 1

        plane_points_time = []
        time_points_time = []

        print(len(plane_points_time))
        #stop
        #############
        cam_axis = []
        cam_position = []

        pos0 = np.array([0,0,0])#plane_matrix_array0[:3,3]
        axis0 = np.array([[1,0,0],[0,1,0],[0,0,1]])#plane_matrix_array0[:3,:3]

        time_points_array4 = []
        time_track = []

        frame_dict_array = []
        center_array = []

        time_cam_intrinsics = []


        with open('/local/tangytob/Summer2023/camera_calibration_synchronization/camera-parameters.json', 'r') as f:
            h36m_json = json.load(f)

        gt_rotation_array = []
        gt_translation_array = []

        distortion_k_array = []
        distortion_p_array = []
        
        points_h36m_3d = []

        points_h36m_3d1 = []
        points_h36m_3d2 = []
        points_h36m_center = []

        plane_matrix_array = []
        plane_dict_array = []
        #The rotation matrix is the camera orientation, but the translation is the extrinsic translation probably
        save_dict_array = []
        points_2d_array = []
        ankle_head_2d_array = []

        pose_2d_array = []

        gt_intrinsics_array = []

        #############################
        # FOCAL INITALIZATION
        '''
        for k1 in list(h36m_json['extrinsics'][subject].keys()):
            continue
            ind_camera = [i for i, n in enumerate(h36m_3d_camera) if int(n) == int(k1)]

            keypoint_index = [
                        0,1,2,3,6,7,8,12,13,14,15,17,18,19,25,26,27 
            ]
            #points_2d = np.array(h36m_2d)[:, [3, 8, 24], :][ind_camera, :, :]
            points_2d = np.array(h36m_2d)[:, keypoint_index, :][ind_camera, :, :]
            print(points_2d.shape, " POINTS IN H36M 2d")
            points_2d_array.append(points_2d)
            #####################

            with open('/local/tangytob/Summer2023/camera_calibration_synchronization/configuration.json', 'r') as f:
                configuration = json.load(f)


            datastore_cal = data.h36m_gt_dataloader(points_2d)        
            
            frame_dir = '/local2/tangytob/h36m/processed/processed/' + subject + '/' + scene + '/imageSequence/' + k1 + '/img_000001.jpg'
            img = mpimg.imread(frame_dir)
            
            ankles, cam_matrix, normal, ankleWorld, ransac_focal, datastore_filtered = run_calibration_ransac.run_calibration_ransac(datastore_cal, '/local/tangytob/Summer2023/camera_calibration_synchronization/hyperparameter.json', img, img.shape[1], img.shape[0], str(k1) + '_', name, skip_frame = configuration['skip_frame'], max_len = configuration['max_len'], min_size = configuration['min_size'])
            
            focal_array.append(ransac_focal)
        '''    
        
        #focal_init = 1145.0#np.median(focal_array)
        #focal_init = np.median(focal_array)
        #focal_init = None
        print(list(h36m_json['extrinsics'][subject].keys()), " THE KEYSSSSSS")
        #stop
        for k1 in list(h36m_json['extrinsics'][subject].keys()):
            print(k1, " THIS IS THE KEY")
            distortion_k_array.append(np.array([h36m_json['intrinsics'][k1]["distortion"][0], h36m_json['intrinsics'][k1]["distortion"][1], h36m_json['intrinsics'][k1]["distortion"][4]]))
            distortion_p_array.append(np.array([h36m_json['intrinsics'][k1]["distortion"][2], h36m_json['intrinsics'][k1]["distortion"][3]]))

            gt_rotation = h36m_json['extrinsics'][subject][k1]['R']
            gt_translation = list(np.array(h36m_json['extrinsics'][subject][k1]['t'])/1000)

            c_rotation, c_translation = util.extrinsics_to_camera_pose(gt_rotation, gt_translation)
            
            gt_intrinsics_array.append(np.array(h36m_json['intrinsics'][k1]["calibration_matrix"]))
            gt_rotation_array.append(gt_rotation)
            gt_translation_array.append(np.array([c_translation[0][0], c_translation[1][0], c_translation[2][0]]))

            with open('/local/tangytob/Summer2023/camera_calibration_synchronization/configuration.json', 'r') as f:
                configuration = json.load(f)

            with open('/local2/tangytob/h36m_hrnet_walking/' + subject + '_Walking_' + str(k1) + '/result_Videos_' + scene + '_.json', 'r') as f:
                points_2d = json.load(f)

            print(len(points_2d["Info"]), " LENGTH OF THE ARRAY")
            datastore = data.coco_mmpose_dataloader(points_2d, scale_x = image_scale, scale_y = image_scale)

            print(datastore.__len__(), " THE DATASTORE")
            #print(datastore.getData(), " THE DATASTORE")
            '''
            for ps in list(datastore.getData().keys()):
                #print(ps, datastore.getData()[ps])
                for pos in datastore.getData()[ps]:
                    print(ps, pos['id'])
            '''
            #stop
            #datastore_cal = data.h36m_gt_dataloader(points_2d)        
            
            frame_dir = '/local2/tangytob/CPSC533R/human3_6/frames/' + subject + '/' + scene + '.' + str(k1) + '.mp4/frame0.jpg'
            img = mpimg.imread(frame_dir)

            with open('/local/tangytob/Summer2023/multiview_synchronization/calibration/h36m_calibration_20220620_160757/' + subject + '_' + k1 + '/calibration.pickle', 'rb') as f:
                pickle_file = pickle.load(f)
            #print(pickle_file)
            #print(pickle_file["ankles"].shape, " HSPAE")
            cam_matrix = np.array(pickle_file["cam_matrix"])#np.array(h36m_json['intrinsics'][k1]["calibration_matrix"])
            normal = np.array(pickle_file["normal"])#np.array(gt_rotation) @ np.array([0,0,1])
            ankleWorld = np.array(pickle_file["ankles"][:, 0])#np.array(gt_translation)
            #datastore_filtered = loaded_list['datastore_filtered']
            
            save_dict = {"cam_matrix":cam_matrix, "ground_normal":normal, "ground_position":ankleWorld}
            ##################################
            #datastore = data.h36m_gt_dataloader(points_2d) 
            #print(datastore.getData())
            #stop
            data_2d = util.get_ankles_heads_dictionary(datastore, cond_tol = confidence)
            pose_2d = util.get_ankles_heads_pose_dictionary(datastore, cond_tol = confidence)
            print(len(pose_2d), len(data_2d), " THIS IS POSE 2D")

            ankle_head_2d_array.append(data_2d)
            pose_2d_array.append(pose_2d)
            
            plane_matrix, basis_matrix = geometry.find_plane_matrix(save_dict["ground_normal"], np.linalg.inv(save_dict['cam_matrix']),save_dict['ground_position'], img.shape[1], img.shape[0])
            
            to_pickle_plane_matrix = {"plane_matrix": plane_matrix,'intrinsics': save_dict['cam_matrix']}

            plane_data_2d = geometry.camera_to_plane(data_2d, cam_matrix, plane_matrix, save_dict['ground_position'], save_dict["ground_normal"], img.shape[1], img.shape[0])

            plane_matrix_array.append(plane_matrix)
            #print(plane_matrix, " PLANE MATRIX!!!")

            plane_dict_array.append(plane_data_2d)
            ##################################
            save_dict_array.append(save_dict)

            #points_h36m_center.append(np.expand_dims(np.array(points_center), axis = 0))
        #stop
        plane_dict_array_experiment0 = {}

        #REF    SYNC   REF          SYNC         REF        SYNC
        #scale, scale, shift start, shift start, shift end, shift end
        #for k in range(len(experiments_time)):
        #print("***************************")
        #print(plane_dict_array[0])
        plane_dict_array_experiment = []

        
        #for cam in [(1,2)]:#cam_comb:
        #for cam in cam_comb:
        
        camera_name = list(h36m_json['extrinsics'][subject].keys())
        '''
        for cam in [0,1,2,3]:
            x_plot = []
            y_plot = []
            frame_plot = []
            plane_dict_array0 = plane_dict_array[cam].copy()
            
            for f_k in list(plane_dict_array0.keys()):
                #print(list(plane_dict_array0[f_k].values())[0][0], list(plane_dict_array0[f_k].values())[0][1])
                x_plot.append(list(plane_dict_array0[f_k].values())[0][0])
                y_plot.append(list(plane_dict_array0[f_k].values())[0][1])
                frame_plot.append(f_k)
            #stop
            df_array = pd.DataFrame(columns=['x', 'y','frame'])
            df_array.x = x_plot
            df_array.y = y_plot
            df_array.frame = frame_plot 

            plotting.plot_slide1(df_array, save_dir, name + '_' + str(camera_name[cam]))
        '''

        for offset in experiments_time:
            
            exp_shift_average = []
            exp_diff_average = []
            exp_gt_average = []
            for cam in cam_comb:
                print(cam)
                cam1 = cam[0]
                cam2 = cam[1]
                print(cam1, cam2, subject, np.dot(np.array(gt_rotation_array[cam1])[2, :], np.array(gt_rotation_array[cam2])[2, :]), " THE CAMS")
                if np.dot(np.array(gt_rotation_array[cam1])[2, :], np.array(gt_rotation_array[cam2])[2, :]) < -0.85:
                    print(" FAILED ")
                    continue

                print(" MADE IT")

                plane_dict_array0 = plane_dict_array[cam1].copy()
                plane_dict_array1 = plane_dict_array[cam2].copy()
                '''
                x_plot = []
                y_plot = []
                frame_plot = []

                for f_k in list(plane_dict_array0.keys()):
                    #print(list(plane_dict_array0[f_k].values())[0][0], list(plane_dict_array0[f_k].values())[0][1])
                    x_plot.append(list(plane_dict_array0[f_k].values())[0][0])
                    y_plot.append(list(plane_dict_array0[f_k].values())[0][1])
                    frame_plot.append(f_k)
                #stop
                plane_dict_array0 = {}
                frame_plot, x_plot, y_plot = time_align.track_smooth(frame_plot, x_plot, y_plot)

                for f_k in range(len(frame_plot)):
                    plane_dict_array0[frame_plot[f_k]] = {0: [x_plot[f_k], y_plot[f_k]]}
                
                #############################################
                x_plot = []
                y_plot = []
                frame_plot = []

                for f_k in list(plane_dict_array1.keys()):
                    #print(list(plane_dict_array1[f_k].values())[0][0], list(plane_dict_array1[f_k].values())[0][1])
                    x_plot.append(list(plane_dict_array1[f_k].values())[0][0])
                    y_plot.append(list(plane_dict_array1[f_k].values())[0][1])
                    frame_plot.append(f_k)
                #stop
                plane_dict_array1 = {}
                frame_plot, x_plot, y_plot = time_align.track_smooth(frame_plot, x_plot, y_plot)

                for f_k in range(len(frame_plot)):
                    plane_dict_array1[frame_plot[f_k]] = {0: [x_plot[f_k], y_plot[f_k]]}
                '''
                #############################################

                track1 = plane_dict_array0.keys()
                track2 = plane_dict_array1.keys()

                set1 = set(track1)
                set2 = set(track2)

                '''
                print(track1, list(h36m_json['extrinsics'][subject].keys())[cam1], " TRACK 1 !!!")
                print(track2, list(h36m_json['extrinsics'][subject].keys())[cam2], " TRACK 2 !!!")
                stop
                continue
                '''
                
                # Find the intersection using the intersection() method
                intersection = sorted(list(set1.intersection(set2)))
                #print(intersection)
                #print(len(intersection), " THE LENGHT")
                
                #1179
                #2706


                shift_avg_array = []
                diff_avg_array = []
                true_offset_array = []

                for exp in range(0, 1):

                    print(len(intersection), " THE SUBSET")
                    test_size = int(len(intersection))
                    subsect, subset_start = util.get_random_windowed_subset(intersection, test_size, offset = offset, start = 0)
                    subsect1, subset_start1 = util.get_random_windowed_subset(intersection, test_size, start = subsect[offset])
                    #subsect1, subset_start1 = util.get_random_windowed_subset(intersection, test_size, start = subset_start, offset = offset)
                    
                    print(subset_start, " subsect")
                    print(subset_start1, " subsect1")
                    #stop
                    true_offset = abs(subset_start - subset_start1)
                    #print(true_offset, " true_offset")
                    #print(subsect, offset, " THE SUBSET")
                    #print(subsect, exp, " SUBSET")
                    #print(plane_dict_array0)

                    #print  (plane_dict_array0, exp," plane_dict_array0 BEFORE !")
                    plane_dict_array_subset0 = {key: plane_dict_array0.get(key, None) for key in subsect}
                    plane_dict_array_subset1 = {key: plane_dict_array1.get(key, None) for key in subsect1}

                    pose_dict_array_subset0 = {key: pose_2d_array[cam1].get(key, None) for key in subsect}
                    pose_dict_array_subset1 = {key: pose_2d_array[cam2].get(key, None) for key in subsect1}

                    '''
                    x_plot = []
                    y_plot = []
                    frame_plot = []

                    for f_k in list(plane_dict_array_subset0.keys()):
                        #print(list(plane_dict_array_subset0[f_k].values())[0][0], list(plane_dict_array_subset0[f_k].values())[0][1])
                        x_plot.append(list(plane_dict_array_subset0[f_k].values())[0][0].copy())
                        y_plot.append(list(plane_dict_array_subset0[f_k].values())[0][1].copy())
                        frame_plot.append(f_k)
                    #stop
                    plane_dict_array_spline0 = {}
                    frame_plot, x_plot, y_plot = time_align.track_smooth(frame_plot, x_plot, y_plot)

                    for f_k in range(len(frame_plot)):
                        plane_dict_array_spline0[frame_plot[f_k]] = {0: [x_plot[f_k], y_plot[f_k]]}
                    
                    #############################################
                    x_plot = []
                    y_plot = []
                    frame_plot = []

                    for f_k in list(plane_dict_array_subset1.keys()):
                        #print(list(plane_dict_array_subset1[f_k].values())[0][0], list(plane_dict_array_subset1[f_k].values())[0][1])
                        x_plot.append(list(plane_dict_array_subset1[f_k].values())[0][0])
                        y_plot.append(list(plane_dict_array_subset1[f_k].values())[0][1])
                        frame_plot.append(f_k)
                    #stop
                    plane_dict_array_spline1 = {}
                    frame_plot, x_plot, y_plot = time_align.track_smooth(frame_plot, x_plot, y_plot)

                    for f_k in range(len(frame_plot)):
                        plane_dict_array_spline1[frame_plot[f_k]] = {0: [x_plot[f_k], y_plot[f_k]]}
                    
                    #############################################          
                    
                    plane_dict_array_no_spline0 = util.rename_keys_to_sequence(plane_dict_array_subset0)
                    plane_dict_array_no_spline1 = util.rename_keys_to_sequence(plane_dict_array_subset1)

                    plane_dict_array_experiment0 = util.rename_keys_to_sequence(plane_dict_array_spline0)
                    plane_dict_array_experiment1 = util.rename_keys_to_sequence(plane_dict_array_spline1)

                    pose_dict_array_experiment0 = util.rename_keys_to_sequence(pose_dict_array_subset0)
                    pose_dict_array_experiment1 = util.rename_keys_to_sequence(pose_dict_array_subset1)
                    '''
                    for mode in [True, False]:
                    #for mode in [False]:
                        
                        mode_string = "no_sync"
                        if mode is True:
                            mode_string = "sync"
                        name_folder = '/' + str(subject) + '_' + str(camera_name[cam1]) + '_' + str(camera_name[cam2]) + '_' + str(offset) + '_' + str(exp) + '_' + mode_string

                        save_dir = './plots/time_' + name + name_folder

                        if os.path.isdir('./plots/time_' + name + name_folder) == False:
                            os.mkdir('./plots/time_' + name + name_folder)

                        if os.path.isdir('./plots/time_' + name + name_folder + '/search_time') == False:
                            os.mkdir('./plots/time_' + name + name_folder + '/search_time')

                        if os.path.isdir('./plots/time_' + name + name_folder + '/search_rot') == False:
                            os.mkdir('./plots/time_' + name + name_folder + '/search_rot')

                        if os.path.isdir('./plots/time_' + name + name_folder + '/ICP') == False:
                            os.mkdir('./plots/time_' + name + name_folder + '/ICP')

                        output_folder = subject + '_' + scene
                        #PLOTS SAVED IN PLOTS NOW!!!
                        
                        if os.path.isdir('./plots/time_' + name + name_folder + '/' + output_folder) == False:
                            os.mkdir('./plots/time_' + name + name_folder + '/' + output_folder)

                        frame_array = []
                        output_path = './plots/time_' + name + name_folder + '/' + output_folder + '/'


                        cam_axis4 = []
                        cam_position4 = []
                        #print(intersection)

                        #55011271	58860488
                        #subsect = util.get_random_windowed_subset(intersection, test_size, start = 1179)
                        #subsect = util.get_random_windowed_subset(intersection, test_size)
                        
                        
                        #print(plane_dict_array_experiment0, " HIIIIII !")
                        #print(plane_dict_array_experiment, " HIIIIII !!")
                        #stop
                    
                        print(subsect1)
                        print(" please work ")            
                        print(subsect)
                        
                        #best_shift_array, best_scale_array, sync_dict_array = ICP.time_all(plane_dict_array_experiment0, [plane_dict_array_experiment1], save_dir = './plots/time_' + name + name_folder, sync = True, name = str(exp) + '_' + str(experiments_time[k]) + '_' + str(camera_name[cam1]) + '_' + str(camera_name[cam2]))
                        #best_shift_array, best_scale_array, sync_dict_array = time_align_multi_person.time_all(plane_dict_array_subset0, [plane_dict_array_subset1], save_dir = './plots/time_' + name + name_folder, sync = mode, name = str(exp) + '_' + str(offset) + '_' + str(camera_name[cam1]) + '_' + str(camera_name[cam2]))
                        best_shift_array = [0]
                        best_scale_array = list(np.zeros(len(best_shift_array))) 
                        
                        if mode == True:
                            best_shift_array = [time_pred[str(offset) + '_' + str(subject) + '_' + str(camera_list[cam1]) + '_' + str(camera_list[cam2])]]
                                
                            #best_shift_array = [time_pred["0_S1_58860488_60457274"]]
                        sync_dict_array = [time_align_inlier.time_knn(1.0, best_shift_array[0], list(plane_dict_array_subset1.keys()), list(plane_dict_array_subset0.keys()))]

                        #sync_dict_array = [time_align.time_knn(1.0, best_shift_array[0], plane_dict_array_no_spline1, plane_dict_array_no_spline0)]
                        #print(" PLEASE WORK !!! ")
                        #print(plane_dict_array0, exp," plane_dict_array0 AFTER !")
                        
                        #print(best_shift_array, best_scale_array, sync_dict_array)

                        #exp_average.append(np.absolute(best_shift_array[0]))
                        with open('./plots/time_' + name + '/result_' + mode_string + '_' + str(offset) + '_.csv','a') as file:
                    
                            writer1 = csv.writer(file)

                            writer1.writerow([str(exp), str(offset), str(best_shift_array[0]), str(true_offset), str(abs(best_shift_array[0] - true_offset)), str(min(subsect)), str(camera_name[cam1]), str(camera_name[cam2]), str(subject)])
                            file.close
                        print(sync_dict_array)
                        if mode == True:
                            shift_avg_array.append(np.absolute(best_shift_array[0]))
                            diff_avg_array.append(abs(best_shift_array[0] - true_offset))
                            true_offset_array.append(true_offset)

                            exp_gt_average.append(true_offset)
                            exp_shift_average.append(best_shift_array[0])
                            exp_diff_average.append(abs(best_shift_array[0] - true_offset))

                        icp_rot_array, icp_init_rot_array, icp_shift_array, init_ref_center_array, init_sync_center_array, time_shift_array, time_scale_array, sync_dict_array, index_array = ICP.icp(plane_dict_array_subset0, [plane_dict_array_subset1], best_shift_array, best_scale_array, sync_dict_array, save_dir = './plots/time_' + name + name_folder, name = '_' + subject)

                        pre_bundle_cal_array = []
                        single_view_cal_array = []
                        cam_intrinsics_array = []
                        pose_2d_array_comb = []
                        #pose_2d_array_experiment = [pose_dict_array_experiment0, pose_dict_array_experiment1]
                        pose_2d_array_experiment = [pose_2d_array[cam1], pose_2d_array[cam2]]
                        #print(pose_2d_array_experiment)
   
                        index_dict = time_align.time_knn_array(1.0, best_shift_array[0] - offset, np.array(list(subsect)))
                        print("***************************************!!!111111")
                        print(index_dict)
                        
                        result_index_list = [int(item) for item in list(index_dict.values())]
                        indices_array = [list(index_dict.keys()), result_index_list]
                        
                        print(indices_array)
                        #stop
                        #print(sync_dict_array[0])
                        #stop
                        for i in range(0, 2):
    
                            #icp_sync_shift_matrix = None

                            init_ref_shift_matrix = None
                            init_sync_shift_matrix = None
                            icp_rot_matrix = None
                            init_rot_matrix = None
                            plane_matrix = plane_matrix_array[cam[i]]
                            

                            if i == 0:
                                init_ref_shift_matrix = np.array([init_ref_center_array[0][0], init_ref_center_array[0][1]])
                                init_sync_shift_matrix = np.array([init_ref_center_array[0][0], init_ref_center_array[0][1]])
                                icp_rot_matrix = np.array([[1,0 ], [0,1]])
                                init_rot_matrix = np.array([[1,0 ], [0,1]])
                                
                                #icp_sync_shift_matrix = np.array([icp_shift_array[0][0], icp_shift_array[0][1]])
                            
                            else:
                                init_ref_shift_matrix = init_ref_center_array[i - 1]
                                init_sync_shift_matrix = init_sync_center_array[i - 1]
                                icp_rot_matrix = icp_rot_array[i - 1]
                                init_rot_matrix = icp_init_rot_array[i - 1]
                                plane_matrix = plane_matrix_array[cam[i]]

                                #icp_sync_shift_matrix = icp_shift_array[i - 1]


                            init_rot_matrix = np.array([[init_rot_matrix[0][0],init_rot_matrix[0][1],0],[init_rot_matrix[1][0],init_rot_matrix[1][1],0],[0,0,1]])
                            icp_rot_matrix = np.array([[icp_rot_matrix[0][0],icp_rot_matrix[0][1],0],[icp_rot_matrix[1][0],icp_rot_matrix[1][1],0],[0,0,1]])
                            
                            peturb_single_view = {'cam_matrix':  save_dict_array[cam[i]]['cam_matrix'],'ground_position':  save_dict_array[cam[i]]['ground_position'],'ground_normal':  save_dict_array[cam[i]]['ground_normal']}
                            peturb_extrinsics = {'init_sync_center_array': init_ref_shift_matrix, 'icp_rot_array': icp_rot_matrix, 'icp_init_rot_array': init_rot_matrix, 'plane_matrix_array': plane_matrix}

                            ###########################
                            print(peturb_single_view, " peturb_single_view")
                            print(peturb_extrinsics, " peturb_extrinsics")
                            ###########################
                            pre_bundle_cal_array.append(peturb_extrinsics)
                            single_view_cal_array.append(peturb_single_view)
                            T01 = pos0#np.array([0,0,0])
                            #print(T01, " TO!!!!!!@#!@#")
                            R01 = icp_rot_matrix @ init_rot_matrix @ plane_matrix[:3,:3]
                            ##########################
                            sync_axis = np.transpose(R01)
                            t01_shift = (T01 + np.array([init_ref_shift_matrix[0], init_ref_shift_matrix[1], 0]))
                            t01_rot_shift = np.linalg.norm(np.array([init_sync_shift_matrix[0], init_sync_shift_matrix[1], 0]))*(init_rot_matrix @ np.array([0,1,0]))
                            t01_rot_rot_shift = (icp_rot_matrix @ t01_rot_shift) + t01_shift
                            #print(plane_matrix[:3,3], " plane matrix")

                            print(icp_rot_matrix, " ICP !!!!!")
                            print(init_rot_matrix, " INIT !!!!!")
                            
                            sync_position = -1*(t01_rot_rot_shift - np.array([init_sync_shift_matrix[0], init_sync_shift_matrix[1], 0]) - plane_matrix[:3,3])#np.transpose(ref_axis) @ T01 - pos0
                            #sync_position = -1*(t01_rot_rot_shift - np.array([init_sync_shift_matrix[0] + icp_sync_shift_matrix[0], init_sync_shift_matrix[1] + icp_sync_shift_matrix[1], 0]) - plane_matrix[:3,3])#np.transpose(ref_axis) @ T01 - pos0

                            cam_axis4.append(np.transpose(R01))
                            cam_position4.append(sync_position)
                            #cam_axis4.append(np.transpose(icp_rot_matrix @ init_rot_matrix @ plane_matrix[:3,:3]))
                            #cam_position4.append(icp_rot_matrix @ init_rot_matrix @ (plane_matrix[:3,3] - np.array([init_ref_shift_matrix[0], init_ref_shift_matrix[1], 0])))

                            cam_intrinsics = save_dict_array[cam[i]]#single_view_array[i]

                            cam_intrinsics_array.append(cam_intrinsics['cam_matrix'])
                            
                            pose_comb = {}
                            for ia in indices_array[i]:
                                #print(ia)
                                if ia in pose_2d_array_experiment[i]:
                                    pose_comb[ia] = pose_2d_array_experiment[i][ia]
                            
                            pose_2d_array_comb.append(pose_comb)
                            #pose_2d_array_comb.append(pose_2d_array[cam[i]][indices_array[i]])
                        print(pose_2d_array_comb[0].keys())
                        print("************")
                        print(pose_2d_array_comb[1].keys())
                        print(pose_2d_array_comb)
                        
                        matched_points = bundle_adjustment_plotly.match_3d_plotly_input2d_farthest_point(pre_bundle_cal_array, single_view_cal_array, pose_2d_array_comb,center_array, output_path, scale = scale, title = 'rotation perturb TRACKS', name = ['perturb'], k = 20)

                        distortion_k_array = []
                        distortion_p_array = []

                        bundle_rotation_matrix_array = cam_axis4
                        bundle_position_matrix_array = cam_position4
                        bundle_intrinsic_matrix_array = cam_intrinsics_array

                        run_name = '_' + str(i) + '_'

                        w0 = 1
                        w1 = 0.5
                        w2 = 1.5
                        w3 = 1

                        bundle_rotation_matrix_array, bundle_position_matrix_array, bundle_intrinsic_matrix_array = bundle_adjustment_plotly_fix.bundle_adjustment_so3_gt(matched_points, bundle_rotation_matrix_array, bundle_position_matrix_array, bundle_intrinsic_matrix_array, gt_rotation_array, gt_translation_array, gt_intrinsics_array, h, distortion_k_array, distortion_p_array, save_dir = output_path, iteration = 100, run_name = run_name, w0 = w0, w1 = w1, w2 = w2, w3 = w3)

                        #print("*************************")
                        #print(cam_axis4)
                        #print("&&&&&&&&&&&&&&&&&&&&&&&&&")
                        gt_rotation_array_comb = [gt_rotation_array[cam[0]], gt_rotation_array[cam[1]]]
                        gt_translation_array_comb = [gt_translation_array[cam[0]], gt_translation_array[cam[1]]]
                        gt_intrinsics_array_comb = [gt_intrinsics_array[cam[0]], gt_intrinsics_array[cam[1]]]
                        
                        cam_position_rotate, cam_axis_rotate, R1, translation_template_rotate, translation_rotate = metrics.procrustes_rotation_translation_template(torch.unsqueeze(torch.tensor(cam_position4), dim = 0).double(), cam_axis4, torch.unsqueeze(torch.tensor(gt_translation_array_comb), dim = 0).double(), gt_rotation_array_comb, use_reflection=False, use_scaling=True)
                        bundle_cam_position_rotate, bundle_cam_axis_rotate, bundle_R1, bundle_translation_template_rotate, bundle_translation_rotate = metrics.procrustes_rotation_translation_template(torch.unsqueeze(torch.tensor(bundle_position_matrix_array), dim = 0).double(),  bundle_rotation_matrix_array, torch.unsqueeze(torch.tensor(gt_translation_array_comb), dim = 0).double(), gt_rotation_array_comb, use_reflection=False, use_scaling=True)
                        
                        plotting.plot_camera_pose([cam_axis_rotate, gt_rotation_array_comb], [cam_position_rotate, gt_translation_array_comb], save_dir, scale = 1, name = "pre_bundle")
                        plotting.plot_camera_pose([bundle_cam_axis_rotate, gt_rotation_array_comb], [bundle_cam_position_rotate, gt_translation_array_comb], save_dir, scale = 1, name = "pose_bundle")

                        results_focal_pred1, results_focal_tsai1, angle_diff1, error_npjpe1, focal_error1, results_position_diff1 = eval_functions.evaluate(cam_intrinsics_array, cam_axis_rotate, [cam_position_rotate], gt_intrinsics_array_comb, gt_rotation_array_comb, [gt_translation_array_comb], output_path, 'pre_bundle')
                        results_focal_pred2, results_focal_tsai2, angle_diff2, error_npjpe2, focal_error2, results_position_diff2 = eval_functions.evaluate(bundle_intrinsic_matrix_array, bundle_cam_axis_rotate, [bundle_cam_position_rotate], gt_intrinsics_array_comb, gt_rotation_array_comb, [gt_translation_array_comb], output_path, 'result_bundle')

                        
                        with open('./plots/time_' + name + '/result_bundle.csv','a') as file:

                            writer1 = csv.writer(file)
                            writer1.writerow([str(subject), str(camera_name[cam1]), str(camera_name[cam2]), str(true_offset), str(best_shift_array[0]), str(abs(best_shift_array[0] - true_offset)), str(exp), mode_string, str(results_focal_pred1), str(results_focal_tsai1), str(angle_diff1), str(error_npjpe1), str(focal_error1), str(results_position_diff1), str(results_focal_pred2), str(results_focal_tsai2), str(angle_diff2), str(error_npjpe2), str(focal_error2), str(results_position_diff2)])
                            file.close
                    #stop
                if mode == True:
                    with open('./plots/time_' + name + '/result_average_sync.csv','a') as file:

                        writer1 = csv.writer(file)
                        writer1.writerow([str(offset), str(subject), str(camera_name[cam1]), str(camera_name[cam2]), str(np.mean(shift_avg_array)), str(np.mean(true_offset_array)), str(np.mean(diff_avg_array))])
                        file.close
            #stop
            with open('./plots/time_' + name + '/result_average_all.csv','a') as file:

                writer1 = csv.writer(file)
                writer1.writerow([str(subject), str(offset), str(np.mean(exp_shift_average)), str(np.mean(exp_gt_average)), str(np.mean(exp_diff_average))])
                file.close