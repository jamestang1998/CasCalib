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
import bundle_adjustment_plotly_fix
import itertools
import copy
import time_align_multi_person
import random
import time_align
import time_align_inlier

#The name of is the current date
name = str(today.strftime('%Y%m%d_%H%M%S')) + '_EPFL_campus4_res50'
#Gets the hyperparamter from hyperparameter.json
threshold_euc, threshold_cos, angle_filter_video, confidence, termination_cond, num_points, h, iter, focal_lr, point_lr = util.hyperparameter('/local/tangytob/Summer2023/multiview_synchronization/scripts_pairwise_bundle_temporal/hyperparameter_campus4.json')

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

scene_num = 0
#EXTRINSIC MATRIX DETAILS
#https://ksimek.github.io/2012/08/22/extrinsic/
#for subject in ['S1', 'S5','S6','S7','S8','S9']:
#for subject in ['S5','S6','S7','S8','S9']:
#for subject in ['S1','S5','S6','S7','S8']:

#REF    SYNC   REF          SYNC         REF        SYNC
#scale, scale, shift start, shift start, shift end, shift end
#experiments_time = [experiments_time[0]]

#experiments_time = [100, 200, 400, 600, 800, 1000]

image_scale = 1.0

#for subject in ['S9', 'S11']:

if os.path.isdir('./plots/time_' + name) == False:
    os.mkdir('./plots/time_' + name)

with open('./plots/time_' + name +  '/result_sync.csv','a') as file:

    writer1 = csv.writer(file)
    writer1.writerow(["shift gt", "shift", "subset", "camera1", "camera2"])
    file.close

with open('./plots/time_' + name + '/result_average_sync.csv','a') as file:

    writer1 = csv.writer(file)
    writer1.writerow(["shift gt", "cam1", "cam2", "shift avg", "shift std"])
    file.close

with open('./plots/time_' + name + '/result_average_all.csv','a') as file:

    writer1 = csv.writer(file)
    writer1.writerow(["shift gt", "shift avg", "shift std", "diff avg", "diff std"])
    file.close
'''
with open('./plots/time_' + name + '/result_bundle.csv','a') as file:

    writer1 = csv.writer(file)
    writer1.writerow(["cam1", "cam2", "offset", "offset pred", "offset diff", "exp", "mode", "focal pre bundle", "focal_tsai", "angle_diff pre bundle", "error_npjpe pre bundle", "focal_error pre bundle", "results_position_diff pre bundle", "focal bundle", "focal_tsai", "angle_diff bundle", "error_npjpe bundle", "focal_error bundle", "results_position_diff bundle"])
    file.close
'''

with open('./plots/time_' + name + '/result_bundle_sync.csv','a') as file:

    writer1 = csv.writer(file)
    writer1.writerow(["cam1", "cam2", "offset", "offset pred", "offset diff", "exp", "focal pre bundle", "focal_tsai", "angle_diff pre bundle", "error_npjpe pre bundle", "focal_error pre bundle", "results_position_diff pre bundle", "focal bundle", "focal_tsai", "angle_diff bundle", "error_npjpe bundle", "focal_error bundle", "results_position_diff bundle"])
    file.close

with open('./plots/time_' + name + '/result_bundle_no_sync.csv','a') as file:

    writer1 = csv.writer(file)
    writer1.writerow(["cam1", "cam2", "offset", "offset pred", "offset diff", "exp", "focal pre bundle", "focal_tsai", "angle_diff pre bundle", "error_npjpe pre bundle", "focal_error pre bundle", "results_position_diff pre bundle", "focal bundle", "focal_tsai", "angle_diff bundle", "error_npjpe bundle", "focal_error bundle", "results_position_diff bundle"])
    file.close

#subject_array = ["S5"]
#subject_array = ["S9", "S11"]
#subject_array = ["S11"]
#experiments_time = [0, 100, 200, 300]
experiments_time = [0, 25, 50]
#experiments_time = [0]
    
focal_array = []


detection_path_array = []
frame_dir_array = []
single_view_array = []
data_array = []

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

tsai_cal = ['campus-tsai-c0.xml', 'campus-tsai-c1.xml', 'campus-tsai-c2.xml']
campus_array_names = ['campus4-c0_avi', 'campus4-c1_avi', 'campus4-c2_avi']
'''
load_dict = [{'cam_matrix': list([[631.33502363,   0.        , 180.        ],
       [  0.        , 631.33502363, 144.        ],
       [  0.        ,   0.        ,   1.        ]]), 'ground_normal': list([-0.01233763, -0.9952843 , -0.09621303]), 'ground_position': list([ 1.87405863,  0.76515014, 14.69145591])}, {'cam_matrix': list([[333.07596527,   0.        , 180.        ],
       [  0.        , 333.07596527, 144.        ],
       [  0.        ,   0.        ,   1.        ]]), 'ground_normal': list([-0.03334644, -0.97676756, -0.21169115]), 'ground_position': list([1.85194396, 0.30550235, 8.95932008])}, {'cam_matrix': list([[867.43926438,   0.        , 180.        ],
       [  0.        , 867.43926438, 144.        ],
       [  0.        ,   0.        ,   1.        ]]), 'ground_normal': list([ 0.01148304, -0.99245271, -0.1220891 ]), 'ground_position': list([ 0.23242678,  0.71130276, 13.11832319])}]
'''

load_dict = [{'cam_matrix': list([[434.99302767,   0.        , 180.        ],
       [  0.        , 434.99302767, 144.        ],
       [  0.        ,   0.        ,   1.        ]]), 'ground_normal': list([-0.02343778, -0.99348305, -0.11154418]), 'ground_position': list([ 1.86490395,  0.77264567, 10.11576797])}, {'cam_matrix': list([[434.99302767,   0.        , 180.        ],
       [  0.        , 434.99302767, 144.        ],
       [  0.        ,   0.        ,   1.        ]]), 'ground_normal': list([-0.04067438, -0.94236456, -0.33210635]), 'ground_position': list([ 1.91172993,  0.30678204, 11.32885048])}, {'cam_matrix': list([[434.99302767,   0.        , 180.        ],
       [  0.        , 434.99302767, 144.        ],
       [  0.        ,   0.        ,   1.        ]]), 'ground_normal': list([ 0.0115356 , -0.998397  , -0.05541088]), 'ground_position': list([0.31897207, 0.70541062, 6.6369932 ])}]

time_pred = {
"0_campus4-c0_avi_campus4-c1_avi": 13,
"0_campus4-c0_avi_campus4-c2_avi": 16,
"0_campus4-c1_avi_campus4-c2_avi": 11,
"25_campus4-c0_avi_campus4-c1_avi": 35,
"25_campus4-c0_avi_campus4-c2_avi": 41,
"25_campus4-c1_avi_campus4-c2_avi": 45,
"50_campus4-c0_avi_campus4-c1_avi": 50,
"50_campus4-c0_avi_campus4-c2_avi": 66,
"50_campus4-c1_avi_campus4-c2_avi": 70
}
cam_comb = util.random_combination(list(range(len(tsai_cal))), 2, np.inf)
#cam_comb = [(0,1), (0,2)]
print(cam_comb)

with open('/local/tangytob/Summer2023/multiview_synchronization/configuration.json', 'r') as f:
    configuration = json.load(f)

num = 0
calib_array = []
for vid in campus_array_names:
    continue
    with open('/local/tangytob/ViTPose/vis_results/res50results/result_' + vid.split('_')[0] + '_.json', 'r') as f:
        points_2d = json.load(f)
    
    datastore_cal = data.coco_mmpose_dataloader(points_2d, bound_lower = 100, bound = 2500)  
    #print(datastore_cal.getData(), " WHATSUP!!!")
    #stop
    #result_terrace1-c0_.json
    '''
    with open('/local/tangytob/ViTPose/vis_results/res50results/result_' + vid.split('_')[0] + '_.json', 'r') as f:
        points_2d = json.load(f)
    datastore_cal = data.coco_mmpose_dataloader(points_2d)
    '''
    '''
    with open('/local/tangytob/ViTPose/hrnet_results/result_' + vid.split('_')[0] + '_.json', 'r') as f:
        points_2d = json.load(f)
    datastore_cal = data.coco_mmpose_dataloader(points_2d)
    '''
    frame_dir = '/local/tangytob/EPFL/videos/Frames/' + vid + '/00000000.jpg'
    img = mpimg.imread(frame_dir)
    
    ankles, cam_matrix, normal, ankleWorld, focal, focal_batch, ransac_focal, datastore_filtered = run_calibration_ransac.run_calibration_ransac(datastore_cal, '/local/tangytob/Summer2023/multiview_synchronization/scripts_temporal/hyperparameter_campus4.json', img, img.shape[1], img.shape[0], name, num, skip_frame = configuration['skip_frame'], max_len = configuration['max_len'], min_size = configuration['min_size'])
    focal_array.append(cam_matrix[0][0])
    calib_array.append({'cam_matrix': cam_matrix, 'ground_normal': normal, 'ground_position': ankleWorld})
    print(ankles, cam_matrix, normal)
    num = num + 1
print(calib_array)

focal_median = np.median(focal_array)

num = 0

calib_array = []
for vid in campus_array_names:
    continue
    with open('/local/tangytob/ViTPose/vis_results/res50results/result_' + vid.split('_')[0] + '_.json', 'r') as f:
        points_2d = json.load(f)
    
    datastore_cal = data.coco_mmpose_dataloader(points_2d, bound_lower = 100, bound = 2500)  
    #print(datastore_cal.getData(), " WHATSUP!!!")
    #stop
    #result_terrace1-c0_.json
    '''
    with open('/local/tangytob/ViTPose/vis_results/res50results/result_' + vid.split('_')[0] + '_.json', 'r') as f:
        points_2d = json.load(f)
    datastore_cal = data.coco_mmpose_dataloader(points_2d)
    '''
    '''
    with open('/local/tangytob/ViTPose/hrnet_results/result_' + vid.split('_')[0] + '_.json', 'r') as f:
        points_2d = json.load(f)
    datastore_cal = data.coco_mmpose_dataloader(points_2d)
    '''
    frame_dir = '/local/tangytob/EPFL/videos/Frames/' + vid + '/00000000.jpg'
    img = mpimg.imread(frame_dir)
    
    ankles, cam_matrix, normal, ankleWorld, focal, focal_batch, ransac_focal, datastore_filtered = run_calibration_ransac.run_calibration_ransac(datastore_cal, '/local/tangytob/Summer2023/multiview_synchronization/scripts_temporal/hyperparameter_campus4.json', img, img.shape[1], img.shape[0], name, str(num) + '_focal_median', skip_frame = configuration['skip_frame'], max_len = configuration['max_len'], min_size = configuration['min_size'], f_init=focal_median)
    #focal_array.append(cam_matrix[0][0])

    calib_array.append({'cam_matrix': cam_matrix, 'ground_normal': normal, 'ground_position': ankleWorld})
    print(ankleWorld, cam_matrix, normal)
    num = num + 1

print(calib_array)

num = 0
for vid in campus_array_names:
    intrinsic_gt_path = '/local/tangytob/EPFL/Calibrations/EPFL_gt_calibrations/EPFL_Known_Scale_Calibration/campus/tsai/single_calibration_' + str(num) + '_.json'
    
    with open(intrinsic_gt_path, 'r') as f:
        intrinsic_gt = json.load(f)
    
    cam_extrinsic_path = '/local/tangytob/EPFL/Calibrations/calibration/campus-tsai-c' + str(num) + '.xml'
    print(cam_extrinsic_path, " THE EXTRISIC PATH")
    cam_extrinsic = minidom.parse(cam_extrinsic_path)

    models_extrinsics = cam_extrinsic.getElementsByTagName('Extrinsic')
    models_instrinsics = cam_extrinsic.getElementsByTagName('Intrinsic')
    models_geometry = cam_extrinsic.getElementsByTagName('Geometry')
    
    tx = float(models_extrinsics[0].attributes.items()[0][1])#/1000.0
    ty = float(models_extrinsics[0].attributes.items()[1][1])#/1000.0
    tz = float(models_extrinsics[0].attributes.items()[2][1])#/1000.0
    #print(tx, ty, tz)
    #stop
    rx = float(models_extrinsics[0].attributes.items()[3][1])
    ry = float(models_extrinsics[0].attributes.items()[4][1])
    rz = float(models_extrinsics[0].attributes.items()[5][1])
    R_x = np.array([[1,         0,                  0                   ],
                    [0,         math.cos(rx), -math.sin(rx) ],
                    [0,         math.sin(rx), math.cos(rx)  ]
                    ])

    R_y = np.array([[math.cos(ry),    0,      math.sin(ry)  ],
                    [0,                     1,      0                   ],
                    [-math.sin(ry),   0,      math.cos(ry)  ]
                    ])

    R_z = np.array([[math.cos(rz),    -math.sin(rz),    0],
                    [math.sin(rz),    math.cos(rz),     0],
                    [0,                     0,                      1]
                    ])
    R_rot = R_z @ R_y @ R_x
    cam_pos = (np.linalg.inv(R_rot)@(-np.array([tx,ty,tz])))
    gt_rotation_array.append(R_rot)
    gt_translation_array.append(cam_pos)

    gt_intrinsics_array.append(intrinsic_gt["intrinsics"])
    
    '''
    with open('/local/tangytob/ViTPose/hrnet_vitpose/' + vid.split('_')[0] + '_avi_vitpose.json', 'r') as f:
        points_2d = json.load(f)
    #datastore = data.vitpose_hrnet(points_2d, bound_lower = 200, bound = 320)    
    datastore = data.vitpose_hrnet(points_2d, bound_lower = 200, bound = 2500)      
    '''
    
    with open('/local/tangytob/ViTPose/vis_results/res50results/result_' + vid.split('_')[0] + '_.json', 'r') as f:
        points_2d = json.load(f)
    
    datastore = data.coco_mmpose_dataloader(points_2d, bound_lower = 100, bound = 2500) 
    
    #datastore = data.coco_mmpose_dataloader(points_2d, bound_lower = 200, bound = 1000) 
    #print(points_2d)
    
    start = True

    frame_dir = '/local/tangytob/EPFL/videos/Frames/' + vid + '/00000000.jpg'
    img = mpimg.imread(frame_dir)
    
    save_dict = load_dict[num]
    cam_matrix = save_dict["cam_matrix"]
    normal = save_dict["ground_normal"]
    ankleWorld = save_dict["ground_position"]

    save_dict_array.append(save_dict)
    ##################################
    keep_list = [100, 125, 150] 
    data_2d = util.get_ankles_heads_dictionary(datastore, cond_tol=confidence, keep = -1, keep_list = keep_list)
    pose_2d = util.get_ankles_heads_pose_dictionary(datastore, cond_tol=confidence, keep = -1, keep_list = keep_list)

    #print(data_2d)
    #stop
    ankle_head_2d_array.append(data_2d)
    pose_2d_array.append(pose_2d)

    plane_matrix, basis_matrix = geometry.find_plane_matrix(save_dict["ground_normal"], np.linalg.inv(save_dict['cam_matrix']),save_dict['ground_position'], img.shape[1], img.shape[0])
    
    to_pickle_plane_matrix = {"plane_matrix": plane_matrix,'intrinsics': save_dict['cam_matrix']}
    
    plane_data_2d = geometry.camera_to_plane(data_2d, cam_matrix, plane_matrix, save_dict['ground_position'], save_dict["ground_normal"], img.shape[1], img.shape[0])
    #print(plane_data_2d, " PLANE DATA")
    #plane_data_2d = util.remove_outliers(plane_data_2d, eps=1.0)
    #stop
    plane_matrix_array.append(plane_matrix)
    #print(plane_matrix, " PLANE MATRIX!!!")
    
    plane_dict_array.append(plane_data_2d)
    ##################################
    save_dict_array.append(save_dict)

    num = num + 1

plane_dict_array_experiment0 = {}

plane_dict_array_experiment = []

camera_name = campus_array_names

for k in range(len(experiments_time)):
    
    shift_avg_array = []
    shift_diff_array = []
    all_shift_array = []
    all_diff_array = []

    for cam in cam_comb:
        print(cam)
        cam1 = cam[0]
        cam2 = cam[1]
        print(cam1, cam2, " THE CAMS")
        if np.dot(np.array(gt_rotation_array[cam1])[2, :], np.array(gt_rotation_array[cam2])[2, :]) < -0.85:
            print(" FAILED ")
            continue
        print(" MADE IT")

        plane_dict_array0 = plane_dict_array[cam1].copy()
        plane_dict_array1 = plane_dict_array[cam2].copy()
        #################################
        '''
        track_ref = {}
        for f_k in list(plane_dict_array0.keys()):
            for tr in list(plane_dict_array0[f_k].keys()):
                x_coord = plane_dict_array0[f_k][tr][0]
                y_coord = plane_dict_array0[f_k][tr][1]
                #print(x_coord,y_coord, " X COORD Y COORD")
                if tr not in track_ref:
                    track_ref[tr] = {f_k: [x_coord, y_coord]}
                else:
                    track_ref[tr][f_k] = [x_coord, y_coord]

        
        track_sync = {}
        for f_k in list(plane_dict_array1.keys()):
            for tr in list(plane_dict_array1[f_k].keys()):
                x_coord = plane_dict_array1[f_k][tr][0]
                y_coord = plane_dict_array1[f_k][tr][1]

                if tr not in track_sync:
                    track_sync[tr] = {f_k: [x_coord, y_coord]}
                else:
                    track_sync[tr][f_k] = [x_coord, y_coord]
        '''
        #################################
        '''
        plane_dict_array0 = {}
        #print(track_ref)

        for tr in track_ref.keys():

            xy_coord = np.array(list(track_ref[tr].values()))
            x_plot = xy_coord[:, 0]#track_ref[tr].values()[0]
            y_plot = xy_coord[:, 1]#track_ref[tr].values()[1]
            if xy_coord.shape[0] < 5:
                continue
            frame_plot = list(track_ref[tr].keys())
            print(len(frame_plot), x_plot.shape, y_plot.shape, " FRAME X Y")
            print(frame_plot)
            frame_plot, x_plot, y_plot = time_align.track_smooth(frame_plot, x_plot, y_plot)

            for fp in range(len(frame_plot)):

                if frame_plot[fp] not in plane_dict_array0:
                    plane_dict_array0[frame_plot[fp]] = {tr: [x_plot[fp], y_plot[fp]]}
                else:
                    plane_dict_array0[frame_plot[fp]][tr] = [x_plot[fp], y_plot[fp]]
        
        plane_dict_array1 = {}
        for tr in track_sync.keys():

            xy_coord = np.array(list(track_sync[tr].values()))
            x_plot = xy_coord[:, 0]
            y_plot = xy_coord[:, 1]
            if xy_coord.shape[0] < 5:
                continue
            frame_plot = list(track_sync[tr].keys())
            #print(xy_coord.shape, " SHAPE X Y !!! ")
            #print(len(frame_plot), x_plot.shape, y_plot.shape, " FRAME X Y !!! ")
            #print(frame_plot)
            print(frame_plot)
            frame_plot, x_plot, y_plot = time_align.track_smooth(frame_plot, x_plot, y_plot)
            #print(frame_plot)
            #stop
            for fp in range(len(frame_plot)):
                print(frame_plot[fp], tr, " THE FP ")
                if frame_plot[fp] not in plane_dict_array1:
                    plane_dict_array1[frame_plot[fp]] = {tr: [x_plot[fp], y_plot[fp]]}
                else:
                    plane_dict_array1[frame_plot[fp]][tr] = [x_plot[fp], y_plot[fp]]
        '''
        print("**************!!!")
        #print(plane_dict_array1)
        
        #stop
        '''
        plane_dict_array1 = {}
        for tr in track_sync.keys():
            
            x_plot = track_sync[tr].values()[0]
            y_plot = track_sync[tr].values()[1]
            frame_plot = track_sync[tr].keys()

            frame_plot, x_plot, y_plot = time_align.track_smooth(frame_plot, x_plot, y_plot)

            for fp in range(len(frame_plot)):
                plane_dict_array1[fp] = [x_plot[fp], y_plot[fp]]
        '''
        '''
        for f_k in list(plane_dict_array0.keys()):
            x_plot.append(list(plane_dict_array0[f_k].values())[0][0])
            y_plot.append(list(plane_dict_array0[f_k].values())[0][1])
            frame_plot.append(f_k)
        #stop
        plane_dict_array0 = {}
        frame_plot, x_plot, y_plot = time_align.track_smooth(frame_plot, x_plot, y_plot)

        for f_k in range(len(frame_plot)):
            plane_dict_array0[frame_plot[f_k]] = {0: [x_plot[f_k], y_plot[f_k]]}
        
        x_plot = []
        y_plot = []
        frame_plot = []

        for f_k in list(plane_dict_array1.keys()):
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

        set1 = list(set(track1))
        set2 = list(set(track2))

        #intersection = sorted(list(set1.intersection(set2)))
        #print(intersection)
        #start = intersection[0]
        #end = intersection[-1]
        #missing_numbers = [x for x in range(start, end + 1) if x not in intersection]
        #print(set1, " SET1")
        #print(set2, " SET2")
        #stop
        # Print the missing numbers
        #print(missing_numbers)
        
        shift_avg_array = []
        diff_avg_array = []
        true_offset_array = []
        #print(" intersection ")
        #print(intersection)

        for exp in range(0, 1):

            #test_size = int(len(intersection))
            subsect, subset_start = util.get_random_windowed_subset(set1, len(set1), offset = experiments_time[k], start = 0)
            print(subsect, " subsect !!!")
            subsect1, subset_start1 = util.get_random_windowed_subset(set2, len(set2), start = experiments_time[k])
            
            print(subsect1, " subsect1 !!!")
            
            print(subset_start, " subsect")
            print(subset_start1, " subsect1")
            #stop
            true_offset = abs(subset_start - subset_start1)

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
            '''
            plane_dict_array_spline0 = plane_dict_array_subset0#time_align.dict_smooth(plane_dict_array_subset0, min_size = 10)
            #############################################
            '''
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
            '''
            plane_dict_array_spline1 = plane_dict_array_subset1#time_align.dict_smooth(plane_dict_array_subset1, min_size = 10)
            #############################################          
            
            plane_dict_array_no_spline0 = util.subtract_keys(plane_dict_array_subset0)
            plane_dict_array_no_spline1 = util.subtract_keys(plane_dict_array_subset1)

            plane_dict_array_experiment0 = util.subtract_keys(plane_dict_array_spline0)
            plane_dict_array_experiment1 = util.subtract_keys(plane_dict_array_spline1)

            pose_dict_array_experiment0 = util.subtract_keys(pose_dict_array_subset0)
            pose_dict_array_experiment1 = util.subtract_keys(pose_dict_array_subset1)
            
            '''
            plane_dict_array_no_spline0 = plane_dict_array_subset0
            plane_dict_array_no_spline1 = plane_dict_array_subset1

            plane_dict_array_experiment0 = plane_dict_array_spline0
            plane_dict_array_experiment1 = plane_dict_array_spline1

            pose_dict_array_experiment0 = pose_dict_array_subset0
            pose_dict_array_experiment1 = pose_dict_array_subset1
            '''
            #for mode in [True, False]:
            #for mode in [False]:
            for mode in [True, False]:

            #for mode in [True]:
                
                mode_string = "no_sync"
                if mode is True:
                    mode_string = "sync"
                name_folder = '/' + str(camera_name[cam1]) + '_' + str(camera_name[cam2]) + '_' + str(experiments_time[k]) + '_' + str(exp) + '_' + mode_string

                save_dir = './plots/time_' + name + name_folder

                if os.path.isdir('./plots/time_' + name + name_folder) == False:
                    os.mkdir('./plots/time_' + name + name_folder)

                if os.path.isdir('./plots/time_' + name + name_folder + '/search_time') == False:
                    os.mkdir('./plots/time_' + name + name_folder + '/search_time')

                if os.path.isdir('./plots/time_' + name + name_folder + '/search_rot') == False:
                    os.mkdir('./plots/time_' + name + name_folder + '/search_rot')

                if os.path.isdir('./plots/time_' + name + name_folder + '/ICP') == False:
                    os.mkdir('./plots/time_' + name + name_folder + '/ICP')

                output_folder = 'campus'
                #PLOTS SAVED IN PLOTS NOW!!!
                
                if os.path.isdir('./plots/time_' + name + name_folder + '/' + output_folder) == False:
                    os.mkdir('./plots/time_' + name + name_folder + '/' + output_folder)

                frame_array = []
                output_path = './plots/time_' + name + name_folder + '/' + output_folder + '/'


                cam_axis4 = []
                cam_position4 = []
                
                
                #print(plane_dict_array_experiment0, " HIIIIII !")
                #print(plane_dict_array_experiment, " HIIIIII !!")
                #stop
            
                print(subsect1)
                print(" please work ")            
                print(subsect)

                ################################
                '''
                x_plot = []
                y_plot = []
                frame_plot = []

                for f_k in list(plane_dict_array_subset0.keys()):
                    print(list(plane_dict_array_subset0[f_k].values())[0][0], list(plane_dict_array_subset0[f_k].values())[0][1])
                    x_plot.append(list(plane_dict_array_subset0[f_k].values())[0][0])
                    y_plot.append(list(plane_dict_array_subset0[f_k].values())[0][1])
                    frame_plot.append(f_k)
                #stop
                df_array = pd.DataFrame(columns=['x', 'y','frame'])
                df_array.x = x_plot
                df_array.y = y_plot
                df_array.frame = frame_plot 

                plotting.plot_slide1(df_array, save_dir, name + '_' + str(camera_name[cam1]))

                #############################################

                x_plot = []
                y_plot = []
                frame_plot = []

                for f_k in list(plane_dict_array_subset1.keys()):

                    x_plot.append(list(plane_dict_array_subset1[f_k].values())[0][0])
                    y_plot.append(list(plane_dict_array_subset1[f_k].values())[0][1])
                    frame_plot.append(f_k)

                df_array = pd.DataFrame(columns=['x', 'y','frame'])
                df_array.x = x_plot
                df_array.y = y_plot
                df_array.frame = frame_plot 

                plotting.plot_slide1(df_array, save_dir, name + '_' + str(camera_name[cam2]))
                '''
                ################################

                
                #best_shift_array, best_scale_array, sync_dict_array = ICP.time_all(plane_dict_array_experiment0, [plane_dict_array_experiment1], save_dir = './plots/time_' + name + name_folder, sync = True, name = str(exp) + '_' + str(experiments_time[k]) + '_' + str(camera_name[cam1]) + '_' + str(camera_name[cam2]))
                #best_shift_array, best_scale_array, sync_dict_array = time_align_multi_person.time_all(plane_dict_array_experiment0, [plane_dict_array_experiment1], save_dir = './plots/time_' + name + name_folder, sync = mode, name = str(exp) + '_' + str(camera_name[cam1]) + '_' + str(camera_name[cam2]), window = 1, dilation = 1)
                
                #sync_dict_array = [time_align_multi_person.time_knn(1.0, best_shift_array[0], plane_dict_array_no_spline1, plane_dict_array_no_spline0)]
                
                best_shift_array = [0]
                best_scale_array = [1]#list(np.zeros(len(best_shift_array))) 
                        
                if mode == True:
                    best_shift_array = [time_pred[str(experiments_time[k]) + '_' + str(campus_array_names[cam1]) + '_' + str(campus_array_names[cam2])]]


                sync_dict_array = [time_align_inlier.time_knn(1.0, best_shift_array[0], list(pose_dict_array_experiment1.keys()), list(pose_dict_array_experiment0.keys()))]


                shift_avg_array.append(np.absolute(best_shift_array[0]))
                shift_diff_array.append(np.absolute(best_shift_array[0] - experiments_time[k]))

                all_shift_array.append(np.absolute(best_shift_array[0]))

                all_diff_array.append(np.absolute(best_shift_array[0] - experiments_time[k]))
                
                with open('./plots/time_' + name + '/result_sync.csv','a') as file:
                
                    writer1 = csv.writer(file)

                    writer1.writerow([str(experiments_time[k]), str(best_shift_array[0]), str(min(subsect1)), str(camera_name[cam1]), str(camera_name[cam2])])
                    file.close

                #print(" PLEASE WORK !!! ")
                #print(plane_dict_array0, exp," plane_dict_array0 AFTER !")
                
                #print(best_shift_array, best_scale_array, sync_dict_array)

                #exp_average.append(np.absolute(best_shift_array[0]))
                '''
                with open('./plots/time_' + name + '/result_sync.csv','a') as file:
            
                    writer1 = csv.writer(file)

                    writer1.writerow([str(exp), str(best_shift_array[0]), str(true_offset), str(abs(best_shift_array[0] - true_offset)), str(min(subsect)), str(camera_name[cam1]), str(camera_name[cam2])])
                    file.close
                '''
                print(sync_dict_array)
                if mode == True:
                    shift_avg_array.append(np.absolute(best_shift_array[0]))
                    diff_avg_array.append(abs(best_shift_array[0] - true_offset))
                    true_offset_array.append(true_offset)

                    #exp_gt_average.append(true_offset)
                    #exp_shift_average.append(best_shift_array[0])
                    #exp_diff_average.append(abs(best_shift_array[0] - true_offset))
                #print(index_array)
                #print("whatsup !!! ")
                icp_rot_array, icp_init_rot_array, icp_shift_array, init_ref_center_array, init_sync_center_array, time_shift_array, time_scale_array, sync_dict_array, index_array = ICP.icp(plane_dict_array_no_spline0, [plane_dict_array_no_spline1], best_shift_array, best_scale_array, sync_dict_array, save_dir = './plots/time_' + name + name_folder, name = '_')
                
                pre_bundle_cal_array = []
                single_view_cal_array = []
                cam_intrinsics_array = []
                pose_2d_array_comb = []
                #pose_2d_array_experiment = [pose_dict_array_experiment0, pose_dict_array_experiment1]
                pose_2d_array_experiment = [pose_2d_array[cam1], pose_2d_array[cam2]]

                #index_dict = time_align_multi_person.time_knn_array(1.0, best_shift_array[0] - offset, np.array(list(subsect)))
                #print("***************************************!!!111111")
                #print(index_dict)

                #indices_array = [list(index_dict.keys()), list(index_dict.values())]
                #print(sync_dict_array[0])
                #print(index_array)
                #print("whatsup")
                #stop
                #indices_array = index_array
                indices_array = [list(sync_dict_array[0].keys()), list(sync_dict_array[0].values())]
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
                            #print(pose_2d_array_experiment[i][ia]," HIII")
                            filtered_frame = {}
                            for pk in list(pose_2d_array_experiment[i][ia].keys()):
                                conf_all = []
                                #print(list(pose_2d_array_experiment[i][ia].values()), " THE AVLULES")
                                
                                pose_val_dict = pose_2d_array_experiment[i][ia][pk][4]
                                pose_all_kp = list(pose_val_dict.keys())
                                for kp_ in pose_all_kp:
                                    #print(kp_)
                                    if kp_ != "id":
                                        conf_all.append(pose_val_dict[kp_][2])
                                conf_all_mean = np.mean(conf_all)
                                #print(conf_all_mean, " conf")
                                if conf_all_mean > 0.81:
                                    filtered_frame[pk] = pose_2d_array_experiment[i][ia][pk]
                                
                                #print(pose_2d_array_experiment[i][ia].keys(), ia, " EXPERIEMTNT")    
                            
                            if len(filtered_frame) == 0:
                                continue
                            pose_comb[ia] = filtered_frame#pose_2d_array_experiment[i][ia]
                    #stop
                    pose_2d_array_comb.append(pose_comb)
                    #pose_2d_array_comb.append(pose_2d_array[cam[i]][indices_array[i]])
                print(pose_2d_array_comb[0].keys())
                print("************")
                print(pose_2d_array_comb[1].keys())
                
                matched_points = bundle_adjustment_plotly_fix.match_3d_plotly_input2d_farthest_point(pre_bundle_cal_array, single_view_cal_array, pose_2d_array_comb,center_array, output_path, scale = scale, title = 'rotation perturb TRACKS', name = ['perturb'], k = 20)

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
                
                #plotting.plot_camera_pose([cam_axis4, gt_rotation_array_comb], [cam_position4, gt_translation_array_comb], save_dir, scale = 1, name = "pre_bundle_no_align")

                plotting.plot_camera_pose([cam_axis_rotate, gt_rotation_array_comb], [cam_position_rotate, gt_translation_array_comb], save_dir, scale = 1, name = "pre_bundle")
                plotting.plot_camera_pose([bundle_cam_axis_rotate, gt_rotation_array_comb], [bundle_cam_position_rotate, gt_translation_array_comb], save_dir, scale = 1, name = "pose_bundle")
                
                results_focal_pred1, results_focal_tsai1, angle_diff1, error_npjpe1, focal_error1, results_position_diff1 = eval_functions.evaluate(cam_intrinsics_array, cam_axis_rotate, [cam_position_rotate], gt_intrinsics_array_comb, gt_rotation_array_comb, [gt_translation_array_comb], output_path, 'pre_bundle')
                results_focal_pred2, results_focal_tsai2, angle_diff2, error_npjpe2, focal_error2, results_position_diff2 = eval_functions.evaluate(bundle_intrinsic_matrix_array, bundle_cam_axis_rotate, [bundle_cam_position_rotate], gt_intrinsics_array_comb, gt_rotation_array_comb, [gt_translation_array_comb], output_path, 'result_bundle')
                '''
                with open('./plots/time_' + name + '/result_bundle.csv','a') as file:

                    writer1 = csv.writer(file)
                    writer1.writerow([str(camera_name[cam1]), str(camera_name[cam2]), str(true_offset), str(best_shift_array[0]), str(abs(best_shift_array[0] - true_offset)), str(exp), mode_string, str(results_focal_pred1), str(results_focal_tsai1), str(angle_diff1), str(error_npjpe1), str(focal_error1), str(results_position_diff1), str(results_focal_pred2), str(results_focal_tsai2), str(angle_diff2), str(error_npjpe2), str(focal_error2), str(results_position_diff2)])
                    file.close
                '''
                if mode == True:
                    with open('./plots/time_' + name + '/result_bundle_sync.csv','a') as file:

                        writer1 = csv.writer(file)
                        writer1.writerow([str(camera_name[cam1]), str(camera_name[cam2]), str(true_offset), str(best_shift_array[0]), str(abs(best_shift_array[0] - true_offset)), str(exp), str(results_focal_pred1), str(results_focal_tsai1), str(angle_diff1), str(error_npjpe1), str(focal_error1), str(results_position_diff1), str(results_focal_pred2), str(results_focal_tsai2), str(angle_diff2), str(error_npjpe2), str(focal_error2), str(results_position_diff2)])
                        file.close
                else:
                    with open('./plots/time_' + name + '/result_bundle_no_sync.csv','a') as file:

                        writer1 = csv.writer(file)
                        writer1.writerow([str(camera_name[cam1]), str(camera_name[cam2]), str(true_offset), str(best_shift_array[0]), str(abs(best_shift_array[0] - true_offset)), str(exp), str(results_focal_pred1), str(results_focal_tsai1), str(angle_diff1), str(error_npjpe1), str(focal_error1), str(results_position_diff1), str(results_focal_pred2), str(results_focal_tsai2), str(angle_diff2), str(error_npjpe2), str(focal_error2), str(results_position_diff2)])
                        file.close

    
            with open('./plots/time_' + name + '/result_average_sync.csv','a') as file:

                writer1 = csv.writer(file)
                writer1.writerow([str(experiments_time[k]), str(camera_name[cam1]), str(camera_name[cam2]), str(np.mean(shift_avg_array)), str(np.std(shift_diff_array))])
                file.close

    with open('./plots/time_' + name + '/result_average_all.csv','a') as file:

        writer1 = csv.writer(file)
        writer1.writerow([str(experiments_time[k]), str(np.mean(all_shift_array)), str(np.std(all_shift_array)), str(np.mean(all_diff_array)), str(np.std(all_diff_array))])
        file.close