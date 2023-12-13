import cv2
import numpy as np
import matplotlib.pyplot as plt
import plotting
import util
from scipy.spatial.distance import directed_hausdorff
import torch
import os
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import math
import knn
import random

from scipy.spatial.distance import cdist
#from simpleicp import PointCloud, SimpleICP
import Icp2d
import multiview_utils

def icp(data_ref, data_sync, best_shift_array, best_scale_array, sync_dict_array, save_dir = None, name = '_cam_'):

    icp_rot_array = []
    icp_init_rot_array = []
    icp_shift_array = []
    init_ref_center_array = []
    init_sync_center_array = []

    time_shift_array = []
    time_scale_array = []
    #sync_dict_array = []

    index_array = []

    for i in range(len(data_sync)):

        best_shift = best_shift_array[i]
        best_scale = best_scale_array[i]
        sync_dict = sync_dict_array[i]

        init_d, init_ref_center, init_sync_center, reflect_true = rot_shift_search_time_weight(data_ref, data_sync[i], sync_dict, best_scale, best_shift, save_dir = save_dir, name = str(i) + name)
        icp_rot, icp_shift, icp_init_rot, index = Icp2d.icp_normalize(data_ref, data_sync[i], sync_dict, init_d, init_ref_center, init_sync_center, reflect_true, save_dir = save_dir, name = str(i), best_scale = best_scale)

        icp_rot_array.append(icp_rot)
        icp_init_rot_array.append(icp_init_rot)
        icp_shift_array.append(icp_shift)
        init_ref_center_array.append(init_ref_center)
        init_sync_center_array.append(init_sync_center)

        time_shift_array.append(best_shift)
        time_scale_array.append(best_scale)
        sync_dict_array.append(sync_dict)

    return icp_rot_array, icp_init_rot_array, icp_shift_array, init_ref_center_array, init_sync_center_array, time_shift_array, time_scale_array, sync_dict_array, index_array

def rot_shift_search_time_weight(data_ref, data_sync, sync_time_dict, scale, offset, save_dir = None, name = None):
    
    if os.path.isdir(save_dir + '/search_rot') == False:
        os.mkdir(save_dir + '/search_rot')

    if os.path.isdir(save_dir + '/search_rot/' + name) == False:
        os.mkdir(save_dir + '/search_rot/' + name)

    ref_coords = []
    ref_time = []
    ref_cor_time = []

    ref_dict = {}
    #print(sync_time_dict, " sync dict")
    #print("**********************************************")
    for fr in list(sync_time_dict.keys()):

        ref_dict[fr] = []
        for tr in list(data_ref[fr].keys()):
            
            ref_coords.append(data_ref[fr][tr][0:2])
            ref_time.append(fr)
            ref_cor_time.append(scale*sync_time_dict[fr] + offset)

            ref_dict[fr].append(data_ref[fr][tr][0:2])
    #print(scale, offset, " scale and offsetttt")
    sync_coords = []
    sync_time = []
    sync_cor_time = []

    sync_dict = {}
    
    for item in sync_time_dict.items():

        sync_dict[item[1]] = []
        for tr in list(data_sync[item[1]].keys()):
            
            sync_coords.append(data_sync[item[1]][tr][0:2])
            sync_time.append(item[1])
            sync_cor_time.append(item[0])
            sync_dict[item[1]].append(data_sync[item[1]][tr][0:2])

    #print(sync_cor_time)
    
    ref_center = np.mean(ref_coords, axis = 0)
    sync_center = np.mean(sync_coords, axis = 0)

    best_d = 0
    best_error = np.inf

    best_x = 1
    best_y = 1

    all_angles = []
    all_error = []
    reflect_true = False

    #################################################
    normalize_ref = np.array(ref_coords) - ref_center
    normalize_rot = np.array(sync_coords) - sync_center#ref_center

    ref_x_size = 1.0#max([abs(normalize_ref_x_max), abs(normalize_ref_x_min)])
    ref_y_size = 1.0#max([abs(normalize_ref_y_max), abs(normalize_ref_y_min)])

    rot_x_size = 1.0#max([abs(normalize_rot_x_max), abs(normalize_rot_x_min)])
    rot_y_size = 1.0#max([abs(normalize_rot_y_max), abs(normalize_rot_y_min)])

    ref_normalize = np.transpose(np.stack([normalize_ref[:, 0]/ref_x_size, normalize_ref[:, 1]/ref_y_size]))
    sync_normalize = np.transpose(np.stack([normalize_rot[:, 0]/rot_x_size, normalize_rot[:, 1]/rot_y_size]))

    sync_space_time = []
    ref_space_time = []

    sync_time = []
    ref_time = []
    
    for i in sync_time_dict.keys():
    
        sync_time.append((i*np.ones(np.array(sync_dict[sync_time_dict[i]])[:, 1].shape)))
        ref_time.append((i*np.ones(np.array(ref_dict[i])[:, 1].shape)))
        
        sync_normalize1 = np.transpose(np.stack([(np.array(sync_dict[sync_time_dict[i]])[:, 0] - sync_center[0])/rot_x_size, (np.array(sync_dict[sync_time_dict[i]])[:, 1] - sync_center[1])/rot_y_size]))
        ref_normalize1 = np.transpose(np.stack([(np.array(ref_dict[i])[:, 0] - ref_center[0])/ref_x_size, (np.array(ref_dict[i])[:, 1] - ref_center[1])/ref_y_size]))

        sync_space_time.append(sync_normalize1)
        ref_space_time.append(ref_normalize1)

    sync_normalize1 = np.concatenate(sync_space_time)
    ref_normalize1 = np.concatenate(ref_space_time)

    sync_time = np.concatenate(sync_time)
    ref_time = np.concatenate(ref_time)

    for reflect in [False]:
        for d in range(50,300):
            rot_coords1, R, o, p, s = rotate(sync_normalize1, origin=[0,0], shift=[0,0], degrees=d, reflect = reflect)
            ref_normalize1_time = np.transpose(np.stack([ref_normalize1[:, 0], ref_normalize1[:, 1], ref_time]))
            rot_coords1_time = np.transpose(np.stack([rot_coords1[:, 0], rot_coords1[:, 1], sync_time]))
            error = multiview_utils.chamfer_distance(ref_normalize1_time, rot_coords1_time, metric='l2', direction='bi')
            
            if best_error > error:
                best_d = d
                best_error = error
                reflect_true = reflect

            
            all_angles.append(d)
            all_error.append(error)

    rot_normalize, R, o, p, s = rotate(sync_normalize1, origin=[0,0], shift=[0,0], degrees=best_d, reflect = reflect_true, scale_x = best_x, scale_y = best_y)

    error = best_error
    if  save_dir is not None:
        fig, ax1 = plt.subplots(1, 1)
        ax1.set_yscale("linear")
        ax1.scatter(all_angles, all_error)

        if name is not None:
            fig.savefig(save_dir + '/search_rot/' + name + '/error_' + str(name) + '_' + str(best_d) + '_degree_' + '.png')
        else:
            fig.savefig(save_dir + '/search_rot/' + name + '/' + 'error_' + str(best_d) + '_degree_' + '.png')
    plt.close('all')

    if  save_dir is not None:
        fig, ax1 = plt.subplots(1, 1)
        ax1.scatter(np.array(rot_normalize)[:, 0], np.array(rot_normalize)[:, 1], c = 'r')
        ax1.scatter(np.array(ref_normalize)[:, 0], np.array(ref_normalize)[:, 1], c = 'b')
        ax1.set_title(str(error))
        fig1, ax2 = plt.subplots(1, 1)
        ax2.scatter(np.array(sync_normalize)[:, 0], np.array(sync_normalize)[:, 1], c = 'r')
        ax2.scatter(np.array(ref_normalize)[:, 0], np.array(ref_normalize)[:, 1], c = 'b')
        ax2.set_title("init")

        if name is not None:
            fig.savefig(save_dir + '/search_rot/' + name + '/best_' + str(name) + '_' + str(best_d) + '_degree_' + '.png')
        else:
            fig.savefig(save_dir + '/search_rot/' + name + '/' + 'best_' + str(best_d) + '_degree_' + '.png')

        fig1.savefig(save_dir + '/search_rot/' + name + '/init.png')
    plt.close('all')
    
    return math.radians(best_d), ref_center, sync_center, reflect_true

def rotate(p, origin=(0, 0), shift=(0,0), degrees=0.0, reflect = False, scale_x = 1.0, scale_y = 1.0):
    angle = np.deg2rad(degrees)

    R = np.array([[scale_x*np.cos(angle), -np.sin(angle)],
                  [np.sin(angle),  scale_y*np.cos(angle)]])

    
    if reflect == True:
        R = np.array([[scale_x*np.cos(angle), -np.sin(angle)],
                    [np.sin(angle),  -1*scale_y*np.cos(angle)]])

    o = np.atleast_2d(origin)
    p = np.atleast_2d(p)

    s = np.atleast_2d(shift)

    #print(p.shape, o.shape, " HELLO")
    if p.shape[0] > 1:
        return np.squeeze((R @ (p.T-o.T) + s.T).T), R, o, p, s
    else:
        return (R @ (p.T-o.T) + s.T).T, R, o, p, s