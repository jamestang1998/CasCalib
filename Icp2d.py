import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
import random
import matplotlib.image as mpimg
import util
import os
import geometry
import ICP
import torch
import math
import multiview_utils

def icp_normalize(data_ref, data_sync, sync_time_dict, init_angle, init_ref_center, init_sync_center, reflect_true, n_iter = 10, save_dir = None, name = None, best_scale = 1.0):
    
    if os.path.isdir(save_dir + '/ICP') == False:
        os.mkdir(save_dir + '/ICP')

    if os.path.isdir(save_dir + '/ICP/' + name) == False:
        os.mkdir(save_dir + '/ICP/' + name)
    
    dilation = 15
    window = 1
    thresh_mag = 0.0
    data_interp_ref = interpolate(data_ref, 2*dilation)
    data_interp_sync = interpolate(data_sync, 2*dilation)

    ref_vel_array = []
    ref_vel_dict = {} 

    #print(data_interp_ref.keys())
    ref_dict = {}
    for rk in list(sync_time_dict.keys()):
        ref_dict[rk] = []

    sync_dict = {}
    for sk in list(sync_time_dict.values()):
        sync_dict[sk] = []

    for track in data_interp_ref.keys():
        
        ref_time = data_interp_ref[track]['frame']
        ref_points = data_interp_ref[track]['points']
        
        if len(ref_points) < 2*dilation + 1:
            continue

        B_set = set(data_interp_ref[track]['frame_actual'])
        indices = [i for i, x in enumerate(ref_time) if x in B_set]

        ref_vel = torch.squeeze(multiview_utils.central_diff(torch.unsqueeze(torch.transpose(torch.tensor(ref_points), 0 , 1), dim = 1).double(), time = 1, dilation = dilation))
        
        ref_vel_array.append(ref_vel[:, indices])

        for ind in range(len(indices)):

            if ref_time[indices[ind]] not in list(sync_time_dict.keys()):
                continue 

            ref_dict[ref_time[indices[ind]]].append({track: ref_vel[:, ind]})

    sync_vel_dict = {} 

    ###################################
    sync_vel_array = []
    for track in data_interp_sync.keys():
        
        sync_frame = data_interp_sync[track]['frame']
        sync_points = data_interp_sync[track]['points']
        sync_time = sync_frame

        if len(sync_points) < 2*dilation + 1:
            continue

        B_set = set(data_interp_sync[track]['frame_actual'])
        indices = [i for i, x in enumerate(sync_time) if x in B_set]
        
        sync_vel = torch.squeeze(multiview_utils.central_diff(torch.unsqueeze(torch.transpose(torch.tensor(sync_points), 0 , 1), dim = 1).double(), time = best_scale, dilation = dilation))
        sync_vel_array.append(sync_vel[:, indices])

        for ind in range(len(indices)):
            if sync_time[indices[ind]] not in list(sync_dict.keys()):
                continue 

            sync_dict[sync_time[indices[ind]]].append({track: sync_vel[:, ind]})
    
    for k in list(ref_dict.keys()):
        myKeys = ref_dict[k]
        if len(myKeys) == 0:
            ref_dict.pop(k)
            continue
        myKeys_0 = list(myKeys[0].keys())
        myKeys_0.sort()
        ref_dict[k] = {i: ref_dict[k][0][i][:2].numpy() for i in myKeys_0}
    
    for k in list(sync_dict.keys()):
    
        myKeys = sync_dict[k]
        if len(myKeys) == 0:
            sync_dict.pop(k)
            continue
        myKeys_0 = list(myKeys[0].keys())
        myKeys_0.sort()
        sync_dict[k] = {i: sync_dict[k][0][i][:2].numpy() for i in myKeys_0}
    
    #############################################################
    ####################################

    frame_sync = list(sync_time_dict.values())
    frame_ref = list(sync_time_dict.keys())

    reflect_coef = 1

    if reflect_true:
        reflect_coef = -1
    
    R_init = np.array([[np.cos(init_angle), -np.sin(init_angle)],
                  [np.sin(init_angle),  reflect_coef*np.cos(init_angle)]])

    best_init_sync_shift = init_sync_center
    #best_
    #top_inlier = 0
    best_rot = np.array([[1,0], [0,1]])
    best_shift = [0,0]#init_ref_center

    ransac_rot = None
    ransac_shift = None

    ref_coords = None
    sync_coords = None

    top_inlier = 0
    sync_untransformed_array = []
    
    #################################################
    ref_coords = []
    for fr in list(sync_time_dict.keys()):

        for tr in list(data_ref[fr].keys()):
            
            ref_coords.append(data_ref[fr][tr][0:2])
    
    sync_coords = []
    
    for item in sync_time_dict.items():
        for tr in list(data_sync[item[1]].keys()):
            
            sync_coords.append(data_sync[item[1]][tr][0:2])
    #################################################
    ref_center = np.mean(ref_coords, axis = 0)
    sync_center = np.mean(sync_coords, axis = 0)

    ref_x_size = 1.0#max([abs(normalize_ref_x_max), abs(normalize_ref_x_min)]) 
    ref_y_size = 1.0#max([abs(normalize_ref_y_max), abs(normalize_ref_y_min)])

    rot_x_size = 1.0#max([abs(normalize_rot_x_max), abs(normalize_rot_x_min)])
    rot_y_size = 1.0#max([abs(normalize_rot_y_max), abs(normalize_rot_y_min)])
    
    index_array = []
    for itr in range(n_iter):

        X_fix = []
        X_mov = []

        X_fix_vel = []
        X_mov_vel = []

        index_dict = {}

        for fr in frame_ref:
            
            if fr not in list(ref_dict.keys()) or sync_time_dict[fr] not in list(sync_dict.keys()):
                print(fr, " FAILURE !!!")
                continue

            fix = np.array(list(data_ref[fr].values()))[:, :2]
            mov = np.array(list(data_sync[sync_time_dict[fr]].values()))
            #print(ref_dict[fr], " ref dict")
            fix_vel = np.array(list(ref_dict[fr].values()))[:, :2]
            mov_vel = np.array(list(sync_dict[sync_time_dict[fr]].values()))

            X_fix_vel.append(fix_vel)

            X_mov_vel.append(np.transpose(best_rot @ (R_init @ np.transpose(np.array(mov_vel)[:, :2]))) + best_shift)

            ref_normalize = np.transpose(np.stack([(fix[:, 0] - ref_center[0])/ref_x_size, (fix[:, 1] - ref_center[1])/ref_y_size]))
            sync_normalize = np.transpose(np.stack([(mov[:, 0] - sync_center[0])/rot_x_size, (mov[:, 1] - sync_center[1])/rot_y_size]))
            
            mov_transformed = np.transpose(best_rot @ (R_init @ np.transpose(np.array(sync_normalize)[:, :2]))) + best_shift

            row_ind, col_ind = multiview_utils.hungarian_assignment_one2one(ref_normalize, mov_transformed)
            
            X_mov.append(np.array(mov_transformed)[col_ind, :])
            X_fix.append(np.array(ref_normalize)[row_ind, :])

            if itr == 0:
                sync_untransformed_array.append(ref_normalize)
        
        print(np.concatenate(X_fix, axis = 0).shape, " SHAPE OF X FIX")
        index_array.append(index_dict)
        
        if np.concatenate(X_fix, axis = 0).shape[0] > 1:
            X_fix = np.squeeze(np.concatenate(X_fix, axis = 0))[:,:2]
        else:
            X_fix = np.concatenate(X_fix, axis = 0)[:,:2]
        
        if np.concatenate(X_mov, axis = 0).shape[0] > 1:
            X_mov = np.squeeze(np.concatenate(X_mov, axis = 0))[:,:2]
        else:
            X_mov = np.concatenate(X_mov, axis = 0)[:,:2]

        if itr == 0:
            if np.concatenate(sync_untransformed_array, axis = 0).shape[0] > 1:
                sync_untransformed_array = np.squeeze(np.concatenate(sync_untransformed_array, axis = 0))[:,:2]
            else:
                sync_untransformed_array = np.concatenate(sync_untransformed_array, axis = 0)[:,:2]

        R, t = geometry.rigid_transform_3D(np.transpose(X_mov), np.transpose(X_fix))

        if math.isnan(R[0][0]):
            print(" ANGLE IS NAN")
            continue

        if R is None:
            print(" R IS NONE")
            continue

        if len(list(R)) == 0:
            print(" R length is 0")
            continue

        best_rot = R @ best_rot
        best_shift = t + best_shift

        if  save_dir is not None:
            fig, ax1 = plt.subplots(1, 1)
            ax1.set_yscale("linear")
            ax1.scatter(np.array(X_fix)[::1, 0], np.array(X_fix)[::1, 1], c = 'b')
            ax1.scatter(np.array(X_mov)[::1, 0], np.array(X_mov)[::1, 1], c = 'r')
            ax1.set_title(str(top_inlier))
            ax1.axis('square')

            if name is not None:
                fig.savefig(save_dir + '/ICP/' + name + '/best_' + str(itr) + '_' + str(top_inlier) + '_' + '.png')
            else:
                fig.savefig(save_dir + '/ICP/' + name + '/' + 'best_' + str(itr) + '_' + str(top_inlier) + '_' + str(itr) + '_' + '.png')
            
            if itr == 0:
                fig, ax2 = plt.subplots(1, 1)
                ax2.set_yscale("linear")
                ax2.scatter(np.array(X_fix)[::1, 0], np.array(X_fix)[::1, 1], c = 'b')
                ax2.scatter(np.array(X_mov)[::1, 0], np.array(X_mov)[::1, 1], c = 'r')
                ax2.set_title(str(top_inlier))
                ax2.axis('square')

                if name is not None:
                    fig.savefig(save_dir + '/ICP/' + name + '/init_' + str(itr) + '_' + str(top_inlier) + '_' + '.png')
                else:
                    fig.savefig(save_dir + '/ICP/' + name + '/' + 'init_' + str(itr) + '_' + str(top_inlier) + '_' + str(itr) + '_' + '.png')
            plt.close('all')
    
    return best_rot, [rot_x_size*best_shift[0], rot_y_size*best_shift[1]], R_init, index_array[-1]

def interpolate(data, min_size = 2):
    
    track_dict = {}
    for fr in list(data.keys()):
        for tr in list(data[fr].keys()):

            if tr in track_dict: 
                track_dict[tr].append({'frame': fr, 'coord': data[fr][tr][0:2]})
            else:
                track_dict[tr] = [{'frame': fr, 'coord': data[fr][tr][0:2]}]

    points_dict = {}
    for t in track_dict.keys():
        
        points_array = []
        frame_array = []
        frame_actual_array = []
        points_actual_array = []
        if len(track_dict[t]) < min_size:
            #print("is it here?")
            continue
            
        for i in range(len(track_dict[t])):

            if i == len(track_dict[t]) - 1:
                points_array.append(track_dict[t][i]['coord'])
                frame_array.append(track_dict[t][i]['frame'])

                frame_actual_array.append(track_dict[t][i]['frame'])
                points_actual_array.append(track_dict[t][i]['coord'])

            elif track_dict[t][i + 1]['frame'] - track_dict[t][i]['frame'] == 1:

                points_array.append(track_dict[t][i]['coord'])
                #points_array.append(track_dict[t][i + 1]['coord'])

                frame_array.append(track_dict[t][i]['frame'])
                #frame_array.append(track_dict[t][i + 1]['frame'])
                frame_actual_array.append(track_dict[t][i]['frame'])
                points_actual_array.append(track_dict[t][i]['coord'])

            else:

                grid = np.arange(track_dict[t][i]['frame'], track_dict[t][i + 1]['frame'])


                points = np.transpose(np.array([track_dict[t][i]['coord'], track_dict[t][i + 1]['coord']]))
                time = [track_dict[t][i]['frame'], track_dict[t][i + 1]['frame']]
                y_new = multiview_utils.interpolate(points, time, grid)

                for intp in range(y_new.shape[1]):
                    points_array.append(y_new[:, intp])
                    frame_array.append(grid[intp])
                
                frame_actual_array.append(track_dict[t][i]['frame'])
                points_actual_array.append(track_dict[t][i]['coord'])
        
        points_dict[t] = {'points': points_array, "frame": frame_array, 'points_actual': points_actual_array, 'frame_actual': frame_actual_array}

    return points_dict

def get_track(data, min_size = 2):
    
    print(data)

    mean_array = []
    track_dict = {}
    for fr in list(data.keys()):
        print(data[fr], " THE DATA")
        for tr in list(data[fr].keys()):
            
            mean_array.append(data[fr][tr][0:2])
            if tr in track_dict: 
                track_dict[tr].append({'frame': fr, 'coord': data[fr][tr][0:2]})
            else:
                track_dict[tr] = [{'frame': fr, 'coord': data[fr][tr][0:2]}]

    mean_center = np.mean(mean_array, axis = 0)
    points_dict = {}
    for t in track_dict.keys():
        
        points_array = []
        frame_array = []
        frame_actual_array = []
        points_actual_array = []
            
        for i in range(len(track_dict[t])):

            points_array.append(track_dict[t][i]['coord'] - mean_center)
            frame_array.append(track_dict[t][i]['frame'])

            points_actual_array.append(track_dict[t][i]['coord'] - mean_center)
            frame_actual_array.append(track_dict[t][i]['frame'])
        
        points_dict[t] = {'points': points_array, "frame": frame_array, 'points_actual': points_actual_array, 'frame_actual': frame_actual_array}
        #print(t, points_dict[t]," HIIIIIIIIIIIIIIIIIIIASDDDDDDDDDDDDDDDDDDDDDDDDDDD")
    return points_dict