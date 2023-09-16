import cv2
import numpy as np
import matplotlib.pyplot as plt
import plotting
import util
from scipy.spatial.distance import directed_hausdorff
import torch
import os
from sklearn.metrics.pairwise import cosine_similarity
import filter
import matplotlib.pyplot as plt
import math
import knn
import random
import plotly.graph_objects as go

from scipy.spatial.distance import cdist
from simpleicp import PointCloud, SimpleICP
import Icp2d

from scipy import signal, fftpack
from scipy.interpolate import RectBivariateSpline
from scipy.interpolate import UnivariateSpline

import numpy as np
from scipy.signal import correlate

import numpy as np
from scipy.spatial.distance import euclidean
import open3d as o3d

def dict_smooth(data, min_size = 2):
    print(data)

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
        
        x_plot = []
        y_plot = []
        frame_plot = []
        #print(track_dict[t])
        #print("************************I")
        for i in range(len(track_dict[t])):
            
            x_plot.append(track_dict[t][i]["coord"][0])
            y_plot.append(track_dict[t][i]["coord"][1])
            frame_plot.append(track_dict[t][i]["frame"])
        
        frame_plot, x_plot, y_plot = track_smooth(frame_plot, x_plot, y_plot)

        for i in range(len(frame_plot)):
            if frame_plot[i] not in points_dict:
                points_dict[frame_plot[i]] = {t: [x_plot[i], y_plot[i], 0.0, 0.0]}
            else:
                points_dict[frame_plot[i]][t] = [x_plot[i], y_plot[i], 0.0, 0.0]
        #print(t, points_dict[t]," HIIIIIIIIIIIIIIIIIIIASDDDDDDDDDDDDDDDDDDDDDDDDDDD")
    return points_dict

def track_smooth(t, x, y):
    # Create the splines for x and y separately
    #print(x, " THIS IS X")
    #print(np.var(x), len(x), " LEN X")
    print(t)
    stop
    spline_x = UnivariateSpline(t, x, k=3, s=np.var(x)*len(x))
    spline_y = UnivariateSpline(t, y, k=3, s=np.var(y)*len(y))

    # Example evaluation points
    t_eval = np.array(list(range(min(t), max(t))))  # Adjust the range and number of points as needed

    # Evaluate the splines
    x_smooth = spline_x(t_eval)
    y_smooth = spline_y(t_eval)
    return t_eval, x_smooth, y_smooth 
def corr(a, b):

    a_b = np.argmax(signal.correlate(a,b))
    b_a = np.argmax(signal.correlate(b,a))
    
    return 

def time_all(data_ref, data_sync, save_dir = None, sync = True, name = '', window = 1, dilation = 1, search_div = 6.0, init_sync = []):
    
    best_shift_array = []
    best_scale_array = []
    sync_dict_array = []
    
    for i in range(len(data_sync)):
        scale_array_init = [1]#[0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2.0, 2.25,2.5,2.75,3.0, 3.25, 3.5]
        
        shift_array_init = list(range(-int(len(list(data_sync[i].keys()))/search_div), int(len(list(data_sync[i].keys()))/search_div)))
        
        if len(init_sync) != 0:
            shift_array_init = init_sync 
        #print(data_ref)
        best_shift, best_scale, sync_dict = time_align_corr(data_ref, data_sync[i], save_dir = save_dir, name = str(i) + '_' + name, sync = sync, scale_search = scale_array_init, shift_search = shift_array_init, trial_name = 'init')
        #best_shift, best_scale, sync_dict = time_align_hungarian(data_ref, data_sync[i], save_dir = save_dir, name = str(i) + '_' + name, sync = sync, scale_search = scale_array_init, shift_search = shift_array_init, trial_name = 'init', window = window, dilation = dilation)
        #scale_array_init = [best_scale]
        #shift_array_init = list(range(int(best_shift) - 20, int(best_shift) + 20))
        #best_shift, best_scale, sync_dict = time_align_corr(data_ref, data_sync[i], save_dir = save_dir, name = str(i) + '_' + name, window = 1, dilation = 15, sync = sync, scale_search = scale_array_init, shift_search = shift_array_init, trial_name = 'refine')

        best_shift_array.append(best_shift)
        best_scale_array.append(best_scale)
        sync_dict_array.append(sync_dict)
    return best_shift_array, best_scale_array, sync_dict_array

def union_and_pad(x1, y1, x2, y2):
    x_union = np.union1d(x1, x2)  # Find the union of times from x1 and x2

    # Pad y1 based on its last or first value to match the union time range
    y1_padded = np.interp(x_union, x1, y1, left=y1[0], right=y1[-1])

    # Pad y2 based on its last or first value to match the union time range
    y2_padded = np.interp(x_union, x2, y2, left=y2[0], right=y2[-1])

    return x_union, y1_padded, y2_padded

def pad(x1, y1, x2, y2):
    x_union = np.union1d(x1, x2)  # Find the union of times from x1 and x2
    x_low = min(x_union)
    x_high = max(x_union)
    # Pad y1 based on its last or first value to match the union time range
    y1_padded = np.interp(x_union, x1, y1, left=y1[0], right=y1[-1])

    # Pad y2 based on its last or first value to match the union time range
    y2_padded = np.interp(x_union, x2, y2, left=y2[0], right=y2[-1])

    return x_union, y1_padded, y2_padded

def time_knn(best_scale, best_shift, data_sync, data_ref):
    shifted_time = best_scale*np.array(list(data_sync)) + best_shift
    argmins, argmins1, d2_sorted = knn.knn(np.expand_dims(list(data_ref), axis = 1), np.expand_dims(shifted_time, axis = 1))

    sync_dict = {}
    for i in range(len(argmins)):
        sync_dict[list(data_ref)[argmins[i]]] = int(list(data_sync)[argmins1[i]])

    return sync_dict 

def time_knn_array(best_scale, best_shift, data_ref):
    #shifted_time = best_scale*np.array(data_sync) + best_shift
    #argmins, argmins1, d2_sorted = knn.knn(np.expand_dims(list(data_ref.keys()), axis = 1), np.expand_dims(shifted_time, axis = 1))

    sync_dict = {}
    for i in range(len(data_ref)):
        sync_dict[data_ref[i]] = (data_ref[i] - best_shift)/best_scale
    return sync_dict 

def time_align_corr(data_ref, data_sync, save_dir = None, name = None, end = True, window = 1, dilation = 1, thresh_mag = 0.0, sync = True, trial_name = 'init', scale_search = [1], shift_search = [0]):
    
    if os.path.isdir(save_dir + '/search_time/' + name) == False:
        os.mkdir(save_dir + '/search_time/' + name)

    #print(data_ref)
    #print(data_sync)

    #data_interp_ref = data_ref#Icp2d.interpolate(data_ref, 2*dilation)
    #data_interp_sync = data_sync#Icp2d.interpolate(data_sync, 2*dilation)
    #print(data_ref.keys(), len(list(data_ref.keys())), " DATA REF")
    data_interp_ref = Icp2d.get_track(data_ref)
    data_interp_sync = Icp2d.get_track(data_sync)
    #print(data_interp_ref.keys(), " DATA interp REF")
    #stop

    ref_vel_array = []
    ref_vel_dict = {} 

    fig1, ax1 = plt.subplots(1, 1)
    fig2, ax2 = plt.subplots(1, 1)
    fig3, ax3 = plt.subplots(1, 1)
    #print(data_ref)   

    ############################
    ref_mean_array = []
    sync_mean_array = []
    
    for track in data_interp_ref.keys():
            
        ref_time = data_interp_ref[track]['frame']
        ref_points = data_interp_ref[track]['points']

        B_set = set(data_interp_ref[track]['frame_actual'])
        indices = [i for i, x in enumerate(ref_time) if x in B_set]
        ref_vel = np.transpose(np.array(ref_points))

        for st in range(len(ref_time)):
            ref_mean_array.append(ref_vel[:, st])
    
    for track in data_interp_sync.keys():
            
        sync_time = data_interp_sync[track]['frame']
        sync_points = data_interp_sync[track]['points']

        B_set = set(data_interp_sync[track]['frame_actual'])
        indices = [i for i, x in enumerate(sync_time) if x in B_set]
        sync_vel = np.transpose(np.array(sync_points))

        for st in range(len(sync_time)):
            sync_mean_array.append(sync_vel[:, st])
                
    ref_mean = np.mean(ref_mean_array, axis = 0)
    sync_mean = np.mean(sync_mean_array, axis = 0)
    ############################
    #print(ref_mean, sync_mean, " ref sync")
    #stop
    ref_time_array = []

    for track in data_interp_ref.keys():
        
        ref_time = data_interp_ref[track]['frame']
        ref_points = data_interp_ref[track]['points']
        if len(ref_points) < 2*dilation + 1:
            continue
        B_set = set(data_interp_ref[track]['frame_actual'])
        indices = [i for i, x in enumerate(ref_time) if x in B_set]
        ref_vel = np.transpose(np.array(ref_points))

        #ax1.scatter(ref_time, np.linalg.norm(np.array(ref_vel), axis = 0).tolist(), c = 'b')
        #ax1.set_yscale("linear")

        #print(ref_vel.shape, " r ef vel")
        for st in range(len(ref_time)):        
            ref_time_array.append(ref_time[st])
            ref_vel_array.append(np.linalg.norm(ref_vel[:, st] - ref_mean))
            if ref_time[st] in ref_vel_dict:
                ref_vel_dict[ref_time[st]].append(ref_vel[:, st] - ref_mean)

            else:
                ref_vel_dict[ref_time[st]] = [ref_vel[:, st] - ref_mean]

    #arg_sort = np.argsort(ref_time_array)
    #ref_time_array = list(np.array(ref_time_array)[arg_sort])
    #ref_vel_array = list(np.array(ref_vel_array)[arg_sort])
       
    ref_vel_dict_avg = {}
    #print(ref_vel_dict)
    #ref_time = ref_time_array
    #ref_vel = ref_vel_array
    '''
    for f in ref_vel_dict.keys():
        #print(ref_vel_dict[f])
        ref_vel_dict_avg[f] = np.linalg.norm(np.mean(np.array(ref_vel_dict[f]), axis = 0))
    '''
    #ref_time = np.array(list(ref_vel_dict_avg.keys()))
    #ref_vel = np.array(list(ref_vel_dict_avg.values()))

    #ref_average = filter.window_average(ref_vel, window = window)

    #############################################################
    sync_vel_dict = {} 

    ###################################
    sync_vel_array = []
    sync_time_array = []

    sync_vel_dict_array = []

    for track in data_interp_sync.keys():
        
        sync_frame = data_interp_sync[track]['frame']
        sync_points = data_interp_sync[track]['points']
        sync_time = sync_frame

        if len(sync_points) < 2*dilation + 1:
            continue

        B_set = set(data_interp_sync[track]['frame_actual'])
        indices = [i for i, x in enumerate(sync_time) if x in B_set]
        
        sync_vel = np.transpose(np.array(sync_points))#torch.squeeze(util.central_diff(torch.unsqueeze(torch.transpose(torch.from_numpy(np.array(sync_points)), 0 , 1), dim = 1).double(), time = 1, dilation = dilation))

        #sync_vel_array.append(sync_vel[:, indices])
        
        sync_vel_dict_array.append((sync_time, np.linalg.norm(np.array(sync_vel), axis = 0).tolist()))
        #ax1.plot(sync_time, np.linalg.norm(np.array(sync_vel), axis = 0).tolist(), c = 'r')
        #ax1.scatter(sync_time, np.linalg.norm(np.array(sync_vel), axis = 0).tolist(), c = 'r')
        #ax1.set_yscale("linear")

        for st in range(len(sync_time)):

            sync_vel_array.append(np.linalg.norm(sync_vel[:, st] - sync_mean))
            sync_time_array.append(sync_time[st])
            
            if sync_time[st] in sync_vel_dict:
                sync_vel_dict[sync_time[st]].append(sync_vel[:, st] - sync_mean)
                
            else:
                sync_vel_dict[sync_time[st]] = [sync_vel[:, st] - sync_mean]

    arg_sort = np.argsort(sync_time_array)
    sync_time_array = list(np.array(sync_time_array)[arg_sort])
    sync_vel_array = list(np.array(sync_vel_array)[arg_sort])
    
    #plt.show()
    
    #sync_time = np.array(list(sync_vel_dict_avg.keys()))
    #sync_vel = np.array(list(sync_vel_dict_avg.values()))
    #sync_average = filter.window_average(sync_vel, window = window)
    #############################################################
    best_sync_time = None
    best_sync_average = None

    best_ref_time = None
    best_ref_average = None

    best_scale = 1.0
    best_shift = 0.0

    error_array = []
    shift_array = []
    #scale_search = [1], shift_search
    #shift_val = list(range(-int(sync_time_init.shape[0]/30), int(sync_time_init.shape[0]/30)))
    #scale_val = [0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2.0, 2.25,2.5,2.75,3.0, 3.25, 3.5]
    shift_val = shift_search
    scale_val = scale_search
    #scale_val = list(np.linspace(0.25, 4, 10)) + [1.0]
    if sync == False:
        shift_val = [0]
        scale_val = [1]

    best_chamfer_dist = np.inf

    '''
    sync_vel_array, indices_sync = util.remove_top_20_percent(sync_vel_array)
    ref_vel_array, indices_ref = util.remove_top_20_percent(ref_vel_array)

    sync_time_array = np.delete(sync_time_array, indices_sync)
    ref_time_array = np.delete(ref_time_array, indices_ref)
    '''
    sync_vel_array = util.normalize_data_median(sync_vel_array)
    ref_vel_array = util.normalize_data_median(ref_vel_array)

    if  save_dir is not None:
        fig, ax1 = plt.subplots(1, 1)
        ax1.set_yscale("linear")
        #ax1.plot(list(sync_vel_dict_avg.keys()), list(sync_vel_dict_avg.values()), c = 'r')
        #ax1.plot(ref_time, ref_average, c = 'b')
        ax1.scatter(sync_time_array, sync_vel_array, c = 'r')
        ax1.scatter(ref_time_array, ref_vel_array, c = 'b')
        ax1.set_title("init")

        if name is not None:
            fig.savefig(save_dir + '/search_time/' + name + '/' + 'init_' + trial_name + '_' + str(name) + '_' + str(best_shift) + '_' + str(best_scale) +'_' + '.png')
        else:
            fig.savefig(save_dir + '/search_time/' + name + '/' + 'init_' + trial_name + '_' + str(best_shift) + '_' + str(best_scale) +'_' + '.png')  

        fig1, ax2 = plt.subplots(1, 1)
        ax2.set_yscale("linear")
        #ax2.plot(shift_array, error_array, c = 'r')
        ax2.scatter(shift_array, error_array, c = 'r')
        ax2.set_title(str(best_chamfer_dist))

        plt.close('all')

    #A = np.transpose(np.stack((ref_time_array, ref_vel_array)))
    #B = np.transpose(np.stack((np.array(sync_time_array), np.array(sync_vel_array))))

    #A = util.remove_outliers_array(A, eps=10)
    #B = util.remove_outliers_array(B, eps=10)

    #ref_time_array = A[:, 0]
    #ref_vel_array = A[:, 1]
    
    #sync_time_array = B[:, 0]
    #sync_vel_array = B[:, 1]
    
    for scale in scale_val:#list(np.linspace(0.5, 2, 5)) + [1.0]:

        shift_array_loop = []
        error_array_loop = []
        chamfer_array_loop = []

        correlation_array_loop = []
        
        for shift in shift_val:#list(range(-int(sync_time_init.shape[0]/64), int(sync_time_init.shape[0]/64))):
            sync_time = scale*np.array(sync_time_array) + shift

            sync_vel = np.array(sync_vel_array)

            #sync_average = filter.window_average(sync_vel/scale, window = window)
            
            #sync_index = np.where(sync_average > thresh_mag)[0]

            #argmins, argmins1, d2_sorted = knn.knn(np.expand_dims(ref_time, axis = 1), np.expand_dims(sync_time, axis = 1))

            #util.match_3d_plotly_input2d_farthest_point(ref_time_array, ref_vel_array, sync_time, sync_vel)
            #print(ref_vel_array)
            #print(ref_time_array)
            
            #x_union, y1_padded, y2_padded = union_and_pad(ref_time_array, ref_vel_array, sync_time, sync_vel)
            #print(x_union, " x union")

            #print(y1_padded, " y1 padded ")
            #print(y2_padded, " y2_padded")
            
            #correlation = np.corrcoef(sync_average[argmins1], ref_average[argmins], rowvar = True)[0,1]
            #mean_error = np.mean(np.abs(sync_average[argmins1] - ref_average[argmins]))

            #A = np.transpose(np.stack((x_union, y1_padded)))
            #B = np.transpose(np.stack((x_union, y2_padded)))
            A = np.transpose(np.stack((np.array(ref_time_array), ref_vel_array)))
            B = np.transpose(np.stack((sync_time, sync_vel)))

            #print(A.shape,  "A SHAPE")
            #stop
            chamfer_dist, A_, B_ = util.hungarian_match_distance(A, B)
            #chamfer_dist = util.chamfer_distance(A, B, metric='l2', direction='bi')
            #print(chamfer_dist, " CHAMFER DIST")
            #correlation = np.corrcoef(y1_padded, y2_padded, rowvar = True)[0,1]
            #mean_error = np.mean(np.abs(y1_padded - y2_padded))
            
            #correlation = correlation_signed
            #error_array.append(mean_error)
            shift_array.append(shift)
            if best_chamfer_dist > chamfer_dist:
                best_scale = scale
                best_shift = shift
                best_chamfer_dist = chamfer_dist
                
                A_array = util.dictionary_to_array(A_)
                B_array = util.dictionary_to_array(B_)

                best_sync_time = list(np.array(B_array)[:, 0])
                best_sync_average = list(np.array(B_array)[:, 1])

                best_ref_time = list(np.array(A_array)[:, 0])
                best_ref_average = list(np.array(A_array)[:, 1])

                '''
                best_sync_time = sync_time
                best_sync_average = sync_vel

                best_ref_time = ref_time_array
                best_ref_average = ref_vel_array
                '''
                #best_sync_time = x_union#sync_time
                #best_sync_average = y2_padded

                #best_ref_time = x_union
                #best_ref_average = y1_padded
            
            chamfer_array_loop.append(chamfer_dist)
            shift_array_loop.append(shift)
            #error_array_loop.append(mean_error)
            #correlation_array_loop.append(correlation)
            #print(chamfer_array_loop)
            #print(shift, "HELLOOOOO")
            ####################
        if  save_dir is not None and sync is True:
            fig1, ax1 = plt.subplots(1, 1)
            #ax1.plot(shift_array_loop, error_array_loop, c = 'r')
            '''
            ax1.scatter(shift_array_loop, error_array_loop, c = 'r')
            ax1.set_yscale("linear")

            fig2, ax2 = plt.subplots(1, 1)
            #ax2.plot(shift_array_loop, correlation_array_loop, c = 'r')
            ax2.scatter(shift_array_loop, correlation_array_loop, c = 'r')
            ax2.set_yscale("linear")
            '''
            fig0, ax0 = plt.subplots(1, 1)
            #ax2.plot(shift_array_loop, correlation_array_loop, c = 'r')
            ax0.scatter(shift_array_loop, chamfer_array_loop, c = 'r')
            ax0.set_yscale("linear")

            '''
            if name is not None:
                #fig1.savefig(save_dir + '/search_time/' + name + '/' + 'error_' + str(scale) + '_' + str(name) + '_' + str(best_shift) + '_' + str(best_scale) +'_' + '.png')
                fig1.savefig(save_dir + '/search_time/' + name + '/' + 'error_' + name + '_' + str(best_shift) + '_' + '.png')
            else:
                #fig1.savefig(save_dir + '/search_time/' + name + '/' + 'error_' + str(scale) + '_' + str(best_shift) + '_' + str(best_scale) +'_' + '.png')  
                fig1.savefig(save_dir + '/search_time/' + name + '/' + 'error_' + name + '_' + str(best_shift) + '_' + '.png') 

            if name is not None:
                #fig2.savefig(save_dir + '/search_time/' + name + '/' + 'cor_' + str(scale) + '_' + str(name) + '_' + str(best_shift) + '_' + str(best_scale) +'_' + '.png')
                fig2.savefig(save_dir + '/search_time/' + name + '/' + 'cor_' + name + '_' + str(best_shift) + '_' + '.png')
            else:
                #fig2.savefig(save_dir + '/search_time/' + name + '/' + 'cor_' + str(scale) + '_' + str(best_shift) + '_' + str(best_scale) +'_' + '.png')
                fig2.savefig(save_dir + '/search_time/' + name + '/' + 'cor_' + name + '_' + str(best_shift) + '_' + '.png')    
            '''
            if name is not None:
                #fig2.savefig(save_dir + '/search_time/' + name + '/' + 'cor_' + str(scale) + '_' + str(name) + '_' + str(best_shift) + '_' + str(best_scale) +'_' + '.png')
                fig0.savefig(save_dir + '/search_time/' + name + '/' + 'chamfer_' + name + '_' + str(best_shift) + '_' + '.png')
            else:
                #fig2.savefig(save_dir + '/search_time/' + name + '/' + 'cor_' + str(scale) + '_' + str(best_shift) + '_' + str(best_scale) +'_' + '.png')
                fig0.savefig(save_dir + '/search_time/' + name + '/' + 'chamfer_' + name + '_' + str(best_shift) + '_' + '.png')    
            plt.close('all')
    #stop
    if  save_dir is not None and sync is True:
        fig, ax1 = plt.subplots(1, 1)
        ax1.set_yscale("linear")
        #ax1.plot(best_sync_time, best_sync_average, c = 'r')
        #ax1.plot(best_ref_time, best_ref_average, c = 'b')
        
        ax1.scatter(best_sync_time, best_sync_average, c = 'r')
        ax1.scatter(best_ref_time, best_ref_average, c = 'b')
        ax1.set_title(str(best_chamfer_dist))

        if name is not None:
            fig.savefig(save_dir + '/search_time/' + name + '/' + 'best_' + trial_name + '_' + str(name) + '_' + str(best_shift) + '_' + str(best_scale) +'_' + '.png')
        else:
            fig.savefig(save_dir + '/search_time/' + name + '/' + 'best_average_' + trial_name + '_' + str(best_shift) + '_' + str(best_scale) +'_' + '.png')  

        #fig1, ax2 = plt.subplots(1, 1)
        #ax2.set_yscale("linear")
        #ax2.plot(shift_array, error_array, c = 'r')
        #ax2.scatter(shift_array, error_array, c = 'r')
        #ax2.set_title(str(best_chamfer_dist))
        '''
        if name is not None:
            fig1.savefig(save_dir + '/search_time/' + name + '/' + 'error_' + str(name) + '_' + str(best_shift) + '_' + str(best_scale) +'_' + '.png')
        else:
            fig1.savefig(save_dir + '/search_time/' + name + '/' + 'error_' + str(best_shift) + '_' + str(best_scale) +'_' + '.png')  
        '''
        plt.close('all')

    shifted_time = best_scale*np.array(list(data_sync.keys())) + best_shift
    argmins, argmins1, d2_sorted = knn.knn(np.expand_dims(list(data_ref.keys()), axis = 1), np.expand_dims(shifted_time, axis = 1))

    sync_dict = {}
    for i in range(len(argmins)):
        sync_dict[list(data_ref.keys())[argmins[i]]] = int(list(data_sync.keys())[argmins1[i]])

    #stop
    return best_shift, best_scale, sync_dict