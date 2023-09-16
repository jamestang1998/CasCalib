import cv2
import numpy as np
import matplotlib.pyplot as plt
import plotting
import util
from scipy.spatial.distance import directed_hausdorff
import icp_pytorch
import torch
import os
from sklearn.metrics.pairwise import cosine_similarity
import filter
import matplotlib.pyplot as plt
import math
import knn
import random
import plotly.graph_objects as go
import icp_pytorch_multi

from scipy.spatial.distance import cdist
from simpleicp import PointCloud, SimpleICP
import Icp2d
import rotation_pca

from scipy import signal, fftpack
from scipy.interpolate import RectBivariateSpline
from scipy.interpolate import UnivariateSpline

import numpy as np
from scipy.signal import correlate

import numpy as np
from scipy.spatial.distance import euclidean
import open3d as o3d
import registration

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
        #best_shift, best_scale, sync_dict = time_align_corr(data_ref, data_sync[i], save_dir = save_dir, name = str(i) + '_' + name, sync = sync, scale_search = scale_array_init, shift_search = shift_array_init, trial_name = 'init')
        best_shift, best_scale, sync_dict = time_align_corr(data_ref, data_sync[i], save_dir = save_dir, name = str(i) + '_' + name, sync = sync, scale_search = scale_array_init, shift_search = shift_array_init, trial_name = 'init', window = window, dilation = dilation)
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
    
    data_interp_ref = Icp2d.get_track(data_ref, 2*dilation)
    data_interp_sync = Icp2d.get_track(data_sync, 2*dilation)

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

    for track in data_interp_ref.keys():
        
        ref_time = data_interp_ref[track]['frame']
        ref_points = data_interp_ref[track]['points']
        if len(ref_points) < 2*dilation + 1:
            continue
        B_set = set(data_interp_ref[track]['frame_actual'])
        indices = [i for i, x in enumerate(ref_time) if x in B_set]
        ref_vel = np.transpose(np.array(ref_points))

        ax1.scatter(ref_time, np.linalg.norm(np.array(ref_vel), axis = 0).tolist(), c = 'b')
        ax1.set_yscale("linear")


        ref_vel_array.append(ref_vel[:, indices])
        #print(ref_vel.shape, " r ef vel")
        for st in range(len(ref_time)):          
            if ref_time[st] in ref_vel_dict:
                ref_vel_dict[ref_time[st]].append(ref_vel[:, st] - ref_mean)

            else:
                ref_vel_dict[ref_time[st]] = [ref_vel[:, st] - ref_mean]

        
    ref_vel_dict_avg = {}
    #print(ref_vel_dict)
    for f in ref_vel_dict.keys():
        ref_vel_dict_avg[f] = np.linalg.norm(np.mean(np.array(ref_vel_dict[f]), axis = 0))
    
    ref_time = np.array(list(ref_vel_dict_avg.keys()))
    ref_vel = np.array(list(ref_vel_dict_avg.values()))

    ref_average = filter.window_average(ref_vel, window = window)

    #############################################################
    sync_vel_dict = {} 

    ###################################
    sync_vel_array = []
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

        sync_vel_array.append(sync_vel[:, indices])
        
        sync_vel_dict_array.append((sync_time, np.linalg.norm(np.array(sync_vel), axis = 0).tolist()))
        #ax1.plot(sync_time, np.linalg.norm(np.array(sync_vel), axis = 0).tolist(), c = 'r')
        ax1.scatter(sync_time, np.linalg.norm(np.array(sync_vel), axis = 0).tolist(), c = 'r')
        ax1.set_yscale("linear")

        for st in range(len(sync_time)):
            
            if sync_time[st] in sync_vel_dict:
                sync_vel_dict[sync_time[st]].append(sync_vel[:, st] - sync_mean)
                
            else:
                sync_vel_dict[sync_time[st]] = [sync_vel[:, st] - sync_mean]

    sync_vel_dict_avg = {}
    for f in sync_vel_dict.keys():
        sync_vel_dict_avg[f] =  np.linalg.norm(np.mean(np.array(sync_vel_dict[f]), axis = 0))
    
    ax2.scatter(list(sync_vel_dict_avg.keys()), list(sync_vel_dict_avg.values()), c = 'r')
    ax2.set_yscale("linear")
    
    ax2.scatter(list(ref_vel_dict_avg.keys()), list(ref_vel_dict_avg.values()), c = 'b')
    ax2.set_yscale("linear")

    ax3.scatter(np.array(list(ref_vel_dict_avg.keys())), np.array(list(ref_vel_dict_avg.values())))
    ax3.scatter(np.array(list(sync_vel_dict_avg.keys())), np.array(list(sync_vel_dict_avg.values())))
    #plt.show()
            
    #sync_average = filter.window_average(sync_vel_init, window = window)
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

    best_correlation = -np.inf
    
    sync_vel_array = np.array(list(sync_vel_dict_avg.values()))# = util.normalize_data_median(np.array(list(sync_vel_dict_avg.values())))
    ref_average = ref_average#util.normalize_data_median(ref_average)

    if  save_dir is not None:
        fig, ax1 = plt.subplots(1, 1)
        ax1.set_yscale("linear")
        #ax1.plot(list(sync_vel_dict_avg.keys()), list(sync_vel_dict_avg.values()), c = 'r')
        #ax1.plot(ref_time, ref_average, c = 'b')
        ax1.scatter(list(sync_vel_dict_avg.keys()), list(sync_vel_array), c = 'r')
        ax1.scatter(ref_time, ref_average, c = 'b')
        ax1.set_title("init")

        if name is not None:
            fig.savefig(save_dir + '/search_time/' + name + '/' + 'init_' + trial_name + '_' + str(name) + '_' + str(best_shift) + '_' + str(best_scale) +'_' + '.png')
        else:
            fig.savefig(save_dir + '/search_time/' + name + '/' + 'init_' + trial_name + '_' + str(best_shift) + '_' + str(best_scale) +'_' + '.png')  

        fig1, ax2 = plt.subplots(1, 1)
        ax2.set_yscale("linear")
        #ax2.plot(shift_array, error_array, c = 'r')
        ax2.scatter(shift_array, error_array, c = 'r')
        ax2.set_title(str(best_correlation))

        plt.close('all')

    for scale in scale_val:#list(np.linspace(0.5, 2, 5)) + [1.0]:

        shift_array_loop = []
        error_array_loop = []

        correlation_array_loop = []
        for shift in shift_val:#list(range(-int(sync_time_init.shape[0]/64), int(sync_time_init.shape[0]/64))):
            sync_time = scale*np.array(list(sync_vel_dict_avg.keys())) + shift
            #sync_time = scale*np.array(list(sync_vel_array)) + shift

            #sync_vel = np.array(list(sync_vel_dict_avg.values()))

            #sync_average = filter.window_average(sync_vel/scale, window = window)
            sync_average = sync_vel_array
            sync_index = np.where(sync_average > thresh_mag)[0]

            argmins, argmins1, d2_sorted = knn.knn(np.expand_dims(ref_time, axis = 1), np.expand_dims(sync_time, axis = 1))

            x_union, y1_padded, y2_padded = union_and_pad(ref_time, ref_average, sync_time, sync_average)
            #print(x_union, " x union")
            #print(y1_padded, " y1 padded ")
            #print(y2_padded, " y2_padded")

            #correlation = np.corrcoef(sync_average[argmins1], ref_average[argmins], rowvar = True)[0,1]
            #mean_error = np.mean(np.abs(sync_average[argmins1] - ref_average[argmins]))
            correlation = np.corrcoef(y1_padded, y2_padded, rowvar = True)[0,1]
            mean_error = np.mean(np.abs(y1_padded - y2_padded))
            
            #correlation = correlation_signed
            correlation_array_loop.append(correlation)
            error_array.append(mean_error)
            shift_array.append(shift)
            if best_correlation < correlation:
                best_scale = scale
                best_shift = shift
                best_correlation = correlation
                '''
                best_sync_time = sync_time[argmins1]
                best_sync_average = sync_average[argmins1]

                best_ref_time = ref_time[argmins]
                best_ref_average = ref_average[argmins]
                '''
                
                best_sync_time = x_union#sync_time
                best_sync_average = y2_padded

                best_ref_time = x_union
                best_ref_average = y1_padded
                

            shift_array_loop.append(shift)
            error_array_loop.append(mean_error)
            ####################
        if  save_dir is not None:
            fig1, ax1 = plt.subplots(1, 1)
            #ax1.plot(shift_array_loop, error_array_loop, c = 'r')
            ax1.scatter(shift_array_loop, error_array_loop, c = 'r')
            ax1.set_yscale("linear")

            fig2, ax2 = plt.subplots(1, 1)
            #ax2.plot(shift_array_loop, correlation_array_loop, c = 'r')
            ax2.scatter(shift_array_loop, correlation_array_loop, c = 'r')
            ax2.set_yscale("linear")
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
            plt.close('all')
            
    if  save_dir is not None:
        fig, ax1 = plt.subplots(1, 1)
        ax1.set_yscale("linear")
        #ax1.plot(best_sync_time, best_sync_average, c = 'r')
        #ax1.plot(best_ref_time, best_ref_average, c = 'b')
        
        ax1.scatter(best_sync_time, best_sync_average, c = 'r')
        ax1.scatter(best_ref_time, best_ref_average, c = 'b')
        ax1.set_title(str(best_correlation))

        if name is not None:
            fig.savefig(save_dir + '/search_time/' + name + '/' + 'best_' + trial_name + '_' + str(name) + '_' + str(best_shift) + '_' + str(best_scale) +'_' + '.png')
        else:
            fig.savefig(save_dir + '/search_time/' + name + '/' + 'best_average_' + trial_name + '_' + str(best_shift) + '_' + str(best_scale) +'_' + '.png')  

        fig1, ax2 = plt.subplots(1, 1)
        ax2.set_yscale("linear")
        #ax2.plot(shift_array, error_array, c = 'r')
        ax2.scatter(shift_array, error_array, c = 'r')
        ax2.set_title(str(best_correlation))
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