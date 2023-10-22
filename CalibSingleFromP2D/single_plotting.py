import numpy as np
import matplotlib.pyplot as plt
import single_util
import matplotlib.cm as cm
import imageio
import os

import matplotlib.patches as mpatches
import scipy.ndimage as sp
from random import choice
import math
import plotly.express as px
import plotly.graph_objects as go
from skimage import io
import single_bundle_intersect
import torch
import single_Icp2d

def plot_slide1(df, save_dir, name):
    #fig = px.scatter(df, x="sp", y="y", animation_frame='ai')
    fig = px.scatter(df, x="x", y="y", animation_frame='frame', title="Plot for SI = " + name.split('_')[-1])
    fig["layout"].pop("updatemenus")
    fig.update_xaxes(range=[-10, 10])
    # Show the interactive graph
    fig.write_html(save_dir + "/" + name + ".html")

def plot_camera_pose(rotation, position, save_dir, scale = 1, name = '', label = ['pred', 'gt']):

    data = []

    for i in range(len(rotation)):

        camera_rotation_gt = rotation[i]
        camera_translation_gt = position[i]
        for cam1 in range(len(camera_rotation_gt)):
                    
                R_mat1 = torch.tensor(camera_rotation_gt[cam1]).double()#so3_exponential_map(R_matrix)[cam1].detach()

                T_mat1 = torch.tensor(camera_translation_gt[cam1]).double()#T_matrix[cam1].detach()
                print(T_mat1, " T MAT 1 ")
                print(R_mat1, " R MAT 1 ")
                trace2 = go.Scatter3d(
                    x=[T_mat1[0].numpy(),T_mat1[0].numpy() + scale*R_mat1[0][0].numpy()],
                    y=[T_mat1[1].numpy(),T_mat1[1].numpy() + scale*R_mat1[0][1].numpy()],
                    z=[T_mat1[2].numpy(),T_mat1[2].numpy() + scale*R_mat1[0][2].numpy()],
                    mode='lines',
                    name= " axis1 " + label[i] + "  " + str(cam1),
                    marker=dict(
                        color='red',
                    )
                )

                trace3 = go.Scatter3d(
                    x=[T_mat1[0].numpy(),T_mat1[0].numpy() + scale*R_mat1[1][0].numpy()],
                    y=[T_mat1[1].numpy(),T_mat1[1].numpy() + scale*R_mat1[1][1].numpy()],
                    z=[T_mat1[2].numpy(),T_mat1[2].numpy() + scale*R_mat1[1][2].numpy()],
                    mode='lines',
                    name= " axis2 " + label[i] + "  " + str(cam1),
                    marker=dict(
                        color='green',
                    )
                )

                trace4 = go.Scatter3d(
                    x=[T_mat1[0].numpy(),T_mat1[0].detach().numpy() + scale*R_mat1[2][0].numpy()],
                    y=[T_mat1[1].numpy(),T_mat1[1].detach().numpy() + scale*R_mat1[2][1].numpy()],
                    z=[T_mat1[2].numpy(),T_mat1[2].detach().numpy() + scale*R_mat1[2][2].numpy()],
                    mode='lines',
                    name=  " axis3 " + label[i] + "  " + str(cam1),
                    marker=dict(
                        color='blue',
                    )
                )

                trace5 = go.Scatter3d(
                    x=[T_mat1[0].numpy()],
                    y=[T_mat1[1].numpy()],
                    z=[T_mat1[2].numpy()],
                    mode='markers+text',
                    name =  " " ,
                    marker=dict(
                        color='black',
                        size = 0.1,
                    ),
                    text = label[i] + "  " + str(cam1),
                    textposition='top right',
                    textfont=dict(color='black')
                )

                data.append(trace2)
                data.append(trace3)
                data.append(trace4)
                data.append(trace5)
    fig = go.Figure(data=data)

    fig.update_layout(scene = dict( aspectmode='data'))

    fig.update_layout(title_text='reconstruction')
            #fig.show()
    fig.write_html(save_dir + '/procrustes_' + name + '_.html')

    fig.data = []

    del fig
    return

def heatmap_velocity(x, v, name, save_dir):
    plt.clf()
    fig, ax = plt.subplots(1, 1)
    
    c, *_ = np.histogram2d(x[:,0], x[:,1], bins=20)
    # Bin Weight Sums:
    heatmap, xbin, ybin = np.histogram2d(x[:,0], x[:,1], bins=20, weights=v)
    extent = [xbin.min(), xbin.max(), ybin.min(), ybin.max()]

    # Render rectangular histogram:
    x = sp.filters.gaussian_filter(heatmap, sigma = 20, order = 0)
    ax.imshow(x.T, extent=extent, origin='lower')

    plt.show()

    fig.savefig(save_dir + '/' + 'imgplot_' + str(name) + '.png')

def generate_3d_plotly_matches_input2d(peturb_extrinsics_array, peturb_single_view_array, cam_axis_array, cam_position_array, all_points_array,center_array, run_name, scale = 1000, title = 'plot', name = ['pred'], frame_init = 0 ,frame_end = 500):
    
    #draw a square
    frame_number = frame_end - frame_init 
    color_array = ['rgba(200,10,10,', 'rgba(10,200,10,', 'rgba(10,10,200,', 'rgba(0,0,0,']

    color_array_string = ['red', 'green', 'blue', 'black']
    data = []

    #for f in range(0, len(list(all_points_array[0].keys())), 10):
    #######################################################
    track_array = []

    track_dict = {}
    for a in range(len(cam_position_array[0])):
        track_dict[a] = {}
    
    #print(track_dict)

    opacity_array = []
    
    for i in range(len(color_array)):
        op = []
        for j in range(0, frame_number):
            #op.append(color_array[i] + str(float(1 - np.exp(-4*j/(frame_number)))) + ')')
            op.append(color_array[i] + '1.0' + ')')
        opacity_array.append(op)

    #for f in range(0, frame_number):
    for f in range(frame_init, frame_end):
        #print(f)
        
        fr = list(all_points_array[0].keys())[f]
 
        #print(fr, " f in plotting")
        #print(all_points_array)
        ########################################
        all_points_ref = all_points_array[0]

        track_array.append(all_points_ref)

        #print(np.array(list(all_points_ref[fr].values())).shape, " SHAOEEEE")
        #print(peturb_single_view_array)
        #print(all_points_ref)
        #print(np.array(list(all_points_ref[fr].values())).shape, " HSPAEASE")
        plane = single_util.plane_ray_intersection_np(np.array(list(all_points_ref[fr].values()))[:, 0], np.array(list(all_points_ref[fr].values()))[:, 1], np.linalg.inv(peturb_single_view_array[0]['cam_matrix']),  peturb_single_view_array[0]["ground_normal"], peturb_single_view_array[0]['ground_position'])
        
        #print(len(peturb_extrinsics_array), " asdasdasd")
        init_ref_shift_matrix = peturb_extrinsics_array[0]['init_sync_center_array']
        icp_rot_matrix = peturb_extrinsics_array[0]['icp_rot_array']
        init_rot_matrix = peturb_extrinsics_array[0]['icp_init_rot_array']
        plane_matrix_array = peturb_extrinsics_array[0]['plane_matrix_array']
        '''
        plane = np.transpose(np.r_[plane, np.ones((1, plane.shape[1]))])
        points_ref = np.transpose(icp_rot_matrix @ init_rot_matrix @ np.transpose(np.transpose(plane_matrix_array @ np.transpose(np.array(plane))) - np.array([[init_ref_shift_matrix[0], init_ref_shift_matrix[1], 0, 0]]))[:3, :])
        '''
        points_ref = []
        '''
        if fr == 83:
            print(plane, " plot ref") 
            print(np.array(list(all_points_ref[fr].values()))[:, 0], np.array(list(all_points_ref[fr].values()))[:, 1], np.linalg.inv(peturb_single_view_array[0]['cam_matrix']),  peturb_single_view_array[0]["ground_normal"], peturb_single_view_array[0]['ground_position'])
        '''
        for k in all_points_ref[fr].keys():

            plane = single_util.plane_ray_intersection_np([np.array(list(all_points_ref[fr][k]))[0]], [np.array(list(all_points_ref[fr][k]))[1]], np.linalg.inv(peturb_single_view_array[0]['cam_matrix']),  peturb_single_view_array[0]["ground_normal"], peturb_single_view_array[0]['ground_position'])
            plane = np.transpose(np.r_[plane, np.ones((1, plane.shape[1]))])

            #print(plane.shape, " HEALSDOAIASDASDASD")
            transformed_ref = np.transpose(icp_rot_matrix @ init_rot_matrix @ np.transpose(np.transpose(plane_matrix_array @ np.transpose(np.array(plane))) - np.array([[init_ref_shift_matrix[0], init_ref_shift_matrix[1], 0, 0]]))[:3, :])
            if k in track_dict[0].keys():
                #track_dict[0][k].append(np.squeeze(transformed_ref))
                if np.linalg.norm(transformed_ref - track_dict[0][k][-1]) < np.inf:
                    track_dict[0][k].append(transformed_ref)
                    points_ref.append(transformed_ref)
            else:
                track_dict[0][k] = []
                #track_dict[0][k].append(np.squeeze(transformed_ref))
                track_dict[0][k].append(transformed_ref)
                points_ref.append(transformed_ref)
        
        points_ref = np.concatenate(points_ref)
        #print(points_ref.shape, " hleoewaokkeaeaewawewea")        

        x_ref = points_ref[:, 0]#[0, 1, 0, 1, 0, 1, 0, 1]
        y_ref = points_ref[:, 1]#[0, 1, 1, 0, 0, 1, 1, 0]
        z_ref = points_ref[:, 2]#[0, 0, 0, 0, 1, 1, 1, 1]
        #print(points_ref.shape, " POITS REFFFFF")
        #print(points_ref, " POITS REFFFFF!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        #stop
        trace1 = go.Scatter3d(
            x=x_ref,
            y=y_ref,
            z=z_ref,
            mode='markers',
            name= 'plane ' + str(0),
            marker_size = 3,
            marker_color=opacity_array[0]#color_array_string[0]
        )
        data.append(trace1)
        '''
        trace2 = go.Scatter3d(
            x=x_ref,
            y=y_ref,
            z=z_ref,
            mode='lines+markers',
            name= 'plane tracks ' + str(0),
            marker_size = 3,
            marker_color=opacity_array[0]#color_array_string[0]
        )
        data.append(trace2)
        '''
        ########################################
        
        for i in range(1, len(all_points_array)):
  
            #print(all_points_array[i].keys())
            if fr not in all_points_array[i].keys():
                continue
            
            all_points = all_points_array[i]
            #print(all_points_ref)

            #######################################
            init_ref_shift_matrix = peturb_extrinsics_array[i]['init_sync_center_array']
            icp_rot_matrix = peturb_extrinsics_array[i]['icp_rot_array']
            init_rot_matrix = peturb_extrinsics_array[i]['icp_init_rot_array']
            plane_matrix_array = peturb_extrinsics_array[i]['plane_matrix_array']
            #print(np.array(list(all_points[fr].values())).shape, " !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            '''
            plane = single_util.plane_ray_intersection_np(np.array(list(all_points[fr].values()))[:, 0], np.array(list(all_points[fr].values()))[:, 1], np.linalg.inv(peturb_single_view_array[i]['cam_matrix']),  peturb_single_view_array[i]["ground_normal"], peturb_single_view_array[i]['ground_position'])
            plane = np.transpose(np.r_[plane, np.ones((1, plane.shape[1]))])
            points = np.transpose(icp_rot_matrix @ init_rot_matrix @ np.transpose(np.transpose(plane_matrix_array @ np.transpose(np.array(plane))) - np.array([[init_ref_shift_matrix[0], init_ref_shift_matrix[1], 0, 0]]))[:3, :])
            '''
            points = []
            for c in all_points_array[i][fr].keys():
                
                plane1 = single_util.plane_ray_intersection_np([np.array(list(all_points_array[i][fr][c]))[0]], [np.array(list(all_points_array[i][fr][c]))[1]], np.linalg.inv(peturb_single_view_array[i]['cam_matrix']),  peturb_single_view_array[i]["ground_normal"], peturb_single_view_array[i]['ground_position'])
                plane1 = np.transpose(np.r_[plane1, np.ones((1, plane1.shape[1]))])
                transformed_sync = np.transpose(icp_rot_matrix @ init_rot_matrix @ np.transpose(np.transpose(plane_matrix_array @ np.transpose(np.array(plane1))) - np.array([[init_ref_shift_matrix[0], init_ref_shift_matrix[1], 0, 0]]))[:3, :])
                
                if c in track_dict[i].keys():
                    if np.linalg.norm(transformed_sync - track_dict[i][c][-1]) < np.inf:
                        track_dict[i][c].append(transformed_sync)
                        points.append(transformed_sync)
                else:
                    track_dict[i][c] = []
                    track_dict[i][c].append(transformed_sync)
                    points.append(transformed_sync)

            points = np.concatenate(points)
            '''
            for k in all_points[fr].keys():
                
                plane = single_util.plane_ray_intersection_np([np.array(list(all_points[fr][k]))[0]], [np.array(list(all_points[fr][k]))[1]], np.linalg.inv(peturb_single_view_array[i]['cam_matrix']),  peturb_single_view_array[i]["ground_normal"], peturb_single_view_array[i]['ground_position'])
                plane = np.transpose(np.r_[plane, np.ones((1, plane.shape[1]))])
                transformed_sync = np.transpose(icp_rot_matrix @ init_rot_matrix @ np.transpose(np.transpose(plane_matrix_array @ np.transpose(np.array(plane))) - np.array([[init_ref_shift_matrix[0], init_ref_shift_matrix[1], 0, 0]]))[:3, :])
                if k in track_dict[0].keys():
                    track_dict[0][k].append(transformed_sync)
                else:
                    track_dict[0][k] = []
                    track_dict[0][k].append(transformed_sync)
            '''
            #######################################
            points_center_ref = np.transpose(np.array(list(center_array[0]))[:2])
            points_center = np.transpose(np.array(list(center_array[i]))[:2])
            #print(points_center.shape, points_ref.shape, points.shape, " points_centerpoints_center")
            #indices = Icp2d.match(all_points_ref[fr][:,:2], all_points[fr][:,:2], center_array[0][:2],  center_array[i][:2])  
            #indices = Icp2d.match(points_ref, points, points_center_ref,  points_center)  
            #indices = Icp2d.match_no_normalize(points_ref, points)
            #A B
            matched_A, matched_B, row_ind, col_ind = single_util.hungarian_assignment(points_ref, points)
            #print(len(indices), " heloasdasdddd")

            x_sync = points[:, 0]#[0, 1, 0, 1, 0, 1, 0, 1]
            y_sync = points[:, 1]#[0, 1, 1, 0, 0, 1, 1, 0]
            z_sync = points[:, 2]#[0, 0, 0, 0, 1, 1, 1, 1]
            
            for ind in range(len(row_ind)):
                
                #sync_ind = indices[ind]
                sync_ind = col_ind[ind]
                ref_ind = row_ind[ind]

                #print([x_ref[ind], x_sync[sync_ind]])
                #print([y_ref[ind], y_sync[sync_ind]])
                #print([z_ref[ind], z_sync[sync_ind]], " ZZZZ")
                
                trace1 = go.Scatter3d(
                    x=[x_ref[ref_ind], x_sync[sync_ind]],
                    y=[y_ref[ref_ind], y_sync[sync_ind]],
                    z=[z_ref[ref_ind], z_sync[sync_ind]],
                    mode='lines+markers',
                    name= 'match lines ' + str(fr) +  ' cam'  + str(i),
                    line={'dash': 'dash'},
                    marker_size = 1,
                    marker_color=color_array_string[i]
                )
                data.append(trace1)
            
            trace1 = go.Scatter3d(
                x=x_sync,
                y=y_sync,
                z=z_sync,
                mode='markers',
                name= 'plane ' + str(i),
                marker_size = 3,
                marker_color=opacity_array[i]#color_array_string[i]
            )
            data.append(trace1)
            '''
            trace2 = go.Scatter3d(
                x=x_sync,
                y=y_sync,
                z=z_sync,
                mode='lines+markers',
                name= 'plane tracks ' + str(i),
                marker_size = 3,
                marker_color=opacity_array[i]#color_array_string[i]
            )
            data.append(trace2)
            '''
    #print(len(data), " data")
    #stop
    #create the coordinate list for the lines
    #print(len(cam_position_array), " JELLOASASD")
    #print(track_dict)
    
    for cam in track_dict.keys():
        for c in track_dict[cam].keys():
            #print(np.array(track_dict[cam][c]))
            #print(np.concatenate(track_dict[cam][c]).shape, cam, c, " SHAPEEEE")

            #print(track_dict[cam][c])
            track_full = np.concatenate(track_dict[cam][c])
            #print(track_full.shape, cam , c, " HIASDASASD")
            trace2 = go.Scatter3d(
                x=track_full[:, 0],
                y=track_full[:, 1],
                z=track_full[:, 2],
                mode='lines',
                name= "track" + ' ' + str(cam) + ' ' + str(c),
                marker=dict(
                    color=color_array_string[cam],
                )
            )
            '''
            trace2 = go.Scatter3d(
                x=np.array(track_dict[cam][c])[:, 0],
                y=np.array(track_dict[cam][c])[:, 1],
                z=np.array(track_dict[cam][c])[:, 2],
                mode='lines',
                name= "track" + ' ' + str(cam) + ' ' + str(c),
                marker=dict(
                    color=color_array_string[cam],
                )
            )
            '''
            data.append(trace2)
    
    '''
    for cam in track_dict.keys():
        for c in track_dict[cam].keys():

            #print(np.array(track_dict[cam][c]).shape, " SHAPEEEE")
            trace2 = go.Scatter3d(
                x=np.array(track_dict[cam][c])[:, 0, 0],
                y=np.array(track_dict[cam][c])[:, 0, 1],
                z=np.array(track_dict[cam][c])[:, 0, 2],
                mode='lines',
                name= str(cam) + ' ' + str(c),
                marker=dict(
                    color=color_array_string[cam],
                )
            )
            data.append(trace2)
    '''
    for a in range(len(cam_position_array)):
        cam_position = cam_position_array[a]
        #print(cam_position[a], " cam position !!!")
        #stop
        cam_axis = cam_axis_array[a]
        for p in range(len(cam_position)):

            trace2 = go.Scatter3d(
                x=[cam_position[p][0],cam_position[p][0] + scale*cam_axis[p][0][0]],
                y=[cam_position[p][1],cam_position[p][1] + scale*cam_axis[p][0][1]],
                z=[cam_position[p][2],cam_position[p][2] + scale*cam_axis[p][0][2]],
                mode='lines',
                name= name[a] + " axis " + str(p),
                marker=dict(
                    color='red',
                )
            )

            trace3 = go.Scatter3d(
                x=[cam_position[p][0],cam_position[p][0] + scale*cam_axis[p][1][0]],
                y=[cam_position[p][1],cam_position[p][1] + scale*cam_axis[p][1][1]],
                z=[cam_position[p][2],cam_position[p][2] + scale*cam_axis[p][1][2]],
                mode='lines',
                name= name[a] + " axis " + str(p),
                marker=dict(
                    color='green',
                )
            )

            trace4 = go.Scatter3d(
                x=[cam_position[p][0],cam_position[p][0] + scale*cam_axis[p][2][0]],
                y=[cam_position[p][1],cam_position[p][1] + scale*cam_axis[p][2][1]],
                z=[cam_position[p][2],cam_position[p][2] + scale*cam_axis[p][2][2]],
                mode='lines',
                name= name[a] + " axis " + str(p),
                marker=dict(
                    color='blue',
                )
            )

            trace5 = go.Scatter3d(
                x=[cam_position[p][0]],
                y=[cam_position[p][1]],
                z=[cam_position[p][2]],
                mode='markers+text',
                name =  name[a] + " " + str(p),
                marker=dict(
                    color='black',
                    size = 0.1,
                ),
                text = name[a] + " " + str(p),
                textposition='top right',
                textfont=dict(color='black')
            )

            data.append(trace2)
            data.append(trace3)
            data.append(trace4)
            data.append(trace5)
    
    fig = go.Figure(data=data)

    fig.update_layout(scene = dict( aspectmode='data'))

    fig.update_layout(title_text=title)
    fig.show()
    fig.write_html(run_name + '/' +  title + '_' + str(frame_init) + '_' + str(frame_end) + '_' + '.html')
    return

#FINISH PLOTTING THE TRACKS !!!!
#frame_init frame_number
def generate_3d_plotly_matches(cam_axis_array, cam_position_array, all_points_array,center_array, run_name, scale = 1000, title = 'plot', name = ['pred'], frame_init = 0 ,frame_end = 500):
    
    #draw a square
    frame_number = frame_end - frame_init 
    color_array = ['rgba(200,10,10,', 'rgba(10,200,10,', 'rgba(10,10,200,', 'rgba(0,0,0,']

    color_array_string = ['red', 'green', 'blue', 'black']
    data = []

    #for f in range(0, len(list(all_points_array[0].keys())), 10):
    #######################################################
    track_array = []

    track_dict = {}
    for a in range(len(cam_position_array[0])):
        track_dict[a] = {}
    
    #print(track_dict)

    opacity_array = []
    
    for i in range(len(color_array)):
        op = []
        for j in range(0, frame_number):
            op.append(color_array[i] + str(float(1 - np.exp(-4*j/(frame_number)))) + ')')
        opacity_array.append(op)

    #for f in range(0, frame_number):
    for f in range(frame_init, frame_end):
        print(f)
        fr = list(all_points_array[0].keys())[f]
        #print(all_points_array)
        ########################################
        all_points_ref = all_points_array[0]

        track_array.append(all_points_ref)

        for k in all_points_ref[fr].keys():

            if k in track_dict[0].keys():
                track_dict[0][k].append(list(all_points_ref[fr][k]))
            else:
                track_dict[0][k] = []
                track_dict[0][k].append(list(all_points_ref[fr][k]))

        #print(np.array(list(all_points_ref[fr].values())).shape, " SHAOEEEE")
        x_ref = np.array(list(all_points_ref[fr].values()))[0,:, 0]#[0, 1, 0, 1, 0, 1, 0, 1]
        y_ref = np.array(list(all_points_ref[fr].values()))[0,:, 1]#[0, 1, 1, 0, 0, 1, 1, 0]
        z_ref = np.array(list(all_points_ref[fr].values()))[0,:, 2]#[0, 0, 0, 0, 1, 1, 1, 1]

        trace1 = go.Scatter3d(
            x=x_ref,
            y=y_ref,
            z=z_ref,
            mode='markers',
            name= 'plane ' + str(0),
            marker_size = 3,
            marker_color=opacity_array[0]#color_array_string[0]
        )
        data.append(trace1)

        trace2 = go.Scatter3d(
            x=x_ref,
            y=y_ref,
            z=z_ref,
            mode='lines+markers',
            name= 'plane tracks ' + str(0),
            marker_size = 3,
            marker_color=opacity_array[0]#color_array_string[0]
        )
        data.append(trace2)
        
        ########################################
        for i in range(1, len(all_points_array)):
  
            #print(all_points_array[i].keys())
            if fr not in all_points_array[i].keys():
                continue

            for c in all_points_array[i][fr].keys():
    
                if c in track_dict[i].keys():
                    track_dict[i][c].append(list(all_points_array[i][fr][c]))
                else:
                    track_dict[i][c] = []
                    track_dict[i][c].append(list(all_points_array[i][fr][c]))

            all_points = all_points_array[i]
            #print(all_points_ref)

            points_ref = np.array(list(all_points_ref[fr].values()))[0,:,:2]
            points = np.array(list(all_points[fr].values()))[0,:,:2]
            points_center_ref = np.transpose(np.array(list(center_array[0]))[:2])
            points_center = np.transpose(np.array(list(center_array[i]))[:2])
            #print(points_center.shape, points_ref.shape, points.shape, " points_centerpoints_center")
            #indices = Icp2d.match(all_points_ref[fr][:,:2], all_points[fr][:,:2], center_array[0][:2],  center_array[i][:2])  
            indices = Icp2d.match(points_ref, points, points_center_ref,  points_center)  
            #print(len(indices), " heloasdasdddd")

            x_sync = np.array(list(all_points[fr].values()))[0,:, 0]#[0, 1, 0, 1, 0, 1, 0, 1]
            y_sync = np.array(list(all_points[fr].values()))[0,:, 1]#[0, 1, 1, 0, 0, 1, 1, 0]
            z_sync = np.array(list(all_points[fr].values()))[0,:, 2]#[0, 0, 0, 0, 1, 1, 1, 1]

            for ind in range(len(indices)):
                
                sync_ind = indices[ind]

                #print([x_ref[ind], x_sync[sync_ind]])
                #print([y_ref[ind], y_sync[sync_ind]])
                #print([z_ref[ind], z_sync[sync_ind]], " ZZZZ")
                
                trace1 = go.Scatter3d(
                    x=[x_ref[ind], x_sync[sync_ind][0]],
                    y=[y_ref[ind], y_sync[sync_ind][0]],
                    z=[z_ref[ind], z_sync[sync_ind][0]],
                    mode='lines+markers',
                    name= 'match lines ' + str(fr) +  ' cam'  + str(i),
                    line={'dash': 'dash'},
                    marker_size = 1,
                    marker_color=color_array_string[i]
                )
                data.append(trace1)

            trace1 = go.Scatter3d(
                x=x_sync,
                y=y_sync,
                z=z_sync,
                mode='markers',
                name= 'plane ' + str(i),
                marker_size = 3,
                marker_color=opacity_array[i]#color_array_string[i]
            )
            data.append(trace1)

            trace2 = go.Scatter3d(
                x=x_sync,
                y=y_sync,
                z=z_sync,
                mode='lines+markers',
                name= 'plane tracks ' + str(i),
                marker_size = 3,
                marker_color=opacity_array[i]#color_array_string[i]
            )
            data.append(trace2)

    #print(len(data), " data")
    #stop
    #create the coordinate list for the lines
    #print(len(cam_position_array), " JELLOASASD")
    
    for cam in track_dict.keys():
        for c in track_dict[cam].keys():

            #print(np.array(track_dict[cam][c]).shape, " SHAPEEEE")
            trace2 = go.Scatter3d(
                x=np.array(track_dict[cam][c])[:, 0, 0],
                y=np.array(track_dict[cam][c])[:, 0, 1],
                z=np.array(track_dict[cam][c])[:, 0, 2],
                mode='lines',
                name= str(cam) + ' ' + str(c),
                marker=dict(
                    color=color_array_string[cam],
                )
            )
            data.append(trace2)
    
    for a in range(len(cam_position_array)):
        cam_position = cam_position_array[a]
        #print(cam_position[a], " cam position !!!")
        #stop
        cam_axis = cam_axis_array[a]
        for p in range(len(cam_position)):

            trace2 = go.Scatter3d(
                x=[cam_position[p][0],cam_position[p][0] + scale*cam_axis[p][0][0]],
                y=[cam_position[p][1],cam_position[p][1] + scale*cam_axis[p][0][1]],
                z=[cam_position[p][2],cam_position[p][2] + scale*cam_axis[p][0][2]],
                mode='lines',
                name= name[a] + " axis " + str(p),
                marker=dict(
                    color='red',
                )
            )

            trace3 = go.Scatter3d(
                x=[cam_position[p][0],cam_position[p][0] + scale*cam_axis[p][1][0]],
                y=[cam_position[p][1],cam_position[p][1] + scale*cam_axis[p][1][1]],
                z=[cam_position[p][2],cam_position[p][2] + scale*cam_axis[p][1][2]],
                mode='lines',
                name= name[a] + " axis " + str(p),
                marker=dict(
                    color='green',
                )
            )

            trace4 = go.Scatter3d(
                x=[cam_position[p][0],cam_position[p][0] + scale*cam_axis[p][2][0]],
                y=[cam_position[p][1],cam_position[p][1] + scale*cam_axis[p][2][1]],
                z=[cam_position[p][2],cam_position[p][2] + scale*cam_axis[p][2][2]],
                mode='lines',
                name= name[a] + " axis " + str(p),
                marker=dict(
                    color='blue',
                )
            )

            trace5 = go.Scatter3d(
                x=[cam_position[p][0]],
                y=[cam_position[p][1]],
                z=[cam_position[p][2]],
                mode='markers+text',
                name =  name[a] + " " + str(p),
                marker=dict(
                    color='black',
                    size = 0.1,
                ),
                text = name[a] + " " + str(p),
                textposition='top right',
                textfont=dict(color='black')
            )

            data.append(trace2)
            data.append(trace3)
            data.append(trace4)
            data.append(trace5)
    
    fig = go.Figure(data=data)

    fig.update_layout(scene = dict( aspectmode='data'))

    fig.update_layout(title_text=title)
    fig.show()
    fig.write_html(run_name + '/' +  title + '_' + str(frame_init) + '_' + str(frame_end) + '_' + '.html')
    return

def generate_3d_plotly_matches_animate(cam_axis_array, cam_position_array, all_points_array, run_name, scale = 1000, title = 'plot', name = ['pred']):
    
    color_array = ['rgba(200,10,10,', 'rgba(10,200,10,', 'rgba(10,10,200,', 'rgba(0,0,0,']
    
    #for i in range(len(all_points_array)):
    all_points = np.array(all_points_array[0])
    #print(all_points.shape, " all_points")
    x = all_points[:, 0]#[0, 1, 0, 1, 0, 1, 0, 1]
    y = all_points[:, 1]#[0, 1, 1, 0, 0, 1, 1, 0]
    z = all_points[:, 2]#[0, 0, 0, 0, 1, 1, 1, 1]

    #spline_model.bs_spline(time, space, t,k, save_dir, name = '')
    #the start and end point for each line
    #pairs = [(0,6), (1,7)]

    print(x.shape, " HELOASDASDASDDDDDDDD")
    print(len(x), " HELOASDASDASDDDDDDDD")

    #stop
    frames = []
    for i in range(len(x)): 
        data = []
        trace1 = go.Scatter3d(
            x=[x[i]],
            y=[y[i]],
            z=[z[i]],
            mode='markers',
            name= 'plane ' + str(i),
            marker_size = 1,
            marker_color='red'
        )
        data.append(trace1)

        #create the coordinate list for the lines
        #print(len(cam_position_array), " JELLOASASD")
        for a in range(len(cam_position_array)):
            cam_position = cam_position_array[a]
            cam_axis = cam_axis_array[a]
            for p in range(len(cam_position)):

                trace2 = go.Scatter3d(
                    x=[cam_position[p][0],cam_position[p][0] + scale*cam_axis[p][0][0]],
                    y=[cam_position[p][1],cam_position[p][1] + scale*cam_axis[p][0][1]],
                    z=[cam_position[p][2],cam_position[p][2] + scale*cam_axis[p][0][2]],
                    mode='lines',
                    name= name[a] + " axis " + str(p),
                    marker=dict(
                        color='red',
                    )
                )

                trace3 = go.Scatter3d(
                    x=[cam_position[p][0],cam_position[p][0] + scale*cam_axis[p][1][0]],
                    y=[cam_position[p][1],cam_position[p][1] + scale*cam_axis[p][1][1]],
                    z=[cam_position[p][2],cam_position[p][2] + scale*cam_axis[p][1][2]],
                    mode='lines',
                    name= name[a] + " axis " + str(p),
                    marker=dict(
                        color='green',
                    )
                )

                trace4 = go.Scatter3d(
                    x=[cam_position[p][0],cam_position[p][0] + scale*cam_axis[p][2][0]],
                    y=[cam_position[p][1],cam_position[p][1] + scale*cam_axis[p][2][1]],
                    z=[cam_position[p][2],cam_position[p][2] + scale*cam_axis[p][2][2]],
                    mode='lines',
                    name= name[a] + " axis " + str(p),
                    marker=dict(
                        color='blue',
                    )
                )

                trace5 = go.Scatter3d(
                    x=[cam_position[p][0]],
                    y=[cam_position[p][1]],
                    z=[cam_position[p][2]],
                    mode='markers+text',
                    name =  name[a] + " " + str(p),
                    marker=dict(
                        color='black',
                        size = 0.1,
                    ),
                    text = name[a] + " " + str(p),
                    textposition='top right',
                    textfont=dict(color='black')
                )

                data.append(trace2)
                data.append(trace3)
                data.append(trace4)
                data.append(trace5)
        frames.append(go.Frame(data = data))
        
    fig = go.Figure(
        data=[
            go.Scatter3d(
            )
        for traces in frames],
        layout=go.Layout( # Styling
            scene=dict(
            ),
            updatemenus=[
                dict(
                    type='buttons',
                    buttons=[
                        dict(
                            label='Play',
                            method='animate',
                            args=[None]
                        )
                    ]
                )
            ]
        ),
        frames=frames
    )
    fig.update_layout(scene = dict( aspectmode='data'))

    #for k, f in enumerate(fig.frames):
    #    print(k, f, " fig frameeee")
    def frame_args(duration):
            return {
                "frame": {"duration": duration},
                "mode": "immediate",
                "fromcurrent": True,
                "transition": {"duration": duration, "easing": "linear"},
            }

    sliders = [
                {
                    "pad": {"b": 10, "t": 60},
                    "len": 0.9,
                    "x": 0.1,
                    "y": 0,
                    "steps": [
                        {
                            "args": [[f.name], frame_args(0)],
                            "label": str(k),
                            "method": "animate",
                        }
                        for k, f in enumerate(fig.frames)
                    ],
                }
            ]

    # Layout
    fig.update_layout(
            title='Slices in volumetric data',
            width=600,
            height=600,
            scene=dict(
                        zaxis=dict(range=[-0.1, 6.8], autorange=False),
                        aspectratio=dict(x=1, y=1, z=1),
                        ),
            updatemenus = [
                {
                    "buttons": [
                        {
                            "args": [None, frame_args(50)],
                            "label": "&#9654;", # play symbol
                            "method": "animate",
                        },
                        {
                            "args": [[None], frame_args(0)],
                            "label": "&#9724;", # pause symbol
                            "method": "animate",
                        },
                    ],
                    "direction": "left",
                    "pad": {"r": 10, "t": 70},
                    "type": "buttons",
                    "x": 0.1,
                    "y": 0,
                }
            ],
            sliders=sliders
    )

    fig.show()
    fig.write_html(run_name + '/' +  title + '_animation_.html', include_plotlyjs='cdn')

    return

def generate_3d_plotly_tracks(cam_axis_array, cam_position_array, all_points_array, run_name, scale = 1000, title = 'plot', name = ['pred']):
    
    #draw a square

    color_array = ['rgba(200,10,10,', 'rgba(10,200,10,', 'rgba(10,10,200,', 'rgba(0,0,0,']
    data = []
    for i in range(len(all_points_array)):
        all_points = np.array(all_points_array[i])
        #print(all_points.shape, " all_points")
        x = all_points[:, 0]#[0, 1, 0, 1, 0, 1, 0, 1]
        y = all_points[:, 1]#[0, 1, 1, 0, 0, 1, 1, 0]
        z = all_points[:, 2]#[0, 0, 0, 0, 1, 1, 1, 1]

        #spline_model.bs_spline(time, space, t,k, save_dir, name = '')
        #the start and end point for each line
        #pairs = [(0,6), (1,7)]

        opacity_array = []
        dict_array = []
        print(x.shape, " HELOASDASDASDDDDDDDD")
        print(len(x), " HELOASDASDASDDDDDDDD")

        for j in range(len(x)):
                #opacity= 1 - np.exp(-j)
                #print(j, " THIS IS J")
                color_alpha = color_array[i] + str(float(1 - np.exp(-4*j/(len(x))))) + ')'
                print(color_alpha, " HELOO")
                opacity_array.append(color_alpha)
        #stop
        '''
        trace1 = go.Scatter3d(
            x=x,
            y=y,
            z=z,
            mode='markers',
            name= 'plane ' + str(i),
            marker_size = 1,
            marker_color=opacity_array
        )
        data.append(trace1)
        '''
        trace_lines = go.Scatter3d(
            x=x,
            y=y,
            z=z,
            mode='lines+markers',
            name= 'lines ' + str(i),
            marker_size = 1,
            marker_color=opacity_array,
            line_color=opacity_array

        )
        
        data.append(trace_lines)

    #create the coordinate list for the lines
    print(len(cam_position_array), " JELLOASASD")
    for a in range(len(cam_position_array)):
        cam_position = cam_position_array[a]
        cam_axis = cam_axis_array[a]
        for p in range(len(cam_position)):
            '''
            print(cam_position[p].shape)
            print(cam_position[p][0], "asadsaaasacacczaafsasascaxas")
            print(cam_position[p][1], "asadsaaasacacczaafsasascaxas")
            print(cam_position[p][2], "asadsaaasacacczaafsasascaxas")
            print(cam_axis[p][0][0])
            print(cam_axis[p][0][1], "1212121ff")
            print(cam_axis[p][0][2])
            '''
            trace2 = go.Scatter3d(
                x=[cam_position[p][0],cam_position[p][0] + scale*cam_axis[p][0][0]],
                y=[cam_position[p][1],cam_position[p][1] + scale*cam_axis[p][0][1]],
                z=[cam_position[p][2],cam_position[p][2] + scale*cam_axis[p][0][2]],
                mode='lines',
                name= name[a] + " axis " + str(p),
                marker=dict(
                    color='red',
                )
            )

            trace3 = go.Scatter3d(
                x=[cam_position[p][0],cam_position[p][0] + scale*cam_axis[p][1][0]],
                y=[cam_position[p][1],cam_position[p][1] + scale*cam_axis[p][1][1]],
                z=[cam_position[p][2],cam_position[p][2] + scale*cam_axis[p][1][2]],
                mode='lines',
                name= name[a] + " axis " + str(p),
                marker=dict(
                    color='green',
                )
            )

            trace4 = go.Scatter3d(
                x=[cam_position[p][0],cam_position[p][0] + scale*cam_axis[p][2][0]],
                y=[cam_position[p][1],cam_position[p][1] + scale*cam_axis[p][2][1]],
                z=[cam_position[p][2],cam_position[p][2] + scale*cam_axis[p][2][2]],
                mode='lines',
                name= name[a] + " axis " + str(p),
                marker=dict(
                    color='blue',
                )
            )

            trace5 = go.Scatter3d(
                x=[cam_position[p][0]],
                y=[cam_position[p][1]],
                z=[cam_position[p][2]],
                mode='markers+text',
                name =  name[a] + " " + str(p),
                marker=dict(
                    color='black',
                    size = 0.1,
                ),
                text = name[a] + " " + str(p),
                textposition='top right',
                textfont=dict(color='black')
            )

            data.append(trace2)
            data.append(trace3)
            data.append(trace4)
            data.append(trace5)

    fig = go.Figure(data=data)

    fig.update_layout(scene = dict( aspectmode='data'))

    fig.update_layout(title_text=title)
    fig.show()
    fig.write_html(run_name + '/' +  title + '.html')
    return

def generate_3d_plotly_multiple_planes(cam_axis_array, cam_position_array, all_points_array, run_name, scale = 1000, title = 'plot', name = ['pred']):
    
    #draw a square

    color_array = ['red', 'blue', 'green', 'black']
    data = []
    #print(all_points_array.shape, " SHAPE")
    for i in range(len(all_points_array)):
        print(i, " HELOASDASDASDASDASDASDASDASD")
        all_points = np.array(all_points_array[i])

        print(all_points.shape, " HEASLOADSASDASDASDASDASDDASDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDD")
        #print(all_points.shape, " all_points")
        x = all_points[:, 0]#[0, 1, 0, 1, 0, 1, 0, 1]
        y = all_points[:, 1]#[0, 1, 1, 0, 0, 1, 1, 0]
        z = all_points[:, 2]#[0, 0, 0, 0, 1, 1, 1, 1]

        #the start and end point for each line
        #pairs = [(0,6), (1,7)]

        # mapping
        #index_list = list(map(str, list(range(all_points.shape[0]))))
        #print(index_list, "indexxxx")
        trace1 = go.Scatter3d(
            x=x,
            y=y,
            z=z,
            #hovertext=index_list,
            #hoverinfo="text",
            mode='markers',
            name= 'plane ' + str(i),
            marker=dict(
                    color=color_array[i],
                    size=1,
                )
        )

        data.append(trace1)

    #create the coordinate list for the lines
    print(len(cam_position_array), " JELLOASASD")
    for a in range(len(cam_position_array)):
        cam_position = cam_position_array[a]
        cam_axis = cam_axis_array[a]
        for p in range(len(cam_position)):
            '''
            print(cam_position[p].shape)
            print(cam_position[p][0], "asadsaaasacacczaafsasascaxas")
            print(cam_position[p][1], "asadsaaasacacczaafsasascaxas")
            print(cam_position[p][2], "asadsaaasacacczaafsasascaxas")
            print(cam_axis[p][0][0])
            print(cam_axis[p][0][1], "1212121ff")
            print(cam_axis[p][0][2])
            '''
            trace2 = go.Scatter3d(
                x=[cam_position[p][0],cam_position[p][0] + scale*cam_axis[p][0][0]],
                y=[cam_position[p][1],cam_position[p][1] + scale*cam_axis[p][0][1]],
                z=[cam_position[p][2],cam_position[p][2] + scale*cam_axis[p][0][2]],
                mode='lines',
                name= name[a] + " axis " + str(p),
                marker=dict(
                    color='red',
                )
            )

            trace3 = go.Scatter3d(
                x=[cam_position[p][0],cam_position[p][0] + scale*cam_axis[p][1][0]],
                y=[cam_position[p][1],cam_position[p][1] + scale*cam_axis[p][1][1]],
                z=[cam_position[p][2],cam_position[p][2] + scale*cam_axis[p][1][2]],
                mode='lines',
                name= name[a] + " axis " + str(p),
                marker=dict(
                    color='green',
                )
            )

            trace4 = go.Scatter3d(
                x=[cam_position[p][0],cam_position[p][0] + scale*cam_axis[p][2][0]],
                y=[cam_position[p][1],cam_position[p][1] + scale*cam_axis[p][2][1]],
                z=[cam_position[p][2],cam_position[p][2] + scale*cam_axis[p][2][2]],
                mode='lines',
                name= name[a] + " axis " + str(p),
                marker=dict(
                    color='blue',
                )
            )

            trace5 = go.Scatter3d(
                x=[cam_position[p][0]],
                y=[cam_position[p][1]],
                z=[cam_position[p][2]],
                mode='markers+text',
                name =  name[a] + " " + str(p),
                marker=dict(
                    color='black',
                    size = 0.1,
                ),
                text = name[a] + " " + str(p),
                textposition='top right',
                textfont=dict(color='black')
            )

            data.append(trace2)
            data.append(trace3)
            data.append(trace4)
            data.append(trace5)

    fig = go.Figure(data=data)

    fig.update_layout(scene = dict( aspectmode='data'))

    fig.update_layout(title_text=title)
    fig.show()
    fig.write_html(run_name + '/' +  title + '.html')
    return

def generate_3d_plotly_points_overlay(cam_axis_array, cam_position_array, all_points_array, run_name, scale = 1000, name = ["cam"], title = 'plot'):
    
    data = []

    color_array = ['green', 'red']
    #draw a square
    for i in range(len(all_points_array)):

        all_points = all_points_array[i]
        all_points = np.array(all_points)
        #print(all_points.shape, " all_points")
        x = all_points[:, 0]#[0, 1, 0, 1, 0, 1, 0, 1]
        y = all_points[:, 1]#[0, 1, 1, 0, 0, 1, 1, 0]
        z = all_points[:, 2]#[0, 0, 0, 0, 1, 1, 1, 1]

        #the start and end point for each line
        #pairs = [(0,6), (1,7)]

        trace1 = go.Scatter3d(
            x=x,
            y=y,
            z=z,
            mode='markers',
            name='markers',
            marker=dict(
                    color=color_array[i],
                    size=1,
                )
        )

        data.append(trace1)

    #create the coordinate list for the lines
    #data = [trace1]
    
    for c in range(len(cam_axis_array)):
        
        cam_axis = cam_axis_array[c]
        cam_position = cam_position_array[c]
        for p in range(len(cam_position)):

            print(cam_position[p].shape)
            print(cam_position[p][0], "asadsaaasacacczaafsasascaxas")
            print(cam_position[p][1], "asadsaaasacacczaafsasascaxas")
            print(cam_position[p][2], "asadsaaasacacczaafsasascaxas")
            print(cam_axis[p][0][0])
            print(cam_axis[p][0][1], "1212121ff")
            print(cam_axis[p][0][2])
            trace2 = go.Scatter3d(
                x=[cam_position[p][0],cam_position[p][0] + scale*cam_axis[p][0][0]],
                y=[cam_position[p][1],cam_position[p][1] + scale*cam_axis[p][0][1]],
                z=[cam_position[p][2],cam_position[p][2] + scale*cam_axis[p][0][2]],
                mode='lines',
                name= name[c] + " " + str(p),
                marker=dict(
                    color='red',
                )
            )

            trace3 = go.Scatter3d(
                x=[cam_position[p][0],cam_position[p][0] + scale*cam_axis[p][1][0]],
                y=[cam_position[p][1],cam_position[p][1] + scale*cam_axis[p][1][1]],
                z=[cam_position[p][2],cam_position[p][2] + scale*cam_axis[p][1][2]],
                mode='lines',
                name= name[c] + " " + str(p),
                marker=dict(
                    color='green',
                )
            )

            trace4 = go.Scatter3d(
                x=[cam_position[p][0],cam_position[p][0] + scale*cam_axis[p][2][0]],
                y=[cam_position[p][1],cam_position[p][1] + scale*cam_axis[p][2][1]],
                z=[cam_position[p][2],cam_position[p][2] + scale*cam_axis[p][2][2]],
                mode='lines',
                name= name[c] + " " + str(p),
                marker=dict(
                    color='blue',
                )
            )


            trace5 = go.Scatter3d(
                x=[cam_position[p][0]],
                y=[cam_position[p][1]],
                z=[cam_position[p][2]],
                mode='markers+text',
                name= name[c] + " " + str(p),
                marker=dict(
                    color='black',
                    size = 0.1,
                ),
                text=name[c] + " " + str(p),
                textposition='top right',
                textfont=dict(color='black')
            )

            data.append(trace2)
            data.append(trace3)
            data.append(trace4)
            data.append(trace5)

    fig = go.Figure(data=data)

    fig.update_layout(scene = dict( aspectmode='data'))

    fig.update_layout(title_text=title)
    fig.show()
    fig.write_html(run_name + '/' +  title + '.html')
    return

def generate_3d_plotly_points(cam_axis, cam_position, all_points, run_name, scale = 1000, title = 'plot'):
    
    #draw a square
    all_points = np.array(all_points)
    #print(all_points.shape, " all_points")
    x = all_points[:, 0]#[0, 1, 0, 1, 0, 1, 0, 1]
    y = all_points[:, 1]#[0, 1, 1, 0, 0, 1, 1, 0]
    z = all_points[:, 2]#[0, 0, 0, 0, 1, 1, 1, 1]

    #the start and end point for each line
    #pairs = [(0,6), (1,7)]

    trace1 = go.Scatter3d(
        x=x,
        y=y,
        z=z,
        mode='markers',
        name='markers',
        marker=dict(
                color='blue',
                size=1,
            )
    )

    #create the coordinate list for the lines
    data = [trace1]
    
    for p in range(len(cam_position)):

        print(cam_position[p].shape)
        print(cam_position[p][0], "asadsaaasacacczaafsasascaxas")
        print(cam_position[p][1], "asadsaaasacacczaafsasascaxas")
        print(cam_position[p][2], "asadsaaasacacczaafsasascaxas")
        print(cam_axis[p][0][0])
        print(cam_axis[p][0][1], "1212121ff")
        print(cam_axis[p][0][2])
        trace2 = go.Scatter3d(
            x=[cam_position[p][0],cam_position[p][0] + scale*cam_axis[p][0][0]],
            y=[cam_position[p][1],cam_position[p][1] + scale*cam_axis[p][0][1]],
            z=[cam_position[p][2],cam_position[p][2] + scale*cam_axis[p][0][2]],
            mode='lines',
            name= "cam " + str(p),
            marker=dict(
                color='red',
            )
        )

        trace3 = go.Scatter3d(
            x=[cam_position[p][0],cam_position[p][0] + scale*cam_axis[p][1][0]],
            y=[cam_position[p][1],cam_position[p][1] + scale*cam_axis[p][1][1]],
            z=[cam_position[p][2],cam_position[p][2] + scale*cam_axis[p][1][2]],
            mode='lines',
            name= "cam " + str(p),
            marker=dict(
                color='green',
            )
        )

        trace4 = go.Scatter3d(
            x=[cam_position[p][0],cam_position[p][0] + scale*cam_axis[p][2][0]],
            y=[cam_position[p][1],cam_position[p][1] + scale*cam_axis[p][2][1]],
            z=[cam_position[p][2],cam_position[p][2] + scale*cam_axis[p][2][2]],
            mode='lines',
            name= "cam " + str(p),
            marker=dict(
                color='blue',
            )
        )

        trace5 = go.Scatter3d(
            x=[cam_position[p][0]],
            y=[cam_position[p][1]],
            z=[cam_position[p][2]],
            mode='markers+text',
            name =  "cam " + str(p),
            marker=dict(
                color='black',
                size = 0.1,
            ),
            text="cam " + str(p),
            textposition='top right',
            textfont=dict(color='black')
        )

        data.append(trace2)
        data.append(trace3)
        data.append(trace4)
        data.append(trace5)

    fig = go.Figure(data=data)

    fig.update_layout(scene = dict( aspectmode='data'))

    fig.update_layout(title_text=title)
    fig.show()
    fig.write_html(run_name + '/' +  title + '.html')
    return

def generate_3d_plotly_animation(view_calibration, extrinsic_array, points_2d_dict, run_name):

    nb_frames = len(extrinsic_array)

    all_points = []
    for i in range(len(points_2d_dict)):
        cam_points = []
        for j in list(points_2d_dict[i].keys()):
            for k in list(points_2d_dict[i][j].keys()):
                #print(points_2d_dict[i][j][k], i, j)
                cam_points.append([points_2d_dict[i][j][k][0], points_2d_dict[i][j][k][1], 1.0, 1.0])
        
        all_points.append(cam_points)
    
    color_array = ['red', 'green', 'blue', 'black']

    #print(torch.tensor((extrinsic_array)).shape, " EXTRINSIC")
    #stop
    frames = []
    for i in range(nb_frames):
        frame_data = []

        for j in range(len(extrinsic_array[i])):
            #print(j, " JJJJJJJ")
            #print(view_calibration[j], j, " view_calibration[j]")
            cam_sync = torch.tensor([[view_calibration[j]['cam_matrix'][0][0], view_calibration[j]['cam_matrix'][0][1], view_calibration[j]['cam_matrix'][0][2], 0], [view_calibration[j]['cam_matrix'][1][0], view_calibration[j]['cam_matrix'][1][1], view_calibration[j]['cam_matrix'][1][2], 0], [view_calibration[j]['cam_matrix'][2][0], view_calibration[j]['cam_matrix'][2][1], view_calibration[j]['cam_matrix'][2][2], 0], [0,0,0,1]])

            center = extrinsic_array[i][j][:3, 3]

            axis_x = extrinsic_array[i][j][:3, 0]
            axis_y = extrinsic_array[i][j][:3, 1]
            axis_z = extrinsic_array[i][j][:3, 2]
            #print(intrinsic_array , "HELOOOOO INTRINSIC")
            points_3d = torch.transpose(bundle_intersect.plane_intersect(torch.transpose(torch.tensor(all_points[j]).double(), 0, 1), torch.tensor(center).double(), torch.inverse(cam_sync).double(), torch.tensor(extrinsic_array[i][j]).double()), 0, 1).numpy()

            #print(points_3d.shape, " POINT 3D !!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

            
            for pt in range(0, points_3d.shape[0], 2000):
                frame_data.append(
                    go.Scatter3d(
                        x=[points_3d[pt][0]],
                        y=[points_3d[pt][1]],
                        z=[points_3d[pt][2]],
                        #name=str(i),
                        marker=dict(
                            color=color_array[j],
                            size=1,
                        )
                    )
                )
            

            frame_data.append(
                go.Scatter3d(
                    x=[center[0],center[0] + axis_x[0]],
                    y=[center[1],center[1] + axis_x[1]],
                    z=[center[2],center[2] + axis_x[2]],
                    mode='lines',
                    name=str(i)
                )
            )

            frame_data.append(
            go.Scatter3d(
                    x=[center[0],center[0] + axis_y[0]],
                    y=[center[1],center[1] + axis_y[1]],
                    z=[center[2],center[2] + axis_y[2]],
                    mode='lines',
                    name=str(i)
                )
            )

            frame_data.append(
            go.Scatter3d(
                    x=[center[0],center[0] + axis_z[0]],
                    y=[center[1],center[1] + axis_z[1]],
                    z=[center[2],center[2] + axis_z[2]],
                    mode='lines',
                    name=str(i)
                )
            )
        frames.append(go.Frame(data = frame_data))

    # Defining figure
    fig = go.Figure(
        data=[
            go.Scatter3d(
            )
        for traces in frames],
        layout=go.Layout( # Styling
            scene=dict(
            ),
            updatemenus=[
                dict(
                    type='buttons',
                    buttons=[
                        dict(
                            label='Play',
                            method='animate',
                            args=[None]
                        )
                    ]
                )
            ]
        ),
        frames=frames
    )
    fig.update_layout(scene = dict( aspectmode='data'))
    '''
    fig = go.Figure(
        data=[
            go.Scatter3d(
            )
        ],
        layout=go.Layout( # Styling
            scene=dict(
            ),
            updatemenus=[
                dict(
                    type='buttons',
                    buttons=[
                        dict(
                            label='Play',
                            method='animate',
                            args=[None]
                        )
                    ]
                )
            ]
        ),
        frames=frames
    )
    '''

    #for k, f in enumerate(fig.frames):
    #    print(k, f, " fig frameeee")
    def frame_args(duration):
            return {
                "frame": {"duration": duration},
                "mode": "immediate",
                "fromcurrent": True,
                "transition": {"duration": duration, "easing": "linear"},
            }

    sliders = [
                {
                    "pad": {"b": 10, "t": 60},
                    "len": 0.9,
                    "x": 0.1,
                    "y": 0,
                    "steps": [
                        {
                            "args": [[f.name], frame_args(0)],
                            "label": str(k),
                            "method": "animate",
                        }
                        for k, f in enumerate(fig.frames)
                    ],
                }
            ]

    # Layout
    fig.update_layout(
            title='Slices in volumetric data',
            width=600,
            height=600,
            scene=dict(
                        zaxis=dict(range=[-0.1, 6.8], autorange=False),
                        aspectratio=dict(x=1, y=1, z=1),
                        ),
            updatemenus = [
                {
                    "buttons": [
                        {
                            "args": [None, frame_args(50)],
                            "label": "&#9654;", # play symbol
                            "method": "animate",
                        },
                        {
                            "args": [[None], frame_args(0)],
                            "label": "&#9724;", # pause symbol
                            "method": "animate",
                        },
                    ],
                    "direction": "left",
                    "pad": {"r": 10, "t": 70},
                    "type": "buttons",
                    "x": 0.1,
                    "y": 0,
                }
            ],
            sliders=sliders
    )

    fig.show()
    fig.write_html(run_name + '/bundle.html')

    return

def generate_3d_plotly(view_calibration, extrinsic_array, points_2d_dict):
    '''
    vol = io.imread("https://s3.amazonaws.com/assets.datacamp.com/blog_assets/attention-mri.tif")
    volume = vol.T

    print(volume, " VOLUME")
    print(volume.shape, " VOLUMEEEE !!!")
    print(volume[0].shape, " vol 0 shape")
    print(volume[1].shape,  " asdqweqwe")
    print(volume[2].shape, " shasspeaseaseasease")
    r, c = volume[0].shape
    '''
    r = 100 
    c = 100
    #print(len(intrinsic_array), "intrinsic_arrayintrinsic_array")
    # Define frames
    nb_frames = len(extrinsic_array)

    print(nb_frames, " NB FRAMES")
    '''
    layout = plotly.graph_objects.Layout(
             scene=dict(
                 scene_aspectmode='data'
         ))
    '''
    '''
    layout = go.Layout(
             scene=dict(
                 aspectmode='manual',
                 aspectratio={ "x": 1, "y": 1,"z": 1 }
         ))
    '''
    
    fig = go.Figure(
        frames=[go.Frame(data=go.Surface(
        z=np.zeros((r, c)),
        cmin=-100, cmax=100
        ),
        name=str(k), # you need to name the frame for the animation to behave properly
        )
        for k in range(nb_frames)])

    fig.update_layout(scene = dict( aspectmode='data'))
    
    # Add data to be displayed before animation starts
    df = px.data.iris()

    #print(points_2d_dict)

    all_points = []
    for i in range(len(points_2d_dict)):
        cam_points = []
        for j in list(points_2d_dict[i].keys()):
            for k in list(points_2d_dict[i][j].keys()):
                #print(points_2d_dict[i][j][k], i, j)
                cam_points.append([points_2d_dict[i][j][k][0], points_2d_dict[i][j][k][1], 1.0, 1.0])
        
        all_points.append(cam_points)
    
    color_array = ['red', 'green', 'blue', 'black']

    #print(torch.tensor((extrinsic_array)).shape, " EXTRINSIC")
    #stop
    for i in range(nb_frames):
        for j in range(len(extrinsic_array[i])):
            
            #print(view_calibration[j], j, " view_calibration[j]")
            cam_sync = torch.tensor([[view_calibration[j]['cam_matrix'][0][0], view_calibration[j]['cam_matrix'][0][1], view_calibration[j]['cam_matrix'][0][2], 0], [view_calibration[j]['cam_matrix'][1][0], view_calibration[j]['cam_matrix'][1][1], view_calibration[j]['cam_matrix'][1][2], 0], [view_calibration[j]['cam_matrix'][2][0], view_calibration[j]['cam_matrix'][2][1], view_calibration[j]['cam_matrix'][2][2], 0], [0,0,0,1]])

            center = extrinsic_array[i][j][:3, 3]

            axis_x = extrinsic_array[i][j][:3, 0]
            axis_y = extrinsic_array[i][j][:3, 1]
            axis_z = extrinsic_array[i][j][:3, 2]
            #print(intrinsic_array , "HELOOOOO INTRINSIC")
            points_3d = torch.transpose(bundle_intersect.plane_intersect(torch.transpose(torch.tensor(all_points[j]).double(), 0, 1), torch.tensor(center).double(), torch.inverse(cam_sync).double(), torch.tensor(extrinsic_array[i][j]).double()), 0, 1).numpy()

            #print(points_3d.shape, " POINT 3D !!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

            for pt in range(0, points_3d.shape[0], 1000):
                fig.add_trace(
                    go.Scatter3d(
                        x=[points_3d[pt][0]],
                        y=[points_3d[pt][1]],
                        z=[points_3d[pt][2]],
                        name=str(i),
                        marker=dict(
                            color=color_array[j],
                            size=1,
                        )
                    )
                )
            '''
            fig.add_trace(
                go.Scatter3d(
                    x=[0],
                    y=[0],
                    z=[20],
                    name=str(i)
                )
            )
            '''
            fig.add_trace(
                go.Scatter3d(
                    x=[center[0],center[0] + axis_x[0]],
                    y=[center[1],center[1] + axis_x[1]],
                    z=[center[2],center[2] + axis_x[2]],
                    mode='lines',
                    name=str(i)
                )
            )

            fig.add_trace(
            go.Scatter3d(
                    x=[center[0],center[0] + axis_y[0]],
                    y=[center[1],center[1] + axis_y[1]],
                    z=[center[2],center[2] + axis_y[2]],
                    mode='lines',
                    name=str(i)
                )
            )

            fig.add_trace(
            go.Scatter3d(
                    x=[center[0],center[0] + axis_z[0]],
                    y=[center[1],center[1] + axis_z[1]],
                    z=[center[2],center[2] + axis_z[2]],
                    mode='lines',
                    name=str(i)
                )
            )
    
    '''
    def frame_args(duration):
        return {
                "frame": {"duration": duration},
                "mode": "immediate",
                "fromcurrent": True,
                "transition": {"duration": duration, "easing": "linear"},
            }

    sliders = [
                {
                    "pad": {"b": 10, "t": 60},
                    "len": 0.9,
                    "x": 0.1,
                    "y": 0,
                    "steps": [
                        {
                            "args": [[f.name], frame_args(0)],
                            "label": str(k),
                            "method": "animate",
                        }
                        for k, f in enumerate(fig.frames)
                    ],
                }
            ]

    # Layout
    fig.update_layout(
            title='Slices in volumetric data',
            width=600,
            height=600,
            scene=dict(
                        zaxis=dict(range=[-0.1, 6.8], autorange=False),
                        aspectratio=dict(x=1, y=1, z=1),
                        ),
            updatemenus = [
                {
                    "buttons": [
                        {
                            "args": [None, frame_args(50)],
                            "label": "&#9654;", # play symbol
                            "method": "animate",
                        },
                        {
                            "args": [[None], frame_args(0)],
                            "label": "&#9724;", # pause symbol
                            "method": "animate",
                        },
                    ],
                    "direction": "left",
                    "pad": {"r": 10, "t": 70},
                    "type": "buttons",
                    "x": 0.1,
                    "y": 0,
                }
            ],
            sliders=sliders
    )
    '''
    #fig.update_scenes(aspectmode="manual", aspectratio=dict(x=1.0, y=1.0, z=1.0))
    fig.show()

    return

# normalize by size of the image
def plot_ground_truth(save_dir, img, scale, line_amount, name, h, gt1_array = [], data1 = [], color_plot = 'green'):

    fig3d = plt.figure(200)
    fig3d.set_size_inches((5, 3))

    ax1 = fig3d.gca(projection='3d')
    fig3d.suptitle('ax3') 

    color_array = ['r', 'g', 'b', 'brown']

    point_3d = []
    for i in range(len(gt1_array)):

        im_width = img[i].shape[1]
        im_height = img[i].shape[0]
        gt1 = gt1_array[i]
        geometry1 = gt1.getElementsByTagName('Geometry')

        width1 = float(geometry1[0].attributes.items()[0][1])
        height1 = float(geometry1[0].attributes.items()[1][1])

        detections = np.array(data1[i])[:, :2]

        detections[:, 0] = detections[:, 0]*(width1/im_width)
        detections[:, 1] = detections[:, 1]*(height1/im_height)

        #print(im_width, im_height, width1, height1, " ASDDDDDDDDDDDDDDDDDD")
        #print(np.array(detections).shape, " detectionsssss")
        #stop

        Intrinsic1 = gt1.getElementsByTagName('Intrinsic')

        focal1 = float(Intrinsic1[0].attributes.items()[0][1])
        cx1 = float(Intrinsic1[0].attributes.items()[2][1])
        cy1 = float(Intrinsic1[0].attributes.items()[3][1])
        sx1 = 0.0#float(Intrinsic1[0].attributes.items()[4][1])
        
        extrinsic1 = gt1.getElementsByTagName('Extrinsic')

        tx1 = float(extrinsic1[0].attributes.items()[0][1])/1000
        ty1 = float(extrinsic1[0].attributes.items()[1][1])/1000
        tz1 = float(extrinsic1[0].attributes.items()[2][1])/1000
        rx1 = float(extrinsic1[0].attributes.items()[3][1])
        ry1 = float(extrinsic1[0].attributes.items()[4][1])
        rz1 = float(extrinsic1[0].attributes.items()[5][1])

        R_x1 = np.array([[1,         0,                  0                   ],
                        [0,         math.cos(rx1), -math.sin(rx1) ],
                        [0,         math.sin(rx1), math.cos(rx1)  ]
                        ])
    
        R_y1 = np.array([[math.cos(ry1),    0,      math.sin(ry1)  ],
                        [0,                     1,      0                   ],
                        [-math.sin(ry1),   0,      math.cos(ry1)  ]
                        ])
    
        R_z1 = np.array([[math.cos(rz1),    -math.sin(rz1),    0],
                        [math.sin(rz1),    math.cos(rz1),     0],
                        [0,                     0,                      1]
                        ])

        intrinsic_matrix = np.array([[focal1, sx1, cx1], [0, focal1, cy1], [0, 0, 1]])
        R_rot1 = R_z1 @ R_y1 @ R_x1
        
        translation1 = np.array([tx1, ty1, tz1])

        print(translation1, " HIIIII")

        x_axis = R_rot1 @ np.array([1,0,0])
        y_axis = R_rot1 @ np.array([0,1,0])
        z_axis = R_rot1 @ np.array([0,0,1])

        print(detections.shape, " SHAPEEE")
        print(intrinsic_matrix, " intrinsic_matrix")
        '''
        plane_3d = single_util.plane_ray_intersection_np(detections[:, 0], detections[:, 1], np.linalg.inv(intrinsic_matrix), z_axis, translation1)

        print(plane_3d.shape, " WEEWQWEWE")
        ax1.scatter(plane_3d[0, :], plane_3d[1, :], plane_3d[2, :], c = color_array[i])
        '''
        ax1.plot([-tx1, -tx1 + x_axis[0]], [-ty1, -ty1 + x_axis[1]], [-tz1, -tz1 + x_axis[2]], c = 'black')
        ax1.plot([-tx1, -tx1 + y_axis[0]], [-ty1, -ty1 + y_axis[1]], [-tz1, -tz1 + y_axis[2]], c = 'black')
        ax1.plot([-tx1, -tx1 + z_axis[0]], [-ty1, -ty1 + z_axis[1]], [-tz1, -tz1 + z_axis[2]], c = color_array[i])
        #break
        point_3d.append(-translation1)
        point_3d.append(np.array([-tx1 + x_axis[0], -ty1 + x_axis[1], -tz1 + x_axis[2]]))
        point_3d.append(np.array([-tx1 + y_axis[0], -ty1 + y_axis[1], -tz1 + y_axis[2]]))
        point_3d.append(np.array([-tx1 + z_axis[0], -ty1 + z_axis[1], -tz1 + z_axis[2]]))

        ############################################################################
    xx, zz = np.meshgrid(range(-20,20), range(-20,20))
    yy =  np.zeros((40,40))
    
    point_3d.append([-20, 0, 0])
    point_3d.append([20, 0, 0])
    point_3d.append([0, 0, -20])
    point_3d.append([0, 0, 20])
    
    print(point_3d)
    #normal
    ax1.plot([0 , 1], [0 , 0], [0, 0], color="black")
    ax1.plot([0 , 0], [0 , 1], [0, 0], color="black")
    ax1.plot([0 , 0], [0 , 0], [0, 1], color="black")

    ax1.plot_surface(xx, yy, zz)


    set_equal_3d(point_3d, ax1)
    plt.show()
    ###################################################

    return


def display_plane_matrix(data, rot_matrix, save_dir, img, from_pickle, scale, line_amount, name, h, color_plot = 'green'):
    '''
    Plots the ground plane and line from ankle to head onto an image and also plots the error line from head detection to ransac predicted head, red means it exceeds threshold. 

    Parameters: ppl_ankle_u: list
                    list of dcpose 2d ankle detections u coordinate (coordinates are (u, v))
                ppl_ankle_v: list
                    list of dcpose 2d ankle detections v coordinate (coordinates are (u, v))
                ppl_head_u: list
                    list of dcpose 2d head detections u coordinate (coordinates are (u, v))
                ppl_head_v: list
                    list of dcpose 2d head detections v coordinate (coordinates are (u, v))
                save_dir: string
                    directory to save plots
                img: np.array
                    frame from the sequuence that is used for plotting
                from_pickle: dictionary
                    dictionary that contains the calibration (must contain camera matrix, normal, and plane center (called ankle))
                scale: float
                    size of the squares in the plotted grid (scale = 1 means the squares are 1 meter)
                line_amount: int
                    amount of lines in the ground plane to plot
                name: string
                    subdirectory in save_dir to save plots
                threshold_euc: float
                    Euclidean threshold for inliers
                threshold_cos: float
                    Cosine threshold for inliers
                h: float
                    assumed height of the people    

    Returns:    None
    '''
    #print(input_dict)
    img_width = img.shape[1]
    img_height = img.shape[0]
            
    ankleWorld = from_pickle['ankleWorld']
    cam_matrix = from_pickle['cam_matrix']
    normal = from_pickle['normal']

    cam_inv = np.linalg.inv(cam_matrix)
    
    fig, ax1 = plt.subplots(1, 1)
    fig.suptitle('Ground plane overlay (birds eye view)')   
    
    plane_world = np.squeeze(single_util.plane_ray_intersection_np([img_width/2.0], [img_height], cam_inv, normal, ankleWorld)) 

    #GRAPHING THE IMAGE VIEW
    
    alpha = 1
    color = 'cyan'
    linewidth = 1
    ax1.scatter(x=0.0, y=0.0, c='black', s=30)

    if save_dir is not None:
        for i in range(-line_amount,line_amount):
                p00 = [(i)*scale,(-line_amount)*scale]
                p01 = [(i)*scale,(line_amount)*scale]
                p10 = [(-line_amount)*scale,(i)*scale]
                p11 = [(line_amount)*scale,(i)*scale]
                
                x = [p00[0],p01[0]]
                y = [p00[1],p01[1]]        
                ax1.plot(x,y, '-r', c=color,  linewidth=linewidth, alpha = alpha) #this part creates the plane
                
                x = [p11[0],p10[0]]
                y = [p11[1],p10[1]]
                
                ax1.plot(x,y, '-k', c=color,  linewidth=linewidth, alpha = alpha) #this part creates the plane

    ankle_x = []
    ankle_y = []

    counter = 0
    #color = ['red' , 'blue', 'green' , 'purple' , 'yellow' , 'pink' , '#f60' , 'black' , 'white']
    for fr in list(data.keys()):
        for tr in list(data[fr].keys()):
            
            #color_plot = "#%06x" % tr
            if counter % 30 != 0:
                counter = counter + 1
                continue
            counter = counter + 1

            ankle_x.append(data[fr][tr][0])
            ankle_y.append(data[fr][tr][1])
            ax1.scatter(data[fr][tr][0],data[fr][tr][1], c=color_plot,  linewidth=linewidth, alpha = alpha)
        

    img_bl_world = cam_inv @ np.array([0,img_height, 1])
    img_br_world = cam_inv @ np.array([img_width,img_height, 1])
    
    img_bl_world = (rot_matrix @ np.append((img_bl_world/np.linalg.norm(img_bl_world))*np.sqrt(2*np.square(scale*line_amount)), 1.0))[0:3]
    img_br_world = (rot_matrix @ np.append((img_br_world/np.linalg.norm(img_br_world))*np.sqrt(2*np.square(scale*line_amount)), 1.0))[0:3]
        
    '''
    img_bl_world = (rot_matrix @ cam_inv @ np.array([0,img_height, 1]))
    img_br_world = (rot_matrix @ cam_inv @ np.array([img_width,img_height, 1]))
    
    img_bl_world = (img_bl_world/np.linalg.norm(img_bl_world))*np.sqrt(2*np.square(scale*line_amount))
    img_br_world = (img_br_world/np.linalg.norm(img_br_world))*np.sqrt(2*np.square(scale*line_amount))
    
    x_lim = max([np.absolute(min(ankle_x + [0])), np.absolute(max(ankle_x + [0]))])

    if save_dir is not None:
        plt.annotate("Camera position", (0, 0))

        ax1.set_xlim([-x_lim - 1, x_lim + 1])
        #ax1.set_ylim([min(ankle_y + [0]) - 1, max(ankle_y + [0]) + 1])
        ax1.plot([0, img_bl_world[0]], [0, img_bl_world[1]], c='red')
        ax1.plot([0, img_br_world[0]], [0, img_br_world[1]], c='red')
        ax1.set_aspect('equal', adjustable='box')
        
        fig.savefig(save_dir + '/' + 'topview_' + str(name) + '.png')
        plt.close('all')
    '''
    if save_dir is not None:
        ax1.plot([0, img_bl_world[0]], [0, img_bl_world[1]], c='red')
        ax1.plot([0, img_br_world[0]], [0, img_br_world[1]], c='red')
        x_lim = max([np.absolute(min(ankle_x + [0])), np.absolute(max(ankle_x + [0]))])
        ax1.set_xlim([-x_lim - 1, x_lim + 1])
        ax1.set_ylim([min(ankle_y + [0]) - 1, max(ankle_y + [0]) + 1])
        plt.annotate("Camera position", (0, 0))
        ax1.set_aspect('equal', adjustable='box')
        fig.savefig(save_dir + '/' + 'topview_' + str(name) + '.png')
        plt.close('all')
    #stop
    #return return_dict, ankle_x, ankle_y

def heatmap_gaussian(ankle_plane_u, ankle_plane_v, save_dir, name):
        
    fig, ax = plt.subplots(1, 1)

    heatmap, xedges, yedges = np.histogram2d(ankle_plane_u, ankle_plane_v, bins=250)
    x = sp.filters.gaussian_filter(heatmap, sigma = 20, order = 0)

    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    ax.imshow(x.T, extent=extent, origin='lower')

    fig.savefig(save_dir + '/' + 'imgplot_' + str(name) + '.png')
    plt.close('all')

def multi_cam_3d(ankle_world, head_world, extrinsic, intrinsic, scale, line_amount, img_width, img_height, h):
    '''
    Plots the ground plane and line from ankle to head onto an image and also plots the error line from head detection to ransac predicted head, red means it exceeds threshold. 

    Parameters: ppl_ankle_u: list
                    list of dcpose 2d ankle detections u coordinate (coordinates are (u, v))
                ppl_ankle_v: list
                    list of dcpose 2d ankle detections v coordinate (coordinates are (u, v))
                ppl_head_u: list
                    list of dcpose 2d head detections u coordinate (coordinates are (u, v))
                ppl_head_v: list
                    list of dcpose 2d head detections v coordinate (coordinates are (u, v))
                img: np.array
                    frame from the sequuence that is used for plotting
                from_pickle: dictionary
                    dictionary that contains the calibration (must contain camera matrix, normal, and plane center (called ankle))
                scale: float
                    size of the squares in the plotted grid (scale = 1 means the squares are 1 meter)
                line_amount: int
                    amount of lines in the ground plane to plot
                h: float
                    assumed height of the people    

    Returns:    None
    '''
            
    ankleWorld = np.array([0,0,0])
    plane_world = ankleWorld
    cam_matrix = intrinsic
    normal = np.array([0,1,0])

    cam_inv = np.linalg.inv(cam_matrix)
    
    fig3d = plt.figure(200)
    fig3d.set_size_inches((5, 3))

    ax1 = fig3d.gca(projection='3d')
    fig3d.suptitle('ax3')  
    
    color = 'cyan'
    
    point_3d = []
    
    for i in range(-line_amount,line_amount):
        for j in range(-line_amount,line_amount):

            ax1.plot([i, i + 1], [0, 0], [j, j], c=color)
            ax1.plot([i, i], [0, 0], [j, j + 1], c=color)
            ax1.plot([i, i + 1], [0, 0],[j + 1, j + 1], c=color)
            ax1.plot([i + 1, i + 1], [0, 0],[j + 1, j], c=color)

            point_3d.append(np.array([i,0,j]))
        
    ax1.scatter(0,0,0, 'black')

    '''
    corner = draw_frustum(ax1, img_width, img_height, cam_matrix, np.sqrt(2*np.square(scale*line_amount)))
    
    point_3d = point_3d + [corner[0], corner[1], corner[2], corner[3], [0,0,0]]
    '''
    for i in range(len(extrinsic)):
        draw_frustum_multi(ax1, img_width, img_height, intrinsic[i], extrinsic[i], scale)

    for i in range(0, len(ankle_world)):
        #points that are not detected by openpose are assignned -1
    
        ax1.scatter(ankle_world[i][0],ankle_world[i][ 1],ankle_world[i][ 2], 'black')
        ax1.scatter(head_world[i][ 0],head_world[i][ 1],head_world[i][ 2], 'black')
        ax1.plot([ankle_world[i][0], head_world[i][0]],[ankle_world[i][1], head_world[i][1]],[ankle_world[i][2], head_world[i][2]], 'blue')
    
    #ax1.plot([0,0], [0,0], [0, scale*(line_amount/2)], 'green')
    set_equal_3d(np.array(ankle_world + head_world + point_3d), ax1)
    plt.show()

def plot_frame_error(ground, pred_array, save_dir, name, label_array = []):

    fig, ax1 = plt.subplots(1, 1)
    cumulative_array = []
    curve_array = []

    sum_array = []

    max_bins = 0
    for i in range(len(pred_array)):
    
        pred = pred_array[i]
        diff = np.abs(np.array(ground) - np.array(pred))
        bins = max(list(diff))

        if max_bins < bins:
            max_bins = bins

    if len(label_array) == 0:
        label_array = list(map(str,list(range(len(pred_array)))))

    for j in range(len(pred_array)):

        pred = pred_array[j]
        diff = np.abs(np.array(ground) - np.array(pred))
        #bins = max(list(diff))
        
        diff_array = []
        for i in range(0, max_bins + 1):
            diff_array.append(list(diff).count(i)/len(list(diff)))

        cumulative = np.cumsum(diff_array)

        cum_dict = {k: v for v, k in enumerate(cumulative)}
        cumulative_array.append(cum_dict)
        
        curves, = ax1.plot(cumulative, label = label_array[j])
        curve_array.append(curves)

        sum_array.append(np.sum(cumulative))
    
    #print(label_array)
    ax1.legend(curve_array, label_array)
    fig.savefig(save_dir + '/' + 'error_' + str(name) + '.png')
    plt.close('all')
    
    return cumulative_array, sum_array

def plot_magnitude_time(ref, sync, save_dir, name, sync_array = None, sync_array1 = None):

    fig, ax1 = plt.subplots(1, 1)
    #print(ref.shape)
    #print(sync.shape)

    ref_center = np.mean(ref[0:2,:], axis = 1)
    sync_center = np.mean(sync[0:2,:], axis = 1)

    for i in range(ref.shape[1]):
        ax1.scatter([ref[2, i]],[np.linalg.norm(ref[0:1, i] - ref_center)], c = 'blue')
    
    for i in range(sync.shape[1]):
        ax1.scatter([sync[2, i]],[np.linalg.norm(sync[0:1, i] - sync_center)], c = 'red') 
    
    if sync_array is not None:
        for s in range(len(sync_array)):
            ax1.plot([sync[2, sync_array[s]], ref[2, sync_array1[s]]],[np.linalg.norm(sync[0:1, sync_array[s]] - sync_center), np.linalg.norm(ref[0:1, sync_array1[s]] - ref_center)], c = 'green') 
    
    fig.savefig(save_dir + '/' + 'magnitude_' + str(name) + '.png')
    plt.close('all')
    return

def plot_velocity(ref, sync, save_dir, name, sync_array = None, sync_array1 = None):
    
    fig, ax1 = plt.subplots(1, 1)
    #print(ref.shape)
    #print(sync.shape)
    #print(ref[2,:])
    #print(sync[2,:])

    ref_diff = np.gradient(ref[:2,:], axis = 1)
    sync_diff = np.gradient(sync[:2,:], axis = 1)

    ref_diff_time = np.gradient(ref[2,:], axis = 0)
    sync_diff_time = np.gradient(sync[2,:], axis = 0)

    #ref_diff = np.append(ref_diff, [0])
    #sync_diff = np.append(sync_diff, [0])
    #print(ref.shape, " REF SHAPE")

    for i in range(ref.shape[1]):
        ax1.scatter([ref[2, i]],[np.linalg.norm(ref_diff[:, i]/ref_diff_time[i])], c = 'blue')
    
    for i in range(sync.shape[1]):
        ax1.scatter([sync[2, i]],[np.linalg.norm(sync_diff[:, i]/sync_diff_time[i])], c = 'red') 

    if sync_array is not None:
        for s in range(len(sync_array)):
            ax1.plot([sync[2, :][sync_array[s]], ref[2, :][sync_array1[s]]],[np.linalg.norm(sync_diff[:, sync_array[s]]/sync_diff_time[s]), np.linalg.norm(ref_diff[:, sync_array1[s]]/ref_diff_time[s])], c = 'green')

    fig.savefig(save_dir + '/' + 'vel_' + str(name) + '.png')
    plt.close('all')
    return

def plot_vel_icp(ref, sync, save_dir, name, sync_array = None, sync_array1 = None):
    
    fig, ax1 = plt.subplots(1, 1)

    for i in range(ref.shape[1]):
        ax1.scatter([ref[1, i]],[ref[0, i]], c = 'blue')
    
    for i in range(sync.shape[1]):
        ax1.scatter([sync[1, i]],[sync[0, i]], c = 'red') 

    if sync_array is not None:
        for s in range(len(sync_array)):
            ax1.plot([sync[1, sync_array[s]],ref[1, sync_array1[s]]],[sync[0, sync_array[s]], ref[0, sync_array1[s]]], c = 'green')

    fig.savefig(save_dir + '/' + 'vel_icp_' + str(name) + '.png')
    plt.close('all')
    return

def make_gif(ankles_array, save_dir, img, from_pickle, scale, line_amount, name):

    ankles_array = np.array(ankles_array)
    if os.path.isdir(save_dir + '/' + 'frames_' + str(name)) == False:
        os.mkdir(save_dir + '/' + 'frames_' + str(name))

    ankle_x = list(np.array(ankles_array)[0, :]) + [0]
    ankle_y = list(np.array(ankles_array)[1, :]) + [0]
    
    x_lim = max([np.absolute(min(ankle_x)), np.absolute(max(ankle_x))])
    y_lim_lower = min(ankle_y)
    y_lim_upper = max(ankle_y)

    for point in range(np.array(ankles_array).shape[1]):
        display_2d_plane([np.transpose(np.array([ankles_array[:, point]]))], save_dir + '/' + 'frames_' + str(name), img, from_pickle, scale, line_amount, point, x_lim = x_lim, y_lim_lower = y_lim_lower, y_lim_upper = y_lim_upper)

    frames = sorted(os.listdir(save_dir + '/' + 'frames_' + str(name)))
    frames.sort(key=lambda x: int(''.join(filter(str.isdigit, x))))
    
    images = []
    for fr in frames:
        filename = save_dir + '/' + 'frames_' + str(name) + '/' + fr
        images.append(imageio.imread(filename))
    imageio.mimsave(save_dir + '/' + str(name) +'.gif', images, fps=30)

def plot_plane(ppl_ankle_u, ppl_ankle_v, ppl_head_u, ppl_head_v, save_dir, img, from_pickle, scale, line_amount, name, threshold_euc, threshold_cos, h):
    '''
    Plots the ground plane and line from ankle to head onto an image and also plots the error line from head detection to ransac predicted head, red means it exceeds threshold. 

    Parameters: ppl_ankle_u: list
                    list of dcpose 2d ankle detections u coordinate (coordinates are (u, v))
                ppl_ankle_v: list
                    list of dcpose 2d ankle detections v coordinate (coordinates are (u, v))
                ppl_head_u: list
                    list of dcpose 2d head detections u coordinate (coordinates are (u, v))
                ppl_head_v: list
                    list of dcpose 2d head detections v coordinate (coordinates are (u, v))
                save_dir: string
                    directory to save plots
                img: np.array
                    frame from the sequuence that is used for plotting
                from_pickle: dictionary
                    dictionary that contains the calibration (must contain camera matrix, normal, and plane center (called ankle))
                scale: int
                    size of the squares in the plotted grid (scale = 1 means the squares are 1 meter)
                line_amount: int
                    amount of lines in the ground plane to plot
                name: string
                    subdirectory in save_dir to save plots
                threshold_euc: float
                    Euclidean threshold for inliers
                threshold_cos: float
                    Cosine threshold for inliers
                h: float
                    assumed height of the people    

    Returns:    None
    '''

    img_width = img.shape[1]
    img_height = img.shape[0]

    fig, ax1 = plt.subplots(1, 1)

    ax1.imshow(img)

    cam_matrix = from_pickle['cam_matrix']
    normal = from_pickle['normal']
    ankleWorld = from_pickle['ankleWorld']
    
    print(normal, " NORMAL")
    print(cam_matrix, " CAM_MATRIX")
    print(ankleWorld, " ANKLEWORLD")

    cam_inv = np.linalg.inv(cam_matrix)
    
    display_homography_horiz_bottom(cam_matrix ,cam_inv ,scale , img_width, img_height, ankleWorld, normal, line_amount, ax_array = [ax1])
    ankle_distance_array = []

    for i in range(0, len(ppl_ankle_u)):
        #points that are not detected by openpose are assignned -1
        if ppl_ankle_u[i] < 0 or ppl_ankle_v[i] < 0 or ppl_head_u[i] < 0 or ppl_head_v[i] < 0:
            continue
            
        person_world = np.squeeze(single_util.plane_ray_intersection_np([ppl_ankle_u[i]], [ppl_ankle_v[i]], cam_inv, normal, ankleWorld))   
        ankle_distance_array.append(np.linalg.norm(person_world))
        ankle_ppl_2d = single_util.perspective_transformation(cam_matrix, person_world)
        head_ppl_2d = single_util.perspective_transformation(cam_matrix, np.array(person_world) + np.squeeze(normal)*h)

        head_vect_pred = np.array([head_ppl_2d[0], head_ppl_2d[1]]) - np.array([ankle_ppl_2d[0], ankle_ppl_2d[1]])
        head_vect_ground = np.array([ppl_head_u[i], ppl_head_v[i]]) - np.array([ppl_ankle_u[i], ppl_ankle_v[i]])

        head_vect_ground_norm = np.linalg.norm(head_vect_ground)
        
        error_cos = 1.0 - single_util.matrix_cosine(np.expand_dims(head_vect_pred, axis = 0), np.expand_dims(head_vect_ground, axis = 0))
        error_norm = np.linalg.norm(np.array([head_ppl_2d[0], head_ppl_2d[1]]) - np.array([ppl_head_u[i], ppl_head_v[i]]))/head_vect_ground_norm
        
        if error_cos < threshold_cos and error_norm < threshold_euc:
            ax1.scatter(x=head_ppl_2d[0], y=head_ppl_2d[1], c='green', s=10)
            ax1.scatter(x=ankle_ppl_2d[0], y=ankle_ppl_2d[1], c='green', s=30)
            ax1.plot([head_ppl_2d[0], ppl_head_u[i]],[head_ppl_2d[1], ppl_head_v[i]], c = 'green')
            ax1.plot([head_ppl_2d[0], ankle_ppl_2d[0]],[head_ppl_2d[1], ankle_ppl_2d[1]], c = 'green')
        
        else:
            ax1.scatter(x=ankle_ppl_2d[0], y=ankle_ppl_2d[1], c='red', s=30)
            ax1.scatter(x=head_ppl_2d[0], y=head_ppl_2d[1], c='red', s=10)
            ax1.plot([head_ppl_2d[0], ppl_head_u[i]],[head_ppl_2d[1], ppl_head_v[i]], c = 'red')
            ax1.plot([head_ppl_2d[0], ankle_ppl_2d[0]],[head_ppl_2d[1], ankle_ppl_2d[1]], c = 'red') 
        
            
    fig.savefig(save_dir + '/' + 'imgplot_' + str(name) + '.png')
    plt.close('all')

# ground plane
def display_homography_horiz_bottom(cam_matrix, cam_inv, scale, img_width, img_height, ankleWorld, normal, line_amount, ax_array = []):
    '''
    creates the groundplane as a mesh on an axes object

    Parameters: cam_matrix: (3,3) np.array
                    The intrinsic camera matrix.
                cam_inv: (3,3) np.array
                    The inverse intrinsic camera matrix.
                scale: Float
                    The scale of the plotted grid.
                img_width: int
                    Pixel width of the image.
                img_height: Int
                    Pixel height of the image.
                ankleWorld: (3,) np.array
                    3d point of an ankle used to initalize the ground plane.
                normal: (3,) np.array
                    Normal vector of the groundplane.
                line_amount: Int
                    The amount of lines to graph. Creates line_amount of horizontal lines and 2*line_amount in the x direction
                ax_array: Array of matplotlib axes objects
                    axes for the grid to be ploted on
    Returns:    ax_array: list of matplotlib ax object
    '''

    plane_world = np.squeeze(single_util.plane_ray_intersection_np([img_width/2.0], [img_height], cam_inv, normal, ankleWorld))
    
    plane_world_2d = cam_matrix @ plane_world
    
    plane_world1 = np.squeeze(single_util.plane_ray_intersection_np([img_width/2.0 + 1], [img_height], cam_inv, normal, ankleWorld))
                
    v_basis = np.array(plane_world1) - np.array(plane_world)
    v_basis = v_basis/np.linalg.norm(v_basis)
    
    u_basis = np.cross(normal, v_basis)
    u_basis = u_basis/np.linalg.norm(u_basis)
                
    x_basis_3d = [plane_world[0] + scale*v_basis[0], plane_world[1] + scale*v_basis[1], plane_world[2] + scale*v_basis[2]]
    y_basis_3d = [plane_world[0] + scale*u_basis[0], plane_world[1] + scale*u_basis[1], plane_world[2] + scale*u_basis[2]]
    z_basis_3d = [plane_world[0] + scale*normal[0], plane_world[1] + scale*normal[1], plane_world[2] + scale*normal[2]]

    x_basis = single_util.perspective_transformation(cam_matrix, x_basis_3d)
    y_basis = single_util.perspective_transformation(cam_matrix, y_basis_3d)
    z_basis = single_util.perspective_transformation(cam_matrix, z_basis_3d)
    
    ax_array[0].scatter(x = plane_world_2d[0]/plane_world_2d[2], y = plane_world_2d[1]/plane_world_2d[2], c = 'black', s = 40)
    ax_array[0].scatter(x = img_width/2.0,y = img_height, c='red',  s=20)
    ax_array[0].scatter(x = x_basis[0],y = x_basis[1], c='orange',  s=20)
    ax_array[0].scatter(x = y_basis[0],y = y_basis[1], c='lime',  s=20)
    ax_array[0].scatter(x = z_basis[0],y = z_basis[1], c='red',  s=20)

    ax_array[0].plot([img_width/2.0, x_basis[0]],[img_height, x_basis[1]], '-k', c='orange',  linewidth=1.0)
    ax_array[0].plot([img_width/2.0, y_basis[0]],[img_height, y_basis[1]], '-k', c='lime',  linewidth=1.0)
    ax_array[0].plot([img_width/2.0, z_basis[0]],[img_height, z_basis[1]], '-k', c='red',  linewidth=1.0)

    rot_matrix = single_util.basis_change_rotation_matrix(cam_matrix, cam_inv, ankleWorld, normal, img_width, img_height)
    
    color = 'cyan'
    color1 = 'cyan'
    for i in range(-line_amount,line_amount):
            p00, p00_3d = single_util.project_point_horiz_bottom(cam_matrix, cam_inv,[(i)*scale,(0)*scale], plane_world, normal, img_width, img_height)
            p01, p01_3d = single_util.project_point_horiz_bottom(cam_matrix, cam_inv,[(i)*scale,(line_amount)*scale], plane_world, normal, img_width, img_height)
            p10, p10_3d = single_util.project_point_horiz_bottom(cam_matrix, cam_inv,[(-line_amount)*scale,(i)*scale], plane_world, normal, img_width, img_height)
            p11, p11_3d = single_util.project_point_horiz_bottom(cam_matrix, cam_inv,[(line_amount)*scale,(i)*scale], plane_world, normal, img_width, img_height)
            
            if p00_3d[2] < 0:
                p00_3d = single_util.plane_line_intersection(p00_3d, p01_3d, [0,0,1], [0,0,0.1])
                p00 = single_util.perspective_transformation(cam_matrix, p00_3d)
            if p01_3d[2] < 0:
                p01_3d = single_util.plane_line_intersection(p01_3d, p00_3d, [0,0,1], [0,0,0.1])
                p01 = single_util.perspective_transformation(cam_matrix, p01_3d)
            if p10_3d[2] < 0:
                p10_3d = single_util.plane_line_intersection(p10_3d, p11_3d, [0,0,1], [0,0,0.1])
                p10 = single_util.perspective_transformation(cam_matrix, p10_3d)
            if p11_3d[2] < 0:
                p11_3d = single_util.plane_line_intersection(p11_3d, p10_3d, [0,0,1], [0,0,0.1])
                p11 = single_util.perspective_transformation(cam_matrix, p11_3d)
            
            p00_plane = rot_matrix @ p00_3d
            p01_plane = rot_matrix @ p01_3d
            p10_plane = rot_matrix @ p10_3d 
            p11_plane = rot_matrix @ p11_3d
            
            alpha = 1
            alpha1 = 1
            linewidth = 1
            
            thresh1, thresh2, thresh3, thresh4 = 2,4,6,8
            if np.linalg.norm(p00_3d) < thresh1 and np.linalg.norm(p01_3d) < thresh1:
                alpha = 1.0
            elif np.linalg.norm(p00_3d) < thresh2 and np.linalg.norm(p01_3d) < thresh2:
                alpha = 0.8
            elif np.linalg.norm(p00_3d) < thresh3 and np.linalg.norm(p01_3d) < thresh3:
                alpha = 0.6
            elif np.linalg.norm(p00_3d) < thresh4 and np.linalg.norm(p01_3d) < thresh4:
                alpha = 0.4
            else:
                alpha = 0.2
                
            if np.linalg.norm(p10_3d) < thresh1 and np.linalg.norm(p11_3d) < thresh1:
                alpha1 = 1.0
            elif np.linalg.norm(p10_3d) < thresh2 and np.linalg.norm(p11_3d) < thresh2:
                alpha1 = 0.8
            elif np.linalg.norm(p10_3d) < thresh3 and np.linalg.norm(p11_3d) < thresh3:
                alpha1 = 0.6
            elif np.linalg.norm(p10_3d) < thresh4 and np.linalg.norm(p11_3d) < thresh4:
                alpha1 = 0.4
            else:
                alpha1 = 0.2
            
            x = [p00[0],p01[0]]
            y = [p00[1],p01[1]] 
            ax_array[0].plot(x,y, '-r', c=color,  linewidth=linewidth, alpha = alpha, zorder=1) #this part creates the plane

            x = [p11[0],p10[0]]
            y = [p11[1],p10[1]]
            if p11_plane[1] > 0 or p10_plane[1] > 0:
                ax_array[0].plot(x,y, '-k', c=color1,  linewidth=linewidth, alpha = alpha1, zorder=1) #this part creates the plane
                
    ax_array[0].set_xlim([-50, img_width + 50])
    ax_array[0].set_ylim([img_height + 50, -50])

    return ax_array

def display_2d_grid_dict_frame(input_dict, save_dir, img, from_pickle, scale, line_amount, name, threshold_euc, threshold_cos, h, color_plot = 'green'):
    '''
    Plots the ground plane and line from ankle to head onto an image and also plots the error line from head detection to ransac predicted head, red means it exceeds threshold. 

    Parameters: ppl_ankle_u: list
                    list of dcpose 2d ankle detections u coordinate (coordinates are (u, v))
                ppl_ankle_v: list
                    list of dcpose 2d ankle detections v coordinate (coordinates are (u, v))
                ppl_head_u: list
                    list of dcpose 2d head detections u coordinate (coordinates are (u, v))
                ppl_head_v: list
                    list of dcpose 2d head detections v coordinate (coordinates are (u, v))
                save_dir: string
                    directory to save plots
                img: np.array
                    frame from the sequuence that is used for plotting
                from_pickle: dictionary
                    dictionary that contains the calibration (must contain camera matrix, normal, and plane center (called ankle))
                scale: float
                    size of the squares in the plotted grid (scale = 1 means the squares are 1 meter)
                line_amount: int
                    amount of lines in the ground plane to plot
                name: string
                    subdirectory in save_dir to save plots
                threshold_euc: float
                    Euclidean threshold for inliers
                threshold_cos: float
                    Cosine threshold for inliers
                h: float
                    assumed height of the people    

    Returns:    None
    '''
    #print(input_dict)
    img_width = img.shape[1]
    img_height = img.shape[0]
            
    ankleWorld = from_pickle['ankleWorld']
    cam_matrix = from_pickle['cam_matrix']
    normal = from_pickle['normal']

    cam_inv = np.linalg.inv(cam_matrix)
    
    rot_matrix = single_util.basis_change_rotation_matrix(cam_matrix, cam_inv, ankleWorld, normal, img_width, img_height)
    
    fig, ax1 = plt.subplots(1, 1)
    fig.suptitle('Ground plane overlay (birds eye view)')   
    
    plane_world = np.squeeze(single_util.plane_ray_intersection_np([img_width/2.0], [img_height], cam_inv, normal, ankleWorld)) 

    #GRAPHING THE IMAGE VIEW
    
    alpha = 1
    color = 'cyan'
    linewidth = 1
    ax1.scatter(x=0.0, y=0.0, c='black', s=30)

    p_center_proj, p_center_3d = single_util.project_point_horiz_bottom(cam_matrix, cam_inv,[0,0], plane_world, normal, img_width, img_height)
    p_center = rot_matrix @ p_center_3d  

    if save_dir is not None:
        for i in range(-line_amount,line_amount):
                p00, p00_3d = single_util.project_point_horiz_bottom(cam_matrix, cam_inv,[(i)*scale,(-line_amount)*scale], plane_world, normal, img_width, img_height)
                p01, p01_3d = single_util.project_point_horiz_bottom(cam_matrix, cam_inv,[(i)*scale,(line_amount)*scale], plane_world, normal, img_width, img_height)
                p10, p10_3d = single_util.project_point_horiz_bottom(cam_matrix, cam_inv,[(-line_amount)*scale,(i)*scale], plane_world, normal, img_width, img_height)
                p11, p11_3d = single_util.project_point_horiz_bottom(cam_matrix, cam_inv,[(line_amount)*scale,(i)*scale], plane_world, normal, img_width, img_height)
                    
                p00 = (rot_matrix @ p00_3d) - p_center 
                p01 = (rot_matrix @ p01_3d) - p_center  
                p10 = (rot_matrix @ p10_3d) - p_center   
                p11 = (rot_matrix @ p11_3d) - p_center  
                
                x = [p00[0],p01[0]]
                y = [p00[1],p01[1]]        
                ax1.plot(x,y, '-r', c=color,  linewidth=linewidth, alpha = alpha) #this part creates the plane
                
                x = [p11[0],p10[0]]
                y = [p11[1],p10[1]]
                
                ax1.plot(x,y, '-k', c=color,  linewidth=linewidth, alpha = alpha) #this part creates the plane
    
    return_dict = {}
    keys = list(input_dict.keys())
    ankle_x = []
    ankle_y = []

    for i in range(0, len(keys)):
        return_dict[keys[i]] = []
        for j in range(0, len(input_dict[keys[i]])):
            #points that are not detected by openpose are assignned -1
            ppl_ankle_u = input_dict[keys[i]][j][0]
            ppl_ankle_v = input_dict[keys[i]][j][1]
            ppl_head_u = input_dict[keys[i]][j][2]
            ppl_head_v = input_dict[keys[i]][j][3]
            #if ppl_ankle_u < 0 or ppl_ankle_v < 0 or ppl_head_u < 0 or ppl_head_v < 0:
            #    continue
            
            ankle_3d = np.squeeze(single_util.plane_ray_intersection_np([ppl_ankle_u], [ppl_ankle_v], cam_inv, normal, ankleWorld))
            person_plane = (rot_matrix @ ankle_3d) 
            #ax1.scatter(x=person_plane[0], y=person_plane[1], c='green', s=30)
            
            ankle_x.append(person_plane[0])
            ankle_y.append(person_plane[1])

            return_dict[keys[i]].append([person_plane[0], person_plane[1]])

            
            #############
            ankle_ppl_2d = single_util.perspective_transformation(cam_matrix, ankle_3d)
            head_ppl_2d = single_util.perspective_transformation(cam_matrix, np.array(ankle_3d) + np.squeeze(normal)*h)

            head_vect_pred = np.array([head_ppl_2d[0], head_ppl_2d[1]]) - np.array([ankle_ppl_2d[0], ankle_ppl_2d[1]])
            head_vect_ground = np.array([ppl_head_u, ppl_head_v]) - np.array([ppl_ankle_u, ppl_ankle_v])

            head_vect_ground_norm = np.linalg.norm(head_vect_ground)
            
            error_cos = 1.0 - single_util.matrix_cosine(np.expand_dims(head_vect_pred, axis = 0), np.expand_dims(head_vect_ground, axis = 0))
            error_norm = np.linalg.norm(np.array([head_ppl_2d[0], head_ppl_2d[1]]) - np.array([ppl_head_u, ppl_head_v]))/head_vect_ground_norm
            
            if save_dir is not None:
                if error_cos < threshold_cos and error_norm < threshold_euc:
                    ax1.scatter(x=person_plane[0], y=person_plane[1], c=color_plot, s=30)
                
                else:
                    ax1.scatter(x=person_plane[0], y=person_plane[1], c='red', s=30)
        
    
    img_bl_world = (rot_matrix @ cam_inv @ np.array([0,img_height, 1]))
    img_br_world = (rot_matrix @ cam_inv @ np.array([img_width,img_height, 1]))
    
    img_bl_world = (img_bl_world/np.linalg.norm(img_bl_world))*np.sqrt(2*np.square(scale*line_amount))
    img_br_world = (img_br_world/np.linalg.norm(img_br_world))*np.sqrt(2*np.square(scale*line_amount))
    
    x_lim = max([np.absolute(min(ankle_x + [0])), np.absolute(max(ankle_x + [0]))])

    if save_dir is not None:
        plt.annotate("Camera position", (0, 0))

        ax1.set_xlim([-x_lim - 1, x_lim + 1])
        ax1.set_ylim([min(ankle_y + [0]) - 1, max(ankle_y + [0]) + 1])
        ax1.plot([0, img_bl_world[0]], [0, img_bl_world[1]], c='red')
        ax1.plot([0, img_br_world[0]], [0, img_br_world[1]], c='red')
        ax1.set_aspect('equal', adjustable='box')
        
        fig.savefig(save_dir + '/' + 'topview_' + str(name) + '.png')
        plt.close('all')

    return return_dict, ankle_x, ankle_y

def display_2d_grid_dict(input_dict, save_dir, img, from_pickle, scale, line_amount, name, threshold_euc, threshold_cos, h, color_plot = 'green'):
    '''
    Plots the ground plane and line from ankle to head onto an image and also plots the error line from head detection to ransac predicted head, red means it exceeds threshold. 

    Parameters: ppl_ankle_u: list
                    list of dcpose 2d ankle detections u coordinate (coordinates are (u, v))
                ppl_ankle_v: list
                    list of dcpose 2d ankle detections v coordinate (coordinates are (u, v))
                ppl_head_u: list
                    list of dcpose 2d head detections u coordinate (coordinates are (u, v))
                ppl_head_v: list
                    list of dcpose 2d head detections v coordinate (coordinates are (u, v))
                save_dir: string
                    directory to save plots
                img: np.array
                    frame from the sequuence that is used for plotting
                from_pickle: dictionary
                    dictionary that contains the calibration (must contain camera matrix, normal, and plane center (called ankle))
                scale: float
                    size of the squares in the plotted grid (scale = 1 means the squares are 1 meter)
                line_amount: int
                    amount of lines in the ground plane to plot
                name: string
                    subdirectory in save_dir to save plots
                threshold_euc: float
                    Euclidean threshold for inliers
                threshold_cos: float
                    Cosine threshold for inliers
                h: float
                    assumed height of the people    

    Returns:    None
    '''
    
    img_width = img.shape[1]
    img_height = img.shape[0]
            
    ankleWorld = from_pickle['ankleWorld']
    cam_matrix = from_pickle['cam_matrix']
    normal = from_pickle['normal']

    cam_inv = np.linalg.inv(cam_matrix)
    
    rot_matrix = single_util.basis_change_rotation_matrix(cam_matrix, cam_inv, ankleWorld, normal, img_width, img_height)
    
    fig, ax1 = plt.subplots(1, 1)
    fig.suptitle('Ground plane overlay (birds eye view)')   
    
    plane_world = np.squeeze(single_util.plane_ray_intersection_np([img_width/2.0], [img_height], cam_inv, normal, ankleWorld)) 

    #GRAPHING THE IMAGE VIEW
    
    alpha = 1
    color = 'cyan'
    linewidth = 1
    ax1.scatter(x=0.0, y=0.0, c='black', s=30)

    p_center_proj, p_center_3d = single_util.project_point_horiz_bottom(cam_matrix, cam_inv,[0,0], plane_world, normal, img_width, img_height)
    p_center = rot_matrix @ p_center_3d  

    for i in range(-line_amount,line_amount):
            p00, p00_3d = single_util.project_point_horiz_bottom(cam_matrix, cam_inv,[(i)*scale,(-line_amount)*scale], plane_world, normal, img_width, img_height)
            p01, p01_3d = single_util.project_point_horiz_bottom(cam_matrix, cam_inv,[(i)*scale,(line_amount)*scale], plane_world, normal, img_width, img_height)
            p10, p10_3d = single_util.project_point_horiz_bottom(cam_matrix, cam_inv,[(-line_amount)*scale,(i)*scale], plane_world, normal, img_width, img_height)
            p11, p11_3d = single_util.project_point_horiz_bottom(cam_matrix, cam_inv,[(line_amount)*scale,(i)*scale], plane_world, normal, img_width, img_height)
                
            p00 = (rot_matrix @ p00_3d) - p_center 
            p01 = (rot_matrix @ p01_3d) - p_center  
            p10 = (rot_matrix @ p10_3d) - p_center   
            p11 = (rot_matrix @ p11_3d) - p_center  
            
            x = [p00[0],p01[0]]
            y = [p00[1],p01[1]]        
            ax1.plot(x,y, '-r', c=color,  linewidth=linewidth, alpha = alpha) #this part creates the plane
            
            x = [p11[0],p10[0]]
            y = [p11[1],p10[1]]
            
            ax1.plot(x,y, '-k', c=color,  linewidth=linewidth, alpha = alpha) #this part creates the plane
    
    return_dict = {}
    keys = list(input_dict.keys())
    ankle_x = []
    ankle_y = []

    for i in range(0, len(keys)):
        return_dict[i] = []
        for j in range(0, len(input_dict[keys[i]])):
            #points that are not detected by openpose are assignned -1
            ppl_ankle_u = input_dict[keys[i]][j][0]
            ppl_ankle_v = input_dict[keys[i]][j][1]
            ppl_head_u = input_dict[keys[i]][j][2]
            ppl_head_v = input_dict[keys[i]][j][3]
            if ppl_ankle_u < 0 or ppl_ankle_v < 0 or ppl_head_u < 0 or ppl_head_v < 0:
                continue
            
            ankle_3d = np.squeeze(single_util.plane_ray_intersection_np([ppl_ankle_u], [ppl_ankle_v], cam_inv, normal, ankleWorld))
            person_plane = (rot_matrix @ ankle_3d) 
            #ax1.scatter(x=person_plane[0], y=person_plane[1], c='green', s=30)
            
            ankle_x.append(person_plane[0])
            ankle_y.append(person_plane[1])

            return_dict[i].append([person_plane[0], person_plane[1]])

            
            #############
            ankle_ppl_2d = single_util.perspective_transformation(cam_matrix, ankle_3d)
            head_ppl_2d = single_util.perspective_transformation(cam_matrix, np.array(ankle_3d) + np.squeeze(normal)*h)

            head_vect_pred = np.array([head_ppl_2d[0], head_ppl_2d[1]]) - np.array([ankle_ppl_2d[0], ankle_ppl_2d[1]])
            head_vect_ground = np.array([ppl_head_u, ppl_head_v]) - np.array([ppl_ankle_u, ppl_ankle_v])

            head_vect_ground_norm = np.linalg.norm(head_vect_ground)
            
            error_cos = 1.0 - single_util.matrix_cosine(np.expand_dims(head_vect_pred, axis = 0), np.expand_dims(head_vect_ground, axis = 0))
            error_norm = np.linalg.norm(np.array([head_ppl_2d[0], head_ppl_2d[1]]) - np.array([ppl_head_u, ppl_head_v]))/head_vect_ground_norm
            
            if error_cos < threshold_cos and error_norm < threshold_euc:
                ax1.scatter(x=person_plane[0], y=person_plane[1], c=color_plot, s=30)
            
            else:
                ax1.scatter(x=person_plane[0], y=person_plane[1], c='red', s=30)
        
    plt.annotate("Camera position", (0, 0))
    
    img_bl_world = (rot_matrix @ cam_inv @ np.array([0,img_height, 1]))
    img_br_world = (rot_matrix @ cam_inv @ np.array([img_width,img_height, 1]))
    
    img_bl_world = (img_bl_world/np.linalg.norm(img_bl_world))*np.sqrt(2*np.square(scale*line_amount))
    img_br_world = (img_br_world/np.linalg.norm(img_br_world))*np.sqrt(2*np.square(scale*line_amount))
    
    x_lim = max([np.absolute(min(ankle_x + [0])), np.absolute(max(ankle_x + [0]))])

    ax1.set_xlim([-x_lim - 1, x_lim + 1])
    ax1.set_ylim([min(ankle_y + [0]) - 1, max(ankle_y + [0]) + 1])
    ax1.plot([0, img_bl_world[0]], [0, img_bl_world[1]], c='red')
    ax1.plot([0, img_br_world[0]], [0, img_br_world[1]], c='red')
    ax1.set_aspect('equal', adjustable='box')
    
    fig.savefig(save_dir + '/' + 'topview_' + str(name) + '.png')
    plt.close('all')

    return return_dict, ankle_x, ankle_y

def display_2d_grid_ref_sync(ppl_ankle_u, ppl_ankle_v, img, from_pickle, scale, line_amount, ax1, color_plot = 'green'):
    '''
    Plots the ground plane and line from ankle to head onto an image and also plots the error line from head detection to ransac predicted head, red means it exceeds threshold. 

    Parameters: ppl_ankle_u: list
                    list of dcpose 2d ankle detections u coordinate (coordinates are (u, v))
                ppl_ankle_v: list
                    list of dcpose 2d ankle detections v coordinate (coordinates are (u, v))
                ppl_head_u: list
                    list of dcpose 2d head detections u coordinate (coordinates are (u, v))
                ppl_head_v: list
                    list of dcpose 2d head detections v coordinate (coordinates are (u, v))
                save_dir: string
                    directory to save plots
                img: np.array
                    frame from the sequuence that is used for plotting
                from_pickle: dictionary
                    dictionary that contains the calibration (must contain camera matrix, normal, and plane center (called ankle))
                scale: float
                    size of the squares in the plotted grid (scale = 1 means the squares are 1 meter)
                line_amount: int
                    amount of lines in the ground plane to plot
                name: string
                    subdirectory in save_dir to save plots
                threshold_euc: float
                    Euclidean threshold for inliers
                threshold_cos: float
                    Cosine threshold for inliers
                h: float
                    assumed height of the people    

    Returns:    None
    '''
    
    img_width = img.shape[1]
    img_height = img.shape[0]
            
    ankleWorld = from_pickle['ankleWorld']
    cam_matrix = from_pickle['cam_matrix']
    normal = from_pickle['normal']

    cam_inv = np.linalg.inv(cam_matrix)
    
    rot_matrix = single_util.basis_change_rotation_matrix(cam_matrix, cam_inv, ankleWorld, normal, img_width, img_height)  
    
    plane_world = np.squeeze(single_util.plane_ray_intersection_np([img_width/2.0], [img_height], cam_inv, normal, ankleWorld)) 

    #GRAPHING THE IMAGE VIEW
    
    alpha = 1
    color = 'cyan'
    linewidth = 1
    ax1.scatter(x=0.0, y=0.0, c='black', s=30)

    p_center_proj, p_center_3d = single_util.project_point_horiz_bottom(cam_matrix, cam_inv,[0,0], plane_world, normal, img_width, img_height)
    p_center = rot_matrix @ p_center_3d  

    for i in range(-line_amount,line_amount):
            p00, p00_3d = single_util.project_point_horiz_bottom(cam_matrix, cam_inv,[(i)*scale,(-line_amount)*scale], plane_world, normal, img_width, img_height)
            p01, p01_3d = single_util.project_point_horiz_bottom(cam_matrix, cam_inv,[(i)*scale,(line_amount)*scale], plane_world, normal, img_width, img_height)
            p10, p10_3d = single_util.project_point_horiz_bottom(cam_matrix, cam_inv,[(-line_amount)*scale,(i)*scale], plane_world, normal, img_width, img_height)
            p11, p11_3d = single_util.project_point_horiz_bottom(cam_matrix, cam_inv,[(line_amount)*scale,(i)*scale], plane_world, normal, img_width, img_height)
                
            p00 = (rot_matrix @ p00_3d) - p_center 
            p01 = (rot_matrix @ p01_3d) - p_center  
            p10 = (rot_matrix @ p10_3d) - p_center   
            p11 = (rot_matrix @ p11_3d) - p_center  
            
            x = [p00[0],p01[0]]
            y = [p00[1],p01[1]]        
            ax1.plot(x,y, '-r', c=color,  linewidth=linewidth, alpha = alpha) #this part creates the plane
            
            x = [p11[0],p10[0]]
            y = [p11[1],p10[1]]
            
            ax1.plot(x,y, '-k', c=color,  linewidth=linewidth, alpha = alpha) #this part creates the plane
    
    ankle_x = []
    ankle_y = []
    for i in range(0, len(ppl_ankle_u)):
        #points that are not detected by openpose are assignned -1
        if ppl_ankle_u[i] < 0 or ppl_ankle_v[i] < 0:
            continue
        
        ankle_3d = np.squeeze(single_util.plane_ray_intersection_np([ppl_ankle_u[i]], [ppl_ankle_v[i]], cam_inv, normal, ankleWorld))
        person_plane = (rot_matrix @ ankle_3d) 
        #ax1.scatter(x=person_plane[0], y=person_plane[1], c='green', s=30)
        
        ankle_x.append(person_plane[0])
        ankle_y.append(person_plane[1])
        
        ax1.scatter(x=person_plane[0], y=person_plane[1], c=color_plot, s=30)
        
        
    plt.annotate("Camera position", (0, 0))
    
    img_bl_world = (rot_matrix @ cam_inv @ np.array([0,img_height, 1]))
    img_br_world = (rot_matrix @ cam_inv @ np.array([img_width,img_height, 1]))
    
    img_bl_world = (img_bl_world/np.linalg.norm(img_bl_world))*np.sqrt(2*np.square(scale*line_amount))
    img_br_world = (img_br_world/np.linalg.norm(img_br_world))*np.sqrt(2*np.square(scale*line_amount))
    
    x_lim = max([np.absolute(min(ankle_x + [0])), np.absolute(max(ankle_x + [0]))])

    ax1.set_xlim([-x_lim - 1, x_lim + 1])
    ax1.set_ylim([min(ankle_y + [0]) - 1, max(ankle_y + [0]) + 1])
    ax1.plot([0, img_bl_world[0]], [0, img_bl_world[1]], c='red')
    ax1.plot([0, img_br_world[0]], [0, img_br_world[1]], c='red')
    ax1.set_aspect('equal', adjustable='box')

    return ankle_x, ankle_y, ax1


def display_2d_grid(ppl_ankle_u, ppl_ankle_v, ppl_head_u, ppl_head_v, save_dir, img, from_pickle, scale, line_amount, name, threshold_euc, threshold_cos, h, color_plot = 'green'):
    '''
    Plots the ground plane and line from ankle to head onto an image and also plots the error line from head detection to ransac predicted head, red means it exceeds threshold. 

    Parameters: ppl_ankle_u: list
                    list of dcpose 2d ankle detections u coordinate (coordinates are (u, v))
                ppl_ankle_v: list
                    list of dcpose 2d ankle detections v coordinate (coordinates are (u, v))
                ppl_head_u: list
                    list of dcpose 2d head detections u coordinate (coordinates are (u, v))
                ppl_head_v: list
                    list of dcpose 2d head detections v coordinate (coordinates are (u, v))
                save_dir: string
                    directory to save plots
                img: np.array
                    frame from the sequuence that is used for plotting
                from_pickle: dictionary
                    dictionary that contains the calibration (must contain camera matrix, normal, and plane center (called ankle))
                scale: float
                    size of the squares in the plotted grid (scale = 1 means the squares are 1 meter)
                line_amount: int
                    amount of lines in the ground plane to plot
                name: string
                    subdirectory in save_dir to save plots
                threshold_euc: float
                    Euclidean threshold for inliers
                threshold_cos: float
                    Cosine threshold for inliers
                h: float
                    assumed height of the people    

    Returns:    None
    '''
    
    img_width = img.shape[1]
    img_height = img.shape[0]
            
    ankleWorld = from_pickle['ankleWorld']
    cam_matrix = from_pickle['cam_matrix']
    normal = from_pickle['normal']

    cam_inv = np.linalg.inv(cam_matrix)
    
    rot_matrix = single_util.basis_change_rotation_matrix(cam_matrix, cam_inv, ankleWorld, normal, img_width, img_height)
    
    fig, ax1 = plt.subplots(1, 1)
    fig.suptitle('Ground plane overlay (birds eye view)')   
    
    plane_world = np.squeeze(single_util.plane_ray_intersection_np([img_width/2.0], [img_height], cam_inv, normal, ankleWorld)) 

    #GRAPHING THE IMAGE VIEW
    
    alpha = 1
    color = 'cyan'
    linewidth = 1
    ax1.scatter(x=0.0, y=0.0, c='black', s=30)

    p_center_proj, p_center_3d = single_util.project_point_horiz_bottom(cam_matrix, cam_inv,[0,0], plane_world, normal, img_width, img_height)
    p_center = rot_matrix @ p_center_3d  

    for i in range(-line_amount,line_amount):
            p00, p00_3d = single_util.project_point_horiz_bottom(cam_matrix, cam_inv,[(i)*scale,(-line_amount)*scale], plane_world, normal, img_width, img_height)
            p01, p01_3d = single_util.project_point_horiz_bottom(cam_matrix, cam_inv,[(i)*scale,(line_amount)*scale], plane_world, normal, img_width, img_height)
            p10, p10_3d = single_util.project_point_horiz_bottom(cam_matrix, cam_inv,[(-line_amount)*scale,(i)*scale], plane_world, normal, img_width, img_height)
            p11, p11_3d = single_util.project_point_horiz_bottom(cam_matrix, cam_inv,[(line_amount)*scale,(i)*scale], plane_world, normal, img_width, img_height)
                
            p00 = (rot_matrix @ p00_3d) - p_center 
            p01 = (rot_matrix @ p01_3d) - p_center  
            p10 = (rot_matrix @ p10_3d) - p_center   
            p11 = (rot_matrix @ p11_3d) - p_center  
            
            x = [p00[0],p01[0]]
            y = [p00[1],p01[1]]        
            ax1.plot(x,y, '-r', c=color,  linewidth=linewidth, alpha = alpha) #this part creates the plane
            
            x = [p11[0],p10[0]]
            y = [p11[1],p10[1]]
            
            ax1.plot(x,y, '-k', c=color,  linewidth=linewidth, alpha = alpha) #this part creates the plane
    
    ankle_x = []
    ankle_y = []
    for i in range(0, len(ppl_ankle_u)):
        #points that are not detected by openpose are assignned -1
        if ppl_ankle_u[i] < 0 or ppl_ankle_v[i] < 0 or ppl_head_u[i] < 0 or ppl_head_v[i] < 0:
            continue
        
        ankle_3d = np.squeeze(single_util.plane_ray_intersection_np([ppl_ankle_u[i]], [ppl_ankle_v[i]], cam_inv, normal, ankleWorld))
        person_plane = (rot_matrix @ ankle_3d) 
        #ax1.scatter(x=person_plane[0], y=person_plane[1], c='green', s=30)
        
        ankle_x.append(person_plane[0])
        ankle_y.append(person_plane[1])

        
        #############
        ankle_ppl_2d = single_util.perspective_transformation(cam_matrix, ankle_3d)
        head_ppl_2d = single_util.perspective_transformation(cam_matrix, np.array(ankle_3d) + np.squeeze(normal)*h)

        head_vect_pred = np.array([head_ppl_2d[0], head_ppl_2d[1]]) - np.array([ankle_ppl_2d[0], ankle_ppl_2d[1]])
        head_vect_ground = np.array([ppl_head_u[i], ppl_head_v[i]]) - np.array([ppl_ankle_u[i], ppl_ankle_v[i]])

        head_vect_ground_norm = np.linalg.norm(head_vect_ground)
        
        error_cos = 1.0 - single_util.matrix_cosine(np.expand_dims(head_vect_pred, axis = 0), np.expand_dims(head_vect_ground, axis = 0))
        error_norm = np.linalg.norm(np.array([head_ppl_2d[0], head_ppl_2d[1]]) - np.array([ppl_head_u[i], ppl_head_v[i]]))/head_vect_ground_norm
        
        if error_cos < threshold_cos and error_norm < threshold_euc:
            ax1.scatter(x=person_plane[0], y=person_plane[1], c=color_plot, s=30)
        
        else:
            ax1.scatter(x=person_plane[0], y=person_plane[1], c='red', s=30)
        
    plt.annotate("Camera position", (0, 0))
    
    img_bl_world = (rot_matrix @ cam_inv @ np.array([0,img_height/2.0, 1]))
    img_br_world = (rot_matrix @ cam_inv @ np.array([img_width,img_height/2.0, 1]))
    print(np.degrees(single_util.angle_between(img_bl_world, img_br_world)), " ANGlE !!!!!!!!!!!!!!!!!!!!!")

    img_bl_world = (img_bl_world/np.linalg.norm(img_bl_world))*np.sqrt(2*np.square(scale*line_amount))
    img_br_world = (img_br_world/np.linalg.norm(img_br_world))*np.sqrt(2*np.square(scale*line_amount))
    
    x_lim = max([np.absolute(min(ankle_x + [0])), np.absolute(max(ankle_x + [0]))])

    ax1.set_xlim([-x_lim - 1, x_lim + 1])
    ax1.set_ylim([min(ankle_y + [0]) - 1, max(ankle_y + [0]) + 1])
    ax1.plot([0, img_bl_world[0]], [0, img_bl_world[1]], c='red')
    ax1.plot([0, img_br_world[0]], [0, img_br_world[1]], c='red')
    ax1.set_aspect('equal', adjustable='box')
    
    fig.savefig(save_dir + '/' + 'topview_' + str(name) + '.png')
    plt.close('all')

    return ankle_x, ankle_y

def display_2d_plane(ankles_array, save_dir, img, from_pickle, scale, line_amount, name, x_lim = None, y_lim_lower = None, y_lim_upper = None, sync = [], sync1 = [], gt_sync = [], color_array = [], cum_sum = []):
    '''
    Plots the ground plane and line from ankle to head onto an image and also plots the error line from head detection to ransac predicted head, red means it exceeds threshold. 

    Parameters: ppl_ankle_u: list
                    list of dcpose 2d ankle detections u coordinate (coordinates are (u, v))
                ppl_ankle_v: list
                    list of dcpose 2d ankle detections v coordinate (coordinates are (u, v))
                ppl_head_u: list
                    list of dcpose 2d head detections u coordinate (coordinates are (u, v))
                ppl_head_v: list
                    list of dcpose 2d head detections v coordinate (coordinates are (u, v))
                save_dir: string
                    directory to save plots
                img: np.array
                    frame from the sequuence that is used for plotting
                from_pickle: dictionary
                    dictionary that contains the calibration (must contain camera matrix, normal, and plane center (called ankle))
                scale: float
                    size of the squares in the plotted grid (scale = 1 means the squares are 1 meter)
                line_amount: int
                    amount of lines in the ground plane to plot
                name: string
                    subdirectory in save_dir to save plots
                threshold_euc: float
                    Euclidean threshold for inliers
                threshold_cos: float
                    Cosine threshold for inliers
                h: float
                    assumed height of the people    

    Returns:    None
    '''

    colors = cm.rainbow(np.linspace(0, 1, len(ankles_array)))

    img_width = img.shape[1]
    img_height = img.shape[0]
            
    ankleWorld = from_pickle['ankleWorld']
    cam_matrix = from_pickle['cam_matrix']
    normal = from_pickle['normal']

    cam_inv = np.linalg.inv(cam_matrix)
    
    rot_matrix = single_util.basis_change_rotation_matrix(cam_matrix, cam_inv, ankleWorld, normal, img_width, img_height)
    
    fig, ax1 = plt.subplots(1, 1)
    fig.suptitle('Ground plane overlay (birds eye view)')   
    
    plane_world = np.squeeze(single_util.plane_ray_intersection_np([img_width/2.0], [img_height], cam_inv, normal, ankleWorld)) 

    #GRAPHING THE IMAGE VIEW
    
    alpha = 1
    color = 'cyan'
    linewidth = 1
    ax1.scatter(x=0.0, y=0.0, c='black', s=30)

    p_center_proj, p_center_3d = single_util.project_point_horiz_bottom(cam_matrix, cam_inv,[0,0], plane_world, normal, img_width, img_height)

    p_center = rot_matrix @ p_center_3d  
    for i in range(-line_amount,line_amount):
            p00, p00_3d = single_util.project_point_horiz_bottom(cam_matrix, cam_inv,[(i)*scale,(-line_amount)*scale], plane_world, normal, img_width, img_height)
            p01, p01_3d = single_util.project_point_horiz_bottom(cam_matrix, cam_inv,[(i)*scale,(line_amount)*scale], plane_world, normal, img_width, img_height)
            p10, p10_3d = single_util.project_point_horiz_bottom(cam_matrix, cam_inv,[(-line_amount)*scale,(i)*scale], plane_world, normal, img_width, img_height)
            p11, p11_3d = single_util.project_point_horiz_bottom(cam_matrix, cam_inv,[(line_amount)*scale,(i)*scale], plane_world, normal, img_width, img_height)
                
            p00 = (rot_matrix @ p00_3d) - p_center
            p01 = (rot_matrix @ p01_3d) - p_center  
            p10 = (rot_matrix @ p10_3d) - p_center   
            p11 = (rot_matrix @ p11_3d) - p_center 
            
            x = [p00[0],p01[0]]
            y = [p00[1],p01[1]]        
            ax1.plot(x,y, '-r', c=color,  linewidth=linewidth, alpha = alpha) #this part creates the plane
            
            x = [p11[0],p10[0]]
            y = [p11[1],p10[1]]
            
            ax1.plot(x,y, '-k', c=color,  linewidth=linewidth, alpha = alpha) #this part creates the plane
    
    ankle_x = []
    ankle_y = []
    color_index = 0
    for ankles in ankles_array:
        for j in range(0, np.array(ankles).shape[1]):
            #points that are not detected by openpose are assignned -1
            ax1.scatter(x=ankles[0][j], y=ankles[1][j], c=colors[color_index].reshape(1,-1), s=30)
            
            ankle_x.append(ankles[0][j])
            ankle_y.append(ankles[1][j])
        
        color_index = color_index + 1

    ankle_ind = 1

    label_index = 0

    if len(color_array) > 0:

        handles = []

        for c_array in color_array:
            error_c = 0
            for c in c_array:
                patch = mpatches.Patch(color='green', label= str(c) + ' ' + str(round(cum_sum[label_index][error_c], 3)) + " off", alpha = 1.0/1.1**(c))
                handles.append(patch)
                error_c = error_c + 1
        label_index = label_index + 1
        plt.legend(handles=handles,bbox_to_anchor=(1.04,1), loc="upper left")

    for sy in range(len(sync)):
        
        for s in range(len(list(sync[sy]))):
            if sync[sy][s] == -1:
                continue
            frames_off = 0.0
            if len(gt_sync) > 0:
                frames_off = np.abs(sync[sy][s] - gt_sync[sy][s])

            ax1.plot([ankles_array[0][0][sync1[sy][s]], ankles_array[1][0][sync[sy][s]]], [ankles_array[0][1][sync1[sy][s]], ankles_array[1][1][sync[sy][s]]], c='green', alpha = 1.0/1.1**(frames_off))

        ankle_ind = ankle_ind + 1
    
    plt.title(name)
    plt.annotate("Camera position", (0, 0))
    
    img_bl_world = (rot_matrix @ cam_inv @ np.array([0,img_height, 1]))
    img_br_world = (rot_matrix @ cam_inv @ np.array([img_width,img_height, 1]))
    
    img_bl_world = (img_bl_world/np.linalg.norm(img_bl_world))*np.sqrt(2*np.square(scale*line_amount))
    img_br_world = (img_br_world/np.linalg.norm(img_br_world))*np.sqrt(2*np.square(scale*line_amount))
    
    if x_lim is None:
        x_lim = max([np.absolute(min(ankle_x + [0])), np.absolute(max(ankle_x + [0]))])

    if y_lim_lower is None:
        y_lim_lower = min(ankle_y + [0])

    if y_lim_upper is None:
        y_lim_upper = max(ankle_y + [0])

    ax1.set_xlim([-x_lim - 1, x_lim + 1])
    ax1.set_ylim([y_lim_lower - 1, y_lim_upper + 1])
    ax1.plot([0, img_bl_world[0]], [0, img_bl_world[1]], c='red')
    ax1.plot([0, img_br_world[0]], [0, img_br_world[1]], c='red')
    ax1.set_aspect('equal', adjustable='box')
        
    fig.savefig(save_dir + '/' + 'combined_top_' + str(name) + '.png')
    plt.close('all')

    return ankle_x, ankle_y

def grid_3d(ax1, img_width, img_height, from_pickle, scale, line_amount, rot_matrix):
    '''
    Plots the ground plane and line from ankle to head onto an image and also plots the error line from head detection to ransac predicted head, red means it exceeds threshold. 

    Parameters: ppl_ankle_u: list
                    list of dcpose 2d ankle detections u coordinate (coordinates are (u, v))
                ppl_ankle_v: list
                    list of dcpose 2d ankle detections v coordinate (coordinates are (u, v))
                ppl_head_u: list
                    list of dcpose 2d head detections u coordinate (coordinates are (u, v))
                ppl_head_v: list
                    list of dcpose 2d head detections v coordinate (coordinates are (u, v))
                img: np.array
                    frame from the sequuence that is used for plotting
                from_pickle: dictionary
                    dictionary that contains the calibration (must contain camera matrix, normal, and plane center (called ankle))
                scale: float
                    size of the squares in the plotted grid (scale = 1 means the squares are 1 meter)
                line_amount: int
                    amount of lines in the ground plane to plot
                h: float
                    assumed height of the people    

    Returns:    None
    '''
    #print(rot_matrix.shape, " 312321321312")
    '''
    if 'init_point' in from_pickle.keys():
        ankleWorld = from_pickle['init_point']
    else:
        ankleWorld = from_pickle['ankleWorld']
    cam_matrix = from_pickle['cam_matrix']
    normal = from_pickle['normal']
    '''
    
    if 'init_point' in from_pickle.keys():
        ankleWorld = from_pickle['init_point'].detach().numpy()
    else:
        ankleWorld = from_pickle['ankleWorld'].detach().numpy()
    cam_matrix = from_pickle['cam_matrix'].detach().numpy()
    normal = from_pickle['normal'].detach().numpy()
    

    cam_inv = np.linalg.inv(cam_matrix)
    
    plane_world = np.squeeze(single_util.plane_ray_intersection_np([img_width/2.0], [img_height], cam_inv, normal, ankleWorld)) 
    
    color = 'cyan'
    
    point_3d = []
    
    #rot_matrix = single_util.basis_change_rotation_matrix(cam_matrix, cam_inv, ankleWorld, normal, img_width, img_height)
    
    for i in range(-line_amount,line_amount):
            p00, p00_3d = single_util.project_point_horiz_bottom(cam_matrix, cam_inv,[(i)*scale,(0)*scale], plane_world, normal, img_width, img_height)
            p01, p01_3d = single_util.project_point_horiz_bottom(cam_matrix, cam_inv,[(i)*scale,(line_amount)*scale], plane_world, normal, img_width, img_height)
            p10, p10_3d = single_util.project_point_horiz_bottom(cam_matrix, cam_inv,[(-line_amount)*scale,(i)*scale], plane_world, normal, img_width, img_height)
            p11, p11_3d = single_util.project_point_horiz_bottom(cam_matrix, cam_inv,[(line_amount)*scale,(i)*scale], plane_world, normal, img_width, img_height)
            
            #point_3d = point_3d + [p00_3d, p01_3d, p10_3d, p11_3d]
            
            p00_plane = rot_matrix @ p00_3d   
            p01_plane = rot_matrix @ p01_3d  
            p10_plane = rot_matrix @ p10_3d   
            p11_plane = rot_matrix @ p11_3d

            #print(p00_plane.shape , "321321321")

            point_3d = point_3d + [p00_plane, p01_plane, p10_plane, p11_plane]

            x = [p00_plane[0],p01_plane[0]]
            y = [p00_plane[1],p01_plane[1]]      
            z = [p00_plane[2],p01_plane[2]]
            ax1.plot(x,y,z, c=color)
            
            x = [p11_plane[0],p10_plane[0]]
            y = [p11_plane[1],p10_plane[1]]      
            z = [p11_plane[2],p10_plane[2]]
            
            ax1.plot(x,y,z, c=color)
            '''
            x = [p00_3d[0],p01_3d[0]]
            y = [p00_3d[1],p01_3d[1]]      
            z = [p00_3d[2],p01_3d[2]]
            ax1.plot(x,y,z, c=color)
            
            x = [p11_3d[0],p10_3d[0]]
            y = [p11_3d[1],p10_3d[1]]      
            z = [p11_3d[2],p10_3d[2]]
            '''
            '''
            if p11_plane[1] > 0 or p10_plane[1] > 0:
                ax1.plot(x,y,z, c=color)
            '''

    return ax1, point_3d

def display_3d_grid(ppl_ankle_u, ppl_ankle_v, ppl_head_u, ppl_head_v, img, from_pickle, scale, line_amount, h):
    '''
    Plots the ground plane and line from ankle to head onto an image and also plots the error line from head detection to ransac predicted head, red means it exceeds threshold. 

    Parameters: ppl_ankle_u: list
                    list of dcpose 2d ankle detections u coordinate (coordinates are (u, v))
                ppl_ankle_v: list
                    list of dcpose 2d ankle detections v coordinate (coordinates are (u, v))
                ppl_head_u: list
                    list of dcpose 2d head detections u coordinate (coordinates are (u, v))
                ppl_head_v: list
                    list of dcpose 2d head detections v coordinate (coordinates are (u, v))
                img: np.array
                    frame from the sequuence that is used for plotting
                from_pickle: dictionary
                    dictionary that contains the calibration (must contain camera matrix, normal, and plane center (called ankle))
                scale: float
                    size of the squares in the plotted grid (scale = 1 means the squares are 1 meter)
                line_amount: int
                    amount of lines in the ground plane to plot
                h: float
                    assumed height of the people    

    Returns:    None
    '''
    
    img_width = img.shape[1]
    img_height = img.shape[0]
            
    ankleWorld = from_pickle['ankleWorld']
    cam_matrix = from_pickle['cam_matrix']
    normal = from_pickle['normal']

    cam_inv = np.linalg.inv(cam_matrix)
    
    fig3d = plt.figure(200)
    fig3d.set_size_inches((5, 3))

    ax1 = fig3d.gca(projection='3d')
    fig3d.suptitle('ax3')  
    
    plane_world = np.squeeze(single_util.plane_ray_intersection_np([img_width/2.0], [img_height], cam_inv, normal, ankleWorld)) 
    
    color = 'cyan'
    
    point_3d = []
    
    rot_matrix = single_util.basis_change_rotation_matrix(cam_matrix, cam_inv, ankleWorld, normal, img_width, img_height)
    
    for i in range(-line_amount,line_amount):
            p00, p00_3d = single_util.project_point_horiz_bottom(cam_matrix, cam_inv,[(i)*scale,(0)*scale], plane_world, normal, img_width, img_height)
            p01, p01_3d = single_util.project_point_horiz_bottom(cam_matrix, cam_inv,[(i)*scale,(line_amount)*scale], plane_world, normal, img_width, img_height)
            p10, p10_3d = single_util.project_point_horiz_bottom(cam_matrix, cam_inv,[(-line_amount)*scale,(i)*scale], plane_world, normal, img_width, img_height)
            p11, p11_3d = single_util.project_point_horiz_bottom(cam_matrix, cam_inv,[(line_amount)*scale,(i)*scale], plane_world, normal, img_width, img_height)
            
            point_3d = point_3d + [p00_3d, p01_3d, p10_3d, p11_3d]
            
            p00_plane = rot_matrix @ p00_3d   
            p01_plane = rot_matrix @ p01_3d  
            p10_plane = rot_matrix @ p10_3d   
            p11_plane = rot_matrix @ p11_3d
            
            x = [p00_3d[0],p01_3d[0]]
            y = [p00_3d[1],p01_3d[1]]      
            z = [p00_3d[2],p01_3d[2]]
            ax1.plot(x,y,z, c=color)
            
            x = [p11_3d[0],p10_3d[0]]
            y = [p11_3d[1],p10_3d[1]]      
            z = [p11_3d[2],p10_3d[2]]
            if p11_plane[1] > 0 or p10_plane[1] > 0:
                ax1.plot(x,y,z, c=color)
        
    ax1.scatter(0,0,0, 'black')
    corner = draw_frustum(ax1, img_width, img_height, cam_matrix, np.sqrt(2*np.square(scale*line_amount)))
    
    point_3d = point_3d + [corner[0], corner[1], corner[2], corner[3], [0,0,0]]
    for i in range(0, len(ppl_ankle_u)):
        #points that are not detected by openpose are assignned -1
        if ppl_ankle_u[i] < 0 or ppl_ankle_v[i] < 0 or ppl_head_u[i] < 0 or ppl_head_v[i] < 0:
            continue

        ankle_3d = np.squeeze(single_util.plane_ray_intersection_np([ppl_ankle_u[i]], [ppl_ankle_v[i]], cam_inv, normal, ankleWorld)) 
        
        head_3d = ankle_3d - normal*h
        point_3d.append(ankle_3d)
    
        ax1.scatter(ankle_3d[0],ankle_3d[1],ankle_3d[2], 'black')
        ax1.scatter(head_3d[0],head_3d[1],head_3d[2], 'black')
        ax1.plot([ankle_3d[0], head_3d[0]],[ankle_3d[1], head_3d[1]],[ankle_3d[2], head_3d[2]], 'blue')
    
    ax1.plot([0,0], [0,0], [0, scale*(line_amount/2)], 'green')
    set_equal_3d(np.array(point_3d), ax1)
    plt.show()

def draw_frustum_multi(ax, img_width, img_height, cam_matrix, extrinsic, scale):
    '''
    computes and plots the top left, top right, bottom left, bottom right coordinates of the view frustum.

    Parameters: ax: matplotlib axes object
                    All coordinates are plotted on this object
                img_width: Float
                    Width of image
                img_height: Float
                    Height of image
                normal: (3,) np.array
                init_point: (3,) np.array 
                    Point used to initalize the ground plane
                cam_matrix:(3,3) np.array
                    intrinsic camera matrix
                scale: Float
                    Scale of the grid. For example scale = 1 means each square is 1 meter by 1 meter.
    Returns:    output: np.array()
        np array that contains the top left, top right, bottom left, bottom right coordinates of the view frustum
    '''
    cam_inv = np.linalg.inv(cam_matrix)
    extrinsic_inv = extrinsic[:, :3]#np.linalg.inv(extrinsic[:, :3])
    translation_inv = -extrinsic[:, 3]
    
    img_bl_world = cam_inv @ np.array([0,img_height, 1])
    img_br_world = cam_inv @ np.array([img_width,img_height, 1])
    
    img_tl_world = cam_inv @ np.array([0,0, 1])
    img_tr_world = cam_inv @ np.array([img_width,0, 1])
    
    img_bl_world = (extrinsic_inv @ (img_bl_world/np.linalg.norm(img_bl_world))*scale) + translation_inv
    img_br_world = (extrinsic_inv @ (img_br_world/np.linalg.norm(img_br_world))*scale) + translation_inv
    img_tl_world = (extrinsic_inv @ (img_tl_world/np.linalg.norm(img_tl_world))*scale) + translation_inv
    img_tr_world = (extrinsic_inv @ (img_tr_world/np.linalg.norm(img_tr_world))*scale) + translation_inv
    
    x = [img_bl_world[0],img_br_world[0]]
    y = [img_bl_world[1],img_br_world[1]]      
    z = [img_bl_world[2],img_br_world[2]]
            
    ax.plot(x,y,z, c='red')
    
    x = [img_tl_world[0],img_tr_world[0]]
    y = [img_tl_world[1],img_tr_world[1]]      
    z = [img_tl_world[2],img_tr_world[2]]
            
    ax.plot(x,y,z, c='blue')
    
    x = [img_bl_world[0],img_tl_world[0]]
    y = [img_bl_world[1],img_tl_world[1]]      
    z = [img_bl_world[2],img_tl_world[2]]
            
    ax.plot(x,y,z, c='green')
    
    x = [img_br_world[0],img_tr_world[0]]
    y = [img_br_world[1],img_tr_world[1]]      
    z = [img_br_world[2],img_tr_world[2]]
            
    ax.plot(x,y,z, c='black')
    
    x = [translation_inv[0],img_br_world[0]]
    y = [translation_inv[1],img_br_world[1]]      
    z = [translation_inv[2],img_br_world[2]]
            
    ax.plot(x,y,z, c='red')
    
    x = [translation_inv[0],img_bl_world[0]]
    y = [translation_inv[1],img_bl_world[1]]      
    z = [translation_inv[2],img_bl_world[2]]
            
    ax.plot(x,y,z, c='red')
    
    x = [translation_inv[0],img_tl_world[0]]
    y = [translation_inv[1],img_tl_world[1]]      
    z = [translation_inv[2],img_tl_world[2]]
            
    ax.plot(x,y,z, c='red')
    
    x = [translation_inv[0],img_tr_world[0]]
    y = [translation_inv[1],img_tr_world[1]]      
    z = [translation_inv[2],img_tr_world[2]]
            
    ax.plot(x,y,z, c='red')

    x_axis = extrinsic[:, 0]
    y_axis = extrinsic[:, 1]
    z_axis = extrinsic[:, 2]
    #print(y_axis, " y axis")

    ax.plot([-extrinsic[:, 3][0], -extrinsic[:, 3][0] + x_axis[0]], [-extrinsic[:, 3][1], -extrinsic[:, 3][1] + x_axis[1]], [-extrinsic[:, 3][2], -extrinsic[:, 3][2] + x_axis[2]], c='blue')
    ax.plot([-extrinsic[:, 3][0], -extrinsic[:, 3][0] + y_axis[0]], [-extrinsic[:, 3][1], -extrinsic[:, 3][1] + y_axis[1]], [-extrinsic[:, 3][2], -extrinsic[:, 3][2] + y_axis[2]], c='green')
    ax.plot([-extrinsic[:, 3][0], -extrinsic[:, 3][0] + z_axis[0]], [-extrinsic[:, 3][1], -extrinsic[:, 3][1] + z_axis[1]], [-extrinsic[:, 3][2], -extrinsic[:, 3][2] + z_axis[2]], c='red')
    
    return np.array([img_bl_world, img_br_world, img_tl_world, img_tr_world])

def draw_frustum(ax, img_width, img_height, cam_matrix, scale):
    '''
    computes and plots the top left, top right, bottom left, bottom right coordinates of the view frustum.

    Parameters: ax: matplotlib axes object
                    All coordinates are plotted on this object
                img_width: Float
                    Width of image
                img_height: Float
                    Height of image
                normal: (3,) np.array
                init_point: (3,) np.array 
                    Point used to initalize the ground plane
                cam_matrix:(3,3) np.array
                    intrinsic camera matrix
                scale: Float
                    Scale of the grid. For example scale = 1 means each square is 1 meter by 1 meter.
    Returns:    output: np.array()
        np array that contains the top left, top right, bottom left, bottom right coordinates of the view frustum
    '''
    cam_inv = np.linalg.inv(cam_matrix)
    
    img_bl_world = cam_inv @ np.array([0,img_height, 1])
    img_br_world = cam_inv @ np.array([img_width,img_height, 1])
    
    img_tl_world = cam_inv @ np.array([0,0, 1])
    img_tr_world = cam_inv @ np.array([img_width,0, 1])
    
    img_bl_world = (img_bl_world/np.linalg.norm(img_bl_world))*scale
    img_br_world = (img_br_world/np.linalg.norm(img_br_world))*scale
    img_tl_world = (img_tl_world/np.linalg.norm(img_tl_world))*scale
    img_tr_world = (img_tr_world/np.linalg.norm(img_tr_world))*scale
    
    x = [img_bl_world[0],img_br_world[0]]
    y = [img_bl_world[1],img_br_world[1]]      
    z = [img_bl_world[2],img_br_world[2]]
            
    ax.plot(x,y,z, c='red')
    
    x = [img_tl_world[0],img_tr_world[0]]
    y = [img_tl_world[1],img_tr_world[1]]      
    z = [img_tl_world[2],img_tr_world[2]]
            
    ax.plot(x,y,z, c='blue')
    
    x = [img_bl_world[0],img_tl_world[0]]
    y = [img_bl_world[1],img_tl_world[1]]      
    z = [img_bl_world[2],img_tl_world[2]]
            
    ax.plot(x,y,z, c='green')
    
    x = [img_br_world[0],img_tr_world[0]]
    y = [img_br_world[1],img_tr_world[1]]      
    z = [img_br_world[2],img_tr_world[2]]
            
    ax.plot(x,y,z, c='black')
    
    x = [0,img_br_world[0]]
    y = [0,img_br_world[1]]      
    z = [0,img_br_world[2]]
            
    ax.plot(x,y,z, c='red')
    
    x = [0,img_bl_world[0]]
    y = [0,img_bl_world[1]]      
    z = [0,img_bl_world[2]]
            
    ax.plot(x,y,z, c='red')
    
    x = [0,img_tl_world[0]]
    y = [0,img_tl_world[1]]      
    z = [0,img_tl_world[2]]
            
    ax.plot(x,y,z, c='red')
    
    x = [0,img_tr_world[0]]
    y = [0,img_tr_world[1]]      
    z = [0,img_tr_world[2]]
            
    ax.plot(x,y,z, c='red')
    
    return np.array([img_bl_world, img_br_world, img_tl_world, img_tr_world])
    
    
def set_equal_3d(point_3d, ax):
    '''
    Creates a bounding box in the 3d plot to make sure every axis is on the same scale.

    Parameters: point_3d: np.array 
                    All the 3d point are stored in this array
                ax: matplotlib axes object
                    All coordinates are plotted on this object
    Returns:    None
    '''
    point_3d = np.array(point_3d)
    print(point_3d.shape, " 312132132")
    X = point_3d[:,0]
    Y = point_3d[:,1]
    Z = point_3d[:,2]

    # Create cubic bounding box to simulate equal aspect ratio
    max_range = np.array([X.max()-X.min(), Y.max()-Y.min(), Z.max()-Z.min()]).max()
    Xb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][0].flatten() + 0.5*(X.max()+X.min())
    Yb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][1].flatten() + 0.5*(Y.max()+Y.min())
    Zb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][2].flatten() + 0.5*(Z.max()+Z.min())

    for xb, yb, zb in zip(Xb, Yb, Zb):
        ax.plot([xb], [yb], [zb], 'w')