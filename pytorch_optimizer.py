import torch
import numpy as np
import timeit
import plotting
import matplotlib.pyplot as plt

import pytorch3d
from pytorch3d.renderer import PerspectiveCameras

from torch.optim import Adam
from torch.optim import SGD

from pytorch3d.transforms.so3 import (
    so3_exponential_map,
    so3_relative_angle,
)

import geometry

def single_view_optimization_torch_3d(init_value, au, av, hu, hv, t1, t2, h, save_dir, img, threshold_euc, threshold_cos, focal_lr = 1e-1, point_lr = 1e-3, iter = 10000, line_amount = 50, plot_scale = 1, conf_array = None):
    '''
    This function optimizes the camerea paramters after ransac using gradient descent.

    Parameters: init_value: Dictionary
                    a dictionary of the form: init_value = {'focal_predicted': focal_predicted, 'normal': normal, 'ankleWorld': ankleWorld, 'ankles': np.array(ankles)}
                au: list
                    list of dcpose 2d ankle detections u coordinate (coordinates are (u, v))
                av: list
                    list of dcpose 2d ankle detections v coordinate (coordinates are (u, v))
                hu: list
                    list of dcpose 2d head detections u coordinate (coordinates are (u, v))
                hv: list
                    list of dcpose 2d head detections v coordinate (coordinates are (u, v))
                t1: float
                    x focal center
                t2: float
                    y focal center
                h: float
                    assumed neck height of the people
                save_dir
                    directory to save plots
                img: matplotlib.image 
                    frame from sequence to be plotted on. If None, then there will be no plotting
                threshold_euc: float
                    Euclidean threshold for RANSAC.
                threshold_cos: float
                    Cosine threshold for RANSAC.
                focal_lr: float
                    Learning rate for the focal length.
                point_lr: float
                    Learning rate for the plane center.
                iter: int
                    Number of iterations for pytorch optimizer
                line_amount: int
                    amount of lines in the ground plane to plot
                plot_scale: int
                    size of the squares in the plotted grid (scale = 1 means the squares are 1 meter)

    Returns: focal_predicted: float
                optimized focal length
             normal.cpu().detach().numpy(): (3,) np.array
                normal vector
             ankleWorld: (3,) np.array
                optimized plane center
             error_array: list
                list of objective function values throughout the iterations
             focal_array: list
                list of focal lengths values throughout the iterations
             normal_array: list
                list of normal vector values throughout the iterations
    '''
    au = torch.tensor(au).double()
    av = torch.tensor(av).double()
    hu = torch.tensor(hu).double()
    hv = torch.tensor(hv).double()

    focal_predicted = init_value['focal_predicted']
    normal_vector = torch.from_numpy(init_value['normal'])

    #print(normal, " Normal")

    ankleWorld = init_value['ankleWorld']
    intrinsic = np.array([normal_vector[0], normal_vector[2], focal_predicted, ankleWorld[0], ankleWorld[1], ankleWorld[2]])  
    
    #ankles = init_value['ankles']
    #ankles = np.array(ankles) - np.tile(np.array([ankleWorld[0],ankleWorld[1], ankleWorld[2]]), (ankles.shape[0], 1))
    #ankles = ankles.reshape(ankles.shape[0]*ankles.shape[1])
    init = list(intrinsic)# + list(ankles)

    start = timeit.timeit()
    print("start")
    plane_matrix, basis_matrix = geometry.find_plane_matrix(init_value['normal'], np.linalg.inv(init_value['cam_matrix']),init_value['ankleWorld'], 2*t1, 2*t2)
    print(plane_matrix)
    print(normal_vector)

    ############################
    print(torch.transpose(torch.from_numpy(plane_matrix[:3,:3]), 0, 1), " PLANE MATRIX ")
    camera_array = PerspectiveCameras(
    R=torch.unsqueeze(pytorch3d.transforms.matrix_to_axis_angle(torch.transpose(torch.from_numpy(plane_matrix[:3,:3]), 0, 1)).double(), dim = 0),
    T=torch.unsqueeze(torch.tensor(init[3:6]).double(), dim = 0),
    focal_length=torch.tensor([init[2]]).double(), 
    principal_point=torch.tensor([[t1,t2]]).double()
    )
    print(camera_array)
    R_matrix = camera_array.R
    point = camera_array.T[0]
    focal = camera_array.focal_length[:,0]
    print(focal.shape, " FOCAL")    
    #stop
    error_array = []
    
    point.requires_grad = True
    focal.requires_grad = True
    #############
    
    optimizer = Adam([
        point,
        focal,
        R_matrix
    ], lr=0.1)
    ############################
    
    ankle_2d_dc = torch.from_numpy(np.stack((au,av)))
    head_2d_dc = torch.from_numpy(np.stack((hu,hv)))

    min_error = np.inf
    min_weight = None
    min_focal = None
    min_point = None
    error_array = []
    focal_array = []
    normal_array = []

    if conf_array is not None:
        conf_array = torch.from_numpy(conf_array/np.sum(conf_array))

    print("*** START LOOP ***")
    #from_pickle = {'ankles': ankles, 'cam_matrix': np.array([[focal_predicted, 0 , t1],[0, focal_predicted, t2],[0 , 0, 1]]), 'normal': normal.cpu().detach().numpy(), 'ankleWorld': [point[0].detach().numpy().item(),  point[1].detach().numpy().item(), point[2].detach().numpy().item()]}
    from_pickle = {'cam_matrix': np.array([[focal_predicted, 0 , t1],[0, focal_predicted, t2],[0 , 0, 1]]), 'normal': normal_vector.cpu().detach().numpy(), 'ankleWorld': [point[0].detach().numpy().item(),  point[1].detach().numpy().item(), point[2].detach().numpy().item()]}
    if img is not None:
        plotting.plot_plane(au, av, hu, hv, save_dir, img, from_pickle, plot_scale, line_amount, 'init', threshold_euc, threshold_cos, h)
        plotting.display_2d_grid(au, av, hu, hv, save_dir, img, from_pickle, plot_scale, line_amount, 'init', threshold_euc, threshold_cos, h)
    for j in range(0, 1):
        
        for i in range(iter):
            #print(so3_exponential_map(R_matrix), " EXPONENTIAL MAPPPPP")
            normal = so3_exponential_map(R_matrix)[0] @ torch.tensor([0,0,1]).double()
            #print(normal, " NOMRAL")
            cam_matrix = torch.squeeze(torch.stack([torch.stack([focal, torch.zeros(1).double(), torch.tensor([t1]).double()]), torch.stack([torch.zeros(1).double(), focal, torch.tensor([t2]).double()]), torch.stack([torch.zeros(1).double(), torch.zeros(1).double(), torch.ones(1).double()])], dim = 0))
            cam_inv = torch.squeeze(torch.stack([torch.stack([1/focal, torch.zeros(1).double(), torch.tensor([-t1]).double()/focal]), torch.stack([torch.zeros(1).double(), 1/focal, torch.tensor([-t2]).double()/focal]), torch.stack([torch.zeros(1).double(), torch.zeros(1).double(), torch.ones(1).double()])], dim = 0))
            
            point_2d = torch.from_numpy(np.stack((au,av, torch.ones(len(au)))))
            
            ankles = torch.matmul(cam_inv, point_2d)
            
            normal_dot_ray = torch.matmul(normal, ankles)
            scale = torch.abs(torch.div(torch.squeeze(torch.dot(normal, point).repeat(len(au), 1)), normal_dot_ray)).repeat(3, 1)
            ankles = torch.transpose(torch.mul(scale,ankles), 0, 1)

            head_3d = ankles + h*normal.repeat(len(au), 1)
            head_2d = torch.div((torch.narrow(torch.transpose(torch.matmul(cam_matrix, torch.transpose(head_3d, 0, 1)), 0 , 1), 1, 0, 2)), head_3d[:, 2:3])

            height_2d = torch.norm(torch.transpose(head_2d_dc, 0, 1) - torch.transpose(ankle_2d_dc, 0, 1), dim = 1)

            head_2d_error = torch.div(torch.norm(head_2d - torch.transpose(head_2d_dc, 0, 1), dim = 1), height_2d)

            if conf_array is not None:
                error_head = torch.mean(head_2d_error*conf_array)
            else: 
                error_head = torch.mean(head_2d_error)

            error = error_head
            error_array.append(error.detach().numpy().item())

            if min_error > error:
                min_error = error.item()

                min_focal = focal[0].detach().numpy().item()
                min_point = [point[0].detach().numpy().item(),  point[1].detach().numpy().item(), point[2].detach().numpy().item()]
            
            
            if i % 1000 == 0 and img is not None:

                fig, ax1 = plt.subplots(1, 1)
                fig1, ax2 = plt.subplots(1, 1)

                ax1.plot(list(range(len(error_array))),error_array, c = 'blue')
                ax2.plot(list(range(len(focal_array))),focal_array, c = 'blue')

                focal_predicted = torch.clone(focal[0]).detach().numpy().item()

                #from_pickle = {'ankles': ankles, 'cam_matrix': cam_matrix.detach().numpy(), 'normal': normal.cpu().detach().numpy(), 'ankleWorld': [point[0].detach().numpy().item(),  point[1].detach().numpy().item(), point[2].detach().numpy().item()]}
                from_pickle = {'cam_matrix': cam_matrix.detach().numpy(), 'normal': normal.cpu().detach().numpy(), 'ankleWorld': [point[0].detach().numpy().item(),  point[1].detach().numpy().item(), point[2].detach().numpy().item()]}
                if img is not None:
                    plotting.plot_plane(au, av, hu, hv, save_dir, img, from_pickle, plot_scale, line_amount, i, threshold_euc, threshold_cos, h)
                    plotting.display_2d_grid(au, av, hu, hv, save_dir, img, from_pickle, plot_scale, line_amount, i, threshold_euc, threshold_cos, h)
                print(error, " iteration", i)   

                fig.savefig(save_dir + '/error.png')
                fig1.savefig(save_dir + '/focal.png')
                plt.close('all')
            
            focal_predicted = focal[0].detach().numpy().item()
            
            focal_array.append(focal_predicted)
            normal_array.append(normal.detach().numpy())

            optimizer.zero_grad()
            error.backward()

            optimizer.step()

    end = timeit.timeit()
    print(point, " end")
    print(end - start, " TIME")
    print("*******************")
    #print(ankles)
    print(min_error, " min error")
    focal_predicted = min_focal
    print(point[0].detach().numpy().item(), " HIIII")
    ankleWorld = min_point
    normal = so3_exponential_map(R_matrix)[0] @ torch.tensor([0,0,1]).double()
    return focal_predicted, normal.cpu().detach().numpy(), ankleWorld, error_array, focal_array, normal_array

def single_view_optimization_torch_3d_old(init_value, au, av, hu, hv, t1, t2, h, save_dir, img, threshold_euc, threshold_cos, focal_lr = 1e-1, point_lr = 1e-3, iter = 10000, line_amount = 50, plot_scale = 1, conf_array = None):
    '''
    This function optimizes the camerea paramters after ransac using gradient descent.

    Parameters: init_value: Dictionary
                    a dictionary of the form: init_value = {'focal_predicted': focal_predicted, 'normal': normal, 'ankleWorld': ankleWorld, 'ankles': np.array(ankles)}
                au: list
                    list of dcpose 2d ankle detections u coordinate (coordinates are (u, v))
                av: list
                    list of dcpose 2d ankle detections v coordinate (coordinates are (u, v))
                hu: list
                    list of dcpose 2d head detections u coordinate (coordinates are (u, v))
                hv: list
                    list of dcpose 2d head detections v coordinate (coordinates are (u, v))
                t1: float
                    x focal center
                t2: float
                    y focal center
                h: float
                    assumed neck height of the people
                save_dir
                    directory to save plots
                img: matplotlib.image 
                    frame from sequence to be plotted on. If None, then there will be no plotting
                threshold_euc: float
                    Euclidean threshold for RANSAC.
                threshold_cos: float
                    Cosine threshold for RANSAC.
                focal_lr: float
                    Learning rate for the focal length.
                point_lr: float
                    Learning rate for the plane center.
                iter: int
                    Number of iterations for pytorch optimizer
                line_amount: int
                    amount of lines in the ground plane to plot
                plot_scale: int
                    size of the squares in the plotted grid (scale = 1 means the squares are 1 meter)

    Returns: focal_predicted: float
                optimized focal length
             normal.cpu().detach().numpy(): (3,) np.array
                normal vector
             ankleWorld: (3,) np.array
                optimized plane center
             error_array: list
                list of objective function values throughout the iterations
             focal_array: list
                list of focal lengths values throughout the iterations
             normal_array: list
                list of normal vector values throughout the iterations
    '''
    au = torch.tensor(au).double()
    av = torch.tensor(av).double()
    hu = torch.tensor(hu).double()
    hv = torch.tensor(hv).double()

    focal_predicted = init_value['focal_predicted']
    normal = torch.from_numpy(init_value['normal'])

    #print(normal, " Normal")

    ankleWorld = init_value['ankleWorld']
    intrinsic = np.array([normal[0], normal[2], focal_predicted, ankleWorld[0], ankleWorld[1], ankleWorld[2]])  
    
    #ankles = init_value['ankles']
    #ankles = np.array(ankles) - np.tile(np.array([ankleWorld[0],ankleWorld[1], ankleWorld[2]]), (ankles.shape[0], 1))
    #ankles = ankles.reshape(ankles.shape[0]*ankles.shape[1])
    init = list(intrinsic)# + list(ankles)

    start = timeit.timeit()
    print("start")

    #############
    plane_matrix, basis_matrix = geometry.find_plane_matrix(init_value['normal'], np.linalg.inv(init_value['cam_matrix']),init_value['ankleWorld'], img.shape[1], img.shape[0])
    
    R_axis_angle = torch.unsqueeze(torch.from_numpy(plane_matrix[:3, :3]), dim=0)
    R_axis_angle = pytorch3d.transforms.matrix_to_axis_angle(R_axis_angle)

    plane_matrix_T = torch.unsqueeze(torch.from_numpy(plane_matrix[:3, 3]), dim=0).clone()
    print([init[2]], " init[2] VALUE")

    print(R_axis_angle.shape, " R axis angle SHAPE")
    print(plane_matrix_T.shape, " plane_matrix_T SHAPE,")
    print(torch.tensor([[t1,t2]]).shape, " t1 t2 SHAPE")
    camera_array = PerspectiveCameras(
    R=R_axis_angle.double(),
    T=plane_matrix_T.double(),
    focal_length=torch.tensor([init[2]]).double(), 
    principal_point=torch.tensor([[t1,t2]]).double()
    )
    print(camera_array)
    R_matrix = camera_array.R
    T_matrix = camera_array.T
    focal = camera_array.focal_length
    p_matrix = camera_array.principal_point
    print(focal.shape, " FOCAL")    
    #stop
    error_array = []
    
    R_matrix.requires_grad = True
    T_matrix.requires_grad = True
    focal.requires_grad = True
    #############
    
    optimizer = Adam([
        R_matrix,
        T_matrix,
        focal
    ], lr=0.1)
    
    ankle_2d_dc = torch.from_numpy(np.stack((au,av)))
    head_2d_dc = torch.from_numpy(np.stack((hu,hv)))

    min_error = np.inf
    min_weight = None
    min_focal = None
    min_point = None
    error_array = []
    focal_array = []
    normal_array = []

    if conf_array is not None:
        conf_array = torch.from_numpy(conf_array/np.sum(conf_array))


    #from_pickle = {'ankles': ankles, 'cam_matrix': np.array([[focal_predicted, 0 , t1],[0, focal_predicted, t2],[0 , 0, 1]]), 'normal': normal.cpu().detach().numpy(), 'ankleWorld': [point[0].detach().numpy().item(),  point[1].detach().numpy().item(), point[2].detach().numpy().item()]}
    #from_pickle = {'cam_matrix': np.array([[focal, 0 , t1],[0, focal, t2],[0 , 0, 1]]), 'normal': normal.cpu().detach().numpy(), 'ankleWorld': [point[0].detach().numpy().item(),  point[1].detach().numpy().item(), point[2].detach().numpy().item()]}
    #if img is not None:
    #    plotting.plot_plane(au, av, hu, hv, save_dir, img, from_pickle, plot_scale, line_amount, 'init', threshold_euc, threshold_cos, h)
    #    plotting.display_2d_grid(au, av, hu, hv, save_dir, img, from_pickle, plot_scale, line_amount, 'init', threshold_euc, threshold_cos, h)
    for j in range(0, 1):
        
        for i in range(iter):
            R_mat1 = torch.transpose(so3_exponential_map(R_matrix)[0], 0, 1)
            print(focal, " FOCAL")
            print(p_matrix, " p_matrix")
            #cam_matrix = torch.squeeze(torch.stack([torch.stack([focal[0, 0], torch.zeros(1).double(), torch.tensor([t1]).double()]), torch.stack([torch.zeros(1).double(), focal[0, 0], torch.tensor([t2]).double()]), torch.stack([torch.zeros(1).double(), torch.zeros(1).double(), torch.ones(1).double()])], dim = 0))
            #cam_inv = torch.squeeze(torch.stack([torch.stack([1/focal[0, 0], torch.zeros(1).double(), torch.tensor([-t1]).double()/focal[0, 0]]), torch.stack([torch.zeros(1).double(), 1/focal[0, 0], torch.tensor([-t2]).double()/focal[0, 0]]), torch.stack([torch.zeros(1).double(), torch.zeros(1).double(), torch.ones(1).double()])], dim = 0))
            
            row1 = torch.stack([focal[0][[0]], torch.tensor([0]).double(), p_matrix[0][[0]]]).double()
            row2 = torch.stack([torch.tensor([0]).double(), focal[0][[1]], p_matrix[0][[1]]]).double()
            row3 = torch.stack([torch.tensor([0]).double(), torch.tensor([0]).double(), torch.tensor([1.0]).double()])

            cam_matrix = torch.squeeze(torch.stack([row1, row2, row3]))
            cam_inv = torch.inverse(cam_matrix)
            point_2d = torch.from_numpy(np.stack((au,av, torch.ones(len(au)))))
            
            ankles = torch.matmul(cam_inv, point_2d)
            
            print(R_mat1.shape, ankles.shape)
    
            print(torch.tensor([[0],[0],[1]]).double().repeat(1, ankles.shape[1]).shape, torch.matmul(R_mat1, ankles).shape, " DOT PRoDUCT HEREEEE")
            #normal_dot_ray = torch.bmm(torch.tensor([[0],[0],[1]]).double().repeat(1, ankles.shape[1]),torch.matmul(R_mat1, ankles), dims = 0)
            print(T_matrix.shape, " T MATRIX")
            normal_dot_ray = torch.einsum('ij,ij->j', torch.tensor([[0],[0],[1]]).double().repeat(1, ankles.shape[1]), torch.matmul(R_mat1, ankles))
            print(torch.squeeze(torch.dot(torch.tensor([0,0,1]).double(), T_matrix[0]).repeat(len(au), 1)).shape, normal_dot_ray.shape, " NORMAL DOT")
            scale = torch.abs(torch.div(torch.squeeze(torch.dot(torch.tensor([0,0,1]).double(), T_matrix[0]).repeat(len(au), 1)), normal_dot_ray)).repeat(3, 1)
            print(scale)
            print(normal_dot_ray.shape, scale.shape)
            ankles = torch.transpose(torch.mul(scale,ankles), 0, 1)

            head_3d = ankles + h*normal.repeat(len(au), 1)
            head_2d = torch.div((torch.narrow(torch.transpose(torch.matmul(cam_matrix, torch.transpose(head_3d, 0, 1)), 0 , 1), 1, 0, 2)), head_3d[:, 2:3])

            height_2d = torch.norm(torch.transpose(head_2d_dc, 0, 1) - torch.transpose(ankle_2d_dc, 0, 1), dim = 1)

            head_2d_error = torch.div(torch.norm(head_2d - torch.transpose(head_2d_dc, 0, 1), dim = 1), height_2d)

            if conf_array is not None:
                error_head = torch.mean(head_2d_error*conf_array)
            else: 
                error_head = torch.mean(head_2d_error)

            error = error_head
            error_array.append(error.detach().numpy().item())

            if min_error > error:
                min_error = error.item()
                min_focal = focal[0][[0]].detach().numpy().item()
            
        
            if i % 1000 == 0 and img is not None:
                '''
                fig, ax1 = plt.subplots(1, 1)
                fig1, ax2 = plt.subplots(1, 1)

                ax1.plot(list(range(len(error_array))),error_array, c = 'blue')
                ax2.plot(list(range(len(focal_array))),focal_array, c = 'blue')

                focal_predicted = torch.clone(focal[0]).detach().numpy().item()

                #from_pickle = {'ankles': ankles, 'cam_matrix': cam_matrix.detach().numpy(), 'normal': normal.cpu().detach().numpy(), 'ankleWorld': [point[0].detach().numpy().item(),  point[1].detach().numpy().item(), point[2].detach().numpy().item()]}
                from_pickle = {'cam_matrix': cam_matrix.detach().numpy(), 'normal': normal.cpu().detach().numpy(), 'ankleWorld': [point[0].detach().numpy().item(),  point[1].detach().numpy().item(), point[2].detach().numpy().item()]}
                if img is not None:
                    plotting.plot_plane(au, av, hu, hv, save_dir, img, from_pickle, plot_scale, line_amount, i, threshold_euc, threshold_cos, h)
                    plotting.display_2d_grid(au, av, hu, hv, save_dir, img, from_pickle, plot_scale, line_amount, i, threshold_euc, threshold_cos, h)
                '''
                print(error, " iteration", i)   
                '''
                fig.savefig(save_dir + '/error.png')
                fig1.savefig(save_dir + '/focal.png')
                plt.close('all')
                '''
            focal_predicted = min_focal = focal[0][[0]].detach().numpy().item()
            
            focal_array.append(focal_predicted)
            normal_array.append(normal.detach().numpy())

            ############
            error.retain_grad()
            T_matrix.retain_grad()
            R_matrix.retain_grad()
            focal.retain_grad()
            #loss = loss/accum_iter
            #print(loss, " HELLOASDASDASD")
            ############
            optimizer.zero_grad()
            error.backward(retain_graph=True)

            optimizer.step()

    end = timeit.timeit()
    print(end - start, " TIME")
    print("*******************")
    #print(ankles)
    print(min_error, " min error")
    focal_predicted = min_focal
    weights =  min_weight
    ankleWorld = min_point

    return focal_predicted, normal.cpu().detach().numpy(), ankleWorld, error_array, focal_array, normal_array
#FREEZE NORMAL
def optimization_focal_dlt_torch(init_value, au, av, hu, hv, t1, t2, h, save_dir, img, threshold_euc, threshold_cos, focal_lr = 1e-1, point_lr = 1e-3, iter = 10000, line_amount = 50, plot_scale = 1, conf_array = None):
    '''
    This function optimizes the camerea paramters after ransac using gradient descent.

    Parameters: init_value: Dictionary
                    a dictionary of the form: init_value = {'focal_predicted': focal_predicted, 'normal': normal, 'ankleWorld': ankleWorld, 'ankles': np.array(ankles)}
                au: list
                    list of dcpose 2d ankle detections u coordinate (coordinates are (u, v))
                av: list
                    list of dcpose 2d ankle detections v coordinate (coordinates are (u, v))
                hu: list
                    list of dcpose 2d head detections u coordinate (coordinates are (u, v))
                hv: list
                    list of dcpose 2d head detections v coordinate (coordinates are (u, v))
                t1: float
                    x focal center
                t2: float
                    y focal center
                h: float
                    assumed neck height of the people
                save_dir
                    directory to save plots
                img: matplotlib.image 
                    frame from sequence to be plotted on. If None, then there will be no plotting
                threshold_euc: float
                    Euclidean threshold for RANSAC.
                threshold_cos: float
                    Cosine threshold for RANSAC.
                focal_lr: float
                    Learning rate for the focal length.
                point_lr: float
                    Learning rate for the plane center.
                iter: int
                    Number of iterations for pytorch optimizer
                line_amount: int
                    amount of lines in the ground plane to plot
                plot_scale: int
                    size of the squares in the plotted grid (scale = 1 means the squares are 1 meter)

    Returns: focal_predicted: float
                optimized focal length
             normal.cpu().detach().numpy(): (3,) np.array
                normal vector
             ankleWorld: (3,) np.array
                optimized plane center
             error_array: list
                list of objective function values throughout the iterations
             focal_array: list
                list of focal lengths values throughout the iterations
             normal_array: list
                list of normal vector values throughout the iterations
    '''
    au = torch.tensor(au).double()
    av = torch.tensor(av).double()
    hu = torch.tensor(hu).double()
    hv = torch.tensor(hv).double()

    focal_predicted = init_value['focal_predicted']
    normal = torch.from_numpy(init_value['normal'])

    #print(normal, " Normal")

    ankleWorld = init_value['ankleWorld']
    intrinsic = np.array([normal[0], normal[2], focal_predicted, ankleWorld[0], ankleWorld[1], ankleWorld[2]])  
    
    #ankles = init_value['ankles']
    #ankles = np.array(ankles) - np.tile(np.array([ankleWorld[0],ankleWorld[1], ankleWorld[2]]), (ankles.shape[0], 1))
    #ankles = ankles.reshape(ankles.shape[0]*ankles.shape[1])
    init = list(intrinsic)# + list(ankles)

    weights = torch.tensor(init[0:2]).double()
    focal = torch.nn.Parameter(torch.tensor([init[2]]).double())
    point = torch.nn.Parameter(torch.tensor(init[3:6]).double())

    start = timeit.timeit()
    print("start")
    
    optimizer = torch.optim.Adam([
                {'params': focal, 'lr':focal_lr},
                {'params': point, 'lr':point_lr}
            ], weight_decay=0.0)
    
    print(point, " POINT BEGGINING")
    
    ankle_2d_dc = torch.from_numpy(np.stack((au,av)))
    head_2d_dc = torch.from_numpy(np.stack((hu,hv)))

    min_error = np.inf
    min_weight = None
    min_focal = None
    min_point = None
    error_array = []
    focal_array = []
    normal_array = []

    if conf_array is not None:
        conf_array = torch.from_numpy(conf_array/np.sum(conf_array))


    #from_pickle = {'ankles': ankles, 'cam_matrix': np.array([[focal_predicted, 0 , t1],[0, focal_predicted, t2],[0 , 0, 1]]), 'normal': normal.cpu().detach().numpy(), 'ankleWorld': [point[0].detach().numpy().item(),  point[1].detach().numpy().item(), point[2].detach().numpy().item()]}
    from_pickle = {'cam_matrix': np.array([[focal_predicted, 0 , t1],[0, focal_predicted, t2],[0 , 0, 1]]), 'normal': normal.cpu().detach().numpy(), 'ankleWorld': [point[0].detach().numpy().item(),  point[1].detach().numpy().item(), point[2].detach().numpy().item()]}
    if img is not None:
        plotting.plot_plane(au, av, hu, hv, save_dir, img, from_pickle, plot_scale, line_amount, 'init', threshold_euc, threshold_cos, h)
        plotting.display_2d_grid(au, av, hu, hv, save_dir, img, from_pickle, plot_scale, line_amount, 'init', threshold_euc, threshold_cos, h)
    for j in range(0, 1):
        
        for i in range(iter):
            
            cam_matrix = torch.squeeze(torch.stack([torch.stack([focal, torch.zeros(1).double(), torch.tensor([t1]).double()]), torch.stack([torch.zeros(1).double(), focal, torch.tensor([t2]).double()]), torch.stack([torch.zeros(1).double(), torch.zeros(1).double(), torch.ones(1).double()])], dim = 0))
            cam_inv = torch.squeeze(torch.stack([torch.stack([1/focal, torch.zeros(1).double(), torch.tensor([-t1]).double()/focal]), torch.stack([torch.zeros(1).double(), 1/focal, torch.tensor([-t2]).double()/focal]), torch.stack([torch.zeros(1).double(), torch.zeros(1).double(), torch.ones(1).double()])], dim = 0))
            
            point_2d = torch.from_numpy(np.stack((au,av, torch.ones(len(au)))))
            
            ankles = torch.matmul(cam_inv, point_2d)
            
            normal_dot_ray = torch.matmul(normal, ankles)
            scale = torch.abs(torch.div(torch.squeeze(torch.dot(normal, point).repeat(len(au), 1)), normal_dot_ray)).repeat(3, 1)
            ankles = torch.transpose(torch.mul(scale,ankles), 0, 1)

            head_3d = ankles + h*normal.repeat(len(au), 1)
            head_2d = torch.div((torch.narrow(torch.transpose(torch.matmul(cam_matrix, torch.transpose(head_3d, 0, 1)), 0 , 1), 1, 0, 2)), head_3d[:, 2:3])

            height_2d = torch.norm(torch.transpose(head_2d_dc, 0, 1) - torch.transpose(ankle_2d_dc, 0, 1), dim = 1)

            head_2d_error = torch.div(torch.norm(head_2d - torch.transpose(head_2d_dc, 0, 1), dim = 1), height_2d)

            if conf_array is not None:
                error_head = torch.mean(head_2d_error*conf_array)
            else: 
                error_head = torch.mean(head_2d_error)

            error = error_head
            error_array.append(error.detach().numpy().item())

            if min_error > error:
                min_error = error.item()
                min_weight = weights.clone()

                min_focal = focal[0].detach().numpy().item()
                min_point = [point[0].detach().numpy().item(),  point[1].detach().numpy().item(), point[2].detach().numpy().item()]
            
            '''
            if i % 1000 == 0 and img is not None:

                fig, ax1 = plt.subplots(1, 1)
                fig1, ax2 = plt.subplots(1, 1)

                ax1.plot(list(range(len(error_array))),error_array, c = 'blue')
                ax2.plot(list(range(len(focal_array))),focal_array, c = 'blue')

                focal_predicted = torch.clone(focal[0]).detach().numpy().item()

                #from_pickle = {'ankles': ankles, 'cam_matrix': cam_matrix.detach().numpy(), 'normal': normal.cpu().detach().numpy(), 'ankleWorld': [point[0].detach().numpy().item(),  point[1].detach().numpy().item(), point[2].detach().numpy().item()]}
                from_pickle = {'cam_matrix': cam_matrix.detach().numpy(), 'normal': normal.cpu().detach().numpy(), 'ankleWorld': [point[0].detach().numpy().item(),  point[1].detach().numpy().item(), point[2].detach().numpy().item()]}
                if img is not None:
                    plotting.plot_plane(au, av, hu, hv, save_dir, img, from_pickle, plot_scale, line_amount, i, threshold_euc, threshold_cos, h)
                    plotting.display_2d_grid(au, av, hu, hv, save_dir, img, from_pickle, plot_scale, line_amount, i, threshold_euc, threshold_cos, h)
                print(error, " iteration", i)   

                fig.savefig(save_dir + '/error.png')
                fig1.savefig(save_dir + '/focal.png')
                plt.close('all')
            '''
            focal_predicted = focal[0].detach().numpy().item()
            
            focal_array.append(focal_predicted)
            normal_array.append(normal.detach().numpy())

            optimizer.zero_grad()
            error.backward()

            optimizer.step()

    end = timeit.timeit()
    print(point, " end")
    print(end - start, " TIME")
    print("*******************")
    #print(ankles)
    print(min_error, " min error")
    focal_predicted = min_focal
    weights =  min_weight

    print(point[0].detach().numpy().item(), " HIIII")
    ankleWorld = min_point

    return focal_predicted, normal.cpu().detach().numpy(), ankleWorld, error_array, focal_array, normal_array