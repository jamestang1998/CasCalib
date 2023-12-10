import numpy as np
import util
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import plotting
import math
import sys 
from scipy.optimize import fsolve
import math

def find_plane_matrix(normal, cam_inv, plane_point, img_width, img_height):

    plane_world_center = np.squeeze(util.plane_ray_intersection_np([img_width/2.0], [img_height], cam_inv, normal, plane_point))
    plane_world_left = np.squeeze(util.plane_ray_intersection_np([img_width/2.0 + 1], [img_height], cam_inv, normal, plane_point))

    plane_dist = np.dot(normal, -plane_world_center)
    closest_point = plane_dist*-1*np.array(normal)

    basis_x = plane_world_left - plane_world_center
    basis_x = basis_x/np.linalg.norm(basis_x)

    plane_world_center = closest_point
    plane_world_left = basis_x + closest_point
    ############################################

    basis_z = np.cross(normal, basis_x)
    basis_z = basis_z/np.linalg.norm(basis_z)

    plane_world_up = (normal + plane_world_center)
    plane_world_left = basis_x + plane_world_center
    plane_world_into = basis_z + plane_world_center

    basis_matrix = np.array([[plane_world_left[0], plane_world_into[0], plane_world_up[0], 0], [plane_world_left[1], plane_world_into[1], plane_world_up[1], 0], [plane_world_left[2], plane_world_into[2], plane_world_up[2], 0], [1,1,1,1]])
    transformation = np.array([[1,0,0,0], [0,1,0,0], [0,0,1,plane_dist], [1,1,1,1]]) @ np.linalg.inv(basis_matrix)
    
    return transformation, basis_matrix

def camera_to_plane(data, cam_matrix, plane_matrix, init_world, normal, img_width, img_height):
    """ 
    Finds finds 2d and 3d coordinates from plane coordinates
    
    Parameters: cam_matrix: (3,3) np.array
                    intrinsic camera matrix
                cam_inv: (3,3) np.array
                    inverse intrinsic camera matrix
                p_plane: (2,) list or array
                    [x, y] coordinates in plane coordinates
                init_world: (3,) np.array
                    3d coordinate used to initlize the ground plane
                normal: (3,) np.array
                    normal vector of ground plane
                img_width: int
                    width of the image
                img_height: int 
                    height of the image
    Returns:    np.array(p_px[:2]): (2,) array
                    p_plane converted into camera coordinates
                grid_location_3d_ij: (3,) array
                    p_plane converted to world coordinates
    """
    cam_inv = np.linalg.inv(cam_matrix)
    return_dict = {}

    x_array = []
    y_array = []
    z_array = []

    plane_x = []
    plane_y = []
    plane_z = []

    all_x = []
    all_y = []
    all_z = []

    rotate_only = []

    return_list = []
    for fr in list(data.keys()):

        return_dict[fr] = {}
        for tr in list(data[fr].keys()):
            
            ankle_3d = np.append(np.squeeze(util.plane_ray_intersection_np([data[fr][tr][0]], [data[fr][tr][1]], cam_inv, normal, init_world)), 1.0)
            ankle_plane_coord = (plane_matrix @ ankle_3d)
            return_dict[fr][tr] = ankle_plane_coord

            rotate_only.append(plane_matrix @ np.append(np.squeeze(util.plane_ray_intersection_np([data[fr][tr][0]], [data[fr][tr][1]], cam_inv, normal, init_world)), 0.0))

            x_array.append(ankle_3d[0]) 
            y_array.append(ankle_3d[1]) 
            z_array.append(ankle_3d[2])        

            plane_x.append(ankle_plane_coord[0])
            plane_y.append(ankle_plane_coord[1])
            plane_z.append(ankle_plane_coord[2])

            all_x.append(ankle_3d[0]) 
            all_y.append(ankle_3d[1]) 
            all_z.append(ankle_3d[2]) 

            all_x.append(ankle_plane_coord[0])
            all_y.append(ankle_plane_coord[1])
            all_z.append(ankle_plane_coord[2])

            return_list.append(ankle_plane_coord[0:2].tolist())
    return return_dict, return_list

# Input: expects 3xN matrix of points
# Returns R,t
# R = 3x3 rotation matrix
# t = 3x1 column vector

# 2xN
def rigid_transform_3D(A, B):

    # find mean column wise
    centroid_A = np.mean(A, axis=1)
    centroid_B = np.mean(B, axis=1)

    # ensure centroids are 3x1
    centroid_A = centroid_A.reshape(-1, 1)
    centroid_B = centroid_B.reshape(-1, 1)

    # subtract mean
    Am = A - centroid_A
    Bm = B - centroid_B

    H = Am @ np.transpose(Bm)

    # find rotation
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    # special reflection case
    if np.linalg.det(R) < 0:
        print("det(R) < R, reflection detected!, correcting for it ...")
        Vt[1,:] *= -1
        R = Vt.T @ U.T

    t = -R @ centroid_A + centroid_B

    return R, np.squeeze(t)