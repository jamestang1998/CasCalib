import numpy as np
import single_util
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

def camera_align(cam_position_gt, cam_axis_gt, cam_position_pred, cam_axis_pred):
    return cam_position_gt_transformed, cam_axis_gt_transformed


def tsai_perspective(coord_3d, translation, f, R, cx, cy, sx, k, ncx, nfx, dx, dy, dpx, dpy):
    coord_rot = R @ coord_3d + translation
    xu = f*coord_rot[0]/coord_rot[2]
    yu = f*coord_rot[1]/coord_rot[2]

    xd, yd = radial(xu, yu, k)

    dx_prime = dx*ncx/nfx 

    xf = sx*xd/dx_prime + cx
    yf = yd/dy + cy

    ###############
    #xf = xu + cx
    #yf = yu + cy
    return xf, yf, coord_rot

def radial(xu, yu, k):

    x,y = fsolve(non_linear,[xu, yu], args=(k, xu, yu))

    return x,y

def non_linear(var, *data):
    k, xu, yu = data
    x,y = var
    #print(k, " adsasdasdad")
    eq1 = xu - x*(1 + k*(x**2 + y**2))
    eq2 = yu - y*(1 + k*(x**2 + y**2))
    return [eq1, eq2]

def grid_to_tv(pos, grid_width, grid_height, tv_origin_x, tv_origin_y, tv_width, tv_height): 
    tv_x = ( (pos % grid_width) + 0.5 ) * (tv_width / grid_width) + tv_origin_x   
    tv_y = ( (pos / grid_width) + 0.5 ) * (tv_height / grid_height) + tv_origin_y

    return tv_x, tv_y

def apply_homography_camera_to_plane(data, cam_matrix, plane_matrix, init_world, normal, img_width, img_height):
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


    for fr in list(data.keys()):

        return_dict[fr] = {}
        for tr in list(data[fr].keys()):
            
            #ankle_3d = np.array([data[fr][tr][0], data[fr][tr][1], 1])
            ankle_3d = np.squeeze(single_util.plane_ray_intersection_np([data[fr][tr][0]], [data[fr][tr][1]], cam_inv, normal, init_world))
            ankle_plane_coord = (plane_matrix @ ankle_3d)            
            print(ankle_plane_coord , " ankle_plane_coord ")
            #print(np.dot(plane_matrix[:, 0], plane_matrix[:, 1]), np.dot(plane_matrix[:, 1], plane_matrix[:, 2]), np.dot(plane_matrix[:, 0], plane_matrix[:, 2]), " DOT PRODUCT")
            return_dict[fr][tr] = ankle_plane_coord

            rotate_only.append(plane_matrix @ np.append(np.squeeze(single_util.plane_ray_intersection_np([data[fr][tr][0]], [data[fr][tr][1]], cam_inv, normal, init_world)), 0.0))
            #print(np.linalg.det(np.array(plane_matrix[0:3,0:3])), " MATRX DET")
            #print(plane_matrix, " PLANE MATRIX")
            #print(np.linalg.norm(plane_matrix[:, 0]), np.linalg.norm(plane_matrix[:, 1]), np.linalg.norm(plane_matrix[:, 2]), " norm")
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

    return return_dict

def find_plane_matrix(normal, cam_inv, plane_point, img_width, img_height):

    plane_world_center = np.squeeze(single_util.plane_ray_intersection_np([img_width/2.0], [img_height], cam_inv, normal, plane_point))
    plane_world_left = np.squeeze(single_util.plane_ray_intersection_np([img_width/2.0 + 1], [img_height], cam_inv, normal, plane_point))

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

    #stop

    #print([basis_x, normal, basis_z])
    #stop
    #basis_matrix = np.array([[plane_world_left[0], plane_world_up[0], plane_world_into[0], 0], [plane_world_left[1], plane_world_up[1], plane_world_into[1], 0], [plane_world_left[2], plane_world_up[2], plane_world_into[2], 0], [1,1,1,1]])
    basis_matrix = np.array([[plane_world_left[0], plane_world_into[0], plane_world_up[0], 0], [plane_world_left[1], plane_world_into[1], plane_world_up[1], 0], [plane_world_left[2], plane_world_into[2], plane_world_up[2], 0], [1,1,1,1]])
    #basis_matrix = np.array([[basis_x[0], normal[0], basis_z[0], plane_world_center[0]], [basis_x[1], normal[1], basis_z[1], plane_world_center[1]], [basis_x[2], normal[2], basis_z[2], plane_world_center[2]], [1,1,1,1]])
    #print(basis_matrix)

    #transformation = np.array([[1,0,0,0], [0,1,0,0], [0,0,1,plane_dist], [0,0,0,0]]) @ np.linalg.inv(basis_matrix)
    #transformation = np.array([[1,0,0,0], [0,1,0,0], [0,0,1,plane_dist], [1,1,1,1]]) @ np.linalg.inv(basis_matrix)
    transformation = np.array([[1,0,0,0], [0,1,0,0], [0,0,1,plane_dist], [1,1,1,1]]) @ np.linalg.inv(basis_matrix)

    #print(transformation @ np.append(plane_world_up, 1))
    #stop
    #transformation[3][3] = 1.0
    #print(np.linalg.inv(transformation), " TRANSFORMaSDASD")
    #print(transformation, " NO INV TRANSFORMaSDASD")
    
    return transformation, basis_matrix

def grid_plane_coordinates(cam_matrix, plane_basis_x, plane_basis_z, cam_inv, p_plane, init_world, normal, img_width, img_height):
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
            
    grid_location_3d_ij = init_world + p_plane[0]*plane_basis_x + p_plane[1]*plane_basis_z
    
    grid_location_2d_ij = single_util.perspective_transformation(cam_matrix, grid_location_3d_ij)
    
    return grid_location_2d_ij, grid_location_3d_ij

def get_3d(data, cam_matrix, init_world, normal):
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
    return_array = [] 

    for fr in list(data.keys()):

        for tr in list(data[fr].keys()):
            
            ankle_3d = np.squeeze(single_util.plane_ray_intersection_np([data[fr][tr][0]], [data[fr][tr][1]], cam_inv, normal, init_world))
            return_array.append(ankle_3d)

    return return_array

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


    for fr in list(data.keys()):

        return_dict[fr] = {}
        for tr in list(data[fr].keys()):
            
            ankle_3d = np.append(np.squeeze(single_util.plane_ray_intersection_np([data[fr][tr][0]], [data[fr][tr][1]], cam_inv, normal, init_world)), 1.0)
            ankle_plane_coord = (plane_matrix @ ankle_3d)
            #print(np.dot(plane_matrix[:, 0], plane_matrix[:, 1]), np.dot(plane_matrix[:, 1], plane_matrix[:, 2]), np.dot(plane_matrix[:, 0], plane_matrix[:, 2]), " DOT PRODUCT")
            return_dict[fr][tr] = ankle_plane_coord

            rotate_only.append(plane_matrix @ np.append(np.squeeze(single_util.plane_ray_intersection_np([data[fr][tr][0]], [data[fr][tr][1]], cam_inv, normal, init_world)), 0.0))
            #print(np.linalg.det(np.array(plane_matrix[0:3,0:3])), " MATRX DET")
            #print(plane_matrix, " PLANE MATRIX")
            #print(np.linalg.norm(plane_matrix[:, 0]), np.linalg.norm(plane_matrix[:, 1]), np.linalg.norm(plane_matrix[:, 2]), " norm")
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
    '''
    arg_max_y = np.argmax(plane_y)
    arg_min_y = np.argmin(plane_y)

    arg_max_z = np.argmax(z_array)
    arg_min_z = np.argmin(z_array)

    print(np.linalg.norm(np.array([plane_x[arg_max_y], plane_y[arg_max_y], plane_z[arg_max_y]]) - np.array([plane_x[arg_min_y], plane_y[arg_min_y], plane_z[arg_min_y]])) , " plane length")
    print(np.linalg.norm(np.array([x_array[arg_max_z], y_array[arg_max_z], z_array[arg_max_z]]) - np.array([x_array[arg_min_z], y_array[arg_min_z], z_array[arg_min_z]])) , " plane length")
    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(111,projection='3d')
    #ax.set_box_aspect((1,1,1))
    #print(plane_x)
    #ax.set_aspect('auto')
    #plotting.set_equal_3d([x_array + plane_x, y_array + plane_y, z_array + plane_z], ax)

    #print(np.array(rotate_only).shape, " SHAOEEEE")
    ax.scatter(np.array(rotate_only)[:, 0], np.array(rotate_only)[:, 1], np.array(rotate_only)[:, 2], c = 'g')

    ax.scatter(x_array, y_array, z_array, c = 'b')
    ax.scatter([0], [0], [0], c = 'black')
    ax.plot([0, 1], [0, 0], [0, 0], c = 'black')
    ax.plot([0, 0], [0, 1], [0, 0], c = 'black')
    ax.plot([0, 0], [0, 0], [0, 1], c = 'black')

    print(plane_matrix, " ASDSASDASDSASASDASA")

    #plane matrix: camera coord -> plane coord
    origin = np.array(plane_matrix) @ np.array([0,0,0, 1])
    new_x = np.array(plane_matrix) @ np.array([1,0,0, 1])
    new_y = np.array(plane_matrix) @ np.array([0,1,0, 1])
    new_z = np.array(plane_matrix) @ np.array([0,0,1, 1])

    print(origin, " ORIGIN")
    ax.scatter([origin[0]], [origin[1]], [origin[2]], c = 'red')
    ax.plot([origin[0], new_x[0]], [origin[1], new_x[1]], [origin[2], new_x[2]], c = 'red')
    ax.plot([origin[0], new_y[0]], [origin[1], new_y[1]], [origin[2], new_y[2]], c = 'red')
    ax.plot([origin[0], new_z[0]], [origin[1], new_z[1]], [origin[2], new_z[2]], c = 'red')

    ax.scatter(plane_x, plane_y, plane_z, c = 'r')
    #ax.scatter(all_x, all_y, all_z, c = 'r')
    plotting.set_equal_3d(np.transpose(np.array([all_x, all_y, all_z])), ax)

    plt.grid()
    plt.show()
    '''
    return return_dict

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

    # sanity check
    #if linalg.matrix_rank(H) < 3:
    #    raise ValueError("rank of H = {}, expecting 3".format(linalg.matrix_rank(H)))

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