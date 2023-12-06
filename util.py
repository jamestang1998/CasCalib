import numpy as np
import data
from scipy import special
import random as rand
import json


def random_combination(image_index, num_points, termination_cond):
    '''
    gets len(image_index) choose num_points combinations of detections, until the number of combinations exceed termination_comd

    Parameters: image_index: list
                    indices of detections in datastore
                num_points: int
                    number of points to solve DLT equations
                termination_cond: int
                    maximum number of combinations
    Returns: samples: list
                Combinations of detections
    '''
    total_comb = int(special.comb(len(image_index),num_points))
    samples = set()
    while len(samples) < termination_cond and len(samples) < total_comb:
        samples.add(tuple(sorted(rand.sample(image_index, num_points))))

    return list(samples)



def get_ankles_heads(datastore, image_index):
    '''
    gets the ankle and head detections from the detections json file
    
    Parameters: datastore: data.py dataloader object
                    Stores the poses for accessing keypoints
                image_index: python int list
                    The selected indices of the json object that contain 2d keypoints
    Returns:    ppl_ankle_u: float np.array 
                    list of ankle x camera coordinates
                ppl_ankle_v: float np.array 
                    list of ankle y camera coordinates
                ppl_head_u: float np.array 
                    list of head x camera coordinates
                ppl_head_v: float np.array 
                    list of head y camera coordinates
                 
    '''
    ppl_ankle_u = []
    ppl_ankle_v = []
    
    ppl_head_u = []
    ppl_head_v = []
    head_conf = []
    ankle_left_conf = []
    ankle_right_conf = []
    
    for ppl in image_index:
        left_ankle = datastore.getitem(ppl)["left_ankle"]
        right_ankle = datastore.getitem(ppl)["right_ankle"]

        ankle_left_conf.append(datastore.getitem(ppl)["left_ankle"][2])
        ankle_right_conf.append(datastore.getitem(ppl)["right_ankle"][2])

        head_x = ((datastore.getitem(ppl)["left_shoulder"][0]) + (datastore.getitem(ppl)["right_shoulder"][0]))/2.0
        head_y = ((datastore.getitem(ppl)["left_shoulder"][1]) + (datastore.getitem(ppl)["right_shoulder"][1]))/2.0
        #ankle_x, ankle_y = determine_foot_bbox(right_ankle, left_ankle, datastore.getitem(ppl))
        max_size = max(np.linalg.norm(np.array([left_ankle[0], left_ankle[1]]) - np.array([head_x, head_y])), np.linalg.norm(np.array([right_ankle[0], right_ankle[1]]) - np.array([head_x, head_y])))
        hgt_thresh = 0.2*max_size
        wide_tresh = 0.2*max_size
        ankle_x, ankle_y = determine_foot(right_ankle,left_ankle, hgt_threshold=hgt_thresh, wide_threshold=wide_tresh)
            
        #head_x = (datastore.getitem(ppl)["Thorax"][0])
        #head_y = (datastore.getitem(ppl)["Thorax"][1])

        head_conf.append((datastore.getitem(ppl)["left_shoulder"][2] + datastore.getitem(ppl)["right_shoulder"][2])/2.0)

        ppl_ankle_u.append(ankle_x)
        ppl_ankle_v.append(ankle_y)

        ppl_head_u.append(head_x)
        ppl_head_v.append(head_y)
    
    return np.array(ppl_ankle_u), np.array(ppl_ankle_v), np.array(ppl_head_u), np.array(ppl_head_v), np.array(head_conf), np.array(ankle_left_conf), np.array(ankle_right_conf)

def determine_foot(rfoot, lfoot, hgt_threshold=80.0, wide_threshold=100.0):
    '''
    Get a single point to represent both ankles given both left and right ankles.

    parameters: rfoot: float
                    right ankle
                lfoot: float
                    left ankle
                hgt_threshold: float
                    height threshold (difference in y or v coordinate)
                wide_threshold: float
                    width threshold (difference in x or u coordinate)
    '''

    
    dist_wide = abs(rfoot[0] - lfoot[0])
    dist_height = abs(rfoot[1] - lfoot[1])
    if dist_height>hgt_threshold: # height
        if rfoot[1]> lfoot[1]:
            ankle_y = rfoot[1]
            ankle_x = rfoot[0]
        else: 
            ankle_y = lfoot[1]
            ankle_x = lfoot[0]
    elif dist_wide>wide_threshold: # wide 
        if rfoot[1] > lfoot[1]:
            ankle_y = rfoot[1]
            ankle_x = rfoot[0]
        else:
            ankle_y = lfoot[1]
            ankle_x = lfoot[0]
    else:
        # normal midpoint
        ankle_x = ((rfoot[0] + lfoot[0])/2.0)
        ankle_y = ((rfoot[1] + lfoot[1])/2.0)

    return ankle_x, ankle_y

#trying to get the poses where the person is NOT moving (aka feet are close together)
def select_indices(datastore, angle_filter_video, confidence, img_width, img_height, skip = 1, min_size = 0, max_len = 1000):
    """ 
    Selects the detection results from openpose that are not too bent or kneeling
    
    Parameters: datastore: data.py dataloader object
                    stores the poses for accessing keypoints
                angle_filter_video: float
                    angle threshold for filtering non standing poses
                confidence: float
                    confidence threshold for confidence of keypoint detections
                img_height: float
                    height of the image
                skip : int, optional (default = 1)
                    The amount of detection frames to skip (ex skip_frame = 1 means use every frame, frame = 2 means use every 2nd etc)
                min_size = int, optional (default = 0)
                    The minimum pixel height of detected peoples (People that are too small in the scene tend to be noisy and very far away)
                max_len = int, optional (default = 1000)
                    The maximum amount of detections to be used (if detections exceed the max_len, then a random sample of size max_len is taken)
    Returns:    process_dict: data.py dataloader object
                    stores the poses for accessing filtered keypoints
    """
    # print("selecting indices")

    filtered_detections = []

    head_ankle_array = []
    
    # print(datastore.__len__(), " total number of poses")
    
    height_array = []

    top_right = 'right_shoulder'
    top_left = 'left_shoulder'

    image_den = 4.0
    image_num = 3.0
    for i in range(0, datastore.__len__(), skip):
        pose_array = datastore.getitem(i)

        for d in range(len(pose_array)):

            pose = pose_array[d]
            
            #print("******************************************************************************************")
            #print(pose['Thorax'], pose['left_ankle'], pose['right_ankle'], d, i ," THORAX LEFT ANKLEEEE")
            
            if confidence > pose[top_left][2] or confidence > pose[top_right][2] or confidence > pose['left_ankle'][2] or confidence > pose['right_ankle'][2]:
                continue
            '''
            if pose[top_right][0] < 0 or pose[top_left][0] < 0 or pose['left_ankle'][0] < 0 or pose['right_ankle'][0] < 0:
                continue
        
            if pose[top_left][1] > img_height - 30 or pose[top_right][1] > img_height - 30 or pose['left_ankle'][1] > img_height - 30 or pose['right_ankle'][1] > img_height - 30:
                continue
            '''
            '''
            if pose['left_ankle'][1] < img_height/image_den or pose['right_ankle'][1] < img_height/image_den:
                continue
            if pose['left_ankle'][1] > image_num*img_height/image_den or pose['right_ankle'][1] > image_num*img_height/image_den:
                continue
            if pose['left_ankle'][0] < img_width/image_den or pose['right_ankle'][0] < img_width/image_den:
                continue
            if pose['left_ankle'][0] > image_num*img_width/image_den or pose['right_ankle'][0] > image_num*img_width/image_den:
                continue
            '''
            
            
            left_ankle = pose["left_ankle"]
            right_ankle = pose["right_ankle"]

            head = (np.array(pose[top_left][:2]) + np.array(pose[top_right][:2]))/2.0 #np.array([pose['Thorax'][:2]])
            #print(head)
            head_x = head[0]
            head_y = head[1]
            max_size = max(np.linalg.norm(np.array([left_ankle[0], left_ankle[1]]) - np.array([head_x, head_y])), np.linalg.norm(np.array([right_ankle[0], right_ankle[1]]) - np.array([head_x, head_y])))
            hgt_thresh = 0.2*max_size
            wide_tresh = 0.2*max_size

            if np.absolute(pose['left_ankle'][1] - pose['right_ankle'][1]) > hgt_thresh:
                continue
                
            if np.absolute(pose['left_ankle'][0] - pose['right_ankle'][0]) > wide_tresh:
                continue
            ankle_average_u, ankle_average_v = determine_foot(right_ankle,left_ankle, hgt_threshold=hgt_thresh, wide_threshold=wide_tresh)
            #ankle_average_u, ankle_average_v = determine_foot(right_ankle, left_ankle)
            #ankle_average_u, ankle_average_v = determine_foot(right_ankle, left_ankle, hgt_threshold=8.0, wide_threshold=10.0)

            ankle_average = np.array([ankle_average_u, ankle_average_v])

            #head = np.array([pose['Thorax'][:2]])
            #head = (np.array([pose[top_left][:2]]) - np.array([pose[top_right][:2]]))/2.0 #np.array([pose['Thorax'][:2]])

            knee_ankle_left = np.array(pose['left_knee'][:2]) - np.array(pose['left_ankle'][:2])
            knee_ankle_right = np.array(pose['right_knee'][:2]) - np.array(pose['right_ankle'][:2])
            knee_hip_left = np.array(pose['left_knee'][:2]) - np.array(pose['left_hip'][:2])
            knee_hip_right = np.array(pose['right_knee'][:2]) - np.array(pose['right_hip'][:2])

            knee_angle_left = angle_between(knee_ankle_left, knee_hip_left)
            knee_angle_right = angle_between(knee_ankle_right, knee_hip_right)

            hip_knee_left = np.array(pose['left_hip'][:2]) - np.array(pose['left_knee'][:2])
            hip_knee_right = np.array(pose['right_hip'][:2]) - np.array(pose['right_knee'][:2])
            hip_shoulder_left = np.array(pose['left_hip'][:2]) - np.array(pose[top_left][:2])
            hip_shoulder_right = np.array(pose['right_hip'][:2]) - np.array(pose[top_right][:2])

            hip_angle_left = angle_between(hip_knee_left, hip_shoulder_left)

            hip_angle_right = angle_between(hip_knee_right, hip_shoulder_right)

            angle_right = np.abs(knee_angle_right - np.pi) + np.abs(hip_angle_right - np.pi)
            angle_left = np.abs(knee_angle_left - np.pi) + np.abs(hip_angle_left - np.pi)

            angle = min([angle_right, angle_left])
            height = np.linalg.norm(head - ankle_average)
            #print('HEIGHT ', height,  " THE ANGLEE", angle, angle_filter_video, d, i, " BEFORE")
            height_array.append(height)
            if height < min_size:
                continue

            if angle > angle_filter_video:
                continue
            
            joint_2d = (ankle_average_u, ankle_average_v, head[0], head[1])
            #print(pose, " POSE")
            filtered_detections.append(pose)
    
    if len(filtered_detections) > max_len:
        index = np.random.choice(len(filtered_detections), max_len, replace=False)
        filtered_detections = list(np.array(filtered_detections)[index])

    filtered_datastore = data.dcpose_dataloader(None)
    filtered_datastore.new_data(filtered_detections)
    filtered_datastore.write_height(np.array(height_array))

    # print(filtered_datastore.__len__(), " Number of ankle for calibration")
    return filtered_datastore

def plane_ray_intersection_np(x_imcoord, y_imcoord, cam_inv, normal, init_point):
    """ 
    Recovers the 3d coordinates from 2d by computing the intersection between the 2d point's ray and the plane
    
    Parameters: x_imcoord: float list or np.array
                    x coordinates in camera coordinates
                y_imcoord: float list or np.array
                    y coordinates in camera coordinates
                cam_inv: (3,3) np.array
                    inverse camera intrinsic matrix
                normal: (3,) np.array
                    normal vector of the plane
                init_point: (3,) np.array
                    3d point on the plane used to initalize the ground plane
    Returns:    ray: (3,) np.array
                    3d coordinates of (x_imcoord, y_imcoord)
    """
    x_imcoord = np.array(x_imcoord)
    y_imcoord = np.array(y_imcoord)
    
    point_2d = np.stack((x_imcoord, y_imcoord, np.ones(x_imcoord.shape[0])))
    
    ray = np.array((cam_inv @ point_2d))
    
    normal_dot_ray = np.transpose(np.array(normal)) @ ray + 1e-15
    
    scale = abs(np.divide(np.repeat(np.dot(normal, init_point), x_imcoord.shape[0]), normal_dot_ray))
    ray[0] = scale*ray[0]
    ray[1] = scale*ray[1]
    ray[2] = scale*ray[2]
    return ray

def perspective_transformation(cam_matrix, coordinate):
    '''
    Converts 3d coordinates into 2d camera coordinates
    
    Parameters: cam_matrix: (3,3) array 
                    The intrinsic camera matrix
                coordinates: (3,1) array
                    3d world coordinates
    Returns:    output: float
                    2d camera coordinates of 3d world coordinates          
    '''
    eps = 1e-20
    coordinate_2d = np.array(cam_matrix) @ np.array(coordinate)  
    return coordinate_2d[:2]/(coordinate_2d[2] + eps)

def angle_between(v1, v2):
    """ 
    Outputs the angle in radians between vectors 'v1' and 'v2'::
    
    Parameters: v1: python float list
                v2: python float list
    Returns:    output: float
                    angle in radians

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

def hyperparameter(filename):
    '''
    Parses the hyperparameter.json file
    
    Parameters: filename: string 
                    This is the path to the hyperparameter file, which is a json file
    Returns:    threshold_euc: float
                    Euclidean threshold of ransac
                threshold_cos: float
                    Cosine threshold of ransac
                angle_filter_video: float
                    Determines how bent a pose can be before it is discarded as 'non standing poses'
                confidence: Int
                    Threshold for how low the detection confidence can be before it is discarded
                termination_cond: Int
                    Number of iterations
                num_points: Int
                    Number of people used to solve DLT system of equations
                h: float
                    assumed height of people in scene  
                optimizer_iteration: int
                    Number of iterations for pytorch optimizer
                focal_lr: float
                    Learning rate for the focal length.
                point_lr: float
                    Learning rate for the plane center.       
    '''

    if isinstance(filename, str):
        with open(filename, 'r') as f_hyperparam:
            file = json.load(f_hyperparam)
    elif isinstance(filename, dict):
            file = filename
    else: 
        raise Exception('Input must either be a string that is a path to a hyperparamter.json file, or is a dictionary with the hyperparameters')
    
    threshold_euc = float(file['threshold_euc'])
    threshold_cos = float(file['threshold_cos'])
    angle_filter_video = float(file['angle_filter_video'])
    confidence = float(file['confidence'])
    termination_cond = int(file['termination_cond'])
    num_points = int(file['num_points'])
    h = float(file['h'])

    optimizer_iteration = int(file["optimizer_iteration"])
    focal_lr = float(file["focal_lr"])
    point_lr = float(file["point_lr"])
    
    return threshold_euc, threshold_cos, angle_filter_video, confidence, termination_cond, num_points, h, optimizer_iteration, focal_lr, point_lr


def unit_vector(vector):
    '''
    Outputs the unit vector of the vector.
    
    Parameters: vector: np.array
                vector to be normalized
    Returns: Output: np.array()
                normalized vector     
    '''
    eps = 1e-15
    return vector / (np.linalg.norm(vector) + eps)

def matrix_cosine(x, y):
    """ 
    Computes the pairwise cosine distance for matrix rows
    
    Parameters: x: N by 2 np array
                y: N by 2 np array
    Returns:    output: np array
                    Pairwise cosine distances of the rows of X and Y
    """
    return np.einsum('ij,ij->i', x, y) / (
              np.linalg.norm(x, axis=1) * np.linalg.norm(y, axis=1) + 1e-15
    )

def basis_change_rotation_matrix(cam_matrix, cam_inv, init_point, normal, img_width, img_height):
    """ 
    Given the camera matrix and the ground plane, constructs the transformation matrix to convert the view into a birds eye perspective.
    
    Parameters: cam_matrix: (3,3) np.array
                    Intrinsic camera matrix
                cam_inv: (3,3) np.array
                    Inverse camera matrix
                init_point: (3,) np.array
                    3d position of the ground plane
                normal: (3,) normal
                    Normal vector of the ground plane
                img_width: float
                    width of the image
                img_height: float
                    height of the image
    Returns:    output: (3,3) np.array
                    transformation matrix
    """    
    plane_world = np.squeeze(plane_ray_intersection_np([img_width/2.0], [img_height], cam_inv, normal, init_point))

    p00, p00_3d = project_point_horiz_bottom(cam_matrix, cam_inv, [0,0], plane_world, normal, img_width, img_height)
    p01, p01_3d = project_point_horiz_bottom(cam_matrix, cam_inv, [0,1], plane_world, normal, img_width, img_height)
    p10, p10_3d = project_point_horiz_bottom(cam_matrix, cam_inv, [1,0], plane_world, normal, img_width, img_height)
    p11, p11_3d = project_point_horiz_bottom(cam_matrix, cam_inv, [1,1], plane_world, normal, img_width, img_height)
    
    new_basis0 = p01_3d - p00_3d
    new_basis1 = p10_3d - p00_3d
    
    new_basis0 = new_basis0/np.linalg.norm(new_basis0)
    new_basis1 = new_basis1/np.linalg.norm(new_basis1)
    
    old_basis0 = np.array([1, 0, 0])
    old_basis1 = np.array([0, 1, 0])
    old_basis2 = np.array([0, 0, 1])
    
    C = np.zeros([3,3], dtype = float)
    C[0] = [np.dot(new_basis0, old_basis0), np.dot(new_basis0, old_basis1), np.dot(new_basis0, old_basis2)]
    C[1] = [np.dot(new_basis1, old_basis0), np.dot(new_basis1, old_basis1), np.dot(new_basis1, old_basis2)]
    C[2] = [np.dot(normal, old_basis0), np.dot(normal, old_basis1), np.dot(normal, old_basis2)]
       
    z_rotation = np.zeros((3,3))
    z_rotation[0] = [np.cos(np.pi/2.0), -1*np.sin(np.pi/2.0), 0]
    z_rotation[1] = [np.sin(np.pi/2.0), np.cos(np.pi/2.0), 0]
    z_rotation[2] = [0,0,1]
    
    flip = np.zeros((3,3))
    flip[0] = [-1, 0, 0]
    flip[1] = [0, 1, 0]
    flip[2] = [0, 0, 1,]
    
    return flip @ z_rotation @ C

def project_point_horiz_bottom(cam_matrix, cam_inv, p_plane, init_world, normal, img_width, img_height):
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
    plane_horiz_world = np.squeeze(plane_ray_intersection_np([img_width/2.0 + 1], [img_height], cam_inv, normal, init_world))

    up_vector = np.array(plane_horiz_world) - np.array(init_world)
    
    up_vector = up_vector/np.linalg.norm(up_vector)

    v = up_vector
    #u = np.cross(v, normal)
    u = np.cross(normal, v)
    
    #normalize 
    v = v/np.linalg.norm(v)
    u = u/np.linalg.norm(u)
            
    grid_location_3d_ij = init_world + p_plane[0]*v + p_plane[1]*u
    
    grid_location_2d_ij = perspective_transformation(cam_matrix, grid_location_3d_ij)
    
    return grid_location_2d_ij, grid_location_3d_ij

def plane_line_intersection(x_0, x_1, normal, init_point):
    """ 
    Given a line defined by 2 3d points, computes the intersection of that line with the plane defined by normal and init_point
    
    Parameters: x_0: (3,3) np.array
                    Starting point of the line
                x_1: (3,3) np.array
                    end point of the line
                normal: (3,3) np.array
                    Normal vector of the plane
                init_point: (3,3) np.array
                    3d position of the plane
    Returns:    output: (3,) np.array
                    3d coordinates of the intersection between the line and the plane
    """        
    ray = np.array(x_1) - np.array(x_0)
        
    scale = np.dot(normal, np.array(init_point) - np.array(x_0))/np.dot(normal, ray)
        
    point_x = np.squeeze(ray[0]*scale)
    point_y = np.squeeze(ray[1]*scale)
    point_z = np.squeeze(ray[2]*scale)

    return np.array([point_x, point_y, point_z]) + np.array(x_0)
