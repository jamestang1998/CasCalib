import numpy as np
import json
import data
from scipy import special
import random as rand

import os
import torch

import scipy.spatial
from sklearn.neighbors import NearestNeighbors
import torch.nn.functional as F
import math
from itertools import tee
from scipy.spatial.distance import cdist
import knn
import math
from scipy.optimize import linear_sum_assignment

import re

from collections import defaultdict
from scipy import stats

import numpy as np
from scipy import stats
from sklearn.cluster import DBSCAN
import random

def match_3d_plotly_input2d_farthest_point(ref_time_array, ref_vel_array, sync_time, sync_vel):

    #print(indices)
    ref_dict = {}
    sync_dict = {}

    for i in range(len(ref_time_array)):

        if ref_time_array[i] in ref_dict:
            ref_dict[i].append(ref_vel_array[i])
        else:
            ref_dict[i] = [ref_vel_array[i]]
    
    for i in range(len(sync_time)):
    
        if sync_vel[i] in sync_dict:
            sync_dict[i].append(sync_vel[i])
        else:
            sync_dict[i] = [sync_vel[i]]

    indices = np.union1d(ref_time_array, sync_time)  # Find the union of times from x1 and x2
    for ind in indices:

        #######################################
        points_array_ref = []
        poses_array_ref = []

        points_array_sync = []
        poses_array_sync = []

        #print(points_array_ref, ind)
        #print(points_array_sync, ind)
        points_ref = np.concatenate(points_array_ref)
        points_sync = np.concatenate(points_array_sync)

        #matched_A, matched_B, row_ind, col_ind = util.hungarian_assignment(points_ref, points)
        #print(points_ref.shape, " points ref  ")
        #stop
        #print(points_ref.shape, points_sync.shape, " POINTS HELLO BEFORE")
        row_ind, col_ind = util.hungarian_assignment_one2one(points_ref, points_sync)
        
        for r_ind in range(len(row_ind)):
            
            #sync_ind = indices[ind]
            sync_ind = col_ind[r_ind]
            ref_ind = row_ind[r_ind]
            #print(list(poses_array_ref[ref_ind].values()), " POSES ARRAY")
            #print(torch.tensor(list(poses_array_ref[ref_ind].values())[:-1]).shape, " POSES ARRAY ")
            index_dict[i1].append(torch.tensor(list(poses_array_ref[ref_ind].values())[:-1]))
            index_dict[i2].append(torch.tensor(list(poses_array_sync[sync_ind].values())[:-1]))

            ankle_dict[i1].append(torch.from_numpy(points_ref[ref_ind]))
            ankle_dict[i2].append(torch.from_numpy(points_sync[sync_ind]))

    return sample_point_array#point_array

def filter_dict(D, A, B):
    if not isinstance(D, dict) or not isinstance(A, list) or not isinstance(B, list):
        raise ValueError("Input types are not valid. D should be a dictionary, A and B should be lists.")
    
    # Create a new dictionary with filtered entries
    new_dict = {key: value for key, value in D.items() if key in A and value in B}
    
    return new_dict

def remove_and_subtract_keys(original_dict, ordered_keys, n, remove_from_start=True):
    if not isinstance(original_dict, dict):
        raise ValueError("Input is not a dictionary.")
    
    if not isinstance(ordered_keys, list) or not all(isinstance(key, int) for key in ordered_keys):
        raise ValueError("Ordered keys should be a list of integers.")
    
    if not isinstance(n, int) or n < 0:
        raise ValueError("n should be a non-negative integer.")
    
    if remove_from_start:
        keys_to_remove = ordered_keys[:n]
        modified_dict = {key - n: value for key, value in original_dict.items() if key not in keys_to_remove}
    else:
        keys_to_remove = ordered_keys[-n:]
        modified_dict = {key: value for key, value in original_dict.items() if key not in keys_to_remove}

    return modified_dict

def rename_keys_to_sequence(array):
    new_dict = {}
    new_key = 0

    for key in sorted(array.keys()):
        if array[key] is None:
            continue
        while new_key in new_dict:
            new_key += 1
        new_dict[new_key] = array[key]
        print(new_key, " HIEA")
        new_key += 1

    return new_dict

def subtract_keys(dictionary):
    if not dictionary:
        return dictionary  # Return the original dictionary if it's empty or None.

    # Get the first key and its corresponding value.
    first_key = next(iter(dictionary))
    # Subtract the first value from all other keys in the dictionary.
    result = {key - first_key: value for key, value in dictionary.items()}

    return result

def get_random_windowed_subset(array, window_size, start = None, offset = 0):
    if window_size <= 0 or window_size > len(array):
        raise ValueError("Window size must be greater than 0 and less than or equal to the array size.")
    
    start_index = start 

    if start_index is None:
        start_index = random.randint(offset, len(array) - window_size - offset)
    
    subset = array[start_index:start_index + window_size]
    
    return subset, start_index

def get_windowed_subsets(array, window_size):
    if window_size <= 0 or window_size > len(array):
        raise ValueError("Window size must be greater than 0 and less than or equal to the array size.")
    
    subsets = []
    for i in range(len(array) - window_size + 1):
        subset = array[i:i + window_size]
        subsets.append(subset)
    
    return subsets

def normalize_data(arr):
    # Calculate the minimum and maximum values in the array
    min_val = np.min(arr)
    max_val = np.max(arr)
    
    # Normalize the data using min-max scaling
    normalized_arr = (arr - min_val) / (max_val - min_val)
    
    return normalized_arr

def normalize_data_median(data):
    """
    Normalize a NumPy array of data by subtracting the median and dividing by the median absolute deviation (MAD).

    Parameters:
    data (numpy.ndarray): The input data array to be normalized.

    Returns:
    numpy.ndarray: The normalized data array.
    """
    median = np.median(data)
    mad = np.median(np.abs(data - median))
    
    normalized_data = (data - median) / mad
    
    return normalized_data

def remove_top_20_percent(arr, per = 0.3):
    # Calculate the number of elements to remove (top 20%)
    num_elements_to_remove = int(len(arr) * per)
    
    # Get the indices of the top 20% elements
    sorted_indices = np.argsort(arr)
    indices_to_remove = sorted_indices[-num_elements_to_remove:]
    
    # Remove the elements from the input array
    modified_arr = np.delete(arr, indices_to_remove)
    
    return modified_arr, indices_to_remove

def remove_outliers(points_dict, eps=0.5, min_samples=2):
    all_points = []

    # Collect all points from the points_dict
    key_key_array = []
    for k1 in points_dict.keys():
        for k2 in points_dict[k1].keys():
            all_points.extend(points_dict[k1][k2])
            key_key_array.append((k1, k2))

    all_points = np.array(all_points)

    all_points = all_points.reshape(-1, 4)

    # Apply DBSCAN clustering to detect outliers
    clustering = DBSCAN(eps=eps, min_samples=min_samples)
    labels = clustering.fit_predict(all_points)

    indices = np.where(labels == 0)[0]
    #print(labels)
    #print("(((())))")
    # Filter out outliers from each subset
    points_dict_filtered = {}
    idx = 0
    
    for i in indices:

        k1 = key_key_array[i][0]
        k2 = key_key_array[i][1]

        if k1 not in points_dict_filtered:
            points_dict_filtered[k1] = {k2: all_points[i]}
        else:
            points_dict_filtered[k1][k2] = all_points[i]

    return points_dict_filtered

def remove_outliers_array(all_points, eps=0.5, min_samples=2):

    # Apply DBSCAN clustering to detect outliers
    clustering = DBSCAN(eps=eps, min_samples=min_samples)
    labels = clustering.fit_predict(all_points)

    indices = np.where(labels == 0)[0]

    return all_points[indices, :]

def sample_3d_normals(theta_range, phi_range, num_samples):
    # Generate random values for theta and phi within the specified ranges
    thetas = np.random.uniform(*theta_range, size=num_samples)
    phis = np.random.uniform(*phi_range, size=num_samples)
    
    # Convert spherical coordinates to Cartesian coordinates
    x = np.sin(thetas) * np.cos(phis)
    y = np.sin(thetas) * np.sin(phis)
    z = np.cos(thetas)
    
    # Create the 3D normal vectors
    normals = np.column_stack((x, y, z))
    
    return normals

def is_point_on_plane(plane_normal, plane_point, point, tolerance=1e-6):
    # Calculate the vector from the given point to the plane point
    vector = point - plane_point
    
    # Calculate the dot product between the vector and the plane normal
    dot_product = np.dot(vector, plane_normal)
    
    # Check if the dot product is within the specified tolerance
    print(np.abs(dot_product))
    if np.abs(dot_product) < tolerance:
        return True
    
    return False

def find_ground_plane(normal_vector, distance, ray1, ray2, ray3):
    # Ensure that the normal vector is a unit vector
    normal_vector /= np.linalg.norm(normal_vector)
    
    # Compute the intersection points of the rays with the plane
    intersection1 = (distance / np.dot(ray1, normal_vector)) * ray1
    intersection2 = (distance / np.dot(ray2, normal_vector)) * ray2
    intersection3 = (distance / np.dot(ray3, normal_vector)) * ray3
    
    # Calculate the centroid of the intersection points
    centroid = (intersection1 + intersection2 + intersection3) / 3.0
    
    # Define the ground plane parameters
    ground_plane_normal = normal_vector
    ground_plane_point = centroid
    
    return ground_plane_normal, ground_plane_point

def generate_normal(seed = None, bound = [0,0,0,0]):
    
    if seed is not None:
        rand.seed(seed)

    p = np.array([rand.uniform(-5 + bound[0], bound[1] + 5), rand.uniform(-3, -1), rand.uniform(-5 + bound[2], 5 + bound[3])])

    z = p / np.linalg.norm(p)

    #z = z

    n = np.array([0,1,0])#np.array([random.uniform(-5, 5), -random.uniform(1, 11), random.uniform(-5, 5)])#np.random.randn(3)  # take a random vector
    n = n / np.linalg.norm(n)
    n -= n.dot(z) * z

    n = n / np.linalg.norm(n)

    if n[1] > 0:
        n = -1*n

    x = np.cross(n, z)
    x = x/np.linalg.norm(x)

    return n, 2*n + p

def find_plane_parameters(point1, point2, point3):
    # Calculate two vectors on the plane
    vector1 = point2 - point1
    vector2 = point3 - point1
    
    # Calculate the normal vector of the plane
    normal_vector = np.cross(vector1, vector2)
    
    # Normalize the normal vector
    normal_vector /= np.linalg.norm(normal_vector)
    
    # Calculate a point on the plane (using point1)
    plane_point = point1
    
    return normal_vector, plane_point

def distance_to_plane(plane_normal, plane_point, point):
    # Calculate the distance between the point and the plane
    # using the formula: distance = |(point - plane_point) . plane_normal| / |plane_normal|
    
    # Vector from the plane point to the given point
    vector = point - plane_point
    
    # Calculate the dot product of the vector and the plane normal
    dot_product = np.dot(vector, plane_normal)
    
    # Calculate the magnitude of the plane normal
    magnitude = np.linalg.norm(plane_normal)
    
    # Calculate the distance
    distance = abs(dot_product) / magnitude
    
    return distance

class Graph:
    def __init__(self):
        self.graph = defaultdict(list)

    def add_edge(self, u, v):
        self.graph[u].append(v)
        self.graph[v].append(u)

    def dfs(self, v, visited, component):
        visited.add(v)
        component.append(v)

        for neighbor in self.graph[v]:
            if neighbor not in visited:
                self.dfs(neighbor, visited, component)

    def get_connected_components(self):
        visited = set()
        components = []

        for vertex in self.graph:
            if vertex not in visited:
                component = []
                self.dfs(vertex, visited, component)
                components.append(component)

        return components

    def get_component_edges(self, component):
        edges = []

        for vertex in component:
            for neighbor in self.graph[vertex]:
                if neighbor in component and (neighbor, vertex) not in edges and (vertex, neighbor) not in edges:
                    edges.append((vertex, neighbor))

        return edges

def multi_view_return(bundle_rotation_matrix_array, bundle_position_matrix_array, bundle_intrinsic_matrix_array, matched_points):

    return

def match_pairs_graph(pair_array, corres):
    g = Graph()

    for i in range(len(corres)):
        ind1 = corres[i][0]
        ind2 = corres[i][1]

        for pair in pair_array[i]:
            
            g.add_edge(str(ind1) + '_' + str(pair[0]), str(ind2) + '_' + str(pair[1]))

    components = g.get_connected_components()
    for component in components:
        component_edges = g.get_component_edges(component)
        print("Component:", component)
        print("Edges:", component_edges)
        print()

        #for edge in component_edges:

    
def partition_pairwise_indices(pairwise_indices):
    sets = []
    
    for pair in pairwise_indices:
        a, b = pair
        
        # Find sets that contain a or b
        sets_with_a = [s for s in sets if a in s]
        sets_with_b = [s for s in sets if b in s]
        
        if len(sets_with_a) > 0 and len(sets_with_b) > 0:
            # Merge sets containing a and b
            merged_set = set.union(*sets_with_a, *sets_with_b)
            sets = [s for s in sets if s not in sets_with_a and s not in sets_with_b]
            sets.append(merged_set)
        elif len(sets_with_a) > 0:
            # Add b to set containing a
            sets_with_a[0].add(b)
        elif len(sets_with_b) > 0:
            # Add a to set containing b
            sets_with_b[0].add(a)
        else:
            # Create a new set with a and b
            sets.append(set([a, b]))
    
    return sets

def multidimensional_frame_hungarian(points):
    num_sets = len(points)
    max_size = max(len(p) for p in points)

    # Create the distance matrix
    distance_matrix = np.zeros((num_sets, max_size, max_size))
    for i, p in enumerate(points):
        for j, q in enumerate(points):
            if i == j:
                continue
            distance_matrix[i, :len(p), :len(q)] = np.linalg.norm(p[:, np.newaxis] - q, axis=2)

    # Modify the distance matrix to satisfy the conditions
    max_distance = np.max(distance_matrix)
    for i in range(num_sets):
        np.fill_diagonal(distance_matrix[i], max_distance)
        for j in range(num_sets):
            if i != j:
                distance_matrix[i, :len(points[i]), :len(points[j])] = max_distance

    # Apply the Hungarian algorithm
    matched_points = []
    for k in range(num_sets):
        row_indices, col_indices = linear_sum_assignment(distance_matrix[k])
        matched_points.extend((k, i, j) for i, j in zip(row_indices, col_indices))

    # Interpret the results
    matched_indices = set()
    for k, i, j in matched_points:
        if j < len(points[k]):
            matched_indices.add((k, i, j))

    matched_points = []
    unmatched_points = []
    for k in range(num_sets):
        for i, p in enumerate(points[k]):
            if (k, i, 0) not in matched_indices:
                unmatched_points.append((k, i))
            for _, _, j in matched_indices:
                if i == j:
                    matched_points.append((k, i, j))

    return matched_points, unmatched_points, matched_indices

def distortion_apply(u, v, cam, Rot, t, k, p):
    # Step 1: Convert pixel coordinate to normalized image coordinate
    R = torch.transpose(Rot, 0, 1)
    cx = cam[0][2]
    cy = cam[1][2] 
    fx = cam[0][0]
    fy = cam[1][1]

    k1, k2, k3 = k
    p1, p2 = p

    # Step 1: Convert pixel coordinate to normalized image coordinate
    x = (u - cx) / fx
    y = (v - cy) / fy
    '''
    print(x, " THE X")
    print(y, " THE Y")
    '''

    # Step 2: Apply inverse distortion to normalized coordinate
    r2 = x*x + y*y
    #x1 = x * (1 + k1*r2 + k2*r2*r2 + k3*r2*r2*r2) + 2*p1*x*y + p2*(r2 + 2*x*x)
    #y1 = y * (1 + k1*r2 + k2*r2*r2 + k3*r2*r2*r2) + p1*(r2 + 2*y*y) + 2*p2*x*y

    x1 = x * (1 + k1*r2**2 + k2*r2**4 + k3*r2**6) + 2*p1*x*y + p2*(r2 + 2*x*x)
    y1 = y * (1 + k1*r2**2 + k2*r2**4 + k3*r2**6) + p1*(r2 + 2*y*y) + 2*p2*x*y


    #print("****************************************************")
    #print(x1, " X1 Y1 !!!!!!!!!!!!!!!!!!!!!!!!")
    #print(y1, " y1 11111111111111111111111")
    #print(u, " U !!!!!!!!!!!!!!!!!!!!!!!!")
    #print(v, " V 11111111111111111111111")

    # Step 3: Convert undistorted normalized coordinate to camera coordinate
    Xc = R[0,0]*x1 + R[0,1]*y1 + R[0,2]*1.0 
    Yc = R[1,0]*x1 + R[1,1]*y1 + R[1,2]*1.0
    Zc = R[2,0]*x1 + R[2,1]*y1 + R[2,2]*1.0
    '''
    print("*****************************************")
    print(Xc, " THE Xc")
    print(Yc, " THE Yc")
    print(Zc, " THE Zc")
    '''

    Xc = Xc + t[0]
    Yc = Yc + t[1]
    Zc = Zc + t[2]
    '''
    print("*****************************************")
    print(Xc, " THE Xc + t")
    print(Yc, " THE Yc + t")
    print(Zc, " THE Zc + t")
    '''

    # Step 4: Convert camera coordinates to world coordinates by dividing by Zc
    X = Xc# / Zc
    Y = Yc# / Zc
    Z = Zc# / Zc # Note: this is just 1.0
    # stack to get world coordinates for all points
    #print(X, " X !!!!!!!!!!!!!!!!!!!!!!!!")
    #print(Y, " Y 11111111111111111111111")
    #print(Z, " Z !!!!!!!!!!!!!!!!!!!!!!!!")
    '''
    print(X, " END Xc")
    print(Y, " END Yc")
    print(Z, " END Zc")
    '''

    world_coords = torch.stack((X,Y,Z), dim=1)
    #print(world_coords, " world_coordsworld_coordsworld_coordsworld_coords")
    return world_coords 

def natural_key(string_):
    """See https://blog.codinghorror.com/sorting-for-humans-natural-sort-order/"""
    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', string_)]

#Use this as reference
#https://ksimek.github.io/2012/08/22/extrinsic/
def extrinsics_to_camera_pose(R, t):
    
    R1 = np.transpose(R)
    t1 = -np.transpose(R) @ t

    return R1, t1

def camera_pose_to_extrinsics(R, t):
    return 

def rotation_matrix_axis_angle(axis, angle):
    """
    Computes the 3 by 3 rotation matrix from an arbitrary axis and angle using NumPy.
    
    Parameters:
        axis (list or array-like): A 3-element list or array representing the axis of rotation.
        angle (float): The angle of rotation in radians.
    
    Returns:
        A 3 by 3 NumPy array representing the rotation matrix.
    """
    axis = np.asarray(axis)
    axis = axis / np.sqrt(np.dot(axis, axis))
    a = np.cos(angle / 2.0)
    b, c, d = -axis * np.sin(angle / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    return np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac), 0],
                     [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab), 0],
                     [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc, 0],
                     [0,0,0,1]])

#A = np.random.rand(10, 2)
#B = np.random.rand(8, 2)

def pairwise_distances(arrays):
    """
    Compute the pairwise distances between all elements in an arbitrary
    number of arrays.
    """
    num_arrays = len(arrays)
    num_elements = tuple(len(arr) for arr in arrays)
    distances = np.zeros(num_elements * num_arrays)

    # Compute the pairwise distances between all elements
    for i, arr1 in enumerate(arrays):
        for j, arr2 in enumerate(arrays):
            indices = tuple(slice(None) if k == i else np.newaxis for k in range(num_arrays))
            distances[indices] += np.sum((arr1[indices] - arr2[indices])**2, axis=-1)

    distances = np.sqrt(distances)
    return distances

def multidimensional_hungarian(arrays):
    '''
    # Example usage
    A = np.random.rand(10, 2)
    B = np.random.rand(8, 2)
    C = np.random.rand(6, 2)
    D = np.random.rand(12, 2)

    arrays = [A, B, C, D]
    '''
    # Calculate the pairwise distances between all elements
    distances = pairwise_distances(arrays)

    # Use the munkres algorithm to find the optimal assignment
    indices = linear_sum_assignment(distances)

    # Extract the matched elements from all arrays
    matched_arrays = [arr[indices[i]] for i, arr in enumerate(arrays)]

    # Print the results
    for i, matched_arr in enumerate(matched_arrays):
        print(f"Matched array {i+1}: {matched_arr}")

def hungarian_assignment_one2one(A, B):
    distances = cdist(A, B)

    # Use the Hungarian algorithm to find the optimal assignment
    row_ind, col_ind = linear_sum_assignment(distances)

    # Extract the matched elements from Array A and the corresponding elements from Array B
    matched_A = A[row_ind]
    matched_B = B[col_ind]

    # Compute the minimum pairwise norm
    '''
    min_pairwise_norm = np.linalg.norm(matched_A - matched_B, axis=1).sum()

    # Print the results
    print("Matched A:", matched_A)
    print("Matched B:", matched_B)
    print("Minimum pairwise norm:", min_pairwise_norm)
    '''

    return row_ind, col_ind

def hungarian_assignment(A, B):

    # Calculate the pairwise distances
    distances = np.zeros((len(A), len(B)))
    for i in range(len(A)):
        for j in range(len(B)):
            distances[i,j] = np.linalg.norm(A[i] - B[j])

    # Use the Hungarian algorithm to find the optimal assignment
    row_ind, col_ind = linear_sum_assignment(distances)

    # Extract the matched elements from Array A and the corresponding elements from Array B
    matched_A = A[row_ind]
    matched_B = B[col_ind]

    # Find the unmatched elements in Array B
    #unmatched_B = np.delete(B, col_ind, axis=0)

    return matched_A, matched_B, row_ind, col_ind

def nearest_neighbor_dp(arrays):
    n = len(arrays)
    k = arrays[0].shape[0]

    # Compute the pairwise distances between all elements in all arrays
    distances = np.zeros((n, k, n, k))
    for i in range(n):
        for j in range(n):
            for p in range(k):
                for q in range(k):
                    distances[i, p, j, q] = np.linalg.norm(arrays[i][p] - arrays[j][q])

    # Initialize the dynamic programming table
    dp = np.zeros((n, k))

    # Compute the optimal solution using dynamic programming
    for i in range(1, n):
        for j in range(k):
            dp[i, j] = np.inf
            for p in range(k):
                dp[i, j] = min(dp[i, j], dp[i-1, p] + distances[i-1, p, i, j])

    # Backtrack to recover the optimal solution
    indices = np.zeros(n, dtype=np.int64)
    indices[n-1] = np.argmin(dp[n-1])
    for i in range(n-2, -1, -1):
        min_index = -1
        min_value = np.inf
        for j in range(k):
            if dp[i, j] + distances[i, j, i+1, indices[i+1]] < min_value:
                min_index = j
                min_value = dp[i, j] + distances[i, j, i+1, indices[i+1]]
        indices[i] = min_index

    return indices

def nearest_neighbor_indices(arrays):
    n = len(arrays)
    result = np.zeros(n, dtype=int)
    result[0] = 0  # Choose arbitrary element from first array
    
    for i in range(1, n):
        last_selected = arrays[result[i-1]][result[i-1], :]
        closest_array_idx, closest_element_idx = min(
            [(j, k) for j in range(n) for k in range(arrays[j].shape[0])],
            key=lambda x: np.linalg.norm(arrays[x[0]][x[1], :] - last_selected)
        )
        result[i] = closest_element_idx
    
    return result

def nearest_neighbor(arrays):
    n = len(arrays)
    result = np.empty((n, 2), dtype=int)
    result[0] = arrays[0][0]  # Choose arbitrary element from first array
    
    for i in range(1, n):
        last_selected = result[i-1]
        closest_element = min(arrays[i], key=lambda x: np.linalg.norm(x - last_selected))
        result[i] = closest_element
    
    return result


def select_elements_2d(arrays):
    n = len(arrays)  # number of arrays
    m = arrays[0].shape[0]  # size of each array
    dp = np.zeros((n, m))  # dynamic programming array to store minimum norms
    parent = np.zeros((n, m), dtype=int)  # parent array to store selected elements

    # Step 1: Calculate the minimum norm for the first array
    dp[0] = np.linalg.norm(arrays[0], axis=1)

    # Step 2: For each subsequent array, calculate the minimum norm for all possible pairs of elements
    for i in range(1, n):
        prev_dp = dp[i - 1]  # minimum norms from the previous array
        for j in range(m):
            curr_array = arrays[i]  # current array
            curr_element = curr_array[j]  # current element
            norms = np.linalg.norm(curr_element - prev_dp, axis=1)  # pairwise norms with previous elements
            min_index = np.argmin(norms)  # index of the minimum norm
            dp[i, j] = norms[min_index]  # update minimum norm
            parent[i, j] = min_index  # store selected element index in parent array

    # Step 3: Update the minimum norm and selected elements for each array
    result = np.zeros((n, 2), dtype=int)
    result[-1] = arrays[-1][np.argmin(dp[-1])]  # select the element with minimum norm from the last array
    for i in range(n - 2, -1, -1):
        result[i] = arrays[i][parent[i + 1, result[i + 1]]]  # select the corresponding element from the parent array

    # Step 4: Return the selected elements
    return result
'''
If I have n amount of 2D arrays,  how to do I select 1 element from each 2D array such that the pairwise difference is mimimal?

To select one element from each 2D array such that the pairwise difference is minimal, you can use a dynamic programming approach. Here are the steps you can follow:

Calculate the minimum difference for the first array: For the first 2D array, simply select any element as the starting point. This will give you a base value for the minimum difference.

For each subsequent 2D array, calculate the minimum difference for all possible pairs of elements: Starting from the second 2D array, you will need to compare each element in the current array with all the previously selected elements from the previous arrays. For each element in the current array, calculate the minimum difference with all previously selected elements, and keep track of the pair of elements that gives the minimum difference.

Update the minimum difference and selected elements for each array: Once you have calculated the minimum difference for the current array, update the minimum difference and selected elements for all the previous arrays. Specifically, for each previous array, update the selected element to be the element that gave the minimum difference when compared with the current element in the current array.

The final solution is the selected element from each array: Once you have completed step 3 for all the arrays, the selected element from each array will give you the minimum pairwise difference.

This approach will have a time complexity of O(n^2m), where n is the number of arrays and m is the size of each array. However, this can be improved to O(nmlogm) by sorting each array and using binary search to find the closest element in the previous arrays.
'''
# SOLVE MINUMUM SUBSET PROBLEM TO GET LOWEST STD
def select_elements(arrays):
    n = len(arrays)  # number of arrays
    m = arrays[0].shape[0]  # size of each array
    dp = np.zeros((n, m))  # dynamic programming array to store minimum differences
    parent = np.zeros((n, m), dtype=int)  # parent array to store selected elements

    # Step 1: Calculate the minimum difference for the first array
    dp[0] = np.zeros(m)

    # Step 2: For each subsequent array, calculate the minimum difference for all possible pairs of elements
    for i in range(1, n):
        prev_dp = dp[i - 1]  # minimum differences from the previous array
        for j in range(m):
            curr_array = arrays[i]  # current array
            curr_element = curr_array[j]  # current element
            differences = np.abs(curr_element - prev_dp)  # pairwise differences with previous elements
            min_index = np.argmin(differences)  # index of the minimum difference
            dp[i, j] = differences[min_index]  # update minimum difference
            parent[i, j] = min_index  # store selected element index in parent array

    # Step 3: Update the minimum difference and selected elements for each array
    result = np.zeros(n, dtype=int)
    result[-1] = np.argmin(dp[-1])  # select the element with minimum difference from the last array
    for i in range(n - 2, -1, -1):
        result[i] = parent[i + 1, result[i + 1]]  # select the corresponding element from the parent array

    # Step 4: Return the selected elements
    return [arrays[i][result[i]] for i in range(n)]
'''
# Example usage
array1 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
array2 = np.array([[2, 3, 4], [5, 6, 7], [8, 9, 10]])
array3 = np.array([[3, 4, 5], [6, 7, 8], [9, 10, 11]])
arrays = [array1, array2, array3]
selected = select_elements(arrays)
print(selected)
'''

def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return rho, phi

def pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return x, y

def get_pose(datastore):
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

    dict_2d = {}
    
    init_dict = datastore.getData()

    for fr in list(init_dict.keys()):

        pose_array = init_dict[fr]

        dict_2d[fr] = {}
        for ppl in list(pose_array):


            ppl.pop("bbox", None)
            ppl.pop("id", None)
            dict_2d = list(ppl.values())
    return dict_2d

def get_ankles_heads_pose_dictionary(datastore, cond_tol = 0.8, keep = -1, keep_list = []):
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

    dict_2d = {}
    
    init_dict = datastore.getData()

    for fr in list(init_dict.keys()):

        pose_array = init_dict[fr]

        dict_2d[fr] = {}
        for ppl in list(pose_array):
            '''
            left_ankle = ppl["left_ankle"]
            right_ankle = ppl["right_ankle"]
            #print(ppl)            
            #stop
            ankle_x, ankle_y = determine_foot(right_ankle,left_ankle, hgt_threshold=8.0, wide_threshold=10.0)
            
            ankle_x = ppl['hip'][0]
            head_x = (ppl["Thorax"][0])
            head_y = (ppl["Thorax"][1])

            dict_2d[fr][ppl['idx']] = ([ankle_x, ankle_y, head_x, head_y, ppl])
            '''
            left_ankle = ppl["left_ankle"]
            right_ankle = ppl["right_ankle"]

            ankle_left_conf = ppl["left_ankle"][2]
            ankle_right_conf = ppl["right_ankle"][2]

            head_conf = (ppl["left_shoulder"][2] + ppl["right_shoulder"][2])/2.0

            avg_cond = (ankle_left_conf + ankle_right_conf + ppl["left_shoulder"][2] + ppl["right_shoulder"][2])/4.0
            #avg_cond = ppl["bbox"][4]
            #print(avg_cond, " HELLO?")
            if fr > keep:
                if avg_cond < cond_tol and fr not in keep_list:
                    continue
            head_x = ((ppl["left_shoulder"][0]) + (ppl["right_shoulder"][0]))/2.0
            head_y = ((ppl["left_shoulder"][1]) + (ppl["right_shoulder"][1]))/2.0
            #ankle_x, ankle_y = determine_foot_bbox(right_ankle, left_ankle, datastore.getitem(ppl))
            max_size = max(np.linalg.norm(np.array([left_ankle[0], left_ankle[1]]) - np.array([head_x, head_y])), np.linalg.norm(np.array([right_ankle[0], right_ankle[1]]) - np.array([head_x, head_y])))
            hgt_thresh = 0.2*max_size
            wide_tresh = 0.2*max_size
            ankle_x, ankle_y = determine_foot(right_ankle,left_ankle, hgt_threshold=hgt_thresh, wide_threshold=wide_tresh)
                
            #head_x = (datastore.getitem(ppl)["Thorax"][0])
            #head_y = (datastore.getitem(ppl)["Thorax"][1])


            ppl.pop("bbox", None)
            dict_2d[fr][ppl['id']] = ([ankle_x, ankle_y, head_x, head_y, ppl, ankle_left_conf, ankle_right_conf, head_conf])
   
        if len(dict_2d[fr]) == 0:
            dict_2d.pop(fr, None)
    return dict_2d

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


def get_ankles_heads_dictionary(datastore, cond_tol = 0.8, keep = -1, keep_list = []):
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

    dict_2d = {}
    
    init_dict = datastore.getData()

    for fr in list(init_dict.keys()):

        pose_array = init_dict[fr]

        dict_2d[fr] = {}
        for ppl in list(pose_array):
            '''
            left_ankle = ppl["left_ankle"]
            right_ankle = ppl["right_ankle"]

            ankle_x, ankle_y = determine_foot(right_ankle,left_ankle, hgt_threshold=8.0, wide_threshold=10.0)
            
            ankle_x = ppl['hip'][0]
            head_x = (ppl["Thorax"][0])
            head_y = (ppl["Thorax"][1])
            '''
            left_ankle = ppl["left_ankle"]
            right_ankle = ppl["right_ankle"]

            ankle_left_conf = ppl["left_ankle"][2]
            ankle_right_conf = ppl["right_ankle"][2]

            avg_cond = (ankle_left_conf + ankle_right_conf + ppl["left_shoulder"][2] + ppl["right_shoulder"][2])/4.0
            #avg_cond = ppl["bbox"][4]
            
            if fr > keep:
                if avg_cond < cond_tol and fr not in keep_list:
                    #print(avg_cond, cond_tol, "HI this shoudlnt happen")
                    continue
            #print(fr, " THIS IS THE FRAME")

            head_x = ((ppl["left_shoulder"][0]) + (ppl["right_shoulder"][0]))/2.0
            head_y = ((ppl["left_shoulder"][1]) + (ppl["right_shoulder"][1]))/2.0
            #ankle_x, ankle_y = determine_foot_bbox(right_ankle, left_ankle, datastore.getitem(ppl))
            max_size = max(np.linalg.norm(np.array([left_ankle[0], left_ankle[1]]) - np.array([head_x, head_y])), np.linalg.norm(np.array([right_ankle[0], right_ankle[1]]) - np.array([head_x, head_y])))
            hgt_thresh = 0.2*max_size
            wide_tresh = 0.2*max_size
            ankle_x, ankle_y = determine_foot(right_ankle,left_ankle, hgt_threshold=hgt_thresh, wide_threshold=wide_tresh)
                
            #head_x = (datastore.getitem(ppl)["Thorax"][0])
            #head_y = (datastore.getitem(ppl)["Thorax"][1])

            head_conf = (ppl["left_shoulder"][2] + ppl["right_shoulder"][2])/2.0
            dict_2d[fr][ppl['id']] = ([ankle_x, ankle_y, head_x, head_y, ankle_left_conf, ankle_right_conf, head_conf])
        if len(dict_2d[fr]) == 0:
            dict_2d.pop(fr, None)


    return dict_2d

def get_ankles_heads_array(datastore):
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

    dict_2d = []
    
    init_dict = datastore.getData()

    for fr in list(init_dict.keys()):

        pose_array = init_dict[fr]

        for ppl in list(pose_array):

            left_ankle = ppl["left_ankle"]
            right_ankle = ppl["right_ankle"]

            ankle_x, ankle_y = determine_foot(right_ankle,left_ankle, hgt_threshold=8.0, wide_threshold=10.0)
                
            head_x = (ppl["Thorax"][0])
            head_y = (ppl["Thorax"][1])

            dict_2d.append([ankle_x, ankle_y, head_x, head_y])

    return dict_2d


def interpolate_multi(coords_input):

    coords = {k:v for k,v in coords_input.items() if v}
    #print(coords)
    #stop
    key_names = list(coords.keys())
    #print(len(key_names), " KEY LENGHT")

    points_dict = {}
    for i in range(len(key_names) - 1):
        coords1 = coords[key_names[i]]
        coords2 = coords[key_names[i + 1]]
        #print(key_names[i + 1], key_names[i])
        if key_names[i + 1] - key_names[i] == 1:
            points_dict[key_names[i]] = coords1
            points_dict[key_names[i + 1]] = coords2
        else:

            index1, index2, d2_sorted = knn.knn(np.array(coords1),np.array(coords2))
            #print(key_names[i + 1], key_names[i], " KEY NAME")
            grid = np.arange(key_names[i], key_names[i + 1])
            #print(grid, " grid")
            #print(range(key_names[i + 1] - key_names[i] - 1), " LISTTTTTTTTTTTTTTTTTT")
            for g in range(key_names[i + 1] - key_names[i]):
                points_dict[grid[g]] = []

            for ind in range(len(index1)):

                points = np.transpose(np.array([coords1[index1[ind]], coords2[index2[ind]]]))
                time = [key_names[i], key_names[i + 1]]
                y_new = interpolate(points, time, grid)
                '''
                print(time, "TIME")
                print(grid, " grid")
                print(y_new.shape, " y new")
                '''

                for intp in range(y_new.shape[1]):
                    points_dict[grid[intp]].append(y_new[:, intp])


    points_dict[key_names[-1]] = coords[key_names[-1]]
    '''
    print(list(points_dict.keys()))
    #print(list(points_dict.keys()))
    print("*********************************************************")
    for k in range(len(list(points_dict.keys())) - 1):
        if list(points_dict.keys())[k + 1] - list(points_dict.keys())[k] > 1:
            print(list(points_dict.keys())[k], list(points_dict.keys())[k + 1], k," points dict")
    
    print(len(list(points_dict.keys())),  " point dict")
    stop
    '''
    #print(list(points_dict.keys()))
    #print(len(list(points_dict.keys())))
    #stop
    return points_dict, list(points_dict.keys())

def diff_multi(coords, dilation = 1):

    first_array = []
    middle_array = []
    last_array = []

    key_names = list(coords.keys())

    #print(coords, " HIIIIIIIIII")
    for i in range(len(key_names)):
        
        if i < dilation:
            coords1 = coords[key_names[i]]
            coords2 = coords[key_names[i + dilation]]    

            index1, index2, d2_sorted = knn.knn(np.array(coords1),np.array(coords2))

            coords1_match = np.array(coords1)[index1]
            coords2_match = np.array(coords2)[index2]

            norm_diff = np.linalg.norm(coords1_match - coords2_match, axis = 1)
            avg_diff = np.mean(norm_diff)

            first_array.append(avg_diff)

        if i >= dilation and i < len(key_names) - dilation:
            coords1 = coords[key_names[i - dilation]]
            coords2 = coords[key_names[i + dilation]]    
            
            index1, index2, d2_sorted = knn.knn(np.array(coords1),np.array(coords2))

            coords1_match = np.array(coords1)[index1]
            coords2_match = np.array(coords2)[index2]

            norm_diff = np.linalg.norm((coords1_match - coords2_match)/2, axis = 1)
            avg_diff = np.mean(norm_diff)

            middle_array.append(avg_diff)

        if i >= len(key_names) - dilation:
            
            coords1 = coords[key_names[i - dilation]]
            coords2 = coords[key_names[i]]    
            
            index1, index2, d2_sorted = knn.knn(np.array(coords1),np.array(coords2))

            coords1_match = np.array(coords1)[index1]
            coords2_match = np.array(coords2)[index2]

            norm_diff = np.linalg.norm(coords1_match - coords2_match, axis = 1)
            avg_diff = np.mean(norm_diff)

            last_array.append(avg_diff)
    
    return first_array + middle_array + last_array

def div_multi(coords, time = 1):
    return list(np.array(coords)/time)

def interpolate(points, time, grid):
    
    f = scipy.interpolate.interp1d(time, points)
    y_new = f(grid)
    return y_new

def unique_matching(list1, list2):

    match_array = []
    #print(list1.shape, list2.shape, " TENOSRRRRRRRRRRRRRRRRRRRRRRRRRRRR")
    #print(list1, list2, " list1 !!!!!!!!!!!!!!!!!!!!!!!!!")
    print(list1.shape, " LIST 111111111111111")
    for i in range(list1.shape[1]):
    
        if (list2.shape[1]) > 0: #When there are elements in list2
            #print(list1.shape, list2.shape, "SHAOEAWAESESSEEEEEEEEE")
            temp_result = np.linalg.norm(list1[i] - list2, axis = 0) #Matrix subtraction
            #print(temp_result.shape, " TEMMERMERERRRRRRRRRRR")
            min_val = np.amin(temp_result) #Getting the minimum value to get closest element
            #print(min_val, " min val")
            #print(min_val.shape, " minvvalalala")
            min_val_index = np.where(temp_result == min_val) #To find index of minimum value
            #print(min_val_index, " MIN VAL INDEX")
            #print(min_val_index, " MIN VAL INDEX !!!!!!!!!!!!!!!!!!!!!!")
            closest_element = np.squeeze(list2[:, min_val_index]) #Actual value of closest element in list2
            #print(closest_element, " CLOSEST ELEMEMNTASDASDAS")
            #print(list2 != closest_element)
            #print(np.array(closest_element).shape, " SHAEASEASEAWEAWE")
            list2 = list2[list2 != closest_element] #Remove closest element after found

            #print(i, list1[i], min_val_index[0][0], closest_element[0]) #List1 Index, Element to find, List2 Index, Closest Element
            match_array.append([i, min_val_index[0][0]])
        else: #All elements are already found

            break
    print(match_array, " MATHC ARAYASASASAS")
    return match_array

def find_matching_index(list1, list2):
    
    inverse_index = { element: index for index, element in enumerate(list1) }

    return [[index, inverse_index[element]]
        for index, element in enumerate(list2) if element in inverse_index]
        
# you must keep track of the matches as well ...

def sync_view(view_dict, view_array, pose_array, ref_sync, view_sync, img_array, view_calibration, view_extrinsic):
    #detect occlusion?
    params_focal = []
    params_normal = []
    params_plane_point = []
    params_rotation = []
    params_rot_point = []

    coord_swap = torch.tensor([[1,0,0], [0,0,1], [0,1,0]]).double()

    for i in range(len(view_calibration)):
    
        v = view_calibration[i]
        v_e = view_extrinsic[i]

        params_focal.append({'params': torch.nn.Parameter(torch.tensor([v['cam_matrix'][0][0]], requires_grad=True).double())})
        params_focal.append({'params': torch.nn.Parameter(torch.tensor([v['cam_matrix'][0][0]], requires_grad=True).double())}) #
        
        params_normal.append({'params': torch.nn.Parameter(torch.tensor([v['normal'][0]]).double(), requires_grad=True)})
        params_normal.append({'params': torch.nn.Parameter(torch.tensor([v['normal'][1]]).double(), requires_grad=True)})
        params_normal.append({'params': torch.nn.Parameter(torch.tensor([v['normal'][2]]).double(), requires_grad=True)})

        params_plane_point.append({'params': torch.nn.Parameter(torch.tensor([v['ankleWorld'][0]]).double(), requires_grad=True)}) # ANKELWORLD MEANS PLANE POINT
        params_plane_point.append({'params': torch.nn.Parameter(torch.tensor([v['ankleWorld'][1]]).double(), requires_grad=True)})
        params_plane_point.append({'params': torch.nn.Parameter(torch.tensor([v['ankleWorld'][2]]).double(), requires_grad=True)})

        params_rotation.append({'params': torch.nn.Parameter(torch.tensor([v_e['angle'][0]]).double(), requires_grad=True)}) #

    for i in range(len(view_calibration)):
            
        #######
        focal_i = torch.absolute(params_focal[2*i]['params'])
        focal1_i = torch.absolute(params_focal[2*i + 1]['params'])

        #print(focal_i, " HIIIASDASD")
        normal_i = torch.squeeze(torch.stack([params_normal[3*i]['params'][0], params_normal[3*i + 1]['params'][0], params_normal[3*i + 2]['params'][0]]))/torch.norm(torch.stack([params_normal[3*i]['params'][0], params_normal[3*i + 1]['params'][0], params_normal[3*i + 2]['params'][0]]))
        init_point_i =  torch.tensor([params_plane_point[3*i]['params'][0], params_plane_point[3*i + 1]['params'][0], params_plane_point[3*i + 2]['params'][0]]).double()
        
        t1 = torch.tensor([img_array[i].shape[1]/2]).double()
        t2 = torch.tensor([img_array[i].shape[0]/2]).double()
        #######
        cam_matrix_i = torch.squeeze(torch.stack([torch.stack([focal_i, torch.zeros(1).double(), torch.tensor([t1]).double()]), torch.stack([torch.zeros(1).double(), focal1_i, torch.tensor([t2]).double()]), torch.stack([torch.zeros(1).double(), torch.zeros(1).double(), torch.ones(1).double()])], dim = 0))
        cam_inv_i = torch.squeeze(torch.stack([torch.stack([1/focal_i, torch.zeros(1).double(), torch.tensor([-t1]).double()/focal_i]), torch.stack([torch.zeros(1).double(), 1/focal1_i, torch.tensor([-t2]).double()/focal1_i]), torch.stack([torch.zeros(1).double(), torch.zeros(1).double(), torch.ones(1).double()])], dim = 0))

        sync_world_i = plane_ray_intersection_torch(view_array[i][0, :], view_array[i][1, :], cam_inv_i, normal_i, init_point_i)
        rot_view_i = coord_swap @ basis_change_rotation_matrix_torch(cam_matrix_i, cam_inv_i, init_point_i, normal_i, torch.tensor([img_array[i].shape[0]]).double(), torch.tensor([img_array[i].shape[1]]).double())

        sync_plane_i = torch.matmul(rot_view_i, sync_world_i)[:, np.array(view_sync[i])]
        sync_center_i = torch.mean(sync_plane_i, dim = 1)

        params_rot_point.append({'params': torch.nn.Parameter(torch.tensor([sync_center_i[0]], requires_grad=True).double())}) # ANKELWORLD MEANS PLANE POINT
        params_rot_point.append({'params': torch.nn.Parameter(torch.tensor([sync_center_i[1]], requires_grad=True).double())})
        params_rot_point.append({'params': torch.nn.Parameter(torch.tensor([sync_center_i[2]], requires_grad=True).double())})

    error_array = []
    ####################
    # ref transform

    i = 0

    all_kp_match = []
    for j in range(0, len(view_calibration)):

        t1_j = torch.tensor([img_array[j].shape[1]/2]).double()
        t2_j = torch.tensor([img_array[j].shape[0]/2]).double()

        focal_j = torch.absolute(params_focal[2*j]['params'])
        focal1_j = torch.absolute(params_focal[2*j + 1]['params'])
        #OPTIMIZE FOR 3 PARAMETERS !!!!
        normal_j = torch.squeeze(torch.stack([params_normal[3*j]['params'][0], params_normal[3*j + 1]['params'][0], params_normal[3*j + 2]['params'][0]]))/torch.norm(torch.stack([params_normal[3*j]['params'][0], params_normal[3*j + 1]['params'][0], params_normal[3*j + 2]['params'][0]]))
        
        init_point_j = torch.squeeze(torch.stack([params_plane_point[3*j]['params'][0], params_plane_point[3*j + 1]['params'][0], params_plane_point[3*j + 2]['params'][0]]).double())

        #print(focal_j, torch.zeros(1).double(), torch.tensor([t1_j]).double(), torch.zeros(1).double(), focal1_j, torch.tensor([t2_j]).double(), torch.zeros(1).double(), torch.zeros(1).double(), torch.ones(1).double())
        cam_matrix_j = torch.squeeze(torch.stack([torch.stack([focal_j, torch.zeros(1).double(), torch.tensor([t1_j]).double()]), torch.stack([torch.zeros(1).double(), focal1_j, torch.tensor([t2_j]).double()]), torch.stack([torch.zeros(1).double(), torch.zeros(1).double(), torch.ones(1).double()])], dim = 0))
        cam_inv_j = torch.squeeze(torch.stack([torch.stack([1/focal_j, torch.zeros(1).double(), torch.tensor([-t1_j]).double()/focal_j]), torch.stack([torch.zeros(1).double(), 1/focal1_j, torch.tensor([-t2_j]).double()/focal1_j]), torch.stack([torch.zeros(1).double(), torch.zeros(1).double(), torch.ones(1).double()])], dim = 0))

        rot_view_j = coord_swap @ basis_change_rotation_matrix_torch(cam_matrix_j, cam_inv_j, init_point_j, normal_j, torch.tensor([img_array[j].shape[1]]).double(), torch.tensor([img_array[j].shape[0]]).double())
        
        #rot_view_j = coord_swap @ R_zj @ R_yj @ R_xj
        #################################
        #print(params_rot_point, " PARAMS ROT POINT")
        #print("******************************************************************")
        #print(params_rot_point[3*i]['params'][0], params_rot_point[3*i + 1]['params'][0], params_rot_point[3*i + 2]['params'][0], " PARAMS")
        sync_center_i = torch.squeeze(torch.transpose(torch.stack([params_rot_point[3*i]['params'], params_rot_point[3*i + 1]['params'], params_rot_point[3*i + 2]['params']]), 0, 1))
        #ref_center_i = torch.stack([params_translation[3*i]['params'][0], params_translation[3*i + 1]['params'][0], params_translation[3*i + 2]['params'][0]])

        sync_center_j = torch.squeeze(torch.transpose(torch.stack([params_rot_point[3*j]['params'], params_rot_point[3*j + 1]['params'], params_rot_point[3*j + 2]['params']]), 0, 1))

        if i == 0:
            ref_center = sync_center_i

        if j == 0:
            ref_center = sync_center_j
        #ref_center_j = torch.stack([params_translation[3*j]['params'][0], params_translation[3*j + 1]['params'][0], params_translation[3*j + 2]['params'][0]])
        #################################
        #######
        focal_i = torch.absolute(params_focal[2*i]['params'])
        focal1_i = torch.absolute(params_focal[2*i + 1]['params'])
        normal_i = torch.squeeze(torch.stack([params_normal[3*i]['params'][0], params_normal[3*i + 1]['params'][0], params_normal[3*i + 2]['params'][0]]))/torch.norm(torch.stack([params_normal[3*i]['params'][0], params_normal[3*i + 1]['params'][0], params_normal[3*i + 2]['params'][0]]))
        #init_point = torch.tensor(view_calibration[i]['ankleWorld'])
        init_point_i =  torch.squeeze(torch.stack([params_plane_point[3*i]['params'][0], params_plane_point[3*i + 1]['params'][0], params_plane_point[3*i + 2]['params'][0]]).double())
        
        t1 = torch.tensor([img_array[i].shape[1]/2]).double()
        t2 = torch.tensor([img_array[i].shape[0]/2]).double()
        #######
        #print(focal_i, " HEWALKLEWQKQQW")
        cam_matrix_i = torch.squeeze(torch.stack([torch.stack([focal_i, torch.zeros(1).double(), torch.tensor([t1]).double()]), torch.stack([torch.zeros(1).double(), focal1_i, torch.tensor([t2]).double()]), torch.stack([torch.zeros(1).double(), torch.zeros(1).double(), torch.ones(1).double()])], dim = 0))
        cam_inv_i = torch.squeeze(torch.stack([torch.stack([1/focal_i, torch.zeros(1).double(), torch.tensor([-t1]).double()/focal_i]), torch.stack([torch.zeros(1).double(), 1/focal1_i, torch.tensor([-t2]).double()/focal1_i]), torch.stack([torch.zeros(1).double(), torch.zeros(1).double(), torch.ones(1).double()])], dim = 0))

        rot_view_i = coord_swap @ basis_change_rotation_matrix_torch(cam_matrix_i, cam_inv_i, init_point_i, normal_i, torch.tensor([img_array[i].shape[0]]).double(), torch.tensor([img_array[i].shape[1]]).double())
    

        rot_y_i = torch.squeeze(torch.stack([torch.stack([torch.cos(params_rotation[i]['params']).double(), torch.tensor([0]).double(), -torch.sin(params_rotation[i]['params']).double()]), torch.stack([torch.tensor([0]).double(), torch.tensor([1]).double(), torch.tensor([0]).double()]), torch.stack([torch.sin(params_rotation[i]['params']).double(), torch.tensor([0]).double().double(), torch.cos(params_rotation[i]['params']).double()])], dim = 0))
        rot_y_j = torch.squeeze(torch.stack([torch.stack([torch.cos(params_rotation[j]['params']).double(), torch.tensor([0]).double(), -torch.sin(params_rotation[j]['params']).double()]), torch.stack([torch.tensor([0]).double(), torch.tensor([1]).double(), torch.tensor([0]).double()]), torch.stack([torch.sin(params_rotation[j]['params']).double(), torch.tensor([0]).double().double(), torch.cos(params_rotation[j]['params']).double()])], dim = 0))
        
        other_cam_i = torch.squeeze((torch.matmul(rot_y_i, (rot_view_i @ torch.unsqueeze(torch.tensor([0,0,0]), 1).double()) - torch.unsqueeze(sync_center_i, dim = 1).repeat(1, 1)) + torch.unsqueeze(ref_center, dim = 1).repeat(1, 1)))

        ##################

        other_cam_j = torch.squeeze((torch.matmul(rot_y_j, (rot_view_j @ torch.unsqueeze(torch.tensor([0,0,0]), 1).double()) - torch.unsqueeze(sync_center_j, dim = 1).repeat(1, 1)) + torch.unsqueeze(ref_center, dim = 1).repeat(1, 1)))

        view = view_dict[j]
        ref = view_dict[0]

        kp_match = {}

        for k in range(view_sync[j].shape[0]):
            
            ref_transform = torch.tensor(ref[list(ref.keys())[ref_sync[j][k]]])
            sync_transform = torch.tensor(view[list(view.keys())[view_sync[j][k]]])
            
            #print(ref_transform.shape, sync_transform.shape, " BEFORRE")

            ref_transform = torch.transpose(ref_transform, 0, 1)
            sync_transform = torch.transpose(sync_transform, 0, 1)

            #print(torch.unsqueeze(ref_transform[0, :], dim = 0), torch.unsqueeze(ref_transform[1, :], dim = 0), torch.ones(1, ref_transform.shape[1]))
            ref_transform = torch.cat((torch.unsqueeze(ref_transform[0, :], dim = 0), torch.unsqueeze(ref_transform[1, :], dim = 0), torch.ones(1, ref_transform.shape[1])))
            sync_transform = torch.cat((torch.unsqueeze(sync_transform[0, :], dim = 0), torch.unsqueeze(sync_transform[1, :], dim = 0), torch.ones(1, sync_transform.shape[1])))
            
            #print(ref_transform.shape, sync_transform.shape, " after")
            torch.transpose(ref_transform - torch.unsqueeze(sync_center_j, dim = 1).repeat(1, ref_transform.shape[1]), 0, 1)

            #print((rot_y_i @ (rot_view_i @ ref_transform - torch.unsqueeze(sync_center_j, dim = 1).repeat(1, ref_transform.shape[1]))).shape, " SHAPE")
            #print(torch.transpose(torch.unsqueeze(ref_center, dim = 0).repeat(ref_transform.shape[1], 1), 0, 1).shape, " QQWPQQWEDADSA")
            ############
            ref_transform = rot_y_i @ (rot_view_i @ ref_transform - torch.unsqueeze(sync_center_j, dim = 1).repeat(1, ref_transform.shape[1])) + torch.transpose(torch.unsqueeze(ref_center, dim = 0).repeat(ref_transform.shape[1], 1), 0, 1)
            sync_transform = rot_y_j @ (rot_view_j @ sync_transform - torch.unsqueeze(sync_center_i, dim = 1).repeat(1, sync_transform.shape[1])) + torch.transpose(torch.unsqueeze(ref_center, dim = 0).repeat(sync_transform.shape[1], 1), 0, 1)
            ############
            #print(ref_transform.shape, sync_transform.shape, " HIEWQWEQWE")
            #pair_array = closest_points(torch.transpose(ref_transform, 0 , 1), torch.transpose(sync_transform, 0 , 1))
            pair_array = find_match_space(torch.transpose(sync_transform, 0 , 1), torch.transpose(ref_transform, 0 , 1))

            #print(list(ref.keys())[ref_sync[j][k]], list(view.keys())[view_sync[j][k]], " 12333333333333")
            kp_match[list(ref.keys())[ref_sync[j][k]]] = {list(view.keys())[view_sync[j][k]]: pair_array}
            #index = np.argmin(pair_dist, axis = 1)
            #print(pair_dist)
        all_kp_match.append(kp_match)

    return all_kp_match

def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)

def eulerAnglesToRotationMatrix(theta):
     
    R_x = np.array([[1,         0,                  0                   ],
                    [0,         math.cos(theta[0]), -math.sin(theta[0]) ],
                    [0,         math.sin(theta[0]), math.cos(theta[0])  ]
                    ])
 
    R_y = np.array([[math.cos(theta[1]),    0,      math.sin(theta[1])  ],
                    [0,                     1,      0                   ],
                    [-math.sin(theta[1]),   0,      math.cos(theta[1])  ]
                    ])
 
    R_z = np.array([[math.cos(theta[2]),    -math.sin(theta[2]),    0],
                    [math.sin(theta[2]),    math.cos(theta[2]),     0],
                    [0,                     0,                      1]
                    ])
 
    R = np.dot(R_z, np.dot( R_y, R_x ))
 
    return R

def isRotationMatrix(R) :
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype = R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6
 
# Calculates rotation matrix to euler angles
# The result is the same as MATLAB except the order
# of the euler angles ( x and z are swapped ).
def rotationMatrixToEulerAngles(R) :
 
    assert(isRotationMatrix(R))
 
    sy = math.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])
 
    singular = sy < 1e-6
 
    if  not singular :
        x = math.atan2(R[2,1] , R[2,2])
        y = math.atan2(-R[2,0], sy)
        z = math.atan2(R[1,0], R[0,0])
    else :
        x = math.atan2(-R[1,2], R[1,1])
        y = math.atan2(-R[2,0], sy)
        z = 0
 
    return np.array([x, y, z])

def plane_coords(input_dict, img, from_pickle):
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
    
    rot_matrix = basis_change_rotation_matrix(cam_matrix, cam_inv, ankleWorld, normal, img_width, img_height)

    #GRAPHING THE IMAGE VIEW
    
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
            
            ankle_3d = np.squeeze(plane_ray_intersection_np([ppl_ankle_u], [ppl_ankle_v], cam_inv, normal, ankleWorld))
            person_plane = (rot_matrix @ ankle_3d) 
            #ax1.scatter(x=person_plane[0], y=person_plane[1], c='green', s=30)
            
            ankle_x.append(person_plane[0])
            ankle_y.append(person_plane[1])

            return_dict[i].append([person_plane[0], person_plane[1]])

    return return_dict, ankle_x, ankle_y

def central_diff(input, time = 1.0, dilation = 1):
    #input is (dim, batch, length)
    kernal_central = torch.tensor([[[-(1/2)*(1/time), 0, (1/2)*(1/time)]]]).double()
    kernal_forward = torch.tensor([[[-(1/time),(1/time)]]]).double()
    
    c_diff = F.conv1d(input, kernal_central, dilation = dilation)
    f_diff = F.conv1d(input, kernal_forward, dilation = dilation)

    return torch.cat((f_diff[:,:, :dilation], c_diff, f_diff[:,:, -dilation:]), dim=2)

def find_match(point_shift, coords_time,coords_ref, coords_ref_time, k):
    
    #dists = torch.transpose(torch.cdist(torch.unsqueeze(coords_time, dim = 1),torch.unsqueeze(coords_ref_time, dim = 1)), 0, 1).detach().numpy()
    dists = torch.cdist(torch.unsqueeze(coords_time, dim = 1),torch.unsqueeze(coords_ref_time, dim = 1)).detach().numpy()
    #for each point in coord ref, finds a correspondance with point shift cr => ps
    #mindists, argmins = torch.min(dists, axis=0)
    #ps -> cr
    #mindists1, argmins1 = torch.min(dists, axis=1)
    idx = np.argpartition(dists, k, axis=1)[:, :k]
    pair_array = []

    for i in range(idx.shape[0]):
        dist_array = idx[i, :]

        if dist_array.shape[0] > 1:
            bool_arr = (dist_array == dist_array[0])
            result = np.all(bool_arr)

            if result:
                continue
        #print(point_shift[i, :].repeat(k, 1).shape, coords_ref[dist_array, :].shape, " .shape.shape.shape.shape")
        ind_dist = torch.norm(point_shift[i, :].repeat(k, 1) - coords_ref[dist_array, :], dim = 1)     
        ind = ind_dist.argmin()
        
        #print(point_shift[i, :].repeat(k, 1).shape, coords_ref[dist_array, :].shape)
        pair_array.append([int(dist_array[ind].item()), i])

    #print(np.array(pair_array), " np.array(pair_array)")
    return np.array(pair_array)

def find_match_time(coords_time,coords_ref_time, k):
    
    #dists = torch.transpose(torch.cdist(torch.unsqueeze(coords_time, dim = 1),torch.unsqueeze(coords_ref_time, dim = 1)), 0, 1).detach().numpy()
    dists = torch.cdist(torch.unsqueeze(coords_time, dim = 1),torch.unsqueeze(coords_ref_time, dim = 1)).detach().numpy()
    #for each point in coord ref, finds a correspondance with point shift cr => ps
    #mindists, argmins = torch.min(dists, axis=0)
    #ps -> cr
    #mindists1, argmins1 = torch.min(dists, axis=1)
    idx = np.argpartition(dists, k, axis=1)[:, :k]
    pair_array = []

    for i in range(idx.shape[0]):
        dist_array = idx[i, :]

        if dist_array.shape[0] > 1:
            bool_arr = (dist_array == dist_array[0])
            result = np.all(bool_arr)

            if result:
                continue
        #print(point_shift[i, :].repeat(k, 1).shape, coords_ref[dist_array, :].shape, " .shape.shape.shape.shape")
        ind_dist = torch.absolute(coords_time[i].repeat(k) - coords_ref_time[dist_array])     
        ind = ind_dist.argmin()
        
        #print(point_shift[i, :].repeat(k, 1).shape, coords_ref[dist_array, :].shape)
        pair_array.append([i, int(dist_array[ind].item())])

    return np.array(pair_array)
'''
def find_match(point_shift, coords_time,coords_ref, coords_ref_time, k):
    #print(point_shift.shape, coords_ref.shape, "coords_refcoords_refcoords_refcoords_refcoords_ref")
    #print(coords_time.shape, " coords_time,coords_time,coords_time,coords_time,coords_time,")
    dists = torch.cdist(point_shift,coords_ref).detach().numpy()

    #for each point in coord ref, finds a correspondance with point shift cr => ps
    #mindists, argmins = torch.min(dists, axis=0)
    #ps -> cr
    #mindists1, argmins1 = torch.min(dists, axis=1)
    idx = np.argpartition(dists, k, axis=0)[:, :k]
    
    pair_array = []

    for i in range(idx.shape[0]):
        dist_array = idx[i, :]

        if dist_array.shape[0] > 1:
            print(dist_array, " dist array")
            bool_arr = (dist_array == dist_array[0])
            #print(bool_arr, " bool_arr")
            result = np.all(bool_arr)

            if result:
                continue

        ind = np.abs(coords_ref_time[dist_array].detach().numpy()
 - coords_time[i].detach().numpy()
).argmin()
        pair_array.append([i, int(dist_array[ind].item())])
    
    print(pair_array)
    #print(np.array(pair_array).shape, " PAIR ARRAY")

    return np.array(pair_array)
'''

def chamfer_distance_time(x, y, x_time, y_time, x_truth_time, y_truth_time, metric='l2', direction='bi'):
    """Chamfer distance between two point clouds
    Parameters
    ----------
    x: numpy array [n_points_x, n_dims]
        first point cloud
    y: numpy array [n_points_y, n_dims]
        second point cloud
    metric: string or callable, default ‘l2’
        metric to use for distance computation. Any metric from scikit-learn or scipy.spatial.distance can be used.
    direction: str
        direction of Chamfer distance.
            'y_to_x':  computes average minimal distance from every point in y to x
            'x_to_y':  computes average minimal distance from every point in x to y
            'bi': compute both
    Returns
    -------
    chamfer_dist: float
        computed bidirectional Chamfer distance:
            sum_{x_i \in x}{\min_{y_j \in y}{||x_i-y_j||**2}} + sum_{y_j \in y}{\min_{x_i \in x}{||x_i-y_j||**2}}
    """
    # INCOPORATE TIME !!!
    if direction=='y_to_x':
        x_nn = NearestNeighbors(n_neighbors=1, leaf_size=1, algorithm='kd_tree', metric=metric).fit(x)
        min_y_to_x, y_x_ind = x_nn.kneighbors(y)
        chamfer_dist = np.mean(min_y_to_x)
    elif direction=='x_to_y':
        y_nn = NearestNeighbors(n_neighbors=1, leaf_size=1, algorithm='kd_tree', metric=metric).fit(y)
        min_x_to_y, x_y_ind = y_nn.kneighbors(x)
        chamfer_dist = np.mean(min_x_to_y)
    elif direction=='bi':
        x_nn = NearestNeighbors(n_neighbors=1, leaf_size=1, algorithm='kd_tree', metric=metric).fit(x)
        min_y_to_x, y_x_ind = x_nn.kneighbors(y)
        y_nn = NearestNeighbors(n_neighbors=1, leaf_size=1, algorithm='kd_tree', metric=metric).fit(y)
        min_x_to_y, x_y_ind = y_nn.kneighbors(x)

        #del x_nn
        #del y_nn
        #print(" HELOOOOOOOOOOOOOOOOOOOOOOOOOOOOOO 656")

        y_x_ind = np.squeeze(y_x_ind)
        x_y_ind = np.squeeze(x_y_ind)

        #print(np.mean(y_x_ind), np.mean(x_y_ind), " INDEXX")
        #print(x_y_ind)
        
        time_diff_x_y = np.squeeze(1 + np.absolute(y_time[x_y_ind] - np.array(x_truth_time)))
        #print(" HELLO 661 ?????")
        time_diff_y_x = np.squeeze(1 + np.absolute(np.array(y_truth_time) - x_time[y_x_ind]))

        print(np.mean(time_diff_x_y), np.mean(time_diff_y_x), " INDEXX")
        #print(" HELOOOOOOOOOOOOOOOOOOOOOOOOOOOOOO 659")
        #print(time_diff_x_y, " x y")
        #print(time_diff_y_x, " y _ x")
        '''
        del y_x_ind
        del x_y_ind
        del x_time
        del y_time
        del x_truth_time
        del y_truth_time
        '''

        min_y_to_x = np.squeeze(min_y_to_x)
        min_x_to_y = np.squeeze(min_x_to_y)

        #print(min_y_to_x)
        x_dist = np.mean(time_diff_y_x*min_y_to_x)
        #print("FINSIEHD X DIST")
        '''
        del time_diff_y_x
        del min_y_to_x
        '''
        y_dist = np.mean(time_diff_x_y*min_x_to_y)
        '''
        del time_diff_x_y
        del min_x_to_y
        '''
        #chamfer_dist = np.mean(np.multiply(time_diff_y_x, min_y_to_x)) + np.mean(np.multiply(time_diff_x_y, min_x_to_y))
        chamfer_dist = x_dist + y_dist
    else:
        raise ValueError("Invalid direction type. Supported types: \'y_x\', \'x_y\', \'bi\'")
        
    return chamfer_dist

def chamfer_distance(x, y, metric='l2', direction='bi'):
    """Chamfer distance between two point clouds
    Parameters
    ----------
    x: numpy array [n_points_x, n_dims]
        first point cloud
    y: numpy array [n_points_y, n_dims]
        second point cloud
    metric: string or callable, default ‘l2’
        metric to use for distance computation. Any metric from scikit-learn or scipy.spatial.distance can be used.
    direction: str
        direction of Chamfer distance.
            'y_to_x':  computes average minimal distance from every point in y to x
            'x_to_y':  computes average minimal distance from every point in x to y
            'bi': compute both
    Returns
    -------
    chamfer_dist: float
        computed bidirectional Chamfer distance:
            sum_{x_i \in x}{\min_{y_j \in y}{||x_i-y_j||**2}} + sum_{y_j \in y}{\min_{x_i \in x}{||x_i-y_j||**2}}
    """
    # INCOPORATE TIME !!!
    if direction=='y_to_x':
        x_nn = NearestNeighbors(n_neighbors=1, leaf_size=1, algorithm='kd_tree', metric=metric).fit(x)
        min_y_to_x = x_nn.kneighbors(y)[0]
        chamfer_dist = np.mean(min_y_to_x)
    elif direction=='x_to_y':
        y_nn = NearestNeighbors(n_neighbors=1, leaf_size=1, algorithm='kd_tree', metric=metric).fit(y)
        min_x_to_y = y_nn.kneighbors(x)[0]
        chamfer_dist = np.mean(min_x_to_y)
    elif direction=='bi':
        x_nn = NearestNeighbors(n_neighbors=1, leaf_size=1, algorithm='kd_tree', metric=metric).fit(x)
        min_y_to_x = x_nn.kneighbors(y)[0]
        y_nn = NearestNeighbors(n_neighbors=1, leaf_size=1, algorithm='kd_tree', metric=metric).fit(y)
        min_x_to_y = y_nn.kneighbors(x)[0]
        chamfer_dist = np.mean(min_y_to_x) + np.mean(min_x_to_y)
    else:
        raise ValueError("Invalid direction type. Supported types: \'y_x\', \'x_y\', \'bi\'")
        
    return chamfer_dist

def hungarian_match_distance(x, y):
    """Chamfer distance between two point clouds
    Parameters
    ----------
    x: numpy array [n_points_x, n_dims]
        first point cloud
    y: numpy array [n_points_y, n_dims]
        second point cloud
    metric: string or callable, default ‘l2’
        metric to use for distance computation. Any metric from scikit-learn or scipy.spatial.distance can be used.
    direction: str
        direction of Chamfer distance.
            'y_to_x':  computes average minimal distance from every point in y to x
            'x_to_y':  computes average minimal distance from every point in x to y
            'bi': compute both
    Returns
    -------
    chamfer_dist: float
        computed bidirectional Chamfer distance:
            sum_{x_i \in x}{\min_{y_j \in y}{||x_i-y_j||**2}} + sum_{y_j \in y}{\min_{x_i \in x}{||x_i-y_j||**2}}
    """

    A = {}

    for subarray in x:
        key = subarray[0]
        value = subarray[1]
        
        # Check if the key is already in the dictionary
        if key in A:
            # If it is, append the value to the existing list
            A[key].append(value)
        else:
            # If not, create a new list with the value
            A[key] = [value]
    
    B = {}

    for subarray in y:
        key = subarray[0]
        value = subarray[1]
        
        # Check if the key is already in the dictionary
        if key in B:
            # If it is, append the value to the existing list
            B[key].append(value)
        else:
            # If not, create a new list with the value
            B[key] = [value]
    
    A, B = merge_dicts_with_defaults(A, B)
    A_time = list(A.keys())
    B_time = list(B.keys())

    common_time = list(set(A_time) & set(B_time))
    # INCOPORATE TIME !!!
    error_array = []
    for i in common_time:
        #print(A[i], B[i])
        matched_A, matched_B, row_ind, col_ind = hungarian_assignment(np.array(A[i]), np.array(B[i]))
        error = np.mean(np.absolute(np.array(matched_A) - np.mean(matched_B)))
        error_array.append(error)

    return np.mean(error_array), A, B

def dictionary_to_array(my_dict):
    result_array = []
    for key, value_list in my_dict.items():
        for value in value_list:
            result_array.append([key, value])
    return result_array

def merge_dicts_with_defaults(dict1, dict2):
    # Get the minimum and maximum keys from both dictionaries
    min_key_dict1 = min(dict1.keys())
    max_key_dict1 = max(dict1.keys())
    min_key_dict2 = min(dict2.keys()) 
    max_key_dict2 = max(dict2.keys())

    # Determine the range of keys to consider
    min_key = int(min(min_key_dict1, min_key_dict2))
    max_key = int(max(max_key_dict1, max_key_dict2))
    #print(min_key, max_key, " THE KEYS")

    # Initialize new dictionaries
    result_dict1 = {}
    result_dict2 = {}

    # Iterate through the range of keys
    last_value1 = None
    last_value2 = None

    for key in range(min_key, max_key + 1):
        if key in dict1:
            result_dict1[key] = dict1[key]
            last_value1 = dict1[key]
        else:
            if key < min_key_dict1 or key > max_key_dict1:
                result_dict1[key] = dict1[min_key_dict1] if key < min_key_dict1 else dict1[max_key_dict1]
            else:
                result_dict1[key] = last_value1

        if key in dict2:
            result_dict2[key] = dict2[key]
            last_value2 = dict2[key]
        else:
            if key < min_key_dict2 or key > max_key_dict2:
                result_dict2[key] = dict2[min_key_dict2] if key < min_key_dict2 else dict2[max_key_dict2]
            else:
                result_dict2[key] = last_value2

    return result_dict1, result_dict2

def find_match_space(point_shift, coords_ref):
    #dists = torch.transpose(torch.cdist(torch.unsqueeze(coords_time, dim = 1),torch.unsqueeze(coords_ref_time, dim = 1)), 0, 1).detach().numpy()
    dists = torch.cdist(torch.unsqueeze(point_shift, dim = 0),torch.unsqueeze(coords_ref, dim = 0)).detach()
    #for each point in coord ref, finds a correspondance with point shift cr => ps
    #mindists, argmins = torch.min(dists, axis=0)
    #ps -> cr
    #mindists1, argmins1 = torch.min(dists, axis=1)
    #idx = np.argpartition(dists, k, axis=1)[:, :k]
    mindists, idx = torch.min(dists, axis=1)

    if idx.shape[1] > 1:
        mindists = torch.squeeze(mindists)
        idx = torch.squeeze(idx)
    pair_array = []

    pair_dict = {}

    j = 0

    for i in idx:
        if i.item() in pair_dict.keys():
            
            if mindists[j] > pair_dict[i.item()][1]: 
                continue
        
        pair_dict[i.item()] = [j, mindists[j].item()]
        j = j + 1

    pair_array = []
    for i in pair_dict.keys():
        pair_array.append([i, pair_dict[i][0]])

    return np.array(pair_array)

def closest_points(coords1, coords2):

    dists = torch.cdist(torch.unsqueeze(coords1, dim = 0),torch.unsqueeze(coords2, dim = 0)).detach()
    #pairs = np.argmin(dists)
    #for each point in coord ref, finds a correspondance with point shift cr => ps
    mindists, argmins = torch.min(dists, axis=0)
    #ps -> cr
    #mindists1, argmins1 = torch.min(dists, axis=1)

    pair_dict = {}

    j = 0

    for i in argmins:

        if i.item() in pair_dict.keys():
            
            if mindists[j] > pair_dict[i.item()][1]: 
                continue
        
        pair_dict[i.item()] = [j, mindists[j].item()]
        j = j + 1

    pair_array = []
    for i in pair_dict.keys():
        pair_array.append([i, pair_dict[i][0]])

    return np.array(pair_array)

def intr_extr_transform(coords, cam_matrix, extrinsic):

    extrinsic_inv = extrinsic[:, :3]
    translation_inv = extrinsic[:, 3]
    #print(extrinsic, " extrinsic")
    transformed_coords = []
    coords_3d = []
    for c in coords:
        cam_coord = extrinsic_inv @ (c + translation_inv)
        #cam_coord = (extrinsic_inv @ c) + translation_inv
        coords_3d.append(cam_coord)
        image_coord = cam_matrix @ cam_coord
        #print(image_coord, " before divide")
        image_coord = image_coord[:2]/image_coord[2]
        transformed_coords.append(image_coord)
        #print(image_coord, " image_coord")
    return np.array(transformed_coords), np.array(coords_3d)

def frame_off(ground, pred):

    frame_dict = {}

    diff = np.abs(np.array(ground) - np.array(pred))
    bins = set(diff)
    
    for i in bins:
        frame_dict[i] = list(diff).count(i)/len(list(diff))
    return frame_dict

def find_nearest_icp(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    #print(array - value)
    #print(idx)
    return idx

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.linalg.norm(array - np.repeat(np.expand_dims(value, axis = 1), array.shape[1], axis=1), axis = 0)).argmin()
    return idx

def rotate(p, origin=(0, 0), degrees=0.0):
    angle = np.deg2rad(degrees)
    R = np.array([[np.cos(angle), -np.sin(angle)],
                  [np.sin(angle),  np.cos(angle)]])
    o = np.atleast_2d(origin)
    p = np.atleast_2d(p)
    return np.squeeze((R @ (p.T-o.T) + o.T).T), R

def reflect(p, origin=(0, 0)):
    R = np.array([[-1, 0],
                  [0,  -1]])
    o = np.atleast_2d(origin)
    p = np.atleast_2d(p)
    return np.squeeze((R @ (p.T-o.T) + o.T).T)
    
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

def perspective_transformation_torch(cam_matrix, coordinate):
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
    coordinate_2d = torch.matmul(cam_matrix, coordinate)
    return coordinate_2d[:2]/(coordinate_2d[2] + eps)
        
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

def get_ankles_heads_frame(datastore, image_index, conf):
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
    frame = []

    frame_dict = {}
    pose_dict = {}

    all_points = []
    
    first_frame = 0#int(datastore.getitem(0)["left_ankle"][3].split('/')[-1].split('.')[0])
    for ppl in image_index:
        
        left_ankle = datastore.getitem(ppl)["left_ankle"]
        right_ankle = datastore.getitem(ppl)["right_ankle"]

        ankle_left_conf.append(datastore.getitem(ppl)["left_ankle"][2])
        ankle_right_conf.append(datastore.getitem(ppl)["right_ankle"][2])

        #ankle_x, ankle_y = determine_foot(right_ankle,left_ankle, hgt_threshold=8.0, wide_threshold=10.0)
        ankle_x, ankle_y = determine_foot(right_ankle,left_ankle, hgt_threshold=4.0, wide_threshold=4.0)
            
        head_x = (datastore.getitem(ppl)["middle"][0])
        head_y = (datastore.getitem(ppl)["middle"][1])

        head_conf.append(datastore.getitem(ppl)["left_ankle"][2])

        ppl_ankle_u.append(ankle_x)
        ppl_ankle_v.append(ankle_y)

        ppl_head_u.append(head_x)
        ppl_head_v.append(head_y)

        frame.append(datastore.getitem(ppl)["middle"][3])

        #frame_name = datastore.getitem(ppl)["middle"][3]
        frame_name = int(datastore.getitem(ppl)["left_ankle"][3].split('/')[-1].split('.')[0]) - first_frame
        
        if frame_name not in frame_dict:
            frame_dict[frame_name] = []
            pose_dict[frame_name] = []

            if datastore.getitem(ppl)["left_ankle"][2] < conf or datastore.getitem(ppl)["right_ankle"][2] < conf:
                continue
            frame_dict[frame_name].append([ankle_x, ankle_y, head_x, head_y])

            kp_array = []
            pose_array = []
            for keypoint in datastore.getitem(ppl).keys():
    
                if keypoint ==  'box1' or  keypoint == 'box2':
                    continue
                kp_array.append([datastore.getitem(ppl)[keypoint][0], datastore.getitem(ppl)[keypoint][1]])

                pose_array.append([datastore.getitem(ppl)[keypoint][0], datastore.getitem(ppl)[keypoint][1]])
            pose_dict[frame_name].append(pose_array)
        else:
            if datastore.getitem(ppl)["left_ankle"][2] < conf or datastore.getitem(ppl)["right_ankle"][2] < conf:
                continue
            frame_dict[frame_name].append([ankle_x, ankle_y, head_x, head_y])

            kp_array = []
            pose_array = []    
            for keypoint in datastore.getitem(ppl).keys():
    
                if keypoint ==  'box1' or  keypoint == 'box2':
                    continue
                kp_array.append([datastore.getitem(ppl)[keypoint][0], datastore.getitem(ppl)[keypoint][1]])

                pose_array.append([datastore.getitem(ppl)[keypoint][0], datastore.getitem(ppl)[keypoint][1]])
            pose_dict[frame_name].append(pose_array)
        #kp_array.append([ankle_x, ankle_y])
        pose_dict = {k:v for k,v in pose_dict.items() if v}
        all_points.append(kp_array)
    return frame_dict, pose_dict, all_points, np.array(ppl_ankle_u), np.array(ppl_ankle_v), np.array(ppl_head_u), np.array(ppl_head_v)

def get_ankles_heads_frame_middle(datastore, image_index, conf):
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
    frame = []

    frame_dict = {}
    pose_dict = {}

    all_points = []
    
    first_frame = 0#int(datastore.getitem(0)["left_ankle"][3].split('/')[-1].split('.')[0])
    for ppl in image_index:
        
        left_ankle = datastore.getitem(ppl)["left_ankle"]
        right_ankle = datastore.getitem(ppl)["right_ankle"]

        ankle_left_conf.append(datastore.getitem(ppl)["left_ankle"][2])
        ankle_right_conf.append(datastore.getitem(ppl)["right_ankle"][2])

        #ankle_x, ankle_y = determine_foot(right_ankle,left_ankle, hgt_threshold=8.0, wide_threshold=10.0)
        ankle_x, ankle_y = determine_foot(right_ankle,left_ankle, hgt_threshold=4.0, wide_threshold=4.0)

        ####################
        #middle of the hip
        ankle_x = (datastore.getitem(ppl)["left_hip"][0] + datastore.getitem(ppl)["right_hip"][0])/2.0
        ###################
            
        head_x = (datastore.getitem(ppl)["middle"][0])
        head_y = (datastore.getitem(ppl)["middle"][1])

        head_conf.append(datastore.getitem(ppl)["left_ankle"][2])

        frame.append(datastore.getitem(ppl)["middle"][3])

        #frame_name = datastore.getitem(ppl)["middle"][3]
        frame_name = int(datastore.getitem(ppl)["left_ankle"][3].split('/')[-1].split('.')[0]) - first_frame

        if frame_name not in frame_dict:
            frame_dict[frame_name] = []
            pose_dict[frame_name] = {'pose': [], 'path': datastore.getitem(ppl)["left_ankle"][3].split('/')[-1].split('.')[0]}#[]

            if datastore.getitem(ppl)["left_ankle"][2] < conf or datastore.getitem(ppl)["right_ankle"][2] < conf:
                continue
            frame_dict[frame_name].append([ankle_x, ankle_y, head_x, head_y])
            ppl_ankle_u.append(ankle_x)
            ppl_ankle_v.append(ankle_y)

            ppl_head_u.append(head_x)
            ppl_head_v.append(head_y)

            kp_array = []
            pose_array = []
            for keypoint in datastore.getitem(ppl).keys():
                #print("****************************")
                #print(datastore.getitem(ppl))
                if keypoint ==  'box1' or  keypoint == 'box2':
                    continue

                kp_array.append([datastore.getitem(ppl)[keypoint][0], datastore.getitem(ppl)[keypoint][1]])
                pose_array.append([datastore.getitem(ppl)[keypoint][0], datastore.getitem(ppl)[keypoint][1]])
            pose_dict[frame_name]['pose'].append(pose_array)
        else:
            if datastore.getitem(ppl)["left_ankle"][2] < conf or datastore.getitem(ppl)["right_ankle"][2] < conf:
                continue
            ppl_ankle_u.append(ankle_x)
            ppl_ankle_v.append(ankle_y)

            ppl_head_u.append(head_x)
            ppl_head_v.append(head_y)
            frame_dict[frame_name].append([ankle_x, ankle_y, head_x, head_y])

            kp_array = []
            pose_array = []
            for keypoint in datastore.getitem(ppl).keys():
    
                if keypoint ==  'box1' or  keypoint == 'box2':
                    continue
                kp_array.append([datastore.getitem(ppl)[keypoint][0], datastore.getitem(ppl)[keypoint][1]])
                pose_array.append([datastore.getitem(ppl)[keypoint][0], datastore.getitem(ppl)[keypoint][1]])
            
            pose_dict[frame_name]['pose'].append(pose_array)
        #kp_array.append([ankle_x, ankle_y])

        all_points.append(kp_array)

    #pose_dict = {k:v for k,v in pose_dict.items() if v}

    key_list = list(pose_dict.keys())
    for k in key_list:
        if len(pose_dict[k]['pose']) == 0:
            #pose_dict.pop(k)
            del pose_dict[k]
    frame_dict = {k:v for k,v in frame_dict.items() if v}

    return frame_dict, pose_dict, all_points, np.array(ppl_ankle_u), np.array(ppl_ankle_v), np.array(ppl_head_u), np.array(ppl_head_v)

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

def findAnglesBetweenTwoVectors(v1s, v2s):
    dot_v1_v2 = np.einsum('ij,ij->i', v1s, v2s)
    dot_v1_v1 = np.einsum('ij,ij->i', v1s, v1s)
    dot_v2_v2 = np.einsum('ij,ij->i', v2s, v2s)

    return np.arccos(dot_v1_v2/(np.sqrt(dot_v1_v1)*np.sqrt(dot_v2_v2)))

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

def plane_ray_intersection_torch(x_imcoord, y_imcoord, cam_inv, normal, init_point):
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
    point_2d = torch.stack((x_imcoord, y_imcoord, torch.ones(x_imcoord.shape[0])))
    
    ray = cam_inv @ point_2d
    #print(torch.unsqueeze(normal, dim = 0).shape, ray.shape, "heqwqewqeweqweqw qqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqq")
    normal_dot_ray = torch.matmul(torch.unsqueeze(normal, dim = 0), ray) + 1e-15
    
    scale = abs(torch.div(torch.dot(normal, init_point).repeat(x_imcoord.shape[0]), normal_dot_ray))
    return scale*ray

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

def project_point_horiz_bottom_torch(cam_matrix, cam_inv, p_plane, init_world, normal, img_width, img_height):
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
    plane_horiz_world = torch.squeeze(plane_ray_intersection_torch(torch.tensor([img_width/2.0 + 1]).double(), torch.tensor([img_height]).double(), cam_inv, normal, init_world))

    up_vector = plane_horiz_world - init_world
    
    up_vector = up_vector/torch.norm(up_vector)

    v = up_vector
    #u = np.cross(v, normal)
    u = torch.cross(normal, v)
    
    #normalize 
    v = v/torch.norm(v)
    u = u/torch.norm(u)
            
    grid_location_3d_ij = init_world + p_plane[0]*v + p_plane[1]*u
    
    grid_location_2d_ij = perspective_transformation_torch(cam_matrix, grid_location_3d_ij)
    
    return grid_location_2d_ij, grid_location_3d_ij

def basis_change_rotation_matrix_torch(cam_matrix, cam_inv, init_point, normal, img_width, img_height):
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
    plane_world = torch.squeeze(plane_ray_intersection_torch(torch.tensor([img_width/2.0]).double(), torch.tensor([img_height]).double(), cam_inv, normal, init_point))

    p00, p00_3d = project_point_horiz_bottom_torch(cam_matrix, cam_inv, torch.tensor([0,0]).double(), plane_world, normal, img_width, img_height)
    p01, p01_3d = project_point_horiz_bottom_torch(cam_matrix, cam_inv, torch.tensor([0,1]).double(), plane_world, normal, img_width, img_height)
    p10, p10_3d = project_point_horiz_bottom_torch(cam_matrix, cam_inv, torch.tensor([1,0]).double(), plane_world, normal, img_width, img_height)
    p11, p11_3d = project_point_horiz_bottom_torch(cam_matrix, cam_inv, torch.tensor([1,1]).double(), plane_world, normal, img_width, img_height)
    
    new_basis0 = p01_3d - p00_3d
    new_basis1 = p10_3d - p00_3d
    
    new_basis0 = new_basis0/torch.norm(new_basis0)
    new_basis1 = new_basis1/torch.norm(new_basis1)
    
    old_basis0 = torch.tensor([1, 0, 0]).double()
    old_basis1 = torch.tensor([0, 1, 0]).double()
    old_basis2 = torch.tensor([0, 0, 1]).double()
    
    #C = torch.zeros([3,3], dtype = float)
    C0 = torch.stack([torch.dot(new_basis0, old_basis0), torch.dot(new_basis0, old_basis1), torch.dot(new_basis0, old_basis2)])
    C1 = torch.stack([torch.dot(new_basis1, old_basis0), torch.dot(new_basis1, old_basis1), torch.dot(new_basis1, old_basis2)])
    C2 = torch.stack([torch.dot(normal, old_basis0), torch.dot(normal, old_basis1), torch.dot(normal, old_basis2)])
    
    C = torch.stack([C0, C1, C2])
    z_rotation = np.zeros((3,3))
    z_rotation[0] = [np.cos(np.pi/2.0), -1*np.sin(np.pi/2.0), 0]
    z_rotation[1] = [np.sin(np.pi/2.0), np.cos(np.pi/2.0), 0]
    z_rotation[2] = [0,0,1]
    
    flip = np.zeros((3,3))
    flip[0] = [-1, 0, 0]
    flip[1] = [0, 1, 0]
    flip[2] = [0, 0, 1,]
    
    return torch.matmul(torch.from_numpy(flip), torch.matmul(torch.from_numpy(z_rotation), C))

def bbox_size(pose):
    
    bbox = pose['bbox']

    #shoulder_left = np.array(pose['left_shoulder'])[:2]
    #shoulder_right = np.array(pose['right_shoulder'])[:2]
    #bbox_width = np.linalg.norm(shoulder_left - shoulder_right)
    bbox_width = np.absolute(bbox[0] - bbox[2])
    bbox_height = np.absolute(bbox[1] - bbox[3])
    return bbox_width, bbox_height

def determine_foot_bbox(rfoot, lfoot, pose, hgt_threshold=0.2, wide_threshold=0.2):
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

    bbox_width, bbox_height = bbox_size(pose)
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
    print("selecting indices")

    filtered_detections = []

    head_ankle_array = []
    
    print(datastore.__len__(), " total number of poses")
    
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

    print(filtered_datastore.__len__(), " Number of ankle for calibration")
    return filtered_datastore