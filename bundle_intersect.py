import torch
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.image as mpimg

def plane_intersect(point_2d, init_point, cam_inv, extrinsic):

    #print(point_2d.shape, init_point.shape, cam_inv.shape, extrinsic.shape, " !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    #print(point_2d.shape, " point_2d shapeeee")
    ray_point = (extrinsic @ cam_inv @ point_2d)[:3, :].double()
    #print(init_point.shape)
    #print(ray_point.shape, " THE RAY POINT")
    ray = -1*(ray_point - torch.unsqueeze(init_point, dim = 1).repeat((1, point_2d.shape[1])).double())

    ray[2, :] = -1*ray[2, :]
    #print(ray.shape, " RAYYY")
    normal = torch.tensor([0,0,1]).double()

    normal_dot_ray = torch.matmul(torch.unsqueeze(normal, dim = 0), ray) + 1e-15
    
    scale = abs(torch.div(torch.dot(normal, init_point).repeat(point_2d.shape[1]), normal_dot_ray))

    #print(scale.shape, ray.shape, init_point.shape, " INIT POINTTTTT")
    return scale.repeat((3,1))*ray + torch.unsqueeze(init_point[:3], dim = 1).repeat((1,point_2d.shape[1]))

def bundle_intersect(pose_ref, pose_sync, ref_pos, view_pos, cam_inv_ref, cam_inv_sync, ref_extrinsic, sync_extrinsic):
    '''
    print(sync_extrinsic.shape, cam_inv_sync.shape, pose_sync.shape, " MADTRIXXXX SHAEeeeeee")
    print(pose_ref)
    print("*****************pose ref****************************")
    print(pose_sync)
    print("******************pose sync***************************")
    '''
    point_ref = ref_extrinsic @ cam_inv_ref @ pose_ref
    point_sync = sync_extrinsic @ cam_inv_sync @ pose_sync
    '''
    print("****************ref*****************************")
    print(point_ref)
    print("*******************sync**************************")
    print(point_sync)
    '''
    #print(ray_ref, ray_sync, " RAY SYNCCCCC")

    #print(ref_pos, "ref_pos")
    #print(view_pos, " ray ref")
    '''
    ray_ref = (ray_ref[0:3, :] - torch.unsqueeze(ref_pos, dim = 1).repeat((1, ray_ref.shape[1])).double())
    ray_sync = (ray_sync[0:3, :] - torch.unsqueeze(view_pos, dim = 1).repeat((1, ray_sync.shape[1])).double())
    '''
    ray_ref = -1*(point_ref[0:3, :] - torch.unsqueeze(ref_pos, dim = 1).repeat((1, point_ref.shape[1])).double())
    ray_sync = -1*(point_sync[0:3, :] - torch.unsqueeze(view_pos, dim = 1).repeat((1, point_sync.shape[1])).double())
    
    ray_ref[2, :] = -1*ray_ref[2, :]
    ray_sync[2, :] = -1*ray_sync[2, :]

    point_ref = ray_ref + torch.unsqueeze(ref_pos, dim = 1).repeat((1, point_ref.shape[1])).double()
    point_sync = ray_sync + torch.unsqueeze(view_pos, dim = 1).repeat((1, point_sync.shape[1])).double()
    
    
    #print(ray_ref.shape, " RAY REF")
    #print(ray_sync.shape, " RAY SYNC")
    
    ###########
    '''
    view_pos = sync_extrinsic['cam_position']
    ref_pos = ref_extrinsic['cam_position']
    '''
    '''
    ref_pos = torch.tensor([0,0,0]).double()
    view_pos  = sync_extrinsic @ torch.tensor([0,0,0, 1]).double()

    view_pos = view_pos[0:3]
    '''
    #print(view_pos.shape, " sjaeeehapeee")
    #print(torch.transpose(torch.unsqueeze(ref_pos, dim = 1).repeat(1, ray_ref.shape[1]), 0, 1).shape, torch.transpose(ray_ref, 0, 1).shape, torch.transpose(torch.unsqueeze(view_pos, dim = 1).repeat(1, ray_sync.shape[1]), 0, 1).shape, torch.transpose(ray_sync, 0, 1).shape, " CLOSET POINTTTT")
    #point_ref_3d, point_sync_3d, dist = closestDistanceBetweenLines(torch.transpose(torch.unsqueeze(ref_pos, dim = 1).repeat(1, ray_ref.shape[1]), 0, 1), torch.transpose(ray_ref, 0, 1), torch.transpose(torch.unsqueeze(view_pos, dim = 1).repeat(1, ray_sync.shape[1]), 0, 1), torch.transpose(ray_sync, 0, 1))
    point_ref_3d, point_sync_3d, dist = closestDistanceBetweenLines(torch.transpose(torch.unsqueeze(ref_pos, dim = 1).repeat(1, ray_ref.shape[1]), 0, 1), torch.transpose(point_ref, 0, 1), torch.transpose(torch.unsqueeze(view_pos, dim = 1).repeat(1, ray_sync.shape[1]), 0, 1), torch.transpose(point_sync, 0, 1))
    #print(ref_pos.shape, ray_ref.shape, point_ref.shape, "HELOOASDASD")
    #point_ref_3d, point_sync_3d, dist = intersect_lstq(torch.transpose(torch.unsqueeze(ref_pos, dim = 1).repeat(1, ray_ref.shape[1]), 0, 1), torch.transpose(ray_ref, 0, 1), torch.transpose(torch.unsqueeze(view_pos, dim = 1).repeat(1, ray_sync.shape[1]), 0, 1), torch.transpose(ray_sync, 0, 1))
    #print(point_ref_3d.shape, " int")
    #point_ref_3d, point_sync_3d, dist = closestDistanceBetweenLines(torch.transpose(torch.unsqueeze(ref_pos, dim = 1).repeat(1, ray_ref.shape[1]), 0, 1), torch.transpose(ray_ref, 0, 1), torch.transpose(torch.unsqueeze(view_pos, dim = 1).repeat(1, ray_sync.shape[1]), 0, 1), torch.transpose(ray_sync, 0, 1))
    #print(point_ref_3d.shape, " close")
    '''
    normal = torch.tensor([0,0,1]).double()

    normal_dot_ray = torch.matmul(torch.unsqueeze(normal, dim = 0), ray_ref) + 1e-15
    
    scale = abs(torch.div(torch.dot(normal, ref_pos).repeat(ray_ref.shape[1]), normal_dot_ray))

    #print(scale.shape, ray.shape, init_point.shape, " INIT POINTTTTT")
    pose_ref_3d = scale.repeat((3,1))*ray_ref + torch.unsqueeze(ref_pos[:3], dim = 1).repeat((1,ray_ref.shape[1]))
    ############################################
    normal = torch.tensor([0,0,1]).double()

    normal_dot_ray = torch.matmul(torch.unsqueeze(normal, dim = 0), ray_sync) + 1e-15
    
    scale = abs(torch.div(torch.dot(normal, view_pos).repeat(ray_sync.shape[1]), normal_dot_ray))

    #print(scale.shape, ray.shape, init_point.shape, " INIT POINTTTTT")
    pose_sync_3d = scale.repeat((3,1))*ray_sync + torch.unsqueeze(view_pos[:3], dim = 1).repeat((1,ray_sync.shape[1]))
    '''
    ##############################################
    #point_ref_3d, point_sync_3d, dist = closestDistanceBetweenLines(torch.transpose(torch.unsqueeze(ref_pos, dim = 1).repeat(1, ray_ref.shape[1]), 0, 1), torch.transpose(pose_ref_3d, 0, 1), torch.transpose(torch.unsqueeze(view_pos, dim = 1).repeat(1, ray_sync.shape[1]), 0, 1), torch.transpose(pose_sync_3d, 0, 1))
    dist = 0

    '''
    pose_ref_2d = ref_cal['cam_matrix'] @ torch.inverse(rot_ref) @ ((torch.inverse(rot_y_ref) @ ((torch.transpose(pose_sync_3d, 0, 1) - translation_ref.repeat(1, sync_reshape.shape[0]))) + rot_point_ref.repeat(1, sync_reshape.shape[0])))
    pose_sync_2d = sync_cal['cam_matrix'] @ torch.inverse(rot_sync) @ ((torch.inverse(rot_y) @ ((torch.transpose(pose_ref_3d, 0, 1) - translation.repeat(1, ref_reshape.shape[0]))) + rot_point.repeat(1, ref_reshape.shape[0])))
    
    depth_ref = pose_ref_2d[2, :]
    depth_sync = pose_sync_2d[2, :]

    pose_ref_2d = torch.div(pose_ref_2d, torch.stack((pose_ref_2d[2, :],pose_ref_2d[2, :], pose_ref_2d[2, :])))[:2, :]
    pose_sync_2d = torch.div(pose_sync_2d, torch.stack((pose_sync_2d[2, :],pose_sync_2d[2, :], pose_sync_2d[2, :])))[:2, :]
    '''
    #return torch.transpose(pose_ref_3d, 0, 1), torch.transpose(pose_sync_3d, 0, 1), point_ref_3d, point_sync_3d, dist
    #return torch.transpose(point_ref, 0, 1), torch.transpose(point_sync, 0, 1), point_ref_3d, point_sync_3d, dist
    return point_ref_3d, point_sync_3d, dist
    #return torch.transpose(ray_ref, 0, 1), torch.transpose(ray_sync, 0, 1), 0

def intersect_lstq(a0,a1,b0,b1):

    # Calculate denomitator
    A = []
    
    A_a1 = []
    B_b1 = []
    '''
    print(a0.shape, a1.shape, b0.shape, b1.shape, " b111")
    print(a1, " a1")
    print(b1, " b1")
    '''
    for i in range(a1.shape[0]):
        up_pad = torch.zeros(3*(i)).double()
        low_pad = torch.zeros(3*a1.shape[0] - 3*(i + 1)).double()

        zero_column = torch.zeros(3*a1.shape[0])

        a_col = torch.cat((torch.cat((up_pad, a1[i, :])), low_pad))
        b_col = torch.cat((torch.cat((up_pad, b1[i, :])), low_pad))

        A_a1.append(a_col)
        A_a1.append(zero_column)
        B_b1.append(zero_column)
        B_b1.append(b_col)

        A.append(a_col)
        A.append(-1*b_col)
    
    A = torch.transpose(torch.stack(A), 0 , 1)
    #print(A, " A")
    B =  b0.reshape(-1,1) - a0.reshape(-1,1)

    A_a1 = torch.transpose(torch.stack(A_a1), 0 , 1)
    B_b1 = torch.transpose(torch.stack(B_b1), 0 , 1)

    #print(A.shape, B.shape, " A AND B")
    soln = torch.linalg.lstsq(A, B, rcond=None).solution

    #print(soln, " SOLN")
    
    #return A, B, 0
    return torch.reshape(A_a1 @ soln + a0.reshape(-1,1), (-1,3)), torch.reshape(B_b1 @ soln + b0.reshape(-1,1), (-1,3)), 0

def multi_view_intersect(pose_ref, pose_sync, ref_cal, sync_cal, ref_extrinsic, sync_extrinsic, ref_sync, view_sync):
    #print(pose_ref.shape, pose_sync.shape, " SHAPESS")
    #print(ref_sync)
    #print(view_sync)
    if ref_sync is None:
        ref_reshape = pose_ref.reshape(-1, 2)
        sync_reshape = pose_sync.reshape(-1, 2)
    else:

        ref_reshape = pose_ref[ref_sync, :, :].reshape(-1, 2)
        sync_reshape = pose_sync[view_sync, :, :].reshape(-1, 2)

    cam_inv_ref = ref_cal['cam_inv']
    cam_inv_sync = sync_cal['cam_inv']

    rot_ref  = ref_extrinsic['cam_to_plane']
    rot_sync = sync_extrinsic['cam_to_plane']

    rot_y = sync_extrinsic['rotation']
    rot_y_ref = ref_extrinsic['rotation']

    translation = torch.unsqueeze(sync_extrinsic['translation'], dim = 1)
    rot_point = torch.unsqueeze(sync_extrinsic['rot_point'], dim = 1)

    ########
    translation_ref = torch.unsqueeze(ref_extrinsic['translation'], dim = 1)
    rot_point_ref = torch.unsqueeze(ref_extrinsic['rot_point'], dim = 1)

    ray_ref =  (rot_y_ref @ ((rot_ref @ cam_inv_ref @ torch.transpose(torch.stack((ref_reshape[:, 0], ref_reshape[:, 1], torch.ones(ref_reshape.shape[0])), dim = 1), 0, 1).double()) - rot_point_ref.repeat(1, ref_reshape.shape[0])))  + translation_ref.repeat(1, ref_reshape.shape[0]) #world coords without scale
    ray_sync = (rot_y @ ((rot_sync @ cam_inv_sync @ torch.transpose(torch.stack((sync_reshape[:, 0], sync_reshape[:, 1], torch.ones(sync_reshape.shape[0])), dim = 1), 0, 1).double()) - rot_point.repeat(1, sync_reshape.shape[0]))) + translation.repeat(1, sync_reshape.shape[0]) #plane without scale
    
    ###########
    view_pos = sync_extrinsic['cam_position']
    ref_pos = ref_extrinsic['cam_position']
    #ref_pos = torch.tensor([0,0,0]).double()

    pose_ref_3d, pose_sync_3d, dist = closestDistanceBetweenLines(torch.transpose(torch.unsqueeze(ref_pos, dim = 1).repeat(1, ray_ref.shape[1]), 0, 1), torch.transpose(ray_ref, 0, 1), torch.transpose(torch.unsqueeze(view_pos, dim = 1).repeat(1, ray_sync.shape[1]), 0, 1), torch.transpose(ray_sync, 0, 1))
    
    #pose_ref_3d = ray_ref
    #pose_sync_3d = ray_sync
    pose_ref_2d = ref_cal['cam_matrix'] @ torch.inverse(rot_ref) @ ((torch.inverse(rot_y_ref) @ ((torch.transpose(pose_sync_3d, 0, 1) - translation_ref.repeat(1, sync_reshape.shape[0]))) + rot_point_ref.repeat(1, sync_reshape.shape[0])))
    pose_sync_2d = sync_cal['cam_matrix'] @ torch.inverse(rot_sync) @ ((torch.inverse(rot_y) @ ((torch.transpose(pose_ref_3d, 0, 1) - translation.repeat(1, ref_reshape.shape[0]))) + rot_point.repeat(1, ref_reshape.shape[0])))
    
    depth_ref = pose_ref_2d[2, :]
    depth_sync = pose_sync_2d[2, :]

    pose_ref_2d = torch.div(pose_ref_2d, torch.stack((pose_ref_2d[2, :],pose_ref_2d[2, :], pose_ref_2d[2, :])))[:2, :]
    pose_sync_2d = torch.div(pose_sync_2d, torch.stack((pose_sync_2d[2, :],pose_sync_2d[2, :], pose_sync_2d[2, :])))[:2, :]

    #pose_sync_2d = pose_ref_3d
    ###########
    return pose_ref_3d, pose_sync_3d


def closestDistanceBetweenLines(a0,a1,b0,b1):
    
    #Given two lines defined by numpy.array pairs (a0,a1,b0,b1)
    #Return the closest points on each segment and their distance
    
    # Calculate denomitator
    #print(a0.shape,a1.shape,b0.shape,b1.shape, "SDAPJASD")
    A = a1 - a0
    B = b1 - b0
    magA = torch.norm(A, dim = 1)
    magB = torch.norm(B, dim = 1)
    
    _A = torch.div(A, torch.transpose(torch.stack((magA, magA, magA)), 0, 1))
    _B = torch.div(B, torch.transpose(torch.stack((magB, magB, magB)), 0, 1))
    
    cross = torch.cross(_A, _B, dim = 1)
    denom = torch.norm(cross, dim = 1)**2
    
    # Lines criss-cross: Calculate the projected closest points
    t = (b0 - a0)
    #detA = torch.det([t, _B, cross])
    #detB = torch.det([t, _A, cross])

    detA = torch.det(torch.stack((t, _B, cross), dim = 2))
    detB = torch.det(torch.stack((t, _A, cross), dim = 2))

    t0 = detA/denom
    t1 = detB/denom

    pA = a0 + (torch.mul(torch.stack((t0, t0, t0), dim = 1), _A)) # Projected closest point on segment A
    pB = b0 + (torch.mul(torch.stack((t1, t1, t1), dim = 1), _B)) # Projected closest point on segment B

    
    return pA,pB, torch.norm(pA-pB, dim = 1)

def closestDistanceBetweenLines_multi(a0,a1,b0,b1):
    
    ''' Given two lines defined by numpy.array pairs (a0,a1,b0,b1)
        Return the closest points on each segment and their distance
    '''
    # Calculate denomitator
    A = a1 - a0
    B = b1 - b0
    magA = torch.norm(A, dim = 1)
    magB = torch.norm(B, dim = 1)
    
    _A = torch.div(A, torch.transpose(torch.stack((magA, magA, magA)), 0, 1))
    _B = torch.div(B, torch.transpose(torch.stack((magB, magB, magB)), 0, 1))
    
    cross = torch.cross(_A, _B, dim = 1)
    denom = torch.norm(cross, dim = 1)**2
    
    # Lines criss-cross: Calculate the projected closest points
    t = (b0 - a0)
    #detA = torch.det([t, _B, cross])
    #detB = torch.det([t, _A, cross])

    detA = torch.det(torch.stack((t, _B, cross), dim = 2))
    detB = torch.det(torch.stack((t, _A, cross), dim = 2))

    t0 = detA/denom
    t1 = detB/denom

    pA = a0 + (torch.mul(torch.stack((t0, t0, t0), dim = 1), _A)) # Projected closest point on segment A
    pB = b0 + (torch.mul(torch.stack((t1, t1, t1), dim = 1), _B)) # Projected closest point on segment B

    
    return pA,pB, torch.norm(pA-pB, dim = 1)