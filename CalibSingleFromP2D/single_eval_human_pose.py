
"""
Common evaluation metrics for 3D human pose estimation.

"""

import torch
import numpy as np

class Metrics:
    """
    This class contains metrics for human poses estimation.
    See examples/train_lightning.py for usage

    """

    def __init__(self, init=0):
        self.init = init

    def mpjpe(self, p_ref, p, use_scaling=True, root_joint=0):
        """
        Computes the Mean Per Joint Positioning Error (MPJPE).

        :param p_ref: The reference pose in format [batchsize, joints, 3]
        :param p: The predicted pose in format [batchsize, joints, 3]
        :param use_scaling: If set to True a scaling step is performed before computing the MPJPE, aka. N-MPJPE
        :param root_joint: index of the root joint
        :return: The mean MPJPE over the batch
        """

        # centralize all poses at their root joint
        p = p - p[:, root_joint:root_joint+1, :]
        p_ref = p_ref - p_ref[:, root_joint:root_joint+1, :]

        if use_scaling:
            p = self.scale_normalize(p, p_ref)

        err = (p - p_ref).norm(p=2, dim=2).mean(axis=1)

        return err

    def PCK(self, p_ref, p, use_scaling=True, root_joint=0, thresh=150.0):
        """
        Computes the Percentage of Correct Keypoints (PCK).
        The threshold is commonly set to 150mm.

        :param p_ref: The reference pose in format [batchsize, joints, 3]
        :param p: The predicted pose in format [batchsize, joints, 3]
        :param use_scaling: If set to True a scaling step is performed before computing the PCK, aka. N-PCK
        :param root_joint: index of the root joint
        :return: The mean PCK over the batch
        """

        num_joints = p.shape[1]

        # centralize all poses at their root joint
        p = p - p[:, root_joint:root_joint+1, :]
        p_ref = p_ref - p_ref[:, root_joint:root_joint+1, :]

        if use_scaling:
            p = self.scale_normalize(p, p_ref)

        err = ((p - p_ref).norm(dim=2) < thresh).sum()/(p_ref.shape[0]*num_joints)*100

        return err

    def AUC(self, p_ref, p, use_scaling=True, root_joint=0):
        """
        Computes the Area Under Curve (AUC) for the Percentage of Correct Keypoints (PCK).
        In contrast to PCK the threshold is variable in the range [0, 150].

        :param p_ref: The reference pose in format [batchsize, joints, 3]
        :param p: The predicted pose in format [batchsize, joints, 3]
        :param use_scaling: If set to True a scaling step is performed before computing the PCK, aka. N-PCK
        :param root_joint: index of the root joint
        :return: The mean AUC over the batch
        """

        # centralize all poses at their root joint
        p = p - p[:, root_joint:root_joint+1, :]
        p_ref = p_ref - p_ref[:, root_joint:root_joint+1, :]

        if use_scaling:
            p = self.scale_normalize(p, p_ref)

        distances = (p - p_ref).norm(dim=2)

        err = 0
        for t in torch.linspace(0, 150, 31):
            err += (distances < t).sum() / (distances.shape[0] * distances.shape[1] * 31)

        return err

    def CPS(self, p_ref, p):
        """
        Computes the Correct Poses Score (CPS) as defined in
        Wandt et al. "CanonPose: Self-Supervised Monocular 3D Human Pose Estimation in the Wild"

        :param p_ref: The reference pose in format [batchsize, joints, 3]
        :param p: The predicted pose in format [batchsize, joints, 3]
        :param use_scaling: If set to True a scaling step is performed before computing the PCK, aka. N-PCK
        :param root_joint: index of the root joint
        :return: The CPS for the batch
        """

        num_joints = p.shape[1]

        p_aligned = self.procrustes(p, p_ref, use_reflection=True, use_scaling=True)

        distances = (p_aligned - p_ref).norm(dim=2)

        err = 0
        for thresh in range(301):
            CP = ((distances < thresh).sum(dim=1) == num_joints)
            err = err + CP.sum()/CP.shape[0]

        return err

    def pmpjpe(self, p_ref, p, use_reflection=True, use_scaling=True):
        """
        Computes the Procrustes aligned MPJPE, aka. P-MPJPE.
        The threshold is commonly set to 150mm.

        :param p_ref: The reference pose in format [batchsize, joints, 3]
        :param p: The predicted pose in format [batchsize, joints, 3]
        :param use_reflection: If set to True a the best reflection is used to compute MPJPE. This is the standard setting.
        :param use_scaling: If set to True a scaling step is performed before computing the MPJPE.
        :param root_joint: index of the root joint
        :return: The mean P-MPJPE over the batch
        """

        p_aligned = self.procrustes(p, p_ref, use_reflection=use_reflection, use_scaling=use_scaling)

        err = (p_ref - p_aligned).norm(p=2, dim=2).mean(axis=1)

        return err

    def scale_normalize(self, poses_inp, template_poses):
        """
        Computes the optimal scale for the input poses to best match the template poses.

        :param poses_inp: The poses that need to be aligned in format [batchsize, joints, 3]
        :param template_poses: The reference pose in format [batchsize, joints, 3]
        :return: The poses after Procrustes alignment in format [batchsize, joints, 3]
        """

        num_joints = poses_inp.shape[1]

        scale_p = poses_inp.reshape(-1, 3 * num_joints).norm(p=2, dim=1, keepdim=True)
        scale_p_ref = template_poses.reshape(-1, 3 * num_joints).norm(p=2, dim=1, keepdim=True)
        scale = scale_p_ref / scale_p
        poses_scaled = (poses_inp.reshape(-1, 3 * num_joints) * scale).reshape(-1, num_joints, 3)

        return poses_scaled

    def procrustes(self, poses_inp, template_poses, use_reflection=True, use_scaling=True):
        """
        Computes the Procrustes alignment between the input poses and the template poses.

        :param poses_inp: The poses that need to be aligned in format [batchsize, joints, 3]
        :param template_poses: The reference pose in format [batchsize, joints, 3]
        :param use_reflection: If set to True a the best reflection is used to compute MPJPE. This is the standard setting.
        :param use_scaling: If set to True a scaling step is performed before computing the MPJPE.
        :return: The poses after Procrustes alignment in format [batchsize, joints, 3]
        """

        num_joints = int(poses_inp.shape[1])

        poses_inp = poses_inp.permute(0, 2, 1)
        template_poses = template_poses.permute(0, 2, 1)

        # translate template
        translation_template = template_poses.mean(axis=2, keepdims=True)
        template_poses_centered = template_poses - translation_template

        # scale template
        scale_t = torch.sqrt((template_poses_centered**2).sum(axis=[1, 2], keepdim=True) / (3*num_joints))
        template_poses_scaled = template_poses_centered / scale_t

        # translate prediction
        translation = poses_inp.mean(axis=2, keepdims=True)
        poses_centered = poses_inp - translation

        # scale prediction
        scale_p = torch.sqrt((poses_centered**2).sum(axis=[1, 2], keepdim=True) / (3*num_joints))
        poses_scaled = poses_centered / scale_p

        # rotation
        U, S, V = torch.svd(torch.matmul(template_poses_scaled, poses_scaled.transpose(2, 1)))
        R = torch.matmul(U, V.transpose(2, 1))

        # avoid reflection
        if not use_reflection:
            # only rotation
            Z = torch.eye(3).repeat(R.shape[0], 1, 1).to(poses_inp.device)
            Z[:, -1, -1] *= R.det()
            R = Z.matmul(R)

        poses_pa = torch.matmul(R, poses_scaled)

        # upscale again
        if use_scaling:
            poses_pa *= scale_t

        poses_pa += translation_template

        return poses_pa.permute(0, 2, 1)
    
    def procrustes_rotation(self, poses_inp, template_poses, use_reflection=True, use_scaling=True):
        """
        Computes the Procrustes alignment between the input poses and the template poses.

        :param poses_inp: The poses that need to be aligned in format [batchsize, joints, 3]
        :param template_poses: The reference pose in format [batchsize, joints, 3]
        :param use_reflection: If set to True a the best reflection is used to compute MPJPE. This is the standard setting.
        :param use_scaling: If set to True a scaling step is performed before computing the MPJPE.
        :return: The poses after Procrustes alignment in format [batchsize, joints, 3]
        """

        num_joints = int(poses_inp.shape[1])

        poses_inp = poses_inp.permute(0, 2, 1)
        template_poses = template_poses.permute(0, 2, 1)

        # translate template
        translation_template = template_poses.mean(axis=2, keepdims=True)
        template_poses_centered = template_poses - translation_template

        # scale template
        scale_t = torch.sqrt((template_poses_centered**2).sum(axis=[1, 2], keepdim=True) / (3*num_joints))
        template_poses_scaled = template_poses_centered / scale_t

        # translate prediction
        translation = poses_inp.mean(axis=2, keepdims=True)
        poses_centered = poses_inp - translation

        # scale prediction
        scale_p = torch.sqrt((poses_centered**2).sum(axis=[1, 2], keepdim=True) / (3*num_joints))
        poses_scaled = poses_centered / scale_p

        # rotation
        U, S, V = torch.svd(torch.matmul(template_poses_scaled, poses_scaled.transpose(2, 1)))
        R = torch.matmul(U, V.transpose(2, 1))

        # avoid reflection
        if not use_reflection:
            # only rotation
            Z = torch.eye(3).repeat(R.shape[0], 1, 1).to(poses_inp.device)
            Z[:, -1, -1] *= R.det()
            R = Z.matmul(R)
        R = torch.tensor([[1.0,0.0,0.0], [0.0,1.0,0.0], [0.0,0.0,1.0]])

        poses_pa = torch.matmul(R, poses_scaled)
        # upscale again
        if use_scaling:
            poses_pa *= scale_t
        #stop
        poses_pa += translation_template

        return poses_pa.permute(0, 2, 1), R, translation_template 

    def procrustes_rotation_translation(self, template_poses, template_axis, poses_inp, poses_axis, use_reflection=True, use_scaling=True):
        """
        Computes the Procrustes alignment between the input poses and the template poses.

        :param poses_inp: The poses that need to be aligned in format [batchsize, joints, 3]
        :param template_poses: The reference pose in format [batchsize, joints, 3]
        :param use_reflection: If set to True a the best reflection is used to compute MPJPE. This is the standard setting.
        :param use_scaling: If set to True a scaling step is performed before computing the MPJPE.
        :return: The poses after Procrustes alignment in format [batchsize, joints, 3]
        """

        num_joints = int(poses_inp.shape[1])

        poses_inp = poses_inp.permute(0, 2, 1)
        template_poses = template_poses.permute(0, 2, 1)
        #print(template_poses.shape, " POSES INPPP")
        # translate template
        translation_template = template_poses[:,:,0:1]#template_poses.mean(axis=2, keepdims=True)
        #translation_sync = poses_inp[:,:,0:1]
        #print(translation_template.shape, " hhddddd")
        template_poses_centered = template_poses - translation_template
        #template_poses_centered = poses_inp - translation_sync

        # scale template
        scale_t = torch.sqrt((template_poses_centered**2).sum(axis=[1, 2], keepdim=True) / (3*num_joints))
        template_poses_scaled = template_poses_centered / scale_t
        #print(template_poses_scaled.shape, " HELOOOASDADSASDdddddddddddd")
        #stop
        # translate prediction
        translation = poses_inp[:,:,0:1]#poses_inp.mean(axis=2, keepdims=True)
        poses_centered = poses_inp - translation

        # scale prediction
        scale_p = torch.sqrt((poses_centered**2).sum(axis=[1, 2], keepdim=True) / (3*num_joints))
        poses_scaled = poses_centered / scale_p
        
        poses_template = []
        poses_sync = []
        for i in range(poses_scaled.shape[2]):
            #print(poses_axis)
            pose_axis = torch.transpose(torch.tensor(poses_axis[i]), 0, 1)
            pose_points = poses_scaled[:,:,i]
            poses_sync.append(pose_points + pose_axis[:,0])
            poses_sync.append(pose_points + pose_axis[:,1])
            poses_sync.append(pose_points + pose_axis[:,2])

        for i in range(template_poses_scaled.shape[2]):
            #print(poses_axis)
            pose_axis = torch.transpose(torch.tensor(template_axis[i]), 0, 1)
            pose_points = template_poses_scaled[:,:,i]
            poses_template.append(pose_points + pose_axis[:,0])
            poses_template.append(pose_points + pose_axis[:,1])
            poses_template.append(pose_points + pose_axis[:,2])   

        pose_axis_template = torch.stack(poses_template, dim = 0).permute(1, 2, 0)
        pose_axis_sync = torch.stack(poses_sync, dim = 0).permute(1, 2, 0)

        print(template_poses_scaled)
        print(poses_scaled)

        all_points_template = torch.concat((pose_axis_template, template_poses_scaled), dim = 2)
        all_points_sync = torch.concat((pose_axis_sync, poses_scaled), dim = 2)

        #print(all_points_template.shape, all_points_poses.shape, " 1231233333333333")
        #stop
        # rotation
        '''
        U, S, V = torch.svd(torch.matmul(template_poses_scaled, poses_scaled.transpose(2, 1)))
        R = torch.matmul(U, V.transpose(2, 1)).double()
        '''
        
        U, S, V = torch.svd(torch.matmul(all_points_template, all_points_sync.transpose(2, 1)))
        R = torch.matmul(U, V.transpose(2, 1)).double()
        print(all_points_template.shape, all_points_sync.shape, " AAAAAAAAAAAAAAASDDDDDDDDDDDDDDDDDD")
        print(template_axis[0], " template axis")
        print(poses_axis[0], " pose aixsss")
        print(R, " rotationssss")
        
        '''
        pose_array = torch.transpose(R @ torch.unsqueeze(torch.tensor(np.stack(poses_axis[0], axis = 0)), dim = 0), 2, 1)
        template_array = R @ torch.unsqueeze(torch.tensor(template_axis[0]), dim = 0)
        print(pose_array.shape, template_array.shape, " pose arrayaaa")
        U, S, V = torch.svd(torch.matmul(template_array, pose_array))
        R = torch.matmul(U, V.transpose(2, 1)).double()
        '''
        '''
        pose_xyz = torch.tensor(np.stack(poses_axis[0], axis = 0))
        template_xyz = torch.tensor(template_axis[0])

        pose_array = torch.transpose(torch.unsqueeze(pose_xyz, dim = 0), 2, 1)
        template_array = torch.unsqueeze(template_xyz, dim = 0)
        U, S, V = torch.svd(torch.matmul(template_array, pose_array))
        R = torch.matmul(U, V.transpose(2, 1)).double()
        '''
        
        '''
        r =  Rotation.from_matrix(R)
        angles = r.as_euler("zyx",degrees=False)
        print(angles)
        
        R = torch.tensor([[math.cos(angles[0][1]),    -math.sin(angles[0][1]),    0],
                [math.sin(angles[0][1]),    math.cos(angles[0][1]),     0],
                [0,                     0,                      1]
                ]).double()
        '''
        print(R, " R")
        #stop
        print(R.det(), " RRRRRRRRR")

        
        '''
        R = torch.tensor([[1.0,0.0,0.0], [0.0,1.0,0.0], [0.0,0.0,1.0]]).double()
        # avoid reflection
        if not use_reflection:
            # only rotation
            Z = torch.eye(3).repeat(R.shape[0], 1, 1).to(poses_inp.device).double()
            Z[:, -1, -1] *= R.det()
            R = Z.matmul(R)
        R = torch.tensor([[1.0,0.0,0.0], [0.0,1.0,0.0], [0.0,0.0,1.0]]).double()
        '''
        #R = torch.tensor([[1.0,0.0,0.0], [0.0,1.0,0.0], [0.0,0.0,1.0]]).double()
        print(poses_scaled, " poses scaled")
        poses_pa = torch.matmul(R, poses_scaled)
        print("******************************************")
        print(R, " rotaiton")
        print(poses_pa, " poses pa")

        transform_axis = []
        for i in range(len(poses_axis)):
    
            #poses_pa_axis = torch.matmul(R, torch.transpose(torch.transpose(torch.tensor(poses_axis[i]), 0, 1), 0, 1))
            print(poses_axis[i], " HIIAISDASDASDASD")
            poses_pa_axis = torch.matmul(R, torch.tensor(poses_axis[i]))
            transform_axis.append(torch.squeeze(poses_pa_axis).numpy())

        # upscale again
        if use_scaling:
            poses_pa *= scale_t

        poses_pa += translation_template

        #return poses_pa.permute(0, 2, 1), R, translation_template 
        return poses_pa.permute(0, 2, 1).numpy(), transform_axis, R, translation_template 
    
    def procrustes_rotation_translation_scale(self, poses_inp, poses_axis, template_poses, template_axis, use_reflection=True, use_scaling=True):
        """
        Computes the Procrustes alignment between the input poses and the template poses.

        :param poses_inp: The poses that need to be aligned in format [batchsize, joints, 3]
        :param template_poses: The reference pose in format [batchsize, joints, 3]
        :param use_reflection: If set to True a the best reflection is used to compute MPJPE. This is the standard setting.
        :param use_scaling: If set to True a scaling step is performed before computing the MPJPE.
        :return: The poses after Procrustes alignment in format [batchsize, joints, 3]
        """

        num_joints = int(poses_inp.shape[1])

        poses_inp = poses_inp.permute(0, 2, 1)
        template_poses = template_poses.permute(0, 2, 1)
        #print(template_poses.shape, " POSES INPPP")
        # translate template
        translation_template = template_poses.mean(axis=2, keepdims=True)
        #translation_sync = poses_inp[:,:,0:1]
        #print(translation_template.shape, " hhddddd")
        template_poses_centered = template_poses - translation_template
        #template_poses_centered = poses_inp - translation_sync

        # scale template
        scale_t = torch.sqrt((template_poses_centered**2).sum(axis=[1, 2], keepdim=True) / (3*num_joints))

        translation = poses_inp.mean(axis=2, keepdims=True)
        poses_centered = poses_inp - translation

        # scale prediction
        scale_p = torch.sqrt((poses_centered**2).sum(axis=[1, 2], keepdim=True) / (3*num_joints))

        template_poses_scaled = template_poses_centered / scale_t

        # translate prediction
        poses_scaled = poses_centered / scale_p
        
        poses_template = []
        poses_sync = []
        for i in range(poses_scaled.shape[2]):
            #print(poses_axis)
            pose_axis = torch.transpose(torch.tensor(poses_axis[i]), 0, 1)
            pose_points = poses_scaled[:,:,i]
            poses_sync.append(pose_points + pose_axis[:,0])
            poses_sync.append(pose_points + pose_axis[:,1])
            poses_sync.append(pose_points + pose_axis[:,2])

        for i in range(template_poses_scaled.shape[2]):
            #print(poses_axis)
            pose_axis = torch.transpose(torch.tensor(template_axis[i]), 0, 1)
            pose_points = template_poses_scaled[:,:,i]
            poses_template.append(pose_points + pose_axis[:,0])
            poses_template.append(pose_points + pose_axis[:,1])
            poses_template.append(pose_points + pose_axis[:,2])   

        pose_axis_template = torch.stack(poses_template, dim = 0).permute(1, 2, 0)
        pose_axis_sync = torch.stack(poses_sync, dim = 0).permute(1, 2, 0)

        print(template_poses_scaled)
        print(poses_scaled)

        all_points_template = torch.concat((pose_axis_template, template_poses_scaled), dim = 2)
        all_points_sync = torch.concat((pose_axis_sync, poses_scaled), dim = 2)

        #print(all_points_template.shape, all_points_poses.shape, " 1231233333333333")
        #stop
        # rotation
        '''
        U, S, V = torch.svd(torch.matmul(template_poses_scaled, poses_scaled.transpose(2, 1)))
        R = torch.matmul(U, V.transpose(2, 1)).double()
        '''
        
        U, S, V = torch.svd(torch.matmul(all_points_template, all_points_sync.transpose(2, 1)))
        R = torch.matmul(U, V.transpose(2, 1)).double()
        print(all_points_template.shape, all_points_sync.shape, " AAAAAAAAAAAAAAASDDDDDDDDDDDDDDDDDD")
        print(template_axis[0], " template axis")
        print(poses_axis[0], " pose aixsss")
        print(R, " rotationssss")
        
        '''
        pose_array = torch.transpose(R @ torch.unsqueeze(torch.tensor(np.stack(poses_axis[0], axis = 0)), dim = 0), 2, 1)
        template_array = R @ torch.unsqueeze(torch.tensor(template_axis[0]), dim = 0)
        print(pose_array.shape, template_array.shape, " pose arrayaaa")
        U, S, V = torch.svd(torch.matmul(template_array, pose_array))
        R = torch.matmul(U, V.transpose(2, 1)).double()
        '''
        '''
        pose_xyz = torch.tensor(np.stack(poses_axis[0], axis = 0))
        template_xyz = torch.tensor(template_axis[0])

        pose_array = torch.transpose(torch.unsqueeze(pose_xyz, dim = 0), 2, 1)
        template_array = torch.unsqueeze(template_xyz, dim = 0)
        U, S, V = torch.svd(torch.matmul(template_array, pose_array))
        R = torch.matmul(U, V.transpose(2, 1)).double()
        '''
        
        '''
        r =  Rotation.from_matrix(R)
        angles = r.as_euler("zyx",degrees=False)
        print(angles)
        
        R = torch.tensor([[math.cos(angles[0][1]),    -math.sin(angles[0][1]),    0],
                [math.sin(angles[0][1]),    math.cos(angles[0][1]),     0],
                [0,                     0,                      1]
                ]).double()
        '''
        print(R, " R")
        #stop
        print(R.det(), " RRRRRRRRR")

        
        '''
        R = torch.tensor([[1.0,0.0,0.0], [0.0,1.0,0.0], [0.0,0.0,1.0]]).double()
        # avoid reflection
        if not use_reflection:
            # only rotation
            Z = torch.eye(3).repeat(R.shape[0], 1, 1).to(poses_inp.device).double()
            Z[:, -1, -1] *= R.det()
            R = Z.matmul(R)
        R = torch.tensor([[1.0,0.0,0.0], [0.0,1.0,0.0], [0.0,0.0,1.0]]).double()
        '''
        #R = torch.tensor([[1.0,0.0,0.0], [0.0,1.0,0.0], [0.0,0.0,1.0]]).double()
        print(poses_scaled, " poses scaled")
        poses_pa = torch.matmul(R, poses_scaled)
        print("******************************************")
        print(R, " rotaiton")
        print(poses_pa, " poses pa")

        transform_axis = []
        for i in range(len(poses_axis)):

            #poses_pa_axis = torch.matmul(R, torch.transpose(torch.transpose(torch.tensor(poses_axis[i]), 0, 1), 0, 1))
            print(poses_axis[i], " HIIAISDASDASDASD")
            poses_pa_axis = torch.matmul(R, torch.tensor(poses_axis[i]))
            transform_axis.append(torch.squeeze(poses_pa_axis).numpy())

        # upscale again
        if use_scaling:
            poses_pa *= scale_t
            poses_scaled *= scale_t

        poses_pa += translation_template

        #return poses_pa.permute(0, 2, 1), R, translation_template 
        return poses_pa.permute(0, 2, 1).numpy(), template_poses_scaled.permute(0, 2, 1).numpy(), transform_axis, R, translation_template 
    
    def procrustes_rotation_translation_template(self, poses_inp, poses_axis, template_poses, template_axis, use_reflection=True, use_scaling=True):
        num_joints = int(poses_inp.shape[1])
        poses_inp = poses_inp.permute(0, 2, 1)
        #print(template_poses, " POSES")
        #print(template_poses.shape, " shape ")
        template_poses = template_poses.permute(0, 2, 1)

        translation_template = template_poses.mean(axis=2, keepdims=True)

        template_poses_centered = template_poses - translation_template

        # scale template
        scale_t = torch.sqrt((template_poses_centered**2).sum(axis=[1, 2], keepdim=True) / (3*num_joints))

        translation = poses_inp.mean(axis=2, keepdims=True)
        poses_centered = poses_inp - translation

        # scale prediction
        scale_p = torch.sqrt((poses_centered**2).sum(axis=[1, 2], keepdim=True) / (3*num_joints))

        template_poses_scaled = template_poses_centered / scale_t

        # translate prediction
        poses_scaled = poses_centered / scale_p
        
        poses_template = []
        poses_sync = []
        for i in range(poses_scaled.shape[2]):
            #print(poses_axis.shape, " P O S E S   A X I S   ")
            pose_axis = torch.transpose(torch.tensor(poses_axis[i]), 0, 1)
            pose_points = poses_scaled[:,:,i]
            poses_sync.append(pose_points + pose_axis[:,0])
            poses_sync.append(pose_points + pose_axis[:,1])
            poses_sync.append(pose_points + pose_axis[:,2])

        for i in range(template_poses_scaled.shape[2]):
            #print(poses_axis)
            pose_axis = torch.transpose(torch.tensor(template_axis[i]), 0, 1)
            pose_points = template_poses_scaled[:,:,i]
            poses_template.append(pose_points + pose_axis[:,0])
            poses_template.append(pose_points + pose_axis[:,1])
            poses_template.append(pose_points + pose_axis[:,2])   

        pose_axis_template = torch.stack(poses_template, dim = 0).permute(1, 2, 0)
        pose_axis_sync = torch.stack(poses_sync, dim = 0).permute(1, 2, 0)

        print(template_poses_scaled)
        print(poses_scaled)

        all_points_template = torch.concat((pose_axis_template, template_poses_scaled), dim = 2)
        all_points_sync = torch.concat((pose_axis_sync, poses_scaled), dim = 2)
        
        #print(all_points_template.shape, all_points_poses.shape, " 1231233333333333")
        #stop
        # rotation
        '''
        U, S, V = torch.svd(torch.matmul(template_poses_scaled, poses_scaled.transpose(2, 1)))
        R = torch.matmul(U, V.transpose(2, 1)).double()
        
        '''
        U, S, V = torch.svd(torch.matmul(all_points_template, all_points_sync.transpose(2, 1)))
        R = torch.matmul(U, V.transpose(2, 1)).double()
        #remove reflections !!!
        Z = torch.eye(3).repeat(R.shape[0], 1, 1).to(poses_inp.device).double()
        Z[:, -1, -1] *= R.det()
        R = Z.matmul(R)
        
        poses_pa = torch.matmul(R, poses_scaled)

        ####

        transform_axis = []
        for i in range(len(poses_axis)):
            
            axis_array = []
            for j in range(len(poses_axis[i])):
                #poses_pa_axis = torch.matmul(R, torch.transpose(torch.transpose(torch.tensor(poses_axis[i]), 0, 1), 0, 1))
                #print(poses_axis[i], " HIIAISDASDASDASD")
                poses_pa_axis = torch.matmul(R, torch.tensor(poses_axis[i][j]))

                #print(poses_pa_axis, " poses_pa_axis")            
                axis_array.append(torch.squeeze(poses_pa_axis).numpy())

            transform_axis.append(axis_array)
        #stop
        # upscale again
        if use_scaling:
            poses_pa *= scale_t

        poses_pa += translation_template

        #print(poses_pa.shape, " poses_paposes_pa")

        #return poses_pa.permute(0, 2, 1), R, translation_template 
        return np.squeeze(poses_pa.permute(0, 2, 1).numpy()), transform_axis, R, translation_template, translation