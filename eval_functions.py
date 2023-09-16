import sys
import os
parent_directory = os.path.abspath(os.path.join(os.getcwd(), '..'))
sys.path.append(parent_directory)
import numpy as np
import math
from xml.dom import minidom
import matplotlib.image as mpimg
import json
import torch
from eval_human_pose import Metrics
metrics = Metrics()

def get_tsai(tsai_path, tsai_intrinsics_path, terrace_tsai_cal):
    
    intrinsic_array = []
    extrinsic_array = []
    cam_position = []
    cam_axis = []

    for itr in range(len(terrace_tsai_cal)):

        file = minidom.parse(tsai_path + terrace_tsai_cal[itr])

        models_extrinsics = file.getElementsByTagName('Extrinsic')
        models_instrinsics = file.getElementsByTagName('Intrinsic')
        models_geometry = file.getElementsByTagName('Geometry')
        
        tx = float(models_extrinsics[0].attributes.items()[0][1])
        ty = float(models_extrinsics[0].attributes.items()[1][1])
        tz = float(models_extrinsics[0].attributes.items()[2][1])
        rx = float(models_extrinsics[0].attributes.items()[3][1])
        ry = float(models_extrinsics[0].attributes.items()[4][1])
        rz = float(models_extrinsics[0].attributes.items()[5][1])

        img_x = float(models_geometry[0].attributes.items()[0][1])
        img_y = float(models_geometry[0].attributes.items()[1][1])
        ncx = float(models_geometry[0].attributes.items()[2][1])
        nfx = float(models_geometry[0].attributes.items()[3][1])
        dx = float(models_geometry[0].attributes.items()[4][1])
        dy = float(models_geometry[0].attributes.items()[5][1])
        dpx = float(models_geometry[0].attributes.items()[6][1])
        dpy = float(models_geometry[0].attributes.items()[7][1])
        
        #print(img_x, img_y, " gasaass")
        f = float(models_instrinsics[0].attributes.items()[0][1])
        kappa = float(models_instrinsics[0].attributes.items()[1][1]) 
        t1 = float(models_instrinsics[0].attributes.items()[2][1]) 
        t2 = float(models_instrinsics[0].attributes.items()[3][1])
        sx = float(models_instrinsics[0].attributes.items()[4][1]) 
        
        cam_matrix = np.array([[f,         0,                  t1                   ],
                            [0,         f, t2 ],
                            [0,         0, 1  ]
                            ])
        
        R_x = np.array([[1,         0,                  0                   ],
                        [0,         math.cos(rx), -math.sin(rx) ],
                        [0,         math.sin(rx), math.cos(rx)  ]
                        ])
    
        R_y = np.array([[math.cos(ry),    0,      math.sin(ry)  ],
                        [0,                     1,      0                   ],
                        [-math.sin(ry),   0,      math.cos(ry)  ]
                        ])
    
        R_z = np.array([[math.cos(rz),    -math.sin(rz),    0],
                        [math.sin(rz),    math.cos(rz),     0],
                        [0,                     0,                      1]
                        ])
        R_rot = R_z @ R_y @ R_x

        extrinsic_array.append(R_rot)

        #cam_pos = (np.linalg.inv(R_rot)@np.array([tx,ty,tz]))
        cam_pos = (np.linalg.inv(R_rot)@(-np.array([tx,ty,tz])))
        cam_position.append(cam_pos)

        #cam_axis.append([R_rot[:,0], R_rot[:,1], R_rot[:,2]])
        cam_axis.append([np.linalg.inv(R_rot)[:,0], np.linalg.inv(R_rot)[:,1], np.linalg.inv(R_rot)[:,2]])

        with open(tsai_intrinsics_path + "single_calibration_" + str(itr) + "_.json", 'r') as f:
            cam_intrinsics = json.load(f)

        intrinsic_array.append(cam_intrinsics["intrinsics"])

    return intrinsic_array, extrinsic_array, cam_position, cam_axis

def evaluate(cam_intrinsics_array, cam_axis, cam_position, intrinsic_array_tsai, cam_axis_tsai, cam_position_tsai, output_path, data = 'result'):
    
    results_focal_pred = []
    results_focal_tsai = []

    results_extriniscs_pred = []
    results_extriniscs_tsai = []

    results_position_diff = []
    print(cam_position)
    for c in range(len(cam_position[0])):
        results_position_diff.append(np.linalg.norm(cam_position[0][c] - cam_position_tsai[0][c]))

    for c in range(len(cam_intrinsics_array)):
        #print(len(intrinsic_array_tsai))
        #print(len(cam_intrinsics_array), " hqwe123132")
        cam_pred = cam_intrinsics_array[c]
        cam_tsai = intrinsic_array_tsai[c]

        #print(cam_pred, " cam pred")
        #print(cam_tsai, " cam tsai")

        results_focal_pred.append(cam_pred[0][0])
        results_focal_tsai.append(cam_tsai[0][0])

    rot_pred_ref = cam_axis[0]
    rot_tsai_ref = cam_axis_tsai[0]

    for c in range(len(cam_axis)):
        rot_pred = cam_axis[c]
        rot_tsai = cam_axis_tsai[c]

        rotation_pred = np.linalg.inv(rot_pred_ref) @ rot_pred
        rotation_tsai = np.linalg.inv(rot_tsai_ref) @ rot_tsai

        pred_angle = np.rad2deg(np.arccos((np.trace(rotation_pred)-1)/2))
        tsai_angle = np.rad2deg(np.arccos((np.trace(rotation_tsai)-1)/2))

        print(pred_angle, " pred angle")
        print(tsai_angle, " tsai angle")
        print(rotation_pred)
        print(rotation_tsai)
        print((np.trace(rotation_pred)-1)/2, " pred")
        print((np.trace(rotation_tsai)-1)/2, " tsai")

        results_extriniscs_pred.append(pred_angle)
        results_extriniscs_tsai.append(tsai_angle)
        #print(pred_angle, " rot_pred")
        #print(tsai_angle, " rot_tsai")

    angle_diff = []
    for c in range(len(cam_axis)):
        rot_pred = cam_axis[c]
        rot_tsai = cam_axis_tsai[c]

        rotation_pred = np.linalg.inv(rot_pred) @ rot_tsai
        
        print((np.trace(rotation_pred)-1)/2, " THE ROTATION PRED TRACE", c)
        pred_angle = np.rad2deg(np.arccos((np.trace(rotation_pred)-1)/2))
        angle_diff.append(pred_angle)
        print(pred_angle, " CAM " + str(c))

    cam_position_torch = torch.tensor(cam_position).double()
    cam_position_tsai_torch = torch.tensor(cam_position_tsai).double()

    #print(cam_position_torch.shape, " HELOOAasssssssssss")
    error_npjpe = metrics.mpjpe(cam_position_tsai_torch, cam_position_torch, use_scaling=True, root_joint=0)

    print(error_npjpe)


    results_focal_pred_array = []
    results_focal_tsai_array = []
    angle_diff_array = []
    error_npjpe_array = []
    focal_error_array = []
    results_position_diff_array = []
    with open(output_path + data + '.csv','w') as file:

        file.write('focal pred, focal tsai, angle difference, nmpjpe, focal percent error, position error (meters)')
        file.write('\n')
        
        for i in range(len(results_focal_pred)):

            focal_error = 100*np.absolute(results_focal_pred[i] - results_focal_tsai[i])/results_focal_tsai[i]
            file.write(str(results_focal_pred[i]) + ',' + str(results_focal_tsai[i])  + ',' + str(angle_diff[i]) + ',' + str(error_npjpe.item()) + ',' + str(focal_error) + ',' + str(results_position_diff[i]))
            file.write('\n')
            
            results_focal_pred_array.append(results_focal_pred[i])
            results_focal_tsai_array.append(results_focal_tsai[i])
            angle_diff_array.append(angle_diff[i],)
            error_npjpe_array.append( error_npjpe.item())
            focal_error_array.append(focal_error)
            results_position_diff_array.append(results_position_diff[i])
            #results_focal_pred[i], results_focal_tsai[i], angle_diff[i], error_npjpe.item(), focal_error, results_position_diff[i]
    
    return np.mean(results_focal_pred_array), np.mean(results_focal_tsai_array), np.mean(angle_diff_array), np.mean(error_npjpe_array), np.mean(focal_error_array), np.mean(results_position_diff_array)