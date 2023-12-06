import torch
import plotly.graph_objects as go

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