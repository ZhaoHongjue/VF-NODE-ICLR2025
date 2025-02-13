'''
Author: Hongjue Zhao
Email:  hongjue2@illinois.edu
Date:   02/12/2025
'''

import matplotlib.pyplot as plt
from math import ceil

def plot_eval_phase(
    ys, pred_ys,
    sampled_ys = None, 
    save_pth: str = None,
):
    '''
    Plot the phase portrait of the ground truth and predictions.
    
    Args:
        - `ys`: `Float[Array, 'traj dim']` - Ground truth.
        - `pred_ys`: `Float[Array, 'traj dim']` - Predictions.
        - `sampled_ys`: `Float[Array, 'traj dim']` - Sampled observations.
        - `save_pth`: `str` - Path to save the plot.
    '''
    if ys.shape[1] == 1 or ys.shape[1] > 3: return
    
    projection = '3d' if ys.shape[1] == 3 else None
    fig = plt.figure(figsize = (8, 6))
    ax = plt.axes(projection = projection)
    
    lw, pad = 2.2, 10
    legend_fs, label_fs, tick_fs = 13, 15, 13
    
    ax.plot(
        *[y for y in ys.T], label = 'Ground truth',
        linestyle = '-', lw = lw, color = 'black',
    )
    ax.plot(
        *[y for y in pred_ys.T], label = 'Predictions',
        linestyle = '-.', lw = lw, color = 'salmon',
    )
    if sampled_ys is not None:
        ax.scatter(
            *[y for y in sampled_ys.T], label = 'Observations',
            s = 10, lw = lw, color = 'cornflowerblue',
        )
        
    ax.set_xlabel('$x_1$', fontsize = label_fs, labelpad = pad)
    ax.set_ylabel('$x_2$', fontsize = label_fs, labelpad = pad)
    plt.tick_params(axis = 'both', labelsize = tick_fs)
    plt.legend(fontsize = legend_fs)
    plt.grid(color = 'grey', linestyle = '--')
    
    if projection:
        ax.set_zlabel('$x_3$', fontsize = label_fs, labelpad = pad)
        ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
        ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
        ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
        ax.xaxis._axinfo["grid"].update({"linewidth":0.5, 'linestyle':'--', 'color':'grey'})
        ax.yaxis._axinfo["grid"].update({"linewidth":0.5, 'linestyle':'--', 'color':'grey'})
        ax.zaxis._axinfo["grid"].update({"linewidth":0.5, 'linestyle':'--', 'color':'grey'})

    is_clean = 'clean' if sampled_ys is None else 'noisy'
    postfix = f'-{is_clean}-phase'
    
    if save_pth:
        plt.savefig(save_pth + postfix + '.pdf')
        plt.savefig(save_pth + postfix + '.svg')
        plt.savefig(save_pth + postfix + '.png')


def plot_eval_tspan(
    ts, ys, pred_ys,
    sampled_ys = None, 
    save_pth: str = None,
):
    '''
    Plot the time series of the ground truth and predictions.
    
    Args:
        - `ts`: `Float[Array, 'traj tspan']` - Time points.
        - `ys`: `Float[Array, 'traj dim']` - Ground truth.
        - `pred_ys`: `Float[Array, 'traj dim']` - Predictions.
        - `sampled_ys`: `Float[Array, 'traj dim']` - Sampled observations.
        - `save_pth`: `str` - Path to save the plot.
    '''
    fig_num = ys.shape[1]
    
    if 3 < fig_num < 15: arrange = (2, ceil(fig_num / 2))
    elif fig_num > 15: arrange = (5, ceil(fig_num / 5))
    else: arrange = (1, fig_num)
    
    fig = plt.figure(figsize = (8*arrange[1], 6*arrange[0]))
    lw, pad = 2.2, 10
    legend_fs, label_fs, tick_fs = 13, 15, 13
    
    for idx in range(fig_num):
        plt.subplot(arrange[0], arrange[1], idx + 1)
        plt.plot(
            ts, ys[:, idx], label = 'Ground Truth',
            linestyle = '-', lw = lw, color = 'black',
        )
        plt.plot(
            ts, pred_ys[:, idx], label = 'Predictions',
            linestyle = '-.', lw = lw, color = 'salmon',
        )
        if sampled_ys is not None:
            plt.scatter(
                ts, sampled_ys[:, idx], label = 'Observations',
                s = 10, lw = lw, color = 'cornflowerblue',
            )
            
        plt.xlabel('$t$', fontsize = label_fs, labelpad = pad)
        plt.ylabel(f'$x_{idx+1}$', fontsize = label_fs, labelpad = pad)
        plt.tick_params(axis = 'both', labelsize = tick_fs)
        plt.legend(fontsize = legend_fs)
        plt.grid(color = 'grey', linestyle = '--')
        
    is_clean = 'clean' if sampled_ys is None else 'noisy'
    postfix = f'-{is_clean}-tspan'
    
    if save_pth:
        plt.savefig(save_pth + postfix + '.pdf')
        plt.savefig(save_pth + postfix + '.svg')
        plt.savefig(save_pth + postfix + '.png')
    
    
# def plot_result_phase(
#     test_x, pred_x, 
#     pred_label: str,
#     color: str = 'salmon',
#     save_pth: str = None
# ):
#     assert len(test_x.shape) == 2
#     assert len(pred_x.shape) == 2
#     if test_x.shape[1] == 1 or test_x.shape[1] > 3: return
    
#     projection = '3d' if test_x.shape[1] == 3 else None
#     fig = plt.figure(figsize = (8, 6))
#     ax = plt.axes(projection = projection)
    
#     lw, pad = 2.2, 10
#     legend_fs, label_fs, tick_fs = 13, 15, 13

#     ax.plot(
#         *[x for x in test_x.T], label = 'odeint',
#         linestyle = '-', lw = lw, color = 'black',
#     )
#     ax.plot(
#         *[x for x in pred_x.T], label = pred_label,
#         linestyle = '--', lw = lw, color = color,
#     )
    
#     ax.set_xlabel('$x_1$', fontsize = label_fs, labelpad = pad)
#     ax.set_ylabel('$x_2$', fontsize = label_fs, labelpad = pad)
#     plt.tick_params(axis = 'both', labelsize = tick_fs)
#     plt.legend(fontsize = legend_fs)
#     plt.grid(color = 'grey', linestyle = '--')
    
#     if projection:
#         ax.set_zlabel('$x_3$', fontsize = label_fs, labelpad = pad)
#         ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
#         ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
#         ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
#         ax.xaxis._axinfo["grid"].update({"linewidth":0.5, 'linestyle':'--', 'color':'grey'})
#         ax.yaxis._axinfo["grid"].update({"linewidth":0.5, 'linestyle':'--', 'color':'grey'})
#         ax.zaxis._axinfo["grid"].update({"linewidth":0.5, 'linestyle':'--', 'color':'grey'})

#     if save_pth:
#         plt.savefig(save_pth + '-phase.pdf')
#         plt.savefig(save_pth + '-phase.svg')
#         plt.savefig(save_pth + '-phase.png')
    
#     # plt.close(fig)
    
    
# def plot_result_tspan(
#     tspan, test_x, pred_x, 
#     pred_label: str,
#     color: str = 'salmon',
#     save_pth: str = None
# ):
#     fig_num = test_x.shape[1]
#     fig = plt.figure(figsize = (8 * fig_num, 6))
#     if fig_num > 3: arrange = (2, ceil(fig_num / 2))
#     else: arrange = (1, fig_num)
    
#     lw, pad = 2.2, 10
#     legend_fs, label_fs, tick_fs = 13, 15, 13
    
#     for idx in range(fig_num):
#         plt.subplot(arrange[0], arrange[1], idx + 1)
#         plt.plot(
#             tspan, test_x[:, idx], label = 'Ground Truth',
#             linestyle = '-', lw = lw, color = 'black',
#         )
#         plt.plot(
#             tspan, pred_x[:, idx], label = pred_label,
#             linestyle = '--', lw = lw, color = color,
#         )
#         plt.xlabel('$t$', fontsize = label_fs, labelpad = pad)
#         plt.ylabel(f'$x_{idx+1}$', fontsize = label_fs, labelpad = pad)
#         plt.tick_params(axis = 'both', labelsize = tick_fs)
#         plt.legend(fontsize = legend_fs)
#         plt.grid(color = 'grey', linestyle = '--')

#     if save_pth:
#         plt.savefig(save_pth + '-tspan.pdf')
#         plt.savefig(save_pth + '-tspan.svg')
#         plt.savefig(save_pth + '-tspan.png')
    
#     # plt.close(fig)