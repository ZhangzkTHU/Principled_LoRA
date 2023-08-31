from plot_utils import compute_svd_series, plot_sv_series
import matplotlib.pyplot as plt
import numpy as np
import os
# from scipy.linalg import svdvals


weights_dict = np.load('log_new/cifar100/2023-8-17 11:25_fullmodel_epoch40_cosineLR_vit_timm_lr5e-05_bs256/vit_timm_patch4_fc.npz', allow_pickle=True)
# figs_dir_all = 'log_new/cifar10/2023-8-16 12:10_fullmodel_epoch50_cosineLR_vit_timm_lr5e-05_bs256/figs_qkv'
# if not os.path.exists(figs_dir_all):
#     os.makedirs(figs_dir_all)

# for key, value in weights_dict.items():
#     print(key, value.shape)
# exit()

# idx_to_name = ['Query', 'Key', 'Value']
idx_to_name = ['proj', 'fc1', 'fc2']

def compute_rank(weights):
    """
    param weights: (3, num_epochs, 768, 768)
    return: effective_rank, stable_rank [3*(num_epochs, ), 3*(num_epochs, )]
    """
    # print(weights.shape)
    # epochs = weights.shape[1]
    epochs = weights[0].shape[0]
    stable_ranks, effective_ranks = np.zeros((3, epochs)), np.zeros((3, epochs))
    for i in range(3):
        print(idx_to_name[i])
        weight = weights[i]  # (num_epochs, 768, 768)
        for j in range(epochs):
            print(f'epoch {j}')
            _, s, _ = np.linalg.svd(weight[j], full_matrices=False)
            # print(s.shape)
            s_normalized = s / s.sum()
            stable_rank = (s**2).sum() / (s.max()**2)
            effective_rank = np.exp(-np.sum(s_normalized * np.log(s_normalized + 1e-16)))
            stable_ranks[i][j] = stable_rank
            effective_ranks[i][j] = effective_rank
            print(f'stable rank: {stable_rank}, effective rank: {effective_rank}')
    return stable_ranks, effective_ranks


for layer_idx in range(12):
    print(f'layer {layer_idx}')


    print('weight')
    weights_all = []
    # module.blocks.3.attn.qkv.weight (25, 2304, 768)
    for name, weight in weights_dict.items():
        # if name == f'module.blocks.{layer_idx}.attn.qkv.weight':
        if name in [f'module.blocks.{layer_idx}.attn.proj.weight', 
                    f'module.blocks.{layer_idx}.mlp.fc1.weight', 
                    f'module.blocks.{layer_idx}.mlp.fc2.weight']:
            weights_all.append(weight)
            # weights_all.append(weight[:, :768])  # (25,768,768)
            # weights_all.append(weight[:, 768:2*768])
            # weights_all.append(weight[:, 2*768:])

    # stable_ranks, effective_ranks = compute_rank(np.stack(weights_all, axis=0))
    stable_ranks, effective_ranks = compute_rank(weights_all)
    
    # exit()

    # print('weight diffnn+1')
    # weights_all_delta = []
    # for i in range(3):
    #     weights_all_delta.append(weights_all[i][1:] - weights_all[i][:-1])
    # # stable_ranks_delta, effective_ranks_delta = compute_rank(np.stack(weights_all_delta, axis=0))
    # stable_ranks_delta, effective_ranks_delta = compute_rank(weights_all_delta)

    print('weight diff0n')
    weights_all_diff = []
    for i in range(3):
        weights_all_diff.append(weights_all[i][1:] - weights_all[i][0])
    # stable_ranks_diff, effective_ranks_diff = compute_rank(np.stack(weights_all_diff, axis=0))
    stable_ranks_diff, effective_ranks_diff = compute_rank(weights_all_diff)



# for layer_idx in range(12):
#     print(f'layer {layer_idx}')
#     figs_dir = os.path.join(figs_dir_all, f'transformer_layer{layer_idx}')
#     if not os.path.exists(figs_dir):
#         os.makedirs(figs_dir)

#     weights_all = []
#     for name, weight in weights_dict.items():
#         if f'blocks.{layer_idx}' in name:
#             print(name, weight.shape)
#             weights_all.append(weight[:, :768])
#             weights_all.append(weight[:, 768:2*768])
#             weights_all.append(weight[:, 2*768:])

#     series_list = []
#     for i in [0, 1, 2]:
#         series_list.append(compute_svd_series(weights_all[i], rank=10))

#     sample_step = 15
#     sigval_num = 60


#     fig = plt.figure(constrained_layout=True, figsize=(21, 24))
#     subfigs = fig.subfigures(nrows=3, ncols=1)
#     for i, subfig in enumerate(subfigs):
#         print(idx_to_name[i], 'weight')
#         sval_series, right_series, left_series = series_list[i]
#         for j in range(sval_series.shape[0]):
#             print(f'epoch{j}', sval_series[j, :sigval_num])
#         subfig.suptitle(idx_to_name[i], fontsize=40, weight='bold', y=0.92)
#         ax = subfig.add_subplot(131, projection='3d')
#         plot_sv_series(ax, sval_series, sample_step=sample_step)

#         ax.set_xlabel('\nSV Index ($i$)', fontsize=20)
#         ax.set_ylabel('\nIteration ($t$)', fontsize=20)
#         ax.set_zlabel('\n$\sigma_i(t)$', fontsize=20)

#         ax.set_title('Singular Values', fontsize=25, y=0.0)

#         ax = subfig.add_subplot(132, projection='3d')

#         plot_sv_series(ax, right_series, color='inferno', sample_step=sample_step)

#         ax.set_xlabel('\nSV Index ($i$)', fontsize=20)
#         ax.set_ylabel('\nIteration ($t$)', fontsize=20)
#         ax.set_zlabel('\n' + r'$\angle(v_i(t), v_i(0))$', fontsize=20)

#         ax.set_title('Right Singular Vectors', fontsize=25, y=0.0)

#         ax = subfig.add_subplot(133, projection='3d')

#         plot_sv_series(ax, left_series, color='cividis', sample_step=sample_step)

#         ax.set_xlabel('\nSV Index ($i$)', fontsize=20)
#         ax.set_ylabel('\nIteration ($t$)', fontsize=20)
#         ax.set_zlabel('\n' + r'$\angle(u_i(t), u_i(0))$', fontsize=20)

#         ax.set_title('Left Singular Vectors', fontsize=25, y=0.0)

#     plt.tight_layout()
#     # plt.savefig(f'./{figs_dir}/{type}_{init_scale}_bn{bn}_subsampled{sample_step}.png')
#     plt.savefig(f'./{figs_dir}/subsampled{sample_step}.png')
#     plt.close()


#     fig = plt.figure(constrained_layout=True, figsize=(21, 24))
#     subfigs = fig.subfigures(nrows=3, ncols=1)
#     for i, subfig in enumerate(subfigs):

#         sval_series, right_series, left_series = series_list[i]
#         sval_series, right_series, left_series = sval_series[:, :sigval_num], right_series[:, :sigval_num], left_series[:, :sigval_num]
#         subfig.suptitle(idx_to_name[i], fontsize=40, weight='bold', y=0.92)
#         ax = subfig.add_subplot(131, projection='3d')
#         plot_sv_series(ax, sval_series)

#         ax.set_xlabel('\nSV Index ($i$)', fontsize=20)
#         ax.set_ylabel('\nIteration ($t$)', fontsize=20)
#         ax.set_zlabel('\n$\sigma_i(t)$', fontsize=20)

#         ax.set_title('Singular Values', fontsize=25, y=0.0)

#         ax = subfig.add_subplot(132, projection='3d')

#         plot_sv_series(ax, right_series, color='inferno')

#         ax.set_xlabel('\nSV Index ($i$)', fontsize=20)
#         ax.set_ylabel('\nIteration ($t$)', fontsize=20)
#         ax.set_zlabel('\n' + r'$\angle(v_i(t), v_i(0))$', fontsize=20)

#         ax.set_title('Right Singular Vectors', fontsize=25, y=0.0)

#         ax = subfig.add_subplot(133, projection='3d')

#         plot_sv_series(ax, left_series, color='cividis')

#         ax.set_xlabel('\nSV Index ($i$)', fontsize=20)
#         ax.set_ylabel('\nIteration ($t$)', fontsize=20)
#         ax.set_zlabel('\n' + r'$\angle(u_i(t), u_i(0))$', fontsize=20)

#         ax.set_title('Left Singular Vectors', fontsize=25, y=0.0)

#     plt.tight_layout()
#     # plt.savefig(f'./{figs_dir}/{type}_{init_scale}_bn{bn}_first{sigval_num}.png')
#     plt.savefig(f'./{figs_dir}/first{sigval_num}.png')
#     plt.close()

#     series_list = []
#     for i in [0, 1, 2]:
#         weight = weights_all[i]
#         weight = np.stack(weight, axis=0)
#         weight = weight[1:] - weight[:-1]
#         series_list.append(compute_svd_series(weight, rank=10))

#     fig = plt.figure(constrained_layout=True, figsize=(21, 24))
#     subfigs = fig.subfigures(nrows=3, ncols=1)
#     for i, subfig in enumerate(subfigs):
        
#         print(idx_to_name[i], 'weight diffnn+1')
#         sval_series, right_series, left_series = series_list[i]
#         for j in range(sval_series.shape[0]):
#             print(f'epoch{i}', sval_series[j, :sigval_num])
#         subfig.suptitle(idx_to_name[i], fontsize=40, weight='bold', y=0.92)
#         ax = subfig.add_subplot(131, projection='3d')
#         plot_sv_series(ax, sval_series, sample_step=sample_step)

#         ax.set_xlabel('\nSV Index ($i$)', fontsize=20)
#         ax.set_ylabel('\nIteration ($t$)', fontsize=20)
#         ax.set_zlabel('\n$\sigma_i(t)$', fontsize=20)

#         ax.set_title('Singular Values', fontsize=25, y=0.0)

#         ax = subfig.add_subplot(132, projection='3d')

#         plot_sv_series(ax, right_series, color='inferno', sample_step=sample_step)

#         ax.set_xlabel('\nSV Index ($i$)', fontsize=20)
#         ax.set_ylabel('\nIteration ($t$)', fontsize=20)
#         ax.set_zlabel('\n' + r'$\angle(v_i(t), v_i(0))$', fontsize=20)

#         ax.set_title('Right Singular Vectors', fontsize=25, y=0.0)

#         ax = subfig.add_subplot(133, projection='3d')

#         plot_sv_series(ax, left_series, color='cividis', sample_step=sample_step)

#         ax.set_xlabel('\nSV Index ($i$)', fontsize=20)
#         ax.set_ylabel('\nIteration ($t$)', fontsize=20)
#         ax.set_zlabel('\n' + r'$\angle(u_i(t), u_i(0))$', fontsize=20)

#         ax.set_title('Left Singular Vectors', fontsize=25, y=0.0)

#     plt.tight_layout()
#     # plt.savefig(f'./{figs_dir}/{type}_{init_scale}_bn{bn}_diffnn+1_subsampled{sample_step}.png')
#     plt.savefig(f'./{figs_dir}/diffnn+1_subsampled{sample_step}.png')
#     plt.close()

#     fig = plt.figure(constrained_layout=True, figsize=(21, 24))
#     subfigs = fig.subfigures(nrows=3, ncols=1)
#     for i, subfig in enumerate(subfigs):

#         sval_series, right_series, left_series = series_list[i]
#         sval_series, right_series, left_series = sval_series[:, :sigval_num], right_series[:, :sigval_num], left_series[:, :sigval_num]
#         subfig.suptitle(idx_to_name[i], fontsize=40, weight='bold', y=0.92)
#         ax = subfig.add_subplot(131, projection='3d')
#         plot_sv_series(ax, sval_series)

#         ax.set_xlabel('\nSV Index ($i$)', fontsize=20)
#         ax.set_ylabel('\nIteration ($t$)', fontsize=20)
#         ax.set_zlabel('\n$\sigma_i(t)$', fontsize=20)

#         ax.set_title('Singular Values', fontsize=25, y=0.0)

#         ax = subfig.add_subplot(132, projection='3d')

#         plot_sv_series(ax, right_series, color='inferno')

#         ax.set_xlabel('\nSV Index ($i$)', fontsize=20)
#         ax.set_ylabel('\nIteration ($t$)', fontsize=20)
#         ax.set_zlabel('\n' + r'$\angle(v_i(t), v_i(0))$', fontsize=20)

#         ax.set_title('Right Singular Vectors', fontsize=25, y=0.0)

#         ax = subfig.add_subplot(133, projection='3d')

#         plot_sv_series(ax, left_series, color='cividis')

#         ax.set_xlabel('\nSV Index ($i$)', fontsize=20)
#         ax.set_ylabel('\nIteration ($t$)', fontsize=20)
#         ax.set_zlabel('\n' + r'$\angle(u_i(t), u_i(0))$', fontsize=20)

#         ax.set_title('Left Singular Vectors', fontsize=25, y=0.0)

#     plt.tight_layout()
#     # plt.savefig(f'./{figs_dir}/{type}_{init_scale}_bn{bn}_diffnn+1_first{sigval_num}.png')
#     plt.savefig(f'./{figs_dir}/diffnn+1_first{sigval_num}.png')
#     plt.close()


#     series_list = []
#     for i in [0, 1, 2]:
#         weight = weights_all[i]
#         weight = np.stack(weight, axis=0)
#         weight = weight[1:] - weight[0]
#         series_list.append(compute_svd_series(weight, rank=10))

#     fig = plt.figure(constrained_layout=True, figsize=(21, 24))
#     subfigs = fig.subfigures(nrows=3, ncols=1)
#     for i, subfig in enumerate(subfigs):
    
#         print(idx_to_name[i], 'weight diff0n')

#         sval_series, right_series, left_series = series_list[i]
#         for j in range(sval_series.shape[0]):
#             print(f'epoch{j}', sval_series[j, :sigval_num])
#         subfig.suptitle(idx_to_name[i], fontsize=40, weight='bold', y=0.92)
#         ax = subfig.add_subplot(131, projection='3d')
#         plot_sv_series(ax, sval_series, sample_step=sample_step)

#         ax.set_xlabel('\nSV Index ($i$)', fontsize=20)
#         ax.set_ylabel('\nIteration ($t$)', fontsize=20)
#         ax.set_zlabel('\n$\sigma_i(t)$', fontsize=20)

#         ax.set_title('Singular Values', fontsize=25, y=0.0)

#         ax = subfig.add_subplot(132, projection='3d')

#         plot_sv_series(ax, right_series, color='inferno', sample_step=sample_step)

#         ax.set_xlabel('\nSV Index ($i$)', fontsize=20)
#         ax.set_ylabel('\nIteration ($t$)', fontsize=20)
#         ax.set_zlabel('\n' + r'$\angle(v_i(t), v_i(0))$', fontsize=20)

#         ax.set_title('Right Singular Vectors', fontsize=25, y=0.0)

#         ax = subfig.add_subplot(133, projection='3d')

#         plot_sv_series(ax, left_series, color='cividis', sample_step=sample_step)

#         ax.set_xlabel('\nSV Index ($i$)', fontsize=20)
#         ax.set_ylabel('\nIteration ($t$)', fontsize=20)
#         ax.set_zlabel('\n' + r'$\angle(u_i(t), u_i(0))$', fontsize=20)

#         ax.set_title('Left Singular Vectors', fontsize=25, y=0.0)

#     plt.tight_layout()
#     # plt.savefig(f'./{figs_dir}/{type}_{init_scale}_bn{bn}_diff0n_subsampled{sample_step}.png')
#     plt.savefig(f'./{figs_dir}/diff0n_subsampled{sample_step}.png')
#     plt.close()

#     fig = plt.figure(constrained_layout=True, figsize=(21, 24))
#     subfigs = fig.subfigures(nrows=3, ncols=1)
#     for i, subfig in enumerate(subfigs):

#         sval_series, right_series, left_series = series_list[i]
#         sval_series, right_series, left_series = sval_series[:, :sigval_num], right_series[:, :sigval_num], left_series[:, :sigval_num]
#         subfig.suptitle(idx_to_name[i], fontsize=40, weight='bold', y=0.92)
#         ax = subfig.add_subplot(131, projection='3d')
#         plot_sv_series(ax, sval_series)

#         ax.set_xlabel('\nSV Index ($i$)', fontsize=20)
#         ax.set_ylabel('\nIteration ($t$)', fontsize=20)
#         ax.set_zlabel('\n$\sigma_i(t)$', fontsize=20)

#         ax.set_title('Singular Values', fontsize=25, y=0.0)

#         ax = subfig.add_subplot(132, projection='3d')

#         plot_sv_series(ax, right_series, color='inferno')

#         ax.set_xlabel('\nSV Index ($i$)', fontsize=20)
#         ax.set_ylabel('\nIteration ($t$)', fontsize=20)
#         ax.set_zlabel('\n' + r'$\angle(v_i(t), v_i(0))$', fontsize=20)

#         ax.set_title('Right Singular Vectors', fontsize=25, y=0.0)

#         ax = subfig.add_subplot(133, projection='3d')

#         plot_sv_series(ax, left_series, color='cividis')

#         ax.set_xlabel('\nSV Index ($i$)', fontsize=20)
#         ax.set_ylabel('\nIteration ($t$)', fontsize=20)
#         ax.set_zlabel('\n' + r'$\angle(u_i(t), u_i(0))$', fontsize=20)

#         ax.set_title('Left Singular Vectors', fontsize=25, y=0.0)

#     plt.tight_layout()
#     # plt.savefig(f'./{figs_dir}/{type}_{init_scale}_bn{bn}_diff0n_first{sigval_num}.png')
#     plt.savefig(f'./{figs_dir}/diff0n_first{sigval_num}.png')
#     plt.close()