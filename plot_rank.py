from plot_utils import compute_svd_series, plot_sv_series
import matplotlib.pyplot as plt
import numpy as np
import os
# from scipy.linalg import svdvals


weights_dict = np.load('log_new/cifar10/2023-8-16 12:10_fullmodel_epoch50_cosineLR_vit_timm_lr5e-05_bs256/vit_timm_patch4_fc.npz', allow_pickle=True)
# figs_dir_all = 'log_new/cifar100/2023-8-17 11:25_fullmodel_epoch40_cosineLR_vit_timm_lr5e-05_bs256/figs_fc'
# if not os.path.exists(figs_dir_all):
#     os.makedirs(figs_dir_all)


# idx_to_name = ['Query', 'Key', 'Value']
idx_to_name = ['proj', 'fc1', 'fc2']

def compute_rank(weights):
    """
    param weights: (3, 768, 768)
    return: effective_rank, stable_rank [3*(num_epochs, ), 3*(num_epochs, )]
    """
    # print(weights.shape)
    # epochs = weights.shape[1]
    epochs = weights[0].shape[0]
    stable_ranks, effective_ranks = np.zeros((3, epochs)), np.zeros((3, epochs))
    for i in range(3):
        # print(idx_to_name[i])
        # print(weights[i].shape)
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



# weights_all = [[] for _ in range(12)]
# stable_ranks_proj, effective_ranks_proj = [[] for _ in range(12)], [[] for _ in range(12)]
# stable_ranks_fc1, effective_ranks_fc1 = [[] for _ in range(12)], [[] for _ in range(12)]
# stable_ranks_fc2, effective_ranks_fc2 = [[] for _ in range(12)], [[] for _ in range(12)]
stable_ranks_all, effective_ranks_all =[[] for _ in range(12)], [[] for _ in range(12)]
# each layer: W(0)
for layer_idx in range(12):
    print(f'layer {layer_idx}')


    print('weight')
    # weights_all = []
    
    proj_weight = weights_dict[f'module.blocks.{layer_idx}.attn.proj.weight']
    fc1_weight = weights_dict[f'module.blocks.{layer_idx}.mlp.fc1.weight']
    fc2_weight = weights_dict[f'module.blocks.{layer_idx}.mlp.fc2.weight']
    weights_all = [proj_weight[[-1]]-proj_weight[0], fc1_weight[[-1]]-fc1_weight[0], fc2_weight[[-1]]-fc2_weight[0]]

    # qkv_weight = weights_dict[f'module.blocks.{layer_idx}.attn.qkv.weight']
    # qkv_weights_all = qkv_weight[-1]-qkv_weight[0]
    # weights_all = [qkv_weights_all[:, :768], qkv_weights_all[:, 768:2*768], qkv_weights_all[:, 2*768:]]

    # stable_ranks, effective_ranks = compute_rank(np.stack(weights_all, axis=0))
    stable_rank, effective_rank = compute_rank(weights_all)
    stable_ranks_all[layer_idx].append(stable_rank)
    effective_ranks_all[layer_idx].append(effective_rank)

    
    # exit()

    # print('weight diffnn+1')
    # weights_all_delta = []
    # for i in range(3):
    #     weights_all_delta.append(weights_all[i][1:] - weights_all[i][:-1])
    # # stable_ranks_delta, effective_ranks_delta = compute_rank(np.stack(weights_all_delta, axis=0))
    # stable_ranks_delta, effective_ranks_delta = compute_rank(weights_all_delta)


stable_ranks_all = np.concatenate(stable_ranks_all, axis=0)
effective_ranks_all = np.concatenate(effective_ranks_all, axis=0)
# print(stable_ranks_all.shape, effective_ranks_all.shape)

idx_to_name = ['proj', 'fc1', 'fc2']
# print(stable_ranks_all.shape)
# idx_to_name = ['Query', 'Key', 'Value']

print('fc1', stable_ranks_all[:, 1, 0])
print('fc2', stable_ranks_all[:, 2, 0])

# import matplotlib.pyplot as plt
# plt.figure()
# for i in range(3):
#     plt.plot(stable_ranks_all[:, i, 0], label=idx_to_name[i])
#     plt.xlabel('Depth')
#     plt.ylabel('Stable Rank')
#     plt.legend()
# plt.savefig(os.path.join(figs_dir_all, 'diff01_stable_rank.png'))

# plt.figure()
# for i in range(3):
#     plt.plot(effective_ranks_all[:, i, 0], label=idx_to_name[i])
#     plt.xlabel('Depth')
#     plt.ylabel('Effective Rank')
#     plt.legend()
# plt.savefig(os.path.join(figs_dir_all, 'diff01_effective_rank.png'))

# plt.figure()
# for i in range(3):
#     plt.plot(stable_ranks_all[:, i, 1], label=idx_to_name[i])
#     plt.xlabel('Depth')
#     plt.ylabel('Stable Rank')
#     plt.legend()
# plt.savefig(os.path.join(figs_dir_all, 'diff025_stable_rank.png'))

# plt.figure()
# for i in range(3):
#     plt.plot(effective_ranks_all[:, i, 1], label=idx_to_name[i])
#     plt.xlabel('Depth')
#     plt.ylabel('Effective Rank')
#     plt.legend()
# plt.savefig(os.path.join(figs_dir_all, 'diff025_effective_rank.png'))




