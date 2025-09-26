# 使用训练后的vae，采样路径观察latent表示
# 观察相同路径的svd表示

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))  # 添加当前目录到 Python 路径


from VaeRepresentationNet.VaeModel import SimulaSVDReductioner
from SingularTrajectory.space import SingularSpace
from utils.utils import augment_trajectory

import numpy as np
import os
import argparse
import baseline
import torch

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))  # 添加当前目录到 Python 路径

from utils import *
from tqdm import tqdm

static_dist=0.3

def calculate_mask( obs_traj):
    if obs_traj.size(1) <= 2:
        mask = (obs_traj[:, -1] - obs_traj[:, -2]).div(1).norm(p=2, dim=-1) > static_dist
    else:
        mask = (obs_traj[:, -1] - obs_traj[:, -3]).div(2).norm(p=2, dim=-1) > static_dist
    return mask



import matplotlib.pyplot as plt
import itertools
import numpy as np
# 假设你的数据
# obs_svd.shape = (5000, 4)
# obs_vae.shape = (5000, 4)

NUM=10

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

def plot_4d_3dcolor(obs_svd, obs_vae, save_path="3dcolor.png", max_points=200):
    fig = plt.figure(figsize=(12,6))
    ax = fig.add_subplot(111, projection='3d')

    # 随机挑选 max_points 个点
    idx = np.random.choice(len(obs_svd), size=min(max_points, len(obs_svd)), replace=False)

    for k in idx:
        # 画 SVD 点 (蓝色圆圈)
        ax.scatter(obs_svd[k,0], obs_svd[k,1], obs_svd[k,2], c='blue', marker='o', s=30)
        # 画 VAE 点 (红色叉号)
        ax.scatter(obs_vae[k,0], obs_vae[k,1], obs_vae[k,2], c='red', marker='x', s=30)
        # 画连线
        ax.plot([obs_svd[k,0], obs_vae[k,0]],
                [obs_svd[k,1], obs_vae[k,1]],
                [obs_svd[k,2], obs_vae[k,2]], c='gray', alpha=0.3)

    ax.set_xlabel("dim1")
    ax.set_ylabel("dim2")
    ax.set_zlabel("dim3")
    ax.set_title("SVD vs VAE (color by correspondence)")

    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"✅ 保存成功: {save_path}")


def plot_4d_parallel(obs_svd, obs_vae, save_path="parallel.png", num_samples=10):
    """
    前 num_samples 个样本的 4D 平行坐标图，SVD vs VAE 一一对应
    """
    obs_svd = obs_svd[:num_samples]
    obs_vae = obs_vae[:num_samples]
    dims = ["dim1", "dim2", "dim3", "dim4"]

    plt.figure(figsize=(10,6))

    for i in range(num_samples):
        # 同一个样本：SVD 蓝线，VAE 红线
        plt.plot(dims, obs_svd[i], color='blue', linestyle='-', marker='o', alpha=0.7)
        plt.plot(dims, obs_vae[i], color='red', linestyle='--', marker='x', alpha=0.7)

        # 在最后一个维度标注样本编号，方便对照
        plt.text(dims[-1], obs_svd[i][-1], f"S{i}", color='blue', fontsize=8)
        plt.text(dims[-1], obs_vae[i][-1], f"V{i}", color='red', fontsize=8)

    plt.title(f"前 {num_samples} 个样本的 4D 对应关系 (SVD vs VAE)")
    plt.xlabel("维度")
    plt.ylabel("数值")
    plt.grid(alpha=0.3)
    plt.tight_layout()

    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"✅ 保存成功: {save_path}")


def plot_4d_pairs(obs_svd, obs_vae, save_path="pairs.png", max_points=10):
    """
    obs_svd, obs_vae: shape (N, 4)
    save_path: 保存路径
    max_points: 最多绘制多少对点（避免5000对太乱，可以取子集）
    """
    dims = [0,1,2,3]
    pairs = list(itertools.combinations(dims, 2))  # 所有维度对
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    # 随机挑选 max_points 个点
    idx = np.random.choice(len(obs_svd), size=min(max_points, len(obs_svd)), replace=False)

    for ax, (i,j) in zip(axes, pairs):
        for k in idx:
            # SVD 点
            ax.scatter(obs_svd[k,i], obs_svd[k,j], c='blue', marker='o', s=30)
            # VAE 点
            ax.scatter(obs_vae[k,i], obs_vae[k,j], c='red', marker='x', s=30)
            # 连线
            ax.plot([obs_svd[k,i], obs_vae[k,i]], [obs_svd[k,j], obs_vae[k,j]], c='gray', alpha=0.3)

        ax.set_xlabel(f"dim {i+1}")
        ax.set_ylabel(f"dim {j+1}")
        ax.set_title(f"dim{i+1} vs dim{j+1}")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"✅ 保存成功: {save_path}")

import matplotlib.pyplot as plt

def plot_reconstructed_trajectories(pred_m_traj, obs_svd_recon_m_traj, obs_vae_recon_m_traj, save_path="recon_traj.png", num_samples=6):
    """
    绘制前 num_samples 条轨迹的真实路径、SVD 重建路径、VAE 重建路径对比
    """
    plt.figure(figsize=(18, 10))

    for i in range(num_samples):
        ax = plt.subplot(2, 3, i+1)

        # 真实路径（黑色实线）
        ax.plot(pred_m_traj[i,:,0], pred_m_traj[i,:,1], color="black", linestyle="-", marker="o", label="真实路径")

        # SVD 重建（蓝色虚线）
        ax.plot(obs_svd_recon_m_traj[i,:,0], obs_svd_recon_m_traj[i,:,1], color="blue", linestyle="--", marker="x", label="SVD 重建")

        # VAE 重建（红色点线）
        ax.plot(obs_vae_recon_m_traj[i,:,0], obs_vae_recon_m_traj[i,:,1], color="red", linestyle="-.", marker="s", label="VAE 重建")

        ax.set_title(f"样本 {i}")
        ax.set_xlabel("X 坐标")
        ax.set_ylabel("Y 坐标")
        ax.legend()
        ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"✅ 保存成功: {save_path}")
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', default="./config/singulartrajectory-transformerdiffusion-zara1.json", type=str, help="config file path")
    parser.add_argument('--tag', default="SingularTrajectory-TEMP", type=str, help="personal tag for the model")
    parser.add_argument('--gpu_id', default="0", type=str, help="gpu id for the model")
    parser.add_argument('--test', default=False, action='store_true', help="evaluation mode")
    args = parser.parse_args()

    print("===== Arguments =====")
    print_arguments(vars(args))

    print("===== Configs =====")
    hyper_params = get_exp_config(args.cfg)
    print_arguments(hyper_params)

    dataset_dir = hyper_params.dataset_dir + hyper_params.dataset + '/'
    obs_len, pred_len, skip = hyper_params.obs_len, hyper_params.pred_len, hyper_params.skip
    
    loader_train = get_dataloader(dataset_dir, 'train', obs_len, pred_len, batch_size=1, skip=skip)
    # loader_val = get_dataloader(dataset_dir, 'val', obs_len, pred_len, batch_size=1)
    # loader_test = get_dataloader(dataset_dir, 'test', obs_len, pred_len, batch_size=1)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    

    vae=SimulaSVDReductioner(hyper_params)
    svdm=SingularSpace(hyper_params,args)
    
    obs_traj, pred_traj = loader_train.dataset.obs_traj, loader_train.dataset.pred_traj
    obs_traj, pred_traj = augment_trajectory(obs_traj, pred_traj)    
    
    # Mask out static trajectory
    mask = calculate_mask(obs_traj)
    obs_m_traj, pred_m_traj = obs_traj[mask], pred_traj[mask]
    obs_s_traj, pred_s_traj = obs_traj[~mask], pred_traj[~mask]

    # Descriptor initialization
    svdm.parameter_initialization(obs_m_traj, pred_m_traj)   
    print("obs_traj.shape=",obs_traj.shape)
    obs_svd,pred_svd = svdm.projection(obs_traj,pred_traj)
    # obs_svd=obs_svd.T.detach().numpy()
    obs_svd=obs_svd.T
    print("obs_svd.shape=",obs_svd.shape)#【50602,4】
    
    vae.load_state_dict(torch.load("/root/SingularTrajectory_Vae/checkpoints/k4_simuSVD/zara2/m_vae_model_best.pth"))
    obs_vae=vae.encode_space(obs_traj)
    print("obs_vae.shape=",obs_vae.shape)#【50602,4】
    
    # plot_4d_pairs(obs_svd, obs_vae)
    # plot_4d_3dcolor(obs_svd, obs_vae, save_path="svd_vae_3d_linked.png", max_points=300)
    # plot_4d_parallel(obs_svd, obs_vae, save_path="svd_vae_parallel.png", num_samples=10)
    
    obs_svd=obs_svd.unsqueeze(0).permute(2,1,0)
    print("obs_svd.shape=",obs_svd.shape)
    obs_svd_recon_m_traj=svdm.reconstruction(obs_svd)
    obs_vae_recon_m_traj=vae.decode_pred(obs_vae)
    
    obs_svd_recon_m_traj=obs_svd_recon_m_traj.squeeze(0)
    obs_vae_recon_m_traj=obs_vae_recon_m_traj.reshape(-1,12,2)
    print("obs_svd_recon_m_traj.shape=",obs_svd_recon_m_traj.shape)
    print("obs_vae_recon_m_traj.shape=",obs_vae_recon_m_traj.shape)
    
    plot_reconstructed_trajectories(pred_m_traj.detach().numpy(), obs_svd_recon_m_traj.detach().numpy(), obs_vae_recon_m_traj.detach().numpy())