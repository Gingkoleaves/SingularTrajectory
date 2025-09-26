import torch
import torch.nn as nn
from .anchor import AdaptiveAnchor
from .space import SingularSpace

class SingularTrajectory(nn.Module):
    r"""The SingularTrajectory model

    Args:
        baseline_model (nn.Module): The baseline model
        hook_func (dict): The bridge functions for the baseline model
        hyper_params (DotDict): The hyper-parameters
    """

    def __init__(self, baseline_model, hook_func,hyper_params,args):
        super().__init__()

        self.baseline_model = baseline_model
        self.hook_func = hook_func
        self.hyper_params = hyper_params
        self.t_obs, self.t_pred = hyper_params.obs_len, hyper_params.pred_len
        self.obs_svd, self.pred_svd = hyper_params.obs_svd, hyper_params.pred_svd
        self.k = hyper_params.k
        self.s = hyper_params.num_samples
        self.dim = hyper_params.traj_dim
        self.static_dist = hyper_params.static_dist
        
        self.attention = nn.MultiheadAttention(embed_dim=self.k, num_heads=1, batch_first=True)

        self.Singular_space_m = SingularSpace(hyper_params=hyper_params,args=args, ms=1, norm_sca=True)
        self.Singular_space_s = SingularSpace(hyper_params=hyper_params,args=args, ms=0, norm_sca=False)
        self.adaptive_anchor_m = AdaptiveAnchor(hyper_params=hyper_params, vae_model=self.Singular_space_m.vae_model)
        self.adaptive_anchor_s = AdaptiveAnchor(hyper_params=hyper_params, vae_model=self.Singular_space_s.vae_model)

    def calculate_parameters(self, obs_traj, pred_traj):
        r"""Generate the Sinuglar space of the SingularTrajectory model

        Args:
            obs_traj (torch.Tensor): The observed trajectory
            pred_traj (torch.Tensor): The predicted trajectory

        Note:
            This function should be called once before training the model.
        """

        # Mask out static trajectory
        mask = self.calculate_mask(obs_traj)
        obs_m_traj, pred_m_traj = obs_traj[mask], pred_traj[mask]
        obs_s_traj, pred_s_traj = obs_traj[~mask], pred_traj[~mask]

        # Descriptor initialization
        data_m = self.Singular_space_m.parameter_initialization(obs_m_traj, pred_m_traj)
        data_s = self.Singular_space_s.parameter_initialization(obs_s_traj, pred_s_traj)

        # Anchor initialization
        self.adaptive_anchor_m.anchor_initialization(*data_m)
        self.adaptive_anchor_s.anchor_initialization(*data_s)
    
    def calculate_adaptive_anchor(self, dataset):
        obs_traj, pred_traj = dataset.obs_traj, dataset.pred_traj
        scene_id = dataset.scene_id
        vector_field = dataset.vector_field
        homography = dataset.homography

        # Mask out static trajectory
        mask = self.calculate_mask(obs_traj)
        obs_m_traj, scene_id_m = obs_traj[mask], scene_id[mask]
        obs_s_traj, scene_id_s = obs_traj[~mask], scene_id[~mask]
        
        n_ped = pred_traj.size(0)
        anchor = torch.zeros((n_ped, self.k, self.s), dtype=torch.float)
        anchor[mask] = self.adaptive_anchor_m.adaptive_anchor_calculation(obs_m_traj, scene_id_m, vector_field, homography, self.Singular_space_m)
        anchor[~mask] = self.adaptive_anchor_s.adaptive_anchor_calculation(obs_s_traj, scene_id_s, vector_field, homography, self.Singular_space_s)

        return anchor

    def calculate_mask(self, obs_traj):
        if obs_traj.size(1) <= 2:
            mask = (obs_traj[:, -1] - obs_traj[:, -2]).div(1).norm(p=2, dim=-1) > self.static_dist
        else:
            mask = (obs_traj[:, -1] - obs_traj[:, -3]).div(2).norm(p=2, dim=-1) > self.static_dist
        return mask

    def nolinear_combination(self, input_data):
        r"""对输入数据做一次非线性变换"""
        # print(input_data.shape,'\n')
        # 假设 input_data shape 为 (self.k, n_ped)
        input_data = input_data.permute(1, 0)  # 变为 (n_ped, self.k)
        # 非线性变换
        output = torch.relu(self.linear(input_data).to(input_data.device))
        # print(output.shape)
        output = output.permute(1, 0)  # 变为 (self.k,n_ped)
        return output
    
    def forward(self, obs_traj, adaptive_anchor, pred_traj=None, addl_info=None):
        r"""The forward function of the SingularTrajectory model

        Args:
            obs_traj (torch.Tensor): The observed trajectory
            pred_traj (torch.Tensor): The predicted trajectory (optional, for training only)
            addl_info (dict): The additional information (optional, if baseline model requires)

        Returns:
            output (dict): The output of the model (recon_traj, loss, etc.)
        """

        n_ped = obs_traj.size(0)

        # Filter out static trajectory
        mask = self.calculate_mask(obs_traj)
        obs_m_traj = obs_traj[mask]
        obs_s_traj = obs_traj[~mask]
        pred_m_traj_gt = pred_traj[mask] if pred_traj is not None else None
        pred_s_traj_gt = pred_traj[~mask] if pred_traj is not None else None

        # Projection
        # print("before projection pred_s_traj_gt.shape=",pred_s_traj_gt.shape) ([195, 12, 2])
        C_m_obs, C_m_pred_gt = self.Singular_space_m.projection(obs_m_traj, pred_m_traj_gt)
        C_s_obs, C_s_pred_gt = self.Singular_space_s.projection(obs_s_traj, pred_s_traj_gt)
        # print("after projection C_s_obs.shape=",C_s_obs.shape)  [8,195]
        C_obs = torch.zeros((self.k, n_ped), dtype=torch.float, device=obs_traj.device)
        C_obs[:, mask], C_obs[:, ~mask] = C_m_obs, C_s_obs

        # Absolute coordinate
        obs_m_ori = self.Singular_space_m.traj_normalizer.traj_ori.squeeze(dim=1).T
        obs_s_ori = self.Singular_space_s.traj_normalizer.traj_ori.squeeze(dim=1).T
        obs_ori = torch.zeros((2, n_ped), dtype=torch.float, device=obs_traj.device)
        obs_ori[:, mask], obs_ori[:, ~mask] = obs_m_ori, obs_s_ori
        obs_ori -= obs_ori.mean(dim=1, keepdim=True)

        ### Adaptive anchor per agent
        # print("before permute adaptive_anchor.shape=",adaptive_anchor.shape)  # torch.Size([512, 8, 20])
        C_anchor = adaptive_anchor.permute(1, 0, 2)
        addl_info["anchor"] = C_anchor.detach().clone()
        # print("after permute C_anchor.shape=",C_anchor.shape) # torch.Size([8, 512, 20])

        # Trajectory prediction
        
        # 对 C_obs 进行自注意力
        # C_obs: [k, n_ped] -> [n_ped, k] for attention, attention input (batch,seq_len,latent_dim)
        C_obs_t = C_obs.T.unsqueeze(0)  # [1, n_ped, k]
        attn_out, _ = self.attention(C_obs_t, C_obs_t, C_obs_t)
        C_obs = attn_out.squeeze(0).T  # [k, n_ped]
        # print("before diffusion.shape=",C_obs.shape) #[4,512]
        
        # C_obs=nolinear_combination(C_obs)
        input_data = self.hook_func.model_forward_pre_hook(C_obs, obs_ori, addl_info)
        output_data = self.hook_func.model_forward(input_data, self.baseline_model)
        C_pred_refine = self.hook_func.model_forward_post_hook(output_data, addl_info) * 0.1

        # print("C_pred_refine.shape=",C_pred_refine.shape) # torch.Size([8, 517, 20])
        C_m_pred = self.adaptive_anchor_m(C_pred_refine[:, mask], C_anchor[:, mask])
        C_s_pred = self.adaptive_anchor_s(C_pred_refine[:, ~mask], C_anchor[:, ~mask])

        # Reconstruction
        # print("before reconstruction C_m_pred.shape=",C_m_pred.shape) # torch.Size([8, 206, 20])
        pred_m_traj_recon = self.Singular_space_m.reconstruction(C_m_pred)
        pred_s_traj_recon = self.Singular_space_s.reconstruction(C_s_pred)
        pred_traj_recon = torch.zeros((self.s, n_ped, self.t_pred, self.dim), dtype=torch.float, device=obs_traj.device)
        pred_traj_recon[:, mask], pred_traj_recon[:, ~mask] = pred_m_traj_recon, pred_s_traj_recon

        output = {"recon_traj": pred_traj_recon}

        if pred_traj is not None:
            C_pred = torch.zeros((self.k, n_ped, self.s), dtype=torch.float, device=obs_traj.device)
            C_pred[:, mask], C_pred[:, ~mask] = C_m_pred, C_s_pred

            # Low-rank approximation for gt trajectory
            C_pred_gt = torch.zeros((self.k, n_ped), dtype=torch.float, device=obs_traj.device)
            C_pred_gt[:, mask], C_pred_gt[:, ~mask] = C_m_pred_gt, C_s_pred_gt
            C_pred_gt = C_pred_gt.detach()

            # Loss calculation
            error_coefficient = (C_pred - C_pred_gt.unsqueeze(dim=-1)).norm(p=2, dim=0)
            error_displacement = (pred_traj_recon - pred_traj.unsqueeze(dim=0)).norm(p=2, dim=-1)
            output["loss_eigentraj"] = error_coefficient.min(dim=-1)[0].mean()
            output["loss_euclidean_ade"] = error_displacement.mean(dim=-1).min(dim=0)[0].mean()
            output["loss_euclidean_fde"] = error_displacement[:, :, -1].min(dim=0)[0].mean()

        return output
