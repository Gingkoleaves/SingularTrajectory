import torch
import torch.nn as nn
from .kmeans import BatchKMeans
from sklearn.cluster import KMeans
import numpy as np
from .homography import image2world, world2image


class AdaptiveAnchor(nn.Module):
    r"""Adaptive anchor model

    Args:
        hyper_params (DotDict): The hyper-parameters
    """

    def __init__(self, hyper_params, vae_model):
        super().__init__()

        self.hyper_params = hyper_params
        self.k = hyper_params.k
        self.s = hyper_params.num_samples
        self.dim = hyper_params.traj_dim
        self.obs_len = hyper_params.obs_len
        self.pred_len = hyper_params.pred_len
        self.vae_model = vae_model

        self.C_anchor = nn.Parameter(torch.zeros((self.k, self.s)))

    def to_Singular_space(self, traj, evec):
        r"""Transform Euclidean trajectories to Singular space coordinates

        Args:
            traj (torch.Tensor): The trajectory to be transformed
            evec (torch.Tensor): The Singular space basis vectors

        Returns:
            C (torch.Tensor): The Singular space coordinates"""

        if self.hyper_params.vae_train:
            # Use the VAE model to encode the trajectory
            traj = self.vae_model.encode_space(traj.cuda())
            C = traj
        else:
            # Euclidean space -> Singular space
            tdim = evec.size(0)
            M = traj.reshape(-1, tdim).T
            C = evec.T.detach() @ M
        
        return C

    def batch_to_Singular_space(self, traj, evec):
        r"""Transform a batch of Euclidean trajectories to Singular space coordinates

        Args:
            traj (torch.Tensor): The batch of trajectories to be transformed
            evec (torch.Tensor): The Singular space basis vectors

        Returns:
            C (torch.Tensor): The Singular space coordinates
        """

        if self.hyper_params.vae_train:
            # Use the VAE model to encode the trajectory

            traj = self.vae_model.encode_space(traj.cuda())
            C = traj
        else:
            # Euclidean space -> Singular space
            tdim = evec.size(0)
            M = traj.reshape(traj.size(0), traj.size(1), tdim).transpose(1, 2)
            C = evec.T.detach() @ M
        return C
    def to_Euclidean_space(self, C, evec):
        r"""Transform Singular space coordinates to Euclidean trajectories

        Args:
            C (torch.Tensor): The Singular space coordinates
            evec (torch.Tensor): The Singular space basis vectors

        Returns:
            traj (torch.Tensor): The Euclidean trajectory"""

        if self.hyper_params.vae_train:
            # Use the VAE model to decode the trajectory
            t = evec.size(0) // self.dim
            if evec.size(0) == self.t_obs * self.dim:
                traj = self.vae_model.decode_obs(C)
            else:
                traj = self.vae_model.decode_pred(C)
            return traj.reshape(-1,t,self.dim)
        else:
            # Singular space -> Euclidean
            t = evec.size(0) // self.dim
            M = evec.detach() @ C
            traj = M.T.reshape(-1, t, self.dim)
            return traj
    
    def batch_to_Euclidean_space(self, C, evec):
        r"""Transform a batch of Singular space coordinates to Euclidean trajectories

        Args:
            C (torch.Tensor): The batch of Singular space coordinates

        Returns:
            traj (torch.Tensor): The batch of Euclidean trajectories
        """
        # C:[20,4,batch]
        if self.hyper_params.vae_train:
            b = C.size(0)
            t = evec.size(0) // self.dim            
            C=C.permute(2,0,1).reshape(-1,self.k)
            
            # Use the VAE model to decode the trajectory
            if evec.size(0) == self.t_obs * self.dim:
                traj = self.vae_model.decode_obs(C)
            else:
                traj = self.vae_model.decode_pred(C)
            return traj.reshape(b,-1,t,self.dim)
        else:
            # Singular space -> Euclidean
            b = C.size(0)
            t = evec.size(0) // self.dim
            M = evec.detach() @ C
            traj = M.transpose(1, 2).reshape(b, -1, t, self.dim)
            return traj

    def anchor_initialization(self, pred_traj_norm, V_pred_trunc):
        r"""Anchor initialization on Singular space

        Args:
            pred_traj_norm (torch.Tensor): The normalized predicted trajectory
            V_pred_trunc (torch.Tensor): The truncated Singular space basis vectors of the predicted trajectory
        Note:
            This function should be called once before training the model.
        """        
        # print("pred_traj_norm.shape=",pred_traj_norm.shape) # ([19148, 12, 2])
        if self.hyper_params.vae_train:
            # Use the VAE model to encode the trajectory
            C_pred = self.vae_model.encode_space(pred_traj_norm.cuda()).detach().cpu().numpy()
        else:
            # Trajectory projection
            C_pred = self.to_Singular_space(pred_traj_norm, evec=V_pred_trunc).T.detach().numpy()

        C_anchor = torch.FloatTensor(KMeans(n_clusters=self.s, random_state=0, init='k-means++', n_init=1).fit(C_pred).cluster_centers_.T)

        # print("C_anchor.shape=",C_anchor.shape) # [4,20]
        
        # Register anchors as model parameters
        self.C_anchor = nn.Parameter(C_anchor.to(self.C_anchor.device))

    def adaptive_anchor_calculation(self, obs_traj, scene_id, vector_field, homography, space):
        r"""Adaptive anchor calculation on Singular space"""

        n_ped = obs_traj.size(0)
        V_trunc = space.V_trunc
        
        space.traj_normalizer.calculate_params(obs_traj.cuda().detach())
        init_anchor = self.C_anchor.unsqueeze(dim=0).repeat_interleave(repeats=n_ped, dim=0).detach()
        init_anchor = init_anchor.permute(2, 1, 0)
        
        # print("init_anchor.shape=",init_anchor.shape,"\n evec.shape=",V_trunc.shape) # [20,4,10131] [24,6]
        
        init_anchor_euclidean = space.batch_to_Euclidean_space(init_anchor, evec=V_trunc)
        init_anchor_euclidean = space.traj_normalizer.denormalize(init_anchor_euclidean).detach().cpu().numpy()
        adaptive_anchor_euclidean = init_anchor_euclidean.copy()
        obs_traj = obs_traj.cpu().numpy()
        
        for ped_id in range(n_ped):
            scene_name = scene_id[ped_id]
            prototype_image = world2image(init_anchor_euclidean[:, ped_id], homography[scene_name])
            startpoint_image = world2image(obs_traj[ped_id], homography[scene_name])[-1]
            endpoint_image = prototype_image[:, -1, :]
            endpoint_image = np.round(endpoint_image).astype(int)
            size = np.array(vector_field[scene_name].shape[1::-1]) // 2
            endpoint_image = np.clip(endpoint_image, a_min= -size // 2, a_max=size + size // 2 -1)
            for s in range(self.s):
                vector = np.array(vector_field[scene_name][endpoint_image[s, 1] + size[1] // 2, endpoint_image[s, 0] + size[0] // 2])[::-1] - size // 2
                if vector[0] == endpoint_image[s, 0] and vector[1] == endpoint_image[s, 1]:
                    continue
                else:
                    nearest_endpoint_image = vector
                    scale_xy = (nearest_endpoint_image - startpoint_image) / (endpoint_image[s] - startpoint_image)
                    prototype_image[s, :, :] = (prototype_image[s, :, :].copy() - startpoint_image) * scale_xy + startpoint_image
                
            prototype_world = image2world(prototype_image, homography[scene_name])
            adaptive_anchor_euclidean[:, ped_id] = prototype_world
        
        adaptive_anchor_euclidean = space.traj_normalizer.normalize(torch.FloatTensor(adaptive_anchor_euclidean).cuda())
        
        # print("before adaptive_anchor.shape=",adaptive_anchor_euclidean.shape) # torch.Size([20, 10131, 12, 2])
        # print("evec.size=",V_trunc.shape) # torch.Size([20, 10131, 12, 2])
        adaptive_anchor = space.batch_to_Singular_space(adaptive_anchor_euclidean, evec=V_trunc)
        # print("after adaptive_anchor.shape=",adaptive_anchor.shape) # torch.Size([20, 4, 10131])
        adaptive_anchor = adaptive_anchor.permute(2, 1, 0).cpu()
        # If you don't want to use an image, return `init_anchor`.
        return adaptive_anchor

    def forward(self, C_residual, C_anchor):
        r"""Anchor refinement on Singular space

        Args:
            C_residual (torch.Tensor): The predicted Singular space coordinates

        Returns:
            C_pred_refine (torch.Tensor): The refined Singular space coordinates
        """
        
        C_pred_refine = C_anchor.detach() + C_residual
        return C_pred_refine
