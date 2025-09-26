from tqdm import tqdm
import torch
import torch.nn as nn
import os
from .VaeModel import VaeDimenReductioner,SharedVaeDimenReductioner,SimulaSVDReductioner,EnhanceVaeReductioner

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))  # 添加当前目录到 Python 路径

from utils.dataloader import get_dataloader

class VaeTrainer:
    r"""A class for training the VAE model for trajectory dimension reduction."""

    def __init__(self,args, hyper_params,ms):
        # Dataset preprocessing
        self.hyper_params = hyper_params
        self.log = {'train_loss': [], 'val_loss': []}

        # Set up the training parameters
        batch_size = hyper_params.batch_size
        vae_lr=hyper_params.vae_lr
        obs_len, pred_len, skip = hyper_params.obs_len, hyper_params.pred_len, hyper_params.skip
        self.dataset_dir = hyper_params.dataset_dir + hyper_params.dataset + '/'
        self.checkpoint_dir = hyper_params.checkpoint_dir + '/' + args.tag + '/' + hyper_params.dataset + '/'

        self.loader_train = get_dataloader(self.dataset_dir, 'train', obs_len, pred_len, batch_size=batch_size, skip=skip)
        self.loader_val = get_dataloader(self.dataset_dir, 'val', obs_len, pred_len, batch_size=batch_size)
        self.loader_test = get_dataloader(self.dataset_dir, 'test', obs_len, pred_len, batch_size=1)

        # 指定 vae-model
        self.model=SimulaSVDReductioner(hyper_params=hyper_params).cuda()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=vae_lr)
        self.ms=ms

    def train(self, epoch):
        self.model.train()
        loss_batch = 0

        for cnt, batch in enumerate(tqdm(self.loader_train, desc=f'Train Epoch {epoch}', mininterval=1)):        
            obs_traj, pred_traj = batch["obs_traj"].cuda(non_blocking=True), batch["pred_traj"].cuda(non_blocking=True)

            # print(f"obs_traj shape: {obs_traj.shape}, pred_traj shape: {pred_traj.shape}")

            self.optimizer.zero_grad()

            obs_recon_obs, pred_recon_pred, obs_recon_pred = self.model(obs_traj,pred_traj)

            # Calculate the loss
            # 这里训练一个Vae，希望它们的latent-space对齐，即同一条轨迹的obs和pred的latent-space相似
            loss = self.model.loss(obs_traj, pred_traj, 
                                   obs_recon_obs=obs_recon_obs, 
                                   pred_recon_pred=pred_recon_pred,
                                   obs_recon_pred=obs_recon_pred
                                   )

            loss[torch.isnan(loss)] = 0
            loss_batch += loss.item()
            
            loss.backward()
            if self.hyper_params.clip_grad is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.hyper_params.clip_grad)

            self.optimizer.step()

        self.log['train_loss'].append(loss_batch / len(self.loader_train))
        
    @torch.no_grad()
    def valid(self, epoch):
        self.model.eval()
        loss_batch = 0

        for cnt, batch in enumerate(tqdm(self.loader_val, desc=f'Valid Epoch {epoch}', mininterval=1)):
            obs_traj, pred_traj = batch["obs_traj"].cuda(non_blocking=True), batch["pred_traj"].cuda(non_blocking=True)

            obs_recon_obs, pred_recon_pred, obs_recon_pred = self.model(obs_traj,pred_traj)

            recon_loss = self.model.loss(obs_traj, pred_traj, 
                                   obs_recon_obs=obs_recon_obs, 
                                   pred_recon_pred=pred_recon_pred,
                                   obs_recon_pred=obs_recon_pred
                                   )
            loss_batch += recon_loss.item()

        num_ped = sum(self.loader_val.dataset.num_peds_in_seq)
        self.log['val_loss'].append(loss_batch / num_ped)

    def fit(self):
        print("Vae Training started...")
        
        epochs= self.hyper_params.vae_s_epochs
        if self.ms==1:
            epochs=self.hyper_params.vae_m_epochs 
            
        for epoch in range(epochs):
            self.train(epoch)
            self.valid(epoch)
            
            print("\n")
            print(f"Epoch {epoch+1} - Train Loss: {self.log['train_loss'][-1]:.4f}")  # 显式打印
            
            # Save the best model
            if epoch == 0 or self.log['val_loss'][-1] < min(self.log['val_loss'][:-1]):
                self.save_model()            

        print("Vae Done.")
        
    def save_model(self, filename='vae_model_best.pth'):
        if self.ms==1:
            filename="m_"+filename
        else :
            filename="s_"+filename
            
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
        model_path = self.checkpoint_dir + filename
        torch.save(self.model.state_dict(), model_path)   

    def load_model(self, filename='vae_model_best.pth'):
        if self.ms==1:
            filename="m_"+filename
        else:
            filename="s_"+filename
        model_path = self.checkpoint_dir + filename
        self.model.load_state_dict(torch.load(model_path))