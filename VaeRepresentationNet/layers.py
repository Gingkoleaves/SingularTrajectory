import torch
import torch.nn as nn
import numpy as np
from numpy.random import rand
import torch.nn.functional as F
from torch import Tensor
import math

# 提供位置编码
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 1000):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        return x + self.pe[:x.size(1)]

# 
class PerAgentEncoder(nn.Module):
    """单个行人轨迹编码器"""
    def __init__(self, input_dim=2, d_model=8, nhead=4, num_layers=1):
        super().__init__()
        self.d_model = d_model        
        # 线性投影层
        self.linear_in = nn.Linear(input_dim, d_model)        
        # 时间编码
        self.pos_encoding = PositionalEncoding(d_model)        
        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 聚合层（使用CLS token或平均池化）
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))

    def forward(self, x: Tensor) -> Tensor:
        # x: (batch_size, seq_len, 2)
        
        batch_size=x.shape[0]
        seq_len= x.shape[1]
        
        # 线性投影
        # print("before x.size=",x.shape) #[9547,8,2]
        x_proj = self.linear_in(x) 
        # print("after linear x_proj.size=",x_proj.shape) #[9547,8, 128]
        # 添加时间
        x_proj = self.pos_encoding(x_proj)# 【batch_size,seqlen,d_model】
        # print("after time x_proj.size=",x_proj.shape) # [9547,8,128]
        
        # 添加CLS token
        cls_tokens = self.cls_token.expand(batch_size,-1,-1)
        x_with_cls = torch.cat([cls_tokens, x_proj], dim=1)
        # print("after cls x_with_cls.size=",x_with_cls.shape) # [9547,9,128]
        
        # Transformer编码
        encoded = self.transformer_encoder(x_with_cls)
        
        # print("encoded.size=",encoded.shape) # [9547,9,128]
        # 取CLS token作为该行人的表示
        agent_embedding = encoded[:, 0]  # (batch_size, d_model)
        
        return agent_embedding

class SceneTransformerVAE(nn.Module):
    """多行人场景级Transformer VAE"""
    def __init__(self, d_model=16, nhead=2, num_layers=1, latent_dim=8):
        super().__init__()
        self.d_model = d_model
        self.latent_dim = latent_dim
        
        # 行人轨迹编码器
        self.agent_encoder = PerAgentEncoder(d_model=d_model)
        
        # 场景级Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, batch_first=True
        )
        self.scene_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # VAE隐变量生成
        self.fc_mu = nn.Linear(d_model, latent_dim)
        self.fc_logvar = nn.Linear(d_model, latent_dim)
        self.pos_encoding=PositionalEncoding(d_model)
        
        # 解码器 - 使用TransformerDecoder
        self.decoder_proj = nn.Linear(latent_dim, d_model)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead, batch_first=True, dropout=0.1
        )
        self.traj_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        
        # 轨迹重建输出层
        self.output_layer = nn.Sequential(
            nn.Linear(d_model, 8),
            nn.ReLU(),
            nn.Linear(8, 2)  # 输出x,y坐标
        )

    def encode(self, trajs: list) -> tuple:
        """
        编码多个行人轨迹
        trajs: (batch_size, seq_len, 2)
        """        
        # 编码每个行人轨迹
        agent_embed = self.agent_encoder(trajs)  # (batch_size, d_model)
        """
        agent_embed.squeeze(0)
        encoded = self.scene_encoder(
            agent_embed
        )        
        # 生成隐变量
        encoded.unsqueeze(0)
        """
        mu = self.fc_mu(agent_embed)  # (batch_size, latent_dim)
        logvar = self.fc_logvar(agent_embed)
        
        return mu, logvar

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: Tensor, target_length: int) -> list:
        """
        从隐变量解码行人轨迹
        z: (batch_size, latent_dim)
        target_length: 要生成的轨迹长度
        """
        batch_size, _ = z.shape
        
        # 投影到decoder维度
        decoder_input = self.decoder_proj(z)  # (batch_size, d_model)  
        
         # 为每个时间步创建位置编码
        time_embeddings = self.pos_encoding.pe[:target_length].unsqueeze(0)
        time_embeddings = time_embeddings.expand(batch_size,-1,-1)
        
        # 准备decoder输入
        decoder_input = decoder_input.unsqueeze(1)
        decoder_input = decoder_input.expand(-1, target_length, -1)
        
        # Transformer解码
        decoded = self.traj_decoder( 
            tgt=decoder_input,
            memory=time_embeddings
        )
        
        # 输出轨迹点
        reconstructed = self.output_layer(decoded)  # (batch_size,target_length,d_model)  
        reconstructed.reshape(-1,target_length,2)        
        return reconstructed

    def forward(self, obs_trajs: list, gt_trajs: list, beta: float = 5.0) -> tuple:
        """
        obs_trajs: 观测轨迹列表 [每个行人: (batch_size, 8, 2)]
        gt_trajs: 真实轨迹列表 [每个行人: (batch_size, 12, 2)]
        """
        # 编码
        mu, logvar = self.encode(obs_trajs)
        z = self.reparameterize(mu, logvar)
        
        # 解码
        recon_trajs = self.decode(z, gt_trajs.shape[-2])
        
        # 计算损失
        recon_loss = 0
        for recon, gt in zip(recon_trajs, gt_trajs):
            recon_loss += F.mse_loss(recon, gt)
        
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        total_loss = recon_loss + beta * kl_loss
        
        return recon_trajs, total_loss, recon_loss, kl_loss

    def sample(self, obs_trajs: list, target_length: int = 12) -> list:
        """从学习到的分布中采样轨迹"""
        with torch.no_grad():
            mu, logvar = self.encode(obs_trajs)
            z = self.reparameterize(mu, logvar)
            sampled_trajs = self.decode(z, target_length)
            return sampled_trajs

        
#encoder：包含linear层->transformer-block->得到每个层的表示->vae
#decoder：位置编码+encoder得到的路径表示进入decoder->[batch,traj_len,d_model]->[batch,traj_len,2]
