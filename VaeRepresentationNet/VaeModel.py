import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torch import Tensor
from .layers import PerAgentEncoder,PositionalEncoding
from .Vae_utili import Extender

class VaeDimenReductioner(nn.Module):
    r"""A class for reducing the dimension of trajectories using VAE.

    Args:
        hyper_params (DotDict): The hyper-parameters
    """

    def __init__(self, seq_len, hyper_params):
        super().__init__()
        self.hyper_params = hyper_params
        self.seq_len = seq_len
        # self.t_obs, self.t_pred = hyper_params.obs_len, hyper_params.pred_len
        self.dim = hyper_params.traj_dim
        self.latent_dim = hyper_params.k

        # Define the input dimension based on the trajectory length and dimension
        self.input_dim  = (self.seq_len) * self.dim
        self.output_dim = (self.seq_len) * self.dim

        # encoder
        self.fc1 = nn.Linear(self.input_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc_mu = nn.Linear(32, self.latent_dim)
        self.fc_logvar = nn.Linear(32, self.latent_dim)

        # decoder
        self.fc3 = nn.Linear(self.latent_dim, 64)
        self.fc4 = nn.Linear(64, 32)

        # Output layer should match the input dimension
        self.fc_out = nn.Linear(32, self.output_dim)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        h1 = F.relu(self.fc2(h1))
        mu = self.fc_mu(h1)
        logvar = self.fc_logvar(h1)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        h3 = F.relu(self.fc4(h3))
        out = self.fc_out(h3)
        return out

    def forward(self, x):
        # 展平为 (batch, features)
        x = x.reshape(-1, self.input_dim)
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        return recon_x, mu, logvar, z

    def loss(self, x, recon_x,  mu, logvar):
        r"""Calculate the VAE loss function

        Args:
            recon_x (torch.Tensor): The reconstructed trajectory
            x (torch.Tensor): The original trajectory
            mu (torch.Tensor): The mean of the latent space
            logvar (torch.Tensor): The log variance of the latent space

        Returns:
            loss (torch.Tensor): The VAE loss value
        """
        BCE = F.mse_loss(recon_x, x.reshape(-1, self.output_dim), reduction='sum')
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return BCE + KLD

    def encode_space(self, x):
        r"""Encode the trajectory into the latent space
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return z
    
class SharedVaeDimenReductioner(nn.Module):
    r"""A class for reducing the dimension of trajectories using a shared VAE.
    Args:
        hyper_params (DotDict): The hyper-parameters
    """
    def __init__(self, hyper_params):
        super().__init__()
        self.hyper_params = hyper_params
        self.t_obs, self.t_pred = hyper_params.obs_len, hyper_params.pred_len

        self.dim = hyper_params.traj_dim
        self.latent_dim = hyper_params.k
        self.alpha=hyper_params.alpha

        self.conv_layers = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.lstm = nn.LSTM(32, 32)  # LSTM
        self.fc_mu = nn.Linear(32, self.latent_dim)          # 双向输出拼接后维度为128
        self.fc_logvar = nn.Linear(32, self.latent_dim)

        # decoder_obs
        self.fc3_obs = nn.Linear(self.latent_dim, 32)
        self.fc4_obs = nn.Linear(32, 16)        
        self.fc_out_obs = nn.Linear(16, self.t_obs*self.dim)

        # decoder_pred
        self.fc3_pred = nn.Linear(self.latent_dim, 32)
        self.fc4_pred = nn.Linear(32, 16)
        self.fc_out_pred = nn.Linear(16, self.t_pred*self.dim)

    def encode(self, x):
        x = x.flatten(start_dim=-2).unsqueeze(1)
        x = self.conv_layers(x)
        
        x = x.permute(2, 0, 1)                          # (seq_len*dim, B, 32)
        _, (h_n, _) = self.lstm(x)                      # h_n: (2, B, 64)
        h_n = h_n.permute(1, 0, 2).flatten(1)   
        
        mu = self.fc_mu(h_n)
        logvar = self.fc_logvar(h_n)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode_obs(self, z):
        h3 = F.relu(self.fc3_obs(z))
        h3 = F.relu(self.fc4_obs(h3))
        out = self.fc_out_obs(h3)
        return out

    def decode_pred(self, z):
        h3 = F.relu(self.fc3_pred(z))
        h3 = F.relu(self.fc4_pred(h3))
        out = self.fc_out_pred(h3)
        return out

    def forward(self, obs, pred):
        # 展平为 (batch, features)
        mu_obs, logvar_obs = self.encode(obs)        
        mu_pred, logvar_pred = self.encode(pred)

        z_obs = self.reparameterize(mu_obs, logvar_obs)
        z_pred = self.reparameterize(mu_pred, logvar_pred)

        obs_recon_obs = self.decode_obs(z_obs)
        pred_recon_pred = self.decode_pred(z_pred)
        obs_recon_pred = self.decode_pred(z_obs)

        obs_recon_obs = {'traj': obs_recon_obs.reshape(-1, self.t_obs* self.dim), 'mu': mu_obs, 'logvar': logvar_obs}
        pred_recon_pred = {'traj': pred_recon_pred.reshape(-1, self.t_pred* self.dim), 'mu': mu_pred, 'logvar': logvar_pred}
        obs_recon_pred = {'traj': obs_recon_pred.reshape(-1, self.t_pred* self.dim)}

        # 返回重构的轨迹和潜在空间的均值和方差
        return obs_recon_obs, pred_recon_pred, obs_recon_pred
    
    def kl_div(self, mu_q, logvar_q, mu_p, logvar_p):
        kl = logvar_p - logvar_q - 1 + (logvar_q.exp() + (mu_q - mu_p).pow(2)) / logvar_p.exp()
        return 0.5 * kl.sum()
    
    def loss(self, obs, pred, obs_recon_obs, pred_recon_pred, obs_recon_pred):
        r"""Calculate the VAE loss function

        Args:
            recon_x (torch.Tensor): The reconstructed trajectory
            x (torch.Tensor): The original trajectory
            mu (torch.Tensor): The mean of the latent space
            logvar (torch.Tensor): The log variance of the latent space

        Returns:
            loss (torch.Tensor): The VAE loss value
        """
        BCE_obs_self = F.mse_loss(obs_recon_obs['traj'], obs.reshape(-1, self.t_obs * self.dim), reduction='sum')
        BCE_pred_self = F.mse_loss(pred_recon_pred['traj'], pred.reshape(-1, self.t_pred * self.dim), reduction='sum')
        BCE_pred_obs = F.mse_loss(obs_recon_pred['traj'], pred.reshape(-1, self.t_pred * self.dim), reduction='sum')
        BCE = BCE_obs_self/3 + BCE_pred_self/3 + BCE_pred_obs/3

        KLD_obs_self  = -0.5 * torch.sum(1 + obs_recon_obs['logvar'] - obs_recon_obs['mu'].pow(2) - obs_recon_obs['logvar'].exp())
        KLD_pred_self = -0.5 * torch.sum(1 + pred_recon_pred['logvar'] - pred_recon_pred['mu'].pow(2) - pred_recon_pred['logvar'].exp())
        # KLD= self.kl_div(obs_recon_obs['mu'],obs_recon_obs['logvar'], pred_recon_pred['mu'], pred_recon_pred['logvar'])
        KLD=KLD_obs_self/2+KLD_pred_self/2
        
        # print("BCE-loss=",BCE,",KLD-loss=",KLD)
        return BCE + self.alpha*KLD

    def encode_space(self, x):
        r"""Encode the trajectory into the latent space
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return z
    
    def freeze(self):
        for param in self.parameters():
            param.requires_grad_(False)
            
class SimulaSVDReductioner(nn.Module):
    r"""A class for reducing the dimension of trajectories using a shared VAE.
    Args:
        hyper_params (DotDict): The hyper-parameters
    """
    def __init__(self, hyper_params):
        super().__init__()
        self.hyper_params = hyper_params
        self.t_obs, self.t_pred = hyper_params.obs_len, hyper_params.pred_len

        self.dim = hyper_params.traj_dim
        self.latent_dim = hyper_params.k
        self.alpha=hyper_params.alpha
        self.beta=hyper_params.beta

        self.conv_layers = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.lstm=nn.LSTM(32,32)
        # self.att=nn.MultiheadAttention(32,2,batch_first=True)
        self.fc_mu = nn.Linear(32, self.latent_dim)          
        self.fc_logvar = nn.Linear(32, self.latent_dim)

        # decoder_obs
        self.fc3_obs = nn.Linear(self.latent_dim, 32)
        self.fc4_obs = nn.Linear(32, 16)        
        self.fc_out_obs = nn.Linear(16, self.t_obs*self.dim)

        # decoder_pred
        self.fc3_pred = nn.Linear(self.latent_dim, 32)
        self.fc4_pred = nn.Linear(32, 16)
        self.fc_out_pred = nn.Linear(16, self.t_pred*self.dim)
                
        self.V_obs_trunc = nn.Parameter(torch.zeros((self.t_obs * self.dim, self.latent_dim)))
        self.V_pred_trunc = nn.Parameter(torch.zeros((self.t_pred * self.dim, self.latent_dim)))
        
    def Para_init(self,obs_trunc,pred_trunc):
        self.V_obs_trunc.cuda()
        self.V_pred_trunc.cuda()
        self.V_obs_trunc= nn.Parameter(obs_trunc.to(self.V_obs_trunc.device))          
        self.V_pred_trunc = nn.Parameter(pred_trunc.to(self.V_pred_trunc.device))

    def encode(self, x):
        x = x.flatten(start_dim=-2).unsqueeze(1)
        x = self.conv_layers(x)    
        
        x = x.permute(2, 0, 1)                          # (seq_len*dim, B, 32)
        _, (h_n, _) = self.lstm(x)                      # h_n: (2, B, 64)
        h_n = h_n.permute(1, 0, 2).flatten(1)  
        
        # h_n,_=self.att(h_n,h_n,h_n)       
        
        # print("encode x.shape=",x.shape)# 517 32 16
        mu = self.fc_mu(h_n)
        logvar = self.fc_logvar(h_n)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode_obs(self, z):
        h3 = F.relu(self.fc3_obs(z))
        h3 = F.relu(self.fc4_obs(h3))
        out = self.fc_out_obs(h3)
        return out

    def decode_pred(self, z):
        h3 = F.relu(self.fc3_pred(z))
        h3 = F.relu(self.fc4_pred(h3))
        out = self.fc_out_pred(h3)
        return out

    def forward(self, obs, pred):
        # 展平为 (batch, features)
        mu_obs, logvar_obs = self.encode(obs)        
        mu_pred, logvar_pred = self.encode(pred)

        z_obs = self.reparameterize(mu_obs, logvar_obs)
        z_pred = self.reparameterize(mu_pred, logvar_pred)

        obs_recon_obs = self.decode_obs(z_obs)
        pred_recon_pred = self.decode_pred(z_pred)
        obs_recon_pred = self.decode_pred(z_obs)

        obs_recon_obs = {'traj': obs_recon_obs.reshape(-1, self.t_obs* self.dim), 'mu': mu_obs, 'logvar': logvar_obs}
        pred_recon_pred = {'traj': pred_recon_pred.reshape(-1, self.t_pred* self.dim), 'mu': mu_pred, 'logvar': logvar_pred}
        obs_recon_pred = {'traj': obs_recon_pred.reshape(-1, self.t_pred* self.dim)}

        # 返回重构的轨迹和潜在空间的均值和方差
        return obs_recon_obs, pred_recon_pred, obs_recon_pred    

    def svd_repr(self,traj,evec):
        # print("evec.shape=",evec.shape) #[16,8]
        # print("traj.shape=",traj.shape) #[8192,16]
        C = traj @ evec.detach() 
        return C
    
    def loss(self, obs, pred, obs_recon_obs, pred_recon_pred, obs_recon_pred):
        r"""Calculate the VAE loss function

        Args:
            recon_x (torch.Tensor): The reconstructed trajectory
            x (torch.Tensor): The original trajectory
            mu (torch.Tensor): The mean of the latent space
            logvar (torch.Tensor): The log variance of the latent space

        Returns:
            loss (torch.Tensor): The VAE loss value
        """      
        svd_obs=self.svd_repr(obs.reshape(-1, self.t_obs * self.dim),self.V_obs_trunc)
        svd_pred=self.svd_repr(pred.reshape(-1, self.t_pred * self.dim),self.V_pred_trunc)
        # print("mu.shape=",obs_recon_obs['mu'].shape) #[512,8]
        MSE=F.mse_loss(obs_recon_obs['mu'],svd_obs)/2+F.mse_loss(pred_recon_pred['mu'],svd_pred)/2
                
        # print("obs_recon_obs.shape=",obs_recon_obs['traj'].shape) #[16,8]
        # print("obs.shape=",obs.shape) #[16,8]
        BCE_obs_self = F.mse_loss(obs_recon_obs['traj'], obs.reshape(-1, self.t_obs * self.dim), reduction='sum')
        BCE_pred_self = F.mse_loss(pred_recon_pred['traj'], pred.reshape(-1, self.t_pred * self.dim), reduction='sum')
        BCE_pred_obs = F.mse_loss(obs_recon_pred['traj'], pred.reshape(-1, self.t_pred * self.dim), reduction='sum')
        BCE = BCE_obs_self/3 + BCE_pred_self/3 + BCE_pred_obs/3

        KLD_obs_self  = -0.5 * torch.sum(1 + obs_recon_obs['logvar'] - obs_recon_obs['mu'].pow(2) - obs_recon_obs['logvar'].exp())
        KLD_pred_self = -0.5 * torch.sum(1 + pred_recon_pred['logvar'] - pred_recon_pred['mu'].pow(2) - pred_recon_pred['logvar'].exp())
        # KLD= self.kl_div(obs_recon_obs['mu'],obs_recon_obs['logvar'], pred_recon_pred['mu'], pred_recon_pred['logvar'])
        KLD=KLD_obs_self/2+KLD_pred_self/2
        
        # print("BCE-loss=",BCE,",KLD-loss=",KLD)
        return BCE + self.alpha*KLD+self.beta*MSE

    def encode_space(self, x):
        r"""Encode the trajectory into the latent space
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return z
    
    def freeze(self):
        for param in self.parameters():
            param.requires_grad_(False)
  
            
class EnhanceVaeReductioner(nn.Module):
    r"""A class for reducing the dimension of trajectories using a shared VAE.
    Args:
        hyper_params (DotDict): The hyper-parameters
    """
    def __init__(self, hyper_params):
        super().__init__()
        self.hyper_params = hyper_params
        self.t_obs, self.t_pred = hyper_params.obs_len, hyper_params.pred_len

        self.dim = hyper_params.traj_dim
        self.pri_latent_dim=8
        self.latent_dim = hyper_params.k
        
        self.alpha=hyper_params.alpha
        self.beta=hyper_params.beta

        
        self.d_model = 8
        self.nhead = 4
        self.num_layers = 3
        
        # 行人轨迹编码器
        self.agent_encoder = PerAgentEncoder(d_model= self.d_model)
        
        # 场景级Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model= self.d_model, nhead= self.nhead, batch_first=True
        )
        self.scene_encoder = nn.TransformerEncoder(encoder_layer, num_layers= self.num_layers)
        
        # VAE隐变量生成
        self.fc_mu = nn.Linear( self.d_model,  self.latent_dim)
        self.fc_logvar = nn.Linear( self.d_model,  self.latent_dim)
        self.pos_encoding=PositionalEncoding( self.d_model)
        
        # 解码器 - 使用TransformerDecoder
        self.decoder_proj = nn.Linear(self.latent_dim,  self.d_model)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model= self.d_model, nhead= self.nhead, batch_first=True, dropout=0.1
        )
        self.traj_decoder = nn.TransformerDecoder(decoder_layer, num_layers= self.num_layers)
        
        # 轨迹重建输出层
        self.output_layer = nn.Sequential(
            nn.Linear( self.d_model, 64),
            nn.ReLU(),
            nn.Linear(64, 2)  # 输出x,y坐标
        )    
        
        # decoder_obs
        self.fc3_obs = nn.Linear(self.latent_dim, 32)
        self.fc4_obs = nn.Linear(32, 16)        
        self.fc_out_obs = nn.Linear(16, self.t_obs*self.dim)

        # decoder_pred
        self.fc3_pred = nn.Linear(self.latent_dim, 32)
        self.fc4_pred = nn.Linear(32, 16)
        self.fc_out_pred = nn.Linear(16, self.t_pred*self.dim)
                
        self.V_obs_trunc = nn.Parameter(torch.zeros((self.t_obs * self.dim, self.latent_dim)))
        self.V_pred_trunc = nn.Parameter(torch.zeros((self.t_pred * self.dim, self.latent_dim)))
        
    def Para_init(self,obs_trunc,pred_trunc):
        self.V_obs_trunc.cuda()
        self.V_pred_trunc.cuda()
        self.V_obs_trunc= nn.Parameter(obs_trunc.to(self.V_obs_trunc.device))          
        self.V_pred_trunc = nn.Parameter(pred_trunc.to(self.V_pred_trunc.device))
   
    def encode(self, trajs):
        """
        编码多个行人轨迹
        trajs: (batch_size, seq_len, 2)
        """        
        # 编码每个行人轨迹
        agent_embed = self.agent_encoder(trajs)  # (batch_size, d_model)
        agent_embed.unsqueeze(0)
        encoded = self.scene_encoder(
            agent_embed
        )        
        # 生成隐变量
        encoded.squeeze(0)
        mu = self.fc_mu(encoded)  # (batch_size, latent_dim1)
        logvar = self.fc_logvar(encoded)
        
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    

    def decode(self, z: Tensor, target_length: int) -> list:
        """
        从隐变量解码行人轨迹
        z: (batch_size, latent_dim1=64)
        target_length: 要生成的轨迹长度
        """
        # print("z.shape=",z.shape) # z.shape= torch.Size([9574, 20, 64])
        batch_size = z.shape[-2]
        
        # 投影到decoder维度
        decoder_input = self.decoder_proj(z)  # (batch_size, d_model)  
        
         # 为每个时间步创建位置编码
        time_embeddings = self.pos_encoding.pe[:target_length].unsqueeze(0)
        time_embeddings = time_embeddings.expand(batch_size,-1,-1)
        
        # 准备decoder输入
        greatbatch=1
        if decoder_input.dim() == 2:
            decoder_input = decoder_input.unsqueeze(0)  # 需要赋值回原变量
            greatbatch = decoder_input.shape[0]
        else:
            greatbatch = decoder_input.shape[0]

        decoded = []
        for i in range(greatbatch):
            # 获取当前batch的decoder输入
            bat_decoder_input = decoder_input[i]  # [batch, d_model]

            # 扩展维度以适应decoder
            bat_decoder_input = bat_decoder_input.unsqueeze(1)  # [batch, 1, d_model]
            bat_decoder_input = bat_decoder_input.expand(-1, target_length, -1)  # [batch, target_length, d_model]

            # Transformer解码 - 注意这里应该使用bat_decoder_input而不是decoder_input
            bat_decoded = self.traj_decoder(
                tgt=bat_decoder_input,
                memory=time_embeddings
            )
            decoded.append(bat_decoded)
        
        # 输出轨迹点
        if len(decoded) == 1:
            decoded_tensor = decoded[0]  # 直接取第一个元素
        else:
            if decoder_input.shape[0]==0:
                return decoder_input
            decoded_tensor = torch.cat(decoded, dim=0)  # 拼接张量
            
        reconstructed = self.output_layer(decoded_tensor)  # (batch_size,target_length,d_model)  
        if(greatbatch):
            reconstructed.reshape(greatbatch,-1,target_length,2)
        else:
            reconstructed.reshape(-1,target_length,2)        
        return reconstructed

    def decode_obs(self, z):
        h3 = F.relu(self.fc3_obs(z))
        h3 = F.relu(self.fc4_obs(h3))
        out = self.fc_out_obs(h3)
        return out

    def decode_pred(self, z):
        h3 = F.relu(self.fc3_pred(z))
        h3 = F.relu(self.fc4_pred(h3))
        out = self.fc_out_pred(h3)
        return out
    
    """

    def decode_obs(self, z):
        return self.decode(z,self.t_obs)

    def decode_pred(self, z):
        return self.decode(z,self.t_pred)
        """
    
    def forward(self, obs, pred):
        """
        obs_trajs: 观测轨迹列表 [每个行人: (batch_size, 8, 2)]
        gt_trajs: 真实轨迹列表 [每个行人: (batch_size, 12, 2)]
        """
        # 编码
        mu_obs, logvar_obs = self.encode(obs)        
        mu_pred, logvar_pred = self.encode(pred)

        z_obs = self.reparameterize(mu_obs, logvar_obs)
        z_pred = self.reparameterize(mu_pred, logvar_pred)        
        
        obs_recon_obs = self.decode_obs(z_obs)
        pred_recon_pred = self.decode_pred(z_pred)
        obs_recon_pred = self.decode_pred(z_obs)

        obs_recon_obs = {'traj': obs_recon_obs.reshape(-1, self.t_obs* self.dim), 'mu': mu_obs, 'logvar': logvar_obs}
        pred_recon_pred = {'traj': pred_recon_pred.reshape(-1, self.t_pred* self.dim), 'mu': mu_pred, 'logvar': logvar_pred}
        obs_recon_pred = {'traj': obs_recon_pred.reshape(-1, self.t_pred* self.dim)}

        # 返回重构的轨迹和潜在空间的均值和方差
        return obs_recon_obs, pred_recon_pred, obs_recon_pred    


    def loss(self, obs, pred, obs_recon_obs, pred_recon_pred, obs_recon_pred):
        r"""Calculate the VAE loss function

        Args:
            recon_x (torch.Tensor): The reconstructed trajectory
            x (torch.Tensor): The original trajectory
            mu (torch.Tensor): The mean of the latent space
            logvar (torch.Tensor): The log variance of the latent space

        Returns:
            loss (torch.Tensor): The VAE loss value
        """
        
        # print("obs_recon_obs.shape=",obs_recon_obs['traj'].shape) #[16,8]
        # print("obs.shape=",obs.shape) #[16,8]
        BCE_obs_self = F.mse_loss(obs_recon_obs['traj'], obs.reshape(-1, self.t_obs * self.dim), reduction='sum')
        BCE_pred_self = F.mse_loss(pred_recon_pred['traj'], pred.reshape(-1, self.t_pred * self.dim), reduction='sum')
        BCE_pred_obs = F.mse_loss(obs_recon_pred['traj'], pred.reshape(-1, self.t_pred * self.dim), reduction='sum')
        BCE = BCE_obs_self/3 + BCE_pred_self/3 + BCE_pred_obs/3

        KLD_obs_self  = -0.5 * torch.sum(1 + obs_recon_obs['logvar'] - obs_recon_obs['mu'].pow(2) - obs_recon_obs['logvar'].exp())
        KLD_pred_self = -0.5 * torch.sum(1 + pred_recon_pred['logvar'] - pred_recon_pred['mu'].pow(2) - pred_recon_pred['logvar'].exp())
        # KLD= self.kl_div(obs_recon_obs['mu'],obs_recon_obs['logvar'], pred_recon_pred['mu'], pred_recon_pred['logvar'])
        KLD=KLD_obs_self/2+KLD_pred_self/2
        
        # print("BCE-loss=",BCE,",KLD-loss=",KLD)
        return BCE + self.alpha*KLD

    def encode_space(self, x):
        r"""Encode the trajectory into the latent space
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return z
    
    def freeze(self):
        for param in self.parameters():
            param.requires_grad_(False)
            
            

class SimulaSVDReductioner(nn.Module):
    r"""A class for reducing the dimension of trajectories using a shared VAE.
    Args:
        hyper_params (DotDict): The hyper-parameters
    """
    def __init__(self, hyper_params):
        super().__init__()
        self.hyper_params = hyper_params
        self.t_obs, self.t_pred = hyper_params.obs_len, hyper_params.pred_len

        self.dim = hyper_params.traj_dim
        self.latent_dim = hyper_params.k
        self.alpha=hyper_params.alpha
        self.beta=hyper_params.beta

        self.conv_layers = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.lstm=nn.LSTM(32,32)
        # self.att=nn.MultiheadAttention(32,2,batch_first=True)
        self.fc_mu = nn.Linear(16, self.latent_dim)          
        self.fc_logvar = nn.Linear(16, self.latent_dim)

        # decoder_obs
        self.fc3_obs = nn.Linear(self.latent_dim, 32)
        self.fc4_obs = nn.Linear(32, 16)        
        self.fc_out_obs = nn.Linear(16, self.t_obs*self.dim)

        # decoder_pred
        self.fc3_pred = nn.Linear(self.latent_dim, 32)
        self.fc4_pred = nn.Linear(32, 16)
        self.fc_out_pred = nn.Linear(16, self.t_pred*self.dim)
                
        self.V_obs_trunc = nn.Parameter(torch.zeros((self.t_obs * self.dim, self.latent_dim)))
        self.V_pred_trunc = nn.Parameter(torch.zeros((self.t_pred * self.dim, self.latent_dim)))
        
    def Para_init(self,obs_trunc,pred_trunc):
        self.V_obs_trunc.cuda()
        self.V_pred_trunc.cuda()
        self.V_obs_trunc= nn.Parameter(obs_trunc.to(self.V_obs_trunc.device))          
        self.V_pred_trunc = nn.Parameter(pred_trunc.to(self.V_pred_trunc.device))

    def encode(self, x):
        print("encode input.shape=",x.shape)
        x = x.flatten(start_dim=-2).unsqueeze(1)
        x = self.conv_layers(x)    
        
        x = x.permute(2, 0, 1)                          # (seq_len*dim, B, 32)
        _, (h_n, _) = self.lstm(x)                      # h_n: (2, B, 64)
        h_n = h_n.permute(1, 0, 2).flatten(1)  
        
        # h_n,_=self.att(h_n,h_n,h_n)       
        
        steps=h_n.shape[-1]//2
        h_n_mu=h_n[...,:steps]
        h_n_logvar=h_n[...,steps:]        
        # print("encode x.shape=",x.shape)# 517 32 16
        mu = self.fc_mu(h_n_mu)
        logvar = self.fc_logvar(h_n_logvar)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode_obs(self,z):        
        h3 = F.relu(self.fc3_obs(z))
        h3 = F.relu(self.fc4_obs(h3))
        out = self.fc_out_obs(h3)
        return out.detach()

    def decode_pred(self, z):
        h3 = F.relu(self.fc3_pred(z))
        h3 = F.relu(self.fc4_pred(h3))
        out = self.fc_out_pred(h3)
        return out
    
    def forward(self, obs, pred):
        # 展平为 (batch, features)
        mu_obs, logvar_obs = self.encode(obs)        
        mu_pred, logvar_pred = self.encode(pred)
        
        mu_pred, logvar_pred =mu_pred, logvar_pred

        z_obs = self.reparameterize(mu_obs, logvar_obs)
        z_pred = self.reparameterize(mu_pred, logvar_pred)

        obs_recon_obs = self.decode_obs(z_obs)
        pred_recon_pred = self.decode_pred(z_pred)
        obs_recon_pred = self.decode_pred(z_obs)

        obs_recon_obs = {'traj': obs_recon_obs.reshape(-1, self.t_obs* self.dim), 'mu': mu_obs, 'logvar': logvar_obs}
        pred_recon_pred = {'traj': pred_recon_pred.reshape(-1, self.t_pred* self.dim), 'mu': mu_pred, 'logvar': logvar_pred}
        obs_recon_pred = {'traj': obs_recon_pred.reshape(-1, self.t_pred* self.dim)}

        # 返回重构的轨迹和潜在空间的均值和方差
        return obs_recon_obs, pred_recon_pred, obs_recon_pred    
    
    def svd_repr(self,traj,evec):
        # print("evec.shape=",evec.shape) #[16,8]
        # print("traj.shape=",traj.shape) #[8192,16]
        C = traj @ evec.detach() 
        return C.detach()
    
    def loss(self, obs, pred, obs_recon_obs, pred_recon_pred, obs_recon_pred):
     
        svd_obs=self.svd_repr(obs.reshape(-1, self.t_obs * self.dim),self.V_obs_trunc)
        svd_pred=self.svd_repr(pred.reshape(-1, self.t_pred * self.dim),self.V_pred_trunc)
        # print("mu.shape=",obs_recon_obs['mu'].shape) #[512,8]
        MSE=F.mse_loss(obs_recon_obs['mu'],svd_obs)/2+F.mse_loss(pred_recon_pred['mu'],svd_pred)/2
                
        # print("obs_recon_obs.shape=",obs_recon_obs['traj'].shape) #[16,8]
        # print("obs.shape=",obs.shape) #[16,8]
        BCE_obs_self = F.mse_loss(obs_recon_obs['traj'], obs.reshape(-1, self.t_obs * self.dim), reduction='sum')
        BCE_pred_self = F.mse_loss(pred_recon_pred['traj'], pred.reshape(-1, self.t_pred * self.dim), reduction='sum')
        BCE_pred_obs = F.mse_loss(obs_recon_pred['traj'], pred.reshape(-1, self.t_pred * self.dim), reduction='sum')
        BCE = BCE_obs_self/3 + BCE_pred_self/3 + BCE_pred_obs/3

        KLD_obs_self  = -0.5 * torch.sum(1 + obs_recon_obs['logvar'] - obs_recon_obs['mu'].pow(2) - obs_recon_obs['logvar'].exp())
        KLD_pred_self = -0.5 * torch.sum(1 + pred_recon_pred['logvar'] - pred_recon_pred['mu'].pow(2) - pred_recon_pred['logvar'].exp())
        # KLD= self.kl_div(obs_recon_obs['mu'],obs_recon_obs['logvar'], pred_recon_pred['mu'], pred_recon_pred['logvar'])
        KLD=KLD_obs_self/2+KLD_pred_self/2
        
        # print("BCE-loss=",BCE,",KLD-loss=",KLD)
        return BCE + self.alpha*KLD+self.beta*MSE
    
    
    def encode_space(self, x):
        r"""Encode the trajectory into the latent space
        """
        # print("encode_space input.shape=",x.shape)
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        # print("z.requires_grad=", z.requires_grad, " grad_fn=", z.grad_fn)
        return z
    
    def freeze(self):
        for param in self.parameters():
            param.requires_grad_(False)


class E2EReductioner(nn.Module):
    r"""A class for reducing the dimension of trajectories using a shared VAE.
    Args:
        hyper_params (DotDict): The hyper-parameters
    """
    def __init__(self, hyper_params):
        super().__init__()
        self.hyper_params = hyper_params
        self.t_obs, self.t_pred = hyper_params.obs_len, hyper_params.pred_len

        self.dim = hyper_params.traj_dim
        self.hidden_dim = 16
        self.latent_dim = hyper_params.k
        self.nhead=2
        self.num_layers=3

        self.Extender=Extender()
        self.linear_proj=nn.Linear(6,self.hidden_dim)
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim,
            nhead=self.nhead,
            dim_feedforward=self.hidden_dim * 2,
            dropout=0.1,
            batch_first=True   # [B, T, C]
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.hidden_dim))  # 使用CLS token学习特征，而非mean
        self.transformer = nn.TransformerEncoder(self.encoder_layer, num_layers=self.num_layers)

        self.fc_mu = nn.Linear(self.hidden_dim, self.latent_dim)          
        self.fc_logvar = nn.Linear(self.hidden_dim, self.latent_dim)

        # decoder_obs
        self.fc3_obs = nn.Linear(self.latent_dim, self.hidden_dim*2)
        self.fc4_obs = nn.Linear(self.hidden_dim*2, self.hidden_dim)        
        self.fc_out_obs = nn.Linear(self.hidden_dim, self.t_obs*self.dim)

        # decoder_pred
        self.fc3_pred = nn.Linear(self.latent_dim, self.hidden_dim*2)
        self.fc4_pred = nn.Linear(self.hidden_dim*2, self.hidden_dim)
        self.fc_out_pred = nn.Linear(self.hidden_dim, self.t_pred*self.dim)
                        
    def encode(self, x):
        B=x.shape[0]
        x=self.Extender.extend(x)    # 【batch,seqlen,6】
        x=self.linear_proj(x)
        
        cls_tokens = self.cls_token.expand(B, 1, -1)  # [B, 1, hidden_dim]
        x=torch.cat([cls_tokens,x],dim=1)
        # print("x.shape=",x.shape) # [batch,13,16]
        x=self.transformer(x)        
        x = x[:,0]       # 取cls
        
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode_obs(self,z):        
        if 0 in z.shape:
            # print("z.shape=",z.shape)
            return torch.zeros(z.shape[0],self.t_obs*self.dim)
        
        h3 = F.relu(self.fc3_obs(z))
        h3 = F.relu(self.fc4_obs(h3))
        out = self.fc_out_obs(h3)
        return out.detach()
    
    def batch_decode_obs(self,z):  
        S, B, L = z.shape
        # print("z.shape=",z.shape)
        z_flat = z.view(S * B, L)         # 展开成 [S*B, L]      
        
        h3 = F.relu(self.fc3_obs(z_flat))
        h3 = F.relu(self.fc4_obs(h3))
        out = self.fc_out_obs(h3)
        
        out = out.reshape(S, B, self.dim*self.t_obs)
        return out

    def decode_pred(self, z):
        if 0 in z.shape:
            # print("z.shape=",z.shape)
            return torch.zeros(z.shape[0],self.t_pred*self.dim)
        
        h3 = F.relu(self.fc3_pred(z))
        h3 = F.relu(self.fc4_pred(h3))
        out = self.fc_out_pred(h3)
        
        return out    
    
    def batch_decode_pred(self,z):  
        S, B, L = z.shape
        # print("z.shape=",z.shape)  # z.shape= torch.Size([20, 9574, 8])
        z_flat = z.reshape(S * B, L)         # 展开成 [S*B, L]  
        
        # print("z_flat.shape=",z_flat.shape)
        h3 = F.relu(self.fc3_pred(z_flat))
        h3 = F.relu(self.fc4_pred(h3))
        out = self.fc_out_pred(h3)
        
        out = out.reshape(S, B, self.dim*self.t_pred)
        return out
    
    def encode_space(self, x):   # x=[batch,seqlen,2]
        r"""Encode the trajectory into the latent space
        """
        # print("x.shape=",x.shape)
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        # print("z.requires_grad=", z.requires_grad, " grad_fn=", z.grad_fn)
        # print("encode_space.z.shape=",z.shape) # torch.Size([206, 8])
        return z
    
    def batch_encode_space(self, x):  # x: [S, B, T, 2]
        S, B, T, D = x.shape
        x_flat = x.reshape(S * B, T, D)         # 展开成 [S*B, T, 2]
        
        batch_size = 1024  # 减小批次大小
        mu_list = []
        logvar_list = []

        # 不这样会报错：transformer不能处理太大的batch
        # RuntimeError: CUDA error: invalid configuration argument
        # Compile with `TORCH_USE_CUDA_DSA` to enable device-side assertions.
        for i in range(0, x_flat.shape[0], batch_size):
            batch_x = x_flat[i:i+batch_size]
            mu_batch, logvar_batch = self.encode(batch_x)
            mu_list.append(mu_batch)
            logvar_list.append(logvar_batch)

        mu = torch.cat(mu_list, dim=0)
        logvar = torch.cat(logvar_list, dim=0)
        
        
        """
        mu, logvar = self.encode(x_flat)     # 调用原来的 encode，返回 [S*B, latent_dim]
        """
        
        # 恢复成 [S, B, latent_dim]
        z = self.reparameterize(mu, logvar)  # [S*B, latent_dim]
        z = z.reshape(S, B, -1)
        return z

    def freeze(self):
        for param in self.parameters():
            param.requires_grad_(False)
  