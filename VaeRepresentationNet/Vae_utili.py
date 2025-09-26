import torch
import torch.nn as nn
import torch.nn.functional as F

class Extender():
    r"""A class for calculating velocity and accelerate-speed.

    """
    def __init__(self,use_Vel=True,use_Acc=True):
        self.use_Vel=use_Vel
        self.use_Acc=use_Acc
        self.dim=2
    
    def extend(self,trajs):
        batch, seqlen, dim = trajs.shape
        vel = torch.zeros(batch, seqlen, dim, device=trajs.device)
        acc = torch.zeros(batch, seqlen, dim, device=trajs.device)

        # 速度
        vel[:, :-1] = trajs[:, 1:] - trajs[:, :-1]
        vel[:, -1] = vel[:, -2]  # 最后一帧复制

        # 加速度
        acc[:, :-1] = vel[:, 1:] - vel[:, :-1]
        acc[:, -1] = acc[:, -2]  # 最后一帧复制
            
        trajs = torch.cat([trajs, vel, acc], dim=-1)        
        return trajs
    
    def batch_extend(self,batch_trajs):
        sample, batch, seqlen, dim = batch_trajs.shape
        vel = torch.zeros(sample, batch, seqlen, dim, device=batch_trajs.device)
        acc = torch.zeros(sample, batch, seqlen, dim, device=batch_trajs.device)
        
        # 速度
        vel[:, :, :-1] = batch_trajs[:, :, 1:] - batch_trajs[:, :, :-1]
        vel[:, :, -1] = vel[:, :, -2]  # 最后一帧复制

        # 加速度
        acc[:, :, :-1] = vel[:, :, 1:] - vel[:, :, :-1]
        acc[:, :, -1] = acc[:, :, -2]  # 最后一帧复制
            
        batch_trajs = torch.cat([batch_trajs, vel, acc], dim=-1)        
        return batch_trajs
    
        