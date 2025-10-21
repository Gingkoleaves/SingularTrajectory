import torch
import torch.nn as nn

# 逐个起始帧【逐批】考虑行人
# 获取当前起始帧的各个行人位置 [num,12,2]
# 对每个行人，计算他与邻居的相对位置、速度差、距离、方位角、mpd等特征
# 将每个行人的邻居特征拼接成 [num,neighbor_num,frames,features]
# 最后得到 [batch,neighbor_num,frames,features]

import torch

# Extract neighbors, return [batch,neighbor_num,frames(12/8),features]

class neighbor_Extractor(nn.Module):
    """
    向量化版本的邻居特征提取
    ----------
    输入：
        trajs: [batch, T, 2]  每条轨迹的坐标序列
        frames: [batch]        每个行人的起始帧编号
        ob_radius: float       邻居判断的感知半径
    输出：
        neighbor_features: [batch, T, max_neighbors, 7]
        neighbor_masks:    [batch, T, max_neighbors]
    """
    def __init__(self):
        super().__init__();
        self.hidden_dim = 16
        self.neighbor_LSTM=nn.LSTM(
            input_size=7,
            hidden_size=self.hidden_dim,
            num_layers=1,
            batch_first=True
        )
        
    def Extracte(self,trajs, frames, ob_radius=2.0):
        batch_size, T, _ = trajs.shape
        device = trajs.device

        # 先计算 pairwise distance mask，按 frame 分组
        neighbor_features_global = [None] * batch_size
        neighbor_masks_global = [None] * batch_size

        for frame in frames.unique():
            idx = (frames == frame).nonzero(as_tuple=True)[0]
            curr_trajs = trajs[idx]  # [num, T, 2]
            num = curr_trajs.shape[0]

            # pairwise 差向量和距离
            # dp: [num, num, T, 2]
            dp = curr_trajs.unsqueeze(1) - curr_trajs.unsqueeze(0)
            dist = dp.norm(dim=-1)  # [num, num, T]
            mask = (dist <= ob_radius) & (dist > 0)  # [num, num, T]

            # 速度差
            v = torch.zeros_like(curr_trajs)
            v[:, 1:, :] = curr_trajs[:, 1:, :] - curr_trajs[:, :-1, :]
            v_i = v.unsqueeze(1)        # [num,1,T,2]
            v_all = v.unsqueeze(0)      # [1,num,T,2]
            dv = v_all - v_i             # [num,num,T,2]

            # bearing & mpd
            v_target = v.unsqueeze(1)   # [num,1,T,2]
            dot_dp_v = (dp * v_target).sum(-1)  # [num,num,T]
            v_norm = v_target.norm(dim=-1) + 1e-8
            bearing = torch.zeros_like(dist)
            mpd = torch.zeros_like(dist)
            valid_mask = (t := torch.arange(T, device=device)).unsqueeze(0).unsqueeze(0) > 0
            bearing[:, :, 1:] = (dot_dp_v[:, :, 1:] / (dist[:, :, 1:] * v_norm[:, :, 1:])).clamp(-1,1)
            cross = dp[:, :, :, 0]*v_target[:, :, :, 1] - dp[:, :, :, 1]*v_target[:, :, :, 0]
            mpd[:, :, 1:] = (torch.abs(cross[:, :, 1:]) / (v_norm[:, :, 1:] + 1e-8))

            # 拼接特征: dp, dv, dist, bearing, mpd
            # [num, num, T, 7]
            features = torch.cat([dp, dv, dist.unsqueeze(-1), bearing.unsqueeze(-1), mpd.unsqueeze(-1)], dim=-1)

            # 根据 mask 提取邻居
            max_n = mask.sum(dim=1).max().item()  # 当前 frame 最大邻居数
            neighbor_features = torch.zeros(num, T, max_n, 7, device=device)
            neighbor_masks = torch.zeros(num, T, max_n, device=device)

            for i in range(num):
                for t in range(T):
                    neigh_idx = mask[i, :, t].nonzero(as_tuple=True)[0]
                    n = neigh_idx.numel()
                    if n > 0:
                        neighbor_features[i, t, :n] = features[i, neigh_idx, t]
                        neighbor_masks[i, t, :n] = 1.0

            # 写入全局
            for k, ped_idx in enumerate(idx):
                neighbor_features_global[ped_idx.item()] = neighbor_features[k]
                neighbor_masks_global[ped_idx.item()] = neighbor_masks[k]

        # 对 batch 对齐不同 frame 最大邻居数
        max_n = max(f.size(1) for f in neighbor_features_global)
        padded_features = []
        padded_masks = []
        for f, m in zip(neighbor_features_global, neighbor_masks_global):
            pad_len = max_n - f.size(1)
            if pad_len > 0:
                f = torch.cat([f, torch.zeros(f.size(0), pad_len, f.size(2), device=device)], dim=1)
                m = torch.cat([m, torch.zeros(m.size(0), pad_len, device=device)], dim=1)
            padded_features.append(f)
            padded_masks.append(m)

        neighbor_features = torch.stack(padded_features, dim=0)
        neighbor_masks = torch.stack(padded_masks, dim=0)



        # Trajectron++ way:
        addi_features = torch.sum(neighbor_features,dim=-2)  # [206,8,7] batch,steps,hidden-vector
        neighbor_aggr,_=self.neighbor_LSTM(addi_features) # [batch,seqlen,hidden_dim]
        neighbor_aggr = neighbor_aggr[:, -1]  # 取最后时间步 [batch, hidden_dim]


        return neighbor_aggr
