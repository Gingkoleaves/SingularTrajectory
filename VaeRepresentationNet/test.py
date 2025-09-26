import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from layers import SceneTransformerVAE

# 设置随机种子以确保可重复性
torch.manual_seed(42)
np.random.seed(42)

OBS_LEN=12

def test_model():
    # 模型参数
    batch_size = 16
    obs_len = OBS_LEN  # 观测轨迹长度
    target_len = 12  # 预测轨迹长度
    d_model = 128
    latent_dim = 64
    
    # 创建模型实例
    model = SceneTransformerVAE(d_model=d_model, latent_dim=latent_dim)
    
    # 打印模型参数数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型总参数量: {total_params:,}")
    
    # 生成模拟数据
    # 观测轨迹 (batch_size, obs_len, 2)
    obs_trajs = torch.randn(batch_size, obs_len, 2)
    # 真实轨迹 (batch_size, target_len, 2)
    gt_trajs = torch.randn(batch_size, target_len, 2)
    
    print(f"输入观测轨迹形状: {obs_trajs.shape}")
    print(f"真实轨迹形状: {gt_trajs.shape}")
    
    # 测试前向传播
    recon_trajs, total_loss, recon_loss, kl_loss = model(obs_trajs, gt_trajs, beta=5.0)
    print(f"重建轨迹形状: {recon_trajs.shape}")
    print(f"总损失: {total_loss.item():.4f}")
    print(f"重建损失: {recon_loss.item():.4f}")
    print(f"KL损失: {kl_loss.item():.4f}")
    
    # 测试编码器
    mu, logvar = model.encode(obs_trajs)
    print(f"隐变量均值形状: {mu.shape}")
    print(f"隐变量对数方差形状: {logvar.shape}")
    
    # 测试重参数化
    z = model.reparameterize(mu, logvar)
    print(f"重参数化后的隐变量形状: {z.shape}")
    
    # 测试解码器
    decoded_trajs = model.decode(z, target_length=target_len)
    print(f"解码后的轨迹形状: {decoded_trajs.shape}")
    
    # 测试采样
    sampled_trajs = model.sample(obs_trajs, target_length=target_len)
    print(f"采样轨迹形状: {sampled_trajs.shape}")
    
    # 检查梯度流动
    total_loss.backward()
    
    # 检查参数是否有梯度
    has_gradients = any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())
    print(f"模型参数是否有梯度: {has_gradients}")
    
    # 可视化一个样本的轨迹
    visualize_trajectories(obs_trajs[0], gt_trajs[0], recon_trajs[0], sampled_trajs[0])
    
    return model

def visualize_trajectories(obs_traj, gt_traj, recon_traj, sampled_traj):
    """可视化轨迹对比"""
    plt.figure(figsize=(12, 10))
    
    # 观测轨迹
    plt.subplot(2, 2, 1)
    plt.plot(obs_traj[:, 0].detach().numpy(), obs_traj[:, 1].detach().numpy(), 
             'bo-', label='Observed', markersize=4)
    plt.title('Observed Trajectory')
    plt.legend()
    plt.axis('equal')
    
    # 真实轨迹
    plt.subplot(2, 2, 2)
    plt.plot(gt_traj[:, 0].detach().numpy(), gt_traj[:, 1].detach().numpy(), 
             'go-', label='Ground Truth', markersize=4)
    plt.title('Ground Truth Trajectory')
    plt.legend()
    plt.axis('equal')
    
    # 重建轨迹
    plt.subplot(2, 2, 3)
    plt.plot(obs_traj[:, 0].detach().numpy(), obs_traj[:, 1].detach().numpy(), 
             'bo-', label='Observed', markersize=4)
    plt.plot(recon_traj[:, 0].detach().numpy(), recon_traj[:, 1].detach().numpy(), 
             'ro-', label='Reconstructed', markersize=4)
    plt.title('Reconstructed vs Observed')
    plt.legend()
    plt.axis('equal')
    
    # 采样轨迹
    plt.subplot(2, 2, 4)
    plt.plot(obs_traj[:, 0].detach().numpy(), obs_traj[:, 1].detach().numpy(), 
             'bo-', label='Observed', markersize=4)
    plt.plot(sampled_traj[:, 0].detach().numpy(), sampled_traj[:, 1].detach().numpy(), 
             'mo-', label='Sampled', markersize=4)
    plt.title('Sampled vs Observed')
    plt.legend()
    plt.axis('equal')
    
    plt.tight_layout()
    plt.savefig('trajectory_comparison.png')
    plt.show()

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from layers import SceneTransformerVAE

# 设置随机种子以确保可重复性
torch.manual_seed(42)
np.random.seed(42)

class MLPBaseline(nn.Module):
    """纯MLP方式的轨迹预测baseline"""
    def __init__(self, input_dim=12*2, hidden_dims=[256, 128, 64], output_dim=12*2):
        super(MLPBaseline, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        # 构建MLP层
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.1))
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, output_dim))
        self.mlp = nn.Sequential(*layers)
        
        # 打印参数量
        total_params = sum(p.numel() for p in self.parameters())
        print(f"MLP Baseline 参数量: {total_params:,}")
    
    def forward(self, obs_trajs, gt_trajs=None):
        """
        前向传播
        obs_trajs: [batch_size, obs_len, 2]
        返回: 预测轨迹 [batch_size, target_len, 2]
        """
        batch_size = obs_trajs.size(0)
        
        # 展平观测轨迹
        x = obs_trajs.reshape(batch_size, -1)  # [batch_size, obs_len*2]
        
        # MLP处理
        output = self.mlp(x)  # [batch_size, target_len*2]
        
        # 重塑为轨迹格式
        pred_trajs = output.reshape(batch_size, -1, 2)  # [batch_size, target_len, 2]
        
        # 计算损失（如果提供了真实轨迹）
        if gt_trajs is not None:
            recon_loss = nn.MSELoss()(pred_trajs, gt_trajs)
            return pred_trajs, recon_loss, recon_loss, torch.tensor(0.0)
        
        return pred_trajs

def test_models_comparison():
    """对比Transformer VAE和MLP Baseline"""
    print("\n" + "="*60)
    print("模型对比测试: Transformer VAE vs MLP Baseline")
    print("="*60)
    
    # 测试参数
    batch_size = 16
    obs_len = 12
    target_len = 12
    
    # 创建模型
    transformer_model = SceneTransformerVAE(d_model=128, latent_dim=64)
    mlp_model = MLPBaseline()
    
    # 生成测试数据
    obs_trajs = torch.randn(batch_size, obs_len, 2)
    gt_trajs = torch.randn(batch_size, target_len, 2)
    
    # 测试Transformer VAE
    print("\n--- Transformer VAE 测试 ---")
    with torch.no_grad():
        recon_trajs, total_loss, recon_loss, kl_loss = transformer_model(obs_trajs, gt_trajs)
        print(f"Transformer VAE - 重建损失: {recon_loss.item():.6f}")
    
    # 测试MLP Baseline
    print("\n--- MLP Baseline 测试 ---")
    with torch.no_grad():
        mlp_trajs, mlp_loss, _, _ = mlp_model(obs_trajs, gt_trajs)
        print(f"MLP Baseline - 重建损失: {mlp_loss.item():.6f}")
    
    # 性能对比
    print(f"\n--- 性能对比 ---")
    print(f"MLP损失 / Transformer损失: {mlp_loss.item()/recon_loss.item():.3f}")
    
    # 可视化对比
    visualize_comparison(obs_trajs[0], gt_trajs[0], recon_trajs[0], mlp_trajs[0])
    
    return transformer_model, mlp_model

def visualize_comparison(obs_traj, gt_traj, transformer_pred, mlp_pred):
    """可视化模型对比"""
    plt.figure(figsize=(15, 10))
    
    # 观测和真实轨迹
    plt.subplot(2, 2, 1)
    plt.plot(obs_traj[:, 0].detach().numpy(), obs_traj[:, 1].detach().numpy(), 
             'bo-', label='Observed', markersize=4, linewidth=2)
    plt.plot(gt_traj[:, 0].detach().numpy(), gt_traj[:, 1].detach().numpy(), 
             'go-', label='Ground Truth', markersize=4, linewidth=2)
    plt.title('Observed and Ground Truth Trajectories')
    plt.legend()
    plt.axis('equal')
    plt.grid(True, alpha=0.3)
    
    # Transformer VAE 预测
    plt.subplot(2, 2, 2)
    plt.plot(obs_traj[:, 0].detach().numpy(), obs_traj[:, 1].detach().numpy(), 
             'bo-', label='Observed', markersize=4, linewidth=2)
    plt.plot(transformer_pred[:, 0].detach().numpy(), transformer_pred[:, 1].detach().numpy(), 
             'ro-', label='Transformer VAE', markersize=4, linewidth=2)
    plt.title('Transformer VAE Prediction')
    plt.legend()
    plt.axis('equal')
    plt.grid(True, alpha=0.3)
    
    # MLP Baseline 预测
    plt.subplot(2, 2, 3)
    plt.plot(obs_traj[:, 0].detach().numpy(), obs_traj[:, 1].detach().numpy(), 
             'bo-', label='Observed', markersize=4, linewidth=2)
    plt.plot(mlp_pred[:, 0].detach().numpy(), mlp_pred[:, 1].detach().numpy(), 
             'mo-', label='MLP Baseline', markersize=4, linewidth=2)
    plt.title('MLP Baseline Prediction')
    plt.legend()
    plt.axis('equal')
    plt.grid(True, alpha=0.3)
    
    # 三者对比
    plt.subplot(2, 2, 4)
    plt.plot(obs_traj[:, 0].detach().numpy(), obs_traj[:, 1].detach().numpy(), 
             'bo-', label='Observed', markersize=4, linewidth=2)
    plt.plot(gt_traj[:, 0].detach().numpy(), gt_traj[:, 1].detach().numpy(), 
             'go-', label='Ground Truth', markersize=4, linewidth=2)
    plt.plot(transformer_pred[:, 0].detach().numpy(), transformer_pred[:, 1].detach().numpy(), 
             'ro-', label='Transformer VAE', markersize=4, linewidth=2)
    plt.plot(mlp_pred[:, 0].detach().numpy(), mlp_pred[:, 1].detach().numpy(), 
             'mo-', label='MLP Baseline', markersize=4, linewidth=2)
    plt.title('All Models Comparison')
    plt.legend()
    plt.axis('equal')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def test_training_comparison():
    """对比训练过程"""
    print("\n" + "="*60)
    print("训练过程对比")
    print("="*60)
    
    # 创建模型和优化器
    transformer_model = SceneTransformerVAE(d_model=128, latent_dim=64)
    mlp_model = MLPBaseline()
    
    transformer_optimizer = torch.optim.Adam(transformer_model.parameters(), lr=1e-3)
    mlp_optimizer = torch.optim.Adam(mlp_model.parameters(), lr=1e-3)
    
    # 训练参数
    num_epochs = 64
    batch_size = 32
    num_batches = 20
    
    transformer_losses = []
    mlp_losses = []
    
    for epoch in range(num_epochs):
        transformer_epoch_loss = 0
        mlp_epoch_loss = 0
        
        for batch_idx in range(num_batches):
            # 生成训练数据
            obs_trajs = torch.randn(batch_size, 12, 2)
            gt_trajs = torch.randn(batch_size, 12, 2)
            
            # 训练Transformer VAE
            transformer_optimizer.zero_grad()
            _, total_loss, recon_loss, _ = transformer_model(obs_trajs, gt_trajs)
            total_loss.backward()
            transformer_optimizer.step()
            transformer_epoch_loss += recon_loss.item()
            
            # 训练MLP Baseline
            mlp_optimizer.zero_grad()
            _, mlp_loss, _, _ = mlp_model(obs_trajs, gt_trajs)
            mlp_loss.backward()
            mlp_optimizer.step()
            mlp_epoch_loss += mlp_loss.item()
        
        # 记录平均损失
        transformer_avg_loss = transformer_epoch_loss / num_batches
        mlp_avg_loss = mlp_epoch_loss / num_batches
        transformer_losses.append(transformer_avg_loss)
        mlp_losses.append(mlp_avg_loss)
        
        print(f'Epoch [{epoch+1}/{num_epochs}]')
        print(f'  Transformer VAE Loss: {transformer_avg_loss:.6f}')
        print(f'  MLP Baseline Loss:    {mlp_avg_loss:.6f}')
        print(f'  Ratio (MLP/Trans):    {mlp_avg_loss/transformer_avg_loss:.3f}')
        print()
    
    # 绘制训练曲线
    plt.figure(figsize=(10, 6))
    plt.plot(transformer_losses, 'r-', label='Transformer VAE', linewidth=2)
    plt.plot(mlp_losses, 'm-', label='MLP Baseline', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Reconstruction Loss')
    plt.title('Training Loss Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('training_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    # 测试原始模型功能
    print("测试原始Transformer VAE模型...")
    model = test_model()
    
    # 测试模型对比
    transformer_model, mlp_model = test_models_comparison()
    
    # 测试训练过程对比
    test_training_comparison()
    
    print("\n所有测试完成！")
    print("MLP Baseline 提供了简单的对比基准，可以评估Transformer VAE的性能优势。")