#!/usr/bin/env python3
"""
AGF系统深度诊断 - 检测PPO模型泛化能力

分析PPO模型对不同距离目标的响应
"""

import sys
sys.path.insert(0, '/home/peking/projects/RL/gym-pybullet-drones')

import numpy as np
from stable_baselines3 import PPO

def analyze_model_response():
    """分析模型对不同目标距离的响应"""
    print("="*70)
    print("🔍 PPO模型泛化能力分析")
    print("="*70)
    
    # 加载模型
    model_path = 'results/save-10.10.2025_09.50.18/final_model.zip'
    print(f"\n加载模型: {model_path}")
    model = PPO.load(model_path)
    
    print(f"\n观测空间维度: {model.observation_space.shape}")
    print(f"观测空间范围: low={model.observation_space.low[0,:5]}, high={model.observation_space.high[0,:5]}")
    
    # 模拟不同距离的目标
    print(f"\n{'='*70}")
    print("模拟测试：无人机在原点，测试不同距离的目标")
    print(f"{'='*70}")
    
    # 构造基础观测（无人机在原点，静止）
    base_obs = np.zeros((1, 75), dtype=np.float32)
    # 位置 [0, 0, 0.5]
    base_obs[0, 0:3] = [0.0, 0.0, 0.5]
    # 其他状态（姿态、速度等）都是0
    
    # 测试不同的目标相对位置
    test_targets = [
        ([0.3, 0.3, 0.0], "近距离 (0.42m)"),
        ([0.5, 0.5, 0.0], "中距离 (0.71m)"),
        ([0.7, 0.7, 0.0], "训练范围边缘 (0.99m)"),
        ([1.0, 1.0, 0.0], "略超训练范围 (1.41m)"),
        ([1.2, 1.2, 0.0], "明显超出 (1.70m)"),
        ([1.5, 1.5, 0.0], "远距离 (2.12m)"),
    ]
    
    print(f"\n{'目标相对位置':<20} {'距离':<10} {'动作预测':<30} {'置信度'}")
    print("-"*80)
    
    for target_rel, desc in test_targets:
        obs = base_obs.copy()
        # 目标相对位置在观测的最后3维
        obs[0, -3:] = target_rel
        
        # 预测动作
        action, _states = model.predict(obs, deterministic=True)
        
        # 计算距离
        distance = np.linalg.norm(target_rel)
        
        # 动作是RPM，范围0-1归一化
        action_str = f"[{action[0,0]:.3f}, {action[0,1]:.3f}, {action[0,2]:.3f}, {action[0,3]:.3f}]"
        confidence = np.mean(np.abs(action[0]))  # 简单的置信度指标
        
        print(f"{str(target_rel):<20} {distance:.2f}m     {action_str:<30} {confidence:.3f}")
    
    print(f"\n{'='*70}")
    print("📊 分析结果")
    print(f"{'='*70}")
    print("""
关键发现:
1. 如果动作预测在 distance > 0.7-1.0m 后趋于平稳/下降，说明模型泛化能力有限
2. 如果置信度在远距离下降，说明模型对远目标响应不足
3. PPO模型可能在训练时只见过 ±0.8m 范围内的目标

解决方案:
A. 短期: 降低APF步长，让目标相对位置保持在模型训练范围内
   step_size = 0.1  # 从0.2减小到0.1
   
B. 中期: 增加中间航点密度，确保相对目标距离 < 0.7m
   
C. 长期: 重新训练模型，使用更大范围的目标距离
""")

if __name__ == "__main__":
    analyze_model_response()
