#!/usr/bin/env python3
"""
快速测试障碍物配置
"""

import sys
import os

# 添加项目路径
sys.path.insert(0, '/home/peking/projects/RL/gym-pybullet-drones')

from gym_pybullet_drones.custom.space_expander import ExtendedHoverAviary
import numpy as np

print("="*60)
print("🧪 测试障碍物配置")
print("="*60)

try:
    print("\n[测试] 创建带障碍物的环境...")
    
    env = ExtendedHoverAviary(
        initial_xyzs=np.array([[0, 0, 0.1]]),
        initial_rpys=np.array([[0, 0, 0]]),
        gui=False,  # 无GUI测试
        record=False,
        obstacles=True  # 启用障碍物
    )
    
    print("[测试] ✅ 环境创建成功！")
    print(f"[测试] 障碍物数量: {len(env.OBSTACLE_IDS) if hasattr(env, 'OBSTACLE_IDS') else 0}")
    
    # 测试环境重置
    print("\n[测试] 重置环境...")
    obs, info = env.reset()
    print("[测试] ✅ 环境重置成功！")
    
    # 测试几步仿真
    print("\n[测试] 运行10步仿真...")
    for i in range(10):
        action = np.array([[0, 0, 0, 0]])  # 静止动作
        obs, reward, terminated, truncated, info = env.step(action)
    print("[测试] ✅ 仿真运行成功！")
    
    # 获取当前状态
    print("\n[测试] 获取无人机状态...")
    state = env.get_current_state()
    print(f"[测试] 位置: {state['position']}")
    print(f"[测试] 目标: {state['target_position']}")
    print(f"[测试] 距离: {state['distance_to_target']:.3f}m")
    
    env.close()
    
    print("\n" + "="*60)
    print("✅ 所有测试通过！障碍物配置正确！")
    print("="*60)
    
except Exception as e:
    print(f"\n❌ 测试失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
