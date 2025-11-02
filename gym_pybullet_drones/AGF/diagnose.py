#!/usr/bin/env python3
"""
AGF系统诊断工具 - 分析原地打转问题

用法:
  python diagnose.py --pos 0.7 0.7 0.5 --target 0.7 0.7 0.5
"""

import numpy as np
import sys

def calculate_distance_to_cylinder(point, cyl_bottom, radius, height):
    """计算点到圆柱的距离"""
    px, py, pz = point
    cx, cy, cz = cyl_bottom
    
    # XY平面距离
    dx = px - cx
    dy = py - cy
    dist_xy = np.sqrt(dx**2 + dy**2)
    
    # 圆柱高度范围
    cyl_top = cz + height
    
    # 情况1: 在圆柱高度范围内
    if cz <= pz <= cyl_top:
        if dist_xy <= radius:
            # 内部
            return radius - dist_xy, "内部"
        else:
            # 外部侧面
            return dist_xy - radius, "外侧"
    
    # 情况2: 上方
    elif pz > cyl_top:
        if dist_xy <= radius:
            return pz - cyl_top, "正上方"
        else:
            # 到顶边缘
            edge = np.array([cx + (dx/dist_xy)*radius, cy + (dy/dist_xy)*radius, cyl_top])
            return np.linalg.norm(point - edge), "上方斜角"
    
    # 情况3: 下方
    else:
        if dist_xy <= radius:
            return cz - pz, "正下方"
        else:
            edge = np.array([cx + (dx/dist_xy)*radius, cy + (dy/dist_xy)*radius, cz])
            return np.linalg.norm(point - edge), "下方斜角"

def diagnose_position(current_pos, target_pos):
    """诊断当前位置"""
    print("="*70)
    print("🔍 AGF系统位置诊断")
    print("="*70)
    
    print(f"\n当前位置: {current_pos}")
    print(f"目标位置: {target_pos}")
    
    # 到目标的距离
    dist_to_target = np.linalg.norm(np.array(target_pos) - np.array(current_pos))
    print(f"\n📏 到目标距离: {dist_to_target:.4f}m")
    
    # 障碍物配置
    obstacles = [
        {'name': '蓝色圆柱', 'pos': [0.0, -0.4, 0.0], 'r': 0.10, 'h': 1.0},
        {'name': '红色圆柱', 'pos': [0.0, +0.4, 0.0], 'r': 0.10, 'h': 1.0}
    ]
    
    print(f"\n🚧 障碍物距离分析:")
    for obs in obstacles:
        dist, status = calculate_distance_to_cylinder(
            current_pos, obs['pos'], obs['r'], obs['h']
        )
        print(f"  {obs['name']}: {dist:.4f}m ({status})")
        
        # 警告
        if dist < 0.3:
            print(f"    ⚠️  警告: 距离过近! (<0.3m)")
        if dist < 0.5:
            print(f"    ⚡ 注意: 在斥力影响范围内 (d0=0.5m)")
    
    # 判定阈值分析
    print(f"\n🎯 目标判定分析:")
    thresholds = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]
    for thresh in thresholds:
        status = "✅ 到达" if dist_to_target < thresh else "❌ 未到达"
        print(f"  阈值 {thresh:.2f}m: {status} (当前距离: {dist_to_target:.4f}m)")
    
    # 建议
    print(f"\n💡 建议:")
    if dist_to_target < 0.3:
        print(f"  ✅ 距离 {dist_to_target:.4f}m < 0.3m，应该判定为到达")
        print(f"  📝 如果系统没有判定到达，请检查 goal_threshold 设置")
    elif dist_to_target < 0.5:
        print(f"  ⚠️  距离 {dist_to_target:.4f}m 在 0.3-0.5m 之间")
        print(f"  📝 建议将 goal_threshold 增加到 {dist_to_target + 0.05:.2f}m")
    else:
        print(f"  ❌ 距离 {dist_to_target:.4f}m 太远，无人机未到达目标")
        print(f"  📝 可能是导航失败，建议检查APF参数")
    
    print(f"\n{'='*70}\n")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--pos', type=float, nargs=3, required=True,
                       help='当前位置 x y z')
    parser.add_argument('--target', type=float, nargs=3, required=True,
                       help='目标位置 x y z')
    args = parser.parse_args()
    
    diagnose_position(args.pos, args.target)
