#!/usr/bin/env python3
"""
AGF避障导航系统 - 测试脚本

测试基于APF的避障导航功能
"""

import sys
import os
import argparse
import numpy as np

# 添加项目路径
sys.path.insert(0, '/home/peking/projects/RL/gym-pybullet-drones')

from gym_pybullet_drones.AGF.agf_navigator import AGFNavigator, find_latest_model
from gym_pybullet_drones.custom.config_continuous import DEFAULT_OUTPUT_FOLDER


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="AGF避障导航系统测试",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  python test_agf_navigation.py                    # 使用最新模型，GUI模式，启用障碍物
  python test_agf_navigation.py --no-gui           # 无GUI模式  
  python test_agf_navigation.py --no-obstacles     # 禁用障碍物（测试基础飞行）
  python test_agf_navigation.py --model model.zip  # 指定模型
  python test_agf_navigation.py --target 0.5 0.5 1.0  # 指定目标
        """
    )
    
    parser.add_argument('--model', type=str, default=None,
                       help='指定模型文件路径 (默认: 自动查找最新模型)')
    
    parser.add_argument('--gui', dest='gui', action='store_true', default=True,
                       help='显示PyBullet GUI界面 (默认: 开启)')
    parser.add_argument('--no-gui', dest='gui', action='store_false',
                       help='不显示GUI界面')
    
    parser.add_argument('--start', type=float, nargs=3, 
                       default=[-1.2, 0.0, 0.5],
                       metavar=('X', 'Y', 'Z'),
                       help='起始位置 (默认: -1.2 0.0 0.5 地图左端)')
    
    parser.add_argument('--target', type=float, nargs=3, 
                       default=[1.2, 0.0, 0.5],
                       metavar=('X', 'Y', 'Z'),
                       help='目标位置 (默认: 1.2 0.0 0.5 地图右端)')
    
    parser.add_argument('--apf-freq', type=int, default=3,
                       help='APF更新频率 (默认: 3, 更频繁的路径更新)')
    
    parser.add_argument('--no-obstacles', dest='obstacles', action='store_false',
                       default=True,
                       help='禁用障碍物（用于测试基础飞行路径）')
    parser.add_argument('--obstacles', dest='obstacles', action='store_true',
                       help='启用障碍物（默认）')
    
    return parser.parse_args()


def print_welcome():
    """打印欢迎信息"""
    print("\n" + "="*70)
    print("🚁 AGF避障导航系统测试")
    print("="*70)
    print("基于人工势场(APF)的智能避障导航")
    print("分层控制架构：APF规划 + PPO执行")
    print("="*70)


def print_test_info(args, model_path):
    """打印测试信息"""
    print(f"\n📋 测试配置:")
    print(f"  模型文件: {model_path}")
    print(f"  GUI模式: {'开启' if args.gui else '关闭'}")
    print(f"  起始位置: ({args.start[0]:.2f}, {args.start[1]:.2f}, {args.start[2]:.2f})")
    print(f"  目标位置: ({args.target[0]:.2f}, {args.target[1]:.2f}, {args.target[2]:.2f})")
    
    # 计算直线距离
    import numpy as np
    distance = np.linalg.norm(np.array(args.target) - np.array(args.start))
    print(f"  直线距离: {distance:.2f}m")
    print(f"  APF更新频率: 每{args.apf_freq}步")
    
    if args.obstacles:
        print(f"\n🚧 障碍物配置:")
        print(f"  障碍物1: 蓝色圆柱 @ (0.6, -0.5, 0.5)")
        print(f"  障碍物2: 红色圆柱 @ (-0.6, +0.5, 0.5)")
        print(f"  两柱间距: 1.0m")
    else:
        print(f"\n✨ 障碍物: 已禁用（测试基础飞行路径）")
    print()


def main():
    """主函数"""
    try:
        # 解析参数
        args = parse_arguments()
        
        # 打印欢迎信息
        print_welcome()
        
        # 确定模型路径
        if args.model:
            model_path = args.model
            print(f"[模型] 使用指定模型: {model_path}")
        else:
            print(f"[模型] 正在查找最新训练模型...")
            model_path = find_latest_model(DEFAULT_OUTPUT_FOLDER)
            print(f"[模型] 找到最新模型: {model_path}")
        
        # 打印测试信息
        print_test_info(args, model_path)
        
        # 创建AGF导航器
        print(f"[系统] 正在初始化AGF导航系统...")
        navigator = AGFNavigator(
            model_path=model_path,
            gui=args.gui,
            record=False,
            apf_update_freq=args.apf_freq,
            start_pos=np.array(args.start),
            obstacles=args.obstacles  # 传递障碍物参数
        )
        
        # 初始化系统
        navigator.initialize()
        
        # 执行导航
        print(f"\n[系统] 开始避障导航测试...")
        result = navigator.navigate_to_target(args.target)
        
        # 显示结果
        if result['success']:
            print(f"\n✅ 测试成功！无人机成功到达目标位置")
        else:
            print(f"\n⚠️ 测试未完全成功")
            print(f"   原因: {result.get('reason', '未到达目标')}")
        
        # 关闭环境
        navigator.close()
        
        print(f"\n{'='*70}")
        print(f"🎉 AGF避障导航测试完成")
        print(f"{'='*70}\n")
        
    except KeyboardInterrupt:
        print(f"\n[系统] 用户中断测试")
        sys.exit(0)
        
    except FileNotFoundError as e:
        print(f"\n❌ 文件未找到: {e}")
        print(f"💡 请确认:")
        print(f"   1. 模型文件路径是否正确")
        print(f"   2. 是否已完成模型训练")
        sys.exit(1)
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
