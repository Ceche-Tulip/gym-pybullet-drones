#!/usr/bin/env python3
"""
快速启动脚本 - 连续导航系统

简化启动流程，自动处理常见配置
"""

import os
import sys

def main():
    """快速启动主函数"""
    
    print("🚁 启动无人机连续导航系统...")
    print("="*50)
    
    # 确保在正确的目录中运行
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 切换到项目根目录
    project_root = os.path.dirname(os.path.dirname(current_dir))
    os.chdir(project_root)
    
    print(f"当前工作目录: {os.getcwd()}")
    
    # 运行连续导航系统
    try:
        import subprocess
        
        # 构建运行命令
        cmd = [
            sys.executable, 
            "-m", 
            "gym_pybullet_drones.custom.rollout_continuous",
            "--gui"
        ]
        
        print(f"执行命令: {' '.join(cmd)}")
        print("="*50)
        
        # 启动系统
        subprocess.run(cmd, check=True)
        
    except subprocess.CalledProcessError as e:
        print(f"❌ 程序运行失败: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print(f"\n🛑 用户中断程序")
        sys.exit(0)
    except Exception as e:
        print(f"❌ 启动失败: {e}")
        print("\n💡 请尝试:")
        print("  1. 确认conda环境已激活: conda activate drones")
        print("  2. 确认已完成模型训练")
        print("  3. 手动运行: python -m gym_pybullet_drones.custom.rollout_continuous")
        sys.exit(1)

if __name__ == "__main__":
    main()