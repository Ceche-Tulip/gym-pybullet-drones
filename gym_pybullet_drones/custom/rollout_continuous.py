"""
连续导航系统 - 主启动程序

基于已训练的PPO模型，实现连续的无人机导航，支持：
- 实时目标更新
- 键盘交互控制
- 暂停/继续功能
- 扩展飞行空间
- 详细状态显示

使用方法：
    $ conda activate drones
    $ python rollout_continuous.py               # 使用最新模型
    $ python rollout_continuous.py --model path/to/model.zip  # 指定模型
    $ python rollout_continuous.py --no-gui     # 无GUI模式
"""

import os
import sys
import argparse
import traceback

# 使用模块形式的导入方式
from gym_pybullet_drones.custom.continuous_navigator import ContinuousNavigator, find_latest_model
from gym_pybullet_drones.custom.config_continuous import *

def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="连续导航系统 - 实现无人机的连续自主导航",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  python rollout_continuous.py                    # 默认设置
  python rollout_continuous.py --no-gui           # 无GUI模式  
  python rollout_continuous.py --model model.zip  # 指定模型
  python rollout_continuous.py --record           # 录制视频
        """
    )
    
    # 模型相关参数
    parser.add_argument('--model', type=str, default=None,
                       help='指定模型文件路径 (默认: 自动查找最新模型)')
    
    # 显示相关参数  
    parser.add_argument('--gui', dest='gui', action='store_true', default=DEFAULT_GUI,
                       help='显示PyBullet GUI界面 (默认: 开启)')
    parser.add_argument('--no-gui', dest='gui', action='store_false',
                       help='不显示GUI界面')
    
    # 录制相关参数
    parser.add_argument('--record', action='store_true', default=DEFAULT_RECORD_VIDEO,
                       help='录制演示视频 (默认: 关闭)')
    
    # 调试相关参数
    parser.add_argument('--verbose', action='store_true', default=DEBUG_CONFIG['verbose'],
                       help='详细输出模式 (默认: 开启)')
    parser.add_argument('--quiet', dest='verbose', action='store_false', 
                       help='简化输出模式')
    
    return parser.parse_args()

def validate_model_path(model_path: str) -> str:
    """
    验证模型路径有效性
    
    参数:
        model_path: 模型文件路径
        
    返回:
        str: 验证后的绝对路径
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件不存在: {model_path}")
    
    if not model_path.endswith('.zip'):
        raise ValueError(f"模型文件格式不正确，应为.zip文件: {model_path}")
    
    return os.path.abspath(model_path)

def print_welcome_message():
    """打印欢迎信息"""
    print("\n" + "="*70)
    print("🚁 无人机连续导航系统")
    print("="*70)
    print("基于强化学习的智能无人机导航演示")
    print("支持实时目标更新、键盘交互控制、暂停/继续等功能")
    print("="*70)

def print_system_info(args, model_path: str):
    """打印系统配置信息"""
    print(f"\n📋 系统配置:")
    print(f"  模型文件: {model_path}")
    print(f"  GUI模式: {'开启' if args.gui else '关闭'}")
    print(f"  视频录制: {'开启' if args.record else '关闭'}")
    print(f"  详细输出: {'开启' if args.verbose else '关闭'}")
    
    # 显示空间配置
    space = TESTING_SPACE
    print(f"\n🌍 飞行空间:")
    print(f"  X轴范围: {space['x_range'][0]} ~ {space['x_range'][1]} 米")
    print(f"  Y轴范围: {space['y_range'][0]} ~ {space['y_range'][1]} 米") 
    print(f"  Z轴范围: {space['z_range'][0]} ~ {space['z_range'][1]} 米")
    
    # 显示控制说明
    print(f"\n🎮 控制说明:")
    print(f"  输入坐标: x y z  (例: 2.5 1.8 2.0)")
    print(f"  暂停/继续: pause / resume")
    print(f"  返回起点: home")
    print(f"  查看状态: current")  
    print(f"  显示帮助: help")
    print(f"  退出程序: exit")
    print()

def main():
    """主函数"""
    try:
        # 解析命令行参数
        args = parse_arguments()
        
        # 打印欢迎信息
        print_welcome_message()
        
        # 确定模型路径
        if args.model:
            model_path = validate_model_path(args.model)
            print(f"[模型] 使用指定模型: {model_path}")
        else:
            print(f"[模型] 正在查找最新训练模型...")
            model_path = find_latest_model(DEFAULT_OUTPUT_FOLDER)
            print(f"[模型] 找到最新模型: {model_path}")
        
        # 显示系统配置
        print_system_info(args, model_path)
        
        # 创建并初始化导航系统
        print(f"[系统] 正在初始化连续导航系统...")
        navigator = ContinuousNavigator(
            model_path=model_path,
            gui=args.gui,
            record=args.record
        )
        
        # 初始化系统组件
        navigator.initialize()
        
        # 启动导航系统
        print(f"[系统] 启动连续导航系统...")
        navigator.start_navigation()
        
    except KeyboardInterrupt:
        print(f"\n[系统] 用户中断程序运行")
        
    except FileNotFoundError as e:
        print(f"\n❌ 文件未找到错误: {e}")
        print(f"💡 请确认:")
        print(f"   1. 模型文件路径是否正确")
        print(f"   2. 是否已完成模型训练") 
        print(f"   3. 训练结果是否保存在 '{DEFAULT_OUTPUT_FOLDER}' 文件夹中")
        sys.exit(1)
        
    except ImportError as e:
        print(f"\n❌ 导入错误: {e}")
        print(f"💡 请确认:")
        print(f"   1. 是否激活了正确的conda环境")
        print(f"   2. 是否安装了所需依赖包")
        print(f"   3. 项目路径是否正确")
        sys.exit(1)
        
    except Exception as e:
        print(f"\n❌ 运行时错误: {e}")
        
        if args.verbose:
            print(f"\n🔍 详细错误信息:")
            traceback.print_exc()
        
        print(f"\n💡 故障排除建议:")
        print(f"   1. 检查模型文件是否完整")
        print(f"   2. 确认GPU/CPU资源是否足够")
        print(f"   3. 重新启动程序")
        print(f"   4. 使用 --verbose 参数查看详细错误")
        
        sys.exit(1)
        
    except:
        print(f"\n❌ 未知错误发生")
        print(f"🔍 详细错误信息:")
        traceback.print_exc()
        sys.exit(1)
        
    finally:
        print(f"\n🎉 感谢使用无人机连续导航系统!")

if __name__ == "__main__":
    main()