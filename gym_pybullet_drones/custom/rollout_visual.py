#!/usr/bin/env python3
"""
可视化观察模式 - 连续导航系统

专为观察无人机飞行轨迹设计的启动模式：
- 禁用键盘输入监听，允许PyBullet摄像头操作
- 仅通过网络接收远程控制命令
- 优化视觉观察体验

使用方法：
    终端1: python rollout_visual.py
    终端2: python -m gym_pybullet_drones.custom.remote_controller
"""

import os
import sys
import time
import argparse
import traceback

# 使用模块形式的导入方式
from gym_pybullet_drones.custom.continuous_navigator import ContinuousNavigator, find_latest_model
from gym_pybullet_drones.custom.config_continuous import *

def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="可视化观察模式 - 连续导航系统",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
观察模式说明:
  - 本模式专为观察飞行轨迹设计
  - 可自由操作PyBullet摄像头（鼠标拖拽、滚轮缩放）
  - 所有控制通过remote_controller.py进行
  - 支持实时LLM轨迹生成和可视化

使用步骤:
  1. 运行本脚本启动渲染环境
  2. 在新终端运行remote_controller.py
  3. 输入circle等命令控制无人机
        """
    )
    
    # 模型相关参数
    parser.add_argument('--model', type=str, default=None,
                       help='指定模型文件路径 (默认: 自动查找最新模型)')
    
    # 录制相关参数
    parser.add_argument('--record', action='store_true', default=False,
                       help='录制演示视频')
    
    return parser.parse_args()

class VisualNavigator(ContinuousNavigator):
    """可视化导航器 - 专为观察优化"""
    
    def __init__(self, model_path: str, record: bool = False):
        # 强制启用GUI，禁用本地键盘输入
        super().__init__(model_path=model_path, gui=True, record=record)
        self.visual_mode = True
        
    def _initialize_controllers(self):
        """重写控制器初始化 - 仅启用网络控制"""
        try:
            # 创建一个虚拟键盘控制器（不启动监听）
            from gym_pybullet_drones.custom.keyboard_controller import KeyboardController, StatusDisplayer
            self.keyboard_controller = KeyboardController()
            # 不调用 start() 方法，保持PyBullet摄像头控制
            
            # 初始化状态显示器
            self.status_displayer = StatusDisplayer(
                update_frequency=1.0/DISPLAY_CONFIG['update_frequency']
            )
            
            # 启动网络服务器
            if self.network_enabled:
                from gym_pybullet_drones.custom.continuous_navigator import NetworkCommandServer
                self.network_server = NetworkCommandServer(self)
                import threading
                self.network_thread = threading.Thread(
                    target=self.network_server.start,
                    daemon=True
                )
                self.network_thread.start()
                
            print(f"[可视化模式] ✅ 网络控制器初始化成功")
            print(f"[可视化模式] 🎮 摄像头控制: 鼠标拖拽旋转，滚轮缩放")
            print(f"[可视化模式] 📱 远程控制: 在新终端运行 remote_controller.py")
            
        except Exception as e:
            print(f"[控制器] ❌ 控制器初始化失败: {e}")
            raise
    
    def start_navigation(self):
        """重写启动导航系统 - 跳过键盘监听"""
        if self.is_running:
            print(f"[导航系统] 系统已在运行中")
            return
            
        print(f"\n" + "="*60)
        print(f"🚁 可视化导航系统启动")
        print(f"="*60)
        
        try:
            # 启动系统组件
            self.is_running = True
            self.stats['start_time'] = time.time()
            
            # 重置环境
            obs, info = self.env.reset()
            
            # 不启动键盘监听，保持PyBullet摄像头控制
            
            print(f"[导航系统] ✅ 可视化系统启动成功")
            print(f"[导航系统] 当前位置: {self.env.get_current_state()['position']}")
            print(f"[导航系统] 当前目标: {self.current_target}")
            print(f"[导航系统] 通过remote_controller.py控制无人机")
            
            # 进入主导航循环
            self._navigation_loop(obs)
            
        except KeyboardInterrupt:
            print(f"\n[导航系统] 收到中断信号，正在退出...")
        except Exception as e:
            print(f"[导航系统] ❌ 系统运行出错: {e}")
        finally:
            self._shutdown()
    
    def _process_user_commands(self):
        """重写命令处理 - 跳过键盘输入，仅处理网络命令"""
        # 可视化模式下不处理键盘输入，所有命令通过网络接收
        pass
    
    def _navigation_loop(self, initial_obs):
        """重写导航循环 - 专为可视化优化"""
        obs = initial_obs
        step_count = 0
        last_status_time = time.time()
        
        print(f"\n🎬 可视化观察模式已启动")
        print(f"   - 可自由操作摄像头观察飞行")
        print(f"   - 通过remote_controller.py控制无人机")
        print(f"   - 输入'circle'开始圆形飞行演示")
        
        while self.is_running:
            # 1. 处理用户命令（仅网络命令）
            self._process_user_commands()
            
            # 2. 检查是否应该退出 - 可视化模式下通过网络命令退出
            
            # 3. 如果暂停，则等待
            if self.paused:
                time.sleep(0.1)
                continue
            
            # 4. 执行无人机控制
            obs = self._execute_control_step(obs)
            
            # 5. 更新状态显示（降低频率，减少输出干扰）
            current_time = time.time()
            if current_time - last_status_time > 3.0:  # 每3秒显示一次状态
                self._update_status_display()
                last_status_time = current_time
            
            # 6. 记录轨迹
            if DEBUG_CONFIG['log_trajectory']:
                self._log_trajectory_point()
            
            step_count += 1
            
            # 控制循环频率
            time.sleep(1.0 / SIMULATION_CONFIG['ctrl_freq'])

def main():
    """主函数"""
    try:
        # 解析命令行参数
        args = parse_arguments()
        
        # 打印启动信息
        print("\n" + "="*70)
        print("👀 无人机可视化观察模式")
        print("="*70)
        print("专为观察飞行轨迹优化 - 支持自由摄像头操作")
        print("="*70)
        
        # 确定模型路径
        if args.model:
            if not os.path.exists(args.model):
                raise FileNotFoundError(f"模型文件不存在: {args.model}")
            model_path = os.path.abspath(args.model)
            print(f"[模型] 使用指定模型: {model_path}")
        else:
            print(f"[模型] 正在查找最新训练模型...")
            model_path = find_latest_model(DEFAULT_OUTPUT_FOLDER)
            print(f"[模型] 找到最新模型: {model_path}")
        
        # 显示配置信息
        print(f"\n📋 可视化配置:")
        print(f"  视频录制: {'开启' if args.record else '关闭'}")
        print(f"  摄像头控制: 自由操作模式")
        print(f"  控制方式: 仅网络远程控制")
        
        # 显示使用说明
        print(f"\n🎮 使用说明:")
        print(f"  1. PyBullet窗口: 鼠标操作摄像头")
        print(f"     - 左键拖拽: 旋转视角")
        print(f"     - 右键拖拽: 平移视角") 
        print(f"     - 滚轮: 缩放")
        print(f"  2. 控制无人机: 新终端运行")
        print(f"     python -m gym_pybullet_drones.custom.remote_controller")
        print(f"  3. 开始飞行: 在控制器中输入'circle'")
        
        # 创建可视化导航器
        print(f"\n[系统] 正在初始化可视化导航系统...")
        navigator = VisualNavigator(
            model_path=model_path,
            record=args.record
        )
        
        # 初始化系统组件
        navigator.initialize()
        
        # 启动导航系统
        print(f"[系统] 启动可视化观察模式...")
        navigator.start_navigation()
        
    except KeyboardInterrupt:
        print(f"\n[系统] 用户中断程序运行")
        
    except Exception as e:
        print(f"\n❌ 运行时错误: {e}")
        traceback.print_exc()
        sys.exit(1)
        
    finally:
        print(f"\n🎉 可视化观察模式结束!")

if __name__ == "__main__":
    main()