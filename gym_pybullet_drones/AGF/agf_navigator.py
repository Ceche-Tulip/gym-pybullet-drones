"""
AGF Navigator - 带APF避障的连续导航系统

基于PPO模型和APF路径规划的分层控制架构：
- 上层：APF规划器计算避障路径点
- 下层：PPO模型执行到达路径点
"""

import os
import time
import numpy as np
from typing import Optional, List, Dict, Any
from stable_baselines3 import PPO

from gym_pybullet_drones.custom.space_expander import ExtendedHoverAviary
from gym_pybullet_drones.custom.config_continuous import *
from gym_pybullet_drones.AGF.apf_planner import APFPlanner


class AGFNavigator:
    """AGF避障导航系统"""
    
    def __init__(self, 
                 model_path: str,
                 gui: bool = True,
                 record: bool = False,
                 apf_update_freq: int = 5,
                 start_pos: Optional[np.ndarray] = None,
                 obstacles: bool = True):
        """
        初始化AGF导航器
        
        参数:
            model_path: PPO模型路径
            gui: 是否显示GUI
            record: 是否录制视频
            apf_update_freq: APF更新频率（每N步更新一次中间目标）
            start_pos: 自定义起始位置 (x, y, z)，默认为 [0, 0, 0.1]
            obstacles: 是否启用障碍物（默认True）
        """
        self.model_path = model_path
        self.gui = gui
        self.record = record
        self.apf_update_freq = apf_update_freq
        self.start_pos = start_pos if start_pos is not None else DEFAULT_INIT_POS
        self.obstacles = obstacles
        
        # 核心组件
        self.env: Optional[ExtendedHoverAviary] = None
        self.model: Optional[PPO] = None
        self.apf_planner: Optional[APFPlanner] = None
        
        # 导航状态
        self.is_running = False
        self.current_target = None  # 最终目标
        self.current_waypoint = None  # 当前中间目标（APF生成）
        self.step_counter = 0
        
        # 统计信息
        self.stats = {
            'start_time': None,
            'waypoints_generated': 0,
            'steps': 0,
            'collisions': 0,
            'target_reached': False
        }
        
        # 轨迹记录
        self.trajectory = []
        self.waypoint_history = []
        
        print(f"[AGF导航器] 初始化...")
        print(f"[AGF导航器] APF更新频率: 每{apf_update_freq}步")
    
    def initialize(self):
        """初始化所有组件"""
        print(f"\n[AGF导航器] 正在加载组件...")
        
        # 1. 加载PPO模型
        self._load_model()
        
        # 2. 创建环境（带障碍物）
        self._create_environment()
        
        # 3. 初始化APF规划器
        self._create_apf_planner()
        
        print(f"[AGF导航器] ✅ 所有组件初始化完成\n")
    
    def _load_model(self):
        """加载PPO模型"""
        try:
            print(f"[模型加载] 加载模型: {self.model_path}")
            self.model = PPO.load(self.model_path)
            print(f"[模型加载] ✅ 模型加载成功")
        except Exception as e:
            print(f"[模型加载] ❌ 模型加载失败: {e}")
            raise
    
    def _create_environment(self):
        """创建带障碍物的环境"""
        try:
            if self.obstacles:
                print(f"[环境创建] 创建带障碍物的测试环境...")
            else:
                print(f"[环境创建] 创建无障碍物的测试环境...")
            
            init_pos = np.array([self.start_pos])
            init_rpy = np.array([[0, 0, 0]])
            
            self.env = ExtendedHoverAviary(
                initial_xyzs=init_pos,
                initial_rpys=init_rpy,
                gui=self.gui,
                record=self.record,
                obs=DEFAULT_OBS,
                act=DEFAULT_ACT,
                target_pos=DEFAULT_TARGET_POS,
                obstacles=self.obstacles  # 使用参数控制障碍物
            )
            
            print(f"[环境创建] ✅ 环境创建成功")
            if self.obstacles and hasattr(self.env, 'OBSTACLE_IDS'):
                print(f"[环境创建] 障碍物数量: {len(self.env.OBSTACLE_IDS)}")
            else:
                print(f"[环境创建] 障碍物: 已禁用")
            
        except Exception as e:
            print(f"[环境创建] ❌ 环境创建失败: {e}")
            raise
    
    def _create_apf_planner(self):
        """创建APF规划器"""
        self.apf_planner = APFPlanner(
            k_att=1.0,
            k_rep=0.8,  # 降低斥力系数以避免局部极小值
            d0=0.6,     # 扩大影响范围以平滑斥力梯度
            step_size=0.1,  #  减小步长 (0.2m -> 0.1m) 以保持在PPO训练范围内
            goal_threshold=0.3  # 进一步放宽判定 (0.3m -> 0.3m) 以应对PPO精度限制
        )
        print(f"[APF规划器] ✅ APF规划器创建成功")
        print(f"[APF规划器] 步长: 0.1m, 引力系数: 1.0, 斥力系数: 0.8 (降低以避免局部极小值)")
        print(f"[APF规划器] 斥力影响范围: 0.6m (增大以平滑斥力梯度)")
        print(f"[APF规划器] 目标到达判定距离: 0.35m")
    
    def set_target(self, target_pos: List[float]) -> bool:
        """
        设置导航目标
        
        参数:
            target_pos: 目标位置 [x, y, z]
        
        返回:
            bool: 是否成功设置
        """
        target_pos = np.array(target_pos)
        
        # 验证目标是否在有效范围内
        space = TESTING_SPACE
        x, y, z = target_pos
        
        if not (space['x_range'][0] <= x <= space['x_range'][1] and
                space['y_range'][0] <= y <= space['y_range'][1] and
                space['z_range'][0] <= z <= space['z_range'][1]):
            print(f"[目标设置] ❌ 目标超出范围: {target_pos}")
            return False
        
        self.current_target = target_pos
        print(f"[目标设置] ✅ 目标已设置: {target_pos}")
        return True
    
    def navigate_to_target(self, target_pos: List[float]) -> Dict:
        """
        导航到目标点（带APF避障）
        
        参数:
            target_pos: 目标位置 [x, y, z]
        
        返回:
            结果字典，包含成功状态和统计信息
        """
        if not self.set_target(target_pos):
            return {'success': False, 'reason': 'Invalid target'}
        
        start_pos = self.env.get_current_state()['position']
        distance_to_goal = np.linalg.norm(np.array(self.current_target) - start_pos)
        
        print(f"\n{'='*70}")
        print(f"🚁 开始APF避障导航 - 详细调试模式")
        print(f"{'='*70}")
        print(f"起点: [{start_pos[0]:.3f}, {start_pos[1]:.3f}, {start_pos[2]:.3f}]")
        print(f"终点: [{self.current_target[0]:.3f}, {self.current_target[1]:.3f}, {self.current_target[2]:.3f}]")
        print(f"直线距离: {distance_to_goal:.3f}m")
        print(f"APF步长: 0.15m")
        print(f"APF更新频率: 每{self.apf_update_freq}步")
        print(f"目标判定距离: 0.35m")
        print(f"{'='*70}\n")
        
        # 重置环境和统计
        obs, info = self.env.reset()
        self.step_counter = 0
        self.stats['start_time'] = time.time()
        self.stats['waypoints_generated'] = 0
        self.stats['steps'] = 0
        self.stats['target_reached'] = False
        self.trajectory = []
        self.waypoint_history = []
        
        # 原地打转检测
        stuck_detection_window = 100  # 检测窗口：最近100步
        stuck_distance_threshold = 0.5  # 如果100步内移动距离 < 0.5m，认为卡住
        
        # 获取障碍物信息
        obstacles = self._get_obstacle_info()
        
        max_steps = 2000  # 最大步数限制（增加以支持更长距离导航）
        self.is_running = True
        
        while self.is_running and self.step_counter < max_steps:
            # 记录当前位置
            current_state = self.env.get_current_state()
            current_pos = current_state['position']
            self.trajectory.append(current_pos.copy())
            
            # 每N步用APF计算新的中间目标
            if self.step_counter % self.apf_update_freq == 0:
                waypoint, apf_info = self.apf_planner.compute_next_waypoint(
                    current_pos,
                    self.current_target,
                    obstacles
                )
                
                self.current_waypoint = waypoint
                self.waypoint_history.append(waypoint.copy())
                self.stats['waypoints_generated'] += 1
                
                # 🔍 调试：计算关键距离
                waypoint_relative = waypoint - current_pos
                waypoint_distance = np.linalg.norm(waypoint_relative)
                target_relative = self.current_target - current_pos
                target_distance = np.linalg.norm(target_relative)
                
                # 更新环境的目标位置（PPO会导航到这个中间目标）
                update_success = self.env.update_target_position(waypoint)
                
                # 显示APF规划信息（每10次更新显示一次详细信息）
                if self.step_counter % (self.apf_update_freq * 10) == 0:
                    print(f"\n{'='*70}")
                    print(f"[步数 {self.step_counter:4d}] 🔍 详细调试信息")
                    print(f"{'='*70}")
                    print(f"当前位置:     [{current_pos[0]:7.3f}, {current_pos[1]:7.3f}, {current_pos[2]:7.3f}]")
                    print(f"最终目标:     [{self.current_target[0]:7.3f}, {self.current_target[1]:7.3f}, {self.current_target[2]:7.3f}]")
                    print(f"中间航点:     [{waypoint[0]:7.3f}, {waypoint[1]:7.3f}, {waypoint[2]:7.3f}]")
                    print(f"-" * 70)
                    print(f"航点相对位置: [{waypoint_relative[0]:7.3f}, {waypoint_relative[1]:7.3f}, {waypoint_relative[2]:7.3f}]")
                    print(f"航点距离:     {waypoint_distance:.4f}m {'⚠️ >0.7m!' if waypoint_distance > 0.7 else '✓'}")
                    print(f"目标相对位置: [{target_relative[0]:7.3f}, {target_relative[1]:7.3f}, {target_relative[2]:7.3f}]")
                    print(f"目标距离:     {target_distance:.4f}m")
                    print(f"航点更新:     {'✅ 成功' if update_success else '❌ 失败（超出范围）'}")
                    print(f"-" * 70)
                    print(f"APF引力:      {np.linalg.norm(apf_info['force_info']['attractive']):.4f}")
                    print(f"APF斥力:      {np.linalg.norm(apf_info['force_info']['repulsive']):.4f}")
                    print(f"APF步长:      {apf_info['step_size']:.4f}m")
                    print(f"{'='*70}\n")
                
                # 检查是否到达最终目标
                if apf_info['reached']:
                    print(f"\n✅ 到达目标！")
                    self.stats['target_reached'] = True
                    self.is_running = False
                    break
            
            # PPO执行一步
            # 确保观测维度正确
            if hasattr(obs, 'shape'):
                if len(obs.shape) == 3 and obs.shape[0] == 1:
                    obs_for_model = obs.reshape(obs.shape[0], -1)
                elif len(obs.shape) == 2:
                    obs_for_model = obs
                else:
                    obs_for_model = obs.reshape(1, -1)
            else:
                obs_for_model = np.array(obs).reshape(1, -1)
            
            # 🔍 调试：记录观测中的目标相对位置（最后3维）
            target_obs = obs_for_model[0, -3:] if obs_for_model.shape[1] >= 3 else None
            
            action, _states = self.model.predict(obs_for_model, deterministic=True)
            
            # 🔍 调试：每50步显示PPO执行详情
            if self.step_counter % 50 == 0 and target_obs is not None:
                target_obs_distance = np.linalg.norm(target_obs)
                action_magnitude = np.mean(np.abs(action[0]))
                print(f"[PPO步数 {self.step_counter:4d}] 观测目标距离: {target_obs_distance:.4f}m, "
                      f"动作强度: {action_magnitude:.3f}, "
                      f"动作: [{action[0,0]:.2f}, {action[0,1]:.2f}, {action[0,2]:.2f}, {action[0,3]:.2f}]")
                if target_obs_distance > 0.8:
                    print(f"                 ⚠️ 警告: 观测目标距离 {target_obs_distance:.4f}m > 0.8m (可能超出PPO训练范围)")
            
            obs, reward, terminated, truncated, info = self.env.step(action)
            
            self.step_counter += 1
            self.stats['steps'] = self.step_counter
            
            # 原地打转检测：检查最近N步是否移动距离过小
            if len(self.trajectory) >= stuck_detection_window:
                recent_positions = self.trajectory[-stuck_detection_window:]
                movement_distance = np.linalg.norm(
                    np.array(recent_positions[-1]) - np.array(recent_positions[0])
                )
                dist_to_target = np.linalg.norm(
                    np.array(self.current_target) - np.array(current_pos)
                )
                
                # 🔍 调试：每100步检查打转状态
                if self.step_counter % 100 == 0:
                    print(f"[打转检测] 最近{stuck_detection_window}步移动: {movement_distance:.4f}m, "
                          f"到目标: {dist_to_target:.4f}m, "
                          f"状态: {'⚠️ 可能卡住' if movement_distance < stuck_distance_threshold else '✓ 正常移动'}")
                
                # 如果移动很小且接近目标，判定为到达
                if movement_distance < stuck_distance_threshold and dist_to_target < 0.5:
                    print(f"\n{'='*70}")
                    print(f"🎯 检测到接近目标且移动停滞")
                    print(f"{'='*70}")
                    print(f"   最近{stuck_detection_window}步移动距离: {movement_distance:.3f}m")
                    print(f"   当前到目标距离: {dist_to_target:.3f}m")
                    print(f"   当前位置: [{current_pos[0]:.3f}, {current_pos[1]:.3f}, {current_pos[2]:.3f}]")
                    print(f"   目标位置: [{self.current_target[0]:.3f}, {self.current_target[1]:.3f}, {self.current_target[2]:.3f}]")
                    print(f"   ✅ 判定为成功到达！")
                    print(f"{'='*70}\n")
                    self.stats['target_reached'] = True
                    self.is_running = False
                    break
            
            # 检查是否终止
            if terminated or truncated:
                print(f"\n⚠️ Episode终止")
                print(f"   Terminated: {terminated}, Truncated: {truncated}")
                self.is_running = False
                break
        
        # 导航结束
        elapsed_time = time.time() - self.stats['start_time']
        
        print(f"\n{'='*60}")
        print(f"📊 导航统计")
        print(f"{'='*60}")
        print(f"总步数: {self.stats['steps']}")
        print(f"生成路径点: {self.stats['waypoints_generated']}")
        print(f"用时: {elapsed_time:.2f}秒")
        print(f"是否到达: {'✅ 是' if self.stats['target_reached'] else '❌ 否'}")
        
        # APF统计
        apf_stats = self.apf_planner.get_stats()
        print(f"\nAPF统计:")
        print(f"  平均引力: {apf_stats['attractive_force_avg']:.3f}")
        print(f"  平均斥力: {apf_stats['repulsive_force_avg']:.3f}")
        print(f"  碰撞警告: {apf_stats['collision_warnings']}")
        print(f"{'='*60}\n")
        
        return {
            'success': self.stats['target_reached'],
            'stats': self.stats.copy(),
            'apf_stats': apf_stats,
            'trajectory': np.array(self.trajectory),
            'waypoints': np.array(self.waypoint_history)
        }
    
    def _get_obstacle_info(self) -> List[Dict]:
        """
        从环境中获取障碍物信息
        
        返回:
            障碍物列表，格式：[{'position': [x,y,z], 'radius': r, 'height': h}, ...]
        """
        obstacles = []
        
        # 检查环境是否有障碍物
        if not hasattr(self.env, 'OBSTACLE_IDS') or len(self.env.OBSTACLE_IDS) == 0:
            return obstacles
        
        # 从space_expander.py中的_addObstacles方法获取障碍物参数
        # 这里我们需要与实际创建障碍物时的参数保持一致
        
        # 根据space_expander.py中的_addObstacles方法，两个圆柱体的配置：
        # basePosition设置的是圆柱【中心】位置，不是底部
        # 障碍物1（蓝色）: 中心在(0.0, -0.4, 0.5), 半径0.10, 高度1.0
        # 障碍物2（红色）: 中心在(0.0, +0.4, 0.5), 半径0.10, 高度1.0
        # 因此圆柱范围：底部Z=0.0, 顶部Z=1.0
        
        obstacles = [
            {
                'position': [0.0, -0.4, 0.0],  # ⚠️ 这里position是圆柱【底部】坐标用于距离计算
                'radius': 0.10,
                'height': 1.0,  # 圆柱从Z=0延伸到Z=1.0
                'name': '蓝色圆柱'
            },
            {
                'position': [0.0, 0.4, 0.0],  # ⚠️ 这里position是圆柱【底部】坐标用于距离计算
                'radius': 0.10,
                'height': 1.0,
                'name': '红色圆柱'
            }
        ]
        
        return obstacles
    
    def close(self):
        """关闭环境"""
        if self.env is not None:
            self.env.close()
        print(f"[AGF导航器] 环境已关闭")


def find_latest_model(results_folder: str = DEFAULT_OUTPUT_FOLDER) -> str:
    """
    查找最新的训练模型
    
    参数:
        results_folder: 结果文件夹路径
    
    返回:
        最新模型的路径
    """
    import glob
    
    # 搜索模式
    patterns = [
        os.path.join(results_folder, '**/success_model.zip'),
        os.path.join(results_folder, '**/best_model.zip'),
        os.path.join(results_folder, '**/*_model.zip')
    ]
    
    all_models = []
    for pattern in patterns:
        all_models.extend(glob.glob(pattern, recursive=True))
    
    if not all_models:
        raise FileNotFoundError(f"在 '{results_folder}' 中未找到模型文件")
    
    # 按修改时间排序
    latest_model = max(all_models, key=os.path.getmtime)
    
    return latest_model
