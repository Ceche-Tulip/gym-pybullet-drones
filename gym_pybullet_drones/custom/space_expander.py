"""
扩展空间的悬停环境

基于原有的obsin_HoverAviary，但放宽了空间限制，
专门用于连续导航测试，不影响训练环境。
"""

import numpy as np
from gym_pybullet_drones.envs.obsin_HoverAviary import HoverAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics, ActionType, ObservationType
from gym_pybullet_drones.custom.config_continuous import TESTING_SPACE, TARGET_TOLERANCE

class ExtendedHoverAviary(HoverAviary):
    """扩展空间的单无人机悬停环境，专门用于连续导航测试"""

    def __init__(self,
                 drone_model: DroneModel=DroneModel.CF2X,
                 initial_xyzs=None,
                 initial_rpys=None,
                 physics: Physics=Physics.PYB,
                 pyb_freq: int = 240,
                 ctrl_freq: int = 30,
                 gui=False,
                 record=False,
                 obs: ObservationType=ObservationType.KIN,
                 act: ActionType=ActionType.RPM,
                 target_pos=None,
                 obstacles=False
                 ):
        """
        初始化扩展空间的RL环境
        
        参数:
            target_pos: 目标位置 [x, y, z]，如果为None则使用默认值
            obstacles: 是否添加障碍物（仅用于测试环境，默认False）
            其他参数与父类相同
        """
        
        # ⚠️ 重要：必须在父类初始化之前设置这些属性
        # 因为父类的 __init__ 会调用 _housekeeping()，
        # 而 _housekeeping() 会调用 _addObstacles()，
        # _addObstacles() 需要访问 self.EXTENDED_SPACE
        
        # 扩展空间配置（在父类初始化前设置）
        self.EXTENDED_SPACE = TESTING_SPACE
        self.TARGET_TOLERANCE_CONFIG = TARGET_TOLERANCE
        
        # 设置目标位置 - 使用更小的测试目标
        if target_pos is not None:
            self.TARGET_POS = np.array(target_pos)
        else:
            # 使用更小、更容易到达的默认目标
            self.TARGET_POS = np.array([0.8, 0.8, 1.2])  # 较小的测试目标
        
        # 调用父类初始化
        super().__init__(
            drone_model=drone_model,
            initial_xyzs=initial_xyzs,
            initial_rpys=initial_rpys,
            physics=physics,
            pyb_freq=pyb_freq,
            ctrl_freq=ctrl_freq,
            gui=gui,
            record=record,
            obs=obs,
            act=act
        )
        
        # 保存障碍物标志（在父类初始化后设置）
        self.OBSTACLES = obstacles
        
        # 扩展episode长度以支持连续导航
        self.EPISODE_LEN_SEC = 300  # 5分钟，足够完成多个目标的连续导航
        
        print(f"[ExtendedHoverAviary] 已创建扩展空间环境")
        print(f"[ExtendedHoverAviary] 空间范围: X{self.EXTENDED_SPACE['x_range']}, Y{self.EXTENDED_SPACE['y_range']}, Z{self.EXTENDED_SPACE['z_range']}")
        print(f"[ExtendedHoverAviary] Episode时长限制: {self.EPISODE_LEN_SEC}秒")
        print(f"[ExtendedHoverAviary] 当前目标位置: {self.TARGET_POS}")

    def _computeTruncated(self):
        """
        重写截断条件，使用扩展的空间限制
        
        返回:
            bool: 是否需要截断当前episode
        """
        state = self._getDroneStateVector(0)
        
        # 获取扩展空间配置
        x_min, x_max = self.EXTENDED_SPACE['x_range']
        y_min, y_max = self.EXTENDED_SPACE['y_range'] 
        z_min, z_max = self.EXTENDED_SPACE['z_range']
        tilt_limit = self.EXTENDED_SPACE['tilt_limit']
        
        # 检查是否超出扩展空间边界
        x_out = state[0] < x_min or state[0] > x_max
        y_out = state[1] < y_min or state[1] > y_max  
        z_out = state[2] < z_min or state[2] > z_max
        tilt_out = abs(state[7]) > tilt_limit or abs(state[8]) > tilt_limit
        
        if x_out or y_out or z_out or tilt_out:
            print(f"[截断详情] 位置=({state[0]:.3f}, {state[1]:.3f}, {state[2]:.3f})")
            print(f"[截断详情] 边界: X[{x_min}, {x_max}], Y[{y_min}, {y_max}], Z[{z_min}, {z_max}]")
            print(f"[截断详情] 倾斜: roll={state[7]:.3f}, pitch={state[8]:.3f}, 限制={tilt_limit}")
            print(f"[截断详情] 超出原因: X={x_out}, Y={y_out}, Z={z_out}, 倾斜={tilt_out}")
            return True
        
        # 检查episode时长（可选的时间限制）
        current_time = self.step_counter/self.PYB_FREQ
        if current_time > self.EPISODE_LEN_SEC:
            print(f"[截断详情] 超时: 当前时间={current_time:.1f}s, 限制={self.EPISODE_LEN_SEC}s")
            return True
            
        return False
    
    def _computeTerminated(self):
        """
        重写终止条件 - 连续导航模式下不因到达目标而终止
        
        在连续导航模式下，我们希望无人机到达目标后继续悬停等待新目标，
        而不是终止episode。只有在严重错误时才终止。
        
        返回:
            bool: 是否需要终止episode（仅在严重错误时）
        """
        # 连续导航模式下不因到达目标而终止episode
        # 这样无人机可以在目标点悬停等待新目标
        return False
    
    def update_target_position(self, new_target):
        """
        更新目标位置（用于连续导航）
        
        参数:
            new_target: 新的目标位置 [x, y, z]
            
        返回:
            bool: 是否成功更新目标位置
        """
        if len(new_target) == 3:
            # 检查目标是否在有效范围内
            x, y, z = new_target
            x_min, x_max = self.EXTENDED_SPACE['x_range']
            y_min, y_max = self.EXTENDED_SPACE['y_range'] 
            z_min, z_max = self.EXTENDED_SPACE['z_range']
            
            if (x_min <= x <= x_max and y_min <= y <= y_max and z_min <= z <= z_max):
                self.TARGET_POS = np.array(new_target)
                print(f"[导航] 目标位置已更新为: ({self.TARGET_POS[0]:.2f}, {self.TARGET_POS[1]:.2f}, {self.TARGET_POS[2]:.2f})")
                return True
            else:
                print(f"[错误] 目标位置超出边界: {new_target}")
                print(f"[错误] 有效范围: X[{x_min}, {x_max}], Y[{y_min}, {y_max}], Z[{z_min}, {z_max}]")
                return False
        else:
            print(f"[错误] 无效的目标位置格式: {new_target}")
            return False
    
    def get_current_state(self):
        """
        获取当前无人机状态信息
        
        返回:
            dict: 包含位置、速度、距离目标等信息的字典
        """
        state = self._getDroneStateVector(0)
        distance_to_target = np.linalg.norm(self.TARGET_POS - state[0:3])
        
        return {
            'position': state[0:3],                    # 当前位置 [x, y, z]
            'velocity': state[10:13],                  # 当前速度 [vx, vy, vz]
            'orientation': state[7:10],                # 当前姿态 [roll, pitch, yaw]
            'target_position': self.TARGET_POS,        # 目标位置
            'distance_to_target': distance_to_target,  # 到目标距离
            'is_near_target': distance_to_target < self.TARGET_TOLERANCE_CONFIG['reach_distance'],
            'step_count': self.step_counter,           # 步数计数
            'time_elapsed': self.step_counter / self.PYB_FREQ,  # 经过时间（秒）
        }
    
    def check_safety_limits(self):
        """
        检查安全限制
        
        返回:
            tuple: (是否安全, 警告信息)
        """
        state = self._getDroneStateVector(0)
        warnings = []
        
        # 检查位置边界
        x_min, x_max = self.EXTENDED_SPACE['x_range']
        y_min, y_max = self.EXTENDED_SPACE['y_range']
        z_min, z_max = self.EXTENDED_SPACE['z_range']
        
        if state[0] <= x_min + 0.5 or state[0] >= x_max - 0.5:
            warnings.append(f"X轴接近边界: {state[0]:.2f}")
        if state[1] <= y_min + 0.5 or state[1] >= y_max - 0.5:
            warnings.append(f"Y轴接近边界: {state[1]:.2f}")
        if state[2] <= z_min + 0.2 or state[2] >= z_max - 0.5:
            warnings.append(f"Z轴接近边界: {state[2]:.2f}")
            
        # 检查倾斜角度
        tilt_limit = self.EXTENDED_SPACE['tilt_limit']
        if abs(state[7]) > tilt_limit * 0.8 or abs(state[8]) > tilt_limit * 0.8:
            warnings.append(f"倾斜角度过大: roll={state[7]:.2f}, pitch={state[8]:.2f}")
        
        # 检查速度
        velocity = np.linalg.norm(state[10:13])
        if velocity > 2.5:  # 速度限制
            warnings.append(f"飞行速度过快: {velocity:.2f}m/s")
        
        is_safe = len(warnings) == 0
        warning_message = "; ".join(warnings) if warnings else "飞行状态正常"
        
        return is_safe, warning_message
    
    def _addObstacles(self):
        """
        在连续导航测试环境中添加静态障碍物
        
        根据TESTING_SPACE的尺寸合理布置障碍物，避免遮挡起点和常用路径。
        当前环境大小: X[-1.5, 1.5], Y[-1.5, 1.5], Z[0.05, 2.5]
        
        障碍物布置策略：
        - 两个对称的圆柱体，位于x=0轴线上
        - Y轴位置互为相反数，形成对称布局
        - 高度相同，测试无人机穿越能力
        """
        import pybullet as p
        
        # 确保有容器存储障碍物ID
        self.OBSTACLE_IDS = []
        
        # 获取空间范围
        x_min, x_max = self.EXTENDED_SPACE['x_range']
        y_min, y_max = self.EXTENDED_SPACE['y_range']
        z_min, z_max = self.EXTENDED_SPACE['z_range']
        
        print(f"[障碍物] 正在创建障碍物...")
        print(f"[障碍物] 环境范围: X[{x_min}, {x_max}], Y[{y_min}, {y_max}], Z[{z_min}, {z_max}]")
        
        # 圆柱体参数配置
        cyl_radius = 0.10        # 圆柱半径 10cm
        cyl_height = 1.0         # 圆柱高度 1.0m
        y_distance = 0.4         # Y轴距离中心的距离（两柱间距为1.2m）
        
        # ==================== 障碍物 1: 蓝色圆柱体 (左侧) ====================
        col_cyl1 = p.createCollisionShape(
            p.GEOM_CYLINDER,
            radius=cyl_radius,
            height=cyl_height,
            physicsClientId=self.CLIENT
        )
        vis_cyl1 = p.createVisualShape(
            p.GEOM_CYLINDER,
            radius=cyl_radius,
            length=cyl_height,
            rgbaColor=[0, 0.5, 1, 0.8],  # 蓝色，稍透明
            physicsClientId=self.CLIENT
        )
        cyl1_id = p.createMultiBody(
            baseMass=0,  # 静态物体
            baseCollisionShapeIndex=col_cyl1,
            baseVisualShapeIndex=vis_cyl1,
            basePosition=[0.0, -y_distance, cyl_height/2],  # x=0, y=-0.6, z=0.5
            physicsClientId=self.CLIENT
        )
        self.OBSTACLE_IDS.append(cyl1_id)
        print(f"[障碍物] ✅ 创建蓝色圆柱 (左侧) @ (0.0, {-y_distance:.1f}, {cyl_height/2:.2f})")
        
        # ==================== 障碍物 2: 红色圆柱体 (右侧) ====================
        col_cyl2 = p.createCollisionShape(
            p.GEOM_CYLINDER,
            radius=cyl_radius,
            height=cyl_height,
            physicsClientId=self.CLIENT
        )
        vis_cyl2 = p.createVisualShape(
            p.GEOM_CYLINDER,
            radius=cyl_radius,
            length=cyl_height,
            rgbaColor=[1, 0.2, 0.2, 0.8],  # 红色
            physicsClientId=self.CLIENT
        )
        cyl2_id = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=col_cyl2,
            baseVisualShapeIndex=vis_cyl2,
            basePosition=[0.0, y_distance, cyl_height/2],  # x=0, y=+0.6, z=0.5
            physicsClientId=self.CLIENT
        )
        self.OBSTACLE_IDS.append(cyl2_id)
        print(f"[障碍物] ✅ 创建红色圆柱 (右侧) @ (0.0, {y_distance:.1f}, {cyl_height/2:.2f})")
        
        print(f"[障碍物] 🎯 共创建 {len(self.OBSTACLE_IDS)} 个对称障碍物")
        print(f"[障碍物] 两柱间距: {y_distance * 2:.1f}m (可供无人机穿越)")
        print(f"[障碍物] 障碍物高度: {cyl_height:.1f}m, 中心高度: {cyl_height/2:.1f}m")