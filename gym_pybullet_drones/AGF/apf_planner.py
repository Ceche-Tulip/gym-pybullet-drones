"""
APF (Artificial Potential Field) 人工势场路径规划器

实现基于势场法的避障路径规划：
- 目标点产生引力场
- 障碍物产生斥力场
- 合力方向指示无人机运动方向
"""

import numpy as np
from typing import List, Dict, Tuple, Optional


class APFPlanner:
    """人工势场路径规划器"""
    
    def __init__(self,
                 k_att: float = 1.0,      # 引力系数
                 k_rep: float = 0.8,      # 斥力系数（降低以减少局部极小值）
                 d0: float = 0.6,         # 斥力影响范围（米）（增大以平滑斥力梯度）
                 step_size: float = 0.2,  # 路径步长（米）
                 goal_threshold: float = 0.1  # 目标距离阈值
                 ):
        """
        初始化APF规划器
        
        参数:
            k_att: 引力增益系数，控制目标吸引力强度
            k_rep: 斥力增益系数，控制障碍物排斥力强度
            d0: 障碍物影响范围，超过此距离斥力为0
            step_size: 每次规划的步长
            goal_threshold: 认为到达目标的距离阈值
        """
        self.k_att = k_att
        self.k_rep = k_rep
        self.d0 = d0
        self.step_size = step_size
        self.goal_threshold = goal_threshold
        
        # 统计信息
        self.stats = {
            'total_waypoints': 0,
            'attractive_force_avg': 0.0,
            'repulsive_force_avg': 0.0,
            'collision_warnings': 0
        }
        
        print(f"[APF规划器] 初始化完成")
        print(f"[APF规划器] 引力系数: {k_att}, 斥力系数: {k_rep}")
        print(f"[APF规划器] 影响范围: {d0}m, 步长: {step_size}m")
    
    def compute_next_waypoint(self, 
                             current_pos: np.ndarray, 
                             target_pos: np.ndarray, 
                             obstacles: List[Dict]) -> Tuple[np.ndarray, Dict]:
        """
        计算下一个避障路径点
        
        参数:
            current_pos: 当前位置 [x, y, z]
            target_pos: 目标位置 [x, y, z]
            obstacles: 障碍物列表，每个障碍物格式：
                      {'position': [x,y,z], 'radius': r, 'height': h}
        
        返回:
            next_waypoint: 下一个路径点 [x, y, z]
            info: 规划信息字典
        """
        current_pos = np.array(current_pos)
        target_pos = np.array(target_pos)
        
        # 检查是否已到达目标
        dist_to_goal = np.linalg.norm(target_pos - current_pos)
        if dist_to_goal < self.goal_threshold:
            return target_pos, {'reached': True, 'force_info': {}}
        
        # 1. 计算目标引力
        f_att = self._compute_attractive_force(current_pos, target_pos)
        
        # 2. 计算障碍物斥力
        f_rep = self._compute_repulsive_force(current_pos, target_pos, obstacles)
        
        # 3. 合力
        f_total = f_att + f_rep
        
        # 4. 计算下一个路径点
        # 如果合力太小，直接朝目标前进
        if np.linalg.norm(f_total) < 1e-6:
            direction = self._normalize(target_pos - current_pos)
        else:
            direction = self._normalize(f_total)
        
        # 限制步长
        actual_step = min(self.step_size, dist_to_goal)
        next_waypoint = current_pos + direction * actual_step
        
        # 更新统计
        self.stats['total_waypoints'] += 1
        self.stats['attractive_force_avg'] = (
            0.9 * self.stats['attractive_force_avg'] + 0.1 * np.linalg.norm(f_att)
        )
        self.stats['repulsive_force_avg'] = (
            0.9 * self.stats['repulsive_force_avg'] + 0.1 * np.linalg.norm(f_rep)
        )
        
        # 碰撞检测警告
        if self._check_collision_risk(current_pos, obstacles):
            self.stats['collision_warnings'] += 1
        
        # 返回信息
        info = {
            'reached': False,
            'force_info': {
                'attractive': f_att,
                'repulsive': f_rep,
                'total': f_total,
                'direction': direction
            },
            'distance_to_goal': dist_to_goal,
            'step_size': actual_step
        }
        
        return next_waypoint, info
    
    def _compute_attractive_force(self, 
                                  current_pos: np.ndarray, 
                                  target_pos: np.ndarray) -> np.ndarray:
        """
        计算目标点的引力
        
        引力模型：F_att = k_att * (target - current)
        引力方向指向目标，大小与距离成正比
        """
        return self.k_att * (target_pos - current_pos)
    
    def _compute_repulsive_force(self, 
                                current_pos: np.ndarray, 
                                target_pos: np.ndarray,
                                obstacles: List[Dict]) -> np.ndarray:
        """
        计算障碍物的斥力
        
        斥力模型：
        - 距离 < d0 时：F_rep = k_rep * (1/d - 1/d0) * (1/d^2) * direction
        - 距离 >= d0 时：F_rep = 0
        
        其中 d 是到障碍物的距离，direction 是远离障碍物的方向
        """
        f_rep_total = np.zeros(3)
        
        for obstacle in obstacles:
            obs_pos = np.array(obstacle['position'])
            obs_radius = obstacle['radius']
            obs_height = obstacle.get('height', 1.0)
            
            # 计算到障碍物表面的距离（考虑圆柱体）
            distance_info = self._distance_to_cylinder(
                current_pos, obs_pos, obs_radius, obs_height
            )
            
            d = distance_info['distance']
            
            # 只有在影响范围内才有斥力
            if d < self.d0:
                # 避免除零
                if d < 0.01:
                    d = 0.01
                
                # 斥力大小：距离越近，斥力越大
                magnitude = self.k_rep * (1.0/d - 1.0/self.d0) * (1.0/(d**2))
                
                # 斥力方向：远离障碍物
                direction = distance_info['direction']
                
                f_rep = magnitude * direction
                f_rep_total += f_rep
        
        return f_rep_total
    
    def _distance_to_cylinder(self, 
                             point: np.ndarray, 
                             cylinder_center: np.ndarray,
                             radius: float, 
                             height: float) -> Dict:
        """
        计算点到圆柱体的最短距离
        
        返回:
            {'distance': float, 'direction': np.ndarray, 'closest_point': np.ndarray}
        """
        # 圆柱体中心在地面，高度方向向上
        cyl_x, cyl_y, cyl_z = cylinder_center
        px, py, pz = point
        
        # 计算XY平面上到圆柱轴线的距离
        dx = px - cyl_x
        dy = py - cyl_y
        dist_xy = np.sqrt(dx**2 + dy**2)
        
        # 计算Z方向的位置关系
        # 圆柱底部在 cyl_z，顶部在 cyl_z + height
        cylinder_bottom = cyl_z
        cylinder_top = cyl_z + height
        
        # 情况1: 点在圆柱高度范围内
        if cylinder_bottom <= pz <= cylinder_top:
            if dist_xy <= radius:
                # 点在圆柱内部
                distance = radius - dist_xy
                # 方向：径向向外
                if dist_xy < 0.01:
                    direction = np.array([1.0, 0.0, 0.0])
                else:
                    direction = np.array([dx/dist_xy, dy/dist_xy, 0.0])
            else:
                # 点在圆柱外部，侧面最近
                distance = dist_xy - radius
                direction = np.array([dx/dist_xy, dy/dist_xy, 0.0])
        
        # 情况2: 点在圆柱上方
        elif pz > cylinder_top:
            if dist_xy <= radius:
                # 在圆柱正上方
                distance = pz - cylinder_top
                direction = np.array([0.0, 0.0, 1.0])
            else:
                # 到顶部边缘的距离
                edge_x = cyl_x + (dx/dist_xy) * radius if dist_xy > 0.01 else cyl_x
                edge_y = cyl_y + (dy/dist_xy) * radius if dist_xy > 0.01 else cyl_y
                edge_z = cylinder_top
                
                diff = point - np.array([edge_x, edge_y, edge_z])
                distance = np.linalg.norm(diff)
                direction = diff / (distance + 1e-6)
        
        # 情况3: 点在圆柱下方
        else:
            if dist_xy <= radius:
                # 在圆柱正下方
                distance = cylinder_bottom - pz
                direction = np.array([0.0, 0.0, -1.0])
            else:
                # 到底部边缘的距离
                edge_x = cyl_x + (dx/dist_xy) * radius if dist_xy > 0.01 else cyl_x
                edge_y = cyl_y + (dy/dist_xy) * radius if dist_xy > 0.01 else cyl_y
                edge_z = cylinder_bottom
                
                diff = point - np.array([edge_x, edge_y, edge_z])
                distance = np.linalg.norm(diff)
                direction = diff / (distance + 1e-6)
        
        return {
            'distance': max(distance, 0.0),
            'direction': direction,
            'closest_point': point - direction * distance
        }
    
    def _normalize(self, vector: np.ndarray) -> np.ndarray:
        """归一化向量"""
        norm = np.linalg.norm(vector)
        if norm < 1e-6:
            return np.zeros_like(vector)
        return vector / norm
    
    def _check_collision_risk(self, 
                              current_pos: np.ndarray, 
                              obstacles: List[Dict]) -> bool:
        """检查是否有碰撞风险"""
        warning_distance = 0.15  # 15cm内警告
        
        for obstacle in obstacles:
            obs_pos = np.array(obstacle['position'])
            obs_radius = obstacle['radius']
            obs_height = obstacle.get('height', 1.0)
            
            distance_info = self._distance_to_cylinder(
                current_pos, obs_pos, obs_radius, obs_height
            )
            
            if distance_info['distance'] < warning_distance:
                return True
        
        return False
    
    def get_stats(self) -> Dict:
        """获取统计信息"""
        return self.stats.copy()
    
    def reset_stats(self):
        """重置统计信息"""
        self.stats = {
            'total_waypoints': 0,
            'attractive_force_avg': 0.0,
            'repulsive_force_avg': 0.0,
            'collision_warnings': 0
        }
