import numpy as np

from gym_pybullet_drones.envs.BaseRLAviary import BaseRLAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics, ActionType, ObservationType

class HoverAviary(BaseRLAviary):
    """单智能体强化学习问题：在指定位置悬停。"""

    ################################################################################
    
    def __init__(self,
                 drone_model: DroneModel=DroneModel.CF2X,  # 无人机模型，默认为CF2X
                 initial_xyzs=None,  # 无人机初始位置坐标，为None时使用默认值
                 initial_rpys=None,  # 无人机初始姿态角(roll-pitch-yaw)，为None时使用默认值
                 physics: Physics=Physics.PYB,  # 物理引擎类型，默认使用PyBullet
                 pyb_freq: int = 240,  # PyBullet物理引擎的更新频率，单位Hz
                 ctrl_freq: int = 30,  # 控制频率，单位Hz，通常为物理引擎频率的因子
                 gui=False,  # 是否显示PyBullet的GUI界面
                 record=False,  # 是否记录模拟视频
                 obs: ObservationType=ObservationType.KIN,  # 观测空间类型，默认为运动学信息
                 act: ActionType=ActionType.RPM  # 动作空间类型，默认为电机转速控制
                 ):
        """单智能体强化学习环境的初始化。

        使用通用的单智能体RL父类。

        参数
        ----------
        drone_model : DroneModel, 可选
            期望的无人机类型（详细信息在`assets`文件夹的.urdf文件中）。
        initial_xyzs: ndarray | None, 可选
            形状为(NUM_DRONES, 3)的数组，包含无人机的初始XYZ位置。
        initial_rpys: ndarray | None, 可选
            形状为(NUM_DRONES, 3)的数组，包含无人机的初始方向（弧度）。
        physics : Physics, 可选
            期望的PyBullet物理实现/自定义动力学。
        pyb_freq : int, 可选
            PyBullet更新的频率（必须是ctrl_freq的倍数）。
        ctrl_freq : int, 可选
            环境更新的频率。
        gui : bool, 可选
            是否使用PyBullet的GUI。
        record : bool, 可选
            是否保存模拟的视频。
        obs : ObservationType, 可选
            观测空间的类型（运动学信息或视觉）
        act : ActionType, 可选
            动作空间的类型（1D或3D；电机转速RPM、推力和力矩，或带PID控制的航点）

        """
        self.TARGET_POS = np.array([0,0,1])  # 目标位置：在原点上方1米高度
        self.EPISODE_LEN_SEC = 8  # 每个回合的最大时长，单位秒
        super().__init__(drone_model=drone_model,
                         num_drones=1,  # 单智能体环境，只有1个无人机
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

    ################################################################################
    
    def _computeReward(self):
        """计算当前奖励值。

        奖励基于无人机与目标位置的距离，距离越近奖励越高。
        使用距离的四次方作为惩罚项，然后从2中减去，保证奖励为正。

        返回
        -------
        float
            奖励值。

        """
        state = self._getDroneStateVector(0)  # 获取无人机的状态向量
        ret = max(0, 2 - np.linalg.norm(self.TARGET_POS-state[0:3])**4)  # 计算基于距离的奖励
        return ret

    ################################################################################
    
    def _computeTerminated(self):
        """计算当前回合是否终止。

        当无人机达到目标位置（距离非常小）时，回合结束。

        返回
        -------
        bool
            当前回合是否结束。

        """
        state = self._getDroneStateVector(0)  # 获取无人机的状态向量
        if np.linalg.norm(self.TARGET_POS-state[0:3]) < .0001:  # 如果无人机与目标位置的距离小于阈值
            return True  # 回合成功结束
        else:
            return False
        
    ################################################################################
    
    def _computeTruncated(self):
        """计算当前回合是否被截断。

        在以下情况回合被截断（提前结束）：
        1. 无人机飞得太远（超出边界）
        2. 无人机倾斜角度太大（不稳定）
        3. 回合时间超过最大限制

        返回
        -------
        bool
            当前回合是否被截断。

        """
        state = self._getDroneStateVector(0)  # 获取无人机的状态向量
        if (abs(state[0]) > 1.5 or abs(state[1]) > 1.5 or state[2] > 2.0  # 当无人机飞得太远时截断
             or abs(state[7]) > .4 or abs(state[8]) > .4  # 当无人机倾斜角度太大时截断
        ):
            return True
        if self.step_counter/self.PYB_FREQ > self.EPISODE_LEN_SEC:  # 如果超过最大回合时间
            return True
        else:
            return False

    ################################################################################
    
    def _computeInfo(self):
        """计算当前信息字典。

        未使用。

        返回
        -------
        dict[str, int]
            虚拟值。

        """
        return {"answer": 42} #### 由深思计算机在750万年内计算出的答案
