from enum import Enum

# 枚举类：无人机模型类型
class DroneModel(Enum):
    """无人机模型枚举类。"""
    CF2X = "cf2x"   # Bitcraze Craziflie 2.0，X型结构
    CF2P = "cf2p"   # Bitcraze Craziflie 2.0，+型结构
    RACE = "racer"  # 竞速无人机，X型结构


################################################################################

# 枚举类：物理仿真类型
class Physics(Enum):
    """物理仿真实现枚举类。"""
    PYB = "pyb"                         # 基础PyBullet物理仿真
    DYN = "dyn"                         # 显式动力学模型
    PYB_GND = "pyb_gnd"                 # 带地面效应的PyBullet物理仿真
    PYB_DRAG = "pyb_drag"               # 带空气阻力的PyBullet物理仿真
    PYB_DW = "pyb_dw"                   # 带下洗气流的PyBullet物理仿真
    PYB_GND_DRAG_DW = "pyb_gnd_drag_dw" # 同时包含地面效应、空气阻力和下洗气流的PyBullet仿真

################################################################################

# 枚举类：摄像头图像类型
class ImageType(Enum):
    """摄像头图像类型枚举类。"""
    RGB = 0     # 彩色图像（红绿蓝及透明度）
    DEP = 1     # 深度图像
    SEG = 2     # 按对象ID分割的图像
    BW = 3      # 黑白图像

################################################################################

# 枚举类：动作输入类型
class ActionType(Enum):
    """动作输入类型枚举类。"""
    RPM = "rpm"                 # 直接输入电机转速
    PID = "pid"                 # PID控制输入
    VEL = "vel"                 # 速度输入（通过PID控制实现）
    ONE_D_RPM = "one_d_rpm"     # 一维转速输入（所有电机相同转速）
    ONE_D_PID = "one_d_pid"     # 一维PID控制输入（所有电机相同PID控制）

################################################################################

# 枚举类：观测信息类型
class ObservationType(Enum):
    """观测信息类型枚举类。"""
    KIN = "kin"     # 运动学信息（位置、速度、角速度等）
    RGB = "rgb"     # 每架无人机视角下的RGB摄像头图像
