"""演示仿真环境与控制算法联合使用的脚本

仿真环境由 `CtrlAviary` 环境运行。
控制算法由 `DSLPIDControl` 中的PID控制器实现。

运行示例
-------
在终端中运行：

    $ python pid.py

说明
-----
无人机在不同高度下，绕着点(0, -0.3)在X-Y平面内沿圆形轨迹运动。

"""
# 系统导入模块
import os
import time
import argparse
from datetime import datetime
import pdb
import math
import random
# 第三方库导入
import numpy as np  # 数值计算库
import pybullet as p  # 物理仿真引擎
import matplotlib.pyplot as plt  # 绘图库

# 项目内模块导入
from gym_pybullet_drones.utils.enums import DroneModel, Physics
from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
from gym_pybullet_drones.utils.Logger import Logger
from gym_pybullet_drones.utils.utils import sync, str2bool

# 默认参数配置
DEFAULT_DRONES = DroneModel("cf2x")  # 默认无人机模型：CF2X四旋翼
DEFAULT_NUM_DRONES = 3  # 默认无人机数量：3架
DEFAULT_PHYSICS = Physics("pyb")  # 默认物理引擎：PyBullet
DEFAULT_GUI = True  # 默认开启图形界面
DEFAULT_RECORD_VISION = False  # 默认不录制视频
DEFAULT_PLOT = True  # 默认绘制仿真结果图表
DEFAULT_USER_DEBUG_GUI = False  # 默认不开启用户调试界面
DEFAULT_OBSTACLES = True  # 默认添加障碍物
DEFAULT_SIMULATION_FREQ_HZ = 240  # 默认仿真频率：240Hz
DEFAULT_CONTROL_FREQ_HZ = 48  # 默认控制频率：48Hz
DEFAULT_DURATION_SEC = 12  # 默认仿真持续时间：12秒
DEFAULT_OUTPUT_FOLDER = 'results'  # 默认结果输出文件夹
DEFAULT_COLAB = False  # 默认不在Colab环境运行

def run(
        drone=DEFAULT_DRONES,  # 无人机模型
        num_drones=DEFAULT_NUM_DRONES,  # 无人机数量
        physics=DEFAULT_PHYSICS,  # 物理引擎类型
        gui=DEFAULT_GUI,  # 是否显示图形界面
        record_video=DEFAULT_RECORD_VISION,  # 是否录制视频
        plot=DEFAULT_PLOT,  # 是否绘制结果图表
        user_debug_gui=DEFAULT_USER_DEBUG_GUI,  # 是否开启用户调试界面
        obstacles=DEFAULT_OBSTACLES,  # 是否添加障碍物
        simulation_freq_hz=DEFAULT_SIMULATION_FREQ_HZ,  # 仿真频率
        control_freq_hz=DEFAULT_CONTROL_FREQ_HZ,  # 控制频率
        duration_sec=DEFAULT_DURATION_SEC,  # 仿真持续时间
        output_folder=DEFAULT_OUTPUT_FOLDER,  # 输出文件夹路径
        colab=DEFAULT_COLAB  # 是否在Colab环境运行
        ):
    #### 初始化仿真环境 #############################
    # 定义无人机初始位置参数
    H = .1  # 基础高度：0.1m
    H_STEP = .05  # 高度步长：0.05m（每架无人机增加的高度差）
    R = .3  # 圆形轨迹半径：0.3m
    
    # 计算每架无人机的初始位置（圆形分布，不同高度）
    INIT_XYZS = np.array([[R*np.cos((i/6)*2*np.pi+np.pi/2), 
                          R*np.sin((i/6)*2*np.pi+np.pi/2)-R, 
                          H+i*H_STEP] for i in range(num_drones)])
    
    # 计算每架无人机的初始姿态角（偏航角递增分布）
    INIT_RPYS = np.array([[0, 0,  i * (np.pi/2)/num_drones] for i in range(num_drones)])

    #### 初始化圆形轨迹 ######################
    PERIOD = 10  # 一个完整圆形轨迹的周期时间（秒）
    NUM_WP = control_freq_hz*PERIOD  # 总路径点数量（控制频率 × 周期）
    
    # 生成圆形轨迹路径点坐标
    TARGET_POS = np.zeros((NUM_WP,3))  # 初始化目标位置数组
    for i in range(NUM_WP):
        # 计算圆形轨迹上每个点的X, Y坐标，Z坐标设为0（相对坐标）
        TARGET_POS[i, :] = (R*np.cos((i/NUM_WP)*(2*np.pi)+np.pi/2)+INIT_XYZS[0, 0], 
                           R*np.sin((i/NUM_WP)*(2*np.pi)+np.pi/2)-R+INIT_XYZS[0, 1], 
                           0)
    
    # 为每架无人机初始化路径点计数器（错开起始位置）
    wp_counters = np.array([int((i*NUM_WP/6)%NUM_WP) for i in range(num_drones)])

    #### 调试轨迹（备选轨迹路径） ######################################
    #### 如果需要使用，请在.computeControlFromState()中取消注释alt. target_pos
    # INIT_XYZS = np.array([[.3 * i, 0, .1] for i in range(num_drones)])
    # INIT_RPYS = np.array([[0, 0,  i * (np.pi/3)/num_drones] for i in range(num_drones)])
    # NUM_WP = control_freq_hz*15
    # TARGET_POS = np.zeros((NUM_WP,3))
    # for i in range(NUM_WP):
    #     if i < NUM_WP/6:  # 第一段：向上向前运动
    #         TARGET_POS[i, :] = (i*6)/NUM_WP, 0, 0.5*(i*6)/NUM_WP
    #     elif i < 2 * NUM_WP/6:  # 第二段：向下向后运动
    #         TARGET_POS[i, :] = 1 - ((i-NUM_WP/6)*6)/NUM_WP, 0, 0.5 - 0.5*((i-NUM_WP/6)*6)/NUM_WP
    #     elif i < 3 * NUM_WP/6:  # 第三段：向右向上运动
    #         TARGET_POS[i, :] = 0, ((i-2*NUM_WP/6)*6)/NUM_WP, 0.5*((i-2*NUM_WP/6)*6)/NUM_WP
    #     elif i < 4 * NUM_WP/6:  # 第四段：向左向下运动
    #         TARGET_POS[i, :] = 0, 1 - ((i-3*NUM_WP/6)*6)/NUM_WP, 0.5 - 0.5*((i-3*NUM_WP/6)*6)/NUM_WP
    #     elif i < 5 * NUM_WP/6:  # 第五段：对角线向上运动
    #         TARGET_POS[i, :] = ((i-4*NUM_WP/6)*6)/NUM_WP, ((i-4*NUM_WP/6)*6)/NUM_WP, 0.5*((i-4*NUM_WP/6)*6)/NUM_WP
    #     elif i < 6 * NUM_WP/6:  # 第六段：对角线向下运动
    #         TARGET_POS[i, :] = 1 - ((i-5*NUM_WP/6)*6)/NUM_WP, 1 - ((i-5*NUM_WP/6)*6)/NUM_WP, 0.5 - 0.5*((i-5*NUM_WP/6)*6)/NUM_WP
    # wp_counters = np.array([0 for i in range(num_drones)])

    #### 创建仿真环境 ################################
    env = CtrlAviary(drone_model=drone,  # 无人机模型
                     num_drones=num_drones,  # 无人机数量
                     initial_xyzs=INIT_XYZS,  # 初始位置
                     initial_rpys=INIT_RPYS,  # 初始姿态角
                     physics=physics,  # 物理引擎类型
                     neighbourhood_radius=10,  # 邻域半径（米）
                     pyb_freq=simulation_freq_hz,  # 物理仿真频率
                     ctrl_freq=control_freq_hz,  # 控制频率
                     gui=gui,  # 是否显示图形界面
                     record=record_video,  # 是否录制视频
                     obstacles=obstacles,  # 是否添加障碍物
                     user_debug_gui=user_debug_gui  # 是否显示用户调试界面
                     )

    #### 从环境中获取PyBullet客户端ID ####
    PYB_CLIENT = env.getPyBulletClient()

    #### 初始化数据记录器 #################################
    logger = Logger(logging_freq_hz=control_freq_hz,  # 记录频率
                   num_drones=num_drones,  # 无人机数量
                   output_folder=output_folder,  # 输出文件夹
                   colab=colab  # 是否在Colab环境运行
                   )

    #### 初始化控制器 ############################
    # 如果使用CF2X或CF2P无人机模型，创建对应的PID控制器
    if drone in [DroneModel.CF2X, DroneModel.CF2P]:
        ctrl = [DSLPIDControl(drone_model=drone) for i in range(num_drones)]

    #### 运行仿真 ####################################
    action = np.zeros((num_drones,4))  # 初始化动作数组（4个电机推力）
    START = time.time()  # 记录仿真开始时间
    
    # 主仿真循环：运行指定时长的仿真步骤
    for i in range(0, int(duration_sec*env.CTRL_FREQ)):

        #### 让橡皮鸭从天而降的彩蛋 #############################
        # 在仿真时间5-10秒内，每10步随机投放一个橡皮鸭
        # if i/env.SIM_FREQ>5 and i%10==0 and i/env.SIM_FREQ<10: 
        #     p.loadURDF("duck_vhacd.urdf", 
        #                [0+random.gauss(0, 0.3), -0.5+random.gauss(0, 0.3), 3], 
        #                p.getQuaternionFromEuler([random.randint(0,360), random.randint(0,360), random.randint(0,360)]), 
        #                physicsClientId=PYB_CLIENT)

        #### 执行仿真步骤 ###################################
        obs, reward, terminated, truncated, info = env.step(action)

        #### 计算当前路径点的控制指令 #############
        for j in range(num_drones):
            # 为每架无人机计算PID控制指令
            action[j, :], _, _ = ctrl[j].computeControlFromState(
                control_timestep=env.CTRL_TIMESTEP,  # 控制时间步长
                state=obs[j],  # 当前无人机状态（位置、速度、姿态等）
                target_pos=np.hstack([TARGET_POS[wp_counters[j], 0:2], INIT_XYZS[j, 2]]),  # 目标位置（圆形轨迹XY + 固定高度Z）
                # target_pos=INIT_XYZS[j, :] + TARGET_POS[wp_counters[j], :],  # 备选目标位置计算方式
                target_rpy=INIT_RPYS[j, :]  # 目标姿态角
            )

        #### 移动到下一个路径点并循环 #####################
        for j in range(num_drones):
            # 更新路径点计数器，到达终点后重新开始循环
            wp_counters[j] = wp_counters[j] + 1 if wp_counters[j] < (NUM_WP-1) else 0

        #### 记录仿真数据 ####################################
        for j in range(num_drones):
            logger.log(drone=j,  # 无人机编号
                      timestamp=i/env.CTRL_FREQ,  # 当前时间戳
                      state=obs[j],  # 无人机状态
                      control=np.hstack([TARGET_POS[wp_counters[j], 0:2], INIT_XYZS[j, 2], INIT_RPYS[j, :], np.zeros(6)])  # 控制目标
                      # control=np.hstack([INIT_XYZS[j, :]+TARGET_POS[wp_counters[j], :], INIT_RPYS[j, :], np.zeros(6)])  # 备选控制记录方式
                      )

        #### 输出信息 ##############################################
        env.render()  # 渲染仿真画面

        #### 同步仿真时间 ###################################
        if gui:
            # 如果开启了图形界面，同步实际时间与仿真时间
            sync(i, START, env.CTRL_TIMESTEP)

    #### 关闭仿真环境 #################################
    env.close()

    #### 保存仿真结果 ###########################
    logger.save()  # 保存日志文件
    logger.save_as_csv("pid")  # 可选：保存为CSV格式

    #### 绘制仿真结果图表 ###########################
    if plot:
        logger.plot()  # 绘制飞行轨迹和性能图表

if __name__ == "__main__":
    #### 定义和解析脚本的命令行参数 ##
    parser = argparse.ArgumentParser(description='使用CtrlAviary和DSLPIDControl的螺旋飞行脚本')
    parser.add_argument('--drone',              default=DEFAULT_DRONES,     type=DroneModel,    help='无人机模型（默认：CF2X）', metavar='', choices=DroneModel)
    parser.add_argument('--num_drones',         default=DEFAULT_NUM_DRONES,          type=int,           help='无人机数量（默认：3）', metavar='')
    parser.add_argument('--physics',            default=DEFAULT_PHYSICS,      type=Physics,       help='物理引擎（默认：PYB）', metavar='', choices=Physics)
    parser.add_argument('--gui',                default=DEFAULT_GUI,       type=str2bool,      help='是否使用PyBullet图形界面（默认：True）', metavar='')
    parser.add_argument('--record_video',       default=DEFAULT_RECORD_VISION,      type=str2bool,      help='是否录制视频（默认：False）', metavar='')
    parser.add_argument('--plot',               default=DEFAULT_PLOT,       type=str2bool,      help='是否绘制仿真结果（默认：True）', metavar='')
    parser.add_argument('--user_debug_gui',     default=DEFAULT_USER_DEBUG_GUI,      type=str2bool,      help='是否在GUI中添加调试线和参数（默认：False）', metavar='')
    parser.add_argument('--obstacles',          default=DEFAULT_OBSTACLES,       type=str2bool,      help='是否在环境中添加障碍物（默认：True）', metavar='')
    parser.add_argument('--simulation_freq_hz', default=DEFAULT_SIMULATION_FREQ_HZ,        type=int,           help='仿真频率，单位Hz（默认：240）', metavar='')
    parser.add_argument('--control_freq_hz',    default=DEFAULT_CONTROL_FREQ_HZ,         type=int,           help='控制频率，单位Hz（默认：48）', metavar='')
    parser.add_argument('--duration_sec',       default=DEFAULT_DURATION_SEC,         type=int,           help='仿真持续时间，单位秒（默认：5）', metavar='')
    parser.add_argument('--output_folder',     default=DEFAULT_OUTPUT_FOLDER, type=str,           help='保存日志的文件夹（默认："results"）', metavar='')
    parser.add_argument('--colab',              default=DEFAULT_COLAB, type=bool,           help='示例是否在notebook中运行（默认："False"）', metavar='')
    ARGS = parser.parse_args()

    # 运行主程序，传入解析后的参数
    run(**vars(ARGS))
