"""脚本展示了如何使用 `gym_pybullet_drones` 的 Gymnasium 接口。

使用 HoverAviary 和 MultiHoverAviary 类作为 PPO 算法的学习环境。

示例
-------
在终端中运行：

    $ python learn.py --multiagent false
    $ python learn.py --multiagent true

注释
-----
这是一个集成 `gym-pybullet-drones` 和强化学习库 `stable-baselines3` 的最小工作示例。

"""
from datetime import datetime
import os
import time
import argparse
import numpy as np
from stable_baselines3 import PPO # 导入 PPO 算法 - stable_baselines3 是主流的 RL 库
from stable_baselines3.common.env_util import make_vec_env # 用于创建向量化环境
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold # 训练回调函数
from stable_baselines3.common.evaluation import evaluate_policy # 策略评估工具

# 导入项目特定的模块 - 这些都在 gym_pybullet_drones 包中定义
from gym_pybullet_drones.utils.Logger import Logger # 用于记录训练日志和绘图
from gym_pybullet_drones.envs.HoverAviary import HoverAviary # 单智能体悬停环境
from gym_pybullet_drones.envs.MultiHoverAviary import MultiHoverAviary # 多智能体悬停环境
from gym_pybullet_drones.utils.utils import sync, str2bool # 同步工具和类型转换
from gym_pybullet_drones.utils.enums import ObservationType, ActionType # 观测和动作类型枚举

# 默认配置参数
DEFAULT_GUI = True                    # 是否显示 PyBullet GUI 界面
DEFAULT_RECORD_VIDEO = False          # 是否录制视频
DEFAULT_OUTPUT_FOLDER = 'results'     # 结果保存文件夹
DEFAULT_COLAB = False                 # 是否在 Colab 环境中运行

DEFAULT_OBS = ObservationType('kin')  # 观测类型：'kin'(运动学) 或 'rgb'(图像)
DEFAULT_ACT = ActionType('one_d_rpm') # 动作类型：'rpm'、'pid'、'vel'、'one_d_rpm'、'one_d_pid'
DEFAULT_AGENTS = 2                    # 多智能体模式下的无人机数量
DEFAULT_MA = False                    # 是否使用多智能体模式

def run(multiagent=DEFAULT_MA, output_folder=DEFAULT_OUTPUT_FOLDER, gui=DEFAULT_GUI, plot=True, colab=DEFAULT_COLAB, record_video=DEFAULT_RECORD_VIDEO, local=True):
    """
    主要的训练和测试函数
    
    参数：
    - multiagent: 是否使用多智能体模式
    - output_folder: 结果输出文件夹
    - gui: 是否显示GUI
    - plot: 是否绘制结果图表
    - colab: 是否在Colab环境运行
    - record_video: 是否录制视频
    - local: 是否本地运行（影响训练时长）
    """
    
    # 创建带时间戳的输出文件夹
    filename = os.path.join(output_folder, 'save-'+datetime.now().strftime("%m.%d.%Y_%H.%M.%S"))
    if not os.path.exists(filename):
        os.makedirs(filename+'/')

    # =================== 环境创建部分 ===================
    if not multiagent:
        # 单智能体环境：使用 HoverAviary
        # make_vec_env 创建向量化环境以提高训练效率
        train_env = make_vec_env(HoverAviary,
                                 env_kwargs=dict(obs=DEFAULT_OBS, act=DEFAULT_ACT),
                                 n_envs=8,  # 并行环境数量，可调整为 2, 4, 8
                                 seed=0
                                 )
        # 创建评估环境（非向量化）
        eval_env = HoverAviary(obs=DEFAULT_OBS, act=DEFAULT_ACT)
    else:
        # 多智能体环境：使用 MultiHoverAviary
        train_env = make_vec_env(MultiHoverAviary,
                                 env_kwargs=dict(num_drones=DEFAULT_AGENTS, obs=DEFAULT_OBS, act=DEFAULT_ACT),
                                 n_envs=8,
                                 seed=0
                                 )
        eval_env = MultiHoverAviary(num_drones=DEFAULT_AGENTS, obs=DEFAULT_OBS, act=DEFAULT_ACT)

    # 打印环境信息 - 用于调试和确认环境配置
    print('[INFO] Action space:', train_env.action_space)      # 动作空间维度
    print('[INFO] Observation space:', train_env.observation_space)  # 观测空间维度

    # =================== 模型创建和训练部分 ===================
    # 创建 PPO 模型
    # PPO (Proximal Policy Optimization) 是一种流行的策略梯度算法
    model = PPO('MlpPolicy',        # 使用多层感知机策略网络
                train_env,          # 训练环境
                # tensorboard_log=filename+'/tb/', # 可选：TensorBoard 日志记录
                verbose=1)          # 详细输出级别

    # =================== 训练目标和回调设置 ===================
    # 根据动作类型和智能体数量设置目标奖励阈值
    if DEFAULT_ACT == ActionType.ONE_D_RPM:
        target_reward = 474.15 if not multiagent else 949.5
    else:
        target_reward = 467. if not multiagent else 920.
    
    # 创建奖励阈值停止回调 - 当达到目标奖励时停止训练
    callback_on_best = StopTrainingOnRewardThreshold(reward_threshold=target_reward,
                                                     verbose=1)
    
    # 创建评估回调 - 定期评估模型性能
    eval_callback = EvalCallback(eval_env,
                                 callback_on_new_best=callback_on_best,  # 新最佳模型时调用
                                 verbose=1,
                                 best_model_save_path=filename+'/',      # 最佳模型保存路径
                                 log_path=filename+'/',                  # 日志保存路径
                                 eval_freq=int(1000),                    # 评估频率（步数）
                                 deterministic=True,                     # 确定性评估
                                 render=False)                           # 评估时不渲染
    
    # =================== 开始训练 ===================
    model.learn(total_timesteps=int(1e7) if local else int(1e2), # 本地运行1千万步，CI测试100步
                callback=eval_callback,                           # 传入评估回调
                log_interval=100)                                # 日志打印间隔

    # 保存最终模型
    model.save(filename+'/final_model.zip')
    print(filename)

    # =================== 打印训练进度 ===================
    # 读取并打印评估结果
    with np.load(filename+'/evaluations.npz') as data:
        for j in range(data['timesteps'].shape[0]):
            print(str(data['timesteps'][j])+","+str(data['results'][j][0]))

    ############################################################
    #################### 训练完成，开始测试 ####################
    ############################################################

    if local:
        input("Press Enter to continue...")  # 本地运行时暂停，等待用户确认

    # =================== 加载最佳模型 ===================
    # 优先加载最佳模型，否则加载最终模型
    if os.path.isfile(filename+'/best_model.zip'):
        path = filename+'/best_model.zip'
    else:
        print("[ERROR]: no model under the specified path", filename)
    model = PPO.load(path)

    # =================== 创建测试环境 ===================
    if not multiagent:
        # 单智能体测试环境
        test_env = HoverAviary(gui=gui,                    # 是否显示GUI
                               obs=DEFAULT_OBS,            # 观测类型
                               act=DEFAULT_ACT,            # 动作类型
                               record=record_video)        # 是否录制视频
        test_env_nogui = HoverAviary(obs=DEFAULT_OBS, act=DEFAULT_ACT)  # 无GUI版本用于评估
    else:
        # 多智能体测试环境
        test_env = MultiHoverAviary(gui=gui,
                                    num_drones=DEFAULT_AGENTS,
                                    obs=DEFAULT_OBS,
                                    act=DEFAULT_ACT,
                                    record=record_video)
        test_env_nogui = MultiHoverAviary(num_drones=DEFAULT_AGENTS, obs=DEFAULT_OBS, act=DEFAULT_ACT)
    
    # =================== 创建日志记录器 ===================
    logger = Logger(logging_freq_hz=int(test_env.CTRL_FREQ),           # 日志频率
                    num_drones=DEFAULT_AGENTS if multiagent else 1,    # 无人机数量
                    output_folder=output_folder,                       # 输出文件夹
                    colab=colab                                        # Colab模式
                    )

    # =================== 策略评估 ===================
    # 使用无GUI环境进行10次评估，获取平均奖励
    mean_reward, std_reward = evaluate_policy(model,
                                              test_env_nogui,
                                              n_eval_episodes=10
                                              )
    print("\n\n\nMean reward ", mean_reward, " +- ", std_reward, "\n\n")

    # =================== 实际测试运行 ===================
    obs, info = test_env.reset(seed=42, options={})  # 重置环境，设置随机种子确保可重复性
    start = time.time()  # 记录开始时间，用于同步
    
    # 运行一个完整的episode
    for i in range((test_env.EPISODE_LEN_SEC+2)*test_env.CTRL_FREQ):
        # 使用训练好的模型预测动作
        action, _states = model.predict(obs,
                                        deterministic=True  # 确定性推理
                                        )
        # 执行动作并获取新状态
        obs, reward, terminated, truncated, info = test_env.step(action)
        obs2 = obs.squeeze()   # 压缩维度
        act2 = action.squeeze()
        
        # 打印详细信息用于调试
        print("Obs:", obs, "\tAction", action, "\tReward:", reward, "\tTerminated:", terminated, "\tTruncated:", truncated)
        
        # 如果是运动学观测，记录到日志中
        if DEFAULT_OBS == ObservationType.KIN:
            if not multiagent:
                # 单智能体日志记录
                logger.log(drone=0,
                           timestamp=i/test_env.CTRL_FREQ,
                           state=np.hstack([obs2[0:3], np.zeros(7), obs2[3:10], act2, np.zeros(6)]),
                           control=np.zeros(12))
            else:
                # 打印观测和动作的形状，帮助调试
                print(f"多智能体模式 - obs形状: {obs.shape}, act形状: {action.shape}")
                
                # 多智能体日志记录 - 改进版本
                for j in range(DEFAULT_AGENTS):
                    # 获取每个无人机的状态 - 注意检查实际形状
                    if len(obs.shape) > 1 and obs.shape[0] == DEFAULT_AGENTS:
                        # 如果obs是按无人机分组的(每个无人机一行)
                        drone_obs = obs[j]
                        drone_act = action[j] if len(action.shape) > 1 else action[j*4:(j+1)*4]
                        
                        # 提取位置、速度、姿态等
                        drone_pos = drone_obs[0:3]
                        drone_vel = drone_obs[3:6] if len(drone_obs) > 5 else np.zeros(3)
                        drone_rpy = drone_obs[6:9] if len(drone_obs) > 8 else np.zeros(3)
                        drone_ang_vel = drone_obs[9:12] if len(drone_obs) > 11 else np.zeros(3)
                    else:
                        # 如果obs是扁平的向量，需要手动分割
                        obs_per_drone = len(obs) // DEFAULT_AGENTS
                        start_idx = j * obs_per_drone
                        
                        drone_obs = obs[start_idx:start_idx+obs_per_drone]
                        drone_act = action[j*4:(j+1)*4] if len(action.shape) == 1 else action[j]
                        
                        # 假设前12个是位置、速度、姿态、角速度(每个3个)
                        if obs_per_drone >= 12:
                            drone_pos = drone_obs[0:3]
                            drone_vel = drone_obs[3:6]
                            drone_rpy = drone_obs[6:9]
                            drone_ang_vel = drone_obs[9:12]
                        else:
                            # 如果观测空间较小，填充一些零
                            drone_pos = drone_obs[0:min(3, len(drone_obs))]
                            if len(drone_pos) < 3:
                                drone_pos = np.pad(drone_pos, (0, 3-len(drone_pos)))
                            
                            drone_vel = np.zeros(3)
                            drone_rpy = np.zeros(3)
                            drone_ang_vel = np.zeros(3)
                    
                    # 打印每个无人机的数据，帮助调试
                    print(f"Drone {j} - pos: {drone_pos}, vel: {drone_vel}, act: {drone_act}")
                    
                    # 确保动作数组形状正确
                    if not isinstance(drone_act, np.ndarray):
                        drone_act = np.array([drone_act])
                    if len(drone_act.shape) == 0:  # 如果是标量
                        drone_act = np.array([drone_act])
                    
                    # 构建完整的20维状态向量(logger.log期望的格式)
                    drone_state = np.zeros(20)
                    drone_state[0:3] = drone_pos            # 位置 x,y,z
                    drone_state[10:13] = drone_vel          # 速度 vx,vy,vz
                    drone_state[7:10] = drone_rpy           # 姿态 roll,pitch,yaw
                    drone_state[13:17] = drone_act if len(drone_act) <= 4 else drone_act[:4]  # 动作
                    drone_state[17:20] = drone_ang_vel      # 角速度
                    
                    # 记录到日志
                    logger.log(drone=j,
                              timestamp=i/test_env.CTRL_FREQ,
                              state=drone_state,
                              control=np.zeros(12))
        
        test_env.render()  # 渲染环境（如果启用了GUI）
        print(terminated)
        
        # 同步帧率 - 确保实时运行
        sync(i, start, test_env.CTRL_TIMESTEP)
        
        if terminated:
            # episode结束时的处理 - 需要具体实现
            break
    
    test_env.close()  # 关闭测试环境

    # =================== 结果可视化 ===================
    # 如果启用绘图且使用运动学观测，则绘制结果
    if plot and DEFAULT_OBS == ObservationType.KIN:
        logger.plot()

# =================== 脚本入口点 ===================
if __name__ == '__main__':
    # 定义并解析命令行参数
    # 创建命令行参数解析器，描述脚本用途
    parser = argparse.ArgumentParser(description='Single agent reinforcement learning example script')
    
    # 添加 multiagent 参数：是否使用多智能体环境（LeaderFollower），默认为 False
    parser.add_argument('--multiagent',         
                        default=DEFAULT_MA,            
                        type=str2bool,      
                        help='Whether to use example LeaderFollower instead of Hover (default: False)', 
                        metavar='')
    
    # 添加 gui 参数：是否显示 PyBullet GUI，默认为 True
    parser.add_argument('--gui',                
                        default=DEFAULT_GUI,           
                        type=str2bool,      
                        help='Whether to use PyBullet GUI (default: True)', 
                        metavar='')
    
    # 添加 record_video 参数：是否录制视频，默认为 False
    parser.add_argument('--record_video',       
                        default=DEFAULT_RECORD_VIDEO,  
                        type=str2bool,      
                        help='Whether to record a video (default: False)', 
                        metavar='')
    
    # 添加 output_folder 参数：日志和模型保存的文件夹，默认为 "results"
    parser.add_argument('--output_folder',      
                        default=DEFAULT_OUTPUT_FOLDER, 
                        type=str,           
                        help='Folder where to save logs (default: "results")', 
                        metavar='')
    
    # 添加 colab 参数：是否在 Colab （指谷歌的云计算环境）环境运行，默认为 False
    parser.add_argument('--colab',              
                        default=DEFAULT_COLAB,         
                        type=bool,          
                        help='Whether example is being run by a notebook (default: "False")', 
                        metavar='')
    
    # 解析命令行参数，返回 Namespace 对象
    ARGS = parser.parse_args()

    # 启动训练和测试流程
    # vars(ARGS) 将 Namespace 对象转换为字典，作为关键字参数传递
    run(**vars(ARGS))
