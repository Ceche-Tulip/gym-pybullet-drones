import os
import time
import argparse
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from gym_pybullet_drones.envs.HoverAviary import HoverAviary
from gym_pybullet_drones.envs.MultiHoverAviary import MultiHoverAviary
from gym_pybullet_drones.utils.enums import ObservationType, ActionType
from gym_pybullet_drones.utils.utils import sync
from gym_pybullet_drones.utils.Logger import Logger

DEFAULT_MODEL_PATH = "results/best_model.zip"
DEFAULT_GUI = True
DEFAULT_OBS = ObservationType('kin')
DEFAULT_ACT = ActionType('one_d_rpm')
DEFAULT_AGENTS = 2
DEFAULT_MA = False

def play(model_path=DEFAULT_MODEL_PATH, multiagent=DEFAULT_MA, gui=DEFAULT_GUI):
    #### Load saved model ####
    if not os.path.isfile(model_path):
        print(f"[ERROR] Model file not found at: {model_path}")
        return

    model = PPO.load(model_path)
    print(f"[INFO] Loaded model from {model_path}")

    #### Create test environment ####
    if not multiagent:
        env = HoverAviary(gui=gui, obs=DEFAULT_OBS, act=DEFAULT_ACT)
    else:
        env = MultiHoverAviary(gui=gui, num_drones=DEFAULT_AGENTS, obs=DEFAULT_OBS, act=DEFAULT_ACT)

    logger = Logger(logging_freq_hz=int(env.CTRL_FREQ),
                    num_drones=DEFAULT_AGENTS if multiagent else 1,
                    output_folder="logs_playback/",
                    colab=False)

    #### Run the simulation ####
    obs, _ = env.reset(seed=42, options={})
    start = time.time()

    for i in range((env.EPISODE_LEN_SEC+2)*env.CTRL_FREQ):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)

        obs2 = obs.squeeze()
        act2 = action.squeeze()

        if DEFAULT_OBS == ObservationType.KIN:
            if not multiagent:
                logger.log(drone=0,
                    timestamp=i/env.CTRL_FREQ,
                    state=np.hstack([obs2[0:3],
                                     np.zeros(4),
                                     obs2[3:15],
                                     act2]),
                    control=np.zeros(12))
            else:
                for d in range(DEFAULT_AGENTS):
                    logger.log(drone=d,
                        timestamp=i/env.CTRL_FREQ,
                        state=np.hstack([obs2[d][0:3],
                                         np.zeros(4),
                                         obs2[d][3:15],
                                         act2[d]]),
                        control=np.zeros(12))

        env.render()
        sync(i, start, env.CTRL_TIMESTEP)
        if terminated:
            break

    env.close()
    logger.plot()

if __name__ == "__main__":
    # 创建命令行参数解析器，并添加相关参数
    parser = argparse.ArgumentParser(description="在PyBullet无人机环境中运行训练好的PPO策略。")
    parser.add_argument('--model_path',
                        type=str,
                        default=DEFAULT_MODEL_PATH,
                        help='保存的策略zip文件路径 (默认: results/best_model.zip)',
                        metavar='')  
    parser.add_argument('--multiagent',
                        type=bool,
                        default=DEFAULT_MA,
                        help='是否使用MultiHoverAviary环境 (默认: False)',
                        metavar='')  
    parser.add_argument('--gui',
                        type=bool,
                        default=DEFAULT_GUI,
                        help='是否启用GUI渲染 (默认: True)',
                        metavar='')  
    # 
    # parser.add_argument('--num_drones',
    #                     type=int,
    #                     default=2,
    #                     help='MultiHoverAviary环境中的无人机数量 (默认: 2)',
    #                     metavar='')  # 保持参数格式一致

    # 解析命令行参数
    args = parser.parse_args()

    # 调用play函数并传入解析后的参数
    play(**vars(args))
