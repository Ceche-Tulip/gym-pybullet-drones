"""
LLM圆形轨迹规划器

基于硅基流动API，使用LLM生成Python代码来计算单无人机的圆形轨迹
支持顺时针/逆时针方向，为后续障碍物避障功能奠定基础

作者: GitHub Copilot
日期: 2025年10月7日
"""

import numpy as np
from openai import OpenAI


def generate_circle_trajectory(init_xyz, num_waypoints=1000, clockwise=False, radius=None):
    """
    使用 LLM 生成 Python 代码，计算单无人机圆形轨迹
    
    参数:
        init_xyz: 无人机当前3D位置 [x, y, z] 或 [[x, y, z]]
        num_waypoints: 轨迹点数量，默认1000个点
        clockwise: 是否顺时针，False为逆时针（默认）
        radius: 指定半径，None则根据当前位置计算
    
    返回:
        numpy.ndarray: 形状为 (1, num_waypoints, 3) 的轨迹数组
        
    注意:
        - 圆心默认为(0, 0)
        - Z值保持无人机当前高度不变
        - 轨迹不包含当前点，从当前角度的下一个点开始
        - 保持原有格式以便后续扩展为多无人机
    """
    
    # 参数处理和验证
    direction_text = "顺时针" if clockwise else "逆时针"
    print(f"[信息]: 请求生成 {num_waypoints} 个轨迹点的{direction_text}圆形路径")
    print(f"[信息]: 基于当前无人机高度生成轨迹")

    # 硅基流动 API 客户端初始化
    client = OpenAI(
        api_key="sk-lnfxstizcghkokwbsekwlixpohvuujhrvhfgezcyidyjqpqr",  # 🔑 在这里填写您的硅基流动 API Key
        base_url="https://api.siliconflow.cn/v1"  # 🌐 在这里填写硅基流动的 base_url (例如: "https://api.siliconflow.cn/v1")
    )

    # 输入格式标准化：确保是列表格式
    if isinstance(init_xyz, np.ndarray):
        if init_xyz.ndim == 1:  # 单无人机 [x, y, z] -> [[x, y, z]]
            init_xyz = [init_xyz.tolist()]
        else:
            init_xyz = init_xyz.tolist()
    elif isinstance(init_xyz, list) and len(init_xyz) == 3 and isinstance(init_xyz[0], (int, float)):
        # 单无人机格式转换
        init_xyz = [init_xyz]
    
    print(f"[调试]: 标准化后的初始位置: {init_xyz}")

    # 构建给LLM的提示词
    prompt = f"""
你是一个无人机仿真领域的Python专家。

请编写Python代码，定义一个名为 `trajectories` 的变量，它是一个形状为 (1, {num_waypoints}, 3) 的NumPy数组，包含单个无人机的3D轨迹点 (X, Y, Z)。

输入数据:
- 无人机当前3D位置列表: {init_xyz}

具体要求:
1. 圆形轨迹设置:
   - 圆心固定在XY平面的 (0, 0) 位置
   - 半径计算: 使用无人机当前位置到圆心的距离作为半径 R = sqrt(x²+y²)
   - Z轴高度: 保持与无人机当前Z位置完全相同，整个轨迹的Z值都等于输入位置的Z值

2. 轨迹点计算:
   - 计算当前角度: angle = np.arctan2(y, x) 
   - 生成 {num_waypoints} 个均匀分布的轨迹点
   - 运动方向: {'顺时针 (角度递减)' if clockwise else '逆时针 (角度递增)'}
   - 重要: 轨迹不包含当前点，从当前角度的下一个位置开始
   - Z轴处理: 所有轨迹点的Z坐标都设为无人机当前的Z位置值
   - 关键: 确保所有数组(x_coords, y_coords, z_coords)长度都等于{num_waypoints}

3. 输出格式要求:
   - 使用numpy数组进行所有计算
   - 最终结果 `trajectories` 的形状必须严格等于 (1, {num_waypoints}, 3)
   - 不要使用数组切片[1:]或其他可能改变数组长度的操作
   - 第一维度为1，表示单个无人机

⚠️ 严格的输出要求:
- 仅输出可执行的Python代码
- 只定义 `trajectories` 变量
- 不要包含: import语句、markdown格式、注释、print语句
"""

    try:
        print("[状态]: 正在向LLM发送请求...")
        
        # 调用LLM API生成代码
        response = client.chat.completions.create(
            model="deepseek-ai/DeepSeek-V3.1-Terminus",  # 🤖 推荐使用代码专用模型，您可以改为其他模型
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,  # 低温度确保代码生成的稳定性
            max_tokens=2048
        )

        # 提取生成的代码
        generated_code = response.choices[0].message.content.strip()
        print("[LLM生成代码]:\n", generated_code)

        # 清理代码格式（移除可能的markdown标记）
        if generated_code.startswith("```"):
            generated_code = generated_code.strip("`").strip()
            if generated_code.startswith("python"):
                generated_code = generated_code[len("python"):].strip()

        # 安全执行生成的代码
        print("[状态]: 正在执行LLM生成的代码...")
        local_variables = {}
        exec(generated_code, {"np": np}, local_variables)

        # 验证生成的轨迹变量
        if "trajectories" not in local_variables:
            raise ValueError("LLM生成的代码中没有定义 'trajectories' 变量")

        trajectories = np.array(local_variables["trajectories"])

        # 验证轨迹格式
        expected_shape = (1, num_waypoints, 3)
        if trajectories.shape != expected_shape:
            raise ValueError(f"轨迹形状错误: 期望 {expected_shape}, 实际 {trajectories.shape}")

        print(f"[成功]: 轨迹生成完成，形状: {trajectories.shape}")
        print(f"[验证]: 起始点 {trajectories[0, 0]}, 结束点 {trajectories[0, -1]}")
        
        return trajectories

    except Exception as e:
        print(f"[错误]: LLM轨迹生成失败: {str(e)}")
        print("[提示]: 请检查API密钥、网络连接和模型可用性")
        return None


def test_circle_trajectory():
    """
    测试函数：验证圆形轨迹生成功能
    """
    print("=" * 60)
    print("🔍 开始测试LLM圆形轨迹生成功能")
    print("=" * 60)
    
    # 测试参数
    test_position = [1.0, 0.0, 1.2]  # 当前位置：距离圆心1米，高度1.5米
    test_waypoints = 100  # 测试用较少点数
    
    print(f"测试参数:")
    print(f"  - 当前位置: {test_position}")
    print(f"  - 轨迹点数: {test_waypoints}")
    print(f"  - 方向: 逆时针")
    print(f"  - 预期高度: 保持在 {test_position[2]} 米")
    
    # 生成轨迹
    trajectory = generate_circle_trajectory(
        init_xyz=test_position,
        num_waypoints=test_waypoints,
        clockwise=False  # 测试逆时针
    )
    
    if trajectory is not None:
        print("\n✅ 测试成功!")
        print(f"生成轨迹形状: {trajectory.shape}")
        
        # 简单验证圆形轨迹
        points_2d = trajectory[0, :, :2]  # 提取XY坐标
        distances = np.sqrt(np.sum(points_2d**2, axis=1))  # 计算到原点距离
        print(f"半径范围: {distances.min():.3f} - {distances.max():.3f}")
        print(f"高度范围: {trajectory[0, :, 2].min():.3f} - {trajectory[0, :, 2].max():.3f}")
        
        # 验证高度是否保持在当前位置
        expected_height = test_position[2]
        actual_heights = trajectory[0, :, 2]
        height_consistent = np.allclose(actual_heights, expected_height)
        print(f"高度一致性检查: {'✅ 通过' if height_consistent else '❌ 失败'}")
        
        return True
    else:
        print("\n❌ 测试失败!")
        return False


if __name__ == "__main__":
    # 运行测试
    test_circle_trajectory()