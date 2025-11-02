# AGF (Artificial Guidance Field) 避障导航系统

## 🎯 系统简介

AGF是一个基于**人工势场(APF)**的智能避障导航系统，采用**分层控制架构**：

- **上层**：APF路径规划器，负责避障策略
- **下层**：PPO控制器，负责执行导航

### 核心特点

✅ **无需重新训练** - 完全利用现有PPO模型  
✅ **实时避障** - APF动态规划安全路径  
✅ **独立模块** - 不影响原有导航系统  
✅ **易于调试** - 清晰的分层架构  

## 📁 文件结构

```
gym_pybullet_drones/AGF/
├── __init__.py              # 模块初始化
├── apf_planner.py           # APF路径规划器
├── agf_navigator.py         # AGF导航器主控制
├── test_agf_navigation.py   # 测试脚本
├── config_agf.py            # 配置文件
└── README.md                # 本文档
```

## 🏗️ 架构设计

### 分层控制流程

```
用户输入目标位置
        ↓
┌────────────────────────────┐
│   APF路径规划器 (上层)      │
│  - 计算目标引力场          │
│  - 计算障碍物斥力场         │
│  - 生成安全中间目标点       │
└──────────┬─────────────────┘
           │ 每5步更新一次
           ↓
┌────────────────────────────┐
│   目标更新逻辑 (中层)       │
│  - 将APF输出设为环境目标   │
└──────────┬─────────────────┘
           │
           ↓
┌────────────────────────────┐
│   PPO控制器 (底层)         │
│  - 观测环境状态            │
│  - 输出电机RPM控制         │
│  - 执行点对点导航          │
└────────────────────────────┘
```

### APF势场模型

#### 引力场 (Attractive Field)
```python
F_att = k_att * (target - current_pos)
```
- 方向：指向目标
- 大小：与距离成正比

#### 斥力场 (Repulsive Field)
```python
当 d < d0 时：
F_rep = k_rep * (1/d - 1/d0) * (1/d²) * direction

当 d >= d0 时：
F_rep = 0
```
- 方向：远离障碍物
- 大小：距离越近斥力越大

## ⚙️ 配置参数

### APF参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `k_att` | 1.0 | 引力系数，控制目标吸引力强度 |
| `k_rep` | 2.0 | 斥力系数，控制障碍物排斥力强度 |
| `d0` | 0.5m | 斥力影响范围，超过此距离斥力为0 |
| `step_size` | 0.2m | 每次规划的步长 |
| `goal_threshold` | 0.1m | 认为到达目标的距离阈值 |

### 更新频率

- **APF更新频率**: 每5步更新一次中间目标
- **太快**: 目标频繁切换，可能导致抖动
- **太慢**: 避障响应不及时

## 🚀 使用方法

### 快速开始

```bash
# 激活conda环境
conda activate drones

# 进入AGF目录
cd /home/peking/projects/RL/gym-pybullet-drones/gym_pybullet_drones/AGF

# 运行测试（使用最新模型）
python test_agf_navigation.py
```

### 命令行参数

```bash
# 无GUI模式
python test_agf_navigation.py --no-gui

# 指定模型
python test_agf_navigation.py --model /path/to/model.zip

# 指定目标位置
python test_agf_navigation.py --target 0.8 0.0 1.0

# 调整APF更新频率
python test_agf_navigation.py --apf-freq 10

# 组合使用
python test_agf_navigation.py --gui --target 0.5 0.5 1.5 --apf-freq 5
```

### Python API使用

```python
from gym_pybullet_drones.AGF.agf_navigator import AGFNavigator

# 创建导航器
navigator = AGFNavigator(
    model_path='path/to/model.zip',
    gui=True,
    record=False,
    apf_update_freq=5
)

# 初始化
navigator.initialize()

# 导航到目标
result = navigator.navigate_to_target([0.8, 0.0, 1.0])

# 查看结果
if result['success']:
    print("导航成功！")
    print(f"用时: {result['stats']['steps']}步")
    print(f"生成路径点: {result['stats']['waypoints_generated']}")

# 关闭
navigator.close()
```

## 🧪 测试场景

### 场景1: 直线穿越障碍物

```bash
python test_agf_navigation.py --target 1.0 0.0 1.0
```

- 起点: (0, 0, 0.1)
- 终点: (1.0, 0.0, 1.0)
- 障碍: 两个圆柱在 (0, ±0.6, 0.5)
- 预期: 无人机从两柱中间穿过

### 场景2: 绕过单侧障碍物

```bash
python test_agf_navigation.py --target 0.5 -0.8 1.0
```

- 起点: (0, 0, 0.1)
- 终点: (0.5, -0.8, 1.0)
- 预期: 无人机绕过左侧圆柱

### 场景3: S形避障

```bash
python test_agf_navigation.py --target 0.8 0.8 1.0
```

- 起点: (0, 0, 0.1)  
- 终点: (0.8, 0.8, 1.0)
- 预期: S形轨迹避开两个障碍物

### 场景4：点对点避障

```bash
python test_agf_navigation.py --start -1.2 -1.2 0.5 --target 0.7 0.7 0.5
```

## 📊 输出信息

### 导航过程输出

```
[步数  000] 位置: [0.0, 0.0, 0.1]
              中间目标: [0.15, 0.0, 0.3]
              到最终目标距离: 1.414m

[步数  050] 位置: [0.3, 0.0, 0.5]
              中间目标: [0.45, 0.0, 0.7]
              到最终目标距离: 1.131m

✅ 到达目标！
```

### 统计信息

```
📊 导航统计
=============================================================
总步数: 250
生成路径点: 50
用时: 12.5秒
是否到达: ✅ 是

APF统计:
  平均引力: 0.856
  平均斥力: 0.324
  碰撞警告: 2
=============================================================
```

## 🔧 调试与优化

### 调整APF参数

如果遇到以下问题，可以调整参数：

**问题1: 无人机离障碍物太近**
```python
# 增加斥力系数或影响范围
k_rep = 3.0  # 原来2.0
d0 = 0.7     # 原来0.5
```

**问题2: 无人机绕路太远**
```python
# 增加引力系数或减小斥力
k_att = 1.5  # 原来1.0
k_rep = 1.5  # 原来2.0
```

**问题3: 路径抖动**
```python
# 增加更新频率或步长
apf_update_freq = 10  # 原来5
step_size = 0.3       # 原来0.2
```

### 修改参数位置

在 `agf_navigator.py` 的 `_create_apf_planner()` 方法中：

```python
def _create_apf_planner(self):
    """创建APF规划器"""
    self.apf_planner = APFPlanner(
        k_att=1.0,      # 修改这里
        k_rep=2.0,      # 修改这里
        d0=0.5,         # 修改这里
        step_size=0.2,  # 修改这里
        goal_threshold=0.1
    )
```

## ⚠️ 注意事项

1. **模型限制**
   - 当前PPO模型未在障碍物环境中训练
   - APF负责避障规划，但PPO的执行可能不完美
   - 建议先在GUI模式下观察效果

2. **障碍物信息**
   - 当前障碍物位置硬编码在 `_get_obstacle_info()` 方法中
   - 如需修改障碍物，需同步修改 `space_expander.py` 和 `agf_navigator.py`

3. **目标范围**
   - 目标必须在有效空间内: X[-1.5, 1.5], Y[-1.5, 1.5], Z[0.05, 2.5]
   - 超出范围会被拒绝

## 🐛 故障排除

### 问题：无法找到模型

```
❌ 文件未找到: 在 'results' 中未找到模型文件
```

**解决方案**:
```bash
# 指定模型路径
python test_agf_navigation.py --model path/to/your/model.zip
```

### 问题：无人机碰撞障碍物

**可能原因**:
1. APF更新频率太低
2. 斥力系数太小
3. PPO执行误差

**解决方案**:
```bash
# 增加更新频率和斥力
python test_agf_navigation.py --apf-freq 3
```

然后在代码中调整 `k_rep = 3.0`

### 问题：无人机不移动

**可能原因**:
1. 目标位置无效
2. 引力和斥力抵消

**解决方案**:
- 检查目标是否在有效范围内
- 尝试不同的目标位置

## 📈 未来改进

- [ ] 动态障碍物支持
- [ ] 3D可视化轨迹
- [ ] 自适应参数调整
- [ ] 多目标点连续导航
- [ ] 与LLM集成的自然语言控制

## 📝 参考文献

- Khatib, O. (1986). "Real-time obstacle avoidance for manipulators and mobile robots"
- 人工势场法 (Artificial Potential Field Method)

---

**开发时间**: 2025年10月24日  
**版本**: 1.0.0  
**状态**: ✅ 可用
