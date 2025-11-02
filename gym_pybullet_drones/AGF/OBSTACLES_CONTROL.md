# 障碍物配置说明

## 📍 障碍物定义位置

你的项目中障碍物在以下文件中被定义和控制：

### 1. **主要代码文件** (space_expander.py)
**位置**: `gym_pybullet_drones/custom/space_expander.py`

- **第217-298行**: `_addObstacles()` 方法 - 实际创建障碍物的代码
- **第28行和第70行**: `obstacles` 参数 - 控制是否启用障碍物

**障碍物配置**:
```python
# 蓝色圆柱体 (位置已更正为实际代码中的位置)
位置: (0.6, -0.5, 0.5)
半径: 0.10m
高度: 1.0m

# 红色圆柱体
位置: (-0.6, 0.5, 0.5)
半径: 0.10m
高度: 1.0m

# 两柱间距: 1.0m
```

### 2. **AGF导航器** (agf_navigator.py)
**位置**: `gym_pybullet_drones/AGF/agf_navigator.py`

- **第113行**: 环境创建时设置 `obstacles=True/False`
- **新增**: 现在支持通过构造函数参数控制

### 3. **配置文件** (config_agf.py)
**位置**: `gym_pybullet_drones/AGF/config_agf.py`

- **第37-52行**: `OBSTACLES_CONFIG` - 障碍物配置信息（文档用途）

---

## 🔧 如何关闭障碍物测试

现在你有**3种方法**来禁用障碍物：

### 方法1: 使用命令行参数 (✅ 推荐)

```bash
# 禁用障碍物
python test_agf_navigation.py --no-obstacles

# 启用障碍物（默认）
python test_agf_navigation.py --obstacles
```

### 方法2: 使用快捷脚本

```bash
# 运行无障碍物测试
./test_no_obstacles.sh

# 或
bash test_no_obstacles.sh
```

### 方法3: 在代码中修改 (不推荐)

如果需要永久禁用，可以修改 `agf_navigator.py`:

```python
# 在 _create_environment() 方法中
self.env = ExtendedHoverAviary(
    # ... 其他参数 ...
    obstacles=False  # 改为 False
)
```

---

## 🧪 测试示例

### 测试1: 无障碍物 - 验证基础飞行
```bash
python test_agf_navigation.py --no-obstacles \
    --start -1.2 0.0 0.5 \
    --target 1.2 0.0 0.5
```
**预期**: 无人机应该沿直线从左端飞到右端

### 测试2: 有障碍物 - 验证避障能力
```bash
python test_agf_navigation.py --obstacles \
    --start -1.2 0.0 0.5 \
    --target 1.2 0.0 0.5
```
**预期**: 无人机应该绕过障碍物到达目标

### 测试3: 自定义路径（无障碍物）
```bash
python test_agf_navigation.py --no-obstacles \
    --start 0.0 0.0 0.5 \
    --target 1.0 1.0 1.5
```

---

## 📊 代码改动总结

### test_agf_navigation.py
- ✅ 添加 `--no-obstacles` 和 `--obstacles` 命令行参数
- ✅ 更新帮助文档
- ✅ 传递 `obstacles` 参数给 `AGFNavigator`
- ✅ 根据障碍物状态调整输出信息

### agf_navigator.py
- ✅ 在 `__init__()` 中添加 `obstacles` 参数
- ✅ 使用 `self.obstacles` 控制环境创建
- ✅ 根据障碍物状态调整输出信息

### 新增文件
- ✅ `test_no_obstacles.sh` - 快捷测试脚本

---

## 💡 使用建议

1. **首次测试**: 建议先用 `--no-obstacles` 验证基础飞行路径是否正确
2. **避障测试**: 确认基础飞行正常后，再用 `--obstacles` 测试避障功能
3. **调试**: 无障碍物模式可以帮助你区分是路径规划问题还是避障算法问题

---

## 🎯 快速开始

```bash
# 进入AGF目录
cd /home/peking/projects/RL/gym-pybullet-drones/gym_pybullet_drones/AGF

# 无障碍物测试（方法1）
python test_agf_navigation.py --no-obstacles

# 无障碍物测试（方法2）
./test_no_obstacles.sh

# 有障碍物测试
python test_agf_navigation.py --obstacles
```

现在你可以轻松切换障碍物状态来测试你的避障方法了！🚀
