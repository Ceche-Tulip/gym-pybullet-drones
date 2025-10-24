"""
障碍物布局可视化脚本

用于展示环境中障碍物的2D俯视图布局
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Circle, Rectangle
import numpy as np

# 设置中文字体 - 尝试多个可能的字体
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei', 'Noto Sans CJK SC', 'SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

# 如果上面的字体都不可用，尝试使用系统字体
try:
    from matplotlib.font_manager import FontProperties
    import os
    # 尝试使用系统中文字体
    chinese_fonts = [
        '/usr/share/fonts/truetype/wqy/wqy-microhei.ttc',
        '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',
        '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
    ]
    for font_path in chinese_fonts:
        if os.path.exists(font_path):
            print(f"✅ 使用字体: {font_path}")
            break
except Exception as e:
    print(f"⚠️ 字体加载警告: {e}")
    print("将使用默认字体，可能无法正确显示中文")

# 环境范围
X_RANGE = [-1.5, 1.5]
Y_RANGE = [-1.5, 1.5]

# 障碍物数据 [x, y, z, 类型, 颜色, 尺寸描述]
obstacles = [
    {'x': 0.8, 'y': 0.8, 'z': 0.4, 'type': 'box', 'color': 'red', 'size': 0.15, 'name': '红色柱'},
    {'x': -0.9, 'y': -0.6, 'z': 0.4, 'type': 'cylinder', 'color': 'blue', 'size': 0.12, 'name': '蓝色柱'},
    {'x': 0.6, 'y': -0.8, 'z': 0.25, 'type': 'box', 'color': 'green', 'size': 0.2, 'name': '绿色块'},
    {'x': -0.7, 'y': 0.5, 'z': 0.6, 'type': 'cylinder', 'color': 'yellow', 'size': 0.08, 'name': '黄色柱'},
    {'x': -1.0, 'y': 0.0, 'z': 0.68, 'type': 'sphere', 'color': 'purple', 'size': 0.18, 'name': '紫色球'},
]

# 推荐测试点
test_points = [
    [0.5, 0.5, 1.0],
    [-0.5, 0.3, 1.0],
    [0.8, -0.5, 1.0],
    [-0.8, -0.3, 1.0],
    [-1.0, 0.2, 1.0],
]

# 创建图形
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

# ============ 左图：2D俯视图 ============
ax1.set_xlim(X_RANGE[0] - 0.2, X_RANGE[1] + 0.2)
ax1.set_ylim(Y_RANGE[0] - 0.2, Y_RANGE[1] + 0.2)
ax1.set_xlabel('X 轴 (米)', fontsize=12)
ax1.set_ylabel('Y 轴 (米)', fontsize=12)
ax1.set_title('障碍物布局 - 俯视图 (2D)', fontsize=14, fontweight='bold')
ax1.grid(True, alpha=0.3, linestyle='--')
ax1.set_aspect('equal')

# 绘制环境边界
boundary = Rectangle((X_RANGE[0], Y_RANGE[0]), 
                     X_RANGE[1] - X_RANGE[0], 
                     Y_RANGE[1] - Y_RANGE[0],
                     linewidth=2, edgecolor='black', facecolor='lightgray', alpha=0.2)
ax1.add_patch(boundary)

# 绘制起始点
ax1.plot(0, 0, 'g*', markersize=20, label='起始点 (0,0)', zorder=10)
ax1.text(0, -0.15, '起点', ha='center', fontsize=10, color='green', fontweight='bold')

# 绘制障碍物
for obs in obstacles:
    if obs['type'] == 'box':
        # 方形障碍物
        rect = Rectangle((obs['x'] - obs['size'], obs['y'] - obs['size']),
                        obs['size'] * 2, obs['size'] * 2,
                        facecolor=obs['color'], alpha=0.6, edgecolor='black', linewidth=2)
        ax1.add_patch(rect)
    elif obs['type'] == 'cylinder' or obs['type'] == 'sphere':
        # 圆形障碍物
        circle = Circle((obs['x'], obs['y']), obs['size'],
                       facecolor=obs['color'], alpha=0.6, edgecolor='black', linewidth=2)
        ax1.add_patch(circle)
    
    # 标注障碍物
    ax1.text(obs['x'], obs['y'], obs['name'], 
            ha='center', va='center', fontsize=9, fontweight='bold', color='white')
    ax1.plot(obs['x'], obs['y'], 'k+', markersize=10)

# 绘制推荐测试点
for i, point in enumerate(test_points, 1):
    ax1.plot(point[0], point[1], 'rx', markersize=12, markeredgewidth=2)
    ax1.text(point[0] + 0.1, point[1] + 0.1, f'T{i}', fontsize=9, color='red')

ax1.legend(loc='upper left', fontsize=10)

# ============ 右图：3D侧视图（高度信息）============
ax2.set_xlim(-1.7, 1.7)
ax2.set_ylim(0, 2.5)
ax2.set_xlabel('X/Y 位置 (米)', fontsize=12)
ax2.set_ylabel('Z 高度 (米)', fontsize=12)
ax2.set_title('障碍物高度 - 侧视图', fontsize=14, fontweight='bold')
ax2.grid(True, alpha=0.3, linestyle='--')

# 绘制地面
ax2.axhline(y=0, color='brown', linewidth=3, label='地面')
ax2.fill_between([-1.7, 1.7], 0, -0.1, color='brown', alpha=0.3)

# 绘制高度参考线
ax2.axhline(y=1.0, color='gray', linewidth=1, linestyle='--', alpha=0.5)
ax2.text(1.55, 1.0, '1.0m', fontsize=9, color='gray')
ax2.axhline(y=2.0, color='gray', linewidth=1, linestyle='--', alpha=0.5)
ax2.text(1.55, 2.0, '2.0m', fontsize=9, color='gray')

# 绘制障碍物高度条
bar_width = 0.15
x_positions = np.linspace(-1.2, 1.2, len(obstacles))

for i, obs in enumerate(obstacles):
    height = obs['z'] * 2  # z是中心高度，总高度约为2倍
    ax2.bar(x_positions[i], height, width=bar_width, 
           color=obs['color'], alpha=0.7, edgecolor='black', linewidth=2)
    ax2.text(x_positions[i], height + 0.1, f"{height:.1f}m", 
            ha='center', fontsize=9, fontweight='bold')
    ax2.text(x_positions[i], -0.15, obs['name'], 
            ha='center', fontsize=8, rotation=15)

# 绘制典型飞行高度范围
flight_zone = Rectangle((-1.7, 0.5), 3.4, 1.5, 
                        linewidth=2, edgecolor='green', 
                        facecolor='lightgreen', alpha=0.15, linestyle='--')
ax2.add_patch(flight_zone)
ax2.text(0, 1.25, '典型飞行区域\n(0.5-2.0m)', 
        ha='center', va='center', fontsize=10, color='darkgreen', 
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

ax2.legend(loc='upper left', fontsize=10)

# 添加总体信息文本
info_text = f"""
环境信息:
• 空间范围: X[{X_RANGE[0]}, {X_RANGE[1]}] m
• 空间范围: Y[{Y_RANGE[0]}, {Y_RANGE[1]}] m  
• 高度范围: Z[0.05, 2.5] m
• 障碍物数量: {len(obstacles)} 个
• 起始位置: (0, 0, 0.1) m
"""

fig.text(0.5, 0.02, info_text, ha='center', fontsize=10, 
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
         family='monospace')

plt.suptitle('🚁 无人机连续导航环境 - 障碍物布局', 
            fontsize=16, fontweight='bold', y=0.98)

plt.tight_layout(rect=(0, 0.08, 1, 0.96))  # 修复类型提示：使用tuple而不是list

# 保存图片
output_path = '/home/peking/projects/RL/gym-pybullet-drones/obstacle_layout.png'
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"✅ 障碍物布局图已保存到: {output_path}")

plt.show()
