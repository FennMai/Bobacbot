# 机器人位置随机化功能说明

本功能允许您控制机器人在仿真环境中的初始位置是固定的还是随机的。

## 功能选项

### randomize_robot_position (布尔值)
- **False** (默认): 机器人使用XML文件中定义的固定位置
- **True**: 机器人位置在指定范围内随机化

## 随机化范围

当启用位置随机化时，机器人的位置将在以下范围内随机选择：

- **X轴**: -2.0 到 2.0 米
- **Y轴**: -2.0 到 2.0 米  
- **朝向角度**: -π 到 π 弧度 (-180° 到 180°)

## 使用方法

### 1. 在 MujocoEnv 中使用

```python
from mujoco_env import MujocoEnv

# 使用固定位置 (默认)
env = MujocoEnv(randomize_robot_position=False)

# 使用随机位置
env = MujocoEnv(randomize_robot_position=True)

# 结合其他选项
env = MujocoEnv(
    randomize_robot_position=True,
    show_images=True,
    show_viewer=True
)
```

### 2. 在 main.py 中使用命令行参数

```bash
# 使用固定位置 (默认)
python main.py --sim --teleop

# 使用随机位置
python main.py --sim --teleop --randomize-robot-position

# 保存演示数据时使用随机位置
python main.py --sim --teleop --save --randomize-robot-position
```

### 3. 命令行参数说明

- `--randomize-robot-position`: 启用机器人位置随机化
- 如果不使用此参数，机器人将使用XML文件中的固定位置

## 默认固定位置

当 `randomize_robot_position=False` 时，机器人使用以下固定位置：
- X: -0.5 米
- Y: 0.0 米
- 朝向: 0.0 弧度 (朝向正X轴方向)

这些值来自 `bobacbot_demo_scene.xml` 中 `base_link` 的定义。

## 应用场景

### 固定位置模式适用于:
- 调试和开发
- 确定性的测试
- 基准测试
- 需要可重复结果的实验

### 随机位置模式适用于:
- 数据增强
- 提高策略的泛化能力
- 模拟真实世界的不确定性
- 训练对位置变化鲁棒的策略

## 注意事项

1. 随机位置是在每次调用 `env.reset()` 时生成的
2. 随机范围可以在 `MujocoSim.__init__()` 中的 `robot_position_range` 字典中修改
3. 机器人的Z坐标(高度)不会随机化，由XML文件中的模型定义决定
4. 立方体等其他物体的位置不受此选项影响，它们有自己的随机化逻辑

## 自定义随机化范围

如果需要修改随机化范围，可以编辑 `mujoco_env.py` 中 `MujocoSim` 类的 `robot_position_range` 字典：

```python
self.robot_position_range = {
    'x': [-1.0, 1.0],    # 修改X轴范围
    'y': [-1.5, 1.5],    # 修改Y轴范围  
    'yaw': [-math.pi/2, math.pi/2]  # 修改朝向范围
}
```
