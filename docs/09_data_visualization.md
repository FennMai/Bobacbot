# 数据可视化文档

## 1. 概述

数据可视化是验证采集质量和调试策略的重要工具。本项目提供了多种可视化工具，包括 Episode 回放、图像查看、轨迹分析等。

### 1.1 可视化工具

| 工具 | 用途 | 输入 |
|------|------|------|
| **replay_episodes.py** | Episode 回放 | Episode 目录 |
| **OpenCV** | 实时图像显示 | 相机数据流 |
| **Matplotlib** | 轨迹绘制 | 状态序列 |
| **HDF5 Viewer** | 数据检查 | HDF5 文件 |

## 2. Episode 回放

### 2.1 基本回放

`replay_episodes.py` 可以在仿真中回放采集的 episode。

```bash
# 回放所有 episodes
python replay_episodes.py --sim --input-dir data/demos

# 回放指定 episode
python replay_episodes.py --sim --input-dir data/demos/20241214T153045123456
```

**回放效果**:
- 机器人会重复执行采集时的动作
- 可以观察任务是否成功完成
- 验证数据记录的准确性

### 2.2 显示图像观测

```bash
# 同时显示相机图像
python replay_episodes.py \
    --sim \
    --input-dir data/demos \
    --show-images
```

**显示内容**:
- `base_image`: 底盘相机视角
- `wrist_image`: 手腕相机视角
- 图像窗口并排显示

### 2.3 执行观测动作

```bash
# 执行观测中的动作而非记录的动作 (测试数据准确性)
python replay_episodes.py \
    --sim \
    --input-dir data/demos \
    --execute-obs
```

**用途**:
- 验证观测数据的准确性
- 检查是否存在状态记录错误
- 理论上应该与正常回放一致

### 2.4 实现原理

```python
def replay_episode(env, episode_dir, show_images=False, execute_obs=False):
    # 1. 重置环境
    env.reset()

    # 2. 加载 Episode 数据
    reader = EpisodeReader(episode_dir)
    print(f'Loaded episode from {episode_dir}')

    # 3. 逐步回放
    start_time = time.time()
    for step_idx, (obs, action) in enumerate(zip(reader.observations, reader.actions)):
        # 3.1 精确控制频率 (10 Hz)
        step_end_time = start_time + step_idx * POLICY_CONTROL_PERIOD
        while time.time() < step_end_time:
            time.sleep(0.0001)

        # 3.2 显示图像 (可选)
        if show_images:
            window_idx = 0
            for k, v in obs.items():
                if v.ndim == 3:
                    cv.imshow(k, cv.cvtColor(v, cv.COLOR_RGB2BGR))
                    cv.moveWindow(k, 640 * window_idx, 0)
                    window_idx += 1
            cv.waitKey(1)

        # 3.3 执行动作
        if execute_obs:
            env.step(obs)  # 执行观测中的动作
        else:
            env.step(action)  # 执行记录的动作
```

## 3. 图像可视化

### 3.1 OpenCV 窗口显示

#### 显示单帧图像

```python
import cv2 as cv
from episode_storage import EpisodeReader

# 加载 episode
reader = EpisodeReader('data/demos/20241214T153045123456')

# 显示第一帧
obs = reader.observations[0]
cv.imshow('base_image', cv.cvtColor(obs['base_image'], cv.COLOR_RGB2BGR))
cv.imshow('wrist_image', cv.cvtColor(obs['wrist_image'], cv.COLOR_RGB2BGR))
cv.waitKey(0)  # 等待按键
```

#### 播放图像序列

```python
for obs in reader.observations:
    cv.imshow('base_image', cv.cvtColor(obs['base_image'], cv.COLOR_RGB2BGR))
    cv.waitKey(100)  # 100ms per frame (10 fps)
```

### 3.2 Matplotlib 静态显示

```python
import matplotlib.pyplot as plt
from episode_storage import EpisodeReader

reader = EpisodeReader('data/demos/20241214T153045123456')

# 显示多帧图像网格
fig, axes = plt.subplots(2, 4, figsize=(12, 6))

for i in range(4):
    obs = reader.observations[i * 20]  # 每隔 20 帧采样
    axes[0, i].imshow(obs['base_image'])
    axes[0, i].set_title(f'Base - Step {i*20}')
    axes[0, i].axis('off')

    axes[1, i].imshow(obs['wrist_image'])
    axes[1, i].set_title(f'Wrist - Step {i*20}')
    axes[1, i].axis('off')

plt.tight_layout()
plt.savefig('episode_frames.png')
plt.show()
```

### 3.3 并排对比

```python
# 对比两个 episode 的同一时刻
reader1 = EpisodeReader('data/demos/episode1')
reader2 = EpisodeReader('data/demos/episode2')

fig, axes = plt.subplots(2, 2, figsize=(10, 10))

step_idx = 50
axes[0, 0].imshow(reader1.observations[step_idx]['base_image'])
axes[0, 0].set_title('Episode 1 - Base')
axes[0, 1].imshow(reader1.observations[step_idx]['wrist_image'])
axes[0, 1].set_title('Episode 1 - Wrist')

axes[1, 0].imshow(reader2.observations[step_idx]['base_image'])
axes[1, 0].set_title('Episode 2 - Base')
axes[1, 1].imshow(reader2.observations[step_idx]['wrist_image'])
axes[1, 1].set_title('Episode 2 - Wrist')

plt.tight_layout()
plt.show()
```

## 4. 轨迹可视化

### 4.1 底盘轨迹

```python
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from episode_storage import EpisodeReader

# 加载多个 episodes
episode_dirs = sorted(Path('data/demos').iterdir())
readers = [EpisodeReader(d) for d in episode_dirs[:10]]

# 提取底盘轨迹
fig, ax = plt.subplots(figsize=(10, 10))

for i, reader in enumerate(readers):
    # 提取 x, y 坐标
    x = [obs['base_pose'][0] for obs in reader.observations]
    y = [obs['base_pose'][1] for obs in reader.observations]

    # 绘制轨迹
    ax.plot(x, y, alpha=0.7, label=f'Episode {i+1}')
    ax.scatter(x[0], y[0], marker='o', s=100, c='g')  # 起点
    ax.scatter(x[-1], y[-1], marker='x', s=100, c='r')  # 终点

ax.set_xlabel('X (m)')
ax.set_ylabel('Y (m)')
ax.set_title('Base Trajectories')
ax.legend()
ax.grid(True)
ax.axis('equal')
plt.savefig('base_trajectories.png')
plt.show()
```

### 4.2 机械臂末端轨迹

```python
# 3D 轨迹可视化
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

for i, reader in enumerate(readers):
    # 提取机械臂末端位置
    x = [obs['arm_pos'][0] for obs in reader.observations]
    y = [obs['arm_pos'][1] for obs in reader.observations]
    z = [obs['arm_pos'][2] for obs in reader.observations]

    ax.plot(x, y, z, alpha=0.7, label=f'Episode {i+1}')
    ax.scatter(x[0], y[0], z[0], marker='o', s=100, c='g')
    ax.scatter(x[-1], y[-1], z[-1], marker='x', s=100, c='r')

ax.set_xlabel('X (m)')
ax.set_ylabel('Y (m)')
ax.set_zlabel('Z (m)')
ax.set_title('Arm End-Effector Trajectories')
ax.legend()
plt.savefig('arm_trajectories.png')
plt.show()
```

### 4.3 时序数据绘制

```python
# 绘制关节角度、速度等时序数据
fig, axes = plt.subplots(3, 1, figsize=(12, 10))

reader = EpisodeReader('data/demos/20241214T153045123456')

# 底盘位姿
t = np.arange(len(reader.observations)) * 0.1  # 时间轴 (10 Hz)
base_x = [obs['base_pose'][0] for obs in reader.observations]
base_y = [obs['base_pose'][1] for obs in reader.observations]
base_theta = [obs['base_pose'][2] for obs in reader.observations]

axes[0].plot(t, base_x, label='X')
axes[0].plot(t, base_y, label='Y')
axes[0].set_ylabel('Position (m)')
axes[0].legend()
axes[0].grid(True)
axes[0].set_title('Base Position')

axes[1].plot(t, base_theta, label='Theta')
axes[1].set_ylabel('Angle (rad)')
axes[1].legend()
axes[1].grid(True)
axes[1].set_title('Base Orientation')

# 夹爪位置
gripper_pos = [obs['gripper_pos'][0] for obs in reader.observations]
axes[2].plot(t, gripper_pos, label='Gripper')
axes[2].set_ylabel('Position (0-1)')
axes[2].set_xlabel('Time (s)')
axes[2].legend()
axes[2].grid(True)
axes[2].set_title('Gripper Position')

plt.tight_layout()
plt.savefig('time_series.png')
plt.show()
```

### 4.4 动作速度分析

```python
# 计算动作变化率
fig, axes = plt.subplots(2, 1, figsize=(12, 8))

# 底盘速度
base_vel_x = np.diff([obs['base_pose'][0] for obs in reader.observations]) / 0.1
base_vel_y = np.diff([obs['base_pose'][1] for obs in reader.observations]) / 0.1
base_vel_theta = np.diff([obs['base_pose'][2] for obs in reader.observations]) / 0.1

t_vel = np.arange(len(base_vel_x)) * 0.1

axes[0].plot(t_vel, base_vel_x, label='Vx')
axes[0].plot(t_vel, base_vel_y, label='Vy')
axes[0].plot(t_vel, base_vel_theta, label='Vtheta')
axes[0].set_ylabel('Velocity')
axes[0].legend()
axes[0].grid(True)
axes[0].set_title('Base Velocity')

# 机械臂速度
arm_vel = np.linalg.norm(np.diff([obs['arm_pos'] for obs in reader.observations], axis=0), axis=1) / 0.1
axes[1].plot(t_vel, arm_vel, label='Arm')
axes[1].set_ylabel('Velocity (m/s)')
axes[1].set_xlabel('Time (s)')
axes[1].legend()
axes[1].grid(True)
axes[1].set_title('Arm End-Effector Velocity')

plt.tight_layout()
plt.savefig('velocities.png')
plt.show()
```

## 5. 数据统计分析

### 5.1 Episode 统计

```python
from pathlib import Path
from episode_storage import EpisodeReader

data_dir = Path('data/demos')
episode_dirs = sorted([d for d in data_dir.iterdir() if d.is_dir()])

# 统计信息
lengths = []
durations = []

for episode_dir in episode_dirs:
    reader = EpisodeReader(episode_dir)
    lengths.append(len(reader))
    durations.append(reader.timestamps[-1] - reader.timestamps[0])

print(f"Total episodes: {len(episode_dirs)}")
print(f"Average length: {np.mean(lengths):.1f} steps")
print(f"Min/Max length: {np.min(lengths)} / {np.max(lengths)} steps")
print(f"Average duration: {np.mean(durations):.1f} s")
print(f"Total samples: {np.sum(lengths)}")

# 绘制分布
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].hist(lengths, bins=20, edgecolor='black')
axes[0].set_xlabel('Episode Length (steps)')
axes[0].set_ylabel('Count')
axes[0].set_title('Episode Length Distribution')
axes[0].axvline(np.mean(lengths), color='r', linestyle='--', label='Mean')
axes[0].legend()

axes[1].hist(durations, bins=20, edgecolor='black')
axes[1].set_xlabel('Episode Duration (s)')
axes[1].set_ylabel('Count')
axes[1].set_title('Episode Duration Distribution')
axes[1].axvline(np.mean(durations), color='r', linestyle='--', label='Mean')
axes[1].legend()

plt.tight_layout()
plt.savefig('episode_statistics.png')
plt.show()
```

### 5.2 动作空间分析

```python
# 分析动作范围
all_base_poses = []
all_arm_pos = []
all_gripper_pos = []

for episode_dir in episode_dirs:
    reader = EpisodeReader(episode_dir)
    all_base_poses.extend([action['base_pose'] for action in reader.actions])
    all_arm_pos.extend([action['arm_pos'] for action in reader.actions])
    all_gripper_pos.extend([action['gripper_pos'] for action in reader.actions])

all_base_poses = np.array(all_base_poses)
all_arm_pos = np.array(all_arm_pos)
all_gripper_pos = np.array(all_gripper_pos)

# 统计
print("Base pose range:")
print(f"  X: [{all_base_poses[:, 0].min():.3f}, {all_base_poses[:, 0].max():.3f}]")
print(f"  Y: [{all_base_poses[:, 1].min():.3f}, {all_base_poses[:, 1].max():.3f}]")
print(f"  Theta: [{all_base_poses[:, 2].min():.3f}, {all_base_poses[:, 2].max():.3f}]")

print("\nArm position range:")
print(f"  X: [{all_arm_pos[:, 0].min():.3f}, {all_arm_pos[:, 0].max():.3f}]")
print(f"  Y: [{all_arm_pos[:, 1].min():.3f}, {all_arm_pos[:, 1].max():.3f}]")
print(f"  Z: [{all_arm_pos[:, 2].min():.3f}, {all_arm_pos[:, 2].max():.3f}]")

print(f"\nGripper range: [{all_gripper_pos.min():.3f}, {all_gripper_pos.max():.3f}]")

# 绘制分布
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

for i, label in enumerate(['X', 'Y', 'Theta']):
    axes[0, i].hist(all_base_poses[:, i], bins=50, edgecolor='black')
    axes[0, i].set_xlabel(f'Base {label}')
    axes[0, i].set_ylabel('Count')
    axes[0, i].set_title(f'Base {label} Distribution')

for i, label in enumerate(['X', 'Y', 'Z']):
    axes[1, i].hist(all_arm_pos[:, i], bins=50, edgecolor='black')
    axes[1, i].set_xlabel(f'Arm {label}')
    axes[1, i].set_ylabel('Count')
    axes[1, i].set_title(f'Arm {label} Distribution')

plt.tight_layout()
plt.savefig('action_distribution.png')
plt.show()
```

## 6. HDF5 数据检查

### 6.1 命令行工具

```bash
# 查看 HDF5 文件结构
h5ls -r data/demos.hdf5

# 输出示例:
# /data                        Group
# /data/demo_0                 Group
# /data/demo_0/actions         Dataset {100, 13}
# /data/demo_0/obs             Group
# /data/demo_0/obs/arm_pos     Dataset {100, 3}
# /data/demo_0/obs/arm_quat    Dataset {100, 4}
# /data/demo_0/obs/base_image  Dataset {100, 84, 84, 3}
# /data/demo_0/obs/base_pose   Dataset {100, 3}
# ...
```

### 6.2 Python 检查

```python
import h5py

# 打开 HDF5 文件
with h5py.File('data/demos.hdf5', 'r') as f:
    print("Episodes:", list(f['data'].keys()))
    print(f"Total episodes: {len(f['data'])}")

    # 查看第一个 episode
    demo_0 = f['data/demo_0']
    print("\nDemo 0 observations:")
    for key in demo_0['obs'].keys():
        dataset = demo_0['obs'][key]
        print(f"  {key}: shape={dataset.shape}, dtype={dataset.dtype}")

    print("\nDemo 0 actions:")
    print(f"  shape={demo_0['actions'].shape}, dtype={demo_0['actions'].dtype}")

    # 检查数据范围
    actions = demo_0['actions'][:]
    print(f"\nAction range:")
    print(f"  Min: {actions.min(axis=0)}")
    print(f"  Max: {actions.max(axis=0)}")
    print(f"  Mean: {actions.mean(axis=0)}")
    print(f"  Std: {actions.std(axis=0)}")
```

### 6.3 可视化 HDF5 数据

```python
import h5py
import matplotlib.pyplot as plt

with h5py.File('data/demos.hdf5', 'r') as f:
    # 加载第一个 episode 的图像
    base_images = f['data/demo_0/obs/base_image'][:]
    wrist_images = f['data/demo_0/obs/wrist_image'][:]

    # 显示几帧
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    for i in range(4):
        idx = i * 20
        axes[0, i].imshow(base_images[idx])
        axes[0, i].set_title(f'Base - Step {idx}')
        axes[0, i].axis('off')

        axes[1, i].imshow(wrist_images[idx])
        axes[1, i].set_title(f'Wrist - Step {idx}')
        axes[1, i].axis('off')

    plt.tight_layout()
    plt.show()
```

## 7. 实时可视化

### 7.1 在仿真中显示轨迹

修改 `mujoco_env.py` 添加轨迹可视化：

```python
class MujocoEnv:
    def __init__(self, ...):
        ...
        self.trajectory = []  # 记录轨迹

    def step(self, action):
        self.command_queue.put(action)
        # 记录底盘位置
        obs = self.get_obs()
        self.trajectory.append(obs['base_pose'][:2].copy())

    def visualize_trajectory(self):
        """在 MuJoCo viewer 中绘制轨迹"""
        trajectory = np.array(self.trajectory)
        # 使用 mujoco.mj_addGeom 绘制线条
        # (需要在 control_callback 中调用)
```

### 7.2 实时数据流可视化

```python
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# 创建实时图表
fig, ax = plt.subplots()
xdata, ydata = [], []
ln, = ax.plot([], [], 'r-')

def init():
    ax.set_xlim(0, 100)
    ax.set_ylim(-1, 1)
    return ln,

def update(frame):
    # 从环境获取最新数据
    obs = env.get_obs()
    xdata.append(frame)
    ydata.append(obs['base_pose'][0])
    ln.set_data(xdata, ydata)
    return ln,

ani = FuncAnimation(fig, update, frames=range(100), init_func=init, blit=True)
plt.show()
```

## 8. 调试可视化

### 8.1 对比预测与真实轨迹

```python
# 加载策略预测和真实执行
predictions = load_policy_predictions()
ground_truth = load_ground_truth()

fig, ax = plt.subplots(figsize=(10, 10))

# 真实轨迹
ax.plot(ground_truth[:, 0], ground_truth[:, 1], 'b-', label='Ground Truth', linewidth=2)
# 预测轨迹
ax.plot(predictions[:, 0], predictions[:, 1], 'r--', label='Prediction', linewidth=2)

ax.legend()
ax.grid(True)
ax.axis('equal')
plt.show()
```

### 8.2 观测-动作对齐检查

```python
# 检查观测和动作的时序对齐
reader = EpisodeReader('data/demos/episode1')

fig, axes = plt.subplots(2, 1, figsize=(12, 8))

t = np.arange(len(reader.observations)) * 0.1

# 观测中的位置
obs_x = [obs['base_pose'][0] for obs in reader.observations]
# 动作中的目标位置
act_x = [action['base_pose'][0] for action in reader.actions]

axes[0].plot(t, obs_x, label='Observation')
axes[0].plot(t, act_x, label='Action', alpha=0.7)
axes[0].set_ylabel('X Position (m)')
axes[0].legend()
axes[0].grid(True)
axes[0].set_title('Observation vs Action Alignment')

# 误差
error = np.array(obs_x) - np.array(act_x)
axes[1].plot(t, error, label='Error')
axes[1].set_xlabel('Time (s)')
axes[1].set_ylabel('Position Error (m)')
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()
plt.show()
```

## 9. 常用可视化脚本

### 9.1 快速查看 Episode

创建 `visualize_episode.py`:

```python
import argparse
import matplotlib.pyplot as plt
from episode_storage import EpisodeReader

def visualize(episode_dir):
    reader = EpisodeReader(episode_dir)

    # 创建子图
    fig = plt.figure(figsize=(15, 10))

    # 图像
    ax1 = plt.subplot(2, 3, 1)
    ax1.imshow(reader.observations[0]['base_image'])
    ax1.set_title('Base Image (First Frame)')
    ax1.axis('off')

    ax2 = plt.subplot(2, 3, 2)
    ax2.imshow(reader.observations[0]['wrist_image'])
    ax2.set_title('Wrist Image (First Frame)')
    ax2.axis('off')

    # 底盘轨迹
    ax3 = plt.subplot(2, 3, 3)
    x = [obs['base_pose'][0] for obs in reader.observations]
    y = [obs['base_pose'][1] for obs in reader.observations]
    ax3.plot(x, y)
    ax3.set_title('Base Trajectory')
    ax3.grid(True)
    ax3.axis('equal')

    # 时序数据
    ax4 = plt.subplot(2, 3, 4)
    t = np.arange(len(reader.observations)) * 0.1
    ax4.plot(t, [obs['base_pose'][0] for obs in reader.observations], label='X')
    ax4.plot(t, [obs['base_pose'][1] for obs in reader.observations], label='Y')
    ax4.set_title('Base Position')
    ax4.legend()
    ax4.grid(True)

    ax5 = plt.subplot(2, 3, 5)
    ax5.plot(t, [obs['arm_pos'][2] for obs in reader.observations])
    ax5.set_title('Arm Height (Z)')
    ax5.grid(True)

    ax6 = plt.subplot(2, 3, 6)
    ax6.plot(t, [obs['gripper_pos'][0] for obs in reader.observations])
    ax6.set_title('Gripper Position')
    ax6.grid(True)

    plt.tight_layout()
    plt.savefig(f'{episode_dir.name}_visualization.png')
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('episode_dir', type=Path)
    args = parser.parse_args()
    visualize(args.episode_dir)
```

使用：
```bash
python visualize_episode.py data/demos/20241214T153045123456
```

## 10. 下一步

数据可视化完成后，参考《数据格式解析》文档了解详细的数据结构和格式规范。
