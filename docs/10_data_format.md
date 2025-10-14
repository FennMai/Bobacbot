# 数据格式解析文档

## 1. 概述

本项目使用两种数据格式：**Episode 格式** 用于数据采集和存储，**HDF5 格式** 用于模型训练。本文档详细解析这两种格式的结构、转换方法和使用场景。

### 1.1 数据格式对比

| 格式 | 用途 | 优点 | 缺点 |
|------|------|------|------|
| **Episode** | 数据采集、回放 | 易读、可扩展、支持视频压缩 | 加载慢、占用空间大 |
| **HDF5** | 模型训练 | 快速读取、高效索引、标准格式 | 不直观、需转换 |

## 2. Episode 格式

### 2.1 目录结构

每个 episode 保存为独立目录，包含 pickle 文件和视频文件：

```
data/demos/20241214T153045123456/
├── data.pkl                # 状态和动作数据
├── base_image.mp4          # 底盘相机视频 (H.264 编码)
└── wrist_image.mp4         # 手腕相机视频 (H.264 编码)
```

**命名规则**:
- 目录名: `YYYYMMDDTHHMMSSffffff` (时间戳精确到微秒)
- 示例: `20241214T153045123456` = 2024年12月14日 15:30:45.123456

### 2.2 data.pkl 结构

`data.pkl` 是 Python pickle 格式，包含一个字典：

```python
{
    'timestamps': [t0, t1, t2, ...],      # 时间戳列表 (float)
    'observations': [obs0, obs1, ...],     # 观测列表 (dict)
    'actions': [act0, act1, ...]           # 动作列表 (dict)
}
```

#### timestamps

Unix 时间戳（秒），浮点数：

```python
timestamps = [
    1702555845.123456,  # 2024-12-14 15:30:45.123456
    1702555845.223456,  # 100ms 后
    1702555845.323456,  # 200ms 后
    ...
]
```

#### observations

观测字典列表，每个观测包含：

```python
obs = {
    'base_pose': np.array([x, y, theta], dtype=float64),  # 底盘位姿
    'arm_pos': np.array([x, y, z], dtype=float64),        # 机械臂位置
    'arm_quat': np.array([x, y, z, w], dtype=float64),    # 机械臂四元数
    'gripper_pos': np.array([pos], dtype=float64),        # 夹爪位置
    'base_image': None,   # 图像存储在 MP4 中，这里为 None
    'wrist_image': None,
}
```

**注意**: 图像字段在磁盘上为 `None`，加载时会从 MP4 恢复。

#### actions

动作字典列表，每个动作包含：

```python
action = {
    'base_pose': np.array([x, y, theta], dtype=float64),
    'arm_pos': np.array([x, y, z], dtype=float64),
    'arm_quat': np.array([x, y, z, w], dtype=float64),
    'gripper_pos': np.array([pos], dtype=float64),
}
```

### 2.3 视频格式

| 字段 | 值 |
|------|-----|
| **编码** | H.264 (avc1) |
| **容器** | MP4 |
| **帧率** | 10 fps |
| **分辨率** | 640×360 (base), 640×480 (wrist) |
| **颜色空间** | RGB (存储为 BGR) |
| **压缩比** | ~10x |

### 2.4 数据加载

使用 `EpisodeReader` 加载数据：

```python
from episode_storage import EpisodeReader

# 加载 episode
reader = EpisodeReader('data/demos/20241214T153045123456')

# 访问数据
print(f"Episode length: {len(reader)}")
print(f"Duration: {reader.timestamps[-1] - reader.timestamps[0]:.2f} s")

# 访问单帧
obs = reader.observations[0]
action = reader.actions[0]

print(f"Base pose: {obs['base_pose']}")
print(f"Base image shape: {obs['base_image'].shape}")  # (360, 640, 3)
```

**加载过程**:
1. 读取 `data.pkl`
2. 从 MP4 解码所有图像帧
3. 将图像恢复到观测字典中

## 3. HDF5 格式

### 3.1 文件结构

HDF5 是分层数据格式，结构类似文件系统：

```
data/demos.hdf5
└── data/                          # 根组
    ├── demo_0/                    # Episode 0
    │   ├── obs/                   # 观测组
    │   │   ├── base_pose          # Dataset (T, 3)
    │   │   ├── arm_pos            # Dataset (T, 3)
    │   │   ├── arm_quat           # Dataset (T, 4)
    │   │   ├── gripper_pos        # Dataset (T, 1)
    │   │   ├── base_image         # Dataset (T, 84, 84, 3)
    │   │   └── wrist_image        # Dataset (T, 84, 84, 3)
    │   └── actions                # Dataset (T, 13)
    ├── demo_1/
    │   └── ...
    └── demo_N/
        └── ...
```

**层级说明**:
- `data/`: 根组，包含所有 episodes
- `demo_{i}/`: 第 i 个 episode
- `obs/`: 观测数据
- `actions`: 动作数据

### 3.2 Dataset 详解

#### 观测数据

| 字段 | 形状 | 类型 | 说明 |
|------|------|------|------|
| `base_pose` | (T, 3) | float32 | 底盘位姿 [x, y, theta] |
| `arm_pos` | (T, 3) | float32 | 机械臂位置 [x, y, z] |
| `arm_quat` | (T, 4) | float32 | 机械臂四元数 [x, y, z, w] |
| `gripper_pos` | (T, 1) | float32 | 夹爪位置 [0-1] |
| `base_image` | (T, 84, 84, 3) | uint8 | 底盘相机图像 (缩放后) |
| `wrist_image` | (T, 84, 84, 3) | uint8 | 手腕相机图像 (缩放后) |

#### 动作数据

| 字段 | 形状 | 类型 | 说明 |
|------|------|------|------|
| `actions` | (T, 13) | float32 | 动作向量 |

**动作向量组成**:
```
[0:3]   base_pose      底盘位姿
[3:6]   arm_pos        机械臂位置
[6:12]  arm_rotation   机械臂旋转 (6D representation)
[12:13] gripper_pos    夹爪位置
```

**注意**: 旋转表示从四元数转换为 6D rotation representation，训练时更稳定。

### 3.3 数据加载

使用 `h5py` 加载：

```python
import h5py

# 打开文件
with h5py.File('data/demos.hdf5', 'r') as f:
    # 列出所有 episodes
    episodes = list(f['data'].keys())
    print(f"Total episodes: {len(episodes)}")

    # 加载第一个 episode
    demo_0 = f['data/demo_0']

    # 加载观测
    base_pose = demo_0['obs/base_pose'][:]  # (T, 3)
    base_image = demo_0['obs/base_image'][:]  # (T, 84, 84, 3)

    # 加载动作
    actions = demo_0['actions'][:]  # (T, 13)

    print(f"Episode 0 length: {len(base_pose)}")
    print(f"Image shape: {base_image.shape}")
    print(f"Action shape: {actions.shape}")
```

**切片加载**:
```python
# 只加载部分数据
start, end = 10, 20
base_image_slice = demo_0['obs/base_image'][start:end]  # (10, 84, 84, 3)
```

## 4. 数据转换

### 4.1 Episode → HDF5

使用 `convert_to_robomimic_hdf5.py`：

```bash
python convert_to_robomimic_hdf5.py \
    --input-dir data/demos \
    --output-path data/demos.hdf5
```

**转换步骤**:

1. **遍历 Episode 目录**

```python
episode_dirs = sorted([child for child in Path(input_dir).iterdir() if child.is_dir()])
```

2. **加载每个 Episode**

```python
reader = EpisodeReader(episode_dir)
```

3. **提取并缩放图像**

```python
for obs in reader.observations:
    for k, v in obs.items():
        if v.ndim == 3:
            v = cv.resize(v, (POLICY_IMAGE_WIDTH, POLICY_IMAGE_HEIGHT))  # 640×360 → 84×84
```

4. **转换旋转表示**

```python
from scipy.spatial.transform import Rotation

actions = [
    np.concatenate((
        action['base_pose'],           # [3]
        action['arm_pos'],             # [3]
        Rotation.from_quat(action['arm_quat']).as_rotvec(),  # [4] → [3] 轴角
        action['gripper_pos'],         # [1]
    )) for action in reader.actions
]
```

**注意**: 这里转换为轴角 (axis-angle)，训练时会进一步转换为 6D rotation。

5. **写入 HDF5**

```python
episode_group = data_group.create_group(f'demo_{episode_idx}')
for k, v in observations.items():
    episode_group.create_dataset(f'obs/{k}', data=np.array(v))
episode_group.create_dataset('actions', data=np.array(actions))
```

### 4.2 HDF5 → Episode (不常用)

如果需要从 HDF5 恢复 Episode 格式：

```python
import h5py
import pickle
from pathlib import Path
from scipy.spatial.transform import Rotation
import cv2 as cv

def hdf5_to_episode(hdf5_path, output_dir, demo_key):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with h5py.File(hdf5_path, 'r') as f:
        demo = f[f'data/{demo_key}']

        # 提取观测
        observations = []
        for i in range(len(demo['actions'])):
            obs = {
                'base_pose': demo['obs/base_pose'][i],
                'arm_pos': demo['obs/arm_pos'][i],
                'arm_quat': demo['obs/arm_quat'][i],
                'gripper_pos': demo['obs/gripper_pos'][i],
                'base_image': demo['obs/base_image'][i],
                'wrist_image': demo['obs/wrist_image'][i],
            }
            observations.append(obs)

        # 提取动作 (需要从 13D 恢复)
        actions = []
        for act in demo['actions']:
            action = {
                'base_pose': act[:3],
                'arm_pos': act[3:6],
                'arm_quat': Rotation.from_rotvec(act[6:9]).as_quat(),  # 轴角 → 四元数
                'gripper_pos': act[12:13],
            }
            actions.append(action)

        # 生成时间戳 (假设 10 Hz)
        timestamps = [i * 0.1 for i in range(len(observations))]

        # 保存 pickle
        with open(output_dir / 'data.pkl', 'wb') as f:
            pickle.dump({'timestamps': timestamps, 'observations': observations, 'actions': actions}, f)

        # 保存视频
        for image_key in ['base_image', 'wrist_image']:
            frames = [obs[image_key] for obs in observations]
            write_frames_to_mp4(frames, output_dir / f'{image_key}.mp4')
```

## 5. 坐标系统和单位

### 5.1 坐标系

所有数据使用**机器人局部坐标系**：

```
    Z (up)
    |
    |
    o----> X (forward)
   /
  /
 Y (left)
```

- **X 轴**: 机器人前方 (正前方为正)
- **Y 轴**: 机器人左侧 (左侧为正)
- **Z 轴**: 竖直向上 (向上为正)

### 5.2 单位

| 量 | 单位 | 范围 |
|---|------|------|
| **位置** | 米 (m) | 任意 |
| **角度** | 弧度 (rad) | `[-π, π]` |
| **四元数** | 无量纲 | 单位四元数 |
| **夹爪** | 归一化 | `[0, 1]` (0=闭合, 1=打开) |
| **时间** | 秒 (s) | Unix 时间戳 |

### 5.3 四元数约定

**格式**: `[x, y, z, w]` (SciPy / Robot 约定)

**转换**:
- MuJoCo 使用 `[w, x, y, z]`
- WebXR 使用 `{x, y, z, w}` (JavaScript 对象)

```python
# Robot → MuJoCo
quat_mujoco = quat_robot[[3, 0, 1, 2]]

# MuJoCo → Robot
quat_robot = quat_mujoco[[1, 2, 3, 0]]
```

### 5.4 旋转表示对比

| 表示 | 维度 | 优点 | 缺点 | 使用场景 |
|------|------|------|------|----------|
| **四元数** | 4 | 无奇异点 | 有冗余 | 仿真、观测 |
| **轴角** | 3 | 紧凑 | 有奇异点 | 数据转换 |
| **6D Rotation** | 6 | 连续、可学习 | 冗余 | 训练 |
| **旋转矩阵** | 9 | 直观 | 冗余大 | 计算 |

**转换关系**:
```python
from scipy.spatial.transform import Rotation

quat = np.array([0, 0, 0, 1])
rot = Rotation.from_quat(quat)

# 转换为不同表示
rotvec = rot.as_rotvec()        # 轴角 (3,)
matrix = rot.as_matrix()        # 旋转矩阵 (3, 3)
euler = rot.as_euler('xyz')     # 欧拉角 (3,)
```

## 6. 数据归一化

### 6.1 训练时的归一化

Diffusion Policy 训练时会自动归一化数据：

```python
# 图像归一化 (在 DiffusionPolicy 中)
image = image.astype(np.float32) / 255.0  # [0, 255] → [0, 1]

# 状态归一化 (在 Dataset 中)
normalizer = LinearNormalizer()
normalizer.fit(data)
normalized_data = normalizer.normalize(data)
```

**归一化方法**:
- **图像**: 除以 255，范围 `[0, 1]`
- **位置**: Min-Max 归一化或 Z-Score 归一化
- **角度**: 直接使用，范围 `[-π, π]`
- **夹爪**: 已归一化，范围 `[0, 1]`

### 6.2 查看归一化参数

```python
import h5py
import numpy as np

with h5py.File('data/demos.hdf5', 'r') as f:
    # 收集所有动作
    all_actions = []
    for demo_key in f['data'].keys():
        all_actions.append(f[f'data/{demo_key}/actions'][:])
    all_actions = np.concatenate(all_actions, axis=0)

    # 计算统计量
    mean = all_actions.mean(axis=0)
    std = all_actions.std(axis=0)
    min_val = all_actions.min(axis=0)
    max_val = all_actions.max(axis=0)

    print("Action statistics:")
    for i in range(13):
        print(f"  Dim {i}: mean={mean[i]:.3f}, std={std[i]:.3f}, min={min_val[i]:.3f}, max={max_val[i]:.3f}")
```

## 7. 数据验证

### 7.1 检查 Episode 完整性

```python
from pathlib import Path
from episode_storage import EpisodeReader

def validate_episode(episode_dir):
    try:
        reader = EpisodeReader(episode_dir)

        # 检查长度一致性
        assert len(reader.timestamps) == len(reader.observations) == len(reader.actions)

        # 检查数据类型
        obs = reader.observations[0]
        assert obs['base_pose'].shape == (3,)
        assert obs['arm_pos'].shape == (3,)
        assert obs['arm_quat'].shape == (4,)
        assert obs['gripper_pos'].shape == (1,)
        assert obs['base_image'].shape == (360, 640, 3)
        assert obs['wrist_image'].shape == (480, 640, 3)

        # 检查动作
        action = reader.actions[0]
        assert action['base_pose'].shape == (3,)
        assert action['arm_pos'].shape == (3,)
        assert action['arm_quat'].shape == (4,)
        assert action['gripper_pos'].shape == (1,)

        # 检查数值范围
        assert 0 <= obs['gripper_pos'][0] <= 1
        assert np.linalg.norm(obs['arm_quat']) > 0.99  # 单位四元数

        print(f"✓ {episode_dir.name} is valid")
        return True

    except Exception as e:
        print(f"✗ {episode_dir.name} is invalid: {e}")
        return False

# 验证所有 episodes
data_dir = Path('data/demos')
episode_dirs = sorted([d for d in data_dir.iterdir() if d.is_dir()])
valid_count = sum(validate_episode(d) for d in episode_dirs)
print(f"\nValid: {valid_count}/{len(episode_dirs)}")
```

### 7.2 检查 HDF5 完整性

```python
import h5py

def validate_hdf5(hdf5_path):
    with h5py.File(hdf5_path, 'r') as f:
        assert 'data' in f
        episodes = list(f['data'].keys())
        print(f"Episodes: {len(episodes)}")

        for demo_key in episodes:
            demo = f[f'data/{demo_key}']

            # 检查观测
            obs_keys = ['base_pose', 'arm_pos', 'arm_quat', 'gripper_pos', 'base_image', 'wrist_image']
            for key in obs_keys:
                assert f'obs/{key}' in demo, f"Missing {key} in {demo_key}"

            # 检查动作
            assert 'actions' in demo
            actions = demo['actions']
            assert actions.shape[1] == 13, f"Action dim should be 13, got {actions.shape[1]}"

            # 检查长度一致性
            length = len(actions)
            for key in obs_keys:
                obs_length = len(demo[f'obs/{key}'])
                assert obs_length == length, f"{key} length mismatch in {demo_key}"

        print(f"✓ HDF5 file is valid")

validate_hdf5('data/demos.hdf5')
```

## 8. 常见问题

### 8.1 Episode 加载失败

**症状**: `FileNotFoundError` 或 `EOFError`

**排查**:
```python
# 检查文件完整性
from pathlib import Path
episode_dir = Path('data/demos/20241214T153045123456')
assert (episode_dir / 'data.pkl').exists()
assert (episode_dir / 'base_image.mp4').exists()
assert (episode_dir / 'wrist_image.mp4').exists()
```

### 8.2 图像颜色异常

**症状**: 图像显示红蓝互换

**原因**: RGB/BGR 混淆

**解决**:
```python
# OpenCV 使用 BGR，需要转换
cv.imshow('image', cv.cvtColor(image, cv.COLOR_RGB2BGR))
```

### 8.3 四元数不是单位四元数

**症状**: `np.linalg.norm(quat) != 1.0`

**原因**: 数值误差或数据损坏

**解决**:
```python
# 归一化四元数
quat = quat / np.linalg.norm(quat)
```

### 8.4 HDF5 动作维度错误

**症状**: `actions.shape[1] != 13`

**原因**: 转换脚本错误

**检查**:
```python
# 确认转换逻辑
action_vector = np.concatenate((
    action['base_pose'],       # 3
    action['arm_pos'],         # 3
    rotvec,                    # 3 (应该是6D rotation)
    action['gripper_pos'],     # 1
))
# 总计应该是 3+3+6+1=13，而不是 3+3+3+1=10
```

**修正**: 检查 `convert_to_robomimic_hdf5.py` 是否正确使用 6D rotation。

## 9. 数据格式最佳实践

### 9.1 采集阶段

- ✅ 使用 Episode 格式（易于回放和调试）
- ✅ 保存原始分辨率图像
- ✅ 记录精确时间戳
- ✅ 每个 episode 独立目录

### 9.2 转换阶段

- ✅ 缩放图像到训练分辨率（84×84）
- ✅ 转换旋转表示（四元数 → 6D rotation）
- ✅ 验证转换结果
- ✅ 保留原始 Episode 数据

### 9.3 训练阶段

- ✅ 使用 HDF5 格式（快速加载）
- ✅ 启用缓存加速
- ✅ 检查数据归一化
- ✅ 监控数据加载时间

## 10. 下一步

至此，所有技术文档已完成。参考《文档索引》(docs/README.md) 查看全部文档列表。
