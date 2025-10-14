# 遥操作数据采集文档

## 1. 概述

本项目通过遥操作（手机或键盘）采集机器人操作演示数据，用于后续的模仿学习训练。数据采集系统采用 **Episode-based** 的设计，每个 episode 包含一个完整的任务轨迹，包括观测序列（机器人状态 + 相机图像）和动作序列。

### 1.1 数据采集流程

```
[启动程序] → [环境重置] → [开始 Episode] → [遥操作控制]
    → [结束 Episode] → [保存/丢弃] → [重置环境] → (循环)
```

### 1.2 支持的控制模式

| 模式 | 输入设备 | 适用场景 |
|------|---------|---------|
| **Teleop** | 手机 WebXR | 6DOF 位姿控制，数据采集 |
| **Keyboard** | 键盘 + 手机 | 推理评估，无需手机遥操作 |
| **Remote Policy** | 策略服务器 + 手机 | 策略推理，手机作为使能设备 |

## 2. Episode 数据结构

### 2.1 内存中的数据

在采集过程中，Episode 数据存储在内存中：

```python
class EpisodeWriter:
    def __init__(self, output_dir):
        # Episode 数据
        self.timestamps = []      # 时间戳列表
        self.observations = []    # 观测序列
        self.actions = []         # 动作序列
```

**数据格式**:
```python
# 单步数据
timestamp = 1733970123.456  # float, Unix 时间戳 (秒)

observation = {
    'base_pose': np.array([x, y, theta]),      # 底盘位姿
    'arm_pos': np.array([x, y, z]),            # 机械臂位置
    'arm_quat': np.array([x, y, z, w]),        # 机械臂四元数
    'gripper_pos': np.array([pos]),            # 夹爪位置
    'base_image': np.array((360, 640, 3)),     # 底盘相机图像
    'wrist_image': np.array((480, 640, 3)),    # 手腕相机图像
}

action = {
    'base_pose': np.array([x, y, theta]),
    'arm_pos': np.array([x, y, z]),
    'arm_quat': np.array([x, y, z, w]),
    'gripper_pos': np.array([pos]),
}
```

### 2.2 磁盘存储格式

Episode 保存到磁盘后，结构如下：

```
data/demos/
├── 20241214T153045123456/          # Episode 文件夹 (时间戳命名)
│   ├── data.pkl                    # 状态和动作数据 (pickle)
│   ├── base_image.mp4              # 底盘相机视频
│   └── wrist_image.mp4             # 手腕相机视频
├── 20241214T153112789012/
│   ├── data.pkl
│   ├── base_image.mp4
│   └── wrist_image.mp4
└── ...
```

**data.pkl 内容**:
```python
{
    'timestamps': [t0, t1, t2, ...],    # 时间戳列表
    'observations': [obs0, obs1, ...],  # 观测列表 (图像字段为 None)
    'actions': [act0, act1, ...]        # 动作列表
}
```

**视频压缩**:
- 编码: H.264 (avc1)
- 帧率: 10 fps
- 分辨率: 原始分辨率 (640×360 或 640×480)
- 压缩比: ~10x (相比原始图像序列)

## 3. EpisodeWriter 实现

### 3.1 初始化

```python
class EpisodeWriter:
    def __init__(self, output_dir):
        self.output_dir = Path(output_dir)
        # Episode 目录名基于时间戳 (精确到微秒)
        self.episode_dir = self.output_dir / datetime.now().strftime('%Y%m%dT%H%M%S%f')
        assert not self.episode_dir.exists()  # 确保不会覆盖已有数据

        # Episode 数据
        self.timestamps = []
        self.observations = []
        self.actions = []

        # 后台写入线程
        self.flush_thread = None
```

**时间戳命名格式**:
- `%Y%m%dT%H%M%S%f`: 例如 `20241214T153045123456`
- 微秒精度避免命名冲突（即使短时间内创建多个 episode）

### 3.2 记录单步数据

```python
def step(self, obs, action):
    self.timestamps.append(time.time())
    self.observations.append(obs)
    self.actions.append(action)
```

**注意**:
- `obs` 和 `action` 是引用，但由于后续会拷贝到磁盘，不会被修改
- 图像数据会占用大量内存（每步 ~1.5 MB），长 episode 可能导致 OOM

### 3.3 保存到磁盘

```python
def _flush(self):
    assert len(self) > 0

    # 1. 创建 Episode 目录
    self.episode_dir.mkdir(parents=True)

    # 2. 提取图像观测
    frames_dict = {}
    for obs in self.observations:
        for k, v in obs.items():
            if v.ndim == 3:  # 图像是 3D 数组
                if k not in frames_dict:
                    frames_dict[k] = []
                frames_dict[k].append(v)
                obs[k] = None  # 清空图像数据，减少 pickle 大小

    # 3. 将图像序列写入 MP4 视频
    for k, frames in frames_dict.items():
        mp4_path = self.episode_dir / f'{k}.mp4'
        write_frames_to_mp4(frames, mp4_path)

    # 4. 保存其余数据为 pickle
    with open(self.episode_dir / 'data.pkl', 'wb') as f:
        pickle.dump({
            'timestamps': self.timestamps,
            'observations': self.observations,  # 图像字段为 None
            'actions': self.actions
        }, f)

    # 5. 打印统计信息
    num_episodes = len([child for child in self.output_dir.iterdir() if child.is_dir()])
    print(f'Saved episode to {self.episode_dir} ({num_episodes} total)')
```

**为什么将图像保存为 MP4？**
1. **压缩**: H.264 压缩比 ~10x，节省磁盘空间
2. **加载速度**: OpenCV 可以快速解码视频
3. **兼容性**: MP4 可以用任何视频播放器查看

### 3.4 异步保存

```python
def flush_async(self):
    print('Saving successful episode to disk...')
    # 在后台线程中保存，避免阻塞主循环
    self.flush_thread = threading.Thread(target=self._flush, daemon=True)
    self.flush_thread.start()

def wait_for_flush(self):
    if self.flush_thread is not None:
        self.flush_thread.join()  # 等待写入完成
        self.flush_thread = None
```

**为什么异步保存？**
- 磁盘写入可能耗时 1-3 秒（取决于 episode 长度）
- 异步保存让用户可以立即开始下一个 episode
- 主循环等待下一个 reset 前会调用 `wait_for_flush()` 确保写入完成

## 4. EpisodeReader 实现

### 4.1 加载 Episode

```python
class EpisodeReader:
    def __init__(self, episode_dir):
        self.episode_dir = episode_dir

        # 1. 加载 pickle 数据
        with open(episode_dir / 'data.pkl', 'rb') as f:
            data = pickle.load(f)
        self.timestamps = data['timestamps']
        self.observations = data['observations']
        self.actions = data['actions']

        assert len(self.timestamps) > 0
        assert len(self.timestamps) == len(self.observations) == len(self.actions)

        # 2. 从 MP4 恢复图像观测
        frames_dict = {}
        for step_idx, obs in enumerate(self.observations):
            for k, v in obs.items():
                if v is None:  # 图像字段
                    # 延迟加载: 只在第一次访问时读取 MP4
                    if k not in frames_dict:
                        mp4_path = episode_dir / f'{k}.mp4'
                        frames_dict[k] = read_frames_from_mp4(mp4_path)

                    # 恢复当前步的图像
                    obs[k] = frames_dict[k][step_idx]  # np.uint8
```

**延迟加载优化**:
- MP4 只在第一次需要时解码
- 所有图像一次性加载到内存（如果需要多次访问）
- 适用于回放和数据转换

### 4.2 MP4 编解码

#### 编码

```python
def write_frames_to_mp4(frames, mp4_path):
    height, width, _ = frames[0].shape
    fourcc = cv.VideoWriter_fourcc(*'avc1')  # H.264 编码
    out = cv.VideoWriter(str(mp4_path), fourcc, POLICY_CONTROL_FREQ, (width, height))

    for frame in frames:
        bgr_frame = cv.cvtColor(frame, cv.COLOR_RGB2BGR)  # OpenCV 使用 BGR
        out.write(bgr_frame)

    out.release()
```

**关键参数**:
- `fourcc`: 'avc1' (H.264)
- `fps`: `POLICY_CONTROL_FREQ = 10` (10 fps)
- 颜色空间转换: RGB → BGR

#### 解码

```python
def read_frames_from_mp4(mp4_path):
    cap = cv.VideoCapture(str(mp4_path))
    frames = []

    while True:
        ret, bgr_frame = cap.read()
        if not ret:
            break
        frames.append(cv.cvtColor(bgr_frame, cv.COLOR_BGR2RGB))

    cap.release()
    return frames
```

## 5. 数据采集主循环

### 5.1 主程序入口

```python
def main(args):
    # 1. 创建环境
    if args.sim:
        from mujoco_env import MujocoEnv
        env = MujocoEnv(
            show_images=True if args.teleop else False,
            randomize_robot_position=args.random
        )

    # 2. 创建策略
    if args.teleop:
        policy = TeleopPolicy()  # 手机遥操作
    elif args.keyboard:
        policy = KeyboardRemotePolicy()  # 键盘控制
    else:
        policy = RemotePolicy()  # 远程策略服务器

    # 3. 主循环
    try:
        while True:
            writer = EpisodeWriter(args.output_dir) if args.save else None
            run_episode(env, policy, writer)
    finally:
        env.close()
```

**命令行参数**:
```bash
python main.py \
    --sim \                   # 使用仿真环境
    --teleop \                # 手机遥操作模式
    --save \                  # 保存数据
    --random \                # 随机化机器人位置
    --output-dir data/demos   # 输出目录
```

### 5.2 Episode 执行循环

```python
def run_episode(env, policy, writer=None):
    # 1. 重置环境
    print('Resetting env...')
    env.reset()
    print('Env has been reset')

    # 2. 等待用户按 "Start episode"
    print('Press "Start episode" in the web app when ready to start new episode')
    policy.reset()
    print('Finished reset, Starting new episode')

    # 3. 执行 Episode
    episode_ended = False
    start_time = time.time()

    for step_idx in count():  # itertools.count(): 无限计数器
        # 3.1 精确控制频率 (10 Hz)
        step_end_time = start_time + step_idx * POLICY_CONTROL_PERIOD
        while time.time() < step_end_time:
            time.sleep(0.0001)

        # 3.2 获取观测
        obs = env.get_obs()

        # 3.3 获取动作
        action = policy.step(obs)

        # 3.4 处理不同类型的动作
        if action is None:
            # 遥操作未启用，跳过
            continue

        elif isinstance(action, dict):
            # 执行动作
            env.step(action)

            # 记录数据
            if writer is not None and not episode_ended:
                writer.step(obs, action)

        elif not episode_ended and action == 'end_episode':
            # Episode 结束
            episode_ended = True
            print('Episode ended')

            if writer is not None and should_save_episode(writer):
                writer.flush_async()

            print('Teleop is now active. Press "Reset env" when ready to proceed.')

        elif action == 'reset_env':
            # 准备重置环境
            break

    # 4. 等待保存完成
    if writer is not None:
        writer.wait_for_flush()
```

**关键点**:
1. **精确频率控制**: 使用 `step_end_time` 计算下一步的目标时间
2. **数据记录时机**: 只在 `action is dict` 时记录（排除 None 和控制信号）
3. **异步保存**: `flush_async()` 后主循环继续，等待用户按 "Reset env"

### 5.3 保存确认

```python
def should_save_episode(writer):
    if len(writer) == 0:
        print('Discarding empty episode')
        return False

    # 提示用户是否保存
    while True:
        user_input = input('Save episode (y/n)? ').strip().lower()
        if user_input == 'y':
            return True
        if user_input == 'n':
            print('Discarding episode')
            return False
        print('Invalid response')
```

**设计考虑**:
- 空 episode 自动丢弃
- 用户可以选择丢弃失败的 episode（如机器人掉落物体）

## 6. 数据采集实践

### 6.1 启动数据采集

```bash
# 基本模式
python main.py --sim --teleop --save --output-dir data/demos

# 随机化机器人位置 (提高泛化能力)
python main.py --sim --teleop --save --random --output-dir data/demos

# 键盘控制模式 (无需手机)
python main.py --sim --keyboard --save --output-dir data/demos
```

### 6.2 操作流程

1. **程序启动**
   - 等待环境初始化
   - 手机连接到服务器 (Teleop 模式)

2. **开始 Episode**
   - 手机点击 "Start episode" 进入 AR 模式
   - 或按空格键 (Keyboard 模式)

3. **执行任务**
   - 手机遥操作控制机器人
   - 完成任务目标 (如抓取物体、放置物体)

4. **结束 Episode**
   - 手机点击 "End episode"
   - 或再次按空格键 (Keyboard 模式)

5. **保存决策**
   - 终端提示: `Save episode (y/n)?`
   - 输入 `y` 保存，`n` 丢弃

6. **重置环境**
   - 手机点击 "Reset env"
   - 或再次按空格键 (Keyboard 模式)

7. **循环采集**
   - 重复步骤 2-6，采集多个 episode

### 6.3 采集建议

#### Episode 长度

- **最短**: 10 步 (~1 秒)
- **推荐**: 50-200 步 (5-20 秒)
- **最长**: 无限制，但内存占用线性增长

#### 采集数量

根据任务复杂度：
- **简单任务** (如推物体): 20-50 episodes
- **中等任务** (如抓取放置): 50-100 episodes
- **复杂任务** (如整理房间): 100-200+ episodes

#### 数据多样性

1. **位置多样性**: 使用 `--random` 随机化机器人初始位置
2. **物体多样性**: 手动改变物体位置/朝向
3. **轨迹多样性**: 采用不同的抓取策略

### 6.4 数据质量检查

采集后使用回放工具验证：

```bash
# 回放所有 episode
python replay_episodes.py --sim --input-dir data/demos

# 显示相机图像
python replay_episodes.py --sim --input-dir data/demos --show-images

# 执行记录的动作 (测试数据准确性)
python replay_episodes.py --sim --input-dir data/demos --execute-obs
```

**检查要点**:
1. 机器人运动是否流畅
2. 图像是否清晰、无花屏
3. Episode 长度是否合理
4. 任务是否成功完成

## 7. 数据统计

### 7.1 Episode 统计

```python
from pathlib import Path
from episode_storage import EpisodeReader

data_dir = Path('data/demos')
episode_dirs = sorted([d for d in data_dir.iterdir() if d.is_dir()])

print(f"Total episodes: {len(episode_dirs)}")

# 统计每个 episode 的长度
lengths = []
for episode_dir in episode_dirs:
    reader = EpisodeReader(episode_dir)
    lengths.append(len(reader))

print(f"Average length: {sum(lengths) / len(lengths):.1f} steps")
print(f"Min length: {min(lengths)} steps")
print(f"Max length: {max(lengths)} steps")
```

输出示例：
```
Total episodes: 85
Average length: 127.3 steps
Min length: 45 steps
Max length: 312 steps
```

### 7.2 磁盘占用

```bash
# 查看总大小
du -sh data/demos

# 查看单个 episode 大小
du -h data/demos/20241214T153045123456
```

典型数据：
- **data.pkl**: ~100 KB (100 steps)
- **base_image.mp4**: ~2 MB
- **wrist_image.mp4**: ~3 MB
- **总计**: ~5 MB per episode

**85 episodes 总计**: ~425 MB

## 8. 常见问题

### 8.1 Episode 过长导致 OOM

**症状**: 采集长 episode 时程序崩溃

**原因**: 图像数据占用大量内存 (~1.5 MB per step)

**解决**:
1. 控制 episode 长度 (<200 steps)
2. 增加系统内存
3. 修改 `EpisodeWriter` 实现流式保存

### 8.2 保存失败

**症状**: Episode 结束后无法保存

**排查**:
1. 检查磁盘空间
   ```bash
   df -h
   ```
2. 检查目录权限
   ```bash
   ls -ld data/demos
   ```
3. 检查日志输出
   ```python
   # 在 _flush() 中添加日志
   print(f"Saving to {self.episode_dir}")
   ```

### 8.3 时间戳命名冲突

**症状**: `FileExistsError: [Errno 17] File exists`

**原因**: 极短时间内创建多个 episode（微秒级）

**解决**: 已使用微秒精度时间戳，几乎不可能冲突。如果仍然发生，检查系统时钟是否正常。

### 8.4 MP4 编码失败

**症状**: 视频保存后无法播放

**排查**:
1. 检查 OpenCV 是否支持 H.264
   ```python
   import cv2 as cv
   print(cv.getBuildInformation())  # 查找 "FFMPEG: YES"
   ```
2. 尝试其他编码器
   ```python
   fourcc = cv.VideoWriter_fourcc(*'mp4v')  # MPEG-4
   ```

### 8.5 图像颜色异常

**症状**: 保存的视频颜色不对（红蓝互换）

**原因**: 忘记 RGB ↔ BGR 转换

**检查**:
```python
# 写入时
bgr_frame = cv.cvtColor(frame, cv.COLOR_RGB2BGR)  # 必须转换

# 读取时
frames.append(cv.cvtColor(bgr_frame, cv.COLOR_BGR2RGB))  # 必须转换
```

## 9. 高级功能

### 9.1 流式保存 (避免 OOM)

修改 `EpisodeWriter` 实现边采集边保存：

```python
class StreamingEpisodeWriter:
    def __init__(self, output_dir):
        self.episode_dir = ...
        self.episode_dir.mkdir(parents=True)

        # 打开文件和视频写入器
        self.pkl_file = open(self.episode_dir / 'data.pkl', 'wb')
        self.video_writers = {...}  # 初始化 cv.VideoWriter

    def step(self, obs, action):
        # 立即写入磁盘，不缓存在内存
        for k, v in obs.items():
            if v.ndim == 3:
                self.video_writers[k].write(v)
        pickle.dump((obs, action), self.pkl_file)
```

**优点**: 内存占用恒定
**缺点**: 无法中途取消保存

### 9.2 自动质量过滤

在保存前自动检查 episode 质量：

```python
def should_save_episode(writer):
    if len(writer) < 10:
        print('Discarding too short episode')
        return False

    # 检查机械臂是否移动足够距离
    first_arm_pos = writer.observations[0]['arm_pos']
    last_arm_pos = writer.observations[-1]['arm_pos']
    distance = np.linalg.norm(last_arm_pos - first_arm_pos)

    if distance < 0.05:  # 5 cm
        print(f'Discarding stationary episode (arm moved {distance:.3f}m)')
        return False

    # 提示用户确认
    while True:
        user_input = input('Save episode (y/n)? ').strip().lower()
        ...
```

### 9.3 数据增强

在保存时进行数据增强：

```python
def augment_episode(observations, actions):
    # 示例: 镜像翻转
    for obs in observations:
        obs['base_pose'][1] *= -1  # y 坐标取反
        obs['base_image'] = np.fliplr(obs['base_image'])  # 水平翻转图像
    for action in actions:
        action['base_pose'][1] *= -1
    return observations, actions
```

## 10. 下一步

数据采集完成后，需要将数据转换为模型训练格式。下一篇文档将介绍如何将 Episode 数据转换为 robomimic HDF5 格式，并进行 Diffusion Policy 训练。
