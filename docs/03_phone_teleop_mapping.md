# 手机遥操作与机械臂物理映射文档

## 1. 概述

本项目使用 WebXR API 将手机作为 3D 输入设备，通过 IMU (惯性测量单元) 获取手机的 6DOF 位姿，并映射到机器人的底盘和机械臂控制上。这种方案比传统的操纵杆更直观，能够实现 hand-to-robot 的自然映射。

### 1.1 系统架构

```
[手机 WebXR]
    ↓ (WebSocket)
[Flask-SocketIO Server]
    ↓ (Queue)
[TeleopController]
    ↓ (坐标转换 + 物理映射)
[MujocoEnv]
```

### 1.2 核心概念

- **Primary Device**: 主设备，用于控制机械臂或底盘
- **Secondary Device**: 辅助设备，用于同时控制底盘（可选）
- **Reference Pose**: 参考位姿，用于相对位移计算
- **Unwrapping**: 角度展开，避免 ±π 跳变

## 2. WebXR 坐标系

### 2.1 坐标系定义

WebXR 使用右手坐标系，但与机器人坐标系不同：

| 轴 | WebXR | 机器人 |
|---|-------|--------|
| **X** | →右 | →前 |
| **Y** | ↑上 | ←左 |
| **Z** | ↓后 | ↑上 |

### 2.2 坐标转换函数

```python
DEVICE_CAMERA_OFFSET = np.array([0.0, 0.02, -0.04])  # iPhone 14 Pro

def convert_webxr_pose(pos, quat):
    """
    将 WebXR 坐标系转换到机器人坐标系
    Args:
        pos: WebXR 位置 {'x': ..., 'y': ..., 'z': ...}
        quat: WebXR 四元数 {'x': ..., 'y': ..., 'z': ..., 'w': ...}
    Returns:
        pos: 机器人位置 [forward, left, up]
        rot: 机器人旋转 (Rotation对象)
    """
    # 坐标轴重新映射: WebXR -> Robot
    # WebXR: +x right, +y up, +z back
    # Robot: +x forward, +y left, +z up
    pos = np.array([-pos['z'], -pos['x'], pos['y']], dtype=np.float64)
    rot = R.from_quat([-quat['z'], -quat['x'], quat['y'], quat['w']])

    # 补偿相机偏移 (手机背面相机不在中心)
    pos = pos + rot.apply(DEVICE_CAMERA_OFFSET)

    return pos, rot
```

**为什么需要相机偏移补偿？**

手机的 WebXR 位姿是基于背面相机的，但我们希望旋转中心在手机中心。对于 iPhone 14 Pro：
- X 偏移: 0.0 (居中)
- Y 偏移: 0.02m (相机在手机上方 2cm)
- Z 偏移: -0.04m (相机突出 4cm)

## 3. TeleopController 设计

### 3.1 状态机

TeleopController 管理多个状态变量：

```python
class TeleopController:
    def __init__(self):
        # 设备管理
        self.primary_device_id = None    # 主设备
        self.secondary_device_id = None  # 辅助设备
        self.enabled_counts = {}         # 每个设备的启用计数

        # 机器人当前状态
        self.base_pose = None  # [x, y, theta]

        # 目标位姿 (控制输出)
        self.targets_initialized = False
        self.base_target_pose = None
        self.arm_target_pos = None
        self.arm_target_rot = None      # Rotation对象
        self.gripper_target_pos = None

        # WebXR 参考位姿
        self.base_xr_ref_pos = None
        self.base_xr_ref_rot_inv = None
        self.arm_xr_ref_pos = None
        self.arm_xr_ref_rot_inv = None

        # 机器人参考位姿
        self.base_ref_pose = None
        self.arm_ref_pos = None
        self.arm_ref_rot = None
        self.arm_ref_base_pose = None
        self.gripper_ref_pos = None
```

### 3.2 设备识别

支持同时使用两个手机：一个控制机械臂，一个控制底盘。

```python
def process_message(self, data):
    device_id = data['device_id']

    # 更新启用计数
    self.enabled_counts[device_id] = (
        self.enabled_counts.get(device_id, 0) + 1
        if 'teleop_mode' in data else 0
    )

    # 分配主次设备 (跳过前2步，因为 WebXR 姿态更新延迟更高)
    if self.enabled_counts[device_id] > 2:
        if self.primary_device_id is None and device_id != self.secondary_device_id:
            self.primary_device_id = device_id
        elif self.secondary_device_id is None and device_id != self.primary_device_id:
            self.secondary_device_id = device_id

    # 设备释放
    elif self.enabled_counts[device_id] == 0:
        if device_id == self.primary_device_id:
            self.primary_device_id = None
            self.base_xr_ref_pos = None
            self.arm_xr_ref_pos = None
        elif device_id == self.secondary_device_id:
            self.secondary_device_id = None
            self.base_xr_ref_pos = None
```

## 4. 底盘控制映射

### 4.1 映射策略

手机的平移和旋转直接映射到底盘的 x, y, yaw：

```
手机前后移动 → 底盘 x (前后)
手机左右移动 → 底盘 y (左右)
手机旋转    → 底盘 yaw (朝向)
```

### 4.2 实现

```python
if data['teleop_mode'] == 'base' or device_id == self.secondary_device_id:
    # 首次启用时记录参考位姿
    if self.base_xr_ref_pos is None:
        self.base_ref_pose = self.base_pose.copy()
        self.base_xr_ref_pos = pos[:2]  # 只用 x, y
        self.base_xr_ref_rot_inv = rot.inv()

    # 位置映射: 相对位移
    self.base_target_pose[:2] = self.base_ref_pose[:2] + (pos[:2] - self.base_xr_ref_pos)

    # 朝向映射: 前向向量的 atan2
    base_fwd_vec_rotated = (rot * self.base_xr_ref_rot_inv).apply([1.0, 0.0, 0.0])
    base_target_theta = self.base_ref_pose[2] + math.atan2(
        base_fwd_vec_rotated[1], base_fwd_vec_rotated[0]
    )

    # 角度展开 (避免 ±π 跳变)
    TWO_PI = 2 * math.pi
    self.base_target_pose[2] += (
        (base_target_theta - self.base_target_pose[2] + math.pi) % TWO_PI - math.pi
    )
```

**关键点**:
- 使用相对位移而非绝对位置，避免手机初始位置影响
- 角度展开确保连续性，如从 179° 到 -179° 实际应该是 +2°

## 5. 机械臂控制映射

### 5.1 映射策略

机械臂控制更复杂，需要考虑底盘坐标系：

```
手机位置 → 机械臂末端位置 (相对底盘)
手机朝向 → 机械臂末端朝向 (相对底盘)
捏合手势 → 夹爪开合
```

### 5.2 位置映射

```python
elif data['teleop_mode'] == 'arm':
    # 首次启用时记录所有参考位姿
    if self.arm_xr_ref_pos is None:
        self.arm_xr_ref_pos = pos
        self.arm_xr_ref_rot_inv = rot.inv()
        self.arm_ref_pos = self.arm_target_pos.copy()
        self.arm_ref_rot = self.arm_target_rot
        self.arm_ref_base_pose = self.base_pose.copy()
        self.gripper_ref_pos = self.gripper_target_pos

    # 底盘坐标系旋转矩阵
    z_rot = R.from_rotvec(np.array([0.0, 0.0, 1.0]) * self.base_pose[2])
    z_rot_inv = z_rot.inv()
    ref_z_rot = R.from_rotvec(np.array([0.0, 0.0, 1.0]) * self.arm_ref_base_pose[2])

    # 位置计算: WebXR 相对位移
    pos_diff = pos - self.arm_xr_ref_pos

    # 补偿底盘运动 (如果使用双设备控制)
    pos_diff += ref_z_rot.apply(self.arm_ref_pos) - z_rot.apply(self.arm_ref_pos)  # 旋转补偿
    pos_diff[:2] += self.arm_ref_base_pose[:2] - self.base_pose[:2]  # 平移补偿

    # 转换到底盘局部坐标系
    self.arm_target_pos = self.arm_ref_pos + z_rot_inv.apply(pos_diff)
```

**为什么需要底盘运动补偿？**

如果同时使用两个手机（一个控制底盘，一个控制机械臂），底盘运动会改变机械臂在世界坐标系下的位置。补偿后，机械臂的局部坐标保持不变。

### 5.3 姿态映射

```python
    # 姿态计算: 考虑底盘旋转
    self.arm_target_rot = (
        z_rot_inv                    # 转到底盘局部坐标系
        * (rot * self.arm_xr_ref_rot_inv)  # WebXR 相对旋转
        * ref_z_rot                  # 补偿底盘旋转
    ) * self.arm_ref_rot            # 叠加初始姿态
```

这个公式看起来复杂，但逻辑是清楚的：
1. 计算手机的相对旋转
2. 补偿底盘的旋转（双设备模式）
3. 转换到当前底盘坐标系
4. 叠加到初始姿态上

### 5.4 夹爪控制

```python
    # 夹爪: 捏合手势的增量
    self.gripper_target_pos = np.clip(
        self.gripper_ref_pos + data['gripper_delta'],
        0.0, 1.0
    )
```

`gripper_delta` 来自 WebXR 的捏合手势，范围通常在 [-0.1, 0.1]。

## 6. 输出动作生成

### 6.1 Step 函数

```python
def step(self, obs):
    # 更新机器人当前状态
    self.base_pose = obs['base_pose']

    # 首次调用时初始化
    if not self.targets_initialized:
        self.base_target_pose = obs['base_pose']
        self.arm_target_pos = obs['arm_pos']
        self.arm_target_rot = R.from_quat(obs['arm_quat'])
        self.gripper_target_pos = obs['gripper_pos']
        self.targets_initialized = True

    # 未启用遥操作时返回 None
    if self.primary_device_id is None:
        return None

    # 生成动作
    arm_quat = self.arm_target_rot.as_quat()
    if arm_quat[3] < 0.0:  # 四元数唯一性
        np.negative(arm_quat, out=arm_quat)

    action = {
        'base_pose': self.base_target_pose.copy(),
        'arm_pos': self.arm_target_pos.copy(),
        'arm_quat': arm_quat,
        'gripper_pos': self.gripper_target_pos.copy(),
    }
    return action
```

## 7. 使用方法

### 7.1 启动遥操作

```bash
# 方式1: 直接启动仿真
python main.py --sim --teleop --save --output-dir data/demos

# 方式2: 测试遥操作 (无数据采集)
python main.py --sim --teleop
```

### 7.2 手机连接

1. 确保手机和电脑在同一 WiFi 网络
2. 程序启动后会打印服务器地址，如 `Starting server at 192.168.1.100:5000`
3. 在手机浏览器打开该地址
4. 点击 "Start AR" 进入 WebXR 模式

### 7.3 操作流程

1. **开始 Episode**: 点击 "Start episode"
2. **控制模式切换**:
   - 按住屏幕 + 选择 "Base": 控制底盘
   - 按住屏幕 + 选择 "Arm": 控制机械臂
3. **夹爪控制**: 双指捏合/展开
4. **结束 Episode**: 点击 "End episode"
5. **重置环境**: 点击 "Reset env"

## 8. 调试技巧

### 8.1 可视化手机位姿

在 `policies.py` 中添加测试代码：

```python
import mujoco
import mujoco.viewer
import numpy as np
from policies import TeleopPolicy

policy = TeleopPolicy()
policy.reset()

# 创建简单场景显示手机位姿
xml = """
<mujoco>
  <worldbody>
    <light directional="true"/>
    <geom name="floor" size="0 0 .05" type="plane" rgba="0.8 0.8 0.8 1"/>
    <body name="phone" pos="0 0 .5" mocap="true">
      <geom type="box" size=".08 .04 .01" rgba=".3 .3 .8 .5"/>
    </body>
  </worldbody>
</mujoco>
"""
m = mujoco.MjModel.from_xml_string(xml)
d = mujoco.MjData(m)
mocap_id = m.body('phone').mocapid[0]

with mujoco.viewer.launch_passive(m, d) as viewer:
    while viewer.is_running():
        mujoco.mj_step(m, d)
        obs = {'base_pose': np.zeros(3), ...}
        action = policy.step(obs)
        if isinstance(action, dict):
            d.mocap_pos[mocap_id] = action['arm_pos']
            d.mocap_quat[mocap_id] = action['arm_quat'][[3, 0, 1, 2]]
        viewer.sync()
```

### 8.2 打印调试信息

取消 `policies.py` 中的注释：

```python
@self.socketio.on('message')
def handle_message(data):
    emit('echo', data['timestamp'])
    print(f"[WebServer] Received: {data}")  # 取消注释
    self.queue.put(data)
```

会打印每条消息：
```
[WebServer] Received: {'device_id': 'abc123', 'teleop_mode': 'arm', 'position': {...}, ...}
```

### 8.3 检查 RTT 延迟

服务器会回传时间戳用于 RTT 计算。正常情况下 5GHz WiFi 的 RTT 应该在 5-10ms。

## 9. 常见问题

### 9.1 手机连不上服务器

**问题**: 手机浏览器打开地址后无响应

**解决**:
1. 检查防火墙是否允许 5000 端口
2. 确认手机和电脑在同一网络
3. 某些网络禁止设备间通信，尝试使用热点

### 9.2 机械臂移动方向反了

**问题**: 手机向前推，机械臂向后移

**解决**: 检查 `convert_webxr_pose` 中的坐标映射是否正确

### 9.3 控制不连贯/卡顿

**问题**: 机械臂移动有停顿

**解决**:
1. 检查 WiFi 信号强度
2. 查看是否有大量丢包
3. 服务器性能是否足够（CPU占用）

### 9.4 双设备控制冲突

**问题**: 两个手机同时控制时机械臂抖动

**解决**: 确保主设备先启用（按住屏幕），辅助设备后启用

## 10. 性能优化

### 10.1 降低延迟

- 使用 5GHz WiFi
- 减少 WebXR pose 更新频率（已在前端限制）
- 消息过时检测（超过 250ms 丢弃）

```python
elif 1000 * time.time() - data['timestamp'] < 250:  # 250 ms
    self._process_message(data)
```

### 10.2 提高精度

- 校准手机相机偏移 `DEVICE_CAMERA_OFFSET`
- 调整参考位姿更新时机
- 过滤手机 IMU 噪声（可选）

## 11. 下一步

手机遥操作是数据采集的核心。下一篇文档将深入介绍 WebXR IMU 数据的获取和通信机制。
