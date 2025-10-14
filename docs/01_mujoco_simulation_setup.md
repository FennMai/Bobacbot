# MuJoCo 仿真环境搭建文档

## 1. 概述

本项目基于 MuJoCo 3.2.4 构建了一个完整的移动操作机器人仿真环境。主要包括 Bobac3 全向底盘、Elephant Robotics eco65b 6自由度机械臂以及 Robotiq 2F-85 夹爪。整个环境采用多进程架构，分离了物理仿真、图像渲染和主控制逻辑。

### 1.1 技术栈

- **物理引擎**: MuJoCo 3.2.4
- **渲染**: OpenGL (离屏渲染)
- **进程通信**: Python multiprocessing + shared memory
- **控制频率**: 10 Hz (100ms周期)
- **图像采集**: 640x360 (底盘相机), 640x480 (手腕相机)

## 2. XML 模型结构

### 2.1 整体架构

项目采用模块化的 XML 文件组织方式：

```
models/
├── bobacbot_v2.xml              # 机器人主模型 (bobac3 + eco65b)
├── bobacbot_demo_scene.xml      # 完整仿真场景
├── bobac3/                      # 底盘模型和 mesh
│   ├── bobac3_base.xml
│   └── meshes/
├── eco65b/                      # 机械臂模型和 mesh
│   ├── eco65b.xml
│   └── meshes/
├── 2f85/                        # 夹爪模型和 mesh
│   ├── 2f85.xml
│   └── meshes/
└── textures/                    # 纹理资源
```

### 2.2 Bobac3 底盘配置

底盘采用全向移动设计，具有3个自由度 (x, y, yaw)。这是相比原版 TidyBot 的最大改动点。

```xml
<body name="base_link" pos="0 0 0.035" euler="0 0 0">
  <!-- 设置总质量和惯性矩 -->
  <inertial pos="0 0 0.035" mass="60" diaginertia="1.0 1.0 1.0"/>

  <!-- 添加 x、y、yaw 三个自由度 -->
  <joint name="joint_x" type="slide" axis="1 0 0"/>
  <joint name="joint_y" type="slide" axis="0 1 0"/>
  <joint name="joint_th" type="hinge" axis="0 0 1"/>

  <!-- 底座主体及其视觉和碰撞几何形状 -->
  <geom class="base/visual" mesh="base_link" material="white"/>
  <geom class="base/collision" mesh="base_link"/>
  ...
</body>
```

**关键点**:
- 使用 `slide` 关节实现平移，`hinge` 关节实现旋转
- 质量设为 60kg，接近实际机器人重量
- 前置深度相机位于 `pos="0.05 0 0.2"` 相对底座

### 2.3 eco65b 机械臂配置

eco65b 是 6 自由度机械臂，相比原版 Kinova Gen3 (7DOF) 减少了一个关节。我在逆运动学和控制器中都做了相应适配。

```xml
<body name="eco65b/base_link" pos="0.1 0 0.29" quat="0 0 0 1">
  <inertial pos="0.00059982 -9.2932E-05 0.043301" mass="0.61127"
            diaginertia="0.00080212 0.00080502 0.00066167"/>
  <geom class="eco65b/visual" mesh="eco65b/base_link"/>
  <geom class="eco65b/collision" mesh="eco65b/base_link"/>

  <!-- 6个关节串联 -->
  <body name="Link1" pos="0 0 0.1625">
    <joint name="joint_1" axis="0 0 1" range="-3.1067 3.1067"/>
    ...
  </body>
</body>
```

**关键参数**:
- 关节角度限制: 参照 eco65b 实际硬件规格
- 机械臂base安装高度: 0.29m (从地面)
- 机械臂base相对底座偏移: x=0.1m (前方 10cm)

### 2.4 执行器配置

```xml
<actuator>
  <!-- 底座执行器 - 使用高增益位置控制 -->
  <position name="joint_x" joint="joint_x" kp="1000000" kv="50000"/>
  <position name="joint_y" joint="joint_y" kp="1000000" kv="50000"/>
  <position name="joint_th" joint="joint_th" kp="50000" kv="1000"/>

  <!-- eco65b 机械臂执行器 -->
  <position class="large_actuator" name="joint_1" joint="joint_1"
            ctrlrange="-3.1923 3.1923"/>
  <position class="large_actuator" name="joint_2" joint="joint_2"
            ctrlrange="-2.61666 2.442"/>
  ...

  <!-- 夹爪执行器 - 使用腱驱动 -->
  <general class="2f85" name="fingers_actuator" tendon="split"
           forcerange="-5 5" ctrlrange="0 255"
           gainprm="0.3137255 0 0" biasprm="0 -100 -10"/>
</actuator>
```

**踩过的坑**:
1. 底座 kp 值一开始设的太低 (10000)，导致跟踪误差大
2. 机械臂关节的 ctrlrange 要严格对应实际关节限位，否则 IK 求解会失败
3. 夹爪的 `gainprm` 和 `biasprm` 需要仔细调试才能获得好的抓取效果

## 3. 场景配置

### 3.1 物理参数设置

```xml
<mujoco model="Bobacbot scene">
  <compiler angle="radian"/>

  <!-- 物理引擎配置 -->
  <option integrator="implicitfast"/>
  <option cone="elliptic" impratio="10"/>

  <!-- 包含主要模型 -->
  <include file="bobacbot_v2.xml"/>
  <include file="textures/textures.xml"/>
  <include file="assets/objects/basic/table_left.xml"/>
  <include file="assets/objects/basic/table_right.xml"/>

  <!-- 统计参数 - 影响渲染视角 -->
  <statistic center="0.25 0 0.6" extent="1.0" meansize="0.05"/>

  <visual>
    <headlight diffuse="0.6 0.6 0.6" ambient="0.1 0.1 0.1" specular="0 0 0"/>
    <rgba haze="0.15 0.25 0.35 1"/>
    <global azimuth="120" elevation="-20"/>
  </visual>
</mujoco>
```

### 3.2 可交互物体

在场景中添加了可抓取的立方体作为操作对象：

```xml
<worldbody>
  <light pos="0 0 1.5" directional="true"/>
  <geom name="floor" size="0 0 0.05" type="plane" material="wood-varnished-panels-mat"/>

  <!-- 可抓取的立方体 -->
  <body name="cube" pos="0.8 -0.2 0.7">
    <freejoint/>
    <geom type="box" size="0.03 0.03 0.03" rgba=".5 .7 .5 1" mass="0.1"/>
  </body>
</worldbody>
```

**设计理由**:
- 初始位置 `pos="0.8 -0.2 0.7"`: 在机器人前方约 0.7m，桌面高度
- `mass="0.1"`: 100g，适合 2F-85 夹爪抓取
- 使用 `freejoint` 实现 6DOF 自由移动

### 3.3 相机配置

系统中配置了两个相机：

```xml
<!-- 底盘前置相机 -->
<camera name="base" pos="0.05 0 0.2"
        euler="0 -0.7853981634 -1.5707963268"
        fovy="52.23384539951277"
        resolution="640 360"/>

<!-- 手腕相机 -->
<camera name="wrist" pos="0.08 0 0"
        euler="0 2.7 1.57"
        fovy="41.83792730009236"
        resolution="640 480"/>
```

**相机位置调试经验**:
- base 相机最初放在 `pos="0.2525 0 0.335"`，但视野太高看不到物体，后调整到更低位置
- wrist 相机 `pos="0.08 0 0"` 确保能看到夹爪和前方物体
- `fovy` 视场角参考实际相机参数设置

## 4. 仿真架构设计

### 4.1 多进程架构

整个仿真采用多进程设计，避免 Python GIL 带来的性能瓶颈：

```
Main Process (main.py)
    ├── Physics Process (mujoco_env.MujocoSim)
    │   ├── Control Thread: 处理控制命令
    │   └── Render Thread: 离屏图像渲染
    ├── Visualizer Process: 显示相机图像 (可选)
    └── Shared Memory:
        ├── ShmState: 机器人状态 (base_pose, arm_pos, arm_quat, gripper_pos)
        └── ShmImage: 相机图像 (base_image, wrist_image)
```

### 4.2 代码实现

#### 共享内存状态

```python
class ShmState:
    """
    用于在多进程间共享机器人状态
    数组布局: [base_pose(3), arm_pos(3), arm_quat(4), gripper_pos(1), initialized(1)]
    """
    def __init__(self, existing_instance=None):
        arr = np.empty(3 + 3 + 4 + 1 + 1)
        if existing_instance is None:
            self.shm = shared_memory.SharedMemory(create=True, size=arr.nbytes)
        else:
            self.shm = shared_memory.SharedMemory(name=existing_instance.shm.name)

        self.data = np.ndarray(arr.shape, buffer=self.shm.buf)
        self.base_pose = self.data[:3]
        self.arm_pos = self.data[3:6]
        self.arm_quat = self.data[6:10]
        self.gripper_pos = self.data[10:11]
        self.initialized = self.data[11:12]
```

#### 共享内存图像

```python
class ShmImage:
    """
    用于在多进程间共享相机图像 (RGB, uint8)
    """
    def __init__(self, camera_name=None, width=None, height=None, existing_instance=None):
        if existing_instance is None:
            self.camera_name = camera_name
            arr = np.empty((height, width, 3), dtype=np.uint8)
            self.shm = shared_memory.SharedMemory(create=True, size=arr.nbytes)
        else:
            self.camera_name = existing_instance.camera_name
            arr = existing_instance.data
            self.shm = shared_memory.SharedMemory(name=existing_instance.shm.name)

        self.data = np.ndarray(arr.shape, dtype=np.uint8, buffer=self.shm.buf)
        self.data.fill(0)
```

### 4.3 离屏渲染

渲染器在独立线程中运行，避免阻塞物理仿真：

```python
class Renderer:
    def __init__(self, model, data, shm_image):
        self.model = model
        self.data = data
        self.image = np.empty_like(shm_image.data)
        self.shm_image = ShmImage(existing_instance=shm_image)

        # 设置相机和渲染上下文
        camera_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA.value,
                                      shm_image.camera_name)
        width, height = model.cam_resolution[camera_id]
        self.camera = mujoco.MjvCamera()
        self.camera.fixedcamid = camera_id
        self.camera.type = mujoco.mjtCamera.mjCAMERA_FIXED

        self.rect = mujoco.MjrRect(0, 0, width, height)
        self.gl_context = mujoco.gl_context.GLContext(width, height)
        self.gl_context.make_current()
        self.mjr_context = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_150.value)
        mujoco.mjr_setBuffer(mujoco.mjtFramebuffer.mjFB_OFFSCREEN.value, self.mjr_context)

        self.scene_option = mujoco.MjvOption()
        self.scene = mujoco.MjvScene(model, 10000)

    def render(self):
        """渲染一帧并写入共享内存"""
        self.gl_context.make_current()
        mujoco.mjv_updateScene(self.model, self.data, self.scene_option, None,
                              self.camera, mujoco.mjtCatBit.mjCAT_ALL.value, self.scene)
        mujoco.mjr_render(self.rect, self.scene, self.mjr_context)
        mujoco.mjr_readPixels(self.image, None, self.rect, self.mjr_context)
        # 垂直翻转 (OpenGL 坐标系)
        self.shm_image.data[:] = np.flipud(self.image)
```

**性能优化**:
- 离屏渲染通常耗时 10-50ms，取决于场景复杂度
- 如果渲染时间 >100ms，系统会打印警告，建议缩小 viewer 窗口
- 渲染和物理仿真在不同线程，互不阻塞

## 5. 环境重置与随机化

### 5.1 机器人位置随机化

训练时可以启用位置随机化来提高策略泛化能力：

```python
class MujocoSim:
    def __init__(self, ..., randomize_robot_position=False):
        ...
        self.randomize_robot_position = randomize_robot_position
        self.robot_position_range = {
            'x': [-0.5, 0.5],    # 前后 1m 范围
            'y': [-0.5, 0.5],    # 左右 1m 范围
            'yaw': [0, 0]        # 朝向固定 (也可以随机化)
        }

    def reset(self):
        mujoco.mj_resetData(self.model, self.data)

        if self.randomize_robot_position:
            self.qpos_base[0] = np.random.uniform(*self.robot_position_range['x'])
            self.qpos_base[1] = np.random.uniform(*self.robot_position_range['y'])
            self.qpos_base[2] = np.random.uniform(*self.robot_position_range['yaw'])
        else:
            self.qpos_base[:] = [0.0, 0.0, 0.0]  # 默认原点

        # 随机化立方体位置
        self.qpos_cube[:2] += np.random.uniform(-0.1, 0.1, 2)
        theta = np.random.uniform(-math.pi, math.pi)
        self.qpos_cube[3:7] = np.array([math.cos(theta/2), 0, 0, math.sin(theta/2)])

        mujoco.mj_forward(self.model, self.data)
        self.base_controller.reset()
        self.arm_controller.reset()
```

### 5.2 重力补偿设置

为了让机器人自身不受重力影响（模拟实际硬件的力矩补偿），但物体受重力：

```python
# 为所有 body 启用重力补偿
self.model.body_gravcomp[:] = 1.0

# 对可交互物体禁用重力补偿
body_names = {self.model.body(i).name for i in range(self.model.nbody)}
for object_name in ['cube']:
    if object_name in body_names:
        self.model.body_gravcomp[self.model.body(object_name).id] = 0.0
```

## 6. 坐标系变换

这是个很容易出错的地方。系统涉及多个坐标系：

- **世界坐标系 (World)**: MuJoCo 全局坐标系
- **底盘坐标系 (Base)**: 随底盘移动旋转
- **机械臂坐标系 (Arm)**: 相对底盘的局部坐标
- **手机坐标系 (WebXR)**: 手机 IMU 输出坐标 (后续文档详述)

### 6.1 末端位姿转换到底盘坐标系

```python
def control_callback(self, *_):
    ...
    # 机械臂末端位置 (世界 -> 底盘局部)
    site_xpos = self.site_xpos.copy()
    site_xpos[2] -= self.base_height  # 减去底盘高度
    site_xpos[:2] -= self.qpos_base[:2]  # 减去底盘平移

    # 旋转到底盘局部坐标系
    mujoco.mju_axisAngle2Quat(self.base_quat_inv, [0, 0, 1], -self.qpos_base[2])
    mujoco.mju_rotVecQuat(self.shm_state.arm_pos, site_xpos, self.base_quat_inv)

    # 机械臂末端姿态 (世界 -> 底盘局部)
    mujoco.mju_mat2Quat(self.site_quat, self.site_xmat)
    mujoco.mju_mulQuat(self.shm_state.arm_quat, self.base_quat_inv, self.site_quat)
```

**关键点**:
- 先做平移变换，再做旋转变换
- `base_height` 是底盘中心到地面的高度 (eco65b/base_link 的 z 坐标)
- 四元数乘法顺序很重要: `q_result = q_base_inv * q_site`

## 7. 启动与测试

### 7.1 基本启动

```bash
# 激活环境
mamba activate bobacbot

# 启动仿真 (默认位置，不随机化)
python mujoco_env.py

# 启动仿真 (随机化机器人位置)
python -c "from mujoco_env import MujocoEnv; MujocoEnv(randomize_robot_position=True)"
```

### 7.2 测试脚本

`mujoco_env.py` 的 `__main__` 中有一个测试循环：

```python
if __name__ == '__main__':
    env = MujocoEnv()
    try:
        env.reset()
        obs = env.get_obs()

        # 保持机械臂位姿不变，随机控制夹爪
        target_base_pose = obs['base_pose']
        target_arm_pos = obs['arm_pos']
        target_arm_quat = obs['arm_quat']

        while True:
            action = {
                'base_pose': target_base_pose,
                'arm_pos': target_arm_pos,
                'arm_quat': target_arm_quat,
                'gripper_pos': np.random.rand(1),  # 夹爪随机开合
            }
            env.step(action)
            obs = env.get_obs()
            print([(k, v.shape) if v.ndim == 3 else (k, v) for k, v in obs.items()])
            time.sleep(POLICY_CONTROL_PERIOD)
    finally:
        env.close()
```

## 8. 常见问题

### 8.1 渲染性能问题

**问题**: 渲染耗时 >100ms，系统打印警告

**解决**:
1. 缩小 MuJoCo viewer 窗口大小
2. 减少场景中的 mesh 复杂度
3. 降低相机分辨率

### 8.2 机械臂初始位置不对

**问题**: 机械臂重置后姿态很奇怪

**解决**: 检查 `ArmController.reset()` 中的 `qpos` 初始值是否与 `eco65b.xml` 的 retract keyframe 一致：

```python
self.qpos[:] = np.array([0.0, 0.52, -1.2, 0.34, 1.57, -1.57])
```

### 8.3 共享内存泄漏

**问题**: 程序异常退出后，再次运行报错 `FileExistsError: [Errno 17] File exists`

**解决**: 手动清理共享内存

```bash
# Linux
rm /dev/shm/psm_*

# macOS
# 共享内存在 /tmp，重启自动清理
```

### 8.4 MuJoCo 查看器无响应

**问题**: viewer 窗口卡死，无法交互

**解决**:
- viewer 运行在独立进程，出现问题时可能需要 kill 进程
- 建议使用 `Ctrl+C` 正常退出，确保调用 `env.close()`
- 必要时可以不启动 viewer: `env = MujocoEnv(show_viewer=False)`

## 9. 总结

至此，我们构建了一个完整的 MuJoCo 仿真环境，具备：
- 模块化的 XML 模型结构
- 高性能的多进程架构
- 可配置的场景和随机化
- 完整的坐标系转换

下一步文档将介绍如何在此基础上实现机械臂的运动控制。
