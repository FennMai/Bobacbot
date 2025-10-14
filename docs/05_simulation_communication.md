# 仿真通信文档

## 1. 概述

本项目采用**多进程架构**来实现高性能的 MuJoCo 仿真环境。不同于传统的单线程仿真，我们将物理仿真、图像渲染和主控制逻辑分离到不同的进程/线程中，通过**共享内存 (Shared Memory)** 和**消息队列 (Queue)** 进行通信。这种设计避免了 Python GIL 的性能瓶颈，同时确保了实时性和高效的数据传输。

### 1.1 架构概览

```
[主进程 - Main Process]
  ├─ MujocoEnv 类
  ├─ 共享内存创建 (ShmState, ShmImage)
  ├─ 命令队列 (command_queue)
  └─ 策略/控制逻辑
        ↓ (通过 command_queue 发送控制命令)
[物理仿真进程 - Physics Process]
  ├─ MujocoSim 类
  ├─ 控制回调 (control_callback, ~1000 Hz)
  ├─ BaseController / ArmController
  └─ 状态发布 (写入 ShmState)
        ↓ (独立线程)
[渲染线程 - Render Thread]
  ├─ Renderer 类
  ├─ 离屏渲染 (OpenGL)
  └─ 图像发布 (写入 ShmImage)
        ↓ (可选)
[可视化进程 - Visualizer Process]
  └─ OpenCV 窗口显示
```

### 1.2 通信机制

| 通信方式 | 用途 | 方向 | 频率 |
|---------|------|------|------|
| **共享内存 (ShmState)** | 机器人状态 | 仿真 → 主进程 | 实时读取 |
| **共享内存 (ShmImage)** | 相机图像 | 渲染 → 主进程/可视化 | 实时读取 |
| **消息队列 (Queue)** | 控制命令 | 主进程 → 仿真 | 10 Hz |
| **消息队列 (Queue)** | 重置信号 | 主进程 → 仿真 | 按需 |

## 2. 共享内存设计

### 2.1 ShmState - 机器人状态共享

`ShmState` 用于在多进程间共享机器人的实时状态，包括底盘位姿、机械臂末端位姿、夹爪位置等。

```python
class ShmState:
    """
    共享内存状态类
    数组布局: [base_pose(3), arm_pos(3), arm_quat(4), gripper_pos(1), initialized(1)]
    """
    def __init__(self, existing_instance=None):
        # 创建包含所有状态信息的数组
        arr = np.empty(3 + 3 + 4 + 1 + 1)  # 总共 12 个浮点数

        if existing_instance is None:
            # 创建新的共享内存 (在主进程中)
            self.shm = shared_memory.SharedMemory(create=True, size=arr.nbytes)
        else:
            # 连接到已存在的共享内存 (在子进程中)
            self.shm = shared_memory.SharedMemory(name=existing_instance.shm.name)

        # 将共享内存缓冲区映射为 numpy 数组
        self.data = np.ndarray(arr.shape, buffer=self.shm.buf)

        # 分配数组的不同部分给各个状态变量
        self.base_pose = self.data[:3]          # 底盘位姿 [x, y, theta]
        self.arm_pos = self.data[3:6]           # 机械臂末端位置 [x, y, z]
        self.arm_quat = self.data[6:10]         # 机械臂末端四元数 [w, x, y, z]
        self.gripper_pos = self.data[10:11]     # 夹爪位置 [0-1]
        self.initialized = self.data[11:12]     # 初始化标志

        self.initialized[:] = 0.0  # 初始化为未初始化状态

    def close(self):
        """关闭共享内存连接"""
        self.shm.close()
```

**内存布局**:
```
Offset | Size | Field
-------|------|---------------
0      | 3    | base_pose
3      | 3    | arm_pos
6      | 4    | arm_quat
10     | 1    | gripper_pos
11     | 1    | initialized
-------|------|---------------
Total: 12 * 8 bytes = 96 bytes
```

**关键设计点**:
1. **数组切片 (Array Slicing)**: 使用 numpy 数组切片直接映射到共享内存，避免拷贝
2. **初始化标志 (initialized)**: 用于同步，确保主进程在状态初始化完成后才开始读取
3. **连接模式**: 主进程创建 (`create=True`)，子进程连接 (`name=...`)

### 2.2 ShmImage - 图像共享

`ShmImage` 用于在多进程间共享相机图像，支持任意分辨率的 RGB 图像。

```python
class ShmImage:
    """
    共享内存图像类
    用于在多进程间共享相机图像数据 (RGB, uint8)
    """
    def __init__(self, camera_name=None, width=None, height=None, existing_instance=None):
        if existing_instance is None:
            # 创建新的图像共享内存
            self.camera_name = camera_name
            arr = np.empty((height, width, 3), dtype=np.uint8)  # RGB 图像
            self.shm = shared_memory.SharedMemory(create=True, size=arr.nbytes)
        else:
            # 连接到已存在的图像共享内存
            self.camera_name = existing_instance.camera_name
            arr = existing_instance.data
            self.shm = shared_memory.SharedMemory(name=existing_instance.shm.name)

        # 将共享内存映射为图像数组
        self.data = np.ndarray(arr.shape, dtype=np.uint8, buffer=self.shm.buf)
        self.data.fill(0)  # 初始化为全黑图像

    def close(self):
        """关闭图像共享内存连接"""
        self.shm.close()
```

**内存大小**:
- 底盘相机: `640 × 360 × 3 = 691,200 bytes` (~675 KB)
- 手腕相机: `640 × 480 × 3 = 921,600 bytes` (~900 KB)

**为什么用 uint8？**
- RGB 图像每个像素 [0, 255]，uint8 节省内存
- 与 OpenCV/PIL 兼容，不需要额外转换

### 2.3 共享内存的生命周期管理

```python
class MujocoEnv:
    def __init__(self, ...):
        # 1. 创建共享内存 (主进程)
        self.shm_state = ShmState()

        if self.render_images:
            self.shm_images = []
            model = mujoco.MjModel.from_xml_path(self.mjcf_path)
            for camera_id in range(model.ncam):
                camera_name = model.camera(camera_id).name
                width, height = model.cam_resolution[camera_id]
                self.shm_images.append(ShmImage(camera_name, width, height))

        # 2. 启动子进程 (会自动连接到共享内存)
        mp.Process(target=self.physics_loop, daemon=True).start()

    def physics_loop(self):
        # 3. 子进程连接到共享内存
        sim = MujocoSim(..., shm_state=self.shm_state, ...)
        # 在 MujocoSim.__init__ 中:
        # self.shm_state = ShmState(existing_instance=shm_state)

    def close(self):
        # 4. 清理共享内存
        self.shm_state.close()
        self.shm_state.shm.unlink()  # 删除共享内存

        if self.render_images:
            for shm_image in self.shm_images:
                shm_image.close()
                shm_image.shm.unlink()
```

**重要**: 必须调用 `unlink()` 来删除共享内存，否则会泄漏（Linux 下位于 `/dev/shm/`）。

## 3. 消息队列通信

### 3.1 命令队列设计

`command_queue` 用于从主进程发送控制命令到物理仿真进程。

```python
class MujocoEnv:
    def __init__(self, ...):
        # 创建容量为 1 的队列
        self.command_queue = mp.Queue(1)

        # 启动物理仿真进程
        mp.Process(target=self.physics_loop, daemon=True).start()

    def step(self, action):
        """执行一步动作"""
        self.command_queue.put(action)

    def reset(self):
        """重置环境"""
        self.shm_state.initialized[:] = 0.0  # 标记为未初始化
        self.command_queue.put('reset')      # 发送重置命令

        # 等待状态初始化完成
        while self.shm_state.initialized == 0.0:
            time.sleep(0.01)
```

**为什么容量为 1？**
- 避免命令积压：最新的命令会覆盖旧的命令
- 低延迟：控制器始终使用最新的目标位姿
- 简化逻辑：不需要处理队列满的情况

### 3.2 控制回调处理

物理仿真进程通过控制回调函数读取命令队列：

```python
class MujocoSim:
    def __init__(self, ..., command_queue, ...):
        self.command_queue = command_queue
        # 设置控制回调函数
        mujoco.set_mjcb_control(self.control_callback)

    def control_callback(self, *_):
        """
        控制回调函数
        在每个仿真步骤 (~1000 Hz) 中被 MuJoCo 调用
        """
        # 检查是否有新的控制命令
        command = None if self.command_queue.empty() else self.command_queue.get()

        if command == 'reset':
            self.reset()
        else:
            # 调用各控制器的控制回调
            self.base_controller.control_callback(command)
            self.arm_controller.control_callback(command)

        # 更新状态到共享内存
        self.shm_state.base_pose[:] = self.qpos_base
        # ... (更新其他状态)
        self.shm_state.initialized[:] = 1.0
```

**非阻塞读取**:
- `command_queue.empty()`: 检查队列是否为空
- `command_queue.get()`: 非阻塞读取（前提是已检查非空）
- 如果队列为空，使用上一次的命令（控制器内部维护）

### 3.3 命令格式

#### 动作命令

```python
action = {
    'base_pose': np.array([x, y, theta]),      # 目标底盘位姿
    'arm_pos': np.array([x, y, z]),            # 目标机械臂位置
    'arm_quat': np.array([x, y, z, w]),        # 目标机械臂四元数
    'gripper_pos': np.array([pos]),            # 目标夹爪位置 [0-1]
}
```

#### 重置命令

```python
command = 'reset'  # 字符串
```

**命令处理逻辑**:
```python
if command == 'reset':
    self.reset()  # 重置环境
elif command is not None:
    # 更新控制器目标
    if 'base_pose' in command:
        self.base_controller.control_callback(command)
    if 'arm_pos' in command:
        self.arm_controller.control_callback(command)
```

## 4. 多进程架构详解

### 4.1 主进程 (Main Process)

主进程负责：
- 创建共享内存和队列
- 启动子进程
- 运行策略/控制逻辑
- 读取观测数据
- 发送控制命令

```python
class MujocoEnv:
    def __init__(self, ...):
        # 1. 创建通信资源
        self.command_queue = mp.Queue(1)
        self.shm_state = ShmState()
        self.shm_images = [...]

        # 2. 启动物理仿真进程
        mp.Process(target=self.physics_loop, daemon=True).start()

        # 3. 启动图像可视化进程 (可选)
        if self.show_images:
            mp.Process(target=self.visualizer_loop, daemon=True).start()

    def get_obs(self):
        """从共享内存读取观测数据"""
        obs = {
            'base_pose': self.shm_state.base_pose.copy(),
            'arm_pos': self.shm_state.arm_pos.copy(),
            'arm_quat': self.shm_state.arm_quat[[1, 2, 3, 0]],  # [w,x,y,z] -> [x,y,z,w]
            'gripper_pos': self.shm_state.gripper_pos.copy(),
        }
        if self.render_images:
            for shm_image in self.shm_images:
                obs[f'{shm_image.camera_name}_image'] = shm_image.data.copy()
        return obs
```

**为什么使用 `.copy()`？**
- 共享内存是动态更新的，必须拷贝数据避免竞态条件
- 拷贝的数据是策略使用的"快照"

### 4.2 物理仿真进程 (Physics Process)

物理仿真进程负责：
- 运行 MuJoCo 物理仿真
- 执行控制回调（处理命令、更新状态）
- 显示 MuJoCo viewer (可选)

```python
def physics_loop(self):
    """物理仿真循环 (在独立进程中运行)"""
    # 创建仿真实例
    sim = MujocoSim(
        self.mjcf_path,
        self.command_queue,
        self.shm_state,
        show_viewer=self.show_viewer,
        randomize_robot_position=self.randomize_robot_position
    )

    # 启动渲染线程 (在同一进程内)
    if self.render_images:
        Thread(target=self.render_loop, args=(sim.model, sim.data), daemon=True).start()

    # 启动仿真 (阻塞调用)
    sim.launch()
```

**为什么是独立进程？**
- 绕过 Python GIL，MuJoCo C++ 代码可以并行运行
- 避免策略计算阻塞物理仿真
- 崩溃隔离（仿真崩溃不影响主进程）

### 4.3 渲染线程 (Render Thread)

渲染线程负责：
- 离屏渲染相机图像 (OpenGL)
- 更新图像到共享内存

```python
def render_loop(self, model, data):
    """渲染循环 (在独立线程中运行)"""
    # 设置渲染器
    renderers = [Renderer(model, data, shm_image) for shm_image in self.shm_images]

    # 持续渲染相机图像
    while True:
        start_time = time.time()
        for renderer in renderers:
            renderer.render()
        render_time = time.time() - start_time

        # 如果渲染时间过长则发出警告
        if render_time > 0.1:  # 10 fps
            print(f'警告: 离屏渲染耗时 {1000 * render_time:.1f} 毫秒')
```

**为什么是线程而非进程？**
- 渲染需要访问 `model` 和 `data`，在同一进程内避免序列化开销
- OpenGL 上下文通常绑定到线程，跨进程传递复杂
- 渲染不是 CPU 密集型（主要是 GPU），线程足够

### 4.4 可视化进程 (Visualizer Process)

可视化进程负责：
- 从共享内存读取图像
- 使用 OpenCV 显示窗口

```python
def visualizer_loop(self):
    """图像可视化循环 (在独立进程中运行)"""
    # 连接到共享内存
    shm_images = [ShmImage(existing_instance=shm_image) for shm_image in self.shm_images]

    last_imshow_time = time.time()
    while True:
        # 控制显示帧率为 10fps
        while time.time() - last_imshow_time < 0.1:
            time.sleep(0.01)
        last_imshow_time = time.time()

        # 显示所有相机图像
        for i, shm_image in enumerate(shm_images):
            cv.imshow(shm_image.camera_name, cv.cvtColor(shm_image.data, cv.COLOR_RGB2BGR))
            cv.moveWindow(shm_image.camera_name, 640 * i, -100)
        cv.waitKey(1)
```

**为什么是独立进程？**
- OpenCV 窗口事件循环可能阻塞
- 允许用户关闭窗口而不影响仿真
- 隔离 GUI 相关的库依赖

## 5. 同步机制

### 5.1 初始化同步

环境重置后，主进程需要等待仿真初始化完成才能读取有效数据。

```python
def reset(self):
    """重置环境到初始状态"""
    # 1. 标记为未初始化
    self.shm_state.initialized[:] = 0.0

    # 2. 发送重置命令
    self.command_queue.put('reset')

    # 3. 等待状态初始化完成
    while self.shm_state.initialized == 0.0:
        time.sleep(0.01)

    # 4. 等待图像渲染完成
    if self.render_images:
        while any(np.all(shm_image.data == 0) for shm_image in self.shm_images):
            time.sleep(0.01)
```

**同步流程**:
```
主进程                        仿真进程
  |                               |
  | initialized = 0.0            |
  |------------------------------>|
  | command_queue.put('reset')   |
  |------------------------------>|
  |                               | mujoco.mj_resetData()
  |                               | self.reset()
  |                               | mj_forward()
  |                               | initialized = 1.0
  |<------------------------------|
  | while initialized == 0.0:    |
  |   sleep(0.01)                |
  | (退出循环)                   |
```

### 5.2 图像同步

图像渲染是异步的，主进程通过检查图像内容判断是否完成：

```python
# 等待图像不全为 0 (假设全黑不是有效图像)
while any(np.all(shm_image.data == 0) for shm_image in self.shm_images):
    time.sleep(0.01)
```

**注意**: 这个假设在某些场景下可能不成立（如相机面向黑色物体）。更可靠的方法是添加图像初始化标志。

### 5.3 命令流中断检测

控制器检测命令流是否中断，如果超过 2.5 个控制周期没有收到命令，维持当前位姿：

```python
def control_callback(self, command):
    if command is not None:
        self.last_command_time = time.time()
        # ... (处理命令)

    # 如果命令流中断，维持当前位姿
    if time.time() - self.last_command_time > 2.5 * POLICY_CONTROL_PERIOD:
        self.otg_inp.target_position = self.qpos  # 底盘
        # self.otg_inp.target_position = self.otg_out.new_position  # 机械臂
        self.otg_res = Result.Working
```

**为什么是 2.5 倍？**
- 控制周期是 100ms (10 Hz)
- 2.5 倍 = 250ms，允许一定的抖动
- 避免策略意外中断时机器人失控

## 6. 性能分析

### 6.1 通信开销

#### 共享内存读写

```python
# 写入 (仿真进程)
self.shm_state.base_pose[:] = self.qpos_base  # ~10 ns (指针赋值)

# 读取 (主进程)
base_pose = self.shm_state.base_pose.copy()  # ~100 ns (拷贝 3 个浮点数)
```

**结论**: 共享内存的读写开销几乎可以忽略 (<1 μs)。

#### 消息队列

```python
# 发送 (主进程)
self.command_queue.put(action)  # ~10 μs (序列化 + 系统调用)

# 接收 (仿真进程)
command = self.command_queue.get()  # ~5 μs (反序列化)
```

**结论**: 队列开销也很小，对 10 Hz 控制频率完全可接受。

### 6.2 渲染开销

```python
# 离屏渲染
for renderer in renderers:
    renderer.render()  # 10-50 ms per camera
```

**瓶颈**: 渲染是最大的性能开销，取决于：
- 场景复杂度 (mesh 数量和精度)
- 相机分辨率
- GPU 性能

**优化建议**:
1. 减少相机分辨率（如 320×180）
2. 简化 mesh 模型
3. 使用 GPU 渲染（MuJoCo 默认）

### 6.3 频率对比

| 组件 | 频率 | 说明 |
|------|------|------|
| **物理仿真** | ~1000 Hz | `model.opt.timestep = 0.002` |
| **控制回调** | ~1000 Hz | 每个仿真步调用 |
| **策略控制** | 10 Hz | `POLICY_CONTROL_PERIOD = 0.1` |
| **图像渲染** | ~20 Hz | 取决于渲染时间 |
| **图像显示** | 10 Hz | `visualizer_loop` 限制 |

## 7. 调试技巧

### 7.1 检查共享内存泄漏

```bash
# Linux
ls -lh /dev/shm/psm_*

# 如果看到遗留的共享内存文件，手动删除
rm /dev/shm/psm_*
```

### 7.2 监控通信频率

在主进程添加统计：

```python
class MujocoEnv:
    def __init__(self, ...):
        self.step_count = 0
        self.last_print_time = time.time()

    def step(self, action):
        self.command_queue.put(action)
        self.step_count += 1

        # 每秒打印一次
        if time.time() - self.last_print_time > 1.0:
            print(f"Command rate: {self.step_count} Hz")
            self.step_count = 0
            self.last_print_time = time.time()
```

应该看到 ~10 Hz。

### 7.3 验证状态同步

```python
def get_obs(self):
    obs = self.shm_state.base_pose.copy()
    print(f"Base pose: {obs}")
    return obs
```

观察输出是否实时更新。如果一直是同一个值，说明仿真进程没有正常更新。

### 7.4 检查队列积压

```python
def control_callback(self, *_):
    if not self.command_queue.empty():
        queue_size = self.command_queue.qsize()
        if queue_size > 0:
            print(f"Warning: Queue has {queue_size} pending commands")
        command = self.command_queue.get()
```

如果队列持续积压，说明仿真频率跟不上命令频率。

## 8. 常见问题

### 8.1 共享内存 FileExistsError

**症状**: 程序异常退出后，再次运行报错

```
FileExistsError: [Errno 17] File exists: '/psm_abc123'
```

**原因**: 上次程序未正确清理共享内存

**解决**:
```bash
rm /dev/shm/psm_*
```

或在代码中添加异常处理：

```python
try:
    env = MujocoEnv()
    # ... (运行代码)
finally:
    env.close()  # 确保调用 close()
```

### 8.2 观测数据不更新

**症状**: `get_obs()` 返回的数据一直不变

**排查**:
1. 检查仿真进程是否启动成功
   ```python
   print(f"Physics process alive: {physics_process.is_alive()}")
   ```
2. 检查 `initialized` 标志是否变为 1.0
   ```python
   print(f"Initialized: {self.shm_state.initialized}")
   ```
3. 检查控制回调是否被调用
   ```python
   def control_callback(self, *_):
       print("Control callback called")  # 添加日志
   ```

### 8.3 图像显示全黑

**症状**: OpenCV 窗口显示全黑图像

**排查**:
1. 检查渲染线程是否启动
2. 检查相机配置是否正确（位置、朝向）
3. 添加渲染日志
   ```python
   def render(self):
       self.gl_context.make_current()
       # ... (渲染代码)
       print(f"Rendered {self.image.shape}, min={self.image.min()}, max={self.image.max()}")
   ```

### 8.4 进程无法结束

**症状**: Ctrl+C 后程序无法退出

**原因**: 子进程未设置 `daemon=True`

**解决**:
```python
mp.Process(target=self.physics_loop, daemon=True).start()
```

设置 `daemon=True` 后，主进程退出时会自动终止子进程。

### 8.5 渲染性能差

**症状**: 离屏渲染耗时 >100ms

**排查**:
1. 检查 MuJoCo viewer 窗口大小（窗口越大，离屏渲染越慢）
2. 减少 mesh 复杂度
3. 降低相机分辨率

**临时方案**: 缩小 viewer 窗口

## 9. 高级优化

### 9.1 零拷贝图像传递

当前实现中，主进程读取图像时会拷贝：

```python
obs[f'{shm_image.camera_name}_image'] = shm_image.data.copy()
```

对于大图像（640×480×3），拷贝开销 ~1ms。可以优化为：

```python
# 直接返回 view (注意: 不能修改)
obs[f'{shm_image.camera_name}_image'] = shm_image.data
```

**风险**: 如果策略修改图像数据，会污染共享内存。

### 9.2 批量图像压缩

在渲染线程中直接压缩为 JPEG：

```python
class Renderer:
    def render(self):
        # ... (渲染)
        _, encoded = cv.imencode('.jpg', self.image, [cv.IMWRITE_JPEG_QUALITY, 90])
        # 写入共享内存 (需要调整 ShmImage 结构)
```

这样可以减少图像数据大小 (~10x)，加快传输速度。

### 9.3 使用 mmap 代替共享内存

对于超大数据（如 Episode Replay Buffer），可以使用 `mmap`：

```python
import mmap

# 创建文件映射
f = open('/tmp/replay_buffer.dat', 'r+b')
mm = mmap.mmap(f.fileno(), length=1024*1024*100)  # 100 MB

# 多进程共享
data = np.ndarray((100, 1000), dtype=np.float32, buffer=mm)
```

## 10. 总结

本项目的通信架构设计有以下特点：

1. **高效**: 共享内存零拷贝，通信开销 <1 μs
2. **实时**: 物理仿真 1000 Hz，控制频率 10 Hz
3. **解耦**: 物理、渲染、控制逻辑完全分离
4. **稳定**: 初始化同步、命令流检测保证系统鲁棒性

下一篇文档将介绍如何使用遥操作进行数据采集，包括 Episode 管理和数据存储格式。
