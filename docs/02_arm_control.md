# 机械臂仿真运动控制文档

## 1. 概述

本项目采用**逆运动学 (IK) + 在线轨迹生成 (OTG)** 的方案实现 eco65b 机械臂的末端位姿控制。相比直接控制关节角度，这种方案更符合实际应用场景，也简化了手机遥操作和策略学习的接口。

### 1.1 控制流程

```
用户/策略输入 (末端位姿)
    ↓
[IK Solver] 求解关节角度
    ↓
[Ruckig OTG] 生成平滑轨迹
    ↓
[MuJoCo] 执行关节控制
    ↓
状态反馈 (实际末端位姿)
```

### 1.2 技术选型

- **IK 算法**: Damped Least Squares (DLS) + Nullspace Optimization
- **轨迹生成**: Ruckig (实时约束轨迹生成)
- **控制频率**: 10 Hz (策略层) / ~1000 Hz (MuJoCo仿真步)
- **机械臂**: 6 DOF (相比原版 Gen3 的 7 DOF 减少一个关节)

## 2. 逆运动学求解

### 2.1 IKSolver 类设计

IK 求解器的核心是 `ik_solver.py`，我基于 [bullet3](https://github.com/bulletphysics/bullet3) 和 [mjctrl](https://github.com/kevinzakka/mjctrl) 的实现做了适配：

```python
class IKSolver:
    def __init__(self, ee_offset=0.0):
        # 加载 eco65b 模型 (不含夹爪)
        self.model = mujoco.MjModel.from_xml_path('models/eco65b/eco65b.xml')
        self.data = mujoco.MjData(self.model)
        self.model.body_gravcomp[:] = 1.0  # 启用重力补偿

        # 关节初始位姿 (retract pose)
        self.qpos0 = self.model.key('retract').qpos

        # pinch_site: 末端执行器的控制点
        self.site_id = self.model.site('pinch_site').id
        self.site_pos = self.data.site(self.site_id).xpos
        self.site_mat = self.data.site(self.site_id).xmat

        # 可选: 末端偏移 (用于补偿夹爪长度)
        self.model.site(self.site_id).pos[2] += ee_offset

        # 预分配数组
        self.err = np.empty(6)  # 位置误差(3) + 旋转误差(3)
        self.jac = np.empty((6, self.model.nv))  # 雅可比矩阵
        self.damping = DAMPING_COEFF * np.eye(6)  # 阻尼项
        self.eye = np.eye(self.model.nv)
```

**关键参数**:
- `DAMPING_COEFF = 1e-12`: 阻尼系数，防止雅可比矩阵奇异
- `MAX_ANGLE_CHANGE = np.deg2rad(45)`: 单次迭代最大角度变化，防止抖动
- `max_iters = 20`: 最大迭代次数
- `err_thresh = 1e-4`: 收敛阈值

### 2.2 求解算法

```python
def solve(self, pos, quat, curr_qpos, max_iters=20, err_thresh=1e-4):
    """
    求解逆运动学
    Args:
        pos: 目标位置 [x, y, z]
        quat: 目标四元数 [x, y, z, w] (Robot convention)
        curr_qpos: 当前关节角度 (6,)
    Returns:
        qpos: 目标关节角度 (6,)
    """
    # 转换四元数格式: Robot [x,y,z,w] -> MuJoCo [w,x,y,z]
    quat = quat[[3, 0, 1, 2]]

    # 设置当前关节配置作为迭代起点
    self.data.qpos = curr_qpos

    for _ in range(max_iters):
        # Step 1: 计算前向运动学
        mujoco.mj_kinematics(self.model, self.data)
        mujoco.mj_comPos(self.model, self.data)

        # Step 2: 计算位置误差
        self.err_pos[:] = pos - self.site_pos

        # Step 3: 计算旋转误差 (四元数 -> 轴角速度)
        mujoco.mju_mat2Quat(self.site_quat, self.site_mat)
        mujoco.mju_negQuat(self.site_quat_inv, self.site_quat)
        mujoco.mju_mulQuat(self.err_quat, quat, self.site_quat_inv)
        mujoco.mju_quat2Vel(self.err_rot, self.err_quat, 1.0)

        # Step 4: 检查收敛
        if np.linalg.norm(self.err) < err_thresh:
            break

        # Step 5: 计算雅可比矩阵
        mujoco.mj_jacSite(self.model, self.data, self.jac_pos, self.jac_rot, self.site_id)

        # Step 6: Damped Least Squares 求解
        # Δq = J^T (J J^T + λI)^{-1} err
        update = self.jac.T @ np.linalg.solve(self.jac @ self.jac.T + self.damping, self.err)

        # Step 7: Nullspace 优化 (让姿态接近 retract pose)
        qpos0_err = np.mod(self.qpos0 - self.data.qpos + np.pi, 2 * np.pi) - np.pi
        null_proj = self.eye - (self.jac.T @ np.linalg.pinv(self.jac @ self.jac.T + self.damping)) @ self.jac
        update += null_proj @ qpos0_err

        # Step 8: 限制更新幅度
        update_max = np.abs(update).max()
        if update_max > MAX_ANGLE_CHANGE:
            update *= MAX_ANGLE_CHANGE / update_max

        # Step 9: 更新关节角度
        mujoco.mj_integratePos(self.model, self.data.qpos, update, 1.0)

    return self.data.qpos.copy()
```

### 2.3 算法解析

#### Damped Least Squares (DLS)

标准的 Jacobian伪逆法在奇异位形附近会不稳定。DLS 通过添加阻尼项改进：

$$ \Delta q = J^T (J J^T + \lambda I)^{-1} e $$

其中：
- $J$: 雅可比矩阵 (6×6)
- $e$: 误差向量 (6×1)
- $\lambda$: 阻尼系数

#### Nullspace Optimization

由于 eco65b 只有 6 个关节，在6D任务空间中几乎没有冗余。但我们仍然利用 nullspace 让机械臂倾向于保持在 retract 姿态附近：

$$ \Delta q_{null} = (I - J^\dagger J) \cdot (q_0 - q) $$

这样可以避免机械臂出现过度扭曲的姿态。

#### 角度展开 (Angle Unwrapping)

在计算 `qpos0_err` 时使用：

```python
qpos0_err = np.mod(self.qpos0 - self.data.qpos + np.pi, 2 * np.pi) - np.pi
```

确保角度差在 [-π, π] 范围内，避免走"远路"。

### 2.4 性能测试

```python
if __name__ == '__main__':
    ik_solver = IKSolver()

    home_pos, home_quat = np.array([0.3, 0.0, 0.2]), np.array([0.0, 1.0, 0.0, 0.0])
    retract_qpos = np.array([0.0, 0.52, -1.2, 0.34, 1.57, -1.57])

    import time
    start_time = time.time()
    for _ in range(1000):
        qpos = ik_solver.solve(home_pos, home_quat, retract_qpos)
    elapsed_time = time.time() - start_time

    print(f'平均每次调用时间: {elapsed_time:.3f} ms')
    # 输出: ~0.1 ms per call (10 kHz)
```

IK 求解速度非常快，完全满足 10Hz 控制频率的需求。

## 3. 在线轨迹生成 (Ruckig)

虽然 IK 求解了目标关节角度，但不能直接跳变过去，需要生成平滑轨迹。我们使用 [Ruckig](https://github.com/pantor/ruckig) 实现实时轨迹规划。

### 3.1 Ruckig 简介

Ruckig 是一个在线轨迹生成库，能够在满足速度、加速度约束的情况下，生成最优的 jerk-limited 轨迹。相比传统的梯形速度规划，Ruckig 生成的轨迹更平滑。

**特点**:
- 支持任意维度 (我们用6维表示6个关节)
- 实时性: 单次计算 <1ms
- 自动处理速度/加速度限制

### 3.2 ArmController 实现

```python
class ArmController:
    def __init__(self, qpos, qvel, ctrl, qpos_gripper, ctrl_gripper, timestep):
        self.qpos = qpos          # 机械臂关节位置
        self.qvel = qvel          # 机械臂关节速度
        self.ctrl = ctrl          # 机械臂控制输出
        self.qpos_gripper = qpos_gripper
        self.ctrl_gripper = ctrl_gripper

        # 逆运动学求解器
        self.ik_solver = IKSolver(ee_offset=0.0)

        # Ruckig OTG
        num_dofs = 6  # eco65b 6个关节
        self.otg = Ruckig(num_dofs, timestep)
        self.otg_inp = InputParameter(num_dofs)
        self.otg_out = OutputParameter(num_dofs)

        # 设置速度和加速度限制
        self.otg_inp.max_velocity = 6 * [math.radians(80)]      # 80°/s
        self.otg_inp.max_acceleration = 6 * [math.radians(240)] # 240°/s²

        self.otg_res = None
        self.last_command_time = None
```

### 3.3 控制回调

控制回调在每个 MuJoCo 仿真步被调用 (~1000 Hz)：

```python
def control_callback(self, command):
    """
    处理外部命令，更新轨迹生成器
    """
    if command is not None:
        self.last_command_time = time.time()

        if 'arm_pos' in command:
            # Step 1: IK 求解目标关节角度
            qpos = self.ik_solver.solve(command['arm_pos'], command['arm_quat'], self.qpos)

            # Step 2: 角度展开 (避免走远路)
            qpos = self.qpos + np.mod((qpos - self.qpos) + np.pi, 2 * np.pi) - np.pi

            # Step 3: 设置 OTG 目标
            self.otg_inp.target_position = qpos
            self.otg_res = Result.Working

        if 'gripper_pos' in command:
            # 夹爪直接位置控制，不经过 OTG
            self.ctrl_gripper[:] = 255.0 * command['gripper_pos']

    # 命令流中断检测 (超过 2.5 个控制周期)
    if time.time() - self.last_command_time > 2.5 * POLICY_CONTROL_PERIOD:
        self.otg_inp.target_position = self.otg_out.new_position
        self.otg_res = Result.Working

    # 更新轨迹生成器
    if self.otg_res == Result.Working:
        self.otg_res = self.otg.update(self.otg_inp, self.otg_out)
        self.otg_out.pass_to_input(self.otg_inp)  # 传递状态
        self.ctrl[:] = self.otg_out.new_position  # 更新控制输出
```

### 3.4 关键设计点

#### 命令流中断处理

如果策略或遥操作意外中断，机械臂会维持在当前位置：

```python
if time.time() - self.last_command_time > 2.5 * POLICY_CONTROL_PERIOD:
    self.otg_inp.target_position = self.otg_out.new_position
```

这比让机械臂突然松弛下垂更安全。

#### Ruckig 状态传递

```python
self.otg_out.pass_to_input(self.otg_inp)
```

将当前步的输出 (位置、速度) 作为下一步的输入，实现连续轨迹。

#### 角度展开的必要性

IK 求解可能返回多组解（如 θ 和 θ+2π）。为了选择"近的"那个解：

```python
qpos = self.qpos + np.mod((qpos - self.qpos) + np.pi, 2 * np.pi) - np.pi
```

## 4. 底盘控制

底盘控制相对简单，因为没有 IK 步骤，直接使用 Ruckig 生成轨迹。

### 4.1 BaseController 实现

```python
class BaseController:
    def __init__(self, qpos, qvel, ctrl, timestep):
        self.qpos = qpos  # [x, y, theta]
        self.qvel = qvel
        self.ctrl = ctrl

        # Ruckig OTG
        num_dofs = 3
        self.otg = Ruckig(num_dofs, timestep)
        self.otg_inp = InputParameter(num_dofs)
        self.otg_out = OutputParameter(num_dofs)

        # 速度和加速度限制
        self.otg_inp.max_velocity = [0.5, 0.5, 3.14]       # x, y: 0.5m/s; θ: 180°/s
        self.otg_inp.max_acceleration = [0.5, 0.5, 2.36]   # x, y: 0.5m/s²; θ: 135°/s²

    def control_callback(self, command):
        if command is not None and 'base_pose' in command:
            self.last_command_time = time.time()
            self.otg_inp.target_position = command['base_pose']
            self.otg_res = Result.Working

        # 命令流中断检测
        if time.time() - self.last_command_time > 2.5 * POLICY_CONTROL_PERIOD:
            self.otg_inp.target_position = self.qpos
            self.otg_res = Result.Working

        # 更新轨迹
        if self.otg_res == Result.Working:
            self.otg_res = self.otg.update(self.otg_inp, self.otg_out)
            self.otg_out.pass_to_input(self.otg_inp)
            self.ctrl[:] = self.otg_out.new_position
```

### 4.2 底盘速度限制调优

我在实际测试中发现，底盘速度设太高会导致：
1. 物理仿真不稳定 (底盘抖动)
2. 机械臂末端位姿跟踪误差增大

最终选择了 0.5 m/s 的保守值。

## 5. 重置与初始化

### 5.1 机械臂重置

```python
def reset(self):
    """重置机械臂到 retract 姿态"""
    # eco65b 的 retract keyframe
    self.qpos[:] = np.array([0.0, 0.52, -1.2, 0.34, 1.57, -1.57])
    self.ctrl[:] = self.qpos
    self.ctrl_gripper[:] = 0.0  # 夹爪闭合

    # 初始化 OTG
    self.last_command_time = time.time()
    self.otg_inp.current_position = self.qpos
    self.otg_inp.current_velocity = self.qvel
    self.otg_inp.target_position = self.qpos
    self.otg_res = Result.Finished
```

这里的 retract 姿态是我在 eco65b.xml 中定义的：

```xml
<keyframe>
  <key name="retract" qpos="0.0 0.52 -1.2 0.34 1.57 -1.57"/>
</keyframe>
```

选择这个姿态的原因：
- 机械臂收缩，不会碰到桌子
- 末端在机器人前方，方便开始操作
- 远离关节限位

### 5.2 底盘重置

```python
def reset(self):
    """重置底盘到原点"""
    self.ctrl[:] = self.qpos  # 不强制归零，保持当前位置

    self.last_command_time = time.time()
    self.otg_inp.current_position = self.qpos
    self.otg_inp.current_velocity = self.qvel
    self.otg_inp.target_position = self.qpos
    self.otg_res = Result.Finished
```

注意我没有强制 `self.qpos[:] = np.zeros(3)`，因为底盘位置可能在采集数据时被随机化。

## 6. 坐标系与单位

### 6.1 位置单位

- 所有位置使用**米 (m)** 为单位
- 角度使用**弧度 (rad)** 为单位

### 6.2 坐标系定义

- **X轴**: 机器人前方
- **Y轴**: 机器人左侧
- **Z轴**: 竖直向上

```
    Z (up)
    |
    |
    o----> X (forward)
   /
  /
 Y (left)
```

### 6.3 四元数格式

非常容易搞混的地方！不同库使用不同的四元数顺序：

| 库/格式 | 顺序 |
|--------|------|
| **MuJoCo** | [w, x, y, z] |
| **SciPy / Robot** | [x, y, z, w] |
| **WebXR** | {x, y, z, w} (JavaScript对象) |

在 IK 求解中需要转换：

```python
# Robot [x,y,z,w] -> MuJoCo [w,x,y,z]
quat_mujoco = quat_robot[[3, 0, 1, 2]]
```

## 7. 调试与测试

### 7.1 测试 IK 求解

```bash
python ik_solver.py
```

会输出：
```
eco65b关节数量: 6
收缩姿态 (弧度): [ 0.    0.52 -1.2   0.34  1.57 -1.57]
收缩姿态 (度数): [  0.  30. -69.  19.  90. -90.]
平均每次调用时间: 0.100 ms
逆运动学求解结果 (度数): [ 10. -20.  30. -15.  45. -60.]
```

### 7.2 可视化机械臂运动

在 `mujoco_env.py` 的测试循环中，可以让机械臂执行特定轨迹：

```python
env = MujocoEnv()
env.reset()

# 获取初始姿态
obs = env.get_obs()
start_pos = obs['arm_pos'].copy()
start_quat = obs['arm_quat'].copy()

# 让机械臂画圆
for t in range(1000):
    theta = 2 * np.pi * t / 1000
    target_pos = start_pos + 0.1 * np.array([np.cos(theta), np.sin(theta), 0])

    action = {
        'base_pose': obs['base_pose'],
        'arm_pos': target_pos,
        'arm_quat': start_quat,
        'gripper_pos': np.array([0.5]),
    }
    env.step(action)
    time.sleep(POLICY_CONTROL_PERIOD)
```

### 7.3 检查 IK 收敛性

有时候目标位姿超出工作空间，IK 无法收敛。可以添加日志检测：

```python
def solve(self, pos, quat, curr_qpos, max_iters=20, err_thresh=1e-4):
    ...
    for i in range(max_iters):
        ...
        if np.linalg.norm(self.err) < err_thresh:
            break

    # 收敛检查
    final_err = np.linalg.norm(self.err)
    if final_err > 0.01:  # 1cm 误差
        print(f'Warning: IK did not converge. Final error: {final_err:.4f}')

    return self.data.qpos.copy()
```

## 8. 常见问题

### 8.1 机械臂抖动

**原因**:
- IK 求解在奇异位形附近
- Ruckig 加速度限制设置过高
- 控制频率不匹配

**解决**:
1. 增大 `DAMPING_COEFF` (但会降低精度)
2. 降低 `max_acceleration`
3. 检查控制命令是否连续

### 8.2 机械臂无法到达目标

**原因**:
- 目标超出工作空间
- 关节限位冲突
- IK 初始猜测不好

**解决**:
1. 在发送命令前检查目标位置是否在工作空间内
2. 检查 `eco65b.xml` 中的 `range` 限制
3. 使用当前关节角度作为 IK 初始猜测（我们已经这样做了）

### 8.3 夹爪控制失效

**原因**:
- `ctrl_gripper` 映射错误
- 腱 (tendon) 配置问题

**解决**:
检查 XML 配置：
```xml
<tendon>
  <fixed name="split">
    <joint joint="right_driver_joint" coef="0.5"/>
    <joint joint="left_driver_joint" coef="0.5"/>
  </fixed>
</tendon>

<actuator>
  <general class="2f85" name="fingers_actuator" tendon="split"
           forcerange="-5 5" ctrlrange="0 255" .../>
</actuator>
```

确保 `ctrlrange="0 255"` 对应我们的控制范围。

### 8.4 速度限制不生效

**问题**: 机械臂移动过快，超过设定的 `max_velocity`

**原因**: Ruckig 的 `timestep` 设置不正确

**解决**:
```python
self.otg = Ruckig(num_dofs, timestep)  # timestep 应该是 MuJoCo 的仿真步长
```

检查：
```python
print(f"MuJoCo timestep: {self.model.opt.timestep}")  # 应该是 0.002 (500Hz)
```

## 9. 性能优化

### 9.1 IK 求解加速

- 使用当前关节角度作为初始猜测（已实现）
- 减少 `max_iters`（收敛通常 <10 次迭代）
- 预计算雅可比矩阵（MuJoCo 已经优化）

### 9.2 Ruckig 参数调优

调整 `max_velocity` 和 `max_acceleration` 平衡速度和平滑度：

| 场景 | max_velocity | max_acceleration |
|------|--------------|------------------|
| **数据采集** | 80°/s | 240°/s² |
| **快速运动** | 120°/s | 360°/s² |
| **精细操作** | 40°/s | 120°/s² |

### 9.3 减少内存分配

IKSolver 中预分配了所有数组：

```python
self.err = np.empty(6)
self.jac = np.empty((6, self.model.nv))
```

避免在循环中创建临时数组。

## 10. 下一步

机械臂控制是整个系统的核心。下一篇文档将介绍如何通过手机遥操作来控制机械臂，涉及 WebXR 坐标系转换和物理映射。
