# 策略部署与推理文档

## 1. 概述

本项目采用 **Client-Server 架构** 进行策略推理。策略服务器 (Policy Server) 运行在 GPU 机器上，负责加载训练好的 Diffusion Policy 模型并执行推理；客户端运行在仿真或真实机器人上，负责采集观测并执行动作。两者通过 **ZeroMQ (ZMQ)** 进行通信。

### 1.1 架构概览

```
[GPU 机器 - Policy Server]
  ├─ DiffusionPolicy 模型加载
  ├─ PolicyWrapper (推理循环)
  ├─ ZMQ 服务器 (port 5555)
  └─ 推理队列管理
        ↓ (ZMQ 通信)
[控制机器 - Client]
  ├─ MujocoEnv 仿真环境
  ├─ RemotePolicy 客户端
  ├─ 观测采集 + 图像编码
  └─ 动作执行
```

### 1.2 通信协议

| 方向 | 内容 | 格式 | 频率 |
|------|------|------|------|
| **Client → Server** | 观测数据 | Pickle | 10 Hz |
| **Server → Client** | 动作序列 | Pickle | 10 Hz |
| **Client → Server** | 重置信号 | Pickle | 按需 |

## 2. 策略服务器实现

### 2.1 DiffusionPolicy 类

`DiffusionPolicy` 封装了训练好的模型，提供统一的推理接口。

```python
class DiffusionPolicy:
    def __init__(self, ckpt_path):
        # 1. 加载 checkpoint
        with open(ckpt_path, 'rb') as f:
            payload = torch.load(f, pickle_module=dill)
        cfg = payload['cfg']

        # 2. 实例化 workspace
        cls = hydra.utils.get_class(cfg._target_)
        workspace = cls(cfg)
        workspace.load_payload(payload)

        # 3. 加载策略模型
        policy = workspace.model
        if cfg.training.use_ema:
            policy = workspace.ema_model
        device = torch.device('cuda')
        policy.eval().to(device)

        # 4. 存储属性
        self.policy = policy
        self.device = device
        self.obs_shape_meta = cfg.shape_meta['obs']
        self.rotation_transformer = RotationTransformer(
            from_rep='rotation_6d',  # 训练时使用 6D rotation
            to_rep='quaternion'       # 输出四元数
        )
        self.warmed_up = False

    def reset(self):
        """重置策略状态 (清空历史观测)"""
        self.policy.reset()

    def step(self, obs_sequence):
        """
        执行推理
        Args:
            obs_sequence: 观测序列 (历史观测 + 当前观测)
        Returns:
            act_sequence: 动作序列 (未来 8 步动作)
        """
        obs_dict = self._convert_obs(obs_sequence)

        with torch.no_grad():
            # 首次推理时预热 (编译 CUDA kernel)
            if not self.warmed_up:
                print('Warming up policy...')
                self.policy.predict_action(obs_dict)
                self.warmed_up = True

            # 推理
            result = self.policy.predict_action(obs_dict)
            action = result['action'][0].detach().to('cpu').numpy()

        act_sequence = self._convert_action(action)
        return act_sequence
```

**关键设计**:
1. **EMA 模型**: 如果训练时使用了 EMA (Exponential Moving Average)，推理时加载 EMA 模型
2. **GPU 加速**: 模型部署在 CUDA 上，推理时间 ~115ms (RTX 4080 Laptop)
3. **预热**: 首次推理会触发 CUDA kernel 编译，耗时较长

### 2.2 观测格式转换

```python
def _convert_obs(self, obs_sequence):
    """
    将观测序列转换为模型输入格式
    Args:
        obs_sequence: 列表，每个元素是一个观测字典
    Returns:
        obs_dict: 模型输入字典
    """
    obs_dict_np = {}

    for key, value in self.obs_shape_meta.items():
        if value.get('type') == 'rgb':
            # 图像观测
            images = np.stack([obs[key] for obs in obs_sequence], axis=0)
            assert images.dtype == np.uint8
            images = images.astype(np.float32) / 255.0  # [0, 255] -> [0, 1]
            images = np.transpose(images, (0, 3, 1, 2))  # (T, H, W, C) -> (T, C, H, W)
            assert images.shape[1:] == tuple(value['shape'])
            obs_dict_np[key] = images
        else:
            # 状态观测 (base_pose, arm_pos, ...)
            obs_dict_np[key] = np.stack([obs[key] for obs in obs_sequence], axis=0).astype(np.float32)

    # 转换为 PyTorch tensor 并添加 batch 维度
    obs_dict = dict_apply(obs_dict_np, lambda x: torch.from_numpy(x).unsqueeze(0).to(self.device))
    return obs_dict
```

**数据流**:
```
numpy (uint8) → [0, 1] (float32) → (T, C, H, W) → PyTorch tensor → GPU
```

### 2.3 动作格式转换

```python
def _convert_action(self, action):
    """
    将模型输出转换为动作字典
    Args:
        action: (T, 13) numpy 数组
            [0:3]   base_pose
            [3:6]   arm_pos
            [6:12]  arm_rotation_6d
            [12:13] gripper_pos
    Returns:
        act_sequence: 动作字典列表
    """
    act_sequence = []
    for act in action:
        action_dict = {
            'base_pose': act[:3],
            'arm_pos': act[3:6],
            'arm_quat': self.rotation_transformer.forward(act[6:12])[[1, 2, 3, 0]],  # 6D -> 四元数 [w,x,y,z] -> [x,y,z,w]
            'gripper_pos': act[12:13],
        }
        act_sequence.append(action_dict)
    return act_sequence
```

**6D Rotation 转换**:
- 训练时使用 6D rotation representation (连续、无奇异点)
- 推理时转换为四元数 (MuJoCo 使用)
- 四元数格式转换: `[w,x,y,z]` → `[x,y,z,w]`

### 2.4 PolicyWrapper - 推理循环

`PolicyWrapper` 负责管理观测历史和动作队列，实现时序对齐。

```python
class PolicyWrapper:
    def __init__(self, policy, n_obs_steps=2, n_action_steps=8):
        self.n_obs_steps = n_obs_steps      # 观测窗口长度
        self.n_action_steps = n_action_steps  # 动作序列长度
        self.obs_queue = queue.Queue()
        self.act_queue = queue.Queue()

        # 启动后台推理线程
        threading.Thread(target=self.inference_loop, args=(policy,), daemon=True).start()

    def reset(self):
        """发送重置信号到推理线程"""
        self.obs_queue.put('reset')

    def step(self, obs):
        """
        提交观测并获取动作
        Args:
            obs: 当前观测
        Returns:
            action: 下一步动作 (如果队列为空则返回 None)
        """
        self.obs_queue.put(obs)
        action = None if self.act_queue.empty() else self.act_queue.get()

        if action is None:
            print('Warning: Unexpected idle action queue. Is the latency budget set too low?')

        return action
```

**推理循环实现**:

```python
def inference_loop(self, policy):
    obs_history = deque(maxlen=self.n_obs_steps)  # 观测历史队列
    start_of_episode = True

    while True:
        # 1. 处理新观测
        if not self.obs_queue.empty():
            obs = self.obs_queue.get()

            if obs == 'reset':
                policy.reset()
                obs_history.clear()
                start_of_episode = True
                # 清空动作队列
                while not self.act_queue.empty():
                    self.act_queue.get()
                continue

            obs_history.append(obs)

        # 2. 执行推理 (当观测历史足够且动作队列未满时)
        if self.act_queue.qsize() < LATENCY_STEPS and len(obs_history) == self.n_obs_steps:
            obs_sequence = list(obs_history)
            act_sequence = policy.step(obs_sequence)

            # 3. 动作队列管理
            if not self.act_queue.empty():
                print('Warning: Unexpected action queue backlog.')

            # 4. 时序对齐
            if start_of_episode:
                # Episode 开始: 丢弃前 LATENCY_STEPS 个动作
                act_sequence = act_sequence[:self.n_action_steps - LATENCY_STEPS]
                start_of_episode = False
            else:
                # 稳定状态: 跳过前 LATENCY_STEPS 个动作 (已在队列中)
                act_sequence = act_sequence[LATENCY_STEPS:self.n_action_steps]

            # 5. 入队动作
            for action in act_sequence:
                self.act_queue.put(action)

        time.sleep(0.001)
```

**时序对齐原理**:

```
控制周期: 100ms (10 Hz)
推理延迟: 200ms (含通信)
延迟步数: LATENCY_STEPS = 2

时间轴:
  t=0   t=1   t=2   t=3   t=4   t=5   t=6
  |-----|-----|-----|-----|-----|-----|-----|
  obs0  obs1  obs2  obs3  obs4  obs5  obs6
        |<--- 推理 --->|
              act[0..7]
                    ↓ (延迟 2 步)
                    执行 act[0]
                          执行 act[1]

Episode 开始:
  - 推理返回 act[0..7]
  - 入队 act[0..5] (丢弃 act[6,7])
  - 2 步后执行 act[0]

稳定状态:
  - 推理返回 act[0..7]
  - 入队 act[2..7] (跳过 act[0,1]，已在队列中)
```

**为什么需要时序对齐？**
- Diffusion Policy 每次推理返回未来 8 步动作
- 推理延迟 ~200ms，相当于 2 个控制周期
- 如果不对齐，动作执行会滞后

### 2.5 PolicyServer - ZMQ 服务器

```python
class PolicyServer:
    def __init__(self, policy):
        self.policy = policy

        # 设置 ZMQ 服务器
        context = zmq.Context()
        self.socket = context.socket(zmq.REP)  # Reply socket
        port = 5555
        self.socket.bind(f'tcp://*:{port}')
        print(f'Server started on port {port}')

    def step(self, obs):
        """
        处理客户端请求
        Args:
            obs: 观测字典 (图像为 JPEG 编码)
        Returns:
            action: 动作字典
        """
        # 解码图像
        for k, v in obs.items():
            if k.endswith('image'):
                v = cv.imdecode(v, cv.IMREAD_COLOR)  # JPEG -> numpy
                obs[k] = v

        # 获取动作
        action = self.policy.step(obs)
        return action

    def run(self):
        """主循环"""
        while True:
            # 等待客户端请求
            req = self.socket.recv_pyobj()
            rep = {}

            # 处理重置请求
            if 'reset' in req:
                self.policy.reset()
                print('Policy has been reset')

            # 处理推理请求
            elif 'obs' in req:
                obs = req['obs']
                action = self.step(obs)
                rep['action'] = action

            # 发送响应
            self.socket.send_pyobj(rep)
```

**通信格式**:

```python
# 重置请求
req = {'reset': True}
rep = {}

# 推理请求
req = {'obs': {
    'base_pose': np.array([...]),
    'arm_pos': np.array([...]),
    'base_image': encoded_jpeg,  # JPEG 编码
    'wrist_image': encoded_jpeg,
}}
rep = {'action': {
    'base_pose': np.array([...]),
    'arm_pos': np.array([...]),
    'arm_quat': np.array([...]),
    'gripper_pos': np.array([...]),
}}
```

## 3. 客户端实现

### 3.1 RemotePolicy 类

客户端使用 `RemotePolicy` 连接到策略服务器。

```python
class RemotePolicy(TeleopPolicy):
    def __init__(self):
        super().__init__()  # 继承 TeleopPolicy (手机作为使能设备)

        # 连接到策略服务器
        context = zmq.Context()
        self.socket = context.socket(zmq.REQ)  # Request socket
        self.socket.connect(f'tcp://{POLICY_SERVER_HOST}:{POLICY_SERVER_PORT}')
        print(f'Connected to policy server at {POLICY_SERVER_HOST}:{POLICY_SERVER_PORT}')

        # 使能标志 (手机按住屏幕时为 True)
        self.enabled = False

    def reset(self):
        """重置策略"""
        super().reset()  # 等待手机按 "Start episode"

        # 检查与服务器的连接
        default_timeout = self.socket.getsockopt(zmq.RCVTIMEO)
        self.socket.setsockopt(zmq.RCVTIMEO, 1000)  # 1 秒超时
        self.socket.send_pyobj({'reset': True})
        try:
            self.socket.recv_pyobj()
        except zmq.error.Again as e:
            raise Exception('Could not communicate with policy server') from e
        self.socket.setsockopt(zmq.RCVTIMEO, default_timeout)

        self.enabled = False  # 需要手机按住屏幕才执行策略

    def _step(self, obs):
        """执行一步策略推理"""
        # Episode 结束后切换回遥操作
        if self.episode_ended:
            return self.teleop_controller.step(obs)

        # 策略未启用时返回 None
        if not self.enabled:
            return None

        # 编码图像为 JPEG
        encoded_obs = {}
        for k, v in obs.items():
            if v.ndim == 3:
                # 缩放到策略输入分辨率
                v = cv.resize(v, (POLICY_IMAGE_WIDTH, POLICY_IMAGE_HEIGHT))
                # JPEG 编码 (压缩 ~10x)
                _, v = cv.imencode('.jpg', v)
                encoded_obs[k] = v
            else:
                encoded_obs[k] = v

        # 发送观测到服务器
        req = {'obs': encoded_obs}
        self.socket.send_pyobj(req)

        # 接收动作
        rep = self.socket.recv_pyobj()
        action = rep['action']

        return action

    def _process_message(self, data):
        """处理手机消息"""
        if self.episode_ended:
            # Episode 结束后使用遥操作
            self.teleop_controller.process_message(data)
        else:
            # 手机按住屏幕时启用策略
            self.enabled = 'teleop_mode' in data
```

**手机作为使能设备**:
- 用户按住手机屏幕时，`self.enabled = True`
- 松开屏幕时，`self.enabled = False`，策略停止执行
- 这是一种安全机制，防止策略失控

### 3.2 图像压缩

```python
# 缩放
v = cv.resize(v, (POLICY_IMAGE_WIDTH, POLICY_IMAGE_HEIGHT))  # 84×84

# JPEG 编码
_, v = cv.imencode('.jpg', v)  # 压缩 ~10x

# 原始: 640×360×3 = 691,200 bytes
# 缩放: 84×84×3 = 21,168 bytes
# 压缩: ~2,000 bytes
```

**为什么需要压缩？**
- 网络传输带宽有限
- ZMQ 传输大数据包会增加延迟
- 策略输入分辨率较低 (84×84)，无需传输高分辨率图像

## 4. 部署流程

### 4.1 环境准备

#### GPU 机器 (策略服务器)

```bash
# 1. 激活训练环境
cd training/diffusion_policy
mamba activate robodiff

# 2. 安装额外依赖
pip install pyzmq opencv-python
```

#### 控制机器 (客户端)

```bash
# 激活运行环境
mamba activate bobacbot

# ZMQ 已包含在 requirements.txt 中
```

### 4.2 启动策略服务器

```bash
cd training/diffusion_policy

# 启动服务器 (指定 checkpoint 路径)
python policy_server.py \
    --ckpt-path data/outputs/2025.07.14/00.30.40_train_diffusion_unet_hybrid_sim_pick_place_v2_July/checkpoints/latest.ckpt
```

输出：
```
Server started on port 5555
```

**Checkpoint 路径**:
- 训练完成后，checkpoint 保存在 `data/outputs/<日期>/<实验名>/checkpoints/`
- 可以使用 `latest.ckpt` (最新) 或 `epoch=XXXX-train_loss=X.XXX.ckpt` (指定 epoch)

### 4.3 启动客户端

#### 本地模式 (服务器和客户端在同一机器)

```bash
python main.py --sim
```

客户端会自动连接到 `localhost:5555`。

#### 远程模式 (服务器和客户端在不同机器)

1. **修改配置**

编辑 `constants.py`:
```python
POLICY_SERVER_HOST = '192.168.1.100'  # GPU 机器 IP
POLICY_SERVER_PORT = 5555
```

2. **创建 SSH 隧道** (可选)

如果 GPU 机器不在同一局域网，可以通过 SSH 隧道连接：

```bash
# 在客户端机器上执行
ssh -L 5555:localhost:5555 user@gpu-machine

# 然后保持 constants.py 中的 HOST 为 localhost
```

3. **启动客户端**

```bash
python main.py --sim
```

### 4.4 执行策略推理

1. **手机连接**

   在手机浏览器访问客户端显示的 URL (如 `http://192.168.1.101:5000`)

2. **开始 Episode**

   点击 "Start episode" 进入 AR 模式

3. **启用策略**

   按住手机屏幕，策略开始执行

4. **观察机器人**

   机器人会根据策略执行任务

5. **停止策略**

   松开手机屏幕，策略停止

6. **结束 Episode**

   点击 "End episode" → "Reset env"

## 5. 性能分析

### 5.1 延迟分解

```
总延迟 (t=0 观测 → t=2 执行):
  - 图像压缩: ~5ms (JPEG 编码)
  - 网络传输: ~10ms (ZMQ, 本地) 或 ~50ms (远程)
  - 推理时间: ~115ms (RTX 4080 Laptop)
  - 动作传输: ~5ms
  - 总计: ~135ms (本地) 或 ~185ms (远程)

延迟步数:
  - LATENCY_BUDGET = 200ms
  - LATENCY_STEPS = 2 步
```

### 5.2 吞吐量

```
推理频率: ~8.7 Hz (115ms per inference)
控制频率: 10 Hz
动作队列: 维持 2 步缓冲
```

**结论**: 推理速度略慢于控制频率，但通过动作队列缓冲可以保持实时性。

### 5.3 GPU 利用率

```bash
# 监控 GPU 使用情况
nvidia-smi -l 1
```

典型数据：
- GPU 占用: ~2 GB (模型 + 中间激活)
- GPU 利用率: ~30% (大部分时间在等待观测)
- 功耗: ~100W

## 6. 调试技巧

### 6.1 测试服务器连接

```python
# 测试脚本
import zmq

context = zmq.Context()
socket = context.socket(zmq.REQ)
socket.connect('tcp://localhost:5555')

# 发送测试请求
socket.send_pyobj({'reset': True})
rep = socket.recv_pyobj()
print('Connection OK:', rep)
```

### 6.2 查看推理日志

在 `policy_server.py` 中取消注释日志：

```python
def inference_loop(self, policy):
    ...
    print(f"[{time.time():.3f}] Inference loop: Starting policy step.")
    act_sequence = policy.step(obs_sequence)
    print(f"[{time.time():.3f}] Inference loop: Policy step returned {len(act_sequence)} actions.")
```

### 6.3 监控动作队列

```python
def step(self, obs):
    self.obs_queue.put(obs)
    action = None if self.act_queue.empty() else self.act_queue.get()

    # 添加日志
    print(f"Action queue size: {self.act_queue.qsize()}")

    if action is None:
        print('Warning: Action queue empty!')

    return action
```

### 6.4 可视化推理输出

```python
def step(self, obs):
    action = self.policy.step(obs)

    # 可视化动作
    print(f"Base pose: {action['base_pose']}")
    print(f"Arm pos: {action['arm_pos']}")
    print(f"Gripper: {action['gripper_pos']}")

    return action
```

## 7. 常见问题

### 7.1 服务器无法启动

**症状**: `Address already in use`

**原因**: 端口 5555 被占用

**解决**:
```bash
# 查找占用进程
lsof -i :5555

# 杀死进程
kill -9 <PID>
```

### 7.2 客户端连接超时

**症状**: `Could not communicate with policy server`

**排查**:
1. 检查服务器是否运行
   ```bash
   ps aux | grep policy_server
   ```
2. 检查防火墙
   ```bash
   sudo ufw allow 5555
   ```
3. 检查网络连通性
   ```bash
   ping <server-ip>
   telnet <server-ip> 5555
   ```

### 7.3 推理速度慢

**症状**: 推理时间 >200ms

**排查**:
1. 检查 GPU 是否可用
   ```python
   print(torch.cuda.is_available())
   print(torch.cuda.get_device_name(0))
   ```
2. 检查模型是否在 GPU 上
   ```python
   print(next(policy.parameters()).device)  # 应该是 cuda:0
   ```
3. 检查 GPU 负载
   ```bash
   nvidia-smi
   ```

### 7.4 策略执行异常

**症状**: 机器人运动不合理

**排查**:
1. 检查观测数据
   ```python
   print(obs)  # 打印观测
   ```
2. 检查动作范围
   ```python
   print(f"Arm pos range: {action['arm_pos'].min()}, {action['arm_pos'].max()}")
   ```
3. 检查模型 checkpoint
   - 确认使用的是正确的 checkpoint
   - 检查训练 loss 是否收敛

### 7.5 动作队列空

**症状**: 频繁出现 "Unexpected idle action queue"

**原因**: 推理速度跟不上控制频率

**解决**:
1. 增加 `LATENCY_BUDGET` (如 250ms → 300ms)
2. 使用更快的 GPU
3. 减小模型大小

## 8. 高级功能

### 8.1 批量推理

修改 `DiffusionPolicy` 支持批量观测：

```python
def step(self, obs_sequence_batch):
    """
    批量推理
    Args:
        obs_sequence_batch: [(obs1, obs2), (obs3, obs4), ...]
    Returns:
        act_sequence_batch: [act_seq1, act_seq2, ...]
    """
    obs_dict = self._convert_obs_batch(obs_sequence_batch)
    with torch.no_grad():
        result = self.policy.predict_action(obs_dict)
        action_batch = result['action'].detach().to('cpu').numpy()
    return [self._convert_action(act) for act in action_batch]
```

### 8.2 多模态策略

支持语言指令输入：

```python
def step(self, obs_sequence, instruction):
    """
    条件策略推理
    Args:
        obs_sequence: 观测序列
        instruction: 自然语言指令 (如 "pick up the red block")
    Returns:
        act_sequence: 动作序列
    """
    obs_dict = self._convert_obs(obs_sequence)
    obs_dict['language'] = self._encode_instruction(instruction)  # CLIP/BERT 编码
    with torch.no_grad():
        result = self.policy.predict_action(obs_dict)
        action = result['action'][0].detach().to('cpu').numpy()
    return self._convert_action(action)
```

### 8.3 策略集成

同时部署多个策略并动态切换：

```python
class MultiPolicyServer:
    def __init__(self, policy_configs):
        self.policies = {
            name: DiffusionPolicy(ckpt_path)
            for name, ckpt_path in policy_configs.items()
        }
        self.current_policy = list(self.policies.values())[0]

    def switch_policy(self, policy_name):
        self.current_policy = self.policies[policy_name]
        print(f'Switched to policy: {policy_name}')

    def step(self, obs):
        return self.current_policy.step(obs)
```

## 9. 下一步

策略部署完成后，下一篇文档将介绍如何训练 Diffusion Policy 模型，包括数据准备、配置文件、训练脚本和超参数调优。
