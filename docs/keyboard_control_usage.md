# 键盘控制模式使用说明

## 概述

新增的键盘控制模式允许您通过按空格键来控制仿真推理的开始和结束，无需使用手机 WebXR 界面。

## 安装依赖

```bash
pip install pynput
```

## 使用方法

### 1. 启动策略服务器

首先确保策略服务器正在运行：

```bash
cd training/diffusion_policy
python policy_server.py
```

您应该看到类似的输出：
```
Server started on port 5555
```

### 2. 启动仿真推理（键盘控制模式）

在另一个终端运行：

```bash
python main.py --sim --keyboard
```

**参数说明：**
- `--sim`：使用仿真环境
- `--keyboard`：启用键盘控制模式

**可选参数：**
- `--random`：随机化机器人初始位置
- `--save`：保存演示数据（通常推理时不需要）

### 3. 操作流程

启动后，您会看到以下提示：

```
============================================================
  Keyboard Control Mode Activated
============================================================
  Press SPACE to start episode
============================================================
```

#### 操作步骤：

1. **按空格键（第一次）**：开始 episode，策略开始推理
   ```
   [Keyboard] Episode started! Policy is now running.
   [Keyboard] Press SPACE again to end episode.
   ```

2. **按空格键（第二次）**：结束当前 episode
   ```
   [Keyboard] Episode ended!
   [Keyboard] Press SPACE to reset environment.
   ```

3. **按空格键（第三次）**：重置环境，准备下一个 episode
   ```
   [Keyboard] Environment will reset...
   [Keyboard] Press SPACE to start next episode.
   ```

4. **按空格键（第四次）**：开始新的 episode（循环回到步骤 1）

### 4. 退出程序

按 `Ctrl+C` 退出程序。

## 完整示例

```bash
# 终端 1: 启动策略服务器
cd training/diffusion_policy
python policy_server.py

# 终端 2: 启动仿真推理（键盘控制）
python main.py --sim --keyboard

# 操作：
# 1. 按空格 → 开始 episode
# 2. 等待策略执行完成
# 3. 按空格 → 结束 episode
# 4. 按空格 → 重置环境
# 5. 按空格 → 开始新 episode
# ... 循环
```

## 与原有模式的对比

### 原有模式（手机控制）

```bash
python main.py --sim
```

- 需要手机连接 WebXR 界面
- 通过手机屏幕按钮控制
- 适合真实演示和数据采集

### 键盘控制模式（新增）

```bash
python main.py --sim --keyboard
```

- ✅ 无需手机，更方便调试
- ✅ 按空格键即可控制
- ✅ 适合快速测试和开发
- ✅ 与原有代码完全兼容

## 状态流转图

```
等待开始 ─────► 运行中 ─────► 已结束 ─────► 等待重置 ─────┐
   ▲             (按空格1)     (按空格2)     (按空格3)      │
   │                                                        │
   └────────────────────────────────────────────────────────┘
                        (按空格4，开始新episode)
```

## 实现细节

### 新增文件

- **keyboard_policy.py**：`KeyboardRemotePolicy` 类
  - 继承自 `RemotePolicy`
  - 使用 `pynput` 库监听键盘事件
  - 模拟 WebXR 的状态转换

### 修改文件

- **main.py**：
  - 添加 `--keyboard` 命令行参数
  - 添加策略选择逻辑（5行代码）

### 关键特性

1. **完全兼容原有代码**：不影响现有的手机控制和遥操作模式
2. **简单易用**：只需按空格键即可控制
3. **状态清晰**：每次按键都有明确的提示信息
4. **自动重置**：支持连续多个 episode 的运行

## 调试技巧

### 查看策略服务器日志

在策略服务器终端，您可以看到推理日志：

```
[1234567890.123] Inference loop: Starting policy step.
[1234567890.238] Inference loop: Policy step returned 8 actions.
[1234567890.239] Inference loop: Finished queuing actions.
```

### 检查连接状态

如果看到以下错误：
```
Exception: Could not communicate with policy server
```

**解决方法：**
1. 确认策略服务器正在运行
2. 检查端口 5555 是否被占用
3. 如果使用远程服务器，确认 SSH 隧道已建立

## 常见问题

### Q: 按空格键没有反应？

**A:** 检查以下几点：
1. 确保终端窗口处于激活状态（焦点在终端上）
2. 确认 `pynput` 已正确安装：`pip list | grep pynput`
3. 查看终端是否有错误信息

### Q: 策略不执行动作？

**A:** 确认：
1. 策略服务器已启动并显示 "Server started on port 5555"
2. 已按空格键开始 episode
3. 查看是否有 "Policy is now running" 的提示

### Q: 如何同时使用键盘控制和机器人位置随机化？

**A:** 添加 `--random` 参数：

```bash
python main.py --sim --keyboard --random
```

### Q: 能否保存键盘控制模式下的数据？

**A:** 可以，添加 `--save` 参数：

```bash
python main.py --sim --keyboard --save --output-dir data/keyboard_demos
```

但通常推理模式下不需要保存数据。

## 技术说明

### 键盘监听原理

使用 `pynput.keyboard.Listener` 在后台线程监听键盘事件：

```python
self.keyboard_listener = keyboard.Listener(on_press=self._on_key_press)
self.keyboard_listener.start()
```

### 状态管理

通过修改 `self.teleop_state` 来模拟 WebXR 的状态更新：

- `'episode_started'`：开始 episode
- `'episode_ended'`：结束 episode
- `'reset_env'`：重置环境

### 线程安全

使用 `threading.Lock` 确保状态更新的线程安全：

```python
with self.space_key_lock:
    self._handle_space_press()
```


