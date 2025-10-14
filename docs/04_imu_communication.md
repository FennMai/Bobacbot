# 手机 IMU 通信调用文档

## 1. 概述

本项目使用手机作为遥操作输入设备，通过 WebXR API 获取手机的 6DOF 位姿数据（来自 IMU 传感器融合），并通过 WebSocket 实时传输到服务器。相比传统的操纵杆或键盘，这种方案能够实现更自然的机器人控制。

### 1.1 通信架构

```
[手机浏览器]
    ↓ WebXR API (获取 IMU 位姿)
[JavaScript Client]
    ↓ Socket.IO (WebSocket)
[Flask-SocketIO Server]
    ↓ Python Queue
[TeleopController]
    ↓ 坐标转换 + 控制映射
[MujocoEnv]
```

### 1.2 技术栈

- **前端**: WebXR Device API (浏览器原生 AR 支持)
- **通信**: Socket.IO 4.7.5 (基于 WebSocket)
- **后端**: Flask + Flask-SocketIO (Python)
- **数据格式**: JSON (实时位姿、控制模式、时间戳)
- **延迟**: 5GHz WiFi 下典型 RTT ~7ms

## 2. WebXR IMU 数据获取

### 2.1 WebXR 会话初始化

手机端使用 WebXR Immersive AR 模式获取传感器数据：

```javascript
function onRequestSession() {
    return navigator.xr.requestSession('immersive-ar', {
        optionalFeatures: ['dom-overlay'],
        domOverlay: { root: document.getElementById('overlay') },
    }).then((session) => {
        xrButton.setSession(session);
        session.isImmersive = true;
        onSessionStarted(session);
    });
}
```

**关键点**:
- `immersive-ar`: 沉浸式 AR 模式，允许访问相机和 IMU
- `dom-overlay`: 保留 UI 交互层（显示 RTT、控制按钮等）
- 浏览器支持: Chrome/Edge (Android), Safari (iOS 仅部分支持)

### 2.2 参考坐标系设置

```javascript
function onSessionStarted(session) {
    session.addEventListener('end', onSessionEnded);
    const canvas = document.createElement('canvas');
    gl = canvas.getContext('webgl', { xrCompatible: true });

    // 请求本地坐标系 (手机初始位置为原点)
    session.requestReferenceSpace('local').then((refSpace) => {
        xrRefSpace = refSpace;
        session.requestAnimationFrame(onXRFrame);  // 开始渲染循环
    });
}
```

**参考坐标系类型**:
- `local`: 手机启动 AR 时的位置为原点，适合遥操作
- `local-floor`: 地面为 z=0 平面
- `unbounded`: 无限制追踪空间（本项目未使用）

### 2.3 实时位姿读取

WebXR 在每一帧回调中提供最新的位姿：

```javascript
function onXRFrame(t, frame) {
    frame.session.requestAnimationFrame(onXRFrame);  // 递归调用

    // 准备发送的数据
    const data = { timestamp: Date.now(), device_id: deviceId };

    // 只有当用户按住屏幕时才发送位姿
    if (touchId !== undefined) {
        const pose = frame.getViewerPose(xrRefSpace);
        if (pose) {
            // 获取位置 (position) 和姿态 (orientation)
            data.teleop_mode = teleopMode;  // 'arm' 或 'base'
            data.position = {
                x: pose.transform.inverse.position.x,
                y: pose.transform.inverse.position.y,
                z: pose.transform.inverse.position.z,
            };
            data.orientation = {
                x: pose.transform.inverse.orientation.x,
                y: pose.transform.inverse.orientation.y,
                z: pose.transform.inverse.orientation.z,
                w: pose.transform.inverse.orientation.w,
            };
        }

        // 机械臂模式下发送夹爪控制
        if (teleopMode === 'arm') {
            data.gripper_delta = touchDeltaY;  // 范围 [-1, 1]
        }
    }

    socket.send(data);  // 发送到服务器
}
```

**数据说明**:
- `pose.transform.inverse`: 相机坐标系相对于参考坐标系的逆变换（即参考坐标系→相机坐标系）
- `position`: 单位米 (m)
- `orientation`: 四元数 (x, y, z, w)，WebXR 格式
- `touchDeltaY`: 夹爪控制量，由触摸滑动计算

### 2.4 触摸事件处理

手机屏幕分为两个区域：左侧 90% 控制机械臂，右侧 10% 控制底盘。

```javascript
canvas.addEventListener('touchstart', (event) => {
    if (touchId === undefined) {
        const touch = event.changedTouches[0];
        touchId = touch.identifier;
        touchStartY = touch.clientY;

        // 根据触摸位置决定控制模式
        teleopMode = touch.clientX < 0.9 * window.innerWidth ? 'arm' : 'base';

        handleTouch(touch);
    }
});

canvas.addEventListener('touchmove', (event) => {
    for (const touch of event.changedTouches) {
        if (touchId === touch.identifier) {
            handleTouch(touch);
        }
    }
});

function handleTouch(touch) {
    // 计算夹爪控制量 (向上滑动=打开，向下滑动=闭合)
    touchDeltaY = (touchStartY - touch.clientY) / (0.2 * window.innerHeight);
    touchDeltaY = Math.min(1, Math.max(-1, touchDeltaY));  // 限制范围
}
```

**设计考虑**:
- 单点触控避免误操作
- 滑动距离归一化到 [-1, 1]
- 屏幕分区让用户快速切换控制模式

## 3. Socket.IO 通信

### 3.1 前端连接

```javascript
// 生成随机设备 ID (支持多设备同时控制)
const deviceId = Math.random().toString(36).substring(2, 15);

// 连接到服务器
const socket = io();  // 自动连接到当前页面的主机

// 监听服务器回传的时间戳 (用于 RTT 计算)
socket.on('echo', (timestamp) => {
    const rtt = Date.now() - timestamp;
    document.getElementById('info').innerText = rttStats.calculate(rtt);
});
```

**连接流程**:
1. 手机浏览器访问 `http://<server-ip>:5000`
2. Socket.IO 自动协商连接方式 (WebSocket 或 HTTP 长轮询)
3. 连接成功后，客户端可以通过 `socket.send(data)` 发送消息

### 3.2 消息格式

发送到服务器的消息是 JSON 对象，包含以下字段：

```javascript
{
    "timestamp": 1733970123456,       // 毫秒时间戳
    "device_id": "abc123xyz",         // 设备唯一标识
    "teleop_mode": "arm",             // 可选，控制模式 ('arm' 或 'base')
    "position": {                     // 可选，手机位置 (米)
        "x": 0.123,
        "y": -0.045,
        "z": 0.678
    },
    "orientation": {                  // 可选，手机姿态 (四元数)
        "x": 0.0,
        "y": 0.707,
        "z": 0.0,
        "w": 0.707
    },
    "gripper_delta": 0.5,             // 可选，夹爪增量控制
    "state_update": "episode_started" // 可选，状态更新
}
```

**消息类型**:
1. **状态消息**: 只包含 `timestamp` 和 `state_update` (episode_started, episode_ended, reset_env)
2. **位姿消息**: 包含 `timestamp`, `device_id`, `teleop_mode`, `position`, `orientation`
3. **夹爪消息**: 机械臂模式下额外包含 `gripper_delta`

### 3.3 RTT 延迟监控

```javascript
class RTTStats {
    constructor(bufferSize) {
        this.bufferSize = bufferSize;
        this.bufferIndex = 0;
        this.rttArray = new Array(bufferSize).fill(0);
    }

    calculate(rtt) {
        this.rttArray[this.bufferIndex] = rtt;
        this.bufferIndex = (this.bufferIndex + 1) % this.bufferSize;

        const minRtt = Math.min(...this.rttArray);
        const avgRtt = this.rttArray.reduce((acc, cur) => acc + cur, 0) / this.bufferSize;
        const maxRtt = Math.max(...this.rttArray);
        const stdDevRtt = Math.sqrt(
            this.rttArray.map((x) => (x - avgRtt) ** 2)
                .reduce((acc, cur) => acc + cur, 0) / this.bufferSize
        );

        return `${minRtt.toFixed(3)}/${avgRtt.toFixed(3)}/${maxRtt.toFixed(3)}/${stdDevRtt.toFixed(3)} ms`;
    }
}

const rttStats = new RTTStats(100);  // 100 样本滚动统计
```

**延迟统计**:
- 最小/平均/最大/标准差
- 5GHz WiFi 下典型值: `5/7/12/2 ms`
- 如果平均 RTT >20ms，建议检查网络环境

## 4. Flask-SocketIO 服务器

### 4.1 WebServer 类设计

```python
class WebServer:
    def __init__(self, queue):
        self.app = Flask(__name__)
        self.socketio = SocketIO(self.app)
        self.queue = queue  # 用于将消息传递给主线程

        @self.app.route('/')
        def index():
            return render_template('index.html')

        @self.socketio.on('message')
        def handle_message(data):
            # 立即回传时间戳用于 RTT 计算
            emit('echo', data['timestamp'])

            # 将消息放入队列，由主线程处理
            self.queue.put(data)

        # 减少 Flask 日志输出
        logging.getLogger('werkzeug').setLevel(logging.WARNING)

    def run(self):
        # 获取本机 IP 地址
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.settimeout(0)
        try:
            s.connect(('8.8.8.8', 1))
            address = s.getsockname()[0]
        except Exception:
            address = '127.0.0.1'
        finally:
            s.close()

        print(f'Starting server at {address}:5000')
        self.socketio.run(self.app, host='0.0.0.0')  # 监听所有网络接口
```

**关键设计**:
- `Queue`: 线程安全的消息队列，解耦网络线程和控制线程
- `emit('echo', ...)`: 立即回传时间戳，最小化 RTT 测量误差
- `host='0.0.0.0'`: 允许局域网内其他设备连接

### 4.2 自动获取 IP 地址

服务器启动时会自动打印可访问的 URL：

```python
# 技巧：通过尝试连接外部 DNS 来获取本机 IP
s.connect(('8.8.8.8', 1))
address = s.getsockname()[0]
```

输出示例：
```
Starting server at 192.168.1.100:5000
```

然后在手机浏览器访问 `http://192.168.1.100:5000`。

### 4.3 消息队列处理

```python
class TeleopPolicy(Policy):
    def __init__(self):
        self.web_server_queue = Queue()
        self.teleop_controller = None

        # 启动 Web 服务器线程
        server = WebServer(self.web_server_queue)
        threading.Thread(target=server.run, daemon=True).start()

        # 启动消息监听线程
        threading.Thread(target=self.listener_loop, daemon=True).start()

    def listener_loop(self):
        while True:
            if not self.web_server_queue.empty():
                data = self.web_server_queue.get()

                # 更新状态
                if 'state_update' in data:
                    self.teleop_state = data['state_update']

                # 处理位姿消息 (过滤过时数据)
                elif 1000 * time.time() - data['timestamp'] < 250:  # 250 ms
                    self._process_message(data)

            time.sleep(0.001)  # 1 kHz 监听频率
```

**消息过滤**:
- 只处理 250ms 内的消息，丢弃过时数据
- 避免网络拥塞时积压的旧命令影响控制

## 5. 延迟优化

### 5.1 延迟来源分析

整个控制回路的延迟包括：

| 环节 | 典型延迟 | 说明 |
|------|----------|------|
| **IMU 采样** | ~16ms | 手机传感器更新率 ~60 Hz |
| **WebXR 处理** | ~5ms | 浏览器融合 IMU 数据 |
| **网络传输 (RTT)** | ~7ms | 5GHz WiFi |
| **服务器处理** | <1ms | Python 解析 + 队列 |
| **控制计算** | <1ms | 坐标转换 + IK |
| **MuJoCo 仿真** | 100ms | 10 Hz 控制周期 |
| **总延迟** | ~130ms | 从手机移动到机器人响应 |

### 5.2 优化策略

#### 减少 IMU 采样延迟

WebXR 的帧率受浏览器限制，通常 60 Hz。可以通过以下方式提高：

```javascript
// 不需要渲染复杂场景，最小化 GPU 负担
gl.clearColor(r, 0, b, 0.5);
gl.clear(gl.COLOR_BUFFER_BIT);  // 只清屏，不画其他内容
```

#### 网络优化

1. **使用 5GHz WiFi**: 相比 2.4GHz，延迟和带宽都更优
2. **消息压缩**: Socket.IO 自动启用 WebSocket 压缩
3. **减少消息大小**: 只发送必要字段

```python
# 原始消息：~150 字节
{
    "timestamp": 1733970123456,
    "device_id": "abc123xyz",
    "teleop_mode": "arm",
    "position": {"x": 0.123, "y": -0.045, "z": 0.678},
    "orientation": {"x": 0.0, "y": 0.707, "z": 0.0, "w": 0.707},
    "gripper_delta": 0.5
}

# 优化后可以用二进制格式进一步压缩 (未实现)
```

#### 过时数据丢弃

```python
elif 1000 * time.time() - data['timestamp'] < 250:  # 250 ms
    self._process_message(data)
```

如果消息在队列中等待超过 250ms，直接丢弃。这样可以避免：
- 网络拥塞时积压的旧命令
- 用户快速移动手机后的"拖尾"效应

### 5.3 延迟监控

#### 前端 RTT 显示

```javascript
socket.on('echo', (timestamp) => {
    const rtt = Date.now() - timestamp;
    document.getElementById('info').innerText = rttStats.calculate(rtt);
});
```

用户可以实时看到网络延迟：

```
5.234/7.123/12.456/1.987 ms
(min/avg/max/std)
```

#### 后端日志

在 `policies.py` 中取消注释以下行：

```python
@self.socketio.on('message')
def handle_message(data):
    emit('echo', data['timestamp'])
    print(f"[WebServer] Received: {data}")  # 取消注释
    self.queue.put(data)
```

会打印每条消息：
```
[WebServer] Received: {'timestamp': 1733970123456, 'device_id': 'abc123', 'teleop_mode': 'arm', ...}
```

## 6. 多设备支持

### 6.1 设备识别

每个手机在连接时生成唯一的 `device_id`：

```javascript
const deviceId = Math.random().toString(36).substring(2, 15);
```

示例: `"a3c9f2k8l1m"`

### 6.2 主次设备分配

系统支持两个手机同时控制：
- **主设备 (Primary)**: 控制机械臂或底盘
- **辅助设备 (Secondary)**: 同时控制底盘

```python
def process_message(self, data):
    device_id = data['device_id']

    # 更新启用计数
    self.enabled_counts[device_id] = (
        self.enabled_counts.get(device_id, 0) + 1
        if 'teleop_mode' in data else 0
    )

    # 分配主设备 (跳过前 2 条消息，因为 WebXR 初始姿态可能不稳定)
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

**设备分配逻辑**:
1. 第一个按住屏幕的手机成为主设备
2. 第二个按住屏幕的手机成为辅助设备（只能控制底盘）
3. 松开屏幕后，设备身份释放

## 7. 使用方法

### 7.1 启动服务器

```bash
# 激活环境
mamba activate bobacbot

# 启动遥操作模式
python main.py --sim --teleop --save --output-dir data/demos
```

程序会输出：
```
Starting server at 192.168.1.100:5000
```

### 7.2 手机连接

1. 确保手机和电脑在同一 WiFi 网络
2. 在手机浏览器（Chrome/Safari）访问 `http://192.168.1.100:5000`
3. 点击 **"Start episode"** 进入 AR 模式
4. 授权相机和运动传感器访问权限

### 7.3 控制操作

#### 开始 Episode
1. 点击 **"Start episode"**
2. 进入 AR 模式后，屏幕左上角显示 RTT 统计

#### 控制机械臂
1. 按住屏幕左侧 90% 区域
2. 移动手机 → 机械臂末端跟随移动
3. 向上滑动 → 夹爪打开
4. 向下滑动 → 夹爪闭合

#### 控制底盘
1. 按住屏幕右侧 10% 区域
2. 移动手机 → 底盘跟随移动
3. 旋转手机 → 底盘旋转

#### 结束 Episode
1. 点击 **"End episode"**
2. 输入 `y` 或 `n` 决定是否保存数据
3. 点击 **"Reset env"** 重置环境

## 8. 调试技巧

### 8.1 检查 WebXR 支持

在手机浏览器的开发者工具中（Chrome: `chrome://inspect`）：

```javascript
navigator.xr.isSessionSupported('immersive-ar').then((supported) => {
    console.log('WebXR AR supported:', supported);
});
```

如果返回 `false`，可能原因：
- 浏览器不支持 WebXR (更新 Chrome 到最新版本)
- 手机硬件不支持 ARCore/ARKit
- HTTPS 要求（localhost 除外）

### 8.2 查看实时位姿

在 `index.html` 的 `onXRFrame` 函数中添加：

```javascript
if (pose) {
    console.log('Position:', data.position);
    console.log('Orientation:', data.orientation);
}
```

### 8.3 网络抓包

使用 Wireshark 捕获 WebSocket 流量：

```
Filter: tcp.port == 5000
```

正常情况下，每帧发送一条 ~150 字节的消息，频率 ~60 Hz。

### 8.4 测试 Socket.IO 连接

独立测试服务器：

```python
from queue import Queue
from policies import WebServer

queue = Queue()
server = WebServer(queue)
server.run()
```

然后在浏览器访问 `http://localhost:5000`，打开控制台应该看到 Socket.IO 连接成功。

## 9. 常见问题

### 9.1 手机连不上服务器

**症状**: 浏览器显示"无法访问此网站"

**排查**:
1. 检查手机和电脑是否在同一网络
   ```bash
   # 电脑上查看 IP
   ifconfig  # Linux/macOS
   ipconfig  # Windows
   ```
2. 检查防火墙是否允许 5000 端口
   ```bash
   # Ubuntu
   sudo ufw allow 5000
   ```
3. 确认服务器正在运行
   ```bash
   ps aux | grep python
   ```

### 9.2 RTT 延迟过高

**症状**: RTT 平均值 >20ms

**原因**:
- 使用 2.4GHz WiFi (干扰严重)
- 距离路由器太远
- 网络拥塞

**解决**:
1. 切换到 5GHz WiFi
2. 靠近路由器
3. 减少同网络设备数量

### 9.3 IMU 数据漂移

**症状**: 手机静止时，机器人仍在缓慢移动

**原因**:
- WebXR 的 `local` 坐标系会累积误差
- 手机传感器校准不准确

**解决**:
1. 重新启动 AR 会话（点击 "Reset env"）
2. 在平坦表面上校准手机陀螺仪（手机设置）
3. 使用参考位姿相对控制（我们已经实现）

### 9.4 控制不流畅/卡顿

**症状**: 机械臂移动有明显停顿

**原因**:
- WebXR 帧率下降（<60 Hz）
- 消息队列积压
- 控制周期不匹配

**解决**:
1. 关闭其他占用 GPU 的应用
2. 检查 `listener_loop` 是否及时处理消息
3. 检查控制周期是否是 100ms (10 Hz)

### 9.5 双设备控制冲突

**症状**: 两个手机同时控制时机械臂抖动

**原因**: 主设备先启用

**解决**: 确保主设备（控制机械臂）先按住屏幕，辅助设备后按住

## 10. 性能分析

### 10.1 消息频率

在 `listener_loop` 中统计消息频率：

```python
last_print_time = time.time()
message_count = 0

def listener_loop(self):
    while True:
        if not self.web_server_queue.empty():
            data = self.web_server_queue.get()
            message_count += 1

            # 每秒打印一次统计
            if time.time() - last_print_time > 1.0:
                print(f"Message rate: {message_count} Hz")
                message_count = 0
                last_print_time = time.time()

            # ... (原有处理逻辑)
```

正常情况下应该看到 ~60 Hz。

### 10.2 队列积压

```python
if not self.web_server_queue.empty():
    queue_size = self.web_server_queue.qsize()
    if queue_size > 10:
        print(f"Warning: Queue backlog = {queue_size}")
```

如果队列持续积压，说明处理速度跟不上接收速度。

## 11. 安全考虑

### 11.1 局域网隔离

当前实现只监听局域网，不暴露到公网：

```python
self.socketio.run(self.app, host='0.0.0.0')  # 只绑定局域网
```

**不要**将 5000 端口映射到公网，除非添加认证机制。

### 11.2 HTTPS 要求

WebXR 需要 HTTPS 或 localhost：
- 开发时使用 `localhost` 或局域网 IP (HTTP 可用)
- 生产环境需要配置 SSL 证书

### 11.3 输入验证

目前消息格式是 JSON，理论上可以注入恶意数据。生产环境建议添加：

```python
@self.socketio.on('message')
def handle_message(data):
    # 验证消息格式
    if 'timestamp' not in data or 'device_id' not in data:
        return  # 忽略无效消息

    if 'position' in data:
        # 检查位置范围
        pos = data['position']
        if abs(pos['x']) > 10 or abs(pos['y']) > 10 or abs(pos['z']) > 10:
            return  # 忽略异常位置

    emit('echo', data['timestamp'])
    self.queue.put(data)
```

## 12. 下一步

手机 IMU 通信是遥操作的核心，但它只是数据采集系统的一部分。下一篇文档将介绍仿真环境的整体通信架构，包括多进程间的数据同步。
