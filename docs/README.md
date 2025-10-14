# Bobacbot 技术文档索引

本目录包含 Bobacbot 项目的完整技术文档，涵盖从仿真环境搭建到模型训练的全流程。

## 文档概览

本项目基于 [TidyBot++](https://github.com/jimmyyhwu/tidybot2) 开发，使用 [Diffusion Policy](https://github.com/real-stanford/diffusion_policy) 进行模仿学习。文档按照实际开发和使用流程组织，适合不同阶段的开发者阅读。

## 文档列表

### 一、环境搭建与仿真

#### [01. MuJoCo 仿真环境搭建](01_mujoco_simulation_setup.md)
详细介绍 MuJoCo 3.2.4 仿真环境的架构设计，包括：
- 机器人模型结构（Bobac3 底盘 + eco65b 机械臂 + Robotiq 夹爪）
- 多进程架构设计（物理仿真、渲染、控制分离）
- 共享内存通信机制
- 场景配置和传感器设置

**适合人群**: 需要理解仿真架构或修改机器人模型的开发者

#### [02. 机械臂仿真运动控制](02_arm_control.md)
深入讲解 6-DOF 机械臂的运动控制实现：
- 逆运动学求解器（Damped Least Squares + 零空间优化）
- Ruckig 实时轨迹生成
- 位姿控制与关节控制
- PID 控制器调优

**适合人群**: 需要理解或改进机械臂控制算法的开发者

### 二、遥操作与数据采集

#### [03. 手机遥操作与机械臂物理映射](03_phone_teleop_mapping.md)
详细说明 WebXR 手机控制到机器人动作的映射关系：
- WebXR 坐标系与机器人坐标系转换
- 位姿数据的物理映射
- 相机偏移校准（iPhone 14 Pro 参数）
- 遥操作模式切换

**适合人群**: 需要调试遥操作体验或适配新设备的开发者

#### [04. 手机 IMU 通信调用](04_imu_communication.md)
讲解 WebXR IMU 数据获取和实时通信：
- WebXR Device API 使用
- Flask-SocketIO 双向通信
- 网络延迟优化（RTT 监控）
- HTTPS 部署和调试

**适合人群**: 需要优化通信性能或排查延迟问题的开发者

#### [05. 仿真通信架构](05_simulation_communication.md)
全面解析多进程通信机制：
- 共享内存布局（ShmState 和 ShmImage）
- 消息队列设计
- 同步与并发控制
- 性能分析（<1μs 延迟）

**适合人群**: 需要深入理解系统架构或优化性能的开发者

#### [06. 遥操作数据采集](06_teleoperation_data_collection.md)
介绍完整的数据采集工作流：
- Episode 数据格式（pickle + MP4）
- EpisodeWriter 异步保存机制
- 数据采集最佳实践
- 数据质量验证

**适合人群**: 需要采集高质量训练数据的研究人员和操作员

### 三、策略训练与部署

#### [07. 策略部署与推理](07_policy_deployment.md)
详细说明 Diffusion Policy 的部署架构：
- Client-Server 架构设计
- ZeroMQ 通信协议
- PolicyWrapper 推理队列管理
- 时序对齐和延迟补偿

**适合人群**: 需要部署训练好的模型进行推理的工程师

#### [08. 模型训练](08_model_training.md)
完整的 Diffusion Policy 训练流程：
- 数据格式转换（Episode → HDF5）
- Hydra 配置文件详解
- 训练参数调优
- 常见训练问题排查

**适合人群**: 需要训练或调优 Diffusion Policy 模型的研究人员

### 四、数据分析与工具

#### [09. 数据可视化](09_data_visualization.md)
提供丰富的数据可视化工具和方法：
- Episode 回放工具使用
- 轨迹可视化（Matplotlib）
- 图像序列查看（OpenCV）
- 数据统计分析

**适合人群**: 需要分析数据质量或调试策略的研究人员

#### [10. 数据格式解析](10_data_format.md)
详细解析项目中使用的所有数据格式：
- Episode 格式规范
- HDF5 格式结构
- 坐标系统和单位约定
- 数据转换和验证

**适合人群**: 需要深入理解数据格式或开发自定义工具的开发者

## 推荐阅读路径

### 🚀 快速上手
1. 先阅读项目根目录的 [README_CN.md](../README_CN.md) 了解基本使用
2. [01. MuJoCo 仿真环境搭建](01_mujoco_simulation_setup.md) - 理解系统架构
3. [06. 遥操作数据采集](06_teleoperation_data_collection.md) - 开始采集数据
4. [09. 数据可视化](09_data_visualization.md) - 验证数据质量

### 🔧 系统开发
1. [01. MuJoCo 仿真环境搭建](01_mujoco_simulation_setup.md) - 架构设计
2. [05. 仿真通信架构](05_simulation_communication.md) - 通信机制
3. [02. 机械臂仿真运动控制](02_arm_control.md) - 控制算法
4. [03. 手机遥操作与机械臂物理映射](03_phone_teleop_mapping.md) - 坐标映射
5. [04. 手机 IMU 通信调用](04_imu_communication.md) - 网络通信

### 🤖 模型训练
1. [06. 遥操作数据采集](06_teleoperation_data_collection.md) - 数据采集
2. [10. 数据格式解析](10_data_format.md) - 理解数据结构
3. [09. 数据可视化](09_data_visualization.md) - 数据分析
4. [08. 模型训练](08_model_training.md) - 训练流程
5. [07. 策略部署与推理](07_policy_deployment.md) - 部署评估

### 🐛 问题排查（运维）
1. [04. 手机 IMU 通信调用](04_imu_communication.md) - 网络延迟问题
2. [05. 仿真通信架构](05_simulation_communication.md) - 进程通信问题
3. [07. 策略部署与推理](07_policy_deployment.md) - 推理性能问题
4. [08. 模型训练](08_model_training.md) - 训练问题排查

## 技术栈

本项目涉及的主要技术：

| 技术 | 版本 | 用途 |
|------|------|------|
| **MuJoCo** | 3.2.4 | 物理仿真 |
| **Diffusion Policy** | - | 模仿学习 |
| **Python** | 3.10.14 | 主要开发语言 |
| **PyTorch** | - | 模型训练 |
| **WebXR** | - | AR 遥操作 |
| **Flask-SocketIO** | - | 实时通信 |
| **ZeroMQ** | - | 策略服务器 |
| **Ruckig** | - | 轨迹生成 |
| **OpenCV** | - | 图像处理 |
| **HDF5** | - | 训练数据存储 |

## 机器人配置

- **底盘**: Bobac3 全向移动底盘（3-DOF: x, y, yaw）
- **机械臂**: Elephant Robotics eco65b（6-DOF）
- **末端执行器**: Robotiq 2F-85 夹爪
- **传感器**:
  - 底盘前置深度相机（base camera）
  - 手腕相机（wrist camera）
- **控制频率**: 10 Hz
- **仿真步长**: 0.001s (1000 Hz)

## 数据流概览

```
[手机 WebXR] → [Socket.IO] → [TeleopPolicy] → [MujocoEnv] → [数据采集]
     ↓                            ↓                  ↓              ↓
  IMU数据                      位姿映射          仿真执行      Episode存储
                                                                    ↓
                                                              [转换为HDF5]
                                                                    ↓
                                                            [Diffusion Policy]
                                                                    ↓
[真实机器人] ← [RemotePolicy] ← [ZMQ通信] ← [PolicyServer] ← [训练checkpoint]
```

## 常用命令速查

```bash
# 启动仿真（遥操作模式）
python main.py --sim --teleop --save --output-dir data/demos

# 启动仿真（键盘模式）
python main.py --sim --keyboard --save --output-dir data/demos

# 回放 episode
python replay_episodes.py --sim --input-dir data/demos

# 转换为训练格式
python convert_to_robomimic_hdf5.py --input-dir data/demos --output-path data/demos.hdf5

# 训练模型
cd training/diffusion_policy
conda activate robodiff
python train.py --config-name=train_diffusion_unet_real_hybrid_workspace task=square_image_abs

# 启动策略服务器
cd training/diffusion_policy
python policy_server.py --ckpt-path data/outputs/.../checkpoints/latest.ckpt

# 运行策略推理
python main.py --sim
```

## 获取帮助

- **项目主页**: https://github.com/jimmyyhwu/tidybot2
- **Diffusion Policy**: https://diffusion-policy.cs.columbia.edu/
- **问题反馈**: 请在项目 GitHub 提交 Issue

## 文档维护

本文档最后更新：2025-10-14

如发现文档有误或需要补充，请提交 Pull Request 或联系项目维护者。

---

**注意**: 阅读文档时，建议结合项目代码一起理解。所有文档中提到的代码片段都可以在项目源码中找到对应实现。
