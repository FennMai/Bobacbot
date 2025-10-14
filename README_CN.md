# Bobacbot - 基于 Diffusion Policy 的全向移动操作机器人

本项目基于 [TidyBot++](https://github.com/jimmyyhwu/tidybot2) 开发，使用 [Diffusion Policy](https://github.com/real-stanford/diffusion_policy) 进行模仿学习训练。

## 项目概述

### 机器人配置
- **底盘**: Bobac3 全向移动底盘（3自由度：x, y, yaw）
- **机械臂**: Elephant Robotics eco65b 6自由度机械臂
- **末端执行器**: Robotiq 2F-85 夹爪
- **传感器**:
  - 底盘前置深度相机（base camera）
  - 手腕相机（wrist camera）

### 主要特性
- 完整的 MuJoCo 仿真环境
- 支持手机 WebXR 遥操作和键盘控制
- 基于 Diffusion Policy 的模仿学习
- 实时图像观测和动作控制
- 数据采集和回放功能

## 环境安装

### 1. 安装 conda/Conda

推荐使用 [conda](https://conda.readthedocs.io/en/latest/installation/conda-installation.html)（Miniforge 版本）以获得更快的依赖解析速度。

```bash
# 也可以使用 conda 替代 conda
curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
bash Miniforge3-$(uname)-$(uname -m).sh
```

### 2. 创建 Python 环境

```bash
# 克隆仓库
git clone <repository_url>
cd Bobacbot

# 创建并激活环境
conda create -n bobacbot python=3.10.14
conda activate bobacbot

# 安装依赖
pip install -r requirements.txt
```

### 3. 设置 Diffusion Policy 训练环境

训练需要单独的环境配置：

```bash
# 进入训练目录
cd training/diffusion_policy

# 创建训练环境
conda env create -f conda_environment.yaml

# 激活训练环境
conda activate robodiff
```

## 快速开始

### 1. 启动仿真环境

#### 方式一：仅查看仿真（不保存数据）
```bash
python main.py --sim
```

#### 方式二：使用键盘控制（推荐用于数据采集）
```bash
# 启动键盘控制模式并保存数据
python main.py --sim --keyboard --save --output-dir data/demos

# 启动后：
# - 按空格键开始新 episode
# - 使用键盘/手机控制机器人
# - 再次按空格键结束 episode
# - 输入 y/n 决定是否保存
```

#### 方式三：使用手机遥操作
```bash
# 1. 启动遥操作模式
python main.py --sim --teleop --save --output-dir data/demos

# 2. 在手机浏览器打开终端显示的 URL (例如：http://192.168.1.100:5000)
# 3. 按照手机界面提示进行操作
```

### 2. 机器人位置随机化

训练时可以启用机器人初始位置随机化以提高策略泛化能力：

```bash
# 启用随机位置
python main.py --sim --keyboard --save --random --output-dir data/demos
```

### 3. 数据回放和验证

采集数据后，可以在仿真中回放验证：

```bash
# 回放所有 episode
python replay_episodes.py --sim --input-dir data/demos

# 同时显示相机图像
python replay_episodes.py --sim --input-dir data/demos --show-images

# 执行观测动作（测试数据准确性）
python replay_episodes.py --sim --input-dir data/demos --execute-obs
```

## 数据采集工作流

### 数据格式

每个 episode 保存为独立文件夹，包含：
- `data.pkl`: 观测和动作序列（pickle 格式）
- `base_image.mp4`: 底盘相机视频
- `wrist_image.mp4`: 手腕相机视频

### 采集步骤

```bash
# 1. 启动数据采集
python main.py --sim --keyboard --save --output-dir data/demos

# 2. 采集多个 episode
# - 每个 episode 控制机器人完成任务
# - 确保动作流畅、任务成功

# 3. 验证数据质量
python replay_episodes.py --sim --input-dir data/demos

# 4. 转换为训练格式
python convert_to_robomimic_hdf5.py --input-dir data/demos --output-path data/demos.hdf5
```

## 模型训练

### 1. 准备训练数据

```bash
# 激活主环境
conda activate bobacbot

# 转换数据格式
python convert_to_robomimic_hdf5.py \
    --input-dir data/demos \
    --output-path data/demos.hdf5

# 将 HDF5 文件复制到训练环境
cp data/demos.hdf5 training/diffusion_policy/data/
```

### 2. 配置训练任务

```bash
cd training/diffusion_policy

# 应用补丁（如果尚未应用）
git checkout 548a52b
git apply ../../diffusion-policy.patch

# 编辑任务配置
# 打开 diffusion_policy/config/task/square_image_abs.yaml
# 修改 name 字段为你的任务名，例如：demos
```

### 3. 开始训练

```bash
# 激活训练环境
conda activate robodiff

# 启动训练
python train.py --config-name=train_diffusion_unet_real_hybrid_workspace task=square_image_abs

# 训练输出保存在：data/outputs/
```

### 4. 监控训练

```bash
# 使用 tensorboard 查看训练曲线
tensorboard --logdir data/outputs
```

## 策略推理（Policy Rollout）

### 1. 启动策略服务器

在 GPU 机器上：

```bash
cd training/diffusion_policy
conda activate robodiff

# 启动策略服务器（指定 checkpoint 路径）
python policy_server.py --ckpt-path data/outputs/<your_run>/checkpoints/epoch=<N>-train_loss=<X>.ckpt
```

### 2. 运行策略评估

在开发机器上（可以是同一台机器）：

```bash
# 1. 创建 SSH 隧道连接策略服务器（如果在不同机器）
ssh -L 5555:localhost:5555 <gpu-machine>

# 2. 启动仿真并运行策略
conda activate bobacbot
python main.py --sim

# 策略会自动连接到 localhost:5555 的服务器
# 使用手机作为使能设备（按住屏幕执行策略）
```

## 项目结构

```
Bobacbot/
├── main.py                      # 主程序入口
├── mujoco_env.py               # MuJoCo 仿真环境
├── policies.py                 # 策略类（遥操作、远程策略等）
├── keyboard_policy.py          # 键盘控制策略
├── episode_storage.py          # Episode 数据存储
├── convert_to_robomimic_hdf5.py # 数据格式转换
├── replay_episodes.py          # 数据回放工具
├── ik_solver.py               # 逆运动学求解器
│
├── models/                     # 机器人模型文件
│   ├── bobacbot_v2.xml        # Bobac3 + eco65b 机器人模型
│   ├── bobacbot_demo_scene.xml # 演示场景
│   ├── bobac3/                # Bobac3 底盘 mesh
│   ├── eco65b/                # eco65b 机械臂 mesh
│   └── 2f85/                  # Robotiq 夹爪 mesh
│
├── data/                       # 数据目录
│   ├── demos/                 # 采集的演示数据
│   └── output/               # 其他输出
│
├── training/                   # 训练相关
│   └── diffusion_policy/      # Diffusion Policy 代码库
│
├── constants.py               # 常量配置
├── requirements.txt           # Python 依赖
└── README_CN.md              # 本文档
```

## 关键参数说明

### main.py 参数

- `--sim`: 使用仿真环境（必须）
- `--teleop`: 启用手机遥操作模式
- `--keyboard`: 启用键盘控制模式
- `--save`: 保存 episode 数据
- `--random`: 随机化机器人初始位置
- `--output-dir`: 数据保存目录（默认：`data/0714`）

### 机器人控制

- **底盘**: 3自由度全向移动（x, y, yaw）
- **机械臂**: 6自由度位姿控制（position + quaternion）
- **夹爪**: 0-1 归一化控制（0=闭合，1=打开）
- **控制频率**: 10 Hz（100ms 周期）

### 观测空间

```python
obs = {
    'base_pose': (3,),           # 底盘位姿 [x, y, theta]
    'arm_pos': (3,),             # 机械臂位置 [x, y, z]
    'arm_quat': (4,),            # 机械臂四元数 [x, y, z, w]
    'gripper_pos': (1,),         # 夹爪位置 [0-1]
    'base_image': (360, 640, 3), # 底盘相机图像
    'wrist_image': (480, 640, 3) # 手腕相机图像
}
```

### 动作空间

```python
action = {
    'base_pose': (3,),      # 目标底盘位姿
    'arm_pos': (3,),        # 目标机械臂位置
    'arm_quat': (4,),       # 目标机械臂四元数
    'gripper_pos': (1,),    # 目标夹爪位置
}
```

## 常见问题

### Q1: 仿真运行缓慢？
A: 尝试缩小 MuJoCo viewer 窗口大小，这会加速离屏渲染。

### Q2: 机械臂重置位置不正确？
A: 检查 `mujoco_env.py` 中 `ArmController.reset()` 的 `qpos` 初始值是否与 eco65b 的 retract keyframe 匹配。

### Q3: 训练时 OOM（内存不足）？
A: 减小 batch size 或降低图像分辨率（在 `constants.py` 修改 `POLICY_IMAGE_WIDTH/HEIGHT`）。

### Q4: 策略推理时无响应？
A: 确认策略服务器已启动，且 SSH 隧道连接正常（`ssh -L 5555:localhost:5555`）。

### Q5: 手机 WebXR 无法连接？
A: 确保手机和电脑在同一网络，浏览器支持 WebXR（推荐 Chrome），使用 HTTPS 或 localhost。

## 参考文献

### TidyBot++
```
@inproceedings{wu2024tidybot,
  title = {TidyBot++: An Open-Source Holonomic Mobile Manipulator for Robot Learning},
  author = {Wu, Jimmy and Chong, William and Holmberg, Robert and Prasad, Aaditya and Gao, Yihuai and Khatib, Oussama and Song, Shuran and Rusinkiewicz, Szymon and Bohg, Jeannette},
  booktitle = {Conference on Robot Learning (CoRL)},
  year = {2024}
}
```

- 项目主页: http://tidybot2.github.io
- 代码仓库: https://github.com/jimmyyhwu/tidybot2
- 论文: https://arxiv.org/abs/2412.10447

### Diffusion Policy
```
@inproceedings{chi2023diffusionpolicy,
  title = {Diffusion Policy: Visuomotor Policy Learning via Action Diffusion},
  author = {Chi, Cheng and Feng, Siyuan and Du, Yilun and Xu, Zhenjia and Cousineau, Eric and Burchfiel, Benjamin and Song, Shuran},
  booktitle = {Robotics: Science and Systems (RSS)},
  year = {2023}
}
```

- 项目主页: https://diffusion-policy.cs.columbia.edu/
- 代码仓库: https://github.com/real-stanford/diffusion_policy

## 致谢

本项目基于以下开源项目：
- [TidyBot++](https://github.com/jimmyyhwu/tidybot2): 全向移动操作机器人平台
- [Diffusion Policy](https://github.com/real-stanford/diffusion_policy): 扩散策略模仿学习
- [MuJoCo](https://mujoco.org/): 物理仿真引擎
- [Ruckig](https://github.com/pantor/ruckig): 实时轨迹生成

## 许可证

本项目继承 TidyBot++ 的许可证。详见 [LICENSE](LICENSE) 文件。
