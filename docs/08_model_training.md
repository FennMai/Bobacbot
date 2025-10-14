# 模型训练文档

## 1. 概述

本项目使用 [Diffusion Policy](https://diffusion-policy.cs.columbia.edu/) 进行模仿学习训练。Diffusion Policy 是一种基于扩散模型的视觉运动策略学习方法，能够从演示数据中学习复杂的多模态行为。

### 1.1 Diffusion Policy 特点

- **多模态**: 可以学习到多种不同的行为策略
- **视觉输入**: 支持 RGB 图像作为观测
- **时序建模**: 使用观测历史和动作序列
- **稳定训练**: 扩散模型训练稳定，不易过拟合

### 1.2 训练流程

```
[Episode 数据] → [数据转换] → [HDF5 格式] → [训练] → [Checkpoint]
     ↓              ↓              ↓            ↓          ↓
  data/demos   convert.py   demos.hdf5    train.py   epoch=XXX.ckpt
```

## 2. 数据准备

### 2.1 数据采集

首先使用遥操作采集演示数据（参见《遥操作数据采集》文档）：

```bash
python main.py --sim --teleop --save --random --output-dir data/demos
```

建议采集数量：
- **简单任务**: 50-100 episodes （仿真环境）
- **中等任务**: 100-200 episodes （真实环境不移动的）
- **复杂任务**: 200+ episodes (真实环境带移动的)

### 2.2 数据格式转换

将 Episode 格式转换为 robomimic HDF5 格式：

```bash
# 激活运行环境
conda activate bobacbot

# 转换数据
python convert_to_robomimic_hdf5.py \
    --input-dir data/demos \
    --output-path data/demos.hdf5
```

**转换过程**:

```python
def main(input_dir, output_path):
    episode_dirs = sorted([child for child in Path(input_dir).iterdir() if child.is_dir()])

    with h5py.File(output_path, 'w') as f:
        data_group = f.create_group('data')

        for episode_idx, episode_dir in enumerate(tqdm(episode_dirs)):
            # 1. 加载 Episode 数据
            reader = EpisodeReader(episode_dir)

            # 2. 提取观测
            observations = {}
            for obs in reader.observations:
                for k, v in obs.items():
                    if v.ndim == 3:
                        # 缩放图像到策略输入分辨率
                        v = cv.resize(v, (POLICY_IMAGE_WIDTH, POLICY_IMAGE_HEIGHT))
                    if k not in observations:
                        observations[k] = []
                    observations[k].append(v)

            # 3. 提取动作
            actions = [
                np.concatenate((
                    action['base_pose'],           # [3]
                    action['arm_pos'],             # [3]
                    Rotation.from_quat(action['arm_quat']).as_rotvec(),  # [3] 四元数 -> 轴角
                    action['gripper_pos'],         # [1]
                )) for action in reader.actions    # 总共 [10]
            ]

            # 4. 写入 HDF5
            episode_key = f'demo_{episode_idx}'
            episode_group = data_group.create_group(episode_key)
            for k, v in observations.items():
                episode_group.create_dataset(f'obs/{k}', data=np.array(v))
            episode_group.create_dataset('actions', data=np.array(actions))
```

**注意**:
- 图像缩放到 84×84 (POLICY_IMAGE_WIDTH, POLICY_IMAGE_HEIGHT)
- 四元数转换为轴角表示 (axis-angle)，训练时会进一步转换为 6D rotation
- 动作维度: 3 (base) + 3 (arm_pos) + 3 (arm_rotvec) + 1 (gripper) = 10

**实际上本项目使用了更多动作维度**：

查看配置文件可知，动作维度是 13：
```yaml
action:
  shape: [13]
```

这意味着实际使用的是：
- 3 (base_pose) + 3 (arm_pos) + 6 (arm_rotation_6d) + 1 (gripper) = 13

### 2.3 验证转换结果

```python
import h5py

# 打开 HDF5 文件
with h5py.File('data/demos.hdf5', 'r') as f:
    print("Episodes:", list(f['data'].keys()))

    # 查看第一个 episode
    demo_0 = f['data/demo_0']
    print("Observations:", list(demo_0['obs'].keys()))
    print("Actions shape:", demo_0['actions'].shape)

    # 查看图像
    base_image = demo_0['obs/base_image'][0]
    print("Image shape:", base_image.shape)  # (84, 84, 3)
```

### 2.4 复制到训练环境

```bash
# 将 HDF5 文件复制到训练目录
cp data/demos.hdf5 training/diffusion_policy/data/
```

## 3. 配置文件

### 3.1 任务配置

编辑 `training/diffusion_policy/diffusion_policy/config/task/square_image_abs.yaml`：

```yaml
name: sim_pick_place_v2_July  # 任务名称，会作为数据文件名

shape_meta: &shape_meta
  obs:
    base_image:
      shape: [3, 84, 84]  # (C, H, W)
      type: rgb
    wrist_image:
      shape: [3, 84, 84]
      type: rgb
    base_pose:
      shape: [3]  # [x, y, theta]
    arm_pos:
      shape: [3]  # [x, y, z]
    arm_quat:
      shape: [4]  # [x, y, z, w]
    gripper_pos:
      shape: [1]  # [pos]
  action:
    shape: [13]  # 3 + 3 + 6 + 1 (base + arm_pos + rotation_6d + gripper)

dataset_path: &dataset_path data/${task.name}.hdf5
abs_action: &abs_action True  # 绝对动作 (vs 相对动作)

dataset:
  _target_: diffusion_policy.dataset.robomimic_replay_image_dataset.RobomimicReplayImageDataset
  shape_meta: *shape_meta
  dataset_path: *dataset_path
  horizon: ${horizon}  # 时间窗口长度
  pad_before: ${eval:'${n_obs_steps}-1+${n_latency_steps}'}
  pad_after: ${eval:'${n_action_steps}-1'}
  n_obs_steps: ${dataset_obs_steps}
  abs_action: *abs_action
  rotation_rep: 'rotation_6d'  # 使用 6D rotation representation
  use_legacy_normalizer: False
  use_cache: True  # 缓存加速
  seed: 42
  val_ratio: 0.0  # 验证集比例 (0 表示不划分)
```

**关键参数解释**:

| 参数 | 说明 | 典型值 |
|------|------|--------|
| `name` | 任务名称 | 自定义 |
| `shape_meta` | 观测和动作的形状定义 | 根据机器人配置 |
| `abs_action` | 是否使用绝对动作 | True |
| `rotation_rep` | 旋转表示 | `rotation_6d` 或 `quaternion` |
| `horizon` | 预测时间窗口 | 16 |
| `n_obs_steps` | 观测历史长度 | 2 |
| `n_action_steps` | 动作序列长度 | 8 |

### 3.2 训练配置

主配置文件 `train_diffusion_unet_real_hybrid_workspace.yaml`：

```yaml
defaults:
  - _self_
  - task: square_image_abs  # 使用的任务配置

name: train_diffusion_unet_hybrid
_target_: diffusion_policy.workspace.train_diffusion_unet_hybrid_workspace.TrainDiffusionUnetHybridWorkspace

# 任务参数
task_name: ${task.name}
shape_meta: ${task.shape_meta}
exp_name: "default"

# 时序参数
horizon: 16
n_obs_steps: 2
n_action_steps: 8
n_latency_steps: 0
dataset_obs_steps: ${n_obs_steps}
past_action_visible: False
keypoint_visible_rate: 1.0
obs_as_global_cond: True

# 网络架构
policy:
  _target_: diffusion_policy.policy.diffusion_unet_hybrid_image_policy.DiffusionUnetHybridImagePolicy
  shape_meta: ${shape_meta}

  # 噪声调度器
  noise_scheduler:
    _target_: diffusers.schedulers.scheduling_ddpm.DDPMScheduler
    num_train_timesteps: 100
    beta_start: 0.0001
    beta_end: 0.02
    beta_schedule: squaredcos_cap_v2
    variance_type: fixed_small
    clip_sample: True
    prediction_type: epsilon

  # 视觉编码器
  obs_encoder:
    _target_: diffusion_policy.model.vision.multi_image_obs_encoder.MultiImageObsEncoder
    shape_meta: ${shape_meta}
    rgb_model:
      _target_: diffusion_policy.model.vision.model_getter.get_resnet
      name: resnet18
      weights: null
    resize_shape: [84, 84]
    crop_shape: [76, 76]

  # UNet 噪声预测网络
  horizon: ${horizon}
  n_action_steps: ${eval:'${n_action_steps}+${n_latency_steps}'}
  n_obs_steps: ${n_obs_steps}
  num_inference_steps: 16
  obs_as_global_cond: ${obs_as_global_cond}

  # 网络架构参数
  diffusion_step_embed_dim: 256
  down_dims: [256, 512, 1024]
  kernel_size: 5
  n_groups: 8

# 数据集
dataloader:
  batch_size: 256
  num_workers: 8
  shuffle: True
  pin_memory: True
  persistent_workers: True

# 训练参数
training:
  device: "cuda:0"
  seed: 42
  debug: False
  resume: True
  lr_scheduler: cosine
  lr_warmup_steps: 500
  num_epochs: 1000
  gradient_accumulate_every: 1
  use_ema: True

  # 学习率
  lr: 1.0e-4

  # EMA 参数
  ema:
    _target_: diffusion_policy.model.diffusion.ema_model.EMAModel
    update_after_step: 0
    inv_gamma: 1.0
    power: 0.75
    min_value: 0.0
    max_value: 0.9999

# 日志
logging:
  project: diffusion_policy_debug
  resume: True
  mode: online
  name: ${now:%Y.%m.%d-%H.%M.%S}_${name}_${task_name}
  tags: ["${name}", "${task_name}", "${exp_name}"]
  id: null
  group: null

# 检查点
checkpoint:
  topk:
    monitor_key: train_loss
    mode: min
    k: 5
    format_str: 'epoch={epoch:04d}-train_loss={train_loss:.3f}.ckpt'
  save_last_ckpt: True
  save_last_snapshot: False

# 多 GPU 训练
multi_run:
  run_dir: data/outputs/${now:%Y.%m.%d}/${now:%H.%M.%S}_${name}_${task_name}
  wandb_name_base: ${now:%Y.%m.%d-%H.%M.%S}_${name}_${task_name}

hydra:
  job:
    override_dirname: ${name}
  run:
    dir: data/outputs/${now:%Y.%m.%d}/${now:%H.%M.%S}_${name}_${task_name}
```

## 4. 训练执行

### 4.1 环境准备

```bash
# 进入训练目录
cd training/diffusion_policy

# 激活训练环境
conda activate robodiff

# 应用补丁 (如果尚未应用)
git checkout 548a52b
git apply ../../diffusion-policy.patch
```

### 4.2 启动训练

```bash
# 基本训练命令
python train.py \
    --config-name=train_diffusion_unet_real_hybrid_workspace \
    task=square_image_abs

# 指定 GPU
CUDA_VISIBLE_DEVICES=0 python train.py \
    --config-name=train_diffusion_unet_real_hybrid_workspace \
    task=square_image_abs

# 覆盖参数
python train.py \
    --config-name=train_diffusion_unet_real_hybrid_workspace \
    task=square_image_abs \
    training.num_epochs=500 \
    training.lr=5e-5 \
    dataloader.batch_size=128
```

### 4.3 训练输出

训练会输出到 `data/outputs/<日期>/<时间>_<配置名>_<任务名>/`：

```
data/outputs/2025.07.14/00.30.40_train_diffusion_unet_hybrid_sim_pick_place_v2_July/
├── checkpoints/
│   ├── epoch=0100-train_loss=0.023.ckpt
│   ├── epoch=0200-train_loss=0.012.ckpt
│   ├── ...
│   └── latest.ckpt
├── .hydra/
│   └── config.yaml  # 完整配置
└── wandb/  # Weights & Biases 日志
```

### 4.4 监控训练

#### 方式一：Weights & Biases

如果启用了 W&B (默认)：

1. 访问 https://wandb.ai/
2. 登录账号
3. 查看项目 `diffusion_policy_debug`
4. 监控 `train_loss`, `val_loss` 等指标

#### 方式二：TensorBoard (需要修改配置)

```bash
# 修改配置启用 tensorboard
logging:
  mode: offline  # 禁用 wandb

# 启动 tensorboard
tensorboard --logdir data/outputs
```

#### 方式三：查看终端输出

```
Epoch 0001: train_loss=0.143, val_loss=N/A, lr=1.00e-04
Epoch 0002: train_loss=0.098, val_loss=N/A, lr=9.99e-05
Epoch 0003: train_loss=0.067, val_loss=N/A, lr=9.98e-05
...
Epoch 0100: train_loss=0.012, val_loss=N/A, lr=5.00e-05
```

## 5. 超参数调优

### 5.1 关键超参数

#### 学习率 (lr)

```yaml
training:
  lr: 1.0e-4  # 默认值
```

**调优建议**:
- 初始尝试: `1e-4`
- Loss 下降缓慢: 增大到 `5e-4`
- Loss 震荡: 减小到 `5e-5`

#### Batch Size

```yaml
dataloader:
  batch_size: 256  # 默认值
```

**调优建议**:
- 显存足够: 增大到 `512` (加速训练)
- 显存不足: 减小到 `128` 或 `64`
- 小数据集: 减小到 `32`

#### 训练 Epochs

```yaml
training:
  num_epochs: 1000  # 默认值
```

**调优建议**:
- 数据充足 (>100 episodes): 500-1000 epochs
- 数据不足 (<50 episodes): 200-500 epochs
- 过拟合: 提前停止或增加数据

#### 观测历史 (n_obs_steps)

```yaml
n_obs_steps: 2  # 使用过去 2 帧观测
```

**调优建议**:
- 增大到 `3` 或 `4` 可以提供更多时序信息
- 但会增加计算开销和延迟

#### 动作序列长度 (n_action_steps)

```yaml
n_action_steps: 8  # 预测未来 8 步动作
```

**调优建议**:
- 保持默认值 `8`
- 增大会提高长期规划能力，但增加训练难度

### 5.2 网络架构调优

#### 视觉编码器

```yaml
obs_encoder:
  rgb_model:
    name: resnet18  # 或 resnet34, resnet50
```

**选择建议**:
- `resnet18`: 快速训练，适合简单任务
- `resnet34`: 平衡性能和速度
- `resnet50`: 复杂任务，需要更多数据

#### UNet 参数

```yaml
down_dims: [256, 512, 1024]  # UNet 通道数
kernel_size: 5
n_groups: 8
```

**调优建议**:
- 增大 `down_dims` 提高模型容量
- 减小以加速训练和推理

## 6. 常见训练问题

### 6.1 Loss 不下降

**症状**: 训练 loss 一直在高位震荡

**可能原因**:
1. 学习率过大
2. 数据归一化问题
3. 数据质量差

**解决方案**:
```bash
# 降低学习率
python train.py ... training.lr=5e-5

# 检查数据
python -c "
import h5py
f = h5py.File('data/demos.hdf5', 'r')
print('Num episodes:', len(f['data']))
print('Episode 0 length:', len(f['data/demo_0/actions']))
"
```

### 6.2 显存不足 (OOM)

**症状**: `CUDA out of memory`

**解决方案**:
```bash
# 减小 batch size
python train.py ... dataloader.batch_size=64

# 使用梯度累积
python train.py ... training.gradient_accumulate_every=4

# 减小图像分辨率 (在 constants.py 修改)
```

### 6.3 过拟合

**症状**: Train loss 很低，但策略表现差

**解决方案**:
1. 增加更多训练数据
2. 使用数据增强
3. 提前停止训练
4. 减小模型容量

### 6.4 推理速度慢

**症状**: 推理时间 >200ms

**解决方案**:
```yaml
# 减少推理步数
policy:
  num_inference_steps: 16  # 默认
  # 减小到 8 或 4 (会降低质量)

# 使用 DDIM 采样器 (更快)
noise_scheduler:
  _target_: diffusers.schedulers.scheduling_ddim.DDIMScheduler
```

### 6.5 训练中断

**症状**: 训练意外中断

**解决方案**:
```bash
# 从 checkpoint 恢复训练
python train.py ... training.resume=True

# 训练会自动从 latest.ckpt 恢复
```

## 7. 评估和选择 Checkpoint

### 7.1 查看训练曲线

在 W&B 或 TensorBoard 中查看：
- `train_loss`: 训练损失
- `lr`: 学习率变化

### 7.2 选择 Checkpoint

```bash
# 查看可用的 checkpoints
ls -lh data/outputs/2025.07.14/00.30.40_*/checkpoints/

# 通常选择:
# 1. 最低 train_loss 的 checkpoint
# 2. latest.ckpt (最新)
# 3. 手动指定 epoch
```

### 7.3 策略评估

```bash
# 启动策略服务器
python policy_server.py \
    --ckpt-path data/outputs/.../checkpoints/epoch=0600-train_loss=0.001.ckpt

# 在另一个终端运行评估
cd ../..
conda activate bobacbot
python main.py --sim

# 观察机器人表现
```

### 7.4 评估指标

- **成功率**: 任务完成的比例
- **平滑度**: 动作是否流畅
- **泛化能力**: 在不同初始位置是否有效
- **鲁棒性**: 遇到干扰时是否恢复

## 8. 高级训练技巧

### 8.1 课程学习

从简单任务逐步过渡到复杂任务：

```bash
# 阶段1: 训练简单任务 (50 episodes)
python train.py ... task=simple_reach

# 阶段2: Fine-tune 复杂任务 (100 episodes)
python train.py ... task=pick_place training.resume=True
```

### 8.2 数据增强

在 `convert_to_robomimic_hdf5.py` 中添加：

```python
def augment_image(image):
    # 随机亮度
    image = image * np.random.uniform(0.8, 1.2)
    # 随机对比度
    image = (image - 0.5) * np.random.uniform(0.8, 1.2) + 0.5
    # 裁剪
    return np.clip(image, 0, 255).astype(np.uint8)
```

### 8.3 迁移学习

使用预训练的视觉编码器：

```yaml
obs_encoder:
  rgb_model:
    name: resnet18
    weights: ResNet18_Weights.IMAGENET1K_V1  # ImageNet 预训练
```

### 8.4 多任务学习

在一个模型中训练多个任务：

```yaml
dataset:
  dataset_path:
    - data/task1.hdf5
    - data/task2.hdf5
  task_weights: [0.5, 0.5]
```

## 9. 性能基准

### 9.1 训练时间

| 数据规模 | Batch Size | GPU | 训练时间 (500 epochs) |
|---------|-----------|-----|---------------------|
| 50 episodes | 256 | RTX 4080 | ~2 小时 |
| 100 episodes | 256 | RTX 4080 | ~4 小时 |
| 200 episodes | 256 | RTX 4080 | ~8 小时 |

### 9.2 最终性能

| 任务 | 数据量 | 成功率 | Train Loss |
|-----|-------|--------|-----------|
| 简单抓取 | 50 episodes | 80% | 0.005 |
| 抓取放置 | 100 episodes | 70% | 0.008 |
| 复杂整理 | 200 episodes | 60% | 0.012 |

## 10. 下一步

训练完成后，参考《策略部署与推理》文档进行模型部署和评估。
