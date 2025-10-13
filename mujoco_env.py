# 作者: Jimmy Wu
# 日期: 2024年10月
#
# 注释: 这是一个基础的仿真环境，用于检验遥操作和模仿学习的
# 真实世界管道。性能指标、奖励信号和终止信号尚未实现。

import math  # 数学运算库
import multiprocessing as mp  # 多进程处理库
import time  # 时间处理库
from multiprocessing import shared_memory  # 多进程共享内存
from threading import Thread  # 线程处理
import cv2 as cv  # OpenCV计算机视觉库
import mujoco  # MuJoCo物理仿真引擎
import mujoco.viewer  # MuJoCo查看器
import numpy as np  # 数值计算库
from ruckig import InputParameter, OutputParameter, Result, Ruckig  # Ruckig轨迹生成库
from constants import POLICY_CONTROL_PERIOD  # 策略控制周期常量
import logging  # 日志记录库
from scipy.spatial.transform import Rotation as R  # 空间旋转变换库
# 重要: 导入逆运动学求解器
from ik_solver import IKSolver
class ShmState:
    """
    共享内存状态类
    用于在多进程间共享机器人状态信息，包括底盘位姿、机械臂位置和四元数、夹爪位置等
    """
    def __init__(self, existing_instance=None):
        # 创建包含所有状态信息的数组: 底盘位姿(3) + 机械臂位置(3) + 机械臂四元数(4) + 夹爪位置(1) + 初始化标志(1)
        arr = np.empty(3 + 3 + 4 + 1 + 1)
        if existing_instance is None:
            # 创建新的共享内存实例
            self.shm = shared_memory.SharedMemory(create=True, size=arr.nbytes)
        else:
            # 连接到已存在的共享内存实例
            self.shm = shared_memory.SharedMemory(name=existing_instance.shm.name)
        # 将共享内存缓冲区映射为numpy数组
        self.data = np.ndarray(arr.shape, buffer=self.shm.buf)
        # 分配数组的不同部分给各个状态变量
        self.base_pose = self.data[:3]          # 底盘位姿 (x, y, theta)
        self.arm_pos = self.data[3:6]           # 机械臂末端位置 (x, y, z)
        self.arm_quat = self.data[6:10]         # 机械臂末端四元数 (w, x, y, z)
        self.gripper_pos = self.data[10:11]     # 夹爪位置 [0-1]
        self.initialized = self.data[11:12]     # 初始化标志
        self.initialized[:] = 0.0               # 初始化为未初始化状态

    def close(self):
        """关闭共享内存连接"""
        self.shm.close()

class ShmImage:
    """
    共享内存图像类
    用于在多进程间共享相机图像数据
    """
    def __init__(self, camera_name=None, width=None, height=None, existing_instance=None):
        if existing_instance is None:
            # 创建新的图像共享内存实例
            self.camera_name = camera_name
            # 创建RGB图像数组 (高度 x 宽度 x 3通道)
            arr = np.empty((height, width, 3), dtype=np.uint8)
            self.shm = shared_memory.SharedMemory(create=True, size=arr.nbytes)
        else:
            # 连接到已存在的图像共享内存实例
            self.camera_name = existing_instance.camera_name
            arr = existing_instance.data
            self.shm = shared_memory.SharedMemory(name=existing_instance.shm.name)
        # 将共享内存缓冲区映射为图像数组
        self.data = np.ndarray(arr.shape, dtype=np.uint8, buffer=self.shm.buf)
        self.data.fill(0)  # 初始化为全黑图像

    def close(self):
        """关闭图像共享内存连接"""
        self.shm.close()

# 改编自 https://github.com/google-deepmind/mujoco/blob/main/python/mujoco/renderer.py
class Renderer:
    """
    MuJoCo渲染器类
    用于离屏渲染相机图像并存储到共享内存中
    """
    def __init__(self, model, data, shm_image):
        self.model = model          # MuJoCo模型
        self.data = data            # MuJoCo数据
        self.image = np.empty_like(shm_image.data)  # 临时图像缓冲区

        # 连接到已存在的共享内存图像
        self.shm_image = ShmImage(existing_instance=shm_image)

        # 设置相机参数
        camera_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA.value, shm_image.camera_name)
        width, height = model.cam_resolution[camera_id]
        self.camera = mujoco.MjvCamera()
        self.camera.fixedcamid = camera_id              # 固定相机ID
        self.camera.type = mujoco.mjtCamera.mjCAMERA_FIXED  # 固定相机类型

        # 设置渲染上下文
        self.rect = mujoco.MjrRect(0, 0, width, height)  # 渲染矩形区域
        self.gl_context = mujoco.gl_context.GLContext(width, height)  # OpenGL上下文
        self.gl_context.make_current()
        self.mjr_context = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_150.value)  # MuJoCo渲染上下文
        mujoco.mjr_setBuffer(mujoco.mjtFramebuffer.mjFB_OFFSCREEN.value, self.mjr_context)  # 设置离屏缓冲

        # 设置场景选项
        self.scene_option = mujoco.MjvOption()
        self.scene = mujoco.MjvScene(model, 10000)  # 最大对象数量为10000

    def render(self):
        """渲染一帧图像并更新共享内存"""
        self.gl_context.make_current()
        # 更新场景
        mujoco.mjv_updateScene(self.model, self.data, self.scene_option, None, self.camera, mujoco.mjtCatBit.mjCAT_ALL.value, self.scene)
        # 渲染场景
        mujoco.mjr_render(self.rect, self.scene, self.mjr_context)
        # 读取像素数据
        mujoco.mjr_readPixels(self.image, None, self.rect, self.mjr_context)
        # 垂直翻转图像并存储到共享内存 (MuJoCo使用OpenGL坐标系，需要翻转)
        self.shm_image.data[:] = np.flipud(self.image)

    def close(self):
        """释放渲染资源"""
        self.gl_context.free()
        self.gl_context = None
        self.mjr_context.free()
        self.mjr_context = None

class BaseController:
    """
    底盘控制器类
    负责控制机器人底盘的运动，使用Ruckig进行在线轨迹生成以实现平滑运动
    """
    def __init__(self, qpos, qvel, ctrl, timestep):
        self.qpos = qpos        # 底盘关节位置 (x, y, theta)
        self.qvel = qvel        # 底盘关节速度
        self.ctrl = ctrl        # 底盘控制输出

        # OTG (在线轨迹生成器) - 用于生成平滑的运动轨迹
        num_dofs = 3  # 自由度数量: x, y, theta
        self.last_command_time = None   # 上次命令时间
        self.otg = Ruckig(num_dofs, timestep)           # Ruckig轨迹生成器
        self.otg_inp = InputParameter(num_dofs)         # 输入参数
        self.otg_out = OutputParameter(num_dofs)        # 输出参数
        # 设置速度和加速度限制
        self.otg_inp.max_velocity = [0.5, 0.5, 3.14]       # 最大速度: x,y为0.5m/s, 角度为180度/s
        self.otg_inp.max_acceleration = [0.5, 0.5, 2.36]   # 最大加速度
        self.otg_res = None     # 轨迹生成结果

    def reset(self):
        """重置底盘控制器到初始状态"""
        # 将底盘初始化到原点
        # self.qpos[:] = np.zeros(3)
        self.ctrl[:] = self.qpos

        # 初始化轨迹生成器
        self.last_command_time = time.time()
        self.otg_inp.current_position = self.qpos       # 当前位置
        self.otg_inp.current_velocity = self.qvel       # 当前速度
        self.otg_inp.target_position = self.qpos        # 目标位置
        self.otg_res = Result.Finished                  # 轨迹生成状态: 已完成

    def control_callback(self, command):
        """
        控制回调函数
        处理外部命令并更新轨迹生成器
        """
        if command is not None:
            self.last_command_time = time.time()
            if 'base_pose' in command:
                # 设置目标底盘位姿
                self.otg_inp.target_position = command['base_pose']
                self.otg_res = Result.Working   # 开始生成新轨迹

        # 如果命令流中断，维持当前位姿 (超过2.5个控制周期)
        if time.time() - self.last_command_time > 2.5 * POLICY_CONTROL_PERIOD:
            self.otg_inp.target_position = self.qpos
            self.otg_res = Result.Working

        # 更新轨迹生成器
        if self.otg_res == Result.Working:
            self.otg_res = self.otg.update(self.otg_inp, self.otg_out)
            self.otg_out.pass_to_input(self.otg_inp)    # 将输出传递给下一步的输入
            self.ctrl[:] = self.otg_out.new_position    # 更新控制输出

class ArmController:
    """
    机械臂控制器类
    负责控制6自由度机械臂的运动，结合逆运动学和轨迹生成实现末端位姿控制
    """
    def __init__(self, qpos, qvel, ctrl, qpos_gripper, ctrl_gripper, timestep):
        self.qpos = qpos                    # 机械臂关节位置
        self.qvel = qvel                    # 机械臂关节速度
        self.ctrl = ctrl                    # 机械臂控制输出
        self.qpos_gripper = qpos_gripper    # 夹爪位置
        self.ctrl_gripper = ctrl_gripper    # 夹爪控制输出

        # 逆运动学求解器
        self.ik_solver = IKSolver(ee_offset=0.0)

        # OTG (在线轨迹生成器)
        # num_dofs = 7  # 7个关节自由度
        num_dofs = 6  # change to 6 for eco65b
        self.last_command_time = None   # 上次命令时间
        self.otg = Ruckig(num_dofs, timestep)           # Ruckig轨迹生成器
        self.otg_inp = InputParameter(num_dofs)         # 输入参数
        self.otg_out = OutputParameter(num_dofs)        # 输出参数
        # # 设置各关节的速度和加速度限制
        # # 前4个关节: 80度/s, 后3个关节: 140度/s
        # self.otg_inp.max_velocity = 4 * [math.radians(80)] + 3 * [math.radians(140)]
        # # 前4个关节: 240度/s², 后3个关节: 450度/s²
        # self.otg_inp.max_acceleration = 4 * [math.radians(240)] + 3 * [math.radians(450)]
        
        # 设置各关节的速度和加速度限制 (调整为6个关节)
        # 根据eco65b的关节特性调整限制
        self.otg_inp.max_velocity = 6 * [math.radians(80)]      # 6个关节都使用80度/s
        self.otg_inp.max_acceleration = 6 * [math.radians(240)] # 6个关节都使用240度/s²

        self.otg_res = None     # 轨迹生成结果

    def reset(self):
        """重置机械臂到"收缩"配置"""
        # 将机械臂初始化到预定义的"收缩"姿态
        # self.qpos[:] = np.array([0.0, -1.0, 0.0, -2.0, 0.0, 0.63, 0.0])
        # 将机械臂初始化到eco65b的预定义"收缩"姿态 (6个关节)
        # 使用eco65b.xml中retract keyframe的值: [0.0, 0.52, -1.2, 0.34, 1.57, -1.57]
        self.qpos[:] = np.array([0.0, 0.52, -1.2, 0.34, 1.57, -1.57])
        print("setting the arm reset: ", self.qpos)
        self.ctrl[:] = self.qpos
        self.ctrl_gripper[:] = 0.0  # 夹爪闭合

        # 初始化轨迹生成器
        self.last_command_time = time.time()
        self.otg_inp.current_position = self.qpos       # 当前关节位置
        self.otg_inp.current_velocity = self.qvel       # 当前关节速度
        self.otg_inp.target_position = self.qpos        # 目标关节位置
        self.otg_res = Result.Finished                  # 轨迹生成状态: 已完成

    def control_callback(self, command):
        """
        控制回调函数
        处理外部命令，包括末端位姿和夹爪控制
        """
        if command is not None:
            self.last_command_time = time.time()

            if 'arm_pos' in command:
                # 使用逆运动学求解新的目标位姿对应的关节角度
                qpos = self.ik_solver.solve(command['arm_pos'], command['arm_quat'], self.qpos)
                # 展开关节角度 (处理角度跳跃问题)
                qpos = self.qpos + np.mod((qpos - self.qpos) + np.pi, 2 * np.pi) - np.pi

                # 设置目标机械臂关节位置
                self.otg_inp.target_position = qpos
                self.otg_res = Result.Working   # 开始生成新轨迹

            if 'gripper_pos' in command:
                # 设置目标夹爪位置 (0-1映射到0-255的控制范围)
                self.ctrl_gripper[:] = 255.0 * command['gripper_pos']  # fingers_actuator, ctrlrange [0, 255]

        # 如果命令流中断，维持当前位姿 (超过2.5个控制周期)
        if time.time() - self.last_command_time > 2.5 * POLICY_CONTROL_PERIOD:
            self.otg_inp.target_position = self.otg_out.new_position
            self.otg_res = Result.Working # 工作中

        # 更新轨迹生成器
        if self.otg_res == Result.Working:
            self.otg_res = self.otg.update(self.otg_inp, self.otg_out)
            self.otg_out.pass_to_input(self.otg_inp)    # 将输出传递给下一步的输入
            self.ctrl[:] = self.otg_out.new_position    # 更新关节控制输出

class MujocoSim:
    """
    MuJoCo仿真主类
    负责管理整个物理仿真环境，包括模型加载、控制器管理和状态更新
    """
    def __init__(self, mjcf_path, command_queue, shm_state, show_viewer=True, randomize_robot_position=False):
        self.model = mujoco.MjModel.from_xml_path(mjcf_path)    # 从XML文件加载MuJoCo模型
        self.data = mujoco.MjData(self.model)                   # 创建仿真数据结构
        self.command_queue = command_queue                      # 命令队列，用于接收外部控制命令
        self.show_viewer = show_viewer                          # 是否显示可视化窗口
        self.randomize_robot_position = randomize_robot_position  # 是否随机化机器人位置
        
        # 定义机器人位置随机化范围 (世界坐标系)
        self.robot_position_range = {
            'x': [-0.5, 0.5],    # X轴范围: -0.5 到 0.5 米
            'y': [-0.5, 0.5],    # Y轴范围: -0.5 到 0.5 米
            'yaw': [0, 0]  # 朝向角度范围: 0 到 0
        }

        # 为除物体外的所有物体启用重力补偿
        self.model.body_gravcomp[:] = 1.0
        body_names = {self.model.body(i).name for i in range(self.model.nbody)}
        for object_name in ['cube']:
            if object_name in body_names:
                # 对指定物体禁用重力补偿 (让物体自然下落)
                self.model.body_gravcomp[self.model.body(object_name).id] = 0.0

        # 缓存数组切片的引用，提高访问效率
        base_dofs = self.model.body('base_link').jntnum.item()  # 底盘自由度数量
        arm_dofs = 6                                            # 机械臂自由度数量 (从7改为6) eco65b
        # 底盘相关数组切片
        self.qpos_base = self.data.qpos[:base_dofs]
        qvel_base = self.data.qvel[:base_dofs]
        ctrl_base = self.data.ctrl[:base_dofs]
        # 机械臂相关数组切片
        qpos_arm = self.data.qpos[base_dofs:(base_dofs + arm_dofs)]
        qvel_arm = self.data.qvel[base_dofs:(base_dofs + arm_dofs)]
        ctrl_arm = self.data.ctrl[base_dofs:(base_dofs + arm_dofs)]
        # 夹爪相关数组切片
        self.qpos_gripper = self.data.qpos[(base_dofs + arm_dofs):(base_dofs + arm_dofs + 1)]
        ctrl_gripper = self.data.ctrl[(base_dofs + arm_dofs):(base_dofs + arm_dofs + 1)]
        # 立方体相关数组切片 (需要根据实际场景调整偏移量)
        self.qpos_cube = self.data.qpos[(base_dofs + arm_dofs + 8):(base_dofs + arm_dofs + 8 + 7)]

        # 创建控制器实例
        self.base_controller = BaseController(self.qpos_base, qvel_base, ctrl_base, self.model.opt.timestep)
        self.arm_controller = ArmController(qpos_arm, qvel_arm, ctrl_arm, self.qpos_gripper, ctrl_gripper, self.model.opt.timestep)

        # 连接到共享内存状态，用于观测数据发布
        self.shm_state = ShmState(existing_instance=shm_state)

        # 用于计算机械臂位置和四元数的变量
        site_id = self.model.site('pinch_site').id      # 获取夹爪位置点ID
        self.site_xpos = self.data.site(site_id).xpos   # 夹爪位置
        self.site_xmat = self.data.site(site_id).xmat   # 夹爪旋转矩阵
        self.site_quat = np.empty(4)                    # 夹爪四元数缓冲区
        self.base_height = self.model.body('eco65b/base_link').pos[2]  # 底盘高度, change to eco65b
        self.base_rot_axis = np.array([0.0, 0.0, 1.0])  # 底盘旋转轴 (Z轴)
        self.base_quat_inv = np.empty(4)                # 底盘逆四元数缓冲区

        # 重置环境到初始状态
        self.reset()

        # 设置控制回调函数
        mujoco.set_mjcb_control(self.control_callback)

    def reset(self):
        """重置仿真环境到初始状态"""
        # 重置仿真数据
        mujoco.mj_resetData(self.model, self.data)

        # 根据选项决定是否随机化机器人位置
        if self.randomize_robot_position:
            # 随机化机器人底盘位置和朝向
            self.qpos_base[0] = np.random.uniform(*self.robot_position_range['x'])    # 随机X位置
            self.qpos_base[1] = np.random.uniform(*self.robot_position_range['y'])    # 随机Y位置
            self.qpos_base[2] = np.random.uniform(*self.robot_position_range['yaw'])  # 随机朝向
            print(f"机器人随机位置: x={self.qpos_base[0]:.2f}, y={self.qpos_base[1]:.2f}, yaw={self.qpos_base[2]:.2f}")
        else:
            # 使用XML文件中定义的默认位置
            # XML中base_link的pos="-0.5 0 0.035"，但这里只控制x,y,yaw，z由模型决定
            self.qpos_base[0] = 0.0  # XML中的默认X位置
            self.qpos_base[1] = 0.0   # XML中的默认Y位置  
            self.qpos_base[2] = 0.0   # XML中的默认朝向
            print("机器人使用默认位置: x=0.0, y=0.0, yaw=0.0")

        # 随机化立方体位置和朝向
        self.qpos_cube[:2] += np.random.uniform(-0.1, 0.1, 2)  # 在xy平面上随机偏移
        theta = np.random.uniform(-math.pi, math.pi)            # 随机旋转角度
        # 将旋转角度转换为四元数 (绕Z轴旋转)
        self.qpos_cube[3:7] = np.array([math.cos(theta / 2), 0, 0, math.sin(theta / 2)])
        
        # 更新物理状态 
        mujoco.mj_forward(self.model, self.data)

        # 重置所有控制器 
        self.base_controller.reset()
        self.arm_controller.reset()

    def control_callback(self, *_):
        """
        控制回调函数
        在每个仿真步骤中被调用，处理控制命令并更新状态
        """
        # 检查是否有新的控制命令
        command = None if self.command_queue.empty() else self.command_queue.get()
        if command == 'reset':
            self.reset()

        # 调用各控制器的控制回调
        self.base_controller.control_callback(command)
        self.arm_controller.control_callback(command)

        # 更新底盘位姿到共享内存
        self.shm_state.base_pose[:] = self.qpos_base

        # 更新机械臂末端位置 (转换到底盘局部坐标系)
        site_xpos = self.site_xpos.copy()
        site_xpos[2] -= self.base_height  # 减去底盘高度偏移
        site_xpos[:2] -= self.qpos_base[:2]  # 减去底盘位置偏移
        # 将全局坐标转换为底盘局部坐标
        mujoco.mju_axisAngle2Quat(self.base_quat_inv, self.base_rot_axis, -self.qpos_base[2])  # 底盘朝向的逆四元数
        mujoco.mju_rotVecQuat(self.shm_state.arm_pos, site_xpos, self.base_quat_inv)  # 机械臂位置 (局部坐标系)

        # 更新机械臂末端四元数 (转换到底盘局部坐标系)
        mujoco.mju_mat2Quat(self.site_quat, self.site_xmat)  # 旋转矩阵转四元数
        mujoco.mju_mulQuat(self.shm_state.arm_quat, self.base_quat_inv, self.site_quat)  # 机械臂四元数 (局部坐标系)

        # 更新夹爪位置 (标准化到0-1范围)
        self.shm_state.gripper_pos[:] = self.qpos_gripper / 0.8  # right_driver_joint, 关节范围 [0, 0.8]

        # 通知reset()函数状态已初始化完成
        self.shm_state.initialized[:] = 1.0

    def launch(self):
        """启动仿真环境"""
        if self.show_viewer:
            # 启动MuJoCo可视化查看器
            mujoco.viewer.launch(self.model, self.data, show_left_ui=False, show_right_ui=False)

        else:
            # 无头模式运行仿真，以实时速度运行
            last_step_time = 0
            while True:
                # 等待直到到达下一个仿真时间步
                while time.time() - last_step_time < self.model.opt.timestep:
                    time.sleep(0.0001)
                last_step_time = time.time()
                mujoco.mj_step(self.model, self.data)  # 执行一步仿真

class MujocoEnv:
    """
    MuJoCo环境主类
    提供高级接口来管理整个仿真环境，包括多进程管理、图像渲染和状态观测
    """
    def __init__(self, render_images=True, show_viewer=True, show_images=False, randomize_robot_position=False):
        # 场景文件路径配置
        # self.mjcf_path = 'models/stanford_tidybot/scene.xml'
        self.mjcf_path = 'models/bobacbot_demo_scene.xml'
        self.render_images = render_images              # 是否渲染图像
        self.show_viewer = show_viewer                  # 是否显示可视化窗口
        self.show_images = show_images                  # 是否显示图像窗口
        self.randomize_robot_position = randomize_robot_position  # 是否随机化机器人位置
        self.command_queue = mp.Queue(1)                # 命令队列，容量为1

        # 创建状态观测的共享内存
        self.shm_state = ShmState()

        # 创建图像观测的共享内存
        if self.render_images:
            self.shm_images = []
            model = mujoco.MjModel.from_xml_path(self.mjcf_path)
            for camera_id in range(model.ncam):
                camera_name = model.camera(camera_id).name
                width, height = model.cam_resolution[camera_id]
                self.shm_images.append(ShmImage(camera_name, width, height))

        # 启动物理仿真进程 (守护进程，主进程结束时自动结束)
        mp.Process(target=self.physics_loop, daemon=True).start()

        if self.render_images and self.show_images:
            # 启动图像可视化进程
            mp.Process(target=self.visualizer_loop, daemon=True).start()

    def physics_loop(self):
        """物理仿真循环 (在独立进程中运行)"""
        # 创建仿真实例
        sim = MujocoSim(self.mjcf_path, self.command_queue, self.shm_state, 
                       show_viewer=self.show_viewer, randomize_robot_position=self.randomize_robot_position)

        # 启动渲染循环
        if self.render_images:
            Thread(target=self.render_loop, args=(sim.model, sim.data), daemon=True).start()

        # 启动仿真 (在创建的同一线程中启动以避免段错误)
        sim.launch()

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
                print(f'警告: 离屏渲染耗时 {1000 * render_time:.1f} 毫秒，尝试缩小MuJoCo查看器窗口来加速离屏渲染')

    def visualizer_loop(self):
        """图像可视化循环 (在独立进程中运行)"""
        shm_images = [ShmImage(existing_instance=shm_image) for shm_image in self.shm_images]
        last_imshow_time = time.time()
        while True:
            # 控制显示帧率为10fps
            while time.time() - last_imshow_time < 0.1:
                time.sleep(0.01)
            last_imshow_time = time.time()
            # 显示所有相机图像
            for i, shm_image in enumerate(shm_images):
                cv.imshow(shm_image.camera_name, cv.cvtColor(shm_image.data, cv.COLOR_RGB2BGR))
                cv.moveWindow(shm_image.camera_name, 640 * i, -100)  # 排列窗口位置
            cv.waitKey(1)

    def reset(self):
        """重置环境到初始状态"""
        self.shm_state.initialized[:] = 0.0     # 标记为未初始化
        self.command_queue.put('reset')         # 发送重置命令

        # 等待状态发布初始化完成
        while self.shm_state.initialized == 0.0:
            time.sleep(0.01)

        # 等待图像渲染初始化完成 (注意: 假设全零不是有效图像)
        if self.render_images:
            while any(np.all(shm_image.data == 0) for shm_image in self.shm_images):
                time.sleep(0.01)

    def get_obs(self):
        """获取当前观测数据"""
        # 调整四元数格式: (w, x, y, z) -> (x, y, z, w)
        arm_quat = self.shm_state.arm_quat[[1, 2, 3, 0]]
        if arm_quat[3] < 0.0:  # 强制四元数唯一性 (w >= 0)
            np.negative(arm_quat, out=arm_quat)
        
        # 构建观测字典
        obs = {
            'base_pose': self.shm_state.base_pose.copy(),    # 底盘位姿
            'arm_pos': self.shm_state.arm_pos.copy(),        # 机械臂末端位置
            'arm_quat': arm_quat,                            # 机械臂末端四元数
            'gripper_pos': self.shm_state.gripper_pos.copy(), # 夹爪位置
        }
        # 添加图像观测
        if self.render_images:
            for shm_image in self.shm_images:
                obs[f'{shm_image.camera_name}_image'] = shm_image.data.copy()
        return obs

    def step(self, action):
        """执行一步动作"""
        # 注意: 我们故意不在这里返回obs，以防止策略使用过时数据
        self.command_queue.put(action)

    def close(self):
        """关闭环境并清理资源"""
        self.shm_state.close()
        self.shm_state.shm.unlink()     # 删除共享内存
        if self.render_images:
            for shm_image in self.shm_images:
                shm_image.close()
                shm_image.shm.unlink()  # 删除图像共享内存

if __name__ == '__main__':
    """
    主函数 - 测试仿真环境
    演示如何使用MujocoEnv类进行基本的环境交互
    """
    # 创建环境实例 - 演示不同的配置选项
    env = MujocoEnv()  # 默认配置: 机器人使用XML中的固定位置
    # env = MujocoEnv(randomize_robot_position=True)     # 启用机器人位置随机化
    # env = MujocoEnv(show_images=True)                  # 显示图像窗口
    # env = MujocoEnv(render_images=False)               # 不渲染图像
    # env = MujocoEnv(randomize_robot_position=True, show_images=True)  # 随机位置 + 显示图像
    
    try:
        # 重置环境并获取初始观测
        env.reset()
        obs = env.get_obs()
        print(f"初始观测:{[(k, v.shape) if v.ndim == 3 else (k, v) for (k, v) in obs.items()]}")
        
        # 保存初始状态作为目标
        target_base_pose = obs['base_pose']     # 目标底盘位姿
        target_arm_pos = obs['arm_pos']         # 目标机械臂位置
        target_arm_quat = obs['arm_quat']       # 目标机械臂四元数
        
        # 主循环 - 保持机械臂位姿不变，随机控制夹爪
        while True:
            # 保持初始机械臂位姿，随机化夹爪位置
            action = {
                'base_pose': target_base_pose,      # 底盘保持不动
                'arm_pos':  target_arm_pos,         # 机械臂位置保持不变
                'arm_quat': target_arm_quat,        # 机械臂姿态保持不变
                'gripper_pos': np.random.rand(1),   # 夹爪位置随机化
            }
            env.step(action)                        # 执行动作
            obs = env.get_obs()                     # 获取新的观测
            print([(k, v.shape) if v.ndim == 3 else (k, v) for (k, v) in obs.items()])
            time.sleep(POLICY_CONTROL_PERIOD)       # 等待控制周期 (注意: 不是精确计时)
    finally:
        # 确保环境被正确关闭
        env.close()
