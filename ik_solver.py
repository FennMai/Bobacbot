# Author: Jimmy Wu
# Date: October 2024
#
# References:
# - https://github.com/bulletphysics/bullet3/blob/master/examples/ThirdPartyLibs/BussIK/Jacobian.cpp
# - https://github.com/kevinzakka/mjctrl/blob/main/diffik_nullspace.py
# - https://github.com/google-deepmind/dm_control/blob/main/dm_control/utils/inverse_kinematics.py

import mujoco
import numpy as np

DAMPING_COEFF = 1e-12 # 阻尼系数
MAX_ANGLE_CHANGE = np.deg2rad(45) # 最大角度变化

class IKSolver:
    def __init__(self, ee_offset=0.0):
        # Load Mujoco model (Kinova) without gripper
        # self.model = mujoco.MjModel.from_xml_path('models/gen72/gen72.xml')
        self.model = mujoco.MjModel.from_xml_path('models/eco65b/eco65b.xml') # change to eco65b
        self.data = mujoco.MjData(self.model)
        # 设置重力补偿
        self.model.body_gravcomp[:] = 1.0

        # 关节初始位姿
        self.qpos0 = self.model.key('retract').qpos
        # 获取 pinch_site 标记的索引，是末端执行器位置点
        self.site_id = self.model.site('pinch_site').id
        self.site_pos = self.data.site(self.site_id).xpos  # 末端当前坐标
        self.site_mat = self.data.site(self.site_id).xmat   # 末端当前旋转矩阵

        # 设置 site 到夹爪末端
        self.model.site(self.site_id).pos[2] += ee_offset  # Apply Z-axis offset for the gripper length

        # 预分配数组提高效率
        self.err = np.empty(6)              # 误差向量 (3D 位置误差 + 3D 朝向误差)
        self.err_pos, self.err_rot = self.err[:3], self.err[3:]
        self.site_quat = np.empty(4)       # site 的当前四元数
        self.site_quat_inv = np.empty(4)   # 当前四元数的逆
        self.err_quat = np.empty(4)        # 四元数误差差值
        self.jac = np.empty((6, self.model.nv))  # 6-D 误差对应的雅可比矩阵
        self.jac_pos, self.jac_rot = self.jac[:3], self.jac[3:]
        self.damping = DAMPING_COEFF * np.eye(6)  # 阻尼项
        self.eye = np.eye(self.model.nv)  # identity matrix for joint nullspace projection

    def solve(self, pos, quat, curr_qpos, max_iters=20, err_thresh=1e-4):
        # quat 转为 MuJoCo 风格[w, x, y, z]
        quat = quat[[3, 0, 1, 2]]

        # 设置当前关节配置
        self.data.qpos = curr_qpos

        for _ in range(max_iters):
            # 计算前向运动学
            mujoco.mj_kinematics(self.model, self.data)
            mujoco.mj_comPos(self.model, self.data)

            # 位置误差：目标 - 当前 site
            self.err_pos[:] = pos - self.site_pos

            # 四元数误差 -> 旋转误差
            mujoco.mju_mat2Quat(self.site_quat, self.site_mat)         # mat -> quat
            mujoco.mju_negQuat(self.site_quat_inv, self.site_quat)    # quat 的逆
            mujoco.mju_mulQuat(self.err_quat, quat, self.site_quat_inv) # quat error (target - current)
            mujoco.mju_quat2Vel(self.err_rot, self.err_quat, 1.0)     # quat error -> axis angle vel

            # 检查误差是否满足阈值
            if np.linalg.norm(self.err) < err_thresh:
                break

            # 雅可比矩阵
            mujoco.mj_jacSite(self.model, self.data, self.jac_pos, self.jac_rot, self.site_id)
            # Damped least squares
            update = self.jac.T @ np.linalg.solve(self.jac @ self.jac.T + self.damping, self.err)

            # Null space 部分（使配置更接近 qpos0）
            qpos0_err = np.mod(self.qpos0 - self.data.qpos + np.pi, 2 * np.pi) - np.pi
            null_proj = self.eye - (self.jac.T @ np.linalg.pinv(self.jac @ self.jac.T + self.damping)) @ self.jac
            update += null_proj @ qpos0_err  # nullspace projection 加权更新

            # 限制更新角度幅度，防抖动
            update_max = np.abs(update).max()
            if update_max > MAX_ANGLE_CHANGE:
                update *= MAX_ANGLE_CHANGE / update_max

            # 应用更新到关节角度上
            mujoco.mj_integratePos(self.model, self.data.qpos, update, 1.0)

        return self.data.qpos.copy()
    
if __name__ == '__main__':
    ik_solver = IKSolver()
    
    # 调整测试参数以适应eco65b机械臂
    # 需要重新确定合适的末端位置和姿态
    home_pos, home_quat = np.array([0.3, 0.0, 0.2]), np.array([0.0, 1.0, 0.0, 0.0])  # 调整为eco65b的合理工作空间
    
    # 使用eco65b的retract关节角度 (6个关节)
    # 从eco65b.xml中的retract keyframe: "0.0 0.52 -1.2 0.34 1.57 -1.57"
    retract_qpos = np.array([0.0, 0.52, -1.2, 0.34, 1.57, -1.57])
    
    print(f"eco65b关节数量: {len(retract_qpos)}")
    print(f"收缩姿态 (弧度): {retract_qpos}")
    print(f"收缩姿态 (度数): {np.rad2deg(retract_qpos).round()}")

    import time
    start_time = time.time()
    for _ in range(1000):
        qpos = ik_solver.solve(home_pos, home_quat, retract_qpos)
    elapsed_time = time.time() - start_time
    print(f'平均每次调用时间: {elapsed_time:.3f} ms')

    # 输出逆运动学求解结果
    result_qpos = ik_solver.solve(home_pos, home_quat, retract_qpos)
    print(f"逆运动学求解结果 (度数): {np.rad2deg(result_qpos).round()}")
    print(f"逆运动学求解结果 (弧度): {result_qpos.round(3)}")
