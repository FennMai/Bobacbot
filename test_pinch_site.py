#!/usr/bin/env python3
"""
测试和验证pinch_site姿态的脚本
用于确定正确的末端执行器姿态
"""

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as R

def test_pinch_site_orientation():
    """测试当前pinch_site的姿态"""
    
    # 加载模型
    model = mujoco.MjModel.from_xml_path('models/eco65b/eco65b.xml')
    data = mujoco.MjData(model)
    
    # 获取pinch_site信息
    site_id = model.site('pinch_site').id
    
    # 设置为retract姿态
    retract_qpos = np.array([0.0, 0.52, -1.2, 0.34, 1.57, -1.57])
    data.qpos[:] = retract_qpos
    
    # 计算前向运动学
    mujoco.mj_kinematics(model, data)
    mujoco.mj_comPos(model, data)
    
    # 获取site的位置和姿态
    site_pos = data.site(site_id).xpos.copy()
    site_mat = data.site(site_id).xmat.copy().reshape(3, 3)
    
    # 转换为四元数 (MuJoCo格式 [w, x, y, z])
    site_quat_mj = np.empty(4)
    mujoco.mju_mat2Quat(site_quat_mj, site_mat.flatten())
    
    # 转换为scipy格式 [x, y, z, w]
    site_quat_scipy = site_quat_mj[[1, 2, 3, 0]]
    
    print("=== Pinch Site 姿态分析 ===")
    print(f"位置: {site_pos}")
    print(f"旋转矩阵:\n{site_mat}")
    print(f"四元数 (MuJoCo格式 [w,x,y,z]): {site_quat_mj}")
    print(f"四元数 (Scipy格式 [x,y,z,w]): {site_quat_scipy}")
    
    # 转换为欧拉角进行直观理解
    r = R.from_quat(site_quat_scipy)
    euler_xyz = r.as_euler('xyz', degrees=True)
    euler_zyx = r.as_euler('zyx', degrees=True)
    
    print(f"欧拉角 (XYZ顺序, 度): {euler_xyz}")
    print(f"欧拉角 (ZYX顺序, 度): {euler_zyx}")
    
    # 分析各个坐标轴方向
    x_axis = site_mat[:, 0]  # 红色轴 (通常是前进方向)
    y_axis = site_mat[:, 1]  # 绿色轴 (通常是左侧方向)
    z_axis = site_mat[:, 2]  # 蓝色轴 (通常是向上方向)
    
    print("\n=== 坐标轴方向分析 ===")
    print(f"X轴方向 (红色): {x_axis}")
    print(f"Y轴方向 (绿色): {y_axis}")
    print(f"Z轴方向 (蓝色): {z_axis}")
    
    return site_pos, site_quat_scipy, site_mat

def suggest_correct_orientation():
    """建议正确的末端执行器姿态"""
    
    print("\n=== 常用的末端执行器姿态建议 ===")
    
    # 几种常见的末端执行器姿态
    orientations = {
        "向下抓取 (Z轴向下)": [0, 0, 0, 1],           # 无旋转
        "向下抓取 (Z轴向下, 旋转180°)": [0, 0, 1, 0],   # 绕Z轴旋转180度
        "水平抓取 (Z轴水平向前)": [0.707, 0, 0, 0.707], # 绕Y轴旋转90度
        "水平抓取 (Z轴水平向右)": [0, 0.707, 0, 0.707], # 绕Z轴旋转90度
        "当前XML中的姿态": [1, 0, 0, 0],               # 绕X轴旋转180度
    }
    
    for desc, quat_scipy in orientations.items():
        # 转换为MuJoCo格式
        quat_mj = [quat_scipy[3], quat_scipy[0], quat_scipy[1], quat_scipy[2]]
        
        # 转换为旋转矩阵
        r = R.from_quat(quat_scipy)
        rot_mat = r.as_matrix()
        euler = r.as_euler('xyz', degrees=True)
        
        print(f"\n{desc}:")
        print(f"  四元数 (scipy [x,y,z,w]): {quat_scipy}")
        print(f"  四元数 (MuJoCo [w,x,y,z]): {quat_mj}")
        print(f"  欧拉角 (度): {euler}")
        print(f"  Z轴方向: {rot_mat[:, 2]}")

def test_different_orientations():
    """测试不同姿态下的逆运动学求解"""
    
    print("\n=== 测试不同姿态的逆运动学求解 ===")
    
    # 这里可以添加逆运动学测试代码
    # 但需要先确定正确的姿态
    pass

if __name__ == '__main__':
    # 测试当前的pinch_site姿态
    pos, quat, mat = test_pinch_site_orientation()
    
    # 提供姿态建议
    suggest_correct_orientation()
    
    print("\n=== 建议的调整步骤 ===")
    print("1. 根据上述分析，确定您希望的末端执行器朝向")
    print("2. 在eco65b.xml中修改pinch_site的quat属性")
    print("3. 重新运行ik_solver测试验证效果")
    print("4. 如果需要，可以调整末端执行器的位置偏移")
