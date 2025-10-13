#!/usr/bin/env python3
"""
测试机器人位置随机化功能
演示如何使用新的 randomize_robot_position 选项
"""

import time
from mujoco_env import MujocoEnv

def test_fixed_position():
    """测试固定位置模式 (默认行为)"""
    print("=== 测试固定位置模式 ===")
    env = MujocoEnv(randomize_robot_position=False, show_viewer=True)
    
    try:
        for i in range(3):
            print(f"\n重置 #{i+1}:")
            env.reset()
            obs = env.get_obs()
            base_pose = obs['base_pose']
            print(f"机器人位置: x={base_pose[0]:.3f}, y={base_pose[1]:.3f}, yaw={base_pose[2]:.3f}")
            time.sleep(2)
    finally:
        env.close()

def test_random_position():
    """测试随机位置模式"""
    print("\n=== 测试随机位置模式 ===")
    env = MujocoEnv(randomize_robot_position=True, show_viewer=True)
    
    try:
        for i in range(5):
            print(f"\n重置 #{i+1}:")
            env.reset()
            obs = env.get_obs()
            base_pose = obs['base_pose']
            print(f"机器人位置: x={base_pose[0]:.3f}, y={base_pose[1]:.3f}, yaw={base_pose[2]:.3f}")
            time.sleep(2)
    finally:
        env.close()

if __name__ == '__main__':
    print("机器人位置随机化功能测试")
    print("按 Ctrl+C 提前退出")
    
    try:
        # 测试固定位置
        test_fixed_position()
        
        print("\n等待3秒后开始随机位置测试...")
        time.sleep(3)
        
        # 测试随机位置
        test_random_position()
        
        print("\n测试完成!")
        
    except KeyboardInterrupt:
        print("\n用户中断，测试结束")
    except Exception as e:
        print(f"\n错误: {e}")
