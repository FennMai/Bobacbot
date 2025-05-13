import mujoco
import glfw
import numpy as np

def display_retract_pose(xml_path):
    # 加载模型与数据
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)

    # 设置重力补偿
    model.body_gravcomp[:] = 1.0

    site_id = model.site('pinch_site').id
    print("site_id =", site_id)
    site_pos = data.site(site_id).xpos  # 末端当前坐标
    print("site_pos =", site_pos)
    site_mat = data.site(site_id).xmat   # 末端当前旋转矩阵
    print("site_mat =", site_mat)
    
    # 获取retract姿态信息
    qpos = model.key('retract').qpos
    # 设置关节角度为 retract 状态
    data.qpos = qpos
    print("current qpos =", data.qpos)

    # 前向动力学以更新位姿
    mujoco.mj_forward(model, data)

    print("site_id =", site_id)
    site_pos = data.site(site_id).xpos  # 末端当前坐标
    print("site_pos =", site_pos)
    site_mat = data.site(site_id).xmat   # 末端当前旋转矩阵
    print("site_mat =", site_mat)
    
    # 初始化 Mujoco 可视化窗口
    glfw.init()
    window = glfw.create_window(1200, 900, "MuJoCo - Position", None, None)
    glfw.make_context_current(window)
    glfw.swap_interval(1)

    # 初始化 Mujoco 的场景和上下文
    scn = mujoco.MjvScene(model, maxgeom=1000)
    cam = mujoco.MjvCamera()
    opt = mujoco.MjvOption()
    con = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_150)

    # 设置相机视角（你可以更改这些数值来调整视角）
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat = np.array([0.5, 0.0, 0.5])  # 观察点
    cam.distance = 2.0
    cam.elevation = -20
    cam.azimuth = 90

    while not glfw.window_should_close(window):
        # 渲染
        viewport = mujoco.MjrRect(0, 0, 0, 0)
        viewport.width, viewport.height = glfw.get_framebuffer_size(window)

        # 更新场景
        mujoco.mjv_updateScene(model, data, opt, None, cam, mujoco.mjtCatBit.mjCAT_ALL, scn)
        mujoco.mjr_render(viewport, scn, con)

        # 交换缓冲区
        glfw.swap_buffers(window)
        glfw.poll_events()

    glfw.terminate()

if __name__ == '__main__':
    xml_path = 'models/gen72/gen72.xml'
    # xml_path = 'models/kinova_gen3/gen3.xml'  
    display_retract_pose(xml_path)
