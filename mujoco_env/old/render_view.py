import mujoco
import mujoco.viewer
import numpy as np
import cv2
from PIL import Image
import glfw
import sys
import pdb


max_width = 1920
max_height = 1080

xml_path = '/home/nayaka/software/mujoco_menagerie/franka_fr3/scene.xml'

# Load your model
model = mujoco.MjModel.from_xml_path(xml_path)
data = mujoco.MjData(model)


# Function to render image from camera
def render_camera(model, data, camera_name):
    camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
    width, height = model.vis.global_.offwidth, model.vis.global_.offheight
    rgba = np.zeros((height, width, 4), dtype=np.uint8)
    depth = np.zeros((height, width, 1), dtype=np.float32)

    con = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_150)


    mujoco.mjr_setBuffer(mujoco.mjtFramebuffer.mjFB_OFFSCREEN, con)
    mujoco.mjv_updateScene(model, data, mujoco.MjvOption(), None, mujoco.MjvCamera(), 
                           mujoco.mjtCatBit.mjCAT_ALL, mujoco.MjvScene())
    mujoco.mjr_render(0, 0, width, height, 0, 0, width, height)
    mujoco.mjr_readPixels(rgba, depth, 0, 0, width, height, 0)
    
    return rgba[:,:,:3]  # Return RGB image

pdb.set_trace()

# Create a viewer
with mujoco.viewer.launch_passive(model, data) as viewer:
    # Simulation loop
    for i in range(1000):  # Run for 1000 steps
        # Step the simulation
        mujoco.mj_step(model, data)
        
        # Update the viewer
        viewer.sync()
        
        # Render image from camera
        if i % 10 == 0:  # Capture image every 10 steps
            # Nd array
            image = render_camera(model, data, "ext_cam")

            result_img = Image.fromarray((image).astype(np.uint8))
            result_img.save(f'{i}.jpg')
        
        # Check for quit
        if viewer.is_stopped():
            break

print("Simulation complete")