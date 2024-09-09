import mujoco
import mujoco.viewer
import pdb
import threading
from PIL import Image
import numpy as np
import sys
import time

# Path to your main XML file (which includes the robot and the scene)
xml_path = '/home/nayaka/software/mujoco_menagerie/franka_fr3/scene.xml'


# Load the model
model = mujoco.MjModel.from_xml_path(xml_path)
data = mujoco.MjData(model)

def run_viewer(model, data):

    # Create a MuJoCo Viewer
    mujoco.viewer.launch_passive(model, data)

def render_image(model, data, params):
    debug = params['debug']
    height = params['height']
    width = params['width']
    camera_names = params['cam_names']

    cam_images = None

    with mujoco.Renderer(model, height, width) as renderer:

        for i in range(len(camera_names)):
            renderer.update_scene(data, camera=camera_names[i])
            img = renderer.render()

            if cam_images is None:
                pdb.set_trace()
                cam_images = img
            else:
                # pdb.set_trace()
                cam_images = np.hstack((cam_images, np.zeros((cam_images.shape[0], 10, 3))))
                cam_images = np.hstack((cam_images, img))


        if debug:
            result = Image.fromarray((cam_images).astype(np.uint8))

            result.save(f'cam_images_{img_idx}.jpg')

        return cam_images


params = {'debug': False, 'height': 720, 'width': 720, 'cam_names': ['ext_cam', 'd435i_camera']} #TODO Separate file

img_idx = 0

with mujoco.viewer.launch_passive(model, data) as viewer:
    # viewer.user_scn.flags[mujoco.mjtRndFlag.mjRND_WIREFRAME] = 1
    viewer.sync()


    while viewer.is_running():
        step_start = time.time()


        img_idx += 1


        while time.time() - step_start < (1.0 / 1000):
            mujoco.mj_step(model, data)

        viewer.sync()

        cur_image = render_image(model, data, params=params)



        data.ctrl[-1] = 255
