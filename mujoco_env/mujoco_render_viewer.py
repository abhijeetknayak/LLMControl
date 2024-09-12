import mujoco
import mujoco.viewer
import pdb
import threading
from PIL import Image
import numpy as np
import sys
import time
from config.file_config import MUJOCO_MODELS_LOC
import yaml
from easydict import EasyDict as edict


def load_config(config_path):
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    return edict(config)


def load_model(xml_path):

    # Load the model
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)

    return model, data


def run_viewer(model, data):

    # Create a MuJoCo Viewer
    mujoco.viewer.launch_passive(model, data)


def render_image(model, data, params):

    # Load parameters from params
    debug = params.debug
    height = params.cameras.height
    width = params.cameras.width
    camera_names = params.cameras.cam_names

    cam_images = None

    with mujoco.Renderer(model, height, width) as renderer:

        for i in range(len(camera_names)):
            renderer.update_scene(data, camera=camera_names[i])
            img = renderer.render()

            if cam_images is None:
                cam_images = img
            else:
                cam_images = np.hstack((cam_images, np.zeros((cam_images.shape[0], 10, 3))))
                cam_images = np.hstack((cam_images, img))

        if debug:
            result = Image.fromarray((cam_images).astype(np.uint8))

            result.save(f'debug/cam_images.jpg')

        return cam_images


if __name__ == "__main__":

    config_path = 'config/render_config.yaml'
    params = load_config(config_path)

    # Path to your main XML file (which includes the robot and the scene)
    xml_path = f'{MUJOCO_MODELS_LOC}/franka_fr3/scene.xml'

    # Mujoco model
    model, data = load_model(xml_path)



    with mujoco.viewer.launch_passive(model, data) as viewer:
        # viewer.user_scn.flags[mujoco.mjtRndFlag.mjRND_WIREFRAME] = 1
        viewer.sync()

        while viewer.is_running():
            step_start = time.time()

            while time.time() - step_start < (1.0 / 1000):
                mujoco.mj_step(model, data)

            viewer.sync()

            cur_image = render_image(model, data, params=params)

            # Update control parameters like this! TODO

            for i in range(20):
                pdb.set_trace()
                data.ctrl[0] += 0.1
                print(data.ctrl)
                
                mujoco.mj_step(model, data)
                viewer.sync()

