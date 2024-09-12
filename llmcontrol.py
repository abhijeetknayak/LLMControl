import openai
import io
import re
import logging
from easydict import EasyDict as edict

from config.file_config import API_KEY

from mujoco_env.mujoco_render_viewer import *
from lang.vqa import *

logging.basicConfig(level=logging.INFO)

def initialize_content(data, image):

    prompt = f""" I am using a Franka FR3 robot arm with the following actuators:
    Actuator 1 - Shoulder Joint Rotation around the Base: Range -2.7 to 2.7 radians,
    Actuator 2 - Shoulder Joint Lifting Up and Down: Range -1.6 to 1.6 radians,
    Actuator 3 - Elbow Joint Rotation of the Upper Arm: Range -2.9 to 2.9 radians,
    Actuator 4 - Elbow Joint Lifting Forearm Up and Down: Range -3.0 to -0.15 radians,
    Actuator 5 - Wrist Joint Rotation of Forearm: Range -2.8 to 2.8 radians,
    Actuator 6 - Wrist Joint Flexion/Extension: Range 0.54 to 4.52 radians,
    Actuator 7 - Wrist Joint Rotation of End Effector: Range -3.02 to 3.02 radians,
    Actuator 8 - Combined Finger Actuator: Range 0 to 255 (0 is fully closed, 255 is fully open),
    The current values of all actuators are  {data.ctrl}. Relate these actuators with the image I have provided.
    The image consists of an external view on the left and the eye-in-hand camera image on the right. The goal of this task is to grasp the red cube.
    Everytime I query you from now on, you need to analyze the current locations of all the joints and provide me with an incremental movement to achieve the end goal.
    I want you to provide a single, simple statement stating which joint(only one) you would actuate for the incremental motion. Also analyze the range of values and 
    the current value of the actuators and give me a floating point value for each motion. Keep in mind I want an incremental motion, so it is still okay if you actuate only one joint. The output should be in the format [actuator number]:[predicted value] separated 
    by commas if there are multiple joints to be actuated. Just give me this, no other text.
    Also, when you give an incremental actuation, make sure that the change in actuation value should be lesser than 20%.
    """

    # image_path = "/home/nayaka/Desktop/LLMControl/start_img.jpg"

    response = send_request(prompt, image_data=image)

    return response


def encode_image_for_openai(im_array):
    image = Image.fromarray(im_array.astype(np.uint8))

    # Save image to a buffer
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    buffer.seek(0)

    # Encode image in Base64
    img_base64 = base64.b64encode(buffer.read()).decode('utf-8')

    return img_base64


def decode_api_response(response):
    actions = {}
    response_json = edict(response.json())

    content = response_json.choices[0].message.content

    pattern = r'(\d+):\s*([-\d.]+)'

    matches = re.findall(pattern, content)

    for match in matches:
        actuator_num, actuator_value = int(match[0]), float(match[1])
        actions[actuator_num - 1] = actuator_value

    return actions


def apply_action(model, data, viewer, action_dict):
    action_indices = []
    for key, value in action_dict.items():
        data.ctrl[key] = value
        action_indices.append(key)

    # Now that data.ctrl is updated, apply mujoco steps to update state
    qpos_target = data.ctrl[action_indices]

    num_steps = 0
    
    while np.abs(np.sum(qpos_target - data.qpos[action_indices])):
        # Apply the step until the change is minimal
        mujoco.mj_step(model, data)
        viewer.sync()

        num_steps += 1

        if num_steps > 500:
            break

    logging.info(f"Applied action and reached a steady state or step limit: Num of steps: {num_steps}")

    return

def run_robot_control_loop():

    INITIALIZED = False


    config_path = 'config/render_config.yaml'
    params = load_config(config_path)

    # Path to your main XML file (which includes the robot and the scene)
    xml_path = f'{MUJOCO_MODELS_LOC}/franka_fr3/scene.xml'

    # Mujoco model
    model, data = load_model(xml_path)

    # Get the image from the env
    with mujoco.viewer.launch_passive(model, data) as viewer:
        # viewer.user_scn.flags[mujoco.mjtRndFlag.mjRND_WIREFRAME] = 1
        viewer.sync()

        while viewer.is_running():
            if not INITIALIZED:
                logging.info("Initializing environment and robot")
                init_start = time.time()
                while time.time() - init_start < 2.0:
                    mujoco.mj_step(model, data)

                viewer.sync()

                cur_image = render_image(model, data, params=params)
                cur_image_base64 = encode_image_for_openai(cur_image)

                initial_response = initialize_content(data, cur_image_base64)
                initial_action = decode_api_response(initial_response)

                logging.info(f"Action computed: {initial_action}")

                # Apply initial action
                apply_action(model, data, viewer, initial_action)
                viewer.sync()
                
                INITIALIZED = True

            step_start = time.time()

            while time.time() - step_start < (1.0 / 1000):
                mujoco.mj_step(model, data)

            viewer.sync()

            cur_image = render_image(model, data, params=params)
            base64_img = encode_image_for_openai(cur_image)

            text_prompt = f""" I am at the current state as shown in the image. My goal still is to grasp the red cube.
            The current joint actuation values are {data.qpos[0:8]}. What will be joint that I should actuate for an incremental action to grasp the cube?
            Analyze the stitched images and reason about which joint movement could bring the end-effctor closer to the red cube.
            Always use the output format mentioned before! And always provide incremental actuations! Do not repeat the last actuation you asked me to apply."""

            response = send_request(text_info=text_prompt, image_data=base64_img)
            new_action = decode_api_response(response)

            logging.info(f"Action computed: {initial_action}")      

            pdb.set_trace()      

            # Apply action
            apply_action(model, data, viewer, new_action)
            viewer.sync()


if __name__ == "__main__":

    # Run Main loop
    run_robot_control_loop()

