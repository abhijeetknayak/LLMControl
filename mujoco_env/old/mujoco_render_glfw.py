import mujoco
import numpy as np
from mujoco import MjData, MjModel
import glfw
import sys


# Load the MuJoCo model from the XML file
xml_path = '/home/nayaka/software/mujoco_menagerie/franka_fr3/scene.xml'
model = MjModel.from_xml_path(xml_path)

# Create a simulation state (data) based on the model
data = MjData(model)

# Initialize a GLFW window
if not glfw.init():
    sys.exit("Failed to initialize GLFW")

window = glfw.create_window(800, 600, "MuJoCo Simulation", None, None)
glfw.make_context_current(window)

# Initialize visualization components
scene = mujoco.MjvScene(model, maxgeom=1000)
cam = mujoco.MjvCamera()
option = mujoco.MjvOption()
context = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_100)

# Set the default camera position
cam.azimuth = 90.0
cam.elevation = -30.0
cam.distance = 5.0
cam.lookat = np.array([0.0, 0.0, 1.0])

# Variables to keep track of the mouse state
last_x, last_y = 0, 0
left_button_pressed = False
middle_button_pressed = False

# Callback function for mouse button events
def mouse_button_callback(window, button, action, mods):
    global left_button_pressed, middle_button_pressed
    if button == glfw.MOUSE_BUTTON_LEFT:
        left_button_pressed = (action == glfw.PRESS)
    elif button == glfw.MOUSE_BUTTON_MIDDLE:
        middle_button_pressed = (action == glfw.PRESS)

# Callback function for mouse motion
def cursor_position_callback(window, xpos, ypos):
    global last_x, last_y
    dx = xpos - last_x
    dy = ypos - last_y

    # Check if Alt or Ctrl is held down for panning
    if glfw.get_key(window, glfw.KEY_LEFT_ALT) == glfw.PRESS or glfw.get_key(window, glfw.KEY_RIGHT_ALT) == glfw.PRESS:
        # Pan camera horizontally and vertically
        cam.lookat[0] -= 0.01 * dx
        cam.lookat[1] += 0.01 * dy
    elif glfw.get_key(window, glfw.KEY_LEFT_CONTROL) == glfw.PRESS or glfw.get_key(window, glfw.KEY_RIGHT_CONTROL) == glfw.PRESS:
        # Pan camera vertically (Z-axis)
        cam.lookat[2] += 0.01 * dy
    elif left_button_pressed:
        # Rotate camera
        cam.azimuth -= 0.4 * dx
        cam.elevation += 0.4 * dy

    last_x, last_y = xpos, ypos
# Callback function for scroll events (zooming)
def scroll_callback(window, xoffset, yoffset):
    cam.distance *= 1.0 - 0.05 * yoffset

# Set the callbacks for mouse input
glfw.set_mouse_button_callback(window, mouse_button_callback)
glfw.set_cursor_pos_callback(window, cursor_position_callback)
glfw.set_scroll_callback(window, scroll_callback)

# Main simulation loop
while not glfw.window_should_close(window):
    # Step the simulation
    mujoco.mj_step(model, data)

    # Update the scene and render it
    mujoco.mjv_updateScene(model, data, option, None, cam, mujoco.mjtCatBit.mjCAT_ALL, scene)
    mujoco.mjr_render(mujoco.MjrRect(0, 0, 800, 600), scene, context)

    # Swap buffers and poll for events
    glfw.swap_buffers(window)
    glfw.poll_events()

# Clean up
glfw.terminate()