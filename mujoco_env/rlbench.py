import mujoco as mj
import glfw
import numpy as np
import time
from OpenGL.GL import *

# Load the MuJoCo model from an XML file
model = mj.MjModel.from_xml_path('/home/nayaka/software/mujoco_menagerie/franka_fr3/fr3.xml')

# Create a simulation instance
data = mj.MjData(model)

# Set up a GLFW window
glfw.init()
window = glfw.create_window(1200, 900, "MuJoCo Simulation", None, None)
glfw.make_context_current(window)

# Create a MuJoCo scene and context
scene = mj.MjvScene(model, maxgeom=1000)
context = mj.MjrContext(model, mj.mjtFontScale.mjFONTSCALE_150)

# Initialize visualization options and camera
opt = mj.MjvOption()
cam = mj.MjvCamera()

# Main simulation loop
while not glfw.window_should_close(window):
    mj.mj_step(model, data)

    # Clear the background to white
    glClearColor(1.0, 1.0, 1.0, 1.0)  # RGBA for white
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

    
    # Update the scene
    mj.mjv_updateScene(model, data, opt, None, cam, mj.mjtCatBit.mjCAT_ALL, scene)

    # Render the scene
    mj.mjr_render(mj.MjrRect(0, 0, 1200, 900), scene, context)

    # Swap buffers and poll for events
    glfw.swap_buffers(window)
    glfw.poll_events()

# Clean up
glfw.terminate()