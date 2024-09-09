import numpy as np
from dm_control import mjcf
from dm_control import viewer

# Load your existing model XML file
model_path = '/home/nayaka/software/mujoco_menagerie/franka_fr3/scene.xml'
model = mjcf.from_path(model_path)

# Add a dynamic object to the model
body = model.worldbody.add(
    mjcf.element('body', name='dynamic_box', pos=[0, 0, 1])
)
body.add(
    mjcf.element('geom', type='box', size=[0.1, 0.1, 0.1], rgba=[0, 1, 0, 1])
)
body.add(
    mjcf.element('joint', type='free', name='box_joint')
)

# Optionally, save the modified model
modified_model_path = 'modified_robot_model.xml'
model.to_file(modified_model_path)

# Load the model using dm_control
env = mjcf.from_path(modified_model_path)

# Create a simulation
sim = env.physics

# Create a viewer for rendering
viewer.launch(env, camera_id=0, azimuth=90.0, elevation=30.0, distance=5.0)

# Run a simple simulation loop
for _ in range(1000):
    sim.step()
    # Access the position of the dynamic object
    pos = sim.data.get_body_xpos("dynamic_box")
    print(f"Dynamic Box Position: {pos}")
