initial_prompt = f""" I am using a Franka FR3 robot arm with the following actuators:
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

initial_prompt_no_d435 = f""" I am using a Franka FR3 robot arm with the following actuators:
    Actuator 1 - Shoulder Joint Rotation around the Base: Range -2.7 to 2.7 radians,
    Actuator 2 - Shoulder Joint Lifting Up and Down: Range -1.6 to 1.6 radians,
    Actuator 3 - Elbow Joint Rotation of the Upper Arm: Range -2.9 to 2.9 radians,
    Actuator 4 - Elbow Joint Lifting Forearm Up and Down: Range -3.0 to -0.15 radians,
    Actuator 5 - Wrist Joint Rotation of Forearm: Range -2.8 to 2.8 radians,
    Actuator 6 - Wrist Joint Flexion/Extension: Range 0.54 to 4.52 radians,
    Actuator 7 - Wrist Joint Rotation of End Effector: Range -3.02 to 3.02 radians,
    Actuator 8 - Combined Finger Actuator: Range 0 to 255 (0 is fully closed, 255 is fully open),
    The current values of all actuators are  {data.ctrl}. Relate these actuators with the image I have provided.
    The image consists of an external view of the robot arm. The goal of this task is to grasp the red cube.
    Everytime I query you from now on, you need to analyze the current locations of all the joints and provide me with an incremental movement to achieve the end goal.
    I want you to provide a single, simple statement stating which joint(only one) you would actuate for the incremental motion. Also analyze the range of values and 
    the current value of the actuators and give me a floating point value for each motion. Keep in mind I want an incremental motion, so it is still okay if you actuate only one joint. The output should be in the format [actuator number]:[predicted value] separated 
    by commas if there are multiple joints to be actuated. Just give me this, no other text.
    Also, when you give an incremental actuation, make sure that the change in actuation value should be lesser than 20%.
    """