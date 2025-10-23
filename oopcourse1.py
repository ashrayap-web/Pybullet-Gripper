import pybullet as p
import pybullet_data
import time
import random
import math
import numpy as np


# ------------------- Setup ------------------- #
cid = p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.resetSimulation()
p.setGravity(0, 0, -10)
p.setRealTimeSimulation(0)
# Plane
p.loadURDF("plane.urdf")



class SimObject:

    def __init__(self, urdf_file=None, pos=None, orientation=None):


        #self.pos, _ = p.getBasePositionAndOrientation(self.body_id)
        self.pos = pos
        self.orientation = orientation

        self.body_id = p.loadURDF(urdf_file, basePosition=self.pos, baseOrientation=self.orientation)


class CubeObject(SimObject):
    def __init__(self, urdf_file="cube_small.urdf", pos=None, orientation=None):
        super().__init__(urdf_file, pos, orientation)


class SphereObject(SimObject):
    def __init__(self, urdf_file="sphere.urdf", pos=None, orientation=None):
        super().__init__(urdf_file, pos, orientation)


class SimGripper:

    def __init__(self, urdf_file, pos=None, orientation=None, target_obj=None):

        #self.pos = pos ## is needed???
        self.OBJ = target_obj

        self.head_offset = [0.2, 0, 0]

        if pos is None:
            self.start_pos = [random.uniform(-1, 1), random.uniform(-1, 1), random.uniform(0.3, 1)]
        else:
            self.start_pos = pos

        base_pos = [self.start_pos[i]-self.head_offset[i] for i in range(3)]
       
        if orientation is None:
            self.orientation = self.set_orientation()
        else:
            self.orientation = orientation

        self.body_id = p.loadURDF(urdf_file, basePosition=base_pos, baseOrientation=self.orientation)

        # Fix gripper with a constraint
        self.pr2_cid = p.createConstraint(
            parentBodyUniqueId=self.body_id,
            parentLinkIndex=-1,
            childBodyUniqueId=-1,
            childLinkIndex=-1,
            jointType=p.JOINT_FIXED,
            jointAxis=[0, 0, 0],
            parentFramePosition=[0.2,0,0],#self.head_offset, # 0.2 to change base pos to gripper head?
            childFramePosition= self.start_pos,
            parentFrameOrientation = [0,0,0,1],
            childFrameOrientation = self.orientation
        )

   
    def get_head_pos(self):
        
        base_pos, _ = p.getBasePositionAndOrientation(self.body_id)
        return np.array(base_pos) + np.array([0.2, 0, 0])

    def set_orientation(self):

        d_vector = np.array(self.OBJ.pos) - np.array(self.start_pos) # vector from gripper to obj
        dx, dy, dz = d_vector

        yaw = np.atan2(dy, dx)
        pitch = np.atan2(-dz, np.sqrt(dx**2 + dy**2))
        roll = 0

        return p.getQuaternionFromEuler([roll, pitch, yaw])


    # ------------------- Helper Functions ------------------- #
    def close_gripper(self):
        for joint in [0,2]:
            p.setJointMotorControl2(self.body_id, joint, p.POSITION_CONTROL, targetPosition=0.1, maxVelocity=1, force=10)

    def open_gripper(self):
        for joint in [0,2]:
            p.setJointMotorControl2(self.body_id, joint, p.POSITION_CONTROL, targetPosition=0.55, maxVelocity=1, force=10)

    def move_gripper(self, x, y, z, force=30):
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
        p.changeConstraint(
            self.pr2_cid,
            jointChildPivot=[x, y, z],
            jointChildFrameOrientation = self.orientation,
            maxForce=force
        )
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)


    def move_towards_obj(self):
        min_dist = 0.085
        speed = 0.005  

        # Get the final target position
        obj_pos_final, _ = p.getBasePositionAndOrientation(self.OBJ.body_id)
        obj_pos_final = np.array(obj_pos_final)
        
        initial_pos = self.get_head_pos()

        # Re-orient towards the object just before moving to ensure accuracy
        d_vector = obj_pos_final - initial_pos
        dx, dy, dz = d_vector
        yaw = np.atan2(dy, dx)
        pitch = np.atan2(-dz, np.sqrt(dx**2 + dy**2))
        self.orientation = p.getQuaternionFromEuler([0, pitch, yaw])

        # The final stopping point
        direction = d_vector / np.linalg.norm(d_vector)
        final_pos = obj_pos_final - min_dist * direction

        # The vector for the entire move
        move_vector = final_pos - initial_pos
        travel_distance = np.linalg.norm(move_vector)

        # Calculate the number of steps needed based on a constant speed
        num_steps = int(travel_distance / speed)
        if num_steps == 0:
            return 

        # Move in small increments of constant size
        for i in range(1, num_steps + 1):
            # Calculate the next intermediate position
            step_pos = initial_pos + (move_vector * (i / num_steps))
            
            self.move_gripper(step_pos[0], step_pos[1], step_pos[2], force=50)
            
            p.stepSimulation()
            time.sleep(1./240.)







#------ CUBE -----------#
cube_height = 0.05 
base_x, base_y = 0.0, 0.0

base_z = 0.025 + cube_height  
#cube_start_pos = [base_x, base_y, base_z]
cube_start_orientation = p.getQuaternionFromEuler([0, 0, 0])


CUBE = CubeObject(pos=[0,0,0.025], orientation=cube_start_orientation)


# ------------------- PR2 Gripper ------------------- #

gripper_start_pos = [-1.5, 3.5, 2.3]
#gripper_start_orientation = p.getQuaternionFromEuler([0, 0, 0])

pr2_gripper = SimGripper("pr2_gripper.urdf", pos=gripper_start_pos, target_obj=CUBE)



# Open fingers initially
joint_positions = [0.550569, 0.0, 0.549657, 0.0]
for i, pos in enumerate(joint_positions):
    p.resetJointState(pr2_gripper.body_id, i, pos)




# ------------------- Pick CUBE ------------------- #


# Open gripper
pr2_gripper.open_gripper()
for _ in range(150):
    p.stepSimulation()
    time.sleep(1./240.)

cube_pos, _ = p.getBasePositionAndOrientation(CUBE.body_id)
cube_x, cube_y, cube_z = cube_pos

#grasp_height = cube_z + 0.1

# The move function now handles its own simulation steps at a constant speed
pr2_gripper.move_towards_obj()


# Close gripper to grasp
pr2_gripper.close_gripper()
for _ in range(300):
    p.stepSimulation()
    time.sleep(1./240.)


# Lift cube
gripper_pos, _ = p.getBasePositionAndOrientation(pr2_gripper.body_id)
x, y, z = cube_pos
lift_height = z + 0.5

pr2_gripper.move_gripper(x, y, lift_height)
for _ in range(200):
    p.stepSimulation()
    time.sleep(1./240.)

'''
p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
p.changeConstraint(
    pr2_gripper.pr2_cid,
    jointChildPivot=[cube_x, cube_y, lift_height],
    jointChildFrameOrientation = pr2_gripper.orientation,
    maxForce=50
)
p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
'''

for _ in range(100):
    p.stepSimulation()
    time.sleep(1./240.)




# Keep GUI open
for _ in range(500):
    p.stepSimulation()
    time.sleep(1./240.)

p.disconnect()
