import pybullet as p
import pybullet_data
import time
import random
import math
import numpy as np
import pandas as pd


class SimObject:

    def __init__(self, name, urdf_file=None, pos=None, orientation=None):
        
        self.name = name
        self.pos = pos
        self.orientation = orientation

        self.body_id = p.loadURDF(urdf_file, basePosition=self.pos, baseOrientation=self.orientation)

        self.pos_grab_before = None
        self.pos_grab_after = None

class CubeObject(SimObject):
    def __init__(self, name, urdf_file="cube_small.urdf", pos=None, orientation=None):
        super().__init__(name, urdf_file, pos, orientation)


class SphereObject(SimObject):
    def __init__(self, name, urdf_file="sphere.urdf", pos=None, orientation=None):
        super().__init__(name, urdf_file, pos, orientation)



class SimGripper:

    def __init__(self, urdf_file, pos=None, orientation=None, target_obj=None):

        #self.pos = pos ## is needed???
        
        self.OBJ = target_obj

        if pos is None:
            self.start_pos = [random.uniform(-1, 1), random.uniform(-1, 1), random.uniform(0.4, 1)]
        else:
            self.start_pos = pos

        
        if orientation is None:
            self.orientation = self.set_orientation()
        else:
            self.orientation = orientation

        self.body_id = p.loadURDF(urdf_file, basePosition=self.start_pos, baseOrientation=self.orientation)

        # Fix gripper with a constraint
        self.pr2_cid = p.createConstraint(
            parentBodyUniqueId=self.body_id,
            parentLinkIndex=-1,
            childBodyUniqueId=-1,
            childLinkIndex=-1,
            jointType=p.JOINT_FIXED,
            jointAxis=[0, 0, 0],
            parentFramePosition=[0,0,0], 
            childFramePosition= self.start_pos,
            parentFrameOrientation = [0,0,0,1],
            childFrameOrientation = self.orientation
        )

        self.grab_start_pos = None
        self.grab_end_pos = None


    def set_orientation(self):

        obj_pos, _ = p.getBasePositionAndOrientation(self.OBJ.body_id)

        d_vector = np.array(obj_pos) - np.array(self.start_pos) # vector from gripper to obj
        dx, dy, dz = d_vector

        
        if np.abs(dx) and np.abs(dy) < 1e-3:
            roll = -np.pi/2
            yaw = 0
            pitch = np.pi/2

        else:
            yaw = np.atan2(dy, dx)
            pitch = np.atan2(-dz, np.sqrt(dx**2 + dy**2))
            roll = 0

        yaw += random.uniform(-np.pi/36, np.pi/36) # 5deg
        pitch += random.uniform(-np.pi/36, np.pi/36)

        #print(f"roll {roll}, pitch {pitch}, yaw {yaw}")

        return p.getQuaternionFromEuler([roll, pitch, yaw])


    # ------------------- Helper Functions ------------------- #
    def close_gripper(self):
        for joint in [0,2]:
            p.setJointMotorControl2(self.body_id, joint, p.POSITION_CONTROL, targetPosition=0.1, maxVelocity=2, force=10)

    def open_gripper(self):
        for joint in [0,2]:
            p.setJointMotorControl2(self.body_id, joint, p.POSITION_CONTROL, targetPosition=0.95, maxVelocity=1, force=10)


    def move_gripper(self, x, y, z, force=80):
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
        p.changeConstraint(
            self.pr2_cid,
            jointChildPivot=[x, y, z],
            jointChildFrameOrientation = self.orientation,
            maxForce=force
        )
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)


    def move_towards_obj(self):
        
        min_dist = 0.30 
        z_offset = 0.005 #0.01

        #obj_pos = np.array(self.OBJ.pos)
        obj_pos, _ = p.getBasePositionAndOrientation(self.OBJ.body_id)
        obj_pos = np.array(obj_pos)
        obj_pos[2] += z_offset

        curr_pos, _ = p.getBasePositionAndOrientation(self.body_id)

        #dist = np.linalg.norm(obj_pos - np.array(curr_pos))

        d_vector = obj_pos - curr_pos
        d_unit_vector = d_vector / np.linalg.norm(d_vector)

        pos_step = obj_pos - min_dist *(d_unit_vector)  # direction vector

        self.move_gripper(pos_step[0], pos_step[1], pos_step[2], force=1000)


    def is_success(self):
        start_pos_cube = np.array(self.OBJ.pos_grab_before)
        end_pos_cube = np.array(self.OBJ.pos_grab_after)

        dist_diff_vec = (np.array(self.grab_end_pos) - np.array(self.grab_start_pos)) - (end_pos_cube - start_pos_cube)
        #dist_diff = np.linalg.norm(dist_diff_vec)
        dist_diff = dist_diff_vec[2]  # height

        if dist_diff < 0.01:
            print(f"\033[32m{self.OBJ.name}: SUCCESS\033[0m")
            return 1.0
        else:
            print(f"\033[31m{self.OBJ.name}: FAILURE\033[0m")
            return 0.0


## ------ UTILITY ------##
'''
def spawn_marker(position, scale=0.01, color=[0, 0, 0, 1]):
    if position:
        visual_shape = p.createVisualShape(
            shapeType=p.GEOM_SPHERE,
            radius=scale,
            rgbaColor=color
        )
        p.createMultiBody(baseVisualShapeIndex=visual_shape, basePosition=position)
'''

count1 = 0
count0 = 0
if __name__ == "__main__":

    # ------------------- Setup ------------------- #
    cid = p.connect(p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    
    
    # camera settings (harmless in DIRECT; kept for compatibility)
    p.resetDebugVisualizerCamera(
        cameraDistance=1.0,
        cameraYaw=50,
        cameraPitch=-35,
        cameraTargetPosition=[0, 0, 0.2]
    )

    # TEST - improve solver
    p.setPhysicsEngineParameter(numSolverIterations=300, erp=0.3, contactERP=0.3)

    # Performance tweak: run in fixed (non-realtime) mode and set a timestep.
    # Removing sleeps and running stepSimulation in tight loops makes the
    # simulation run as fast as possible in DIRECT mode.
    p.setRealTimeSimulation(0)
    try:
        p.setTimeStep(1.0/240.0)
    except Exception:
        # Older/newer pybullet versions may not expose setTimeStep; ignore if unavailable
        pass


    n = 10000
    data = np.empty((120,8)) # 7 features, 1 output


    for i in range(n):

        p.resetSimulation()
        p.setGravity(0, 0, -10)
        # Plane
        p.loadURDF("plane.urdf")

        #------ CUBE -----------#
        cube_height = 0.05  # half-height

        cube_start_orientation = p.getQuaternionFromEuler([0, 0, 0])


        CUBE = CubeObject(f"Cube{i+1}", pos=[0.0,0.0,0.5], orientation=cube_start_orientation)

        # let cube fall to place (no sleeps -> runs at CPU speed)
        for _ in range(100):
            p.stepSimulation()

        ## CHECK CUBE POS
        cube_pos, _ = p.getBasePositionAndOrientation(CUBE.body_id)
        #print(f"CUBE POS: {cube_pos}")
        #spawn_marker(cube_pos)


        # ------------------- PR2 Gripper ------------------- #

        # error with this pos??
        #gripper_start_pos = [0.5, 0.1, 0.05]  # always give z >= 0.04
        #gripper_start_pos = [0, 0.0, 0.5]
        #gripper_start_orientation = p.getQuaternionFromEuler([0, 0, 0])

        pr2_gripper = SimGripper("pr2_gripper.urdf", pos=None, target_obj=CUBE)

        # Open fingers initially
        joint_positions = [0.550569, 0.0, 0.549657, 0.0]
        for joint_idx, pos in enumerate(joint_positions):
            p.resetJointState(pr2_gripper.body_id, joint_idx, pos)

        # ------------------- Pick CUBE ------------------- #

        # Open gripper
        pr2_gripper.open_gripper()
        for _ in range(50):
            p.stepSimulation()


        pr2_gripper.move_towards_obj()
        for _ in range(250):
            p.stepSimulation()


        # Close gripper to grasp
        pr2_gripper.close_gripper()
        for _ in range(150):
            p.stepSimulation()

        CUBE.pos_grab_before, _ = p.getBasePositionAndOrientation(CUBE.body_id)

        # Lift cube
        pr2_gripper.grab_start_pos, _ = p.getBasePositionAndOrientation(pr2_gripper.body_id)
        x, y, z = pr2_gripper.grab_start_pos
        #print(x,y,z)
        lift_height = z + 0.3

        pr2_gripper.move_gripper(x, y, lift_height)
        for _ in range(200):
            p.stepSimulation()

        CUBE.pos_grab_after, _ = p.getBasePositionAndOrientation(CUBE.body_id)
        pr2_gripper.grab_end_pos, _ = p.getBasePositionAndOrientation(pr2_gripper.body_id)

        for _ in range(100):
            p.stepSimulation()

        # print success or fail
        result = pr2_gripper.is_success()
        if result == 1 and count1<60:
            count1 += 1
            data[(count1+count0-1),:] = np.hstack([pr2_gripper.start_pos, pr2_gripper.orientation, result])
        elif result ==0 and count0<60:
            count0 += 1
            data[(count1+count0-1),:] = np.hstack([pr2_gripper.start_pos, pr2_gripper.orientation, result])
        elif count1==60 and count0==60:
            break
            
        
        ### ADD POSE TO DATASET
        

        # remove gripper and cube from world
        #p.removeBody(pr2_gripper.body_id)
        #p.removeBody(CUBE.body_id)

        # Keep GUI open
        # for _ in range(50):
        #     p.stepSimulation()
        #     time.sleep(1./240.)


    # end
    p.disconnect()


    cols = ["x","y","z","qx","qy","qz","qw","Result"]

    df = pd.DataFrame(data, columns=cols)

    print(df)