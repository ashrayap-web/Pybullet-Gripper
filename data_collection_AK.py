import pybullet as p
import pybullet_data
import time
import random
import math
import numpy as np
import pandas as pd

from abc import ABC, abstractmethod


class SimObject:

    def __init__(self, name, urdf_file=None, pos=None, orientation=None, scale=1.0):
       
        self.name = name
        self.pos = pos
        self.orientation = orientation

        self.body_id = p.loadURDF(urdf_file, basePosition=self.pos, baseOrientation=self.orientation, globalScaling=scale)

        self.pos_grab_before = None
        self.pos_grab_after = None


class CubeObject(SimObject):
    def __init__(self, name, urdf_file="cube_small.urdf", pos=None, orientation=None, scale=1.0):
        super().__init__(name, urdf_file, pos, orientation, scale)


class SphereObject(SimObject):
    def __init__(self, name, urdf_file="sphere.urdf", pos=None, orientation=None, scale=1.0):
        super().__init__(name, urdf_file, pos, orientation, scale)


class CylinderObject(SimObject):
    def __init__(self, name, urdf_file="cylinder.urdf", pos=None, orientation=None, scale=0.55):
        super().__init__(name, urdf_file, pos, orientation, scale)

        p.changeDynamics(self.body_id, -1, lateralFriction=1.0)



class SimGripper(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def set_orientation(self):
        pass
    
    @abstractmethod
    def close_gripper(self):
        pass
    
    @abstractmethod
    def open_gripper(self):
        pass

    @abstractmethod
    def move_gripper(self, x, y, z, force=80):
        pass

    @abstractmethod
    def move_towards_obj(self):
       pass

    @abstractmethod
    def is_success(self):
        pass


class PR2Gripper(SimGripper):

    def __init__(self, urdf_file, pos=None, orientation=None, target_obj=None):

        #self.pos = pos ## is needed???
       
        self.OBJ = target_obj

        self.start_pos = np.array(pos if pos is not None else
                            [random.uniform(-1, 1), 
                            random.uniform(-1, 1), 
                            random.uniform(0.6, 1)
                            ], dtype=float)
        

        self.orientation = np.array(orientation if orientation is not None else
                                    self.set_orientation(), dtype=float)
        

        self.body_id = p.loadURDF(urdf_file, basePosition=self.start_pos.tolist(), baseOrientation=self.orientation.tolist())

        # Fix gripper with a constraint
        self.pr2_cid = p.createConstraint(
            parentBodyUniqueId=self.body_id,
            parentLinkIndex=-1,
            childBodyUniqueId=-1,
            childLinkIndex=-1,
            jointType=p.JOINT_FIXED,
            jointAxis=[0, 0, 0],
            parentFramePosition=[0,0,0],
            childFramePosition= self.start_pos.tolist(),
            parentFrameOrientation = [0,0,0,1],
            childFrameOrientation = self.orientation.tolist()
        )

        self.grab_start_pos = None
        self.grab_end_pos = None


    def set_orientation(self):

        obj_pos, _ = p.getBasePositionAndOrientation(self.OBJ.body_id)

        d_vector = np.array(obj_pos) - self.start_pos # vector from gripper to obj
        dx, dy, dz = d_vector

        tolerance = 1e-3
        if np.abs(dx) < tolerance and np.abs(dy) < tolerance:
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

        if isinstance(self.OBJ, CubeObject):
            f = 10
        elif isinstance(self.OBJ, CylinderObject):
            f = 10

        for joint in [0,2]:
            p.setJointMotorControl2(self.body_id, joint, p.POSITION_CONTROL, targetPosition=0.1, maxVelocity=2, force=f)

    def open_gripper(self):
        for joint in [0,2]:
            p.setJointMotorControl2(self.body_id, joint, p.POSITION_CONTROL, targetPosition=0.95, maxVelocity=2, force=10)


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

        obj_pos, _ = p.getBasePositionAndOrientation(self.OBJ.body_id)
        obj_pos = np.array(obj_pos)
        obj_pos[2] += z_offset

        curr_pos, _ = p.getBasePositionAndOrientation(self.body_id)

        d_vector = obj_pos - np.array(curr_pos)
        d_unit_vector = d_vector / np.linalg.norm(d_vector)

        pos_step = obj_pos - min_dist *(d_unit_vector)  # direction vector

        self.move_gripper(pos_step[0], pos_step[1], pos_step[2], force=1000)


    def is_success(self):
        start_pos_obj = np.array(self.OBJ.pos_grab_before)
        end_pos_obj = np.array(self.OBJ.pos_grab_after)

        dist_diff_vec = (np.array(self.grab_end_pos) - np.array(self.grab_start_pos)) - (end_pos_obj - start_pos_obj)
        #dist_diff = np.linalg.norm(dist_diff_vec)
        dist_diff = dist_diff_vec[2]  # height

        _, angular_vel = p.getBaseVelocity(self.OBJ.body_id)

        if dist_diff < 0.01 and np.linalg.norm(angular_vel) < 0.15:
            print(f"\033[32m{self.OBJ.name}: SUCCESS\033[0m")
            return 1.0
        else:
            print(f"\033[31m{self.OBJ.name}: FAILURE\033[0m")
            return 0.0


#----------SIM LOOP--------------#

if __name__ == "__main__":

    COLLECT_DATA = False

    # ------------------- Setup ------------------- #
    cid = p.connect(p.GUI)
    #cid = p.connect(p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
   

    # camera settings
    p.resetDebugVisualizerCamera(
        cameraDistance=1.0,
        cameraYaw=50,
        cameraPitch=-35,
        cameraTargetPosition=[0, 0, 0.2]
    )

    # TEST - improve solver
    p.setPhysicsEngineParameter(numSolverIterations=300, erp=0.3, contactERP=0.3)


    n = 10000
    data = np.empty((n,8)) # 6 features (7 if quaternions), + 1 output
    # disclude roll since mostly 0?

    for i in range(n):

        p.resetSimulation()
        p.setGravity(0, 0, -10)
        p.setRealTimeSimulation(0)

        # Plane
        p.loadURDF("plane.urdf")

        #------ CUBE -----------#
        #cube_start_orientation = p.getQuaternionFromEuler([0, 0, 0])
        #cube_z = 0.03
                                             # cube min height
        #CUBE = CubeObject(f"Cube{i+1}", pos=[0.0,0.0,cube_z], orientation=cube_start_orientation)


        #------ Cylinder -----------#

        cylinder_start_orientation = p.getQuaternionFromEuler([0, 0, 0])
        cylinder_z = 0.06
                                             # cube min height
        CYLINDER = CylinderObject(f"Cylinder{i+1}", pos=[0.0,0.0, cylinder_z], orientation=cylinder_start_orientation)

        # ------- SET CURRENT OBJECT --------#
        current_obj = CYLINDER

        # let object fall to place
        for _ in range(50):
            p.stepSimulation()
            time.sleep(1./240.)

        ## CHECK CUBE POS
        obj_pos, _ = p.getBasePositionAndOrientation(current_obj.body_id)


        # ------------------- PR2 Gripper ------------------- #

        #pos0 = [-0.5, 0.0, 0.09]  # always give z >= 0.04
        #o = p.getQuaternionFromEuler([0, 0, 0])

        curr_gripper = PR2Gripper("pr2_gripper.urdf", pos=None, orientation=None, target_obj=current_obj)

        # Open fingers initially
        joint_positions = [0.550569, 0.0, 0.549657, 0.0]
        for joint_idx, pos in enumerate(joint_positions):
            p.resetJointState(curr_gripper.body_id, joint_idx, pos)

        # ------------------- Pick CUBE ------------------- #

        # Open gripper
        curr_gripper.open_gripper()
        for _ in range(50):
            p.stepSimulation()
            time.sleep(1./240.)


        curr_gripper.move_towards_obj()
        for _ in range(50):
            p.stepSimulation()
            time.sleep(1./240.)


        # Close gripper to grasp
        curr_gripper.close_gripper()
        for _ in range(150):
            p.stepSimulation()
            time.sleep(1./240.)

        current_obj.pos_grab_before, _ = p.getBasePositionAndOrientation(current_obj.body_id)

        # Lift cube
        curr_gripper.grab_start_pos, _ = p.getBasePositionAndOrientation(curr_gripper.body_id)
        x, y, z = curr_gripper.grab_start_pos
        #print(x,y,z)
        lift_height = z + 0.3

        curr_gripper.move_gripper(x, y, lift_height)
        for _ in range(50):
            p.stepSimulation()
            time.sleep(1./240.)

        # wait 3s - 720
        for _ in range(120):
            p.stepSimulation()
            time.sleep(1./240.)

        current_obj.pos_grab_after, _ = p.getBasePositionAndOrientation(current_obj.body_id)
        curr_gripper.grab_end_pos, _ = p.getBasePositionAndOrientation(curr_gripper.body_id)

        # print success or fail
        result = curr_gripper.is_success()

        ### ADD POSE TO DATASET
        #euler_angles = p.getEulerFromQuaternion(curr_gripper.orientation)
        data[i,:] = np.hstack([curr_gripper.start_pos, curr_gripper.orientation, result]) #euler_angles


        # Keep GUI open
        for _ in range(50):
            p.stepSimulation()
            time.sleep(1./240.)


    # end
    p.disconnect()


    cols = ["x","y","z","qx","qy","qz","qw","Result"]
    #cols = ["x","y","z","roll","pitch","yaw","Result"]

    df = pd.DataFrame(data, columns=cols)
    print(df.head())

    print(df["Result"].value_counts())

    if COLLECT_DATA:
        df.to_csv("poses_dataset_euler.csv", index=False)
