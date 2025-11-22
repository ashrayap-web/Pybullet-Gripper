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

class CylinderObject(SimObject):
    def __init__(self, name, urdf_file="cylinder.urdf", pos=None, orientation=None, scale=0.55):
        super().__init__(name, urdf_file, pos, orientation, scale)
        p.changeDynamics(self.body_id, -1, lateralFriction=1.0)

# ---------------------------------------------------------
# GRIPPER INTERFACE
# ---------------------------------------------------------

class SimGripper(ABC):
    def __init__(self, urdf_file, pos=None, orientation=None, target_obj=None):
        self.OBJ = target_obj
        self.start_pos = np.array(pos if pos is not None else [random.uniform(-1, 1), random.uniform(-1, 1), random.uniform(0.6, 1)], dtype=float)
        self.grab_start_pos = None
        self.grab_end_pos = None
    @abstractmethod
    def set_orientation(self):
        pass
    
    @abstractmethod
    def open_gripper(self):
        pass
    
    @abstractmethod
    def close_gripper(self):
        pass
    
    @abstractmethod
    def move_gripper(self, x, y, z, force=80):
        pass
    @abstractmethod
    def move_towards_obj(self):
        pass
    def set_orientation(self):
        obj_pos, _ = p.getBasePositionAndOrientation(self.OBJ.body_id)
        d_vector = np.array(obj_pos) - self.start_pos 
        dx, dy, dz = d_vector
        if np.abs(dx) < 1e-3 and np.abs(dy) < 1e-3:
            roll, yaw, pitch = -np.pi/2, 0, np.pi/2
        else:
            pitch  = np.atan2(-dz, np.sqrt(dx**2 + dy**2)) 
            yaw = np.atan2(dy, dx)
            roll = 0
            
        yaw += random.uniform(-np.pi/36, np.pi/36) 
        pitch += random.uniform(-np.pi/36, np.pi/36)
        return p.getQuaternionFromEuler([roll, pitch, yaw])
    def is_success(self):
        start_pos_obj = np.array(self.OBJ.pos_grab_before)
        end_pos_obj = np.array(self.OBJ.pos_grab_after)
        dist_diff_vec = (np.array(self.grab_end_pos) - np.array(self.grab_start_pos)) - (end_pos_obj - start_pos_obj)
        _, angular_vel = p.getBaseVelocity(self.OBJ.body_id)
        if dist_diff_vec[2] < 0.01 and np.linalg.norm(angular_vel) < 0.15:
            print(f"\033[32m{self.OBJ.name}: SUCCESS\033[0m")
            return 1.0
        else:
            print(f"\033[31m{self.OBJ.name}: FAILURE\033[0m")
            return 0.0


# ---------------------------------------------------------
# PR2 GRIPPER (two finger)
# ---------------------------------------------------------
class PR2Gripper(SimGripper):
    def __init__(self, urdf_file, pos=None, orientation=None, target_obj=None):
        super().__init__(urdf_file, pos, orientation, target_obj)
        self.orientation = np.array(orientation if orientation is not None else self.set_orientation(), dtype=float)
        self.body_id = p.loadURDF(urdf_file, basePosition=self.start_pos.tolist(), baseOrientation=self.orientation.tolist())
        self.cid = p.createConstraint(self.body_id, -1, -1, -1, p.JOINT_FIXED, [0, 0, 0], [0,0,0], self.start_pos.tolist(), [0,0,0,1], self.orientation.tolist())

    def close_gripper(self):
        for joint in [0,2]:
            p.setJointMotorControl2(self.body_id, joint, p.POSITION_CONTROL, targetPosition=0.1, maxVelocity=2, force=10)

    def open_gripper(self):
        for joint in [0,2]:
            p.setJointMotorControl2(self.body_id, joint, p.POSITION_CONTROL, targetPosition=0.95, maxVelocity=2, force=10)

    def move_gripper(self, x, y, z, force=80):
        p.changeConstraint(self.cid, jointChildPivot=[x, y, z], jointChildFrameOrientation = self.orientation, maxForce=force)

    def move_towards_obj(self):
        obj_pos, _ = p.getBasePositionAndOrientation(self.OBJ.body_id)
        obj_pos = np.array(obj_pos); obj_pos[2] += 0.005
        curr_pos, _ = p.getBasePositionAndOrientation(self.body_id)
        d_vec = obj_pos - np.array(curr_pos)
        pos_step = obj_pos - 0.30 * (d_vec / np.linalg.norm(d_vec))
        self.move_gripper(pos_step[0], pos_step[1], pos_step[2], force=1000)




# ---------------------------------------------------------
# ThreeFingerHand gripper
# ---------------------------------------------------------
class ThreeFingerHand(SimGripper):
    GRASP_JOINTS = [1, 4, 7]
    PRESHAPE_JOINTS = [2, 5, 8]
    UPPER_JOINTS = [3, 6, 9]

    def __init__(self, urdf_file, pos=None, orientation=None, target_obj=None):
        super().__init__(urdf_file, pos, orientation, target_obj)
        base_orientation_quat = self.set_orientation()
        TWIST_ANGLE = np.pi / 2 
        twist_quat = p.getQuaternionFromEuler([np.pi/2, 0, TWIST_ANGLE])
        twisted_orientation_quat = p.multiplyTransforms(
            positionA=[0, 0, 0],
            orientationA=base_orientation_quat,
            positionB=[0, 0, 0],
            orientationB=twist_quat,
        )[1] 
        self.orientation = np.array(orientation if orientation is not None else twisted_orientation_quat, dtype=float)
        self.body_id = p.loadURDF(urdf_file, basePosition=self.start_pos.tolist(), baseOrientation=self.orientation.tolist(), globalScaling=1.0)
        self.gripper_id = self.body_id 
        self.open = False
        self.num_joints = p.getNumJoints(self.body_id)
        
        # Filter the lists. If joint index >= num_joints, remove it.
        self.GRASP_JOINTS = [j for j in self.GRASP_JOINTS if j < self.num_joints]
        self.PRESHAPE_JOINTS = [j for j in self.PRESHAPE_JOINTS if j < self.num_joints]
        self.UPPER_JOINTS = [j for j in self.UPPER_JOINTS if j < self.num_joints]
        # ----------------------------------------------------------

        # Movement Constraint
        self.cid = p.createConstraint(
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

    def preshape(self):
        """Move fingers into preshape pose."""
        for i in [2, 5, 8]:
            p.setJointMotorControl2(self.gripper_id, i, p.POSITION_CONTROL,
                                    targetPosition=-0.7, maxVelocity=0.5, force=1)
        self.open = False
        time.sleep(1)
        
        # Removed time.sleep(1) to avoid slowing down the dataset generation too much

    def open_gripper(self):
        """Gradually open fingers until fully open."""
        closed, iteration = True, 0
        while closed and not self.open:
            joints = self.get_joint_positions()
            closed = False
            for k in range(self.num_joints):
                if k in self.PRESHAPE_JOINTS and joints[k] >= 0.9:
                    self._apply_joint_command(k, joints[k] - 0.01)
                    closed = True
                elif k in self.UPPER_JOINTS and joints[k] <= 0.9:
                    self._apply_joint_command(k, joints[k] - 0.01)
                    closed = True
                elif k in self.GRASP_JOINTS and joints[k] <= 0.9:
                    self._apply_joint_command(k, joints[k] - 0.01)
                    closed = True
            iteration += 1
            if iteration > 1000: # Reduced iteration limit for speed
                break
            p.stepSimulation()
        self.open = True

    def _apply_joint_command(self, joint, target):
        p.setJointMotorControl2(self.gripper_id, joint, p.POSITION_CONTROL,
                                targetPosition=target, maxVelocity=2, force=9999)

    def get_joint_positions(self):
        return [p.getJointState(self.gripper_id, i)[0] for i in range(self.num_joints)]

    def close_gripper(self): 
        "Close gripper to grab object"
        self._apply_joint_command(
            joint=7,
            target=-0.5)
        for j in [1, 4, 7]:
            self._apply_joint_command(
                joint=j, target=0.3)
        time.sleep(2)
        self.open=False
        
    def move_gripper(self, x, y, z, force=80):
        p.changeConstraint(
            self.cid,
            jointChildPivot=[x, y, z],
            jointChildFrameOrientation = self.orientation,
            maxForce=force
        )

    def move_towards_obj(self):
        min_dist = 0.17
        z_offset = 0 
        obj_pos, _ = p.getBasePositionAndOrientation(self.OBJ.body_id)
        obj_pos = np.array(obj_pos); obj_pos[2] += z_offset
        curr_pos, _ = p.getBasePositionAndOrientation(self.body_id)
        d_vec = obj_pos - np.array(curr_pos)
        pos_step = obj_pos - min_dist *(d_vec / np.linalg.norm(d_vec)) 
        self.move_gripper(pos_step[0], pos_step[1], pos_step[2], force=1000)

    
#----------SIM LOOP--------------#

if __name__ == "__main__":

    COLLECT_DATA = False

    # ------------------- Setup ------------------- #
    cid = p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.resetDebugVisualizerCamera(cameraDistance=1.0, cameraYaw=50, cameraPitch=-35, cameraTargetPosition=[0, 0, 0.2])
    p.setPhysicsEngineParameter(numSolverIterations=300, erp=0.3, contactERP=0.3)

    n = 10000
    data = np.empty((n,8)) 

    for i in range(n):

        p.resetSimulation()
        p.setGravity(0, 0, -10)
        p.setRealTimeSimulation(0)
        p.loadURDF("plane.urdf")

        #------ Cylinder -----------#
        cylinder_start_orientation = p.getQuaternionFromEuler([0, 0, 0])
        CYLINDER = CylinderObject(f"Cylinder{i+1}", pos=[0.0,0.0, 0.06], orientation=cylinder_start_orientation)
        current_obj = CYLINDER

        for _ in range(50): p.stepSimulation(); time.sleep(1./240.)

        
        #-------------Choosing the gripper------------#
    
        #curr_gripper = ThreeFingerHand("./threeFingers/sdh/sdh.urdf", pos=None, orientation=None, target_obj=current_obj)
        
        curr_gripper = PR2Gripper("pr2_gripper.urdf", pos=None, orientation=None, target_obj=current_obj)
        
        # ---------------------------------------------------

        # Open
        curr_gripper.open_gripper()
        for _ in range(50): p.stepSimulation(); time.sleep(1./240.)

        curr_gripper.move_towards_obj()
        for _ in range(80): p.stepSimulation(); time.sleep(1./240.)

        # Close
        curr_gripper.close_gripper()
        for _ in range(150): p.stepSimulation(); time.sleep(1./240.)

        current_obj.pos_grab_before, _ = p.getBasePositionAndOrientation(current_obj.body_id)

        # Lift
        curr_gripper.grab_start_pos, _ = p.getBasePositionAndOrientation(curr_gripper.body_id)
        x, y, z = curr_gripper.grab_start_pos
        curr_gripper.move_gripper(x, y, z + 0.3)
        
        for _ in range(50): p.stepSimulation(); time.sleep(1./240.)

        # Wait
        for _ in range(120): p.stepSimulation(); time.sleep(1./240.)

        current_obj.pos_grab_after, _ = p.getBasePositionAndOrientation(current_obj.body_id)
        curr_gripper.grab_end_pos, _ = p.getBasePositionAndOrientation(curr_gripper.body_id)

        result = curr_gripper.is_success()
        data[i,:] = np.hstack([curr_gripper.start_pos, curr_gripper.orientation, result])

        for _ in range(50): p.stepSimulation(); time.sleep(1./240.)


    p.disconnect()
    cols = ["x","y","z","qx","qy","qz","qw","Result"]
    df = pd.DataFrame(data, columns=cols)
    print(df.head())

    if COLLECT_DATA:
        df.to_csv("poses_dataset_sdh.csv", index=False)