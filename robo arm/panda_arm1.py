import math
import time

import numpy as np
import pandas as pd
import pybullet as p
import pybullet_data


class CubeObject:
    def __init__(self, name: str, init_pos=None, orientation=None, scale=1.0):
        self.name = name

        self.init_pos = np.array(init_pos) if init_pos else self.generate_pos()
        self.orientation = np.array(orientation) if orientation else p.getQuaternionFromEuler([0, 0, 0])

        self.body_id = p.loadURDF(
            "cube_small.urdf",
            basePosition=self.init_pos,
            baseOrientation=self.orientation,
            globalScaling=scale,
        )

        p.changeVisualShape(self.body_id, -1, rgbaColor=[1,0,0,1])

        self.start_pos = None
        self.end_pos = None

    def generate_pos(self):

        min_dist = 0.2 # smaller than this will collide with arm base
        min_cube_height = 0.024988

        # x y z
        while True:
            x, y, z = np.random.uniform(-0.7, 0.7, size=3)

            if (abs(x) > min_dist or abs(y) > min_dist) and z > min_cube_height:
                return np.array([x,y,z])
            
    def get_pos(self):

        curr_pos, _ = p.getBasePositionAndOrientation(self.body_id)
        return curr_pos
    
    def get_angular_velocity(self):
        _, angular_vel = p.getBaseVelocity(self.body_id)
        return angular_vel


class PandaArm:
    def __init__(self, collect_data):

        self.collect_data = collect_data

        self.robot_id = p.loadURDF("franka_panda/panda.urdf", useFixedBase=True)
        self.ee_link = 11
        self.arm_joint_indices = [0, 1, 2, 3, 4, 5, 6]
        self.finger_joint_indices = [9, 10]
        self.rest_pose = [0.0, -np.pi/4, 0.0, -3*np.pi/4, 0.0, np.pi/2, np.pi/4]
        
        for idx, pos in zip(self.arm_joint_indices, self.rest_pose):
            p.resetJointState(self.robot_id, idx, pos)

        self.open_gripper()
        

    def step(self, steps=150):
        for _ in range(steps):
            p.stepSimulation()

            if not self.collect_data:
                time.sleep(1./240.)


    def set_arm_joints(self, joint_positions, steps=150):
        for idx, pos in zip(self.arm_joint_indices, joint_positions):
            p.setJointMotorControl2(
                self.robot_id,
                idx,
                p.POSITION_CONTROL,
                targetPosition=pos,
                force=300,          # good force, speed combo
                maxVelocity=5.0
            )
        self.step(steps)


    def move_ee(self, target_pos, target_orn=None, steps=150):

        #target_orn = target_orn or p.getQuaternionFromEuler([np.pi, 0.0, 0.0])

        if target_orn is None:

            joint_positions = p.calculateInverseKinematics(
                self.robot_id,
                self.ee_link,
                target_pos,
                #target_orn,
                maxNumIterations=100,
                residualThreshold=1e-4,
            )
        
        else:
            joint_positions = p.calculateInverseKinematics(
                self.robot_id,
                self.ee_link,
                target_pos,
                target_orn,
                maxNumIterations=100,
                residualThreshold=1e-4,
            )

        self.set_arm_joints(joint_positions[: len(self.arm_joint_indices)], steps)


    def open_gripper(self, steps=100):
        for j in self.finger_joint_indices:
            p.setJointMotorControl2(
                self.robot_id, j, p.POSITION_CONTROL, targetPosition=0.04, force=20
            )
        self.step(steps)

    def close_gripper(self, steps=100):
        for j in self.finger_joint_indices:
            p.setJointMotorControl2(
                self.robot_id, j, p.POSITION_CONTROL, targetPosition=0.0, force=100
            )
        self.step(steps)

    def pick(self, obj: CubeObject, target_orn=None):

        #obj_pos, _ = p.getBasePositionAndOrientation(obj.body_id)
        obj.start_pos = obj.get_pos()
        self.grasp_h = obj.start_pos[2] + 0.015
        aproach_h = self.grasp_h + 0.01
        self.lift_h = 0.40

        x, y, _ = obj.start_pos

        # go slightly above first to prevent hitting cube
        self.move_ee([x, y, aproach_h], target_orn, steps=250)
        self.move_ee([x, y, self.grasp_h], target_orn, steps=50)
        self.close_gripper()
        self.move_ee([x, y, self.lift_h], target_orn, steps=150)
        obj.end_pos = obj.get_pos()


    def is_success(self, obj: CubeObject):
        start_pos_obj = np.array(obj.start_pos)
        end_pos_obj = np.array(obj.end_pos)

        dist_diff_vec = (np.array(self.lift_h) - np.array(self.grasp_h)) - (end_pos_obj - start_pos_obj)
        height_diff = dist_diff_vec[2]  # height

        angular_vel = obj.get_angular_velocity()

        if height_diff < 0.014 and np.linalg.norm(angular_vel) < 0.15:
            print(f"\033[32m{obj.name}: SUCCESS\033[0m")
            return 1.0
        else:
            print(f"\033[31m{obj.name}: FAILURE\033[0m")
            return 0.0

# to do
class Simulation:
    def __init__(self, collect_data):

        self.collect_data = collect_data

        if self.collect_data:
            self.cid = p.connect(p.DIRECT)
        else:
            self.cid = p.connect(p.GUI)

        if self.cid < 0:
            raise RuntimeError("connection failed")

        p.setAdditionalSearchPath(pybullet_data.getDataPath())

        # camera settings
        p.resetDebugVisualizerCamera(
            cameraDistance=2.0,
            cameraYaw=50,
            cameraPitch=-35,
            cameraTargetPosition=[0, 0, 0.1]
        )

    def reset_sim(self):
        p.resetSimulation()
        p.setGravity(0, 0, -9.81)
        p.loadURDF("plane.urdf")

    def step(self, steps=100):
        for _ in range(steps):
            p.stepSimulation()

            if not self.collect_data:
                time.sleep(1./240.)



if __name__ == "__main__":

    COLLECT_DATA = False

    sim = Simulation(COLLECT_DATA)
    N_SIMS = 10000

    data = np.empty((N_SIMS,3)) # 2 features (x,y) + 1 output (success or failure)

    # SIM LOOP
    for i in range(N_SIMS):

        sim.reset_sim()

        #cube_pos = [0.626147, -0.474556, 0.025]
        cube = CubeObject(f"cube{i}", init_pos=None)
        sim.step()

        # load arm
        arm = PandaArm(COLLECT_DATA)

        # pick
        ori = p.getQuaternionFromEuler([np.pi, 0.0, 0.0])
        arm.pick(cube, target_orn=ori)

        result = arm.is_success(cube)
        data[i,:] = np.hstack([cube.start_pos[0:2], result])
        
        sim.step(steps=50)

    p.disconnect()

    cols = ["x","y","Result"] # no z since constant (0.024988)

    df = pd.DataFrame(data, columns=cols)
    print(df.head(),"\n")

    print(df["Result"].value_counts())

    if COLLECT_DATA:
        df.to_csv("arm_reachability_down.csv", index=False)