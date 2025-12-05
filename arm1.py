import math
import time

import numpy as np
import pybullet as p
import pybullet_data


class CubeObject:
    def __init__(self, name: str, pos, orientation=None, scale=1.0):
        self.name = name
        self.pos = pos
        self.orientation = orientation or p.getQuaternionFromEuler([0, 0, 0])
        self.body_id = p.loadURDF(
            "cube_small.urdf",
            basePosition=self.pos,
            baseOrientation=self.orientation,
            globalScaling=scale,
        )


class PandaArm:
    def __init__(self):
        self.robot_id = p.loadURDF("franka_panda/panda.urdf", useFixedBase=True)
        self.ee_link = 11
        self.arm_joint_indices = [0, 1, 2, 3, 4, 5, 6]
        self.finger_joint_indices = [9, 10]
        self.rest_pose = [0.0, -math.pi / 4, 0.0,-3 * math.pi / 4,0.0,math.pi / 2,math.pi / 4,]
        self.open_gripper()
        self.set_arm_joints(self.rest_pose)
        self.step_sim(120)

    def step_sim(self, steps=240):
        for _ in range(steps):
            p.stepSimulation()
            time.sleep(1.0 / 240.0)

    def set_arm_joints(self, joint_positions):
        for idx, pos in zip(self.arm_joint_indices, joint_positions):
            p.setJointMotorControl2(
                self.robot_id,
                idx,
                p.POSITION_CONTROL,
                targetPosition=pos,
                force=180,  # higher force to reduce lag/overshoot
            )

    def move_ee(self, target_pos, target_orn=None, steps=120):
        target_orn = target_orn or p.getQuaternionFromEuler([math.pi, 0.0, 0.0])
        joint_positions = p.calculateInverseKinematics(
            self.robot_id,
            self.ee_link,
            target_pos,
            target_orn,
            maxNumIterations=100,
            residualThreshold=1e-4,
        )
        self.set_arm_joints(joint_positions[: len(self.arm_joint_indices)])
        self.step_sim(steps)

    def open_gripper(self):
        for j in self.finger_joint_indices:
            p.setJointMotorControl2(
                self.robot_id, j, p.POSITION_CONTROL, targetPosition=0.04, force=20
            )
        self.step_sim(60)

    def close_gripper(self):
        for j in self.finger_joint_indices:
            p.setJointMotorControl2(
                self.robot_id, j, p.POSITION_CONTROL, targetPosition=0.0, force=100
            )
        self.step_sim(120)

    def pick(self, obj: CubeObject):
        approach_h = 0.30
        grasp_h = obj.pos[2] + 0.02
        lift_h = 0.35
        x, y, _ = obj.pos

        #self.move_ee([x, y, approach_h])
        self.move_ee([x, y, grasp_h], steps=180)
        self.close_gripper()
        self.move_ee([x, y, lift_h])


if __name__ == "__main__":
    cid = p.connect(p.GUI)
    if cid < 0:
        raise RuntimeError("PyBullet connection failed")

    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.resetSimulation()
    p.setGravity(0, 0, -9.81)
    p.setTimeStep(1.0 / 240.0)

    p.loadURDF("plane.urdf")

    cube = CubeObject("cube1", pos=[0.6, 0.0, 0.025])
    arm = PandaArm()
    arm.pick(cube)

    print("Picked cube; close the window or interrupt to exit.")
    while True:
        p.stepSimulation()
        time.sleep(1.0 / 240.0)

