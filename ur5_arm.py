import pybullet as p
import pybullet_data
import time
import numpy as np
from collections import namedtuple

class SimObject:

    def __init__(self, name, urdf_file=None, pos=None, orientation=None, scale=1.0):
       
        self.name = name
        self.pos = pos
        self.orientation = orientation

        self.body_id = p.loadURDF(urdf_file, basePosition=self.pos, baseOrientation=self.orientation, globalScaling=scale)

        self.pos_grab_before = None
        self.pos_grab_after = None


class CubeObject(SimObject):
    def __init__(self, name, urdf_file="ur5_grasp_object_pybullet/urdf/cube_small.urdf", pos=None, orientation=None, scale=1.0):
        super().__init__(name, urdf_file, pos, orientation, scale)

    def load_cube(self):
        pass


class UR5Robotiq85:
    def __init__(self, base_position=[0, 0, 0], base_orientation=[0, 0, 0, 1]):
        
        self.robot_id = p.loadURDF(
            "ur5_grasp_object_pybullet/urdf/ur5_robotiq_85.urdf",
            basePosition=base_position,
            baseOrientation=base_orientation,
            useFixedBase=True
        )
        
        # Get joint info
        self.num_joints = p.getNumJoints(self.robot_id)
        self.joint_indices = [i for i in range(self.num_joints) 
                             if p.getJointInfo(self.robot_id, i)[2] != p.JOINT_FIXED]
        
        self.gripper_angles = [0, 0.085]   # length between tips at max open and close pos
        self.end_effector_index = 7
        self.arm_num_dofs = 6
        self.max_velocity = 3  # Maximum joint velocity
        
        # Parse joint info for IK constraints
        self.__parse_joint_info__()
    
    def __parse_joint_info__(self):
        """Get joint information including limits and ranges for IK"""
        # Use the same approach as main.py for consistency
        jointInfo = namedtuple('jointInfo',
                               ['id', 'name', 'type', 'lowerLimit', 'upperLimit', 'maxForce', 'maxVelocity', 'controllable'])
        self.joints = []
        self.controllable_joints = []
        
        for i in range(self.num_joints):
            info = p.getJointInfo(self.robot_id, i)
            jointID = info[0]
            jointName = info[1].decode("utf-8") if info[1] else ""
            jointType = info[2]
            jointLowerLimit = info[8]
            jointUpperLimit = info[9]
            jointMaxForce = info[10]
            jointMaxVelocity = info[11]
            controllable = jointType != p.JOINT_FIXED
            if controllable:
                self.controllable_joints.append(jointID)
            self.joints.append(
                jointInfo(jointID, jointName, jointType, jointLowerLimit, jointUpperLimit, jointMaxForce, jointMaxVelocity, controllable)
            )
        
        self.arm_controllable_joints = self.controllable_joints[:self.arm_num_dofs]
        self.arm_lower_limits = [j.lowerLimit for j in self.joints if j.controllable][:self.arm_num_dofs]
        self.arm_upper_limits = [j.upperLimit for j in self.joints if j.controllable][:self.arm_num_dofs]
        self.arm_joint_ranges = [ul - ll for ul, ll in zip(self.arm_upper_limits, self.arm_lower_limits)]
        
        # Rest poses for IK solver (helps find valid solutions)
        self.arm_rest_poses = [-1.57, -1.54, 1.34, -1.37, -1.57, 0.0]

    def get_joint_states(self):
        """Get current joint positions and velocities"""
        states = p.getJointStates(self.robot_id, self.joint_indices)
        positions = [state[0] for state in states]
        velocities = [state[1] for state in states]
        return positions, velocities
    
    def set_joint_positions(self, positions):
        """Set joint positions for the arm"""
        for i, pos in enumerate(positions):
            p.setJointMotorControl2(
                self.robot_id,
                self.arm_controllable_joints[i],
                p.POSITION_CONTROL,
                targetPosition=pos,
                maxVelocity=self.max_velocity
            )

    
    def open_gripper(self):
        open_angle = 0.715 - np.arcsin((max(self.gripper_angles) - 0.010) / 0.1143)
        p.setJointMotorControl2(self.robot_id, 8, p.POSITION_CONTROL, open_angle)
    
    def close_gripper(self):
        close_angle = 0.715 - np.arcsin((min(self.gripper_angles) - 0.010) / 0.1143)
        p.setJointMotorControl2(self.robot_id, 8, p.POSITION_CONTROL, close_angle)

    def solve_IK(self, target_pos, target_ori=None):
        
        if target_ori is None:
            # Without orientation constraint, but still use limits for better solutions
            joint_positions = p.calculateInverseKinematics(
                    self.robot_id,
                    self.end_effector_index,
                    target_pos,
                    lowerLimits=self.arm_lower_limits,
                    upperLimits=self.arm_upper_limits,
                    jointRanges=self.arm_joint_ranges,
                    restPoses=self.arm_rest_poses,
                )
        else:
            # With orientation constraint, need joint limits and rest poses for reliable IK
            joint_positions = p.calculateInverseKinematics(
                    self.robot_id,
                    self.end_effector_index,
                    target_pos,
                    target_ori,
                    lowerLimits=self.arm_lower_limits,
                    upperLimits=self.arm_upper_limits,
                    jointRanges=self.arm_joint_ranges,
                    restPoses=self.arm_rest_poses,
                )

        # Return only the arm joint positions (first 6 DOF)
        return joint_positions[:self.arm_num_dofs]
    

    def move_to(self, target_pos, target_ori):
        joint_positions = self.solve_IK(target_pos, target_ori)
        self.set_joint_positions(joint_positions)


# Usage
if __name__ == "__main__":
    
    
    # Connect to PyBullet (GUI or DIRECT mode)
    physics_client = p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())

    # camera settings
    p.resetDebugVisualizerCamera(
        cameraDistance=1.5,
        cameraYaw=50,
        cameraPitch=-35,
        cameraTargetPosition=[0, 0, 0.2]
    )


    N_SIMS = 1

    for i in range(N_SIMS):

        p.resetSimulation()
        p.setGravity(0, 0, -10)
        p.setRealTimeSimulation(0)

        # Load plane and robot
        p.loadURDF("plane.urdf")

        #------ CUBE -----------#
        cube_start_orientation = p.getQuaternionFromEuler([0, 0, 0])
        cube_z = 0.03
                                                # cube min height
        CUBE = CubeObject(f"Cube{i+1}", pos=[0.5,0.0,cube_z], orientation=cube_start_orientation)

            
        robot = UR5Robotiq85()
        
        # Example: set home position
        home_position = [0, -1.57, 1.57, -1.57, -1.57, 0]
        robot.set_joint_positions(home_position)
        
        for _ in range(200):
            p.stepSimulation()
            time.sleep(1./240.)

        # pick
        target_pos = [0.5, 0.0, 0.2] #0.6
        target_orientation = p.getQuaternionFromEuler([0,np.pi, 0])
        robot.move_to(target_pos, target_ori=target_orientation)
        
        # Wait for robot to reach target position
        for _ in range(200):
            p.stepSimulation()
            time.sleep(1./240.)

        robot.close_gripper()

        # Run simulation
        for _ in range(400):
            p.stepSimulation()
            time.sleep(1./240.)


    p.disconnect()