import numpy as np

class LegKinematics:
    def __init__(self, BASE_SH_LEN, LINK_LENGTH):
        """
        Initialize the LegKinematics class with base to shoulder length and link length.

        Args:
            BASE_SH_LEN (float): The length from the base to the shoulder joint.
            LINK_LENGTH (float): The length of the leg links (assuming equal length for simplicity).
        """
        self.BASE_SH_LEN = BASE_SH_LEN
        self.LINK_LENGTH = LINK_LENGTH

    def compute_Inverse_Kinematics(self, foot_p, base_pos):
        """
        Compute the angles for the hip and knee based on the foot position and base position.
        
        Args:
            foot_p (list): The position of the foot [x, y].
            base_pos (list): The position of the base [x, y, theta].
        
        Returns:
            tuple: The computed angles (q2, q3) for the hip and knee.
        """
        base_to_sh = np.array([-self.BASE_SH_LEN/2, 0.0])
        
        # Front legs
        foot_pos = np.array(foot_p[:2])
        
        base_pos_w = np.array(base_pos[:2])
        theta = -base_pos[2]
        
        R_l_w = np.array([[np.cos(theta), np.sin(theta)],
                          [-np.sin(theta), np.cos(theta)]])
        
        sh_pos_l = -foot_pos + (base_pos_w + R_l_w.dot(base_to_sh))
        sh_pos_l = np.array([-sh_pos_l[1], -sh_pos_l[0]])  # IMPORTANTISSIMO PER MULINEX
        d = (sh_pos_l[0]**2 + sh_pos_l[1]**2 - 2*(self.LINK_LENGTH**2)) / (2*(self.LINK_LENGTH**2))

        # Hip and knee computation
        q3 = np.arctan2(np.sqrt(1 - d**2), d)
        q2 = np.arctan2(sh_pos_l[0], -sh_pos_l[1]) - np.arctan2(self.LINK_LENGTH*np.sin(q3), self.LINK_LENGTH + self.LINK_LENGTH*np.cos(q3))

        return q2, q3
       
    def compute_Forward_Kinematics(self, base_pos, q2, q3):
        """
        Compute the foot position based on the base position and leg angles q2, q3.

        Args:
            base_pos (list): The position of the base [x, y, theta].
            q2 (float): The hip angle.
            q3 (float): The knee angle.

        Returns:
            np.array: The position of the foot [x, y].
        """
        theta = base_pos[2]
        R_w_l = np.array([[np.cos(theta), -np.sin(theta)],
                        [np.sin(theta), np.cos(theta)]])
        
        # Length from shoulder to foot in the leg's coordinate frame
        L1 = self.LINK_LENGTH * np.cos(q2)
        L2 = self.LINK_LENGTH * np.cos(q2 + q3)
        shoulder_to_foot_l = np.array([L1 + L2, self.LINK_LENGTH * np.sin(q2) + self.LINK_LENGTH * np.sin(q2 + q3)])
        
        # Convert to world frame
        shoulder_to_foot_w = R_w_l.dot(shoulder_to_foot_l)
        
        # Position of the shoulder in world frame
        base_to_shoulder_w = np.array([self.BASE_SH_LEN, 0])
        shoulder_pos_w = base_pos[:2] + R_w_l.dot(base_to_shoulder_w)
        
        # Foot position in world frame
        foot_pos_w = shoulder_pos_w + shoulder_to_foot_w
        
        return foot_pos_w
    
    def compute_Inverse_Kinematics_trajectory(self, foot_p_trajectory, base_pos_trajectory):
        """
        Compute the trajectories of angles for the hip and knee based on the trajectories of the foot position and base position.
        
        Args:
            foot_p_trajectory (list of lists): The trajectory of the foot positions [[x, y], ...].
            base_pos_trajectory (list of lists): The trajectory of the base positions [[x, y, theta], ...].
        
        Returns:
            tuple of lists: The computed trajectories of angles (q2, q3) for the hip and knee.
        """
        q2_trajectory = []
        q3_trajectory = []

        # Iterate over the trajectories
        for foot_p, base_pos in zip(foot_p_trajectory, base_pos_trajectory):
            # Call compute_Inverse_Kinematics for each position in the trajectory
            q2, q3 = self.compute_Inverse_Kinematics(foot_p, base_pos)
            q2_trajectory.append(q2)
            q3_trajectory.append(q3)
        
        return q2_trajectory, q3_trajectory
    
    def calculate_leg_torque(self, external_force, joint_angles):
        """
        Calculate the torque required at the joint level for a given external force on the foot,
        using the Jacobian matrix.

        Parameters:
        - external_force: A numpy array representing the external force applied to the foot.
        - joint_angles: A numpy array of the current angles of the joints in the leg.

        Returns:
        - torque: A numpy array representing the torque required at each joint.
        """
        l1, l2 = self.LINK_LENGTH, self.LINK_LENGTH  # Assuming both links have the same length for simplicity
        theta1, theta2 = joint_angles

        J = np.array([[-l1*np.sin(theta1) - l2*np.sin(theta1 + theta2), -l2*np.sin(theta1 + theta2)],
                      [l1*np.cos(theta1) + l2*np.cos(theta1 + theta2), l2*np.cos(theta1 + theta2)]])

        # Calculate torque: τ = J^T * F
        torque = J.T.dot(external_force)

        return torque
    
    def calculate_torque_trajectories(self, q2_trajectory, q3_trajectory, external_force_trajectory):
        """
        Calculate the torque trajectories for given q2, q3 angles trajectory and external force applied on the foot.

        Args:
            q2_trajectory (list of float): Trajectory of q2 angles over time.
            q3_trajectory (list of float): Trajectory of q3 angles over time.
            external_force_trajectory (list of list of float): Trajectory of external forces applied on the foot over time.

        Returns:
            tuple of lists: Two lists containing the trajectories of torques required at the hip (q2) and knee (q3) joints over time.
        """
        torque2_trajectory = []
        torque3_trajectory = []

        for q2, q3, external_force in zip(q2_trajectory, q3_trajectory, external_force_trajectory):
            joint_angles = np.array([q2, q3])
            torque = self.calculate_leg_torque(external_force, joint_angles)
            torque2_trajectory.append(torque[0])
            torque3_trajectory.append(torque[1])

        return torque2_trajectory, torque3_trajectory
