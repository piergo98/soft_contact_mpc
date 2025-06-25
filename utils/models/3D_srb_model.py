import casadi as ca
import numpy as np
import matplotlib.pyplot as plt
import dill
from numbers import Number


class SingleRigidBody3D:
    # Class for a 3D Single Rigid Body model with feet.

    n_states = 18  # Increased for 3D (x, y, z, roll, pitch, yaw, foot positions, velocities)
    n_controls = 12  # Increased for 3D forces and velocities
    g = 9.81
    friction = 0.6

    def __init__(self, 
                mass:                           float,
                inertia_tensor:                 np.ndarray,  # 3x3 inertia tensor
                leg_length:                     float,
                lower_leg_length:               float,
                upper_leg_length:               float,
                com_hip_distance:               float,
                box_length:                     float,
                box_width:                      float,
                box_height:                     float,
                minimum_distance_h:             float,
                minimum_foot2hip_dist:          float) -> None:

        self.m = mass
        self.I = inertia_tensor  # 3x3 inertia tensor
        self.leg_length = leg_length
        self.l1 = lower_leg_length
        self.l2 = upper_leg_length
        self.l3 = com_hip_distance
        self.l_box = box_length
        self.w_box = box_width
        self.h_box = box_height
        self.dmin = minimum_distance_h
        self.min_f2h_dist = minimum_foot2hip_dist

        self.IK_front = self.inverse_kinematics(leg='F')
        self.IK_hind = self.inverse_kinematics(leg='H')

    def stance_dynamics(self, x_cartesian: ca.SX, u: ca.SX):
        """
        3D Stance dynamics in Cartesian coordinates.

        Args:
            x_cartesian (ca.SX): Cartesian state.
            u           (ca.SX): Control input.

        Returns:
            ca.SX: State derivative.
        """
        x, y, z, roll, pitch, yaw, x_f, y_f, z_f, x_h, y_h, z_h, x_dot, y_dot, z_dot, roll_dot, pitch_dot, yaw_dot = ca.vertsplit(x_cartesian)
        F_xf, F_yf, F_zf, F_xh, F_yh, F_zh, v_xf, v_yf, v_zf, v_xh, v_yh, v_zh = ca.vertsplit(u)
        
        # Foot positions are fixed in stance
        x_f_dot = 0
        y_f_dot = 0
        z_f_dot = 0
        x_h_dot = 0
        y_h_dot = 0
        z_h_dot = 0
        
        # Linear accelerations
        x_ddot = (F_xf + F_xh) / self.m
        y_ddot = (F_yf + F_yh) / self.m
        z_ddot = (F_zf + F_zh) / self.m - self.g
        
        # Calculate moment arms
        r_f = ca.vertcat(x_f - x, y_f - y, z_f - z)
        r_h = ca.vertcat(x_h - x, y_h - y, z_h - z)
        
        # Force vectors
        F_f = ca.vertcat(F_xf, F_yf, F_zf)
        F_h = ca.vertcat(F_xh, F_yh, F_zh)
        
        # Calculate torques (cross products)
        tau_f = ca.cross(r_f, F_f)
        tau_h = ca.cross(r_h, F_h)
        
        # Total torque
        tau_total = tau_f + tau_h
        
        # Angular accelerations (simplified - assuming diagonal inertia tensor)
        roll_ddot = tau_total[0] / self.I[0, 0]
        pitch_ddot = tau_total[1] / self.I[1, 1]
        yaw_ddot = tau_total[2] / self.I[2, 2]

        return ca.vertcat(
            x_dot, y_dot, z_dot, roll_dot, pitch_dot, yaw_dot,
            x_f_dot, y_f_dot, z_f_dot, x_h_dot, y_h_dot, z_h_dot,
            x_ddot, y_ddot, z_ddot, roll_ddot, pitch_ddot, yaw_ddot
        )
        
    def push_dynamics(self, x_cartesian: ca.SX, u: ca.SX):
        """
        3D Push dynamics in Cartesian coordinates.

        Args:
            x_cartesian (ca.SX): Cartesian state
            u           (ca.SX): Control input

        Returns:
            ca.SX: State derivative
        """
        x, y, z, roll, pitch, yaw, x_f, y_f, z_f, x_h, y_h, z_h, x_dot, y_dot, z_dot, roll_dot, pitch_dot, yaw_dot = ca.vertsplit(x_cartesian)
        F_xf, F_yf, F_zf, F_xh, F_yh, F_zh, v_xf, v_yf, v_zf, v_xh, v_yh, v_zh = ca.vertsplit(u)

        # Rotation matrix from body to world frame (simplified)
        R = self.rotation_matrix(roll, pitch, yaw)
        
        # Front foot can move (push)
        foot_vel_world = ca.mtimes(R, ca.vertcat(v_xf, v_yf, v_zf))
        omega = ca.vertcat(roll_dot, pitch_dot, yaw_dot)
        r_f = ca.vertcat(x_f - x, y_f - y, z_f - z)
        
        x_f_dot = x_dot + foot_vel_world[0] - ca.cross(omega, r_f)[0]
        y_f_dot = y_dot + foot_vel_world[1] - ca.cross(omega, r_f)[1]
        z_f_dot = z_dot + foot_vel_world[2] - ca.cross(omega, r_f)[2]
        
        # Hind foot is fixed
        x_h_dot = 0
        y_h_dot = 0
        z_h_dot = 0
        
        # Linear accelerations
        x_ddot = F_xh / self.m
        y_ddot = F_yh / self.m
        z_ddot = F_zh / self.m - self.g
        
        # Calculate moment arm and torque for hind foot
        r_h = ca.vertcat(x_h - x, y_h - y, z_h - z)
        F_h = ca.vertcat(F_xh, F_yh, F_zh)
        tau_h = ca.cross(r_h, F_h)
        
        # Angular accelerations
        roll_ddot = tau_h[0] / self.I[0, 0]
        pitch_ddot = tau_h[1] / self.I[1, 1]
        yaw_ddot = tau_h[2] / self.I[2, 2]

        return ca.vertcat(
            x_dot, y_dot, z_dot, roll_dot, pitch_dot, yaw_dot,
            x_f_dot, y_f_dot, z_f_dot, x_h_dot, y_h_dot, z_h_dot,
            x_ddot, y_ddot, z_ddot, roll_ddot, pitch_ddot, yaw_ddot
        )

    def flight_dynamics(self, x_cartesian: ca.SX, u: ca.SX):
        """
        3D Flight dynamics in Cartesian coordinates.

        Args:
            x_cartesian (ca.SX): Cartesian state
            u           (ca.SX): Control input

        Returns:
            ca.SX: State derivative
        """
        x, y, z, roll, pitch, yaw, x_f, y_f, z_f, x_h, y_h, z_h, x_dot, y_dot, z_dot, roll_dot, pitch_dot, yaw_dot = ca.vertsplit(x_cartesian)
        F_xf, F_yf, F_zf, F_xh, F_yh, F_zh, v_xf, v_yf, v_zf, v_xh, v_yh, v_zh = ca.vertsplit(u)

        # Rotation matrix
        R = self.rotation_matrix(roll, pitch, yaw)
        omega = ca.vertcat(roll_dot, pitch_dot, yaw_dot)
        
        # Both feet can move in flight
        r_f = ca.vertcat(x_f - x, y_f - y, z_f - z)
        r_h = ca.vertcat(x_h - x, y_h - y, z_h - z)
        
        foot_vel_f = ca.mtimes(R, ca.vertcat(v_xf, v_yf, v_zf))
        foot_vel_h = ca.mtimes(R, ca.vertcat(v_xh, v_yh, v_zh))
        
        x_f_dot = x_dot + foot_vel_f[0] - ca.cross(omega, r_f)[0]
        y_f_dot = y_dot + foot_vel_f[1] - ca.cross(omega, r_f)[1]
        z_f_dot = z_dot + foot_vel_f[2] - ca.cross(omega, r_f)[2]
        
        x_h_dot = x_dot + foot_vel_h[0] - ca.cross(omega, r_h)[0]
        y_h_dot = y_dot + foot_vel_h[1] - ca.cross(omega, r_h)[1]
        z_h_dot = z_dot + foot_vel_h[2] - ca.cross(omega, r_h)[2]
        
        # In flight, only gravity acts on the body
        x_ddot = 0.0
        y_ddot = 0.0
        z_ddot = -self.g
        
        # No external torques in flight
        roll_ddot = 0.0
        pitch_ddot = 0.0
        yaw_ddot = 0.0

        return ca.vertcat(
            x_dot, y_dot, z_dot, roll_dot, pitch_dot, yaw_dot,
            x_f_dot, y_f_dot, z_f_dot, x_h_dot, y_h_dot, z_h_dot,
            x_ddot, y_ddot, z_ddot, roll_ddot, pitch_ddot, yaw_ddot
        )
        
    def land_dynamics(self, x_cartesian: ca.SX, u: ca.SX):
        """
        3D Land dynamics in Cartesian coordinates.

        Args:
            x_cartesian (ca.SX): Cartesian state
            u           (ca.SX): Control input

        Returns:
            ca.SX: State derivative
        """
        x, y, z, roll, pitch, yaw, x_f, y_f, z_f, x_h, y_h, z_h, x_dot, y_dot, z_dot, roll_dot, pitch_dot, yaw_dot = ca.vertsplit(x_cartesian)
        F_xf, F_yf, F_zf, F_xh, F_yh, F_zh, v_xf, v_yf, v_zf, v_xh, v_yh, v_zh = ca.vertsplit(u)

        # Front foot fixed
        x_f_dot = 0
        y_f_dot = 0
        z_f_dot = 0
        
        # Hind foot can move
        R = self.rotation_matrix(roll, pitch, yaw)
        omega = ca.vertcat(roll_dot, pitch_dot, yaw_dot)
        r_h = ca.vertcat(x_h - x, y_h - y, z_h - z)
        
        foot_vel_h = ca.mtimes(R, ca.vertcat(v_xh, v_yh, v_zh))
        
        x_h_dot = x_dot + foot_vel_h[0] - ca.cross(omega, r_h)[0]
        y_h_dot = y_dot + foot_vel_h[1] - ca.cross(omega, r_h)[1]
        z_h_dot = z_dot + foot_vel_h[2] - ca.cross(omega, r_h)[2]
        
        # Linear accelerations
        x_ddot = F_xf / self.m
        y_ddot = F_yf / self.m
        z_ddot = F_zf / self.m - self.g
        
        # Calculate moment arm and torque for front foot
        r_f = ca.vertcat(x_f - x, y_f - y, z_f - z)
        F_f = ca.vertcat(F_xf, F_yf, F_zf)
        tau_f = ca.cross(r_f, F_f)
        
        # Angular accelerations
        roll_ddot = tau_f[0] / self.I[0, 0]
        pitch_ddot = tau_f[1] / self.I[1, 1]
        yaw_ddot = tau_f[2] / self.I[2, 2]

        return ca.vertcat(
            x_dot, y_dot, z_dot, roll_dot, pitch_dot, yaw_dot,
            x_f_dot, y_f_dot, z_f_dot, x_h_dot, y_h_dot, z_h_dot,
            x_ddot, y_ddot, z_ddot, roll_ddot, pitch_ddot, yaw_ddot
        )
        
    def rotation_matrix(self, roll, pitch, yaw):
        """
        Returns 3D rotation matrix from roll, pitch, yaw Euler angles.
        
        Args:
            roll (ca.SX): Roll angle
            pitch (ca.SX): Pitch angle
            yaw (ca.SX): Yaw angle
            
        Returns:
            ca.SX: 3x3 rotation matrix
        """
        # Roll rotation
        R_roll = ca.vertcat(
            ca.horzcat(1, 0, 0),
            ca.horzcat(0, ca.cos(roll), -ca.sin(roll)),
            ca.horzcat(0, ca.sin(roll), ca.cos(roll))
        )
        
        # Pitch rotation
        R_pitch = ca.vertcat(
            ca.horzcat(ca.cos(pitch), 0, ca.sin(pitch)),
            ca.horzcat(0, 1, 0),
            ca.horzcat(-ca.sin(pitch), 0, ca.cos(pitch))
        )
        
        # Yaw rotation
        R_yaw = ca.vertcat(
            ca.horzcat(ca.cos(yaw), -ca.sin(yaw), 0),
            ca.horzcat(ca.sin(yaw), ca.cos(yaw), 0),
            ca.horzcat(0, 0, 1)
        )
        
        # Combined rotation matrix
        return ca.mtimes(R_yaw, ca.mtimes(R_pitch, R_roll))
        
    def jacobian(self, q, leg):
        """Returns inversed transposed Jacobian for 3D kinematics.

        Args:
            q (list): Joint angles
            leg (str): Which leg ('F' or 'H')

        Returns:
            ca.SX or np.ndarray: 3x3 Jacobian matrix
        """
        # For 3D kinematics, this would be more complex
        # This is a simplified approach that needs further development
        J = ca.SX(3, 3) if not isinstance(q[0], Number) else np.empty((3,3))
        
        # Implement 3D kinematics Jacobian calculation
        # This would depend on the specific leg configuration in 3D space
        
        # Placeholder for now - this would need to be properly implemented
        if leg == 'H':
            # Implement hind leg jacobian
            pass
        elif leg == 'F':
            # Implement front leg jacobian
            pass
            
        return J
    
    def inverse_kinematics(self, leg):
        """Inverse kinematics of a leg in 3D.

        Args:
            leg (str): Which leg ('F' or 'H')

        Returns:
            ca.Function: Function to compute joint angles
        """
        # Define the symbolic variables
        base_pos = ca.SX.sym('base_pos', 3)  # x, y, z
        orientation = ca.SX.sym('orientation', 3)  # roll, pitch, yaw
        foot_pos = ca.SX.sym('foot_pos', 3)  # x, y, z foot position
        
        # This is a placeholder - a complete 3D IK solution would be implemented here
        # based on the specific leg configuration and geometry
        
        # For now, return a dummy function
        q = ca.SX.sym('q', 3)  # Assuming 3 joint angles per leg in 3D
        
        q = ca.Function('q', [base_pos, orientation, foot_pos], [q])
        
        return q
