import casadi as ca
from math import sqrt
import matplotlib.pyplot as plt
import numpy as np
import dill
from numbers import Number


class SingleRigidBody3D:
    # Class for a 3D Single Rigid Body model with feet.

    n_states = 25  # (r, q, v, omega, p1, p2, p3, p4)
    n_controls = 24  # (F_1, F_2, F_3, F_4, v_1, v_2, v_3, v_4)
    n_feet = 4  # Number of feet
    g = np.array([0, 0, 9.81])  # Gravity vector in 3D
    friction = 0.6

    def __init__(self,
                params:     dict, 
        ) -> None:

        self.m = params['mass']
        self.I = np.array(params['inertia_tensor']).reshape(3, 3)
        self.leg_length = params['leg_length']
        self.l1 = params['l1']
        self.l2 = params['l2']
        self.l3 = params['l3']
        self.l_box = params['l_box']
        self.w_box = params['w_box']
        self.h_box = sqrt(self.leg_length**2 - self.l_box**2)  # [m]
        self.dmin = params['dmin']
        self.min_f2h_dist = params['dmin']  
        self.gamma = params['gamma']
        self.gamma_v = params['gamma_v']
        
        self.sigmoid, self.sigmoid_v = self.sigmoid_function(self.gamma, self.gamma_v)
        

        # self.IK_front = self.inverse_kinematics(leg='F')
        # self.IK_hind = self.inverse_kinematics(leg='H')
        
    def rigid_body_dynamics(self, h_terrain: ca.Function):
        """
        3D Rigid body dynamics in Cartesian coordinates.

        Args:
            h_terrain (ca.Function): Function to compute terrain height at x position.

        Returns:
            ca.SX: State derivative.
        """
        # Define symbolic variables for state and control input
        x_cartesian = ca.SX.sym('x_cartesian', self.n_states)  # State vector in Cartesian coordinates
        u = ca.SX.sym('u', self.n_controls)  # Control input vector
        
        # Position and orientation
        x, y, z = ca.vertsplit(x_cartesian[:3])                         # Position of CoM in inertial frame
        r = ca.vertcat(x, y, z)                                         # Position vector
        qw, qx, qy, qz = ca.vertsplit(x_cartesian[3:7])                 # Quaternion orientation [w, x, y, z]
        q = ca.vertcat(qw, qx, qy, qz)                                  # Quaternion vector
        
        # Linear and angular velocities
        v_x, v_y, v_z = ca.vertsplit(x_cartesian[7:10])                 # Linear velocity
        v = ca.vertcat(v_x, v_y, v_z)                                   # Linear velocity vector
        omega_x, omega_y, omega_z = ca.vertsplit(x_cartesian[10:13])    # Angular velocity
        omega = ca.vertcat(omega_x, omega_y, omega_z)                   # Angular velocity vector
        
        # Foot positions (assuming 4 feet in 3D model)
        p1x, p1y, p1z = ca.vertsplit(x_cartesian[13:16])                # Foot 1 (LF)
        p1 = ca.vertcat(p1x, p1y, p1z)                                  # Foot 1 position vector (LF)
        p2x, p2y, p2z = ca.vertsplit(x_cartesian[16:19])                # Foot 2 (RF)
        p2 = ca.vertcat(p2x, p2y, p2z)                                  # Foot 2 position vector (RF)
        p3x, p3y, p3z = ca.vertsplit(x_cartesian[19:22])                # Foot 3 (LH) 
        p3 = ca.vertcat(p3x, p3y, p3z)                                  # Foot 3 position vector (LH)
        p4x, p4y, p4z = ca.vertsplit(x_cartesian[22:25])                # Foot 4 (RH)
        p4 = ca.vertcat(p4x, p4y, p4z)                                  # Foot 4 position vector (RH)
        
        # Normalize quaternion to prevent drift during integration (optional but good practice)
        norm_q = ca.norm_2(q)
        if norm_q > 0:
            q = q / norm_q
        
        # Extract contact forces from u
        F_x1, F_y1, F_z1 = ca.vertsplit(u[:3])                          # Force on foot 1 (LF) 
        F_1 = ca.vertcat(
            self.sigmoid(h_terrain(p1x) - p1z)*F_x1, 
            self.sigmoid(h_terrain(p1x) - p1z)*F_y1, 
            self.sigmoid(h_terrain(p1x) - p1z)*F_z1
        )                                                               # Force vector for foot 1 (LF)
        F_x2, F_y2, F_z2 = ca.vertsplit(u[3:6])                         # Force on foot 2 (RF)
        F_2 = ca.vertcat(
            self.sigmoid(h_terrain(p2x) - p2z)*F_x2, 
            self.sigmoid(h_terrain(p2x) - p2z)*F_y2, 
            self.sigmoid(h_terrain(p2x) - p2z)*F_z2
        )                                                               # Force vector for foot 2 (RF)
        F_x3, F_y3, F_z3 = ca.vertsplit(u[6:9])                         # Force on foot 3 (LH)
        F_3 = ca.vertcat(
            self.sigmoid(h_terrain(p3x) - p3z)*F_x3, 
            self.sigmoid(h_terrain(p3x) - p3z)*F_y3, 
            self.sigmoid(h_terrain(p3x) - p3z)*F_z3
        )                                                               # Force vector for foot 3 (LH)
        F_x4, F_y4, F_z4 = ca.vertsplit(u[9:12])                        # Force on foot 4 (RH)
        F_4 = ca.vertcat(
            self.sigmoid(h_terrain(p4x) - p4z)*F_x4, 
            self.sigmoid(h_terrain(p4x) - p4z)*F_y4, 
            self.sigmoid(h_terrain(p4x) - p4z)*F_z4
        )                                                               # Force vector for foot 4 (RH)
        
        # Extract foot velocities from u
        v_x1, v_y1, v_z1 = ca.vertsplit(u[12:15])                       # Velocity of foot 1 (LF)
        v_1 = ca.vertcat(
            self.sigmoid_v(-h_terrain(p1x) + p1z)*v_x1, 
            self.sigmoid_v(-h_terrain(p1x) + p1z)*v_y1, 
            self.sigmoid_v(-h_terrain(p1x) + p1z)*v_z1
        )                                                               # Velocity vector for foot 1 (LF)
        v_x2, v_y2, v_z2 = ca.vertsplit(u[15:18])                       # Velocity of foot 2 (RF)
        v_2 = ca.vertcat(
            self.sigmoid_v(-h_terrain(p2x) + p2z)*v_x2, 
            self.sigmoid_v(-h_terrain(p2x) + p2z)*v_y2, 
            self.sigmoid_v(-h_terrain(p2x) + p2z)*v_z2
        )                                                               # Velocity vector for foot 2 (RF)
        v_x3, v_y3, v_z3 = ca.vertsplit(u[18:21])                       # Velocity of foot 3 (LH)
        v_3 = ca.vertcat(
            self.sigmoid_v(-h_terrain(p3x) + p3z)*v_x3, 
            self.sigmoid_v(-h_terrain(p3x) + p3z)*v_y3, 
            self.sigmoid_v(-h_terrain(p3x) + p3z)*v_z3
        )                                                               # Velocity vector for foot 3 (LH)
        v_x4, v_y4, v_z4 = ca.vertsplit(u[21:24])                       # Velocity of foot 4 (RH)
        v_4 = ca.vertcat(
            self.sigmoid_v(-h_terrain(p4x) + p4z)*v_x4, 
            self.sigmoid_v(-h_terrain(p4x) + p4z)*v_y4, 
            self.sigmoid_v(-h_terrain(p4x) + p4z)*v_z4
        )                                                               # Velocity vector for foot 4 (RH)
        
        
        # --- Kinematics ---
        dp = v  # Rate of change of position is linear velocity
        
        dq = self.quat_dot(q, omega)  # Quaternion derivative
        
        # --- Dynamics ---
        # Linear Dynamics: Newton's Second Law
        F_tot = F_1 + F_2 + F_3 + F_4  # Total force from all feet
        dv = (F_tot / self.m) - self.g  # Linear acceleration (gravity included)
        
        # Angular Dynamics: Euler's Equations
        # Convert orientation quaternion to rotation matrix (body to inertial)
        R_wb = self.quat_to_rot_matrix(q)
        
        # Add torques from each foot
        F = [F_1, F_2, F_3, F_4]
        Tau_total_body = ca.SX.zeros(3)  # Initialize total torque vector
        for i, p in enumerate([p1, p2, p3, p4]):
            # Calculate torque for each foot
            r_i = p - r
            # Check 3D vectors
            if r_i.shape[0] != 3 or F[i].shape[0] != 3:
                raise ValueError(f"Cross product requires 3D vectors, got shapes {r_i.shape} and {F[i].shape}")
            tau_contact_inertial = ca.cross(r_i, F[i])  # Torque = r x F
            # Convert torque to body frame
            tau_contact_body = ca.transpose(R_wb) @ tau_contact_inertial
            Tau_total_body += tau_contact_body  # Sum torques from all feet
            
        # Calculate angular acceleration (Euler's equation)
        # I * d(omega)/dt + omega x (I * omega) = Tau_total_body
        # d(omega)/dt = I_inv * (Tau_total_body - omega x (I * omega))
        
        # Check 3D vectors
        if omega.shape[0] != 3 or (self.I @ omega).shape[0] != 3:
            raise ValueError(f"Cross product requires 3D vectors, got shapes {omega.shape} and {(self.I @ omega).shape}")
        domega = np.linalg.inv(self.I) @ (Tau_total_body - ca.cross(omega, self.I @ omega))
        
        dxdt = ca.vertcat(
            dp,         # Linear velocity
            dq,         # Quaternion derivative
            dv,         # Linear acceleration
            domega,     # Angular acceleration
            v_1,        # Foot 1 velocity (LF)
            v_2,        # Foot 2 velocity (RF)
            v_3,        # Foot 3 velocity (LH)
            v_4,        # Foot 4 velocity (RH)
        )
        
        self.dynamics = ca.Function(
            'SRB',
            [x_cartesian, u],
            [dxdt],
            ['x_cartesian', 'u'],
            ['dxdt'],
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
    
    def quat_to_rot_matrix(self, q):
        """
        Converts a quaternion [w, x, y, z] to a 3x3 rotation matrix R.
        R transforms a vector from body frame to inertial frame.
        
        Args:
            q (ca.SX): Quaternion vector [w, x, y, z]
            
        Returns:
            ca.SX: 3x3 rotation matrix R
        """
        w, x, y, z = ca.vertsplit(q)
        
        R = ca.SX(3, 3)
        
        R[0, 0] = 1 - 2 * (y**2 + z**2)
        R[0, 1] = 2 * (x * y - w * z)
        R[0, 2] = 2 * (x * z + w * y)
        R[1, 0] = 2 * (x * y + w * z)
        R[1, 1] = 1 - 2 * (x**2 + z**2)
        R[1, 2] = 2 * (y * z - w * x)
        R[2, 0] = 2 * (x * z - w * y)
        R[2, 1] = 2 * (y * z + w * x)
        R[2, 2] = 1 - 2 * (x**2 + y**2)
        return R
    
    def quat_multiply(self, q1, q2):
        """
        Multiplies two quaternions q1 and q2.
        q1 * q2 = [w1*w2 - x1*x2 - y1*y2 - z1*z2, w1*x2 + x1*w2 + y1*z2 - z1*y2, w1*y2 - x1*z2 + y1*w2 + z1*x2, w1*z2 + x1*y2 - y1*x2 + z1*w2]
        """
        w1, x1, y1, z1 = ca.vertsplit(q1)
        w2, x2, y2, z2 = ca.vertsplit(q2)
        
        w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
        x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
        y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
        z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
        
        return ca.vertcat(w, x, y, z)
    
    def quat_dot(self, q, omega_body):
        """
        Calculates the quaternion derivative q_dot given quaternion q and
        angular velocity omega in the body frame.
        q_dot = 0.5 * q * [0, omega_x, omega_y, omega_z]
        """
        omega_quat = ca.vertcat(0, omega_body)  # Convert angular velocity to quaternion form
        q_dot = 0.5 * self.quat_multiply(q, omega_quat)
        return q_dot
    
    def sigmoid_function(self, gamma=1000, gamma_v=1000):
        """
        This method is used to define the sigmoid function.
        
        Args:
            z: float
                Input value.
            gamma: float
                Gain parameter.
                
        Returns:
            Sigmoid: casadi.Function
                Sigmoid function.
            Sigmoid_V: casadi.Function
                Sigmoid function for velocity.
        """
        # Contact activation function
        z = ca.SX.sym('z')
        sig = 1 / (1 + ca.exp(-gamma*z))
        sig_v = 1 / (1 + ca.exp(-gamma_v*z))
        Sigmoid = ca.Function('Sigmoid', [z], [sig])
        Sigmoid_V = ca.Function('Sigmoid_V', [z], [sig_v])
        
        return Sigmoid, Sigmoid_V
