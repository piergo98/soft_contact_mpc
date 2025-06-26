import casadi as ca
import numpy as np
import matplotlib.pyplot as plt
import dill
from numbers import Number


class SigleRigidBody2D:
    # Class for a 2D Single Rigid Body model with feet.

    n_states = 10
    n_controls = 8   
    g = 9.81
    friction = 0.6

    def __init__(self, 
                mass:                           float,
                inertia:                        float,
                leg_length:                     float,
                lower_leg_length:               float,
                upper_leg_length:               float,
                com_hip_distance:               float,
                box_length:                     float,
                box_heigth:                     float,
                minimum_distance_h:             float,
                minimum_foot2hip_dist:          float,
                gamma:                          float,
                gamma_v:                        float,
                ) -> None:

        self.m  = mass
        self.I  = inertia
        self.leg_length = leg_length
        self.l1 = lower_leg_length
        self.l2 = upper_leg_length
        self.l3 = com_hip_distance
        self.l_box = box_length
        self.h_box = box_heigth
        self.dmin = minimum_distance_h
        self.min_f2h_dist = minimum_foot2hip_dist
        
        self.gamma = gamma
        self.gamma_v = gamma_v
        
        self.sigmoid, self.sigmoid_v = self.sigmoid_function(self.gamma, self.gamma_v)

        self.IK_front = self.inverse_kinematics(leg='F')
        self.IK_hind = self.inverse_kinematics(leg='H')
        
    def rigid_body_dynamics(self, h_terrain: ca.Function):
        """
        2D Rigid body dynamics in Cartesian coordinates.

        Args:
            h_terrain (ca.Function): Function to compute terrain height at x position.

        Returns:
            ca.SX: State derivative.
        """
        # Define symbolic variables for state and control input
        x_cartesian = ca.SX.sym('x_cartesian', self.n_states)  # State vector in Cartesian coordinates
        u = ca.SX.sym('u', self.n_controls)  # Control input vector
        
        # Extract states
        x, z, theta, x_dot, z_dot, theta_dot, x_f, z_f, x_h, z_h = ca.vertsplit(x_cartesian)

        # Extract control inputs
        F_xf, F_zf, F_xh, F_zh, v_xf, v_zf, v_xh, v_zh = ca.vertsplit(u)

        # Calculate terrain height at foot positions
        h_f = h_terrain(x_f)
        h_h = h_terrain(x_h)

        # Apply sigmoid functions to compute contact forces based on height difference
        F_xf_applied = self.sigmoid(h_f - z_f) * F_xf
        F_zf_applied = self.sigmoid(h_f - z_f) * F_zf
        F_xh_applied = self.sigmoid(h_h - z_h) * F_xh
        F_zh_applied = self.sigmoid(h_h - z_h) * F_zh

        # Rigid body kinematics
        x_dot_eq = x_dot
        z_dot_eq = z_dot
        theta_dot_eq = theta_dot

        # Foot kinematics
        x_f_dot = (v_xf + x_dot - theta_dot * (z_f - z)) * self.sigmoid_v(-h_f + z_f)
        z_f_dot = (v_zf + z_dot + theta_dot * (x_f - x)) * self.sigmoid_v(-h_f + z_f)
        x_h_dot = (v_xh + x_dot - theta_dot * (z_h - z)) * self.sigmoid_v(-h_h + z_h)
        z_h_dot = (v_zh + z_dot + theta_dot * (x_h - x)) * self.sigmoid_v(-h_h + z_h)

        # Rigid body dynamics
        x_ddot = (F_xf_applied + F_xh_applied) / self.m
        z_ddot = (F_zf_applied + F_zh_applied) / self.m - self.g
        theta_ddot = (F_xf_applied * (z_f - z) - F_zf_applied * (x_f - x) + 
                      F_xh_applied * (z_h - z) - F_zh_applied * (x_h - x)) / self.I

        # Combine all state derivatives
        dxdt = ca.vertcat(
            x_dot_eq, z_dot_eq, theta_dot_eq,
            x_ddot, z_ddot, theta_ddot,
            x_f_dot, z_f_dot, x_h_dot, z_h_dot,
        )
        
        self.dynamics = ca.Function(
            'SRB',
            [x_cartesian, u],
            [dxdt],
            ['x_cartesian', 'u'],
            ['dxdt'],
        )
        
    def jacobian(self, q, leg):
        """Returns inversed transposed Jacobian.

        Args:
            q (_type_): _description_
            leg (_type_): _description_

        Returns:
            _type_: _description_
        """
        J = ca.SX(3, 3) if not isinstance(q[0], Number) else np.empty((3,3))
        
        if leg == 'H':
            q1, q2, q3 = q[0], q[1], q[2]
            a = 0
            b = 0
            c = 1
            d = -self.l1 * np.sin(q3 + q2)
            e =  self.l1 * np.cos(q3 + q2)
            f = 1
            g = -self.l2 * np.sin(q3) - self.l1 * np.sin(q3 + q2)
            h =  self.l2 * np.cos(q3) + self.l1 * np.cos(q3 + q2)
            i = 1
        elif leg == 'F':
            q4, q5, q6 = q[0], q[1], q[2]
            a = -self.l2 * np.sin(q4) - self.l1 * np.sin(q4 + q5)
            b =  self.l2 * np.cos(q4) + self.l1 * np.cos(q4 + q5)
            c = 1
            d = -self.l1 * np.sin(q4 + q5)
            e =  self.l1 * np.cos(q4 + q5)
            f = 1
            g = 0
            h = 0
            i = 1              

        J[0, 0] = e*i - f*h
        J[0, 1] = c*h - b*i
        J[0, 2] = b*f - c*e
        J[1, 0] = f*g - d*i
        J[1, 1] = a*i - c*g
        J[1, 2] = c*d - a*f
        J[2, 0] = d*h - e*g
        J[2, 1] = b*g - a*h
        J[2, 2] = a*e - b*d
        J *= 1 / (a*(e*i - f*h) - b*(d*i - f*g) + c*(d*h - e*g))
        
        return J
    
    def inverse_kinematics(self, leg):
        """Inverse kinematics of a leg.

        Args:
            x (_type_): _description_
            z (_type_): _description_
            leg (_type_): _description_

        Returns:
            _type_: _description_
        """
        # Define the symbolic variables
        base_pos = ca.SX.sym('base_pos', 2)
        pitch_com = ca.SX.sym('pitch_com')
        foot_pos = ca.SX.sym('foot_pos', 2)
        
        # Define the rotation matrix
        R_l_w = np.array([[np.cos(pitch_com), -np.sin(pitch_com)],
                            [np.sin(pitch_com), np.cos(pitch_com)]])
        
        # Define the base to shoulder vector
        if leg == 'F':
            base_to_shoulder = foot_pos - (base_pos + R_l_w @ np.array([self.l3, 0]))
        elif leg == 'H':
            base_to_shoulder = foot_pos - (base_pos - R_l_w @ np.array([self.l3, 0]))
        
        d = (base_to_shoulder[0]**2 + base_to_shoulder[1]**2 - 2*(self.l1**2))/(2*(self.l1**2))
        
        #  Hip and knee computation
        knee_angle = ca.atan2(ca.sqrt(1 - d**2), d)
        hip_angle = ca.atan2(base_to_shoulder[1], base_to_shoulder[0]) - ca.atan2(self.l1 * ca.sin(knee_angle), self.l1 + self.l1 * ca.cos(knee_angle))
        
        q = ca.vertcat(hip_angle, knee_angle)
        
        q = ca.Function('q', [base_pos, pitch_com, foot_pos], [q])
        
        return q
    
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
            
    
