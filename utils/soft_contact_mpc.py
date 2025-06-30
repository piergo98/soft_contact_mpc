import casadi as ca
import csv
import matplotlib.pyplot as plt
import numpy as np


class SoftContactMPC:
    class Constraint():
        def __init__(self, g, lbg, ubg, name=None):
            self.g = g
            self.lbg = lbg
            self.ubg = ubg
            self.name = name
    def __init__(self, model):
        
        # Load the model
        self.model = model
        
    def setup_problem(self, problem_params):
        """
        This method is used to set up the optimization problem.
        Args:
            params: dict
                Dictionary containing the parameters of the optimization problem.
        """
        # Load problem parameters
        params = problem_params['optimization_problem']
        
        # Number of time steps
        self.N = params['N']
        # Time horizon
        self.T = params['horizon']
        # Discretization time step
        self.DT = self.T / self.N
        # Number of integrations between two steps
        self.n_int = params['n_int']
        # Integration step
        self.h = self.DT / self.n_int
        # Number of optimization variables
        
        self.n_opti = (self.model.n_states + self.model.n_controls) * (self.N) + self.model.n_states
        
        # Set initial state and control input
        self.x0 = params['initial_state']
        init_force = self.model.m / self.model.n_feet * self.model.g
        u0 = [*init_force] * self.model.n_feet
        u0 += [0.0] * self.model.n_velocities * self.model.n_feet
        
        # Default values
        self.x_step = params['x_step']
        self.y_step = params['y_step']
        self.z_step = params['z_step'] 
        
        epsilon = 0
        
        # Desired final state
        x_fin = self.x0[0] + self.x_step
        y_fin = self.x0[1] + self.y_step
        z_fin = self.x0[2] + self.z_step
        self.x_des = np.array([
            x_fin,
            y_fin,
            z_fin,
            1.0,
            0.0,
            0.0,
            0.0,    #
            0.0,
            0.0,
            0.0,    #
            0.0,
            0.0,
            x_fin + self.model.length,  # LF
            y_fin + self.model.width,
            self.z_step,
            x_fin + self.model.l1,      # RF
            y_fin - self.model.l2,
            self.z_step,
            x_fin - self.model.l1,      # LH
            y_fin + self.model.l2,
            self.z_step,
            x_fin - self.model.l3,      # RH
            y_fin - self.model.l3,
        ]).reshape(-1, 1)
        
        # Vector centers for obstacle avoidance            
        self.r_c = 0.03     # Radius contact points in m
        self.r_ob = 0.01    # Radius obstacle circles in m
        self.nc_ob = int(self.z_step/(2*self.r_ob) + 1)   # number of collision circle needed
        self.C_OB = []
        for index in range(self.nc_ob):
            self.C_OB += [[self.x_step, index*2*self.r_ob]]
        
        print('Setting up optimization problem:')
        
        # Empty NLP
        self.w = []
        self.w0 = []
        self.lbw = []
        self.ubw = []
        self.cost = 0
        self.constraints = []
        self.lbconstr = []
        self.ubconstr = []
            
        # Initial state contraint
        self.Xk = ca.SX.sym('X0', self.model.n_states)
        self.w += [self.Xk]
        self.lbw += self.x0
        self.ubw += self.x0
        self.w0 += self.x0
        self.w0 += u0
        
        # Set all the methods for the optimization problem
        self.terrain(slope=params['slope'])
        self.integrator(problem_params['cost'])
        
        for i in range(self.N):
            self.set_control_constraints(i, params)
            self.multiple_shooting(i, params)
            self.set_quaternion_constraints(i)
            self.set_bounding_box(i)
            
        self.set_terminal_cost(problem_params['cost']['terminal'])
        
        # Set the initial guess for the optimization variables
        self.set_initial_guess(params['initial_state'], u0)
    
    def terrain(self, slope=0.0):
        """
        This method is used to define the terrain profile.
        
        Args:
            x: float
                Input value.
                
        Returns:
            float
                Output value of the terrain profile.
        """
        x_t = ca.SX.sym('x_t')
        # h_t = h_j * (1 / (1 + exp(-300*(x_t - x_j))))          # Step function bild like a sigmoid
        h_t = slope*x_t
        # h_t = 0.0*sin(1000*x_t)
        h_terrain = ca.Function('h_terrain', [x_t], [h_t])
        
        # Build thhe rigid body dynamics function
        self.model.rigid_body_dynamics(h_terrain)
        return h_terrain
        
    def cost_function(self, x_cartesian: ca.SX, u: ca.SX, gains):
        """
        This method is used to define the cost function of the optimization problem.
        
        Args:
            gains: dict
                Dictionary containing the gains of the cost function.
        """
        # Extract variables from the state vector
        r = x_cartesian[0:3]         # CoM position
        q = x_cartesian[3:7]         # CoM orientation (quaternion)
        v = x_cartesian[7:10]        # CoM velocity
        omega = x_cartesian[10:13]   # CoM angular velocity
        p1 = x_cartesian[13:16]      # LF foot position
        p2 = x_cartesian[16:19]      # RF foot position
        p3 = x_cartesian[19:22]      # LH foot position
        p4 = x_cartesian[22:25]      # RH foot position
        
        # Extract variables from the control input vector
        F1 = u[0:3]                  # LF foot force
        F2 = u[3:6]                  # RF foot force
        F3 = u[6:9]                  # LH foot force
        F4 = u[9:12]                 # RH foot force
        v_p1 = u[12:15]              # LF foot velocity
        v_p2 = u[15:18]              # RF foot velocity
        v_p3 = u[18:21]              # LH foot velocity
        v_p4 = u[21:24]              # RH foot velocity
        
        # Extract gains from the input dictionary
        k_pos = gains['k_pos']
        k_orientation = gains['k_orientation']
        k_vel = gains['k_vel']
        k_omega = gains['k_omega']
        k_foot = gains['k_foot']
        k_force = gains['k_force']
        k_foot_vel = gains['k_vel']
        
        # Tracking term
        tracking = k_pos * ca.sumsqr(r - self.x_des[0:3])
        
        # Orientation term
        orientation = k_orientation * ca.sumsqr(self.model.quat_multiply(self.model.quat_conjugate(q), self.x_des[3:7]))
        
        # Velocity term
        velocity = k_vel * ca.sumsqr(v - self.x_des[7:10])
        
        # Angular velocity term
        angular_velocity = k_omega * ca.sumsqr(omega - self.x_des[10:13])
        
        # Foot position term
        foot_position = k_foot * (ca.sumsqr(p1 - self.x_des[13:16]) + 
                                  ca.sumsqr(p2 - self.x_des[16:19]) + 
                                  ca.sumsqr(p3 - self.x_des[19:22]) + 
                                  ca.sumsqr(p4 - self.x_des[22:25]))
        
        # Force term
        force = k_force * (ca.sumsqr(F1) + 
                           ca.sumsqr(F2) + 
                           ca.sumsqr(F3) + 
                           ca.sumsqr(F4))
        
        # Foot velocity term
        foot_velocity = k_foot_vel * (ca.sumsqr(v_p1) + 
                                      ca.sumsqr(v_p2) + 
                                      ca.sumsqr(v_p3) + 
                                      ca.sumsqr(v_p4))
        
        # Cost function
        L_ = tracking + orientation + velocity + angular_velocity + foot_position + force + foot_velocity
        
        return L_  # Return the cost function value
    
    def ode(self, params):
        """
        This method is used to define the ordinary differential equation of the optimization problem.
        """     
        # Define symbolic variables for the computations
        state = ca.SX.sym('x', self.model.n_states)
        control = ca.SX.sym('u', self.model.n_controls)
        
        # Define the dynamics of the system
        dynamics = ca.Function('dynamics', 
                               [state, control], 
                               [self.model.dynamics(state, control), self.cost_function(state, control, params)], 
                            )
        
        return dynamics
    
    def integrator(self, params):
        """
        This method is used to define the integrator of the optimization problem.
        """
        # Define the integrator
        ode = self.ode(params)
        X0 = ca.SX.sym('X0', self.model.n_states)
        U = ca.SX.sym('U', self.model.n_controls)
        X = X0
        Q = 0
        for j in range(self.n_int):
            k1, k1_q = ode(X, U)
            k2, k2_q = ode(X + self.h/2 * k1, U)
            k3, k3_q = ode(X + self.h/2 * k2, U)
            k4, k4_q = ode(X + self.h * k3, U)
            X = X+self.h/6*(k1 + 2*k2 + 2*k3 + k4)
            Q = Q + self.h/6*(k1_q + 2*k2_q + 2*k3_q + k4_q)
            # X = X + self.h*k1
            # Q = Q + self.h*k1_q
        self.update_state = ca.Function('update_state', 
                                        [X0, U], 
                                        [X, Q],
                                    )
                
    def set_initial_guess(self, x0, u0):
        """
        This method is used to define the initial values of the optimization variables.
        """    
        step_height = 0.05
        frequency = 4
        t = np.linspace(0, self.T, self.N+1)
        print('Setting initial guess...', end=' ')
        
        x       = np.linspace(x0[0], self.x_des[0], self.N+1)
        y       = np.linspace(x0[1], self.x_des[1], self.N+1)
        z       = np.linspace(x0[2], self.x_des[2], self.N+1)
        qw      = np.linspace(x0[3], self.x_des[3], self.N+1)
        qx      = np.linspace(x0[4], self.x_des[4], self.N+1)
        qy      = np.linspace(x0[5], self.x_des[5], self.N+1)
        qz      = np.linspace(x0[6], self.x_des[6], self.N+1)
        
        vx      = np.linspace(x0[7], self.x_des[7], self.N+1)
        vy      = np.linspace(x0[8], self.x_des[8], self.N+1)
        vz      = np.linspace(x0[9], self.x_des[9], self.N+1)
        wx      = np.linspace(x0[10], self.x_des[10], self.N+1)
        wy      = np.linspace(x0[11], self.x_des[11], self.N+1)
        wz      = np.linspace(x0[12], self.x_des[12], self.N+1)
        
        # Contact points
        p1_x    = np.linspace(x0[13], self.x_des[13], self.N+1)
        p1_y    = np.linspace(x0[14], self.x_des[14], self.N+1)
        p1_z    = np.maximum(step_height*np.sin(2*np.pi*frequency*t), 0)
        p2_x    = np.linspace(x0[15], self.x_des[15], self.N+1)
        p2_y    = np.linspace(x0[16], self.x_des[16], self.N+1)
        p2_z    = np.maximum(step_height*np.sin(2*np.pi*frequency*t - np.pi/4), 0)
        p3_x    = np.linspace(x0[17], self.x_des[17], self.N+1)
        p3_y    = np.linspace(x0[18], self.x_des[18], self.N+1)
        p3_z    = np.maximum(step_height*np.sin(2*np.pi*frequency*t + np.pi/4), 0)
        p4_x    = np.linspace(x0[19], self.x_des[19], self.N+1)
        p4_y    = np.linspace(x0[20], self.x_des[20], self.N+1)
        p4_z    = np.maximum(step_height*np.sin(2*np.pi*frequency*t - np.pi/2), 0)
        

        for k in range(self.N):
            self.w0 += [x[k], y[k], z[k], qw[k], qx[k], qy[k], qz[k],
                        vx[k], vy[k], vz[k], wx[k], wy[k], wz[k],
                        p1_x[k], p1_y[k], p1_z[k],
                        p2_x[k], p2_y[k], p2_z[k],
                        p3_x[k], p3_y[k], p3_z[k],
                        p4_x[k], p4_y[k], p4_z[k],]
            if k != self.N-1:
                self.w0 += u0
            
        print('Done')
    
    def set_bounds_x(self, states_lb, states_ub):
        """
        This method is used to set the bounds of the state variables.
        """
        self.lbw += states_lb
        self.ubw += states_ub
        
    def set_bounds_u(self, inputs_lb, inputs_ub):
        """
        This method is used to set the bounds of the control input variables.
        """
        self.lbw += inputs_lb
        self.ubw += inputs_ub
        
    def add_constraint(self, g, lbg, ubg, name=None):
        """
        This method is used to add a constraint to the optimization problem.
        """
        if name is not None:
            for constraint in self.constraints:
                if constraint.name == name:
                    raise ValueError(f"Constraint {name} already exists.")
        
        self.constraints += [self.Constraint(
            g=g,
            lbg=np.array([lbg]).flatten(),
            ubg=np.array([ubg]).flatten(),
            name=name,
        )]
        
    def update_constraint(self, name, g=None, lbg=None, ubg=None):
        for constraint in self.constraints:
            if constraint.name == name:
                if g is not None:
                    constraint.g = g
                if lbg is not None:
                    constraint.lbg = np.array([lbg]).flatten()
                if ubg is not None:
                    constraint.ubg = np.array([ubg]).flatten()
                return
            
        raise ValueError(f"Constraint {name} not found.")
    
    def multiple_shooting(self, i, params) -> None:
        """
        This method is used to define the multiple shooting constraint of the optimization problem.
        """
        x_next, J_next = self.update_state(self.Xk, self.Uk)
        self.cost += J_next
        
        self.Xk = ca.SX.sym(f"X_{i+1}", self.model.n_states)
        self.w += [self.Xk]
        self.set_bounds_x(params['state_lower_bound'], params['state_upper_bound'])
        
        
        # Continuity constraint
        self.add_constraint(
            g=[self.Xk - x_next],
            lbg=np.zeros(self.model.n_states),
            ubg=np.zeros(self.model.n_states),
            name=f"Multiple shooting {i}",
        )
    
    def set_control_constraints(self, i, params) -> None:
        """
        This method is used to set the control constraints of the optimization problem.
        
        Args:
            i: int
                Index of the control input.
            u_lb: np.array
                Lower bound of the control input.
            u_ub: np.array
                Upper bound of the control input.
        """
        self.Uk = ca.SX.sym(f"U_{i}", self.model.n_controls)
        self.w += [self.Uk]
        self.set_bounds_u(params['input_lower_bound'], params['input_upper_bound'])
        
        # Friction cone constraints
        for k in range(self.model.n_feet):
            # Contact force components
            # Get components of the force
            fx = self.Uk[3*k]  # Force in x direction
            fy = self.Uk[3*k + 1]  # Force in y direction
            fz = self.Uk[3*k + 2]  # Force in z direction (normal force)

            # Friction cone constraint: sqrt(fx^2 + fy^2) <= mu * fz
            # This ensures the tangential forces don't exceed friction limit
            tangential_force = fx**2 + fy**2
            normal_force = fz
            
            # Add friction cone constraint
            self.add_constraint(
                g=[tangential_force - (self.model.friction * normal_force)**2],
                lbg=[-ca.inf],
                ubg=[0],
                name=f"Friction cone {i}_{k+1}"
            )
        
        # Force can only push, not pull (normal force must be positive)
        # self.add_constraint(
        #     g=[normal_force],
        #     lbg=[0],
        #     ubg=[ca.inf],
        #     name=f"Normal force {i}"
        # )
        
    def set_bounding_box(self, i) -> None:
        """
        This method is used to set the bounding box constraints of the optimization problem.
        
        Args:
            i: int
                Index of the control input.
        """
        # Extract the feet position from the state
        # Extract feet positions from the state
        # Each foot has 3 components (x, y, z)
        r = self.Xk[0:3]         # CoM position
        q = self.Xk[3:7]         # CoM orientation (quaternion)
        p1 = self.Xk[13:16]      # First foot position
        p2 = self.Xk[16:19]      # Second foot position
        p3 = self.Xk[19:22]      # Third foot position
        p4 = self.Xk[22:25]      # Fourth foot position
        
        R = self.model.quat_to_rot_matrix(q)  # Convert quaternion to rotation matrix
        
        for k, p in enumerate([p1, p2, p3, p4]):
            # Bounding box constraints for each foot
            # Assuming the bounding box is a square with side length 0.1 centered at the foot position
            self.add_constraint(
                g=[ca.transpose(R) @ (p - r)],
                lbg=[-self.model.l_box, -self.model.w_box, -self.model.h_box],
                ubg=[self.model.l_box, self.model.w_box, self.model.h_box],
                name=f"Bounding box foot {i}_{k+1}"
            )
            
    def set_quaternion_constraints(self, i):
        """
        This method is used to set the unit quaternion constraints of the optimization problem.
        Args:
            i: int
                Index.
        """
        # Quaternion normalization constraint: ||q|| = 1
        q = self.Xk[3:7]
        self.add_constraint(
            g=[ca.sumsqr(q) - 1],
            lbg=[0],
            ubg=[0],
            name=f"Quaternion normalization {i}"
        )
        
        # Set a non-negative real part of the quaternion to avoid ambiguity
        # self.add_constraint(
        #     g = [q[0]],  # Ensure the real part (q0) is non-negative
        #     lbg=[0],
        #     ubg=[ca.inf],
        #     name=f"Quaternion non-negative real part {i}"
        # )
            
    def set_obstacle_avoidance(self, i, params):
        """
        This method is used to set the obstacle avoidance constraints of the optimization problem.
        
        Args:
            i: int
                Index of the control input.
            params: dict
                Dictionary containing the parameters of the optimization problem.
        """
        # Obstacle avoidance constraints
        if self.z_step != 0.0 and self.x_step != 0.0:
            for j in range(self.nc_ob):
                x_ob = self.C_OB[j][0]
                y_ob = self.C_OB[j][1]
                # Distance from the contact point to the obstacle center
                distance = ca.sqrt((self.state[i][0] - x_ob)**2 + (self.state[i][1] - y_ob)**2)
                # Constraint: distance >= r_c + r_ob
                self.add_constraint(
                    g=[distance - (self.r_c + self.r_ob)],
                    lbg=[0],
                    ubg=[ca.inf],
                    name=f"Obstacle avoidance {i}_{j}"
                )
        
    def set_terminal_cost(self, gains) -> None:
        """
        This method is used to set the terminal cost of the optimization problem.
        
        Args:
            x_des: np.array
                Desired state.
            gains: dict
                Dictionary containing the gains of the terminal cost.
        """
        self.cost += gains['alpha']*(self.Xk[0] - self.x_des[0])**2 \
                + gains['beta']*(self.Xk[1] - self.x_des[1])**2 \
                + gains['gamma']*(self.Xk[2] - self.x_des[2])**2    
    
    def create_solver(self, opts):
        """
        This method is used to create the solver of the optimization problem.
        """
        g = []
        for constraint in self.constraints:
            g += constraint.g
        
        problem = {
            'f': self.cost,
            'x': ca.vertcat(*self.w),
            'g': ca.vertcat(*g)
        }
                
        # opts = {
        #     'ipopt.max_iter': 5e3,
        #     # 'ipopt.gradient_approximation': 'finite-difference-values',
        #     # 'ipopt.hessian_approximation': 'limited-memory', 
        #     # 'ipopt.hsllib': "/usr/local/libhsl.so",
        #     # 'ipopt.linear_solver': 'mumps',
        #     # 'ipopt.mu_strategy': 'adaptive',
        #     # 'ipopt.adaptive_mu_globalization': 'kkt-error',
        #     'ipopt.tol': 1e-6,
        #     'ipopt.acceptable_tol': 1e-4,
        #     'ipopt.print_level': 3
        # }
                        
        self.solver = ca.nlpsol('solver', 'ipopt', problem, opts)
        
    def solve(self):
        """
        This method is used to solve the optimization problem.
        """
        lbg = np.empty(0)
        ubg = np.empty(0)
        for constraint in self.constraints:
            lbg = np.concatenate((lbg, constraint.lbg))
            ubg = np.concatenate((ubg, constraint.ubg))
        #     print(f"Constraint {constraint.name}")
            
        # input()
        
        print('Solving optimization problem...')
        r = self.solver(
            x0=self.w0,
            lbx=self.lbw, ubx=self.ubw,
            lbg=lbg, ubg=ubg,
        )
        
        sol = r['x'].full().flatten()
        
        self.opt_var_0 = sol
        
        self.extract_solution(sol)
        
    def extract_solution(self, sol):
        """
        This method is used to extract the solution from the optimization problem.
        
        Args:
            sol: np.array
                Solution of the optimization problem.       
        """
        # Collect the optimal trajectory of the CoM
        self.r_opt, self.q_opt, self.v_opt, self.omega_opt = [], [], [], []
        
        # Collect the optimal trajectory of the contact points
        self.p1_opt, self.p2_opt, self.p3_opt, self.p4_opt = [], [], [], []
        
        # Collect the optimal control inputs
        self.F1_opt, self.F2_opt, self.F3_opt, self.F4_opt = [], [], [], []
        self.v_p1_opt, self.v_p2_opt, self.v_p3_opt, self.v_p4_opt = [], [], [], []
        
        for i in range(self.N):
            # Extract state and control variables for this timestep
            offset = i * (self.model.n_states + self.model.n_controls)
            
            # Extract CoM position (first 3 elements of state)
            self.r_opt.append(
                sol[offset:offset+self.model.n_com]
            )
            
            # Extract CoM orientation (next 4 elements)
            self.q_opt.append(
                sol[offset+self.model.n_com:offset+self.model.n_com+self.model.n_quat]
            )
            
            # Extract CoM velocity (next 3 elements)
            self.v_opt.append(
                sol[offset+self.model.n_com+self.model.n_quat:offset+self.model.n_com+self.model.n_quat+self.model.n_linear_vel]
            )
            
            # Extract CoM angular velocity (next 3 elements)
            self.omega_opt.append(
                sol[offset+self.model.n_com+self.model.n_quat+self.model.n_linear_vel:offset+self.model.n_com+self.model.n_quat+self.model.n_linear_vel+self.model.n_angular_vel]
            )
            
            # Extract foot positions
            feet_index = offset+self.model.n_com+self.model.n_quat+self.model.n_linear_vel+self.model.n_angular_vel
            self.p1_opt.append(
                sol[feet_index:feet_index+self.model.n_foot_vars]
            )
            feet_index += self.model.n_foot_vars
            self.p2_opt.append(
                sol[feet_index:feet_index+self.model.n_foot_vars]
            )
            feet_index += self.model.n_foot_vars
            self.p3_opt.append(
                sol[feet_index:feet_index+self.model.n_foot_vars]
            )
            feet_index += self.model.n_foot_vars
            self.p4_opt.append(
                sol[feet_index:feet_index+self.model.n_foot_vars]
            )
            
            # Extract control inputs if not the last timestep
            if i < self.N - 1:
                control_offset = offset + self.model.n_states
                
                # Extract foot forces
                self.F1_opt.append(sol[control_offset:control_offset+self.model.n_forces])
                control_offset += self.model.n_forces
                self.F2_opt.append(sol[control_offset:control_offset+self.model.n_forces])
                control_offset += self.model.n_forces
                self.F3_opt.append(sol[control_offset:control_offset+self.model.n_forces])
                control_offset += self.model.n_forces
                self.F4_opt.append(sol[control_offset:control_offset+self.model.n_forces])
                
                # Extract foot velocities
                control_offset += self.model.n_forces
                self.v_p1_opt.append(sol[control_offset:control_offset+self.model.n_velocities])
                control_offset += self.model.n_velocities
                self.v_p2_opt.append(sol[control_offset:control_offset+self.model.n_velocities])
                control_offset += self.model.n_velocities
                self.v_p3_opt.append(sol[control_offset:control_offset+self.model.n_velocities])
                control_offset += self.model.n_velocities
                self.v_p4_opt.append(sol[control_offset:control_offset+self.model.n_velocities])
        
    def visualize(self):
        """
        This method is used to visualize the solution of the optimization problem.
        """
        
        # Convert lists to numpy arrays for easier plotting
        r_opt = np.array(self.r_opt)
        p1_opt = np.array(self.p1_opt)
        p2_opt = np.array(self.p2_opt)
        p3_opt = np.array(self.p3_opt)
        p4_opt = np.array(self.p4_opt)
        
        # Plot 3D CoM trajectory
        # Create a 3D plot
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot 3D CoM trajectory
        ax.plot(r_opt[:, 0], r_opt[:, 1], r_opt[:, 2], label='CoM Trajectory', color='blue')
        # Add initial and final points
        ax.scatter(r_opt[0, 0], r_opt[0, 1], r_opt[0, 2], color='blue', s=100, label='Initial CoM Position')
        ax.scatter(r_opt[-1, 0], r_opt[-1, 1], r_opt[-1, 2], color='blue', s=100, label='Final CoM Position')
        
        # Plot 3D foot trajectories
        ax.plot(p1_opt[:, 0], p1_opt[:, 1], p1_opt[:, 2], label='LF Foot Trajectory', color='red')
        ax.plot(p2_opt[:, 0], p2_opt[:, 1], p2_opt[:, 2], label='RF Foot Trajectory', color='green')
        ax.plot(p3_opt[:, 0], p3_opt[:, 1], p3_opt[:, 2], label='LH Foot Trajectory', color='orange')
        ax.plot(p4_opt[:, 0], p4_opt[:, 1], p4_opt[:, 2], label='RH Foot Trajectory', color='purple')
        # Add initial and final foot positions
        ax.scatter(p1_opt[0, 0], p1_opt[0, 1], p1_opt[0, 2], color='red', s=100, label='Initial LF Foot Position')
        ax.scatter(p1_opt[-1, 0], p1_opt[-1, 1], p1_opt[-1, 2], color='red', s=100, label='Final LF Foot Position')
        ax.scatter(p2_opt[0, 0], p2_opt[0, 1], p2_opt[0, 2], color='green', s=100, label='Initial RF Foot Position')
        ax.scatter(p2_opt[-1, 0], p2_opt[-1, 1], p2_opt[-1, 2], color='green', s=100, label='Final RF Foot Position')
        ax.scatter(p3_opt[0, 0], p3_opt[0, 1], p3_opt[0, 2], color='orange', s=100, label='Initial LH Foot Position')
        ax.scatter(p3_opt[-1, 0], p3_opt[-1, 1], p3_opt[-1, 2], color='orange', s=100, label='Final LH Foot Position')
        ax.scatter(p4_opt[0, 0], p4_opt[0, 1], p4_opt[0, 2], color='purple', s=100, label='Initial RH Foot Position')
        ax.scatter(p4_opt[-1, 0], p4_opt[-1, 1], p4_opt[-1, 2], color='purple', s=100, label='Final RH Foot Position')
        
        ax.set_xlabel('X Position (m)')
        ax.set_ylabel('Y Position (m)')
        ax.set_zlabel('Z Position (m)')
        ax.set_title('3D CoM and Foot Trajectories')
        ax.legend()
        ax.grid(True)
        # ax.axis.set_box_aspect([1,1,1])  # Equal aspect ratio
        
        plt.show()

    def write_to_csv(self, csv_name, q2_traj, q3_traj, torque2_traj, torque3_traj):
        """
        Writes the trajectories and torques for joints 2 and 3 into a CSV file.

        Args:
            csv_name: str
                The name of the CSV file to write to.
            q2_traj: list
                The trajectory of joint 2.
            q3_traj: list
                The trajectory of joint 3.
            torque2_traj: list
                The torque trajectory of joint 2.
            torque3_traj: list
                The torque trajectory of joint 3.
        """
        with open(csv_name, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["q2", "q3", "torque2", "torque3"])  # Writing the header
            for q2, q3, torque2, torque3 in zip(q2_traj, q3_traj, torque2_traj, torque3_traj):
                writer.writerow([q2, q3, torque2, torque3])
    
    def step(self, x0, u0, params):
        """
        This method is used to solve the optimization problem (use it in an MPC fashion).
        
        Args:
            x0: np.array
                Initial state.
            u0: np.array
                Initial control input.
        """
        self.set_initial_guess(x0, u0)
        
        self.create_solver(params)
        
        return self.solve()
        