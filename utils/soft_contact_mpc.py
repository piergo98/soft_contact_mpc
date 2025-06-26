import casadi as ca
import numpy as np
import csv

class SoftContactMPC:
    class Constraint():
        def __init__(self, g, lbg, ubg, name=None):
            self.g = g
            self.lbg = lbg
            self.ubg = ubg
            self.name = name
    def __init__(self, model, x0, u0, nx, nu, N, T, n_int=2):
        
        # Load the model
        self.model = model
        
    def set_up_problem(self, params, **kwargs):
        """
        This method is used to set up the optimization problem.
        Args:
            params: dict
                Dictionary containing the parameters of the optimization problem.
        """
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
        self.u0 = params['initial_control']
        
        # Define symbolic variables for the computations
        self.x = ca.SX.sym('x', self.model.n_states)
        self.u = ca.SX.sym('u', self.model.n_controls)
        
        # Default values
        self.h_step = 0.0
        self.x_step = 0.0
        self.step = 0.0
        epsilon = 0
            
        for key, value in kwargs.items():
            print(f"Setting {key} to {value}")
            if key == 'h_t':
                self.h_step = value
            elif key == 'x_t':
                self.x_step = value
                epsilon = 0.3
            elif key == 'x_motion':
                self.x_step = value
            else:
                self.x_step = 1.0
        
        # Desired final state
        x_fin = self.x0[0] + self.x_step + epsilon
        y_fin = self.x0[1] + self.h_step
        self.x_des = np.array([x_fin, y_fin, 0.0, x_fin+self.model.l3, self.h_step, x_fin-self.model.l3, self.h_step, 0.0, 0.0, 0.0]).reshape(-1, 1)
        
        # Vector centers for obstacle avoidance            
        self.r_c = 0.03     # Radius contact points in m
        self.r_ob = 0.01    # Radius obstacle circles in m
        self.nc_ob = int(self.h_step/(2*self.r_ob) + 1)   # number of collision circle needed
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
        self.w0 += self.u0
        
        # Set all the methods for the optimization problem
        self.terrain(slope=params['slope'])
        self.integrator()
        
        for i in range(self.N):
            self.set_control_constraints(i, params)
            self.multiple_shooting(i, params)
            self.set_bounding_box(i)
            
        self.set_terminal_cost()
        
        # Set the initial guess for the optimization variables
        self.set_initial_guess()
    
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
        
    def cost_function(self, gains, xdes=None):
        """
        This method is used to define the cost function of the optimization problem.
        
        Args:
            gains: dict
                Dictionary containing the gains of the cost function.
        """
        # Check if the desired state is defined
        if xdes is None:
            xdes = np.zeros(self.model.n_states)
        
        k_force = gains['k_force']
        k_vel = gains['k_vel']
        k_pos = gains['k_pos']
        
        # Cost function
        self.L = k_pos*(self.x[0] - xdes[0])**2 + k_vel*(self.x[2] - xdes[2])**2
    
    def ode(self):
        """
        This method is used to define the ordinary differential equation of the optimization problem.
        """     
        
        dynamics = ca.Function('dynamics', 
                               [self.x, self.u], 
                               [self.model.dynamics(self.x, self.u), self.L], 
                               ['x0', 'u0'], 
                               ['xf', 'qf']
                            )
        
        return dynamics
    
    def integrator(self):
        """
        This method is used to define the integrator of the optimization problem.
        """
        # Define the integrator
        ode = self.ode()
        X0 = ca.SX.sym('X0', self.model.n_states)
        U = ca.SX.sym('U', self.model.n_controls)
        X = X0
        Q = 0
        for j in range(self.n_int):
            k1, k1_q = ode(X, U)
            # k2, k2_q = ode(X + self.h/2 * k1, U)
            # k3, k3_q = ode(X + self.h/2 * k2, U)
            # k4, k4_q = ode(X + self.h * k3, U)
            # X = X+self.h/6*(k1 + 2*k2 + 2*k3 + k4)
            # Q = Q + self.h/6*(k1_q + 2*k2_q + 2*k3_q + k4_q)
            X = X + self.h*k1
            Q = Q + self.h*k1_q
        self.update_state = ca.Function('update_state', 
                                        [X0, U], 
                                        [X, Q],
                                    )
                
    def set_initial_guess(self, x0, u0):
        """
        This method is used to define the initial values of the optimization variables.
        """    
        # Initial dynamics propagation with constant input
        w0 = []
        w0 += x0.tolist()
        x0_k = x0
        u0_k = u0.tolist()

        z0_g = []  
        z0_p = []
        Fz_p0 = []
        vz_p0 = []
        
        # Set the initial state in the constraints of the optimization variables
        self.lb_opt_var[:self.model.n_states] = x0.tolist()
        self.ub_opt_var[:self.model.n_states] = x0.tolist()    
        # Define the terrain profile
        h_terrain = self.terrain()

        terrain = []
        for k in range(self.N):
            w0 += u0_k
            terrain += [(h_terrain(k/100)).full().flatten()[0]]
            x0_k = self.update_state(x0=x0_k, u0=u0_k)  # return a DM type structure

            x0_k = x0_k['xf'].full().flatten()

            w0 += x0_k.tolist()
            
            # Extract CoM and contact points state
            z0_g += [x0_k[0]]
            z0_p += [x0_k[1]]

            # Extract controls
            Fz_p0 += [(u0_k[0])]   
            vz_p0 += [((u0_k[1]))]
            
        
        return w0, z0_g, z0_p, Fz_p0, vz_p0, terrain
    
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
        x_next, J_next = self.update_state(x0=self.state[i], u0=self.inputs[i])
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
            tangential_force = ca.sqrt(fx**2 + fy**2)
            normal_force = fz
            
            # Add friction cone constraint
            self.add_constraint(
                g=[tangential_force - self.model.friction * normal_force],
                lbg=[-ca.inf],
                ubg=[0],
                name=f"Friction cone {k}"
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
        
        R = self.model.quat_to_rot(q)  # Convert quaternion to rotation matrix
        
        for k, p in enumerate([p1, p2, p3, p4]):
            # Bounding box constraints for each foot
            # Assuming the bounding box is a square with side length 0.1 centered at the foot position
            self.add_constraint(
                g=[ca.transpose(R) @ (p - r)],
                lbg=[-self.model.l_box, -self.model.w_box, -self.model.h_box],
                ubg=[self.model.l_box, self.model.w_box, self.model.h_box],
                name=f"Bounding box foot {k+1}"
            )
            
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
        if self.h_step != 0.0 and self.x_step != 0.0:
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
            xdes: np.array
                Desired state.
            gains: dict
                Dictionary containing the gains of the terminal cost.
        """
        self.cost += gains['alpha']*(self.state[-1][0] - self.xdes[0])**2 \
                + gains['beta']*(self.state[-1][1] - self.xdes[1])**2 \
                + gains['gamma']*(self.state[-1][2] - self.xdes[2])**2    
    
    def create_solver(self, params):
        """
        This method is used to create the solver of the optimization problem.
        """
        g = []
        for constraint in self.constraints:
            g += constraint.g
        
        problem = {
            'f': self.cost,
            'x': ca.vertcat(*self.opt_var),
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
                        
        self.solver = ca.nlpsol('solver', 'ipopt', problem, params['ipopt'])
        
    def solve(self):
        """
        This method is used to solve the optimization problem.
        """
        lbg = np.empty(0)
        ubg = np.empty(0)
        for constraint in self.constraints:
            lbg = np.concatenate((lbg, constraint.lbg))
            ubg = np.concatenate((ubg, constraint.ubg))
    
        r = self.solver(
            x0=self.opt_var_0,
            lbx=self.lb_opt_var, ubx=self.ub_opt_var,
            lbg=lbg, ubg=ubg,
        )
        
        sol = r['x'].full().flatten()
        
        self.opt_var_0 = sol
        
        # add optimal solution extraction and plot
        # Optimal trajectory CoM
        z_g_opt = []
        # Optimal velocity CoM
        vz_g_opt = []
        # Optimal trajectory contact points
        z_p_opt = []
        # Optimal controls
        Fz_p_opt = []
        vz_p_opt = []
        # Optimal controls with sigmoid function
        Fz_p_opt_sig = []
        vz_p_opt_sig = []
        
        sig, sig_v = self.sigmoid_function(self.gamma, self.gamma_v)
        
        for i in range(self.N):
            z_g_opt += [sol[i*(self.model.n_states+self.model.n_controls)]]
            z_p_opt += [sol[i*(self.model.n_states+self.model.n_controls) + 1]]
            vz_g_opt += [sol[i*(self.model.n_states+self.model.n_controls) + 2]]
            Fz_p_opt += [(sol[i*(self.model.n_states+self.model.n_controls) + 3])]
            vz_p_opt += [(sol[i*(self.model.n_states+self.model.n_controls) + 4])]
            Fz_p_opt_sig += [((sol[i*(self.model.n_states+self.model.n_controls) + 3])*sig(-z_p_opt[i])).full().flatten()[0]]
            vz_p_opt_sig += [((sol[i*(self.model.n_states+self.model.n_controls) + 4])*sig_v(z_p_opt[i])).full().flatten()[0]]
            

        return z_g_opt, z_p_opt, vz_g_opt, Fz_p_opt, vz_p_opt, Fz_p_opt_sig, vz_p_opt_sig


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
    
    def step(self, x0, u0):
        """
        This method is used to solve the optimization problem (use it in an MPC fashion).
        
        Args:
            x0: np.array
                Initial state.
            u0: np.array
                Initial control input.
        """
        self.set_initial_guess(x0, u0)
        
        self.create_solver()
        
        return self.solve()
        