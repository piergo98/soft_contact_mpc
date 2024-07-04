import casadi as ca
import numpy as np

class SoftContactMPC:
    class Constraint():
        def __init__(self, g, lbg, ubg, name=None):
            self.g = g
            self.lbg = lbg
            self.ubg = ubg
            self.name = name
    def __init__(self, x0, u0, nx, nu, N, T, n_int=2):
        # Number of states
        self.nx = nx
        # Number of control input
        self.nu = nu
        # Number of time steps
        self.N = N
        # Time horizon
        self.T = T
        # Discretization time step
        self.DT = T / N
        # Number of integrations between two steps
        self.n_int = n_int
        # Integration step
        self.h = self.DT / self.n_int
        # Number of optimization variables
        self.n_opti = (self.nx + self.nu) * (self.N) + self.nx
        
        # Define symbolic variables for the computations
        self.x = ca.MX.sym('x', self.nx)
        self.u = ca.MX.sym('u', self.nu)
        
        # Define the symbolic variables used for the optimization
        self.state = [ca.MX.sym("X_0", self.nx)]
        self.inputs = []
        self.opt_var = self.state.copy()
        for i in range(self.N):
            self.inputs += [ca.MX.sym(f"U_{i}", self.nu)]
            self.state += [ca.MX.sym(f"X_{i+1}", self.nx)]
            self.opt_var += [self.inputs[i], self.state[i+1]]
                
        # self.opt_var_0 = self.init_values(x0, u0)
        
        self.lb_opt_var = - np.ones(self.n_opti) * np.inf
        self.ub_opt_var =   np.ones(self.n_opti) * np.inf
        
        self.cost = 0
        self.constraints = []
        
    def sigmoid_function(self, gamma=1000, gamma_v=1000):
        """
        This method is used to define the sigmoid function.
        
        Args:
            z: float
                Input value.
            gamma: float
                Gain parameter.
                
        Returns:
            float
                Output value of the sigmoid function.
        """
        # Contact activation function
        z = ca.SX.sym('z')
        sig = 1 / (1 + ca.exp(-gamma*z))
        sig_v = 1 / (1 + ca.exp(-gamma_v*z))
        Sigmoid = ca.Function('Sigmoid', [z], [sig])
        Sigmoid_V = ca.Function('Sigmoid_V', [z], [sig_v])
        
        return Sigmoid, Sigmoid_V
    
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
        x_t = ca.MX.sym('x_t')
        # h_t = h_j * (1 / (1 + exp(-300*(x_t - x_j))))          # Step function bild like a sigmoid
        h_t = slope*x_t
        # h_t = 0.0*sin(1000*x_t)
        h_terrain = ca.Function('h_terrain', [x_t], [h_t])
        
        return h_terrain
    
    def load_model(self, parameters):
        """
        This method is used to load the model parameters and to build the model that
        approximates a leg of the quadruped robot.
        
        Args:
            parameters: dict
                Dictionary containing the model parameters.
        """
        ## Model parameters
        # Mass
        m = parameters['m'] 
        # Inertia
        I = parameters['I']
        # Quadruped dimensions
        length = parameters['length']
        width = parameters['width']  
        heigth = parameters['height']  

        # Leg length
        leg_length = parameters['leg_length']
        dmin = parameters['dmin']
        l_box = parameters['l_box']
        self.gamma = parameters['gamma']
        self.gamma_v = parameters['gamma_v']
        
        h_box = (leg_length**2 - l_box**2)**(1/2) - dmin
        # Gravity
        g = 9.81
        
        # Extract the sigmoid functions
        sig, sig_v = self.sigmoid_function(self.gamma, self.gamma_v)
        
        # Contact forces
        Fz_P = sig(-self.x[1])*self.u[0]
        
        # Single rigid body single contact point 1D (jumping leg)
        x_dot0 = self.x[2]
        x_dot1 = sig_v(self.x[1])*(self.u[1] + self.x[2])  
        x_dot2 = Fz_P/m - g

        self.x_dot = ca.vertcat(x_dot0, x_dot1, x_dot2)
        
    def cost_function(self, gains, xdes=None):
        """
        This method is used to define the cost function of the optimization problem.
        
        Args:
            gains: dict
                Dictionary containing the gains of the cost function.
        """
        # Check if the desired state is defined
        if xdes is None:
            xdes = np.zeros(self.nx)
        
        x_dot = self.x_dot
        
        k_force = gains['k_force']
        k_vel = gains['k_vel']
        k_pos = gains['k_pos']
        
        # Cost function
        self.L = k_pos*(self.x[0] - xdes[0])**2 + k_vel*(self.x[2] - xdes[2])**2
    
    def ode(self):
        """
        This method is used to define the ordinary differential equation of the optimization problem.
        """     
        
        dynamics = ca.Function('dynamics', [self.x, self.u], [self.x_dot, self.L], ['x0', 'u0'], ['xf', 'qf'])
        
        return dynamics
    
    def integrator(self):
        """
        This method is used to define the integrator of the optimization problem.
        """
        # Define the integrator
        ode = self.ode()
        X0 = ca.MX.sym('X0', self.nx)
        U = ca.MX.sym('U', self.nu)
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
        self.update_state = ca.Function('update_state', [X0, U], [X, Q], ['x0', 'u0'], ['xf', 'qf'])
        
    def init_values(self, x0, u0):
        """
        This method is used to define the initial values of the optimization variables.
        """    
        # Initial dynamics propagation with constant input
        w0 = []
        w0 += x0.tolist()
        x0_k = x0
        u0_k = u0.tolist()
        print(f"u0_k: {u0_k}")
        z0_g = []  
        z0_p = []
        Fz_p0 = []
        vz_p0 = []
        
        # Set the initial state in the constraints of the optimization variables
        self.lb_opt_var[:self.nx] = x0.tolist()
        self.ub_opt_var[:self.nx] = x0.tolist()    
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
    
    def set_bounds_x(self, inputs_lb, inputs_ub):
        """
        This method is used to set the bounds of the state variables.
        """
        for i in range(self.N):
            self.lb_opt_var[i*(self.nx+self.nu):i*(self.nx+self.nu)+self.nx] = inputs_lb
            self.ub_opt_var[i*(self.nx+self.nu):i*(self.nx+self.nu)+self.nx] = inputs_ub
        
    def set_bounds_u(self, inputs_lb, inputs_ub):
        """
        This method is used to set the bounds of the control input variables.
        """
        for i in range(self.N):
            self.lb_opt_var[i*(self.nx+self.nu)+self.nx:i*(self.nx+self.nu)+self.nx+self.nu] = inputs_lb
            self.ub_opt_var[i*(self.nx+self.nu)+self.nx:i*(self.nx+self.nu)+self.nx+self.nu] = inputs_ub
        
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
    
    def multiple_shooting(self):
        """
        This method is used to define the multiple shooting constraint of the optimization problem.
        """
        for i in range(self.N):
            F = self.update_state(x0=self.state[i], u0=self.inputs[i])
            Xk_end = F['xf']
            self.cost += F['qf']
            
            # Continuity constraint
            self.add_constraint(
                g=[Xk_end - self.state[i+1]],
                lbg=np.zeros(self.nx),
                ubg=np.zeros(self.nx),
                name=f"Multiple shooting {i}",
            )
    
    def set_terminal_cost(self, xdes, gains):
        """
        This method is used to set the terminal cost of the optimization problem.
        
        Args:
            xdes: np.array
                Desired state.
            gains: dict
                Dictionary containing the gains of the terminal cost.
        """
        self.cost += gains['alpha']*(self.state[-1][0] - xdes[0])**2 + gains['beta']*(self.state[-1][1] - xdes[1])**2 + gains['gamma']*(self.state[-1][2] - xdes[2])**2    
    
    def create_solver(self):
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
                
        opts = {
            'ipopt.max_iter': 5e3,
            # 'ipopt.gradient_approximation': 'finite-difference-values',
            # 'ipopt.hessian_approximation': 'limited-memory', 
            # 'ipopt.hsllib': "/usr/local/libhsl.so",
            # 'ipopt.linear_solver': 'mumps',
            # 'ipopt.mu_strategy': 'adaptive',
            # 'ipopt.adaptive_mu_globalization': 'kkt-error',
            'ipopt.tol': 1e-6,
            'ipopt.acceptable_tol': 1e-4,
            'ipopt.print_level': 3
        }
                        
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
            z_g_opt += [sol[i*(self.nx+self.nu)]]
            z_p_opt += [sol[i*(self.nx+self.nu) + 1]]
            vz_g_opt += [sol[i*(self.nx+self.nu) + 2]]
            Fz_p_opt += [(sol[i*(self.nx+self.nu) + 3])]
            vz_p_opt += [(sol[i*(self.nx+self.nu) + 4])]
            Fz_p_opt_sig += [((sol[i*(self.nx+self.nu) + 3])*sig(-z_p_opt[i])).full().flatten()[0]]
            vz_p_opt_sig += [((sol[i*(self.nx+self.nu) + 4])*sig_v(z_p_opt[i])).full().flatten()[0]]
            

        return z_g_opt, z_p_opt, vz_g_opt, Fz_p_opt, vz_p_opt, Fz_p_opt_sig, vz_p_opt_sig
    
    def step(self, x0, u0):
        """
        This method is used to solve the optimization problem (use it in an MPC fashion).
        
        Args:
            x0: np.array
                Initial state.
            u0: np.array
                Initial control input.
        """
        self.init_values(x0, u0)
        
        self.create_solver()
        
        return self.solve()
        