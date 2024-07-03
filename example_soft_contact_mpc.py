import numpy as np
import matplotlib.pyplot as plt
from utils.soft_contact_mpc import SoftContactMPC

# Define the system parameters
params = {
    'm': 1.5,  # mass
    'I': 0.35,  # moment of inertia
    'length': 0.235,  # length
    'width': 0.3,  # width
    'height': 0.4,  # height
    'leg_length': 0.35,  # leg length
    'dmin': 0.03,  # minimum distance
    'l_box': 0.002,  # box length
    'gamma': 10.0,  # penalty on the distance
    'gamma_v': 10.0,  # penalty on the velocity
}

# Define the initial state
x0 = np.array([0.05, -0.0075, 0.0])

# Define the initial control input
u0 = np.array([20.0, 0.0])

# Define the desired state
xd = np.array([1.5, 1.0, 0.0])

# Define the MPC parameters
n_states = 3
n_controls = 2
n_horizon = 10
time_horizon = 0.5

soft_contact_mpc = SoftContactMPC(x0, u0, n_states, n_controls, n_horizon, time_horizon, n_int=1)

# Set bounds for the optimization variables
# Bound controls
v_bw = 10.0
f_bw = 100
lbu = np.array([-f_bw, -v_bw])
ubu = np.array([f_bw, v_bw])
soft_contact_mpc.set_bounds_u(lbu, ubu)

# Bound states
z_g_min = 0.0
z_g_max = 20.0
z_p_min = -0.1
z_p_max = 10.0
vz_g_min = -2000.0
vz_g_max = 2000.0
lb = np.array([z_g_min, z_p_min, vz_g_min])
ub = np.array([z_g_max, z_p_max, vz_g_max])
soft_contact_mpc.set_bounds_x(lb, ub)


# Load the model
soft_contact_mpc.load_model(params)

# Define the gains for the cost function
gains = {
    'k_force': 1e-4,
    'k_vel': 0,
    'k_pos': 1e4,
}
# Create the cost function
soft_contact_mpc.cost_function(gains, xdes=xd)

# # Add initial state constraints
# soft_contact_mpc.add_constraint([soft_contact_mpc.state[0]], np.reshape(x0, (-1,1)), np.reshape(x0, (-1,1)))

# Define the multiple shooting constraints
soft_contact_mpc.integrator()
soft_contact_mpc.multiple_shooting()

# Bounding box constraints
h_box = (params['leg_length']**2 - params['l_box']**2)**(1/2) - params['dmin']  # [m]
for i in range(1, soft_contact_mpc.N+1):
    bounding_box = [soft_contact_mpc.state[i][0] - soft_contact_mpc.state[i][1]]
    lower_bound = params['dmin']
    upper_bound = params['dmin'] + h_box
    soft_contact_mpc.add_constraint(bounding_box, lower_bound, upper_bound)
    
# Set terminal cost
terminal_gains = {
    'alpha': 0,
    'beta': 0.0,
    'gamma': 0,
}
soft_contact_mpc.set_terminal_cost(xd, terminal_gains)

# Set initial guess
soft_contact_mpc.opt_var_0, z0_g, z0_p, Fz_p0, vz_p0, terrain = soft_contact_mpc.init_values(x0, u0)

tgrid0 = np.linspace(0, soft_contact_mpc.T, soft_contact_mpc.N)

# Plot initial condition
plt.figure(1)
plt.plot(tgrid0, z0_g, '-o', label='z_G')
plt.plot(tgrid0, z0_p, '-o', label='z_P')
plt.plot(tgrid0, terrain, '-', label='terrain')
plt.xlabel('t')
plt.ylabel('pos')
plt.legend()
plt.grid()

# Plot controls
plt.figure(2)
plt.subplot(2, 1, 1)
plt.plot(tgrid0, Fz_p0, '-o', label='Fz_P')
plt.xlabel('t')
plt.ylabel('u')
plt.legend()
plt.grid()
plt.subplot(2, 1, 2)
plt.plot(tgrid0, vz_p0, '-o', label='vz_P') 
plt.xlabel('t')
plt.ylabel('u')
plt.legend()
plt.grid()
# plt.show()

# Create the optimization problem
soft_contact_mpc.create_solver()

z_g_opt, z_p_opt, vz_g_opt, Fz_p_opt, vz_p_opt, Fz_p_opt_sig, vz_p_opt_sig = soft_contact_mpc.solve()


# Plot the results
tgrid = np.linspace(0, soft_contact_mpc.T, soft_contact_mpc.N)
#print(tgrid)

# print optimal trajectory
plt.figure(3)
plt.plot(tgrid, z_g_opt, '-o', label='z_G')
plt.plot(tgrid, z_p_opt, '-o', label='z_P')
plt.plot(tgrid, terrain, '-', label='terrain')
plt.xlabel('t')
plt.ylabel('pos')
plt.legend()
plt.grid()

# print velocity
plt.figure(4)
plt.plot(tgrid, vz_g_opt, '-o', label='vz_G')
plt.xlabel('t')
plt.ylabel('vel')
plt.legend()
plt.grid()


# print optimal controls
plt.figure(5)
plt.subplot(2, 1, 1)
plt.step(tgrid, Fz_p_opt, '-o', label='Fz_P', where='post')
plt.step(tgrid, Fz_p_opt_sig, '-o', label='Fz_P_sig', where='post')
plt.xlabel('t')
plt.ylabel('u')
plt.legend()
plt.grid()
plt.subplot(2, 1, 2)
plt.plot(tgrid, vz_p_opt, '-o', label='vz_P')
plt.plot(tgrid, vz_p_opt_sig, '-o', label='vz_P_sig')
plt.xlabel('t')
plt.ylabel('u')
plt.legend()
plt.grid()
plt.show()