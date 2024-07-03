from casadi import *
import matplotlib.pyplot as plt
import numpy as np
# from math import sqrt
import scipy
from scipy import io
# from pathlib import Path, PureWindowsPath            #Use only for Windows
from datetime import date

MAX_OPT_NUMBER = 1

# Direct multiple shooting optimization for contacts dynamics
N_states = 3
N_controls = 2

## Simulation parameters
# Simulation time
T = 0.5
# Horizon length (number of optimization steps)
N = 50
# Discretization step (time between two optimization steps)
DT = T/N
# Number of integrations between two steps
n_int = 2
# Integration step
h = DT/n_int
print(f"int_step = {h}")
## Model parameters
# Mass
m = 1.5  # [kg]
# Inertia
I = 0.35  # [kg*m^2] 0.25
# Quadruped dimensions
length = 0.235 # [m]
width = 0.30  # [m]
heigth = 0.40  # [m]

# Leg length
leg_length = 0.35  # [m]
dmin = 0.03  # [m]
# h_box = 0.11  # [m]
# l_box = (leg_length**2 - (h_box+dmin)**2)**(1/2)  # [m]
# l_box = 0.1077  #[m]
l_box = 0.002 #[m]
h_box = (leg_length**2 - l_box**2)**(1/2) - dmin  # [m]
# l_box = l_box/1.5
# l_box = 0.04
print(f"h_box = {h_box}")
print(f"l_box = {l_box}")

# Bound controls
v_bw = 10.0
f_bw = 60 

# Gravity
g = 9.81

# Soft Contact Param (Sigmoid Model)
mu = 0.6            # Friction cone param
hist = 0.0

# Velocity cones parameter
mu_v = 1.0

# activation Function param
gamma = 1000
gamma_F = gamma #500 #1670
gamma_V = gamma

# Jump task setting
x_j = 0.0           # x where obstacle start
h_j = 0.0           # obstacle height

# Target distanza
target_jump = 0.5    # [m] 
target_forward = 0.2    # [m] 
#Target Velocità
v_xG = 1.5      # [m/s]


## Function to determine the height of the terrain
x_t = MX.sym('x_t')
# h_t = h_j * (1 / (1 + exp(-300*(x_t - x_j))))          # Step function bild like a sigmoid
slope = 0.0
h_t = slope*x_t
# h_t = 0.0*sin(1000*x_t)
h_terrain = Function('h_terrain', [x_t], [h_t])


# Initial Condition
# x = [z_G, z_P, dz_G]
x0 = [0.05, -0.005, 0.0] 
# u = [Fz_P, vz_P]
u0 = [40, 0.0]   # [N, m/s]


# Desired final state
# x_jump = [length/2, x0[1]+h_j + target_jump, 0.0, length, x0[4]+h_j+ target_jump, 0.0, x0[6]+h_j+ target_jump, 0.0, 0.0, 0.0]
# x_des = [length/2 + target_forward, x0[1]+h_j, 0.0, x0[3]+target_forward, x0[4]+h_j, x0[5]+target_forward, x0[6]+h_j, 0.0, 0.0, 0.0] # su X
x_des = x0
x_jump = [2.0, 1.0, 0.0]

print(f"x_des = {x_des}")

# State and control variables
u = MX.sym("u", N_controls)    # control
x = MX.sym("x", N_states)      # states


# L = -1e10*(x[4]**2 + x[6]**2)

########################################################################################################################
## Model Equations 1D Single Rigid Body
########################################################################################################################


# Contact activation function
z = MX.sym('z')
sig = 1 / (1 + exp(-gamma*z))
sig_v = 1 / (1 + exp(-gamma_V*z))
Sigmoid = Function('Sigmoid', [z], [sig])
Sigmoid_V = Function('Sigmoid_V', [z], [sig_v])


# u = [Fz_P, vz_P]
# x = [z_G, z_P, dz_G]

# Contact forces
Fz_P = Sigmoid(-x[1])*u[0]

# Single rigid body single contact point 1D (jumping leg)
x_dot0 = x[2]
x_dot1 = Sigmoid_V(x[1])*(u[1] + x[2])  
x_dot2 = Fz_P/m - g


x_dot = vertcat(x_dot0, x_dot1, x_dot2)

# Objective term
k_force_z = 1e-2
k_vel_z = 1e0
k_zG0 = 0.0
k_pz = 0.0
min_h = 0.0

# L = dot(u, u) + x[2]**2  # + (x[1] - x0[1])**2 + x[2]**2 # (x[0] - x_des[0])**2   # Control cost
# L = dot(x, x)
L = ( k_force_z*(u[0]**2)  
    + k_vel_z*(x_dot1**2)   
    ) #+ 1e12*(x[1]-x_jump[1])**2# + k_pz*((x[4] - min_h)**2 + (x[6] - min_h)**2) + 1e6*((x[1]-x_des[1])**2 + (x[4]-x_des[4])**2 + (x[6]-x_des[6])**2 + (x[0]-x_des[0])**2 + (x[3]-x_des[3])**2 + (x[5]-x_des[5])**2 )

dynamics_contact_HF = Function('dynamics_contact_HF', [x, u], [x_dot, L])

## Integrator Contact H-F Phase
# Fixed step Runge-Kutta 4 integrator

X0 = MX.sym('X0', N_states)
U = MX.sym('U', N_controls)
X = X0
Q = 0
for j in range(n_int):
    k1, k1_q = dynamics_contact_HF(X, U)
    # k2, k2_q = dynamics_contact_HF(X + h/2 * k1, U)
    # k3, k3_q = dynamics_contact_HF(X + h/2 * k2, U)
    # k4, k4_q = dynamics_contact_HF(X + h * k3, U)
    # X = X+h/6*(k1 + 2*k2 + 2*k3 + k4)
    # Q = Q + h/6*(k1_q + 2*k2_q + 2*k3_q + k4_q)
    X = X + h*k1
    Q = Q + h*k1_q
F_contact_HF = Function('F_contact_HF', [X0, U], [X, Q], ['x0', 'u0'], ['xf', 'qf'])

# warm_start = scipy.io.loadmat(r"C:\Users\vince\Documents\UniPi\Magistrale\DA FARE\Tesi\py_lab\Gait-and-trajectory-optimization-for-legged-robot\Workspaces\Two_leg_srb_SC\NotBad\x_0.79_y_0.34_N_400_T_2.0_gamma_600_hist_0.0_2023-02-12.mat")
# warm_start = scipy.io.loadmat("/home/vince/Documents/cp/paper1/Workspaces/Two_legs_srb_SC/task_20cm_1sec/2024-02-19/13:7/x_0.3946_y_0.2_vG_0.22_xj_0.0_ht_0.0_N_100_T_1.0_gamma_1000_ma27_task20cm_1sec_2024-02-19 13:07:48.770729.mat")
# print(warm_start)
# w_last = warm_start['opt_state'][0]



########################################################################################################################
# Evaluate at a test point
#Fk = F_flight(x0=x0, u0=u0, Tp_i=Tp0[0])
# print(Fk['xf'])
# print(Fk['qf'])

# Start with an empty NLP
w = []
w0 = []    # Initial guess
lbw = []
ubw = []
J = 0
G = []
lbg = []
ubg = []

# "Lift" initial conditions
Xk = MX.sym('X_0', N_states)
w += [Xk]   # It will create a structure like [X0, U_0, X1, U_1 ...]
lbw += x0  # bound only for states
ubw += x0

# Initial dynamics propagation with constant input
x0_k = x0
u0_k = u0
w0 += x0_k
w0 += u0_k
z0_g = []  
z0_p = []
Fz_p0 = []
vz_p0 = []

terrain = []
for k in range(N):

    terrain += [(h_terrain(k/100)).full().flatten()[0]]
    x0_k = F_contact_HF(x0=x0_k, u0=u0_k)  # return a DM type structure

    x0_k = x0_k['xf'].full().flatten()

    w0 += x0_k.tolist()
    # Extract CoM and contact points state
    z0_g += [x0_k[0]]
    z0_p += [x0_k[1]]

    # Extract controls
    Fz_p0 += [(u0_k[0])]   
    vz_p0 += [((u0_k[1]))]

    if k != N-1:
        w0 += u0_k



w_last = w0
# Plot the results
tgrid0 = np.linspace(0, T, N)

# print initial condition
plt.figure(1)
plt.plot(tgrid0, z0_g, '-o', label='z_G')
plt.plot(tgrid0, z0_p, '-o', label='z_P')
plt.plot(tgrid0, terrain, '-', label='terrain')
plt.xlabel('t')
plt.ylabel('pos')
plt.legend()
plt.grid()

# print controls
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
plt.show()

input("Press Enter to continue... and solve the NLP!")

for i in range(1, MAX_OPT_NUMBER + 1):
    # Formulate the NLP
    for k in range(N):
        # New NLP variable for the control
        Uk = MX.sym('U_' + str(k), N_controls)
        w += [Uk]
        lbw += [-f_bw, -v_bw]
        ubw += [f_bw, v_bw]

        
        # Integrate till the end of the interval

        Fk = F_contact_HF(x0=Xk, u0=Uk)  # return a DM type structure

        Xk_end = Fk['xf']


        J = J+Fk['qf']

        # New NLP variable for state at end of interval
        Xk = MX.sym('X_' + str(k+1), N_states)
        w += [Xk]
        lbw += [-inf, -inf, -inf]
        ubw += [inf, inf, inf]

        # Add continuity constraint
        G += [Xk_end-Xk]
        lbg += [0, 0, 0]
        ubg += [0, 0, 0]


        # Bounding box for contact point
        G += [Xk[0] - Xk[1]]
        lbg += [dmin]
        ubg += [dmin + h_box]

        # if k == N-1:
            # J = J + 1e12*(Xk[0]-x_jump[0])**2 #+ 1e8*(Xk[4]-x_jump[1])**2 + 1e8*(Xk[6]-x_jump[1])**2

        # if k == N-1:
        #     #   Xk_target = Xk_end
        #     G += [Xk[0]-x_des[0]]
        #     lbg += [-0.01]
        #     ubg += [-0.01]
        #     G += [Xk[1]-x_des[1]]
        #     lbg += [-0.01]
        #     ubg += [0.01]
            # G += [Xk[2]-x_des[2]]
            # lbg += [-0.001]
            # ubg += [0.001]


    # Add final state cost term
    beta = 1e7
    J = J  + beta * ((Xk_end[2]-x_des[2])**2) + 1e12*(Xk_end[0]-x_jump[0])**2

    print(f"J = {J}")
    print(f"g = {G}")

    linear_solver = 'ma27'
    # Create an NLP solver
    prob = {'f': J, 'x': vertcat(*w), 'g': vertcat(*G)}
    # prob = {'x': vertcat(*w), 'g': vertcat(*G)}
    # NLP solver options
    # opts = {'ipopt.max_iter': 1e4, 'warn_initial_bounds': 1,'ipopt.linear_solver': linear_solver, 'ipopt.hessian_approximation': 'limited-memory', 'ipopt.tol': 1e-1}
    opts = {'ipopt.max_iter': 1e4, 'warn_initial_bounds': 1, 'ipopt.tol': 1e-8}#,'ipopt.linear_solver': linear_solver, 'ipopt.tol': 1e-8}
    solver = nlpsol('solver', 'ipopt', prob, opts) 

    # Solve the NLP
    sol = solver(x0=w_last, lbx=lbw, ubx=ubw, lbg=lbg, ubg=ubg)
    w_last = sol['x']


# Extract the solution
w_opt = sol['x'].full().flatten()


# print(f"w_opt = {w_opt}")

# Optimal trajectory CoM
z_g_opt = []
# Optimal trajectory contact points
z_p_opt = []
# Optimal controls
Fz_p_opt = []
vz_p_opt = []

for i in range(N):
    z_g_opt += [w_opt[i*(N_states+N_controls)]]
    z_p_opt += [w_opt[i*(N_states+N_controls) + 1]]

    Fz_p_opt += [((w_opt[i*(N_states+N_controls) + 3])*Sigmoid(-z_p_opt[i])).full().flatten()[0]]
    vz_p_opt += [w_opt[i*(N_states+N_controls) + 4]]





# Plot the results
tgrid = np.linspace(0, T, N)
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

# print optimal controls
plt.figure(4)
plt.subplot(2, 1, 1)
plt.plot(tgrid, Fz_p_opt, '-o', label='Fz_P')
plt.xlabel('t')
plt.ylabel('u')
plt.legend()
plt.grid()
plt.subplot(2, 1, 2)
plt.plot(tgrid, vz_p_opt, '-o', label='vz_P')
plt.xlabel('t')
plt.ylabel('u')
plt.legend()
plt.grid()
plt.show()


#plt.savefig(f'/home/pietro/Tesi/PythonFiles/Gait-and-trajectory-optimization-for-legged-robot/Python_plots/Optimal_control_{x_des[0]}_HC.png')




# Initial dynamics propagation with optimal inputs
xp_k = x0
up_k = [Fz_p_opt[0], vz_p_opt[0]]
# wp = debug list
wp = []
wp += xp_k
wp += up_k
xp_g = []  # [x0[0]]
yp_g = []
for k in range(N):

    controls = [Fz_p_opt[k],vz_p_opt[k]]

    xp_k = F_contact_HF(x0=xp_k, u0=controls)  # return a DM type structure

    xp_k = xp_k['xf']
    #print(f"xp_k = {xp_k}")
    xpt = np.transpose(xp_k.__array__())
    xpl = xpt.tolist()
    xp_k = xpl[0]
    wp += xp_k
    xp_g += [xp_k[0]]
    yp_g += [xp_k[1]]
    if k != N-2:
        wp += up_k

plt.figure(5)
plt.plot(tgrid, xp_g, '-o', label='xp')
plt.plot(tgrid, yp_g, '--', label='yp')
plt.axhline(y=x_des[1])


plt.xlabel('t')
plt.ylabel('pos')
plt.legend()
#plt.savefig(f'/home/pietro/Tesi/PythonFiles/Gait-and-trajectory-optimization-for-legged-robot/Python_plots/Optimal_traj_{x_des[0]}_HC.png')

plt.figure(6)
plt.plot(tgrid, y_F_opt, '-', label='yF')
plt.plot(tgrid, y_H_opt, '--', label='yH')
# plt.axhline(y=x_des[1])
plt.xlabel('t')
plt.ylabel('c_pos')
plt.legend()

plt.figure(7)
plt.plot(x_g_opt, y_g_opt)
plt.xlabel('x [m]')
plt.ylabel('y [m]')
plt.show()

from datetime import datetime
## Save Workspace
today = date.today()
todaytime = datetime.now()
hour = todaytime.hour
minute = todaytime.minute
name_file = f'x_{x_des[0]}_y_{x_des[1]}_vG_{v_xG}_xj_{x_j}_ht_{h_j}_N_{N}_T_{T}_gamma_{gamma}_{linear_solver}_task20cm_1sec_{todaytime}.mat'
time = str(hour) + ':' + str(minute) 
#Linux
import os

dir = f'/home/vince/Documents/cp/Sig_ILC/Mulinex/Workspaces/SingleLeg/Planned_trajectories/' + str(today) + '/' + time
os.makedirs(dir, exist_ok=True)
name_file = dir + '/' + name_file
scipy.io.savemat(name_file, mdict={"opt_state": w_opt, 'prop_state': wp})

# Open script py
f = open("/home/vince/Documents/cp/Sig_ILC/Mulinex/Task_SC_def/single_leg/task_20cm/SIG_ILC_planner_mulinex.py", 'r')

# Leggi le prime righe
first_rows = f.readlines()

# Save rows on a new file
with open(dir + '/' + "opt_info.txt", "w") as info_file:
    for riga in first_rows:
        info_file.write(riga)

# Chiudi il file
f.close()
 

#Windows
# path = Path(r"C:\Users\vince\Documents\UniPi\Magistrale\DA FARE\Tesi\py_lab\Gait-and-trajectory-optimization-for-legged-robot\Workspaces\Two_leg_srb_SC\Jump", name_file)
# scipy.io.savemat(path, mdict={"opt_state": w_opt, 'prop_state': wp})
