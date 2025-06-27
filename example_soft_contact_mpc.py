import matplotlib.pyplot as plt
import numpy as np
import yaml


from utils.soft_contact_mpc import SoftContactMPC
from utils.models.srb_3D_model import SingleRigidBody3D



PLOT = True

# Define the system parameters
# Load system parameters from YAML file
with open('config/params.yaml', 'r') as file:
    params = yaml.safe_load(file)
    
# Load the problem parameters
with open('config/problem.yaml', 'r') as file:
    problem_params = yaml.safe_load(file)

# Define the model
model = SingleRigidBody3D(params['model'])

# Initialize the MPC
soft_contact_mpc = SoftContactMPC(model)

# Set up the problem
soft_contact_mpc.setup_problem(problem_params)

# Create opt problem
soft_contact_mpc.create_solver(problem_params['solver'])

# Solve the optimization problem
sol = soft_contact_mpc.solve()

# Visualize the results
if PLOT:
    soft_contact_mpc.visualize()









