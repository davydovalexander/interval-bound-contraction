import torch
import numpy as np
import matplotlib.pyplot as plt

from nn_controller import NN_IBP
from dynamics import pendulum_f, pendulum_control_matrix
from torchdiffeq import odeint

# 
# LaTeX + Font Setup
# 
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    "axes.labelsize": 9.5,      # larger label font
    "font.size": 9,             # base font size
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
})

# 
# Closed-loop dynamics
# 
def f(t, x):
    u_x = model(x.unsqueeze(0)).squeeze()
    dx = pendulum_f(x).view(-1) + (pendulum_control_matrix() @ u_x.view(-1,1)).view(-1)
    return dx

# Load model
model = NN_IBP(input_dim=2, hidden_dims=[16,16], output_scale=40)
model.load_state_dict(torch.load('inverted_pendulum_controller.pth', weights_only=True))
model.eval()

# 
# Simulation parameters
# 
T = 5
dt = 0.01
t = torch.arange(0, T, dt)

# Random ICs
N = 20
thetas = (torch.rand(N) - 0.5) * torch.pi*(75/50)
thetadots = (torch.rand(N) - 0.5) * 9.0
inits = torch.stack([thetas, thetadots], dim=1)

# 
# IEEE single-column figure
# 
fig, ax = plt.subplots(figsize=(3.5, 2.8))   # ~IEEE column width

# 
# Compute vector field grid (NO GRAD)
# 
with torch.no_grad():
    theta_min, theta_max = -75*np.pi/100, 75*np.pi/100
    thetadot_min, thetadot_max = -4.5, 4.5

    n_grid = 30
    theta = torch.linspace(theta_min, theta_max, n_grid)
    thetadot = torch.linspace(thetadot_min, thetadot_max, n_grid)

    TH, THD = torch.meshgrid(theta, thetadot, indexing='ij')

    dTH = torch.zeros_like(TH)
    dTHD = torch.zeros_like(THD)

    for i in range(n_grid):
        for j in range(n_grid):
            x = torch.tensor([TH[i, j], THD[i, j]], dtype=torch.float32)
            dx = f(0, x)
            dTH[i, j] = dx[0]
            dTHD[i, j] = dx[1]

# 
# Vector field (thicker arrows)
# 
ax.quiver(
    TH, THD, dTH, dTHD,
    color='gray', alpha=0.8
)

# 
# Flow lines (thicker)
# 
for k in range(N):
    init = inits[k]
    xs = odeint(f, init, t, method='dopri5', rtol=1e-6, atol=1e-8)
    xs_np = xs.detach().numpy()
    ax.plot(xs_np[:,0], xs_np[:,1], linewidth=1.2, alpha=0.9)

# 
# Labels & formatting
# 
ax.set_xlabel(r"$\theta$ (rad)", labelpad=1)
ax.set_ylabel(r"$\dot{\theta}$ (rad/s)", labelpad=1)
ax.set_title(r"Closed-Loop Vector Field", fontsize=10)

ax.set_xlim(theta_min, theta_max)
ax.set_ylim(thetadot_min, thetadot_max)

# Make ticks denser + readable
ax.tick_params(axis='both', which='major', length=3, width=0.8)

plt.tight_layout(pad=0.1)
plt.savefig('pendulum-vector-field-new.pdf', dpi=400, bbox_inches='tight')
plt.show()
