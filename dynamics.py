import torch

def polynomial_f(x):
    # Polynomial vector field
    # TODO - make batched compatible
    return torch.tensor([-x[0] + x[2], 
    x[0]**2 - x[1] - 2*x[0]*x[2] + x[2], 
    -x[1]])

def polynomial_Df(x):
    """
    Jacobian of polynomial vector field for batched input.
    
    Args:
        x: Tensor of shape (b, 3) or (3,)
    Returns:
        Jacobian: Tensor of shape (b, 3, 3)
    """
    # Ensure batch dimension
    if x.ndim == 1:
        x = x.unsqueeze(0)

    batch_size = x.shape[0]
    J = torch.zeros(batch_size, 3, 3, dtype=x.dtype, device=x.device)

    # Fill in the Jacobian
    J[:, 0, 0] = -1
    J[:, 0, 1] = 0
    J[:, 0, 2] = 1

    J[:, 1, 0] = 2*x[:, 0] - 2*x[:, 2]
    J[:, 1, 1] = -1
    J[:, 1, 2] = -2*x[:, 0] + 1

    J[:, 2, 0] = 0
    J[:, 2, 1] = -1
    J[:, 2, 2] = 0

    return J


def polynomial_jac_bounds(l, u):
    # Interval bounds on Jacobian of polynomial dynamics given interval
    # bounds on the input
    # l, u: (b, 3) or (3,) if unbatched
    # Ensure batching
    if l.ndim == 1:
        l = l.unsqueeze(0)
        u = u.unsqueeze(0)
    
    B = l.shape[0]
    device = l.device
    dtype = l.dtype
    
    lower = torch.zeros((B, 3, 3), device=device, dtype=dtype)
    upper = torch.zeros((B, 3, 3), device=device, dtype=dtype)
    
    # Constant entries
    lower[:, 0, 0] = -1
    lower[:, 0, 2] = 1
    lower[:, 1, 1] = -1
    lower[:, 2, 1] = -1
    
    upper[:, 0, 0] = -1
    upper[:, 0, 2] = 1
    upper[:, 1, 1] = -1
    upper[:, 2, 1] = -1
    
    # Fill in l/u dependent entries
    lower[:, 1, 0] = 2*l[:, 0] - 2*u[:, 2]
    lower[:, 1, 2] = -2*u[:, 0] + 1
    
    upper[:, 1, 0] = 2*u[:, 0] - 2*l[:, 2]
    upper[:, 1, 2] = -2*l[:, 0] + 1
    
    return (lower, upper)

def polynomial_control_matrix():
    # Function for the constant control matrix B for the polynomial dynamics
    return torch.tensor([[0.], [0.], [1.]])

def pendulum_f(x):
    g = 10.
    l = 1.
    return torch.tensor([[x[1]], [g/l * torch.sin(x[0])]])

def pendulum_Df(x):
    """
    Jacobian of pendulum vector field for batched input.
    
    Args:
        x: Tensor of shape (b, 2) or (2,)
    Returns:
        Jacobian: Tensor of shape (b, 2, 2)
    """
    g = 10.
    l = 1.
    # Ensure batch dimension
    if x.ndim == 1:
        x = x.unsqueeze(0)

    batch_size = x.shape[0]
    J = torch.zeros(batch_size, 2, 2, dtype=x.dtype, device=x.device)

    # Fill in the Jacobian
    J[:, 0, 0] = 0.
    J[:, 0, 1] = 1.

    J[:, 1, 0] = g/l * torch.cos(x[:, 0])
    J[:, 1, 1] = 0

    return J

def pendulum_jac_bounds(l, u):
    # Interval bounds on Jacobian of pendulum dynamics given interval
    # bounds on the input
    # l, u: (b, 2) or (2,) if unbatched
    g = 10.
    ell = 1.
    # Ensure batching
    if l.ndim == 1:
        l = l.unsqueeze(0)
        u = u.unsqueeze(0)
    
    b = l.shape[0]
    device = l.device
    dtype = l.dtype
    
    lower = torch.zeros((b, 2, 2), device=device, dtype=dtype)
    upper = torch.zeros((b, 2, 2), device=device, dtype=dtype)
    
    # Constant entries
    lower[:, 0, 0] = 0.
    lower[:, 0, 1] = 1
    lower[:, 1, 1] = 0.
    
    upper[:, 0, 0] = 0
    upper[:, 0, 1] = 1
    upper[:, 1, 1] = 0.
    
    # Fill in l/u dependent entries
    # lower[:, 1, 0] = -g/ell
    # upper[:, 1, 0] = g/ell*torch.maximum(-torch.cos(l[:,0]), -torch.cos(u[:,0]))
    lower[:, 1, 0] = g/ell*torch.minimum(torch.cos(l[:,0]), torch.cos(u[:,0]))
    upper[:, 1, 0] = g/ell
    
    return (lower, upper)

def pendulum_control_matrix():
    # Function for the constant control matrix B for the pendulum dynamics
    m = 1.
    l = 1.
    return torch.tensor([[0.], [1/(m*l**2)]])