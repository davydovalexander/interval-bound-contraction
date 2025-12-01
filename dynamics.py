import torch

def polynomial_f(x):
    # Polynomial vector field
    # 
    return torch.tensor([-x[0] + x[2], 
    x[0]**2 - x[1] - 2*x[0]*x[2] + x[2], 
    -x[1]])

def polynomial_f_bounds(l, u):
    # Ensure batching
    if l.ndim == 1:
        l = l.unsqueeze(0)
        u = u.unsqueeze(0)
    
    b = l.shape[0]
    device = l.device
    dtype = l.dtype
    
    lower = torch.zeros((b, 3), device=device, dtype=dtype)
    upper = torch.zeros((b, 3), device=device, dtype=dtype)

    # f1 = -x1 + x3
    lower[:, 0] = -u[:, 0] + l[:, 2]
    upper[:, 0] = -l[:, 0] + u[:, 2]

    # f3 = -x2
    lower[:, 2] = -u[:, 1]
    upper[:, 2] = -l[:, 1]

    # products for f2 = x1^2 - x2 - 2 x1 x3 + x3
    four_prods = torch.stack([
        l[:, 0] * l[:, 2],
        l[:, 0] * u[:, 2],
        u[:, 0] * l[:, 2],
        u[:, 0] * u[:, 2]
    ], dim=0)  # shape (4, b)

    max_fp, _ = torch.max(four_prods, dim=0)
    min_fp, _ = torch.min(four_prods, dim=0)

    lower[:, 1] = (torch.minimum(u[:, 0]**2, l[:, 0]**2)
                   - u[:, 1] + l[:, 2]
                   - 2 * max_fp)

    upper[:, 1] = (torch.maximum(u[:, 0]**2, l[:, 0]**2)
                   - l[:, 1] + u[:, 2]
                   - 2 * min_fp)

    return lower, upper



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

    # Jacobian
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
    # Interval bounds on Jacobian of polynomial dynamics
    # 
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
    # inverted pendulum drift term
    g = 10.
    l = 1.
    return torch.tensor([[x[1]], [g/l * torch.sin(x[0])]])

def sin_interval(l, u):
    """
    Tight interval bound for sin([l,u]).
    Handles arbitrary-size intervals.
    """
    # start with endpoints
    s_l = torch.sin(l)
    s_u = torch.sin(u)

    smin = torch.minimum(s_l, s_u)
    smax = torch.maximum(s_l, s_u)

    # If interval width >= 2π → sin covers full range
    full = (u - l) >= 2*torch.pi
    smin = torch.where(full, -torch.ones_like(smin), smin)
    smax = torch.where(full,  torch.ones_like(smax), smax)

    # critical points for sin: π/2 + 2kπ (max = 1), -π/2 + 2kπ (min = -1)
    # Compute the nearest k for each bound
    k1 = torch.ceil((l - torch.pi/2) / (2*torch.pi))
    crit_max = k1 * 2*torch.pi + torch.pi/2
    contains_max = crit_max <= u

    k2 = torch.ceil((l + torch.pi/2) / (2*torch.pi))
    crit_min = k2 * 2*torch.pi - torch.pi/2
    contains_min = crit_min <= u

    # apply corrections
    smax = torch.where(contains_max, torch.ones_like(smax), smax)
    smin = torch.where(contains_min, -torch.ones_like(smin), smin)

    return smin, smax

def cos_interval(l, u):
    # use cos(theta) = sin(theta + pi/2)
    return sin_interval(l + torch.pi/2, u + torch.pi/2)

def pendulum_f_bounds(l, u):
    # bounds on f(x)
    if l.ndim == 1:
        l = l.unsqueeze(0); u = u.unsqueeze(0)
    b = l.shape[0]
    device = l.device
    g = 10.; ell = 1.
    lower = torch.zeros((b,2), device=device)
    upper = torch.zeros((b,2), device=device)
    lower[:,0] = l[:,1]; upper[:,0] = u[:,1]
    smin, smax = sin_interval(l[:,0], u[:,0])
    lower[:,1] = g/ell * smin
    upper[:,1] = g/ell * smax
    return lower, upper

def pendulum_Df(x):
    """
    Jacobian of pendulum vector field.
    
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

    # Jacobian
    J[:, 0, 0] = 0.
    J[:, 0, 1] = 1.

    J[:, 1, 0] = g/l * torch.cos(x[:, 0])
    J[:, 1, 1] = 0

    return J

def pendulum_jac_bounds(l, u):
    """
    Tight interval bounds on Jacobian of pendulum dynamics.
    J = [[0, 1],
         [g/l * cos(theta), 0]]
    """
    g = 10.
    ell = 1.

    if l.ndim == 1:
        l = l.unsqueeze(0)
        u = u.unsqueeze(0)

    b = l.shape[0]
    device = l.device
    dtype = l.dtype

    lower = torch.zeros((b, 2, 2), device=device, dtype=dtype)
    upper = torch.zeros((b, 2, 2), device=device, dtype=dtype)

    # constant entries
    lower[:,0,1] = upper[:,0,1] = 1

    # cos interval
    cmin, cmax = cos_interval(l[:,0], u[:,0])

    lower[:,1,0] = g/ell * cmin
    upper[:,1,0] = g/ell * cmax

    return lower, upper

def pendulum_control_matrix():
    # Function for the constant control matrix B for the pendulum dynamics
    m = 1.
    l = 1.
    return torch.tensor([[0.], [1/(m*l**2)]])