import torch

def partition_hyperrectangle(xunder, xover, n_parts):
    """
    Partition a hyperrectangle [xunder, xover] into n_parts equally sized sub-rectangles.

    Args:
        xunder (torch.Tensor): Lower corner, shape (d,)
        xover (torch.Tensor): Upper corner, shape (d,)
        n_parts (int): Number of partitions (must be a^d for some integer a)

    Returns:
        sub_xunder (torch.Tensor): Lower corners of sub-rectangles, shape (n_parts, d)
        sub_xover (torch.Tensor): Upper corners of sub-rectangles, shape (n_parts, d)
    """
    dim = xunder.shape[0]
    # Determine number of splits along each axis
    splits_per_dim = round(n_parts ** (1.0 / dim))
    assert splits_per_dim ** dim == n_parts, \
        "n_parts must be a perfect power of the number of dimensions"

    # Create equally spaced points along each dimension
    edges = [
        torch.linspace(xunder[i], xover[i], splits_per_dim + 1)
        for i in range(dim)
    ]

    # Lower and upper bounds for each small rectangle
    lower_grid = torch.stack(torch.meshgrid(*[e[:-1] for e in edges], indexing='ij'), dim=-1)
    upper_grid = torch.stack(torch.meshgrid(*[e[1:] for e in edges], indexing='ij'), dim=-1)

    sub_xunder = lower_grid.reshape(-1, dim)
    sub_xover = upper_grid.reshape(-1, dim)

    return sub_xunder, sub_xover

def compute_metzler_upper_bound(jac_bounds, controller_bounds):
    # from interval bounds on P * Df and P * B * Du, compute entrywise upper
    # bounds on |A(x)|_{Mzr} and return it
    
    jac_lower, jac_upper = jac_bounds       # (b, n, n)
    Du_lower, Du_upper = controller_bounds  # (b, n, n)
    
    # Transpose only the last two dims for batching
    jac_lower_T = jac_lower.transpose(-1, -2)
    jac_upper_T = jac_upper.transpose(-1, -2)
    Du_lower_T = Du_lower.transpose(-1, -2)
    Du_upper_T = Du_upper.transpose(-1, -2)
    
    # Compute LMI lower and upper bounds
    lmi_lower = 0.5 * (jac_lower + jac_lower_T + Du_lower + Du_lower_T)
    lmi_upper = 0.5 * (jac_upper + jac_upper_T + Du_upper + Du_upper_T)
    
    # Elementwise maximum for the Metzler bound
    mat_abs = torch.maximum(lmi_upper, -lmi_lower)
    
    # Extract diagonals in batched way
    diag_mat_abs = torch.diagonal(mat_abs, dim1=-2, dim2=-1)
    diag_lmi_upper = torch.diagonal(lmi_upper, dim1=-2, dim2=-1)
    
    # Zero out diagonal of mat_abs, then replace with lmi_upper diagonal
    result = mat_abs - torch.diag_embed(diag_mat_abs) + torch.diag_embed(diag_lmi_upper)
    
    return result

def compute_metzler_nonconstant_NCM(jac_bounds, 
                                    controller_bounds, 
                                    Mdot_f_bounds,
                                    Mdot_Bu_bounds):
    # from interval bounds on P * Df and P * B * Du, compute entrywise upper
    # bounds on |A(x)|_{Mzr} and return it
    
    jac_lower, jac_upper = jac_bounds       # (b, n, n)
    Du_lower, Du_upper = controller_bounds  # (b, n, n)
    Mdot_f_lower, Mdot_f_upper = Mdot_f_bounds # (b, n, n)
    Mdot_Bu_lower, Mdot_Bu_upper = Mdot_Bu_bounds # (b, n, n)

    # Transpose only the last two dims for batching
    jac_lower_T = jac_lower.transpose(-1, -2)
    jac_upper_T = jac_upper.transpose(-1, -2)
    Du_lower_T = Du_lower.transpose(-1, -2)
    Du_upper_T = Du_upper.transpose(-1, -2)
    
    # Compute LMI lower and upper bounds
    lmi_lower = (jac_lower + jac_lower_T + Du_lower + Du_lower_T) + Mdot_f_lower + Mdot_Bu_lower
    lmi_upper = (jac_upper + jac_upper_T + Du_upper + Du_upper_T) + Mdot_f_upper + Mdot_Bu_upper
    
    # Elementwise maximum for the Metzler bound
    assert (lmi_lower <= lmi_upper).all(), "Interval invalid: lmi_lower > lmi_upper"
    mat_abs = torch.maximum(lmi_upper, -lmi_lower)
    
    # Extract diagonals in batched way
    diag_mat_abs = torch.diagonal(mat_abs, dim1=-2, dim2=-1)
    diag_lmi_upper = torch.diagonal(lmi_upper, dim1=-2, dim2=-1)
    
    # Zero out diagonal of mat_abs, then replace with lmi_upper diagonal
    result = mat_abs - torch.diag_embed(diag_mat_abs) + torch.diag_embed(diag_lmi_upper)
    
    return result

def multiply_two_interval_matrices(A_lower, A_upper, B_lower, B_upper):
    """
    Computes tight interval bounds for C = A @ B
    given interval bounds on A and B.

    Args:
        A_lower, A_upper: (..., n, m) interval bounds for A
        B_lower, B_upper: (..., m, p) interval bounds for B

    Returns:
        C_lower, C_upper: (..., n, p) interval bounds for C
    """
    # Expand dimensions so we can broadcast multiply over k
    # Shapes: (..., n, m, 1) and (..., 1, m, p)
    A_l = A_lower.unsqueeze(-1)
    A_u = A_upper.unsqueeze(-1)
    B_l = B_lower.unsqueeze(-3)
    B_u = B_upper.unsqueeze(-3)

    # All 4 combinations for each term in the sum
    prod1 = A_l * B_l
    prod2 = A_l * B_u
    prod3 = A_u * B_l
    prod4 = A_u * B_u

    # Termwise lower and upper bounds (before summing over k)
    term_lower = torch.minimum(
        torch.minimum(prod1, prod2),
        torch.minimum(prod3, prod4)
    )
    term_upper = torch.maximum(
        torch.maximum(prod1, prod2),
        torch.maximum(prod3, prod4)
    )

    # Sum over k (the shared dimension between A and B)
    C_lower = term_lower.sum(dim=-2)
    C_upper = term_upper.sum(dim=-2)

    return C_lower, C_upper

def bound_Mdot(grad_M_lower, grad_M_upper, f_lower, f_upper):
    """
    Compute bounds on Mdot_{ij} = grad_M_{ij}^T f
    given bounds on grad_M and f.

    Args:
        grad_M_lower: (b, n, n, d) lower bounds on gradient of M entries
        grad_M_upper: (b, n, n, d) upper bounds on gradient of M entries
        f_lower: (b, d) lower bound on f
        f_upper: (b, d) upper bound on f

    Returns:
        Mdot_lower: (b, n, n) lower bounds on Mdot
        Mdot_upper: (b, n, n) upper bounds on Mdot
    """
    # Expand f bounds to match grad_M bounds
    f_lower_exp = f_lower[:, None, None, :]  # (b, 1, 1, d)
    f_upper_exp = f_upper[:, None, None, :]  # (b, 1, 1, d)

    # Compute all 4 combinations for each coordinate
    prod1 = grad_M_lower * f_lower_exp
    prod2 = grad_M_lower * f_upper_exp
    prod3 = grad_M_upper * f_lower_exp
    prod4 = grad_M_upper * f_upper_exp

    # Elementwise min/max across the 4 products
    prod_min = torch.min(torch.min(prod1, prod2), torch.min(prod3, prod4))
    prod_max = torch.max(torch.max(prod1, prod2), torch.max(prod3, prod4))

    # Sum over the last dimension to get dot product bounds
    Mdot_lower = prod_min.sum(dim=-1)
    Mdot_upper = prod_max.sum(dim=-1)

    return Mdot_lower, Mdot_upper