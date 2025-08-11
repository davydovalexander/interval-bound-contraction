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