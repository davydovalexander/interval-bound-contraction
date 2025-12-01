import torch
import torch.nn as nn

class NN_IBP(nn.Module):
    def __init__(self, input_dim=3, hidden_dims=[16, 16], output_dim=1,
                 activation='softplus', output_scale=40.0):
        super().__init__()
        self.hidden_layers = nn.ModuleList()
        self.activations = nn.ModuleList()
        self.output_scale = float(output_scale)

        dims = [input_dim] + hidden_dims
        for i in range(len(hidden_dims)):
            self.hidden_layers.append(nn.Linear(dims[i], dims[i+1]))
            if activation == 'softplus':
                self.activations.append(nn.Softplus())
            else:
                raise ValueError(f"Unsupported activation: {activation}")

        self.output_layer = nn.Linear(hidden_dims[-1], output_dim)

    # ------------------------------
    # Compute network output at input zero 
    # ------------------------------
    def _compute_zero_output(self, device=None, dtype=None):
        h0 = torch.zeros(1, self.hidden_layers[0].in_features,
                         device=device if device else next(self.parameters()).device,
                         dtype=dtype if dtype else next(self.parameters()).dtype)
        for layer, act in zip(self.hidden_layers, self.activations):
            h0 = act(layer(h0))
        z0 = self.output_layer(h0)
        return z0

    # ------------------------------
    # Forward pass
    # ------------------------------
    def forward(self, x):
        h = x
        for layer, act in zip(self.hidden_layers, self.activations):
            h = act(layer(h))
        z0 = self._compute_zero_output(device=x.device, dtype=x.dtype)
        z = self.output_layer(h) - z0
        y = self.output_scale * torch.tanh(z / self.output_scale)
        return y

    # ------------------------------
    # Scaled tanh IBP / derivative bounds
    # ------------------------------
    def _scaled_tanh_ibp(self, l, u):
        s = self.output_scale
        return s * torch.tanh(l / s), s * torch.tanh(u / s)

    def _scaled_tanh_derivative_bounds(self, l, u):
        s = self.output_scale
        dl = l / s
        du = u / s
        d1 = 1.0 - torch.tanh(dl)**2
        d2 = 1.0 - torch.tanh(du)**2
        d_lower = torch.minimum(d1, d2)
        d_upper_endpoints = torch.maximum(d1, d2)
        crosses_zero = (l <= 0.0) & (u >= 0.0)
        one = torch.ones_like(d_upper_endpoints)
        d_upper = torch.where(crosses_zero, one, d_upper_endpoints)
        return d_lower, d_upper

    # ------------------------------
    # IBP forward pass
    # ------------------------------
    def IBP_forward(self, lower, upper, elision_matrix=None, return_preact=False):
        for layer, act in zip(self.hidden_layers, self.activations):
            W = layer.weight
            b = layer.bias
            W_pos = torch.clamp(W, min=0)
            W_neg = torch.clamp(W, max=0)
            lower_new = lower @ W_pos.T + upper @ W_neg.T + b
            upper_new = upper @ W_pos.T + lower @ W_neg.T + b
            lower = act(lower_new)
            upper = act(upper_new)

        W = self.output_layer.weight.clone()
        b = self.output_layer.bias.clone()
        if elision_matrix is not None:
            W = elision_matrix @ W
            b = elision_matrix @ b

        lower_pre = lower @ torch.clamp(W, min=0).T + upper @ torch.clamp(W, max=0).T + b
        upper_pre = upper @ torch.clamp(W, min=0).T + lower @ torch.clamp(W, max=0).T + b

        # Subtract dynamic z0
        z0 = self._compute_zero_output(device=lower.device, dtype=lower.dtype)
        lower_pre = lower_pre - z0
        upper_pre = upper_pre - z0

        lower_post, upper_post = self._scaled_tanh_ibp(lower_pre, upper_pre)

        if return_preact:
            return lower_pre, upper_pre, lower_post, upper_post
        return lower_post, upper_post

    # ------------------------------
    # Hidden layer bounds / derivatives
    # ------------------------------
    def get_hidden_pre_activation_bounds(self, x_lower, x_upper):
        def linear_interval(layer, l, u):
            W = layer.weight
            b = layer.bias
            W_pos = torch.clamp(W, min=0)
            W_neg = torch.clamp(W, max=0)
            lower = l @ W_pos.T + u @ W_neg.T + b
            upper = u @ W_pos.T + l @ W_neg.T + b
            return lower, upper

        def activation_interval(act, l, u):
            return act(l), act(u)

        bounds = []
        l, u = x_lower, x_upper
        for layer, act in zip(self.hidden_layers, self.activations):
            l_pre, u_pre = linear_interval(layer, l, u)
            bounds.append((l_pre, u_pre))
            l, u = activation_interval(act, l_pre, u_pre)
        return bounds

    def activation_derivative(self, x):
        return torch.sigmoid(x)  # softplus derivative

    def get_diagonal_bounds_from_intermediate(self, pre_act_bounds):
        diagonal_bounds = []
        for l, u in pre_act_bounds:
            dl = self.activation_derivative(l)
            du = self.activation_derivative(u)
            diagonal_bounds.append((dl, du))
        return diagonal_bounds

    def get_diag_bounds(self, l, u):
        pre_act_bounds = self.get_hidden_pre_activation_bounds(l, u)
        return self.get_diagonal_bounds_from_intermediate(pre_act_bounds)

    # ------------------------------
    # Interval matrix multiplication utilities
    # ------------------------------
    def left_multiply_by_diag(self, J_lower, J_upper, P_lower, P_upper):
        J_lower_exp = J_lower.unsqueeze(-1)
        J_upper_exp = J_upper.unsqueeze(-1)
        ll = J_lower_exp * P_lower
        lu = J_lower_exp * P_upper
        ul = J_upper_exp * P_lower
        uu = J_upper_exp * P_upper
        lower = torch.minimum(torch.minimum(ll, lu), torch.minimum(ul, uu))
        upper = torch.maximum(torch.maximum(ll, lu), torch.maximum(ul, uu))
        return lower, upper

    def left_multiply_by_constant_matrix(self, P_lower, P_upper, W):
        W_pos = torch.clamp(W, min=0)
        W_neg = torch.clamp(W, max=0)
        lower = torch.einsum('kn,bnm->bkm', W_pos, P_lower) + torch.einsum('kn,bnm->bkm', W_neg, P_upper)
        upper = torch.einsum('kn,bnm->bkm', W_pos, P_upper) + torch.einsum('kn,bnm->bkm', W_neg, P_lower)
        return lower, upper
    
    # ------------------------------
    # Full product bound
    # ------------------------------
    def compute_full_product_bound(self, diag_bounds_list, elision_matrix):
        batch = diag_bounds_list[0][0].shape[0]
        W1 = self.hidden_layers[0].weight.clone()
        P_lower = W1.unsqueeze(0).expand(batch, -1, -1).clone()
        P_upper = W1.unsqueeze(0).expand(batch, -1, -1).clone()

        for i in range(1, len(self.hidden_layers)):
            J_lower, J_upper = diag_bounds_list[i - 1]
            P_lower, P_upper = self.left_multiply_by_diag(J_lower, J_upper, P_lower, P_upper)
            W_i = self.hidden_layers[i].weight
            P_lower, P_upper = self.left_multiply_by_constant_matrix(P_lower, P_upper, W_i)

        J_lower, J_upper = diag_bounds_list[-1]
        P_lower, P_upper = self.left_multiply_by_diag(J_lower, J_upper, P_lower, P_upper)

        W_out = self.output_layer.weight.clone()
        if elision_matrix is not None:
            W_out = elision_matrix @ W_out

        return self.left_multiply_by_constant_matrix(P_lower, P_upper, W_out)

    # ------------------------------
    # Final Jacobian bounds with output derivative
    # ------------------------------
    def compute_Du_bounds(self, l, u, elision_matrix=None):
        diag_bounds = self.get_diag_bounds(l, u)
        pre_lower, pre_upper = self.compute_full_product_bound(diag_bounds, elision_matrix)
        lower_pre, upper_pre, _, _ = self.IBP_forward(l, u, elision_matrix=elision_matrix, return_preact=True)
        d_lower, d_upper = self._scaled_tanh_derivative_bounds(lower_pre, upper_pre)
        return self.left_multiply_by_diag(d_lower, d_upper, pre_lower, pre_upper)


## This class doesn't have input saturation or u(0) = 0
class NN_IBP_old(nn.Module):
    def __init__(self, input_dim=3, hidden_dims=[16, 16], output_dim=1, activation='softplus'):
        # Make number of layers modular 
        super().__init__()

        self.hidden_layers = nn.ModuleList()
        self.activations = nn.ModuleList()

        dims = [input_dim] + hidden_dims
        for i in range(len(hidden_dims)):
            self.hidden_layers.append(nn.Linear(dims[i], dims[i+1]))
            if activation == 'softplus':
                self.activations.append(nn.Softplus())
            # Todo: support tanh and relu activations
            else:
                raise ValueError(f"Unsupported activation: {activation}")

        self.output_layer = nn.Linear(hidden_dims[-1], output_dim)

    def forward(self, x):
        for layer, act in zip(self.hidden_layers, self.activations):
            x = act(layer(x))
        return self.output_layer(x)

    def IBP_forward(self, lower, upper, elision_matrix = None):
        """
        Interval Bound Propagation forward pass.
        Args:
            lower (tensor): shape (batch, in_dim)
            upper (tensor): shape (batch, in_dim)
        Returns:
            lower_out, upper_out (tensors): shape (batch, out_dim)
        """
        # Pass through hidden layers
        for layer, act in zip(self.hidden_layers, self.activations):
            W = layer.weight
            b = layer.bias

            W_pos = torch.clamp(W, min=0)
            W_neg = torch.clamp(W, max=0)

            # Linear transformation
            lower_new = lower @ W_pos.T + upper @ W_neg.T + b
            upper_new = upper @ W_pos.T + lower @ W_neg.T + b

            # Activation (monotonic)
            lower = act(lower_new)
            upper = act(upper_new)

        # Output layer, no activation
        W = self.output_layer.weight.clone()
        b = self.output_layer.bias.clone()
        # Premultiply by constant matrix B or MB if M is constant
        if elision_matrix is not None:
            W = elision_matrix @ W
            b = elision_matrix @ b

        W_pos = torch.clamp(W, min=0)
        W_neg = torch.clamp(W, max=0)

        lower_out = lower @ W_pos.T + upper @ W_neg.T + b
        upper_out = upper @ W_pos.T + lower @ W_neg.T + b

        return lower_out, upper_out

    def get_hidden_pre_activation_bounds(self, x_lower, x_upper):
        """
        Compute pre-activation interval bounds for each hidden layer.

        Args:
            x_lower: Tensor of shape (b, input_dim)
            x_upper: Tensor of shape (b, input_dim)

        Returns:
            List of tuples (l_i, u_i): pre-activation bounds at each hidden layer
        """
        def linear_interval(layer: nn.Linear, l, u):
            W = layer.weight
            b = layer.bias
            W_pos = torch.clamp(W, min=0)
            W_neg = torch.clamp(W, max=0)

            lower = torch.matmul(l, W_pos.T) + torch.matmul(u, W_neg.T) + b
            upper = torch.matmul(u, W_pos.T) + torch.matmul(l, W_neg.T) + b
            return lower, upper

        def activation_interval(act, l, u):
            return act(l), act(u)

        bounds = []

        l, u = x_lower, x_upper
        for layer, act in zip(self.hidden_layers, self.activations):
            l_pre, u_pre = linear_interval(layer, l, u)
            bounds.append((l_pre, u_pre))
            l, u = activation_interval(act, l_pre, u_pre)

        return bounds
    
    def activation_derivative(self, x):
        # only works for softplus activation
        return torch.sigmoid(x)

    def get_diagonal_bounds_from_intermediate(self, pre_act_bounds):
        """
        Compute bounds on each of the J_i.

        Args:
            pre_act_bounds: List of pre-activation interval bounds (z_k_lower, z_k_upper)

        Returns:
            List of tuples (l_i, u_i): upper and lower bounds on the diagonal entries of each J_i
        """
        diagonal_bounds = []
        for i in range(len(pre_act_bounds)):
            lower_diag_bound = self.activation_derivative(pre_act_bounds[i][0])
            upper_diag_bound = self.activation_derivative(pre_act_bounds[i][1])
            diagonal_bounds.append((lower_diag_bound, upper_diag_bound))
        return diagonal_bounds

    def get_diag_bounds(self, l, u):
        # From interval bounds on the inputs, directly gets bounds on J_i
        pre_act_bounds = self.get_hidden_pre_activation_bounds(l, u)
        return self.get_diagonal_bounds_from_intermediate(pre_act_bounds)
    
    def left_multiply_by_diag(self, J_lower, J_upper, P_lower, P_upper):
        """
        Compute bounds on J @ P, where J is diagonal
        with diag entries in [J_lower, J_upper], and P is an interval matrix.

        Args:
            J_lower, J_upper: (b, n)
            P_lower, P_upper: (b, n, m)

        Returns:
            (lower, upper): (b, n, m)
        """
        # Expand diagonal bounds to match P's shape for broadcasting
        J_lower_exp = J_lower.unsqueeze(-1)  # (b, n, 1)
        J_upper_exp = J_upper.unsqueeze(-1)  # (b, n, 1)

        # Four possible products
        ll = J_lower_exp * P_lower
        lu = J_lower_exp * P_upper
        ul = J_upper_exp * P_lower
        uu = J_upper_exp * P_upper

        # Elementwise min/max over the four cases
        lower = torch.minimum(torch.minimum(ll, lu), torch.minimum(ul, uu))
        upper = torch.maximum(torch.maximum(ll, lu), torch.maximum(ul, uu))

        return lower, upper


    def left_multiply_by_constant_matrix(self, P_lower, P_upper, W):
        """
        Interval multiplication: W @ P_lower/P_upper
        """
        b, n, m = P_lower.shape
        k = W.shape[0]
        
        W_exp = W.unsqueeze(0).expand(b, -1, -1)  # (b, k, n)
        
        # Compute all four combinations
        P_lower_exp = P_lower.unsqueeze(1).expand(-1, k, -1, -1)  # (b, k, n, m)
        P_upper_exp = P_upper.unsqueeze(1).expand(-1, k, -1, -1)  # (b, k, n, m)
        W_exp_exp = W_exp.unsqueeze(-1)  # (b, k, n, 1)
        
        terms = torch.stack([
            W_exp_exp * P_lower_exp,
            W_exp_exp * P_upper_exp
        ], dim=0)  # (2, b, k, n, m)
        
        lower = torch.min(terms, dim=0).values.sum(dim=2)  # sum over n
        upper = torch.max(terms, dim=0).values.sum(dim=2)
        
        return lower, upper

    def compute_full_product_bound(self, diag_bounds_list, elision_matrix):
        """
        Computes interval bounds on the product:
        W_out · J_{N-1} · W_{N-1} · J_{N-2} · ... · J_1 · W_1

        Batched version.

        Args:
            diag_bounds_list: list of (J_lower_i, J_upper_i), each of shape (b, n_i)
                            length = N-1
            elision_matrix: optional matrix to multiply with W_out at the end

        Returns:
            (M_lower, M_upper): tensors of shape (b, k, m) — bounds on the final matrix product
        """
        batch_size = diag_bounds_list[0][0].shape[0]
        assert len(diag_bounds_list) == len(self.hidden_layers), \
            f"Expected {len(self.hidden_layers)} diag bounds but got {len(diag_bounds_list)}"

        # Step 1: Start from the rightmost matrix: W_1
        W1 = self.hidden_layers[0].weight.clone()  # (n1, m1)
        P_lower = W1.unsqueeze(0).expand(batch_size, -1, -1).clone()
        P_upper = W1.unsqueeze(0).expand(batch_size, -1, -1).clone()

        # Step 2: Loop through J_1, W_2, ..., J_N
        for i in range(1, len(self.hidden_layers)):
            # Multiply on the left with diag(J_i)
            J_lower, J_upper = diag_bounds_list[i - 1]  # (b, n_i)
            P_lower, P_upper = self.left_multiply_by_diag(J_lower, J_upper, P_lower, P_upper)

            # Multiply on the left with constant W_i
            W_i = self.hidden_layers[i].weight  # (n_{i+1}, n_i)
            P_lower, P_upper = self.left_multiply_by_constant_matrix(P_lower, P_upper, W_i)

        # Step 3: Final left multiplication with J_N
        J_lower, J_upper = diag_bounds_list[-1]
        P_lower, P_upper = self.left_multiply_by_diag(J_lower, J_upper, P_lower, P_upper)

        # Step 4: Multiply by W_out
        W_out = self.output_layer.weight.clone()  # (k, n_last)
        if elision_matrix is not None:
            W_out = elision_matrix @ W_out
        final_lower, final_upper = self.left_multiply_by_constant_matrix(P_lower, P_upper, W_out)

        return final_lower, final_upper

    
    def compute_Du_bounds(self, l, u, elision_matrix = None):
        diag_bounds = self.get_diag_bounds(l, u)
        return self.compute_full_product_bound(diag_bounds, elision_matrix)
        