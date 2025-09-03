import torch
import torch.nn as nn

class NN_IBP(nn.Module):
    def __init__(self, input_dim=3, hidden_dims=[16, 16], output_dim=1, activation='softplus', trainable_NCM=False):
        # Make number of layers modular 
        super().__init__()

        self.hidden_layers = nn.ModuleList()
        self.activations = nn.ModuleList()
        if trainable_NCM:
            self.P = nn.Linear(input_dim, input_dim, bias = False)
        else:
            self.P = nn.Linear(input_dim, input_dim, bias = False)
            self.P.weight.requires_grad = False

        dims = [input_dim] + hidden_dims
        for i in range(len(hidden_dims)):
            self.hidden_layers.append(nn.Linear(dims[i], dims[i+1]))
            if activation == 'softplus':
                self.activations.append(nn.Softplus())
            # Todo: support tanh and relu activations
            else:
                raise ValueError(f"Unsupported activation: {activation}")

        self.output_layer = nn.Linear(hidden_dims[-1], output_dim)

    def constant_NCM(self):
        return self.P.weight @ torch.transpose(self.P.weight, 0, 1) + torch.eye(self.P.weight.shape[-1])

    def forward(self, x):
        for layer, act in zip(self.hidden_layers, self.activations):
            x = act(layer(x))
        return self.output_layer(x)

    def IBP_forward(self, lower, upper, ellision_matrix = None):
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

        # Output layer (no activation here, but same bound logic)
        W = self.output_layer.weight.clone()
        b = self.output_layer.bias.clone()
        if ellision_matrix is not None:
            W = ellision_matrix @ W
            b = ellision_matrix @ b

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
        Compute bounds on W @ P, given bounds on P.

        Args:
            P_lower, P_upper: (b, n, m)
            W: (k, n) constant matrix

        Returns:
            (lower, upper): (b, k, m)
        """
        # Use batch matrix multiply: (b, k, n) @ (b, n, m)
        W_exp = W.unsqueeze(0).expand(P_lower.shape[0], -1, -1)  # (b, k, n)

        X1 = torch.bmm(W_exp, P_lower)
        X2 = torch.bmm(W_exp, P_upper)

        lower = torch.minimum(X1, X2)
        upper = torch.maximum(X1, X2)

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
        