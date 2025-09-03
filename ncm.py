import torch
import torch.nn as nn
import torch.nn.functional as F

class NCM_NN(nn.Module):
    """
    Maps x in R^d to a PD matrix M(x) = N(x)^T N(x) + eps * I,
    where N(x) in R^{d x d} is produced by a small Softplus MLP.
    Provides interval bounds on M and on grad_x M_{ij}.
    """
    def __init__(self, d: int, hidden_sizes=(16, 16), eps: float = 1e-3, bias: bool = True):
        super().__init__()
        self.d = d
        self.eps = eps

        layers = []
        in_dim = d
        for h in hidden_sizes:
            layers += [nn.Linear(in_dim, h, bias=bias), nn.Softplus()]
            in_dim = h
        layers += [nn.Linear(in_dim, d * d, bias=bias)]  # final linear, no activation
        self.mlp = nn.Sequential(*layers)

        # Cache the linear layers for Jacobian products
        self.linear_layers = [m for m in self.mlp if isinstance(m, nn.Linear)]

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_uniform_(m.weight, a=5**0.5)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    # ---------- Forward ----------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (d,) or (B, d)
        returns: (d, d) or (B, d, d)
        """
        single = (x.ndim == 1)
        if single:
            x = x.unsqueeze(0)

        B = x.shape[0]
        N_flat = self.mlp(x)                      # (B, d*d)
        N = N_flat.view(B, self.d, self.d)        # (B, d, d)
        M = torch.matmul(N.transpose(-1, -2), N)  # (B, d, d)
        if self.eps != 0.0:
            I = torch.eye(self.d, device=x.device, dtype=x.dtype).unsqueeze(0)
            M = M + self.eps * I
        return M.squeeze(0) if single else M

    # ---------- Interval arithmetic helpers ----------
    @staticmethod
    def _ibp_linear(lower, upper, W: torch.Tensor, b: torch.Tensor):
        """
        IBP through y = x @ W^T + b  (PyTorch Linear)
        lower/upper: (B, in), W: (out, in), b: (out,)
        returns bounds (B, out)
        """
        W_pos = torch.clamp(W, min=0)
        W_neg = torch.clamp(W, max=0)
        y_lower = lower @ W_pos.T + upper @ W_neg.T + b
        y_upper = upper @ W_pos.T + lower @ W_neg.T + b
        return y_lower, y_upper

    @staticmethod
    def _ibp_softplus(lower, upper):
        # Softplus is monotone increasing
        return F.softplus(lower), F.softplus(upper)

    def _preact_bounds_and_N_bounds(self, l: torch.Tensor, u: torch.Tensor):
        """
        IBP through the MLP to produce:
        - list of pre-activation bounds [(z1_lower, z1_upper), ..., (z_{L-1}_lower, z_{L-1}_upper)]
          for each Softplus layer (used for derivative sigmoids)
        - bounds on N_flat (B, d*d) -> reshaped to (B, d, d)
        """
        if l.ndim == 1: l = l.unsqueeze(0)
        if u.ndim == 1: u = u.unsqueeze(0)
        B, din = l.shape
        assert din == self.d

        preacts = []
        lower, upper = l, u
        modules = list(self.mlp)

        i = 0
        while i < len(modules):
            lin = modules[i];  assert isinstance(lin, nn.Linear)
            zL, zU = self._ibp_linear(lower, upper, lin.weight, lin.bias)

            # If next is Softplus, store pre-activation bounds then pass through Softplus
            if i + 1 < len(modules) and isinstance(modules[i + 1], nn.Softplus):
                preacts.append((zL, zU))
                lower, upper = self._ibp_softplus(zL, zU)
                i += 2
            else:
                # Final linear (to d*d). These are N_flat bounds.
                N_lower, N_upper = zL, zU
                break

        N_lower = N_lower.view(B, self.d, self.d)
        N_upper = N_upper.view(B, self.d, self.d)
        return preacts, (N_lower, N_upper)

    @staticmethod
    def _diag_bounds_from_preacts(preacts):
        """
        Given pre-activation bounds, return bounds on J_i = diag(sigmoid(z_i)).
        Since sigmoid is increasing, bounds are [sigmoid(zL), sigmoid(zU)].
        Returns list of (J_lower, J_upper) as (B, n_i) vectors.
        """
        diag_bounds = []
        for zL, zU in preacts:
            J_lower = torch.sigmoid(zL)
            J_upper = torch.sigmoid(zU)
            diag_bounds.append((J_lower, J_upper))
        return diag_bounds

    @staticmethod
    def _left_mul_by_diag_batched(J_lower, J_upper, P_lower, P_upper):
        """
        Bounds for (diag(J) @ P), with batch support.
        J_*: (B, n)
        P_*: (B, n, m)
        returns: (B, n, m)
        """
        JL = J_lower.unsqueeze(-1)  # (B, n, 1)
        JU = J_upper.unsqueeze(-1)
        ll = JL * P_lower
        lu = JL * P_upper
        ul = JU * P_lower
        uu = JU * P_upper
        lower = torch.minimum(torch.minimum(ll, lu), torch.minimum(ul, uu))
        upper = torch.maximum(torch.maximum(ll, lu), torch.maximum(ul, uu))
        return lower, upper

    @staticmethod
    def _left_mul_by_const_batched(P_lower, P_upper, W):
        """
        Bounds for (W @ P) with constant W.
        P_*: (B, n, m), W: (k, n)
        returns: (B, k, m)
        """
        W_pos = torch.clamp(W, min=0)
        W_neg = torch.clamp(W, max=0)
        # Use einsum to broadcast over batch
        lower = torch.einsum('kn,bnm->bkm', W_pos, P_lower) + torch.einsum('kn,bnm->bkm', W_neg, P_upper)
        upper = torch.einsum('kn,bnm->bkm', W_pos, P_upper) + torch.einsum('kn,bnm->bkm', W_neg, P_lower)
        return lower, upper

    def _jacobian_bounds_of_N_flat(self, l, u):
        """
        Bounds on Jacobian d vec(N) / dx as interval matrix:
        result lower/upper: (B, d*d, d).  (out_dim, in_dim)
        Uses product bound: W_L J_{L-1} W_{L-1} ... J_1 W_1
        """
        preacts, _ = self._preact_bounds_and_N_bounds(l, u)
        diag_bounds = self._diag_bounds_from_preacts(preacts)

        B = l.shape[0] if l.ndim > 1 else 1
        W_list = [lin.weight for lin in self.linear_layers]  # [W1, W2, ..., W_L]
        # Start with P = W1 (broadcasted to batch)
        P_lower = W_list[0].unsqueeze(0).expand(B, -1, -1).clone()
        P_upper = P_lower.clone()

        # For each hidden layer i (paired with Softplus), apply J_i then next W
        for i in range(len(diag_bounds)):
            JL, JU = diag_bounds[i]                      # (B, n_i)
            P_lower, P_upper = self._left_mul_by_diag_batched(JL, JU, P_lower, P_upper)
            P_lower, P_upper = self._left_mul_by_const_batched(P_lower, P_upper, W_list[i + 1])

        # Now P_* are bounds on (d vec(N)/dx): (B, d*d, d)
        return P_lower, P_upper

    # ---------- Public: interval bounds on M ----------
    def output_bounds(self, l, u):
        """
        Bounds on M(x) = N(x)^T N(x) + eps I, given x in [l, u].
        Returns: (M_lower, M_upper) both (B, d, d)
        """
        _, (N_lower, N_upper) = self._preact_bounds_and_N_bounds(l, u)

        # Vectorized bounds for N^T N (no Python loops)
        B, d, _ = N_lower.shape
        Nl_i = N_lower.permute(0, 2, 1).unsqueeze(-1)  # (B, d, d, 1)
        Nu_i = N_upper.permute(0, 2, 1).unsqueeze(-1)
        Nl_j = N_lower.unsqueeze(1)                    # (B, 1, d, d)
        Nu_j = N_upper.unsqueeze(1)
        cands = torch.stack([Nl_i*Nl_j, Nl_i*Nu_j, Nu_i*Nl_j, Nu_i*Nu_j], dim=-1)  # (B,d,d,d,4)
        prod_min = cands.min(dim=-1).values  # (B, d, d, d)
        prod_max = cands.max(dim=-1).values
        M_lower = prod_min.sum(dim=2)        # sum over k
        M_upper = prod_max.sum(dim=2)

        if self.eps != 0.0:
            I = torch.eye(self.d, device=N_lower.device, dtype=N_lower.dtype).unsqueeze(0)
            M_lower = M_lower + self.eps * I
            M_upper = M_upper + self.eps * I
        return M_lower, M_upper

    # ---------- Public: interval bounds on grad_x M_{ij} ----------
    def grad_M_bounds(self, l, u):
        """
        Bounds on gradients ∇_x M_{ij}(x) for x in [l,u].
        Returns:
            (grad_lower, grad_upper) with shape (B, d, d, d_in),
            where grad_*(b, i, j, :) bounds ∇_x M_{ij} for batch b.
        """
        if l.ndim == 1: l = l.unsqueeze(0)
        if u.ndim == 1: u = u.unsqueeze(0)
        B, d = l.shape
        assert d == self.d

        # 1) Bounds on N and 2) bounds on J-chain Jacobian of N (flattened)
        preacts, (N_lower, N_upper) = self._preact_bounds_and_N_bounds(l, u)
        G_lower, G_upper = self._jacobian_bounds_of_N_flat(l, u)  # (B, d*d, d_in)

        # Reshape Jacobian bounds to (B, k, i, d_in) where entry is ∇_x N_{k i}
        G_lower = G_lower.view(B, self.d, self.d, d)  # (B, k, i, d_in)
        G_upper = G_upper.view(B, self.d, self.d, d)

        # Prepare index grids to broadcast over all (i, j)
        device = l.device
        I_idx, J_idx = torch.meshgrid(
            torch.arange(self.d, device=device),
            torch.arange(self.d, device=device),
            indexing='ij'
        )  # each (d, d)

        # Gather tensors in (B, k, i, j, d_in) or (B, k, i, j)
        # A-term uses G_{k i} with N_{k j}
        GL_ki = G_lower[:, :, I_idx, :]                    # (B, k, i, j, d_in)
        GU_ki = G_upper[:, :, I_idx, :]                    # (B, k, i, j, d_in)
        NL_kj = N_lower[:, :, J_idx]                       # (B, k, i, j)
        NU_kj = N_upper[:, :, J_idx]                       # (B, k, i, j)

        # B-term uses G_{k j} with N_{k i}
        GL_kj = G_lower[:, :, J_idx, :]                    # (B, k, i, j, d_in)
        GU_kj = G_upper[:, :, J_idx, :]                    # (B, k, i, j, d_in)
        NL_ki = N_lower[:, :, I_idx]                       # (B, k, i, j)
        NU_ki = N_upper[:, :, I_idx]                       # (B, k, i, j)

        # Interval product (vector interval) * (scalar interval) for A-term
        def vec_times_scalar_interval(vL, vU, sL, sU):
            # v*: (B,k,i,j,d_in), s*: (B,k,i,j) -> (B,k,i,j,d_in)
            sL = sL.unsqueeze(-1); sU = sU.unsqueeze(-1)
            c1 = vL * sL
            c2 = vL * sU
            c3 = vU * sL
            c4 = vU * sU
            low = torch.minimum(torch.minimum(c1, c2), torch.minimum(c3, c4))
            upp = torch.maximum(torch.maximum(c1, c2), torch.maximum(c3, c4))
            return low, upp

        A_low, A_upp = vec_times_scalar_interval(GL_ki, GU_ki, NL_kj, NU_kj)
        B_low, B_upp = vec_times_scalar_interval(GL_kj, GU_kj, NL_ki, NU_ki)

        # Sum over k (rows) -> (B, i, j, d_in)
        grad_lower = (A_low + B_low).sum(dim=1)
        grad_upper = (A_upp + B_upp).sum(dim=1)

        return grad_lower, grad_upper


class NCM_NN_triangular(nn.Module):
    """
    Maps x in R^d to a PD matrix M(x) = N(x)^T N(x) + eps * I,
    where N(x) in R^{d x d} is lower-triangular and produced by a small Softplus MLP.
    Provides interval bounds on M and on grad_x M_{ij}.
    """
    def __init__(self, d: int, hidden_sizes=(16, 16), eps: float = 1e-3, bias: bool = True):
        super().__init__()
        self.d = d
        self.eps = eps
        self.tril_size = d * (d + 1) // 2  # number of lower-triangular entries

        layers = []
        in_dim = d
        for h in hidden_sizes:
            layers += [nn.Linear(in_dim, h, bias=bias), nn.Softplus()]
            in_dim = h
        layers += [nn.Linear(in_dim, self.tril_size, bias=bias)]  # final linear, no activation
        self.mlp = nn.Sequential(*layers)

        # Cache the linear layers for Jacobian products
        self.linear_layers = [m for m in self.mlp if isinstance(m, nn.Linear)]

        # Precompute lower-triangular indices for scattering
        self.register_buffer("tril_idx", torch.tril_indices(d, d, 0))

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_uniform_(m.weight, a=5**0.5)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    # ---------- Utility: scatter flat → triangular ----------
    def flat_to_tril(self, N_flat: torch.Tensor) -> torch.Tensor:
        """
        Convert (B, tril_size) into (B, d, d) lower-triangular matrix.
        """
        B = N_flat.shape[0]
        N = torch.zeros(B, self.d, self.d, device=N_flat.device, dtype=N_flat.dtype)
        N[:, self.tril_idx[0], self.tril_idx[1]] = N_flat
        return N

    # ---------- Forward ----------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (d,) or (B, d)
        returns: (d, d) or (B, d, d)
        """
        single = (x.ndim == 1)
        if single:
            x = x.unsqueeze(0)

        B = x.shape[0]
        N_flat = self.mlp(x)                           # (B, tril_size)
        N = self.flat_to_tril(N_flat)                  # (B, d, d)
        M = torch.matmul(N.transpose(-1, -2), N)       # (B, d, d)
        if self.eps != 0.0:
            I = torch.eye(self.d, device=x.device, dtype=x.dtype).unsqueeze(0)
            M = M + self.eps * I
        return M.squeeze(0) if single else M

    # ---------- Interval arithmetic helpers ----------
    @staticmethod
    def _ibp_linear(lower, upper, W: torch.Tensor, b: torch.Tensor):
        W_pos = torch.clamp(W, min=0)
        W_neg = torch.clamp(W, max=0)
        y_lower = lower @ W_pos.T + upper @ W_neg.T + b
        y_upper = upper @ W_pos.T + lower @ W_neg.T + b
        return y_lower, y_upper

    @staticmethod
    def _ibp_softplus(lower, upper):
        return F.softplus(lower), F.softplus(upper)

    def _preact_bounds_and_N_bounds(self, l: torch.Tensor, u: torch.Tensor):
        if l.ndim == 1: l = l.unsqueeze(0)
        if u.ndim == 1: u = u.unsqueeze(0)
        B, din = l.shape
        assert din == self.d

        preacts = []
        lower, upper = l, u
        modules = list(self.mlp)

        i = 0
        while i < len(modules):
            lin = modules[i];  assert isinstance(lin, nn.Linear)
            zL, zU = self._ibp_linear(lower, upper, lin.weight, lin.bias)

            if i + 1 < len(modules) and isinstance(modules[i + 1], nn.Softplus):
                preacts.append((zL, zU))
                lower, upper = self._ibp_softplus(zL, zU)
                i += 2
            else:
                # Final linear (to tril_size). These are N_flat bounds.
                N_lower_flat, N_upper_flat = zL, zU
                break

        N_lower = self.flat_to_tril(N_lower_flat)
        N_upper = self.flat_to_tril(N_upper_flat)
        return preacts, (N_lower, N_upper)

    @staticmethod
    def _diag_bounds_from_preacts(preacts):
        diag_bounds = []
        for zL, zU in preacts:
            J_lower = torch.sigmoid(zL)
            J_upper = torch.sigmoid(zU)
            diag_bounds.append((J_lower, J_upper))
        return diag_bounds

    @staticmethod
    def _left_mul_by_diag_batched(J_lower, J_upper, P_lower, P_upper):
        JL = J_lower.unsqueeze(-1)
        JU = J_upper.unsqueeze(-1)
        ll = JL * P_lower
        lu = JL * P_upper
        ul = JU * P_lower
        uu = JU * P_upper
        lower = torch.minimum(torch.minimum(ll, lu), torch.minimum(ul, uu))
        upper = torch.maximum(torch.maximum(ll, lu), torch.maximum(ul, uu))
        return lower, upper

    @staticmethod
    def _left_mul_by_const_batched(P_lower, P_upper, W):
        W_pos = torch.clamp(W, min=0)
        W_neg = torch.clamp(W, max=0)
        lower = torch.einsum('kn,bnm->bkm', W_pos, P_lower) + torch.einsum('kn,bnm->bkm', W_neg, P_upper)
        upper = torch.einsum('kn,bnm->bkm', W_pos, P_upper) + torch.einsum('kn,bnm->bkm', W_neg, P_lower)
        return lower, upper

    def _jacobian_bounds_of_N_flat(self, l, u):
        preacts, _ = self._preact_bounds_and_N_bounds(l, u)
        diag_bounds = self._diag_bounds_from_preacts(preacts)

        B = l.shape[0] if l.ndim > 1 else 1
        W_list = [lin.weight for lin in self.linear_layers]
        P_lower = W_list[0].unsqueeze(0).expand(B, -1, -1).clone()
        P_upper = P_lower.clone()

        for i in range(len(diag_bounds)):
            JL, JU = diag_bounds[i]
            P_lower, P_upper = self._left_mul_by_diag_batched(JL, JU, P_lower, P_upper)
            P_lower, P_upper = self._left_mul_by_const_batched(P_lower, P_upper, W_list[i + 1])

        return P_lower, P_upper  # (B, tril_size, d)

    # ---------- Public: interval bounds on M ----------
    def output_bounds(self, l, u):
        _, (N_lower, N_upper) = self._preact_bounds_and_N_bounds(l, u)

        B, d, _ = N_lower.shape
        Nl_i = N_lower.permute(0, 2, 1).unsqueeze(-1)
        Nu_i = N_upper.permute(0, 2, 1).unsqueeze(-1)
        Nl_j = N_lower.unsqueeze(1)
        Nu_j = N_upper.unsqueeze(1)
        cands = torch.stack([Nl_i*Nl_j, Nl_i*Nu_j, Nu_i*Nl_j, Nu_i*Nu_j], dim=-1)
        prod_min = cands.min(dim=-1).values
        prod_max = cands.max(dim=-1).values
        M_lower = prod_min.sum(dim=2)
        M_upper = prod_max.sum(dim=2)

        if self.eps != 0.0:
            I = torch.eye(self.d, device=N_lower.device, dtype=N_lower.dtype).unsqueeze(0)
            M_lower = M_lower + self.eps * I
            M_upper = M_upper + self.eps * I
        return M_lower, M_upper

    # ---------- Public: interval bounds on grad_x M_{ij} ----------
    def grad_M_bounds(self, l, u):
        if l.ndim == 1: l = l.unsqueeze(0)
        if u.ndim == 1: u = u.unsqueeze(0)
        B, d = l.shape
        assert d == self.d

        preacts, (N_lower, N_upper) = self._preact_bounds_and_N_bounds(l, u)
        G_lower, G_upper = self._jacobian_bounds_of_N_flat(l, u)  # (B, tril_size, d_in)

        # Scatter Jacobians into (B, k, i, d_in)
        G_lower_full = torch.zeros(B, d, d, d, device=l.device)
        G_upper_full = torch.zeros(B, d, d, d, device=l.device)
        G_lower_full[:, self.tril_idx[0], self.tril_idx[1], :] = G_lower
        G_upper_full[:, self.tril_idx[0], self.tril_idx[1], :] = G_upper

        I_idx, J_idx = torch.meshgrid(
            torch.arange(self.d, device=l.device),
            torch.arange(self.d, device=l.device),
            indexing='ij'
        )

        GL_ki = G_lower_full[:, :, I_idx, :]
        GU_ki = G_upper_full[:, :, I_idx, :]
        NL_kj = N_lower[:, :, J_idx]
        NU_kj = N_upper[:, :, J_idx]

        GL_kj = G_lower_full[:, :, J_idx, :]
        GU_kj = G_upper_full[:, :, J_idx, :]
        NL_ki = N_lower[:, :, I_idx]
        NU_ki = N_upper[:, :, I_idx]

        def vec_times_scalar_interval(vL, vU, sL, sU):
            sL = sL.unsqueeze(-1); sU = sU.unsqueeze(-1)
            c1 = vL * sL
            c2 = vL * sU
            c3 = vU * sL
            c4 = vU * sU
            low = torch.minimum(torch.minimum(c1, c2), torch.minimum(c3, c4))
            upp = torch.maximum(torch.maximum(c1, c2), torch.maximum(c3, c4))
            return low, upp

        A_low, A_upp = vec_times_scalar_interval(GL_ki, GU_ki, NL_kj, NU_kj)
        B_low, B_upp = vec_times_scalar_interval(GL_kj, GU_kj, NL_ki, NU_ki)

        grad_lower = (A_low + B_low).sum(dim=1)
        grad_upper = (A_upp + B_upp).sum(dim=1)

        return grad_lower, grad_upper
