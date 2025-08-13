import torch
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

from ncm import NCM_NN
from dynamics import pendulum_jac_bounds, pendulum_control_matrix, pendulum_f_bounds
from nn_controller import NN_IBP

from utils import compute_metzler_nonconstant_NCM, partition_hyperrectangle, multiply_two_interval_matrices, bound_Mdot

# Only for polynomial dynamics for now

if __name__ == '__main__':
        # --- Hyperparameters ---
    eps = 1e-8
    learning_rate = 1e-2
    num_epochs = 100000
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    xunder = torch.tensor([-torch.pi/3, -0.5])
    eye = torch.eye(3).to(device=device)
    xover = -xunder

    xunder, xover = partition_hyperrectangle(xunder, xover, 10**2)

    # --- Send model to device ---
    model = NN_IBP(input_dim = 2, hidden_dims=[64,64], trainable_NCM=False)
    ncm_model = NCM_NN(d = 2, eps = 0.1)
    model = model.to(device)
    ncm_model = ncm_model.to(device)

    B = pendulum_control_matrix()
    # --- Loss and optimizer ---
    optimizer = optim.Adam(list(model.parameters()) 
                           + list(ncm_model.parameters()), 
                           lr=learning_rate)

    # --- Training loop ---
    for epoch in range(num_epochs):
        # ---- Training ----
        model.train()
        # Check this function for bugs
        M_bounds = ncm_model.output_bounds(xunder, xover)
        Du_bounds = model.compute_Du_bounds(xunder, xover, elision_matrix=B)
        M_times_Du_bounds = multiply_two_interval_matrices(*M_bounds, *Du_bounds)

        #print(Du_bounds[0].shape)
        Df_lower, Df_upper = pendulum_jac_bounds(xunder, xover)
        M_times_Df_bounds = multiply_two_interval_matrices(*M_bounds, Df_lower, Df_upper)
        #print(Df_bounds[0].shape)
        # eigenbounds = max_eig_over_hyperrectangles(Df, xunder, xover, NCM)
        grad_M_bounds = ncm_model.grad_M_bounds(xunder, xover)
        f_bounds = pendulum_f_bounds(xunder, xover)
        Mdot_f_bounds = bound_Mdot(*grad_M_bounds, *f_bounds)
        Bu_lower, Bu_upper = model.IBP_forward(xunder, xover, ellision_matrix=B)
        Mdot_Bu_bounds = bound_Mdot(*grad_M_bounds, Bu_lower, Bu_upper)

        B_Mzr = compute_metzler_nonconstant_NCM((Df_lower, Df_upper), 
                                                Du_bounds,
                                                Mdot_f_bounds,
                                                Mdot_Bu_bounds)
        # B_Mzr = compute_metzler_upper_bound_new(eigenbounds, Du_bounds)
        # improve conditioning
        B_Mzr = B_Mzr + eps * torch.eye(B_Mzr.shape[-1], device=B_Mzr.device).unsqueeze(0)
        try:
            eigs = torch.linalg.eigvalsh(B_Mzr)
        except:
            print('poor conditioning')
            print(B_Mzr[0])
            
            break
        max_eig = eigs.amax(dim=tuple(range(1, eigs.ndim)))
        # max_eig = max_eig_metzler_shifted(B_Mzr)
        # max_eig = B_Mzr.sum(dim=-1)
        # print(max_eig)
        loss = torch.sum(torch.relu(max_eig))
        if torch.isnan(loss):
            print('NaN detected in loss')
            break

        if loss <= 1e-7:
            print('At epoch ', epoch+1, ' loss has hit 0, valid closed-loop contracting controller')
            # torch.save(model.state_dict(), 'inverted_pendulum.pth')
            break

        optimizer.zero_grad()
        loss.backward()
        # nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        for name, param in model.named_parameters():
            if param.grad is not None and torch.isnan(param.grad).any():
                print(f"NaN detected in gradients of {name}")

        if epoch % 1000 == 999:
            print(f"Epoch [{epoch+1}/{num_epochs}] "
                f"Loss: {loss:.4f} ")
            
