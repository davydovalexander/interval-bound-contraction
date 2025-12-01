import torch
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

from ncm import NCM_NN
from dynamics import pendulum_jac_bounds, pendulum_control_matrix, pendulum_f_bounds
from nn_controller import NN_IBP

from utils import compute_metzler_nonconstant_NCM, partition_hyperrectangle, multiply_two_interval_matrices, bound_Mdot


if __name__ == '__main__':
        # --- Hyperparameters ---
    eps = 1e-6
    learning_rate = 1e-2
    num_epochs = 20000
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    xunder_full = torch.tensor([-torch.pi/100, -0.05])
    eye = torch.eye(3).to(device=device)
    xover_full = -xunder_full
    epoch_last_updated = 0
    domain_part = 16

    xunder, xover = partition_hyperrectangle(xunder_full, xover_full, domain_part**2)

    # --- Send model to device ---
    model = NN_IBP(input_dim = 2, hidden_dims=[16,16])
    ncm_model = NCM_NN(d = 2, hidden_sizes=[32,32], eps = 0.1, constant_NCM=False)
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
        ncm_model.train()
        # Check this function for bugs
        M_lower, M_upper = ncm_model.output_bounds(xunder, xover)
        Du_lower, Du_upper = model.compute_Du_bounds(xunder, xover, elision_matrix=None)
        Du_lower, Du_upper = multiply_two_interval_matrices(B.unsqueeze(dim=0), B.unsqueeze(dim=0), Du_lower, Du_upper)
        M_times_Du_bounds = multiply_two_interval_matrices(M_lower,
                                                           M_upper,
                                                            Du_lower,
                                                            Du_upper)

        #print(Du_bounds[0].shape)
        Df_lower, Df_upper = pendulum_jac_bounds(xunder, xover)
        M_times_Df_bounds = multiply_two_interval_matrices(M_lower,
                                                           M_upper,
                                                        Df_lower, 
                                                        Df_upper)
        #print(Df_bounds[0].shape)
        # eigenbounds = max_eig_over_hyperrectangles(Df, xunder, xover, NCM)
        grad_M_lower, grad_M_upper = ncm_model.grad_M_bounds(xunder, xover)
        f_lower, f_upper = pendulum_f_bounds(xunder, xover)
        Mdot_f_bounds = bound_Mdot(grad_M_lower, grad_M_upper, f_lower, f_upper)
        # Bu_lower, Bu_upper = model.IBP_forward(xunder, xover, elision_matrix=B)
        u_lower, u_upper = model.IBP_forward(xunder, xover, elision_matrix=None)
        Bu_lower, Bu_upper = multiply_two_interval_matrices(B.unsqueeze(dim=0), B.unsqueeze(dim=0), u_lower, u_upper)
        u0 = model(torch.zeros(2, device=device))
        # Bu_lower = Bu_lower - (B.unsqueeze(0) @ u0.unsqueeze(-1))   # check shapes carefully
        # Bu_upper = Bu_upper - (B.unsqueeze(0) @ u0.unsqueeze(-1))
        Mdot_Bu_bounds = bound_Mdot(grad_M_lower, grad_M_upper, Bu_lower, Bu_upper)

        B_Mzr = compute_metzler_nonconstant_NCM(M_times_Df_bounds, 
                                                M_times_Du_bounds,
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
        # max_eig = eigs.amax(dim=tuple(range(1, eigs.ndim)))
        # print(max_eig)
        # loss = torch.sum(torch.relu(max_eig))
        loss = torch.sum(torch.relu(eigs))
        if torch.isnan(loss):
            print('NaN detected in loss')
            break

        if loss <= 1e-7:
            # torch.save(model.state_dict(), 'inverted_pendulum_bigger.pth')
            # torch.save(ncm_model.state_dict(), 'inv_pend_ncm_bigger.pth')
            # print('At epoch ', epoch+1, ' loss has hit 0, valid closed-loop contracting controller')
            print('At epoch ', epoch+1, ' loss has hit 0, increasing domain')
            shift = torch.tensor([torch.pi/100, 0.06], device=xunder.device, dtype=xunder.dtype)

            xunder_full = xunder_full - shift
            xover_full = -xunder_full

            xunder, xover = partition_hyperrectangle(xunder_full, xover_full, 16**2)

            print('new xover', xover_full)
            epoch_last_updated = epoch
            continue

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
            
        if epoch - epoch_last_updated >= 2000:
            print("further partitioning domain")
            domain_part += 2
            xunder, xover = partition_hyperrectangle(xunder_full, xover_full, domain_part**2)
            epoch_last_updated = epoch

    torch.save(model.state_dict(), 'small.pth')
    torch.save(ncm_model.state_dict(), 'small_ncm.pth')
    print('saved models')