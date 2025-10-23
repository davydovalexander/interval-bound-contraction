import torch
import torch.optim as optim

from ncm import NCM_NN
from dynamics import polynomial_jac_bounds, polynomial_control_matrix
from nn_controller import NN_IBP

from utils import compute_metzler_upper_bound, partition_hyperrectangle, multiply_two_interval_matrices, bound_Mdot
# Only for polynomial dynamics for now

if __name__ == '__main__':
        # --- Hyperparameters ---
    eps = 1e-8
    learning_rate = 1e-2
    num_epochs = 20000
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    xunder = -100*torch.ones(3)
    eye = torch.eye(3).to(device=device)
    xover = 100*torch.ones(3)

    xunder, xover = partition_hyperrectangle(xunder, xover, 5**3)

    # --- Send model to device ---
    model = NN_IBP(input_dim = 3, hidden_dims=[32,32])
    ncm_model = NCM_NN(d = 3, hidden_sizes=[1,1], eps = 0.1, constant_NCM=True)
    model = model.to(device)
    ncm_model = ncm_model.to(device)

    B = polynomial_control_matrix()
    # --- Loss and optimizer ---
    optimizer = optim.Adam(list(model.parameters()) 
                           + list(ncm_model.parameters()), 
                           lr=learning_rate)

    # --- Training loop ---
    for epoch in range(num_epochs):
        # ---- Training ----
        model.train()
        # Check this function for bugs
        M = ncm_model(torch.tensor([0]))
        Du_lower, Du_upper = model.compute_Du_bounds(xunder, xover, elision_matrix=M @ B)
        # M_times_Du_bounds = multiply_two_interval_matrices(M_lower,
        #                                                    M_upper,
        #                                                     Du_lower,
        #                                                     Du_upper)

        #print(Du_bounds[0].shape)
        Df_lower, Df_upper = polynomial_jac_bounds(xunder, xover)
        M_times_Df_bounds = multiply_two_interval_matrices(M,
                                                           M,
                                                        Df_lower, 
                                                        Df_upper)
        #print(Df_bounds[0].shape)
        # eigenbounds = max_eig_over_hyperrectangles(Df, xunder, xover, NCM)
        
        B_Mzr = compute_metzler_upper_bound(M_times_Df_bounds, 
                                            (Du_lower, Du_upper))
        # B_Mzr = compute_metzler_upper_bound_new(eigenbounds, Du_bounds)
        # improve conditioning
        B_Mzr = B_Mzr + eps * torch.eye(B_Mzr.shape[-1], device=B_Mzr.device).unsqueeze(0)
        try:
            eigs = torch.linalg.eigvalsh(B_Mzr)
        except:
            print('poor conditioning')
            print(B_Mzr[0])
            
            break
        loss = torch.sum(torch.relu(eigs))
        if torch.isnan(loss):
            print('NaN detected in loss')
            break

        if loss <= 1e-7:
            print('At epoch ', epoch+1, ' loss has hit 0, valid closed-loop contracting controller')
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
            

    print('contraction metric', ncm_model(torch.tensor([0])))