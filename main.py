import torch
import torch.optim as optim

from dynamics import jac_bounds, control_matrix
from nn_controller import NN_IBP

from utils import compute_metzler_upper_bound, partition_hyperrectangle

# Only for polynomial dynamics for now

if __name__ == '__main__':
        # --- Hyperparameters ---
    eps = 1e-8
    learning_rate = 1e-2
    num_epochs = 20000
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    xunder = -10*torch.ones(3)
    eye = torch.eye(3).to(device=device)
    xover = 10*torch.ones(3)

    xunder, xover = partition_hyperrectangle(xunder, xover, 10**3)

    # --- Send model to device ---
    model = NN_IBP(hidden_dims=[32,32], trainable_NCM=True)
    model = model.to(device)

    B = control_matrix()
    # --- Loss and optimizer ---
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # --- Training loop ---
    for epoch in range(num_epochs):
        # ---- Training ----
        model.train()
        # Check this function for bugs
        NCM = model.constant_NCM()
        Du_bounds = model.compute_Du_bounds(xunder, xover, elision_matrix=NCM @ B)

        #print(Du_bounds[0].shape)
        Df_lower, Df_upper = jac_bounds(xunder, xover)
        Df_lower, Df_upper = model.left_multiply_by_constant_matrix(Df_lower, Df_upper, NCM)
        #print(Df_bounds[0].shape)
        # eigenbounds = max_eig_over_hyperrectangles(Df, xunder, xover, NCM)
        B_Mzr = compute_metzler_upper_bound((Df_lower, Df_upper), Du_bounds)
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
            

    print('contraction metric', model.constant_NCM())