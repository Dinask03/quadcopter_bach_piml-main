import torch
import torch.nn as nn
from a_System_dynamics.system_dynamics import system_dynamics

_mse_loss = nn.MSELoss()

def data_loss(model, X_curr_NN, U_curr_NN, X_curr, X_next, dt,
              mass=2.0, inertia=torch.tensor([0.0217, 0.0217, 0.04]), g=9.81,
              channel_weights=None, lambda_corr=0.0):
    """
    model outputs only Δx_dot (correction).
    X_curr, X_next are in physical units.
    dt: (B,) or (B,1)
    """

    if dt.ndim == 1:
        dt = dt.view(-1, 1)

    # baseline physics derivative
    x_dot_phys = system_dynamics(X_curr, U_curr_NN, mass, inertia=inertia, g=g)  # (B,12)

    # NN correction in derivative space
    delta_x_dot = model(X_curr_NN, U_curr_NN)  # (B,12)

    # corrected derivative
    x_dot_total = x_dot_phys + delta_x_dot

    # integrate one Euler step
    X_next_pred = X_curr + dt * x_dot_total

    # Wrap angles before comparing (optional but recommended)
    # Compare roll/pitch/yaw with angle-wrapping safe representation:
    X_next_pred_wrapped = X_next_pred.clone()
    X_next_true_wrapped = X_next.clone()

    for idx in [3, 4, 5]:  # roll, pitch, yaw indices in your 12-state
        X_next_pred_wrapped[:, idx] = torch.atan2(torch.sin(X_next_pred[:, idx]), torch.cos(X_next_pred[:, idx]))
        X_next_true_wrapped[:, idx] = torch.atan2(torch.sin(X_next[:, idx]), torch.cos(X_next[:, idx]))

    # channel-wise MSE (to keep your weighting approach)
    per_channel_losses = torch.mean((X_next_pred_wrapped - X_next_true_wrapped) ** 2, dim=0)  # (12,)

    if channel_weights is None:
        cw = torch.ones(12, device=X_curr.device, dtype=X_curr.dtype)
    else:
        cw = channel_weights.to(device=X_curr.device, dtype=X_curr.dtype)

    loss_data = torch.sum(cw * per_channel_losses)

    # Optional: discourage huge corrections (PINN-ish regularization)
    loss_corr = torch.mean(delta_x_dot ** 2)

    total_loss = loss_data + lambda_corr * loss_corr
    return total_loss, {"loss_data": loss_data.detach(), "loss_corr": loss_corr.detach()}