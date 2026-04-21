import torch
import torch.nn as nn
from a_System_dynamics.system_dynamics import (
    system_dynamics,
    euler_to_quat, quat_mul, quat_normalize, omega_to_quat_delta, quat_to_euler, wrap_to_pi,
)

_mse_loss = nn.MSELoss()


def compute_new_pos_angles(delta_vw, X_curr, U_curr, dt, mass, inertia, g):
    """
    Integrate one Euler step with NN derivative corrections.

    delta_vw: (B,6) NN output = derivative corrections [v̇_corr(3), ω̇_corr(3)]
              v̇_corr in m/s², ω̇_corr in rad/s².  dt-agnostic (no /dt needed).
    X_curr:   (B,12) [x,y,z, roll,pitch,yaw, vx,vy,vz, w_roll,w_pitch,w_yaw]
    U_curr:   (B,4)
    dt:       (B,) or (B,1)
    returns:  X_next (B,12)
    """
    if dt.ndim == 1:
        dt = dt.view(-1, 1)

    # baseline physics derivative
    x_dot_phys = system_dynamics(X_curr, U_curr, mass, inertia=inertia, g=g)  # (B,12)

    delta_v = delta_vw[:, 0:3]  # (B,3) derivative correction for linear velocity  [m/s²]
    delta_w = delta_vw[:, 3:6]  # (B,3) derivative correction for angular velocity [rad/s²]

    # corrected next velocities: physics derivative + NN derivative correction, then Euler step
    v_next = X_curr[:, 6:9]  + (x_dot_phys[:, 6:9]  + delta_v) * dt  # (B,3)
    w_next = X_curr[:, 9:12] + (x_dot_phys[:, 9:12] + delta_w) * dt  # (B,3)

    # integrate position in world frame using trapezoid rule
    v_curr = X_curr[:, 6:9]
    pos_next = X_curr[:, 0:3] + 0.5 * (v_curr + v_next) * dt  # (B,3)

    # integrate attitude using quaternion (singularity-free, no tan(pitch) / cos(pitch))
    roll0  = X_curr[:, 3]
    pitch0 = X_curr[:, 4]
    yaw0   = X_curr[:, 5]

    q0    = euler_to_quat(roll0, pitch0, yaw0)   # (B,4)
    dq    = omega_to_quat_delta(w_next, dt)       # (B,4)

    # w_next is expressed in the body frame; q_next = q0 ⊗ dq
    q_next   = quat_mul(q0, dq)
    q_next   = quat_normalize(q_next)
    rpy_next = quat_to_euler(q_next)              # (B,3)
    rpy_next = wrap_to_pi(rpy_next)

    # physics-only baseline for remaining state channels (acc terms kept from physics)
    x_next_base = X_curr + x_dot_phys * dt

    # assemble final next state
    X_next = x_next_base.clone()
    X_next[:, 0:3]  = pos_next
    X_next[:, 3:6]  = rpy_next
    X_next[:, 6:9]  = v_next
    X_next[:, 9:12] = w_next

    return X_next


def data_loss(model, X_curr, U_curr_NN, X_next, dt,
              mass=2.0, inertia=torch.tensor([0.0217, 0.0217, 0.04]), g=9.81,
              channel_weights=None, lambda_corr=0.0):
    """
    One-step data loss with NN derivative corrections.

    model outputs [v̇_corr(3), ω̇_corr(3)] – derivative corrections in physical units.
    X_curr, X_next are in physical units.
    dt: (B,) or (B,1)

    channel_weights: (12,) tensor weighting each state channel.
    lambda_corr:     L2 penalty on the NN correction magnitude (keeps corrections small).
    """
    if dt.ndim == 1:
        dt = dt.view(-1, 1)

    delta_vw = model(X_curr, U_curr_NN)  # (B,6)

    X_next_pred = compute_new_pos_angles(delta_vw, X_curr, U_curr_NN, dt, mass, inertia, g)  # (B,12)

    # channel-wise MSE over all 12 state channels
    per_channel_losses = torch.mean((X_next_pred - X_next) ** 2, dim=0)  # (12,)

    if channel_weights is None:
        # Positions/angles get a small weight so the loss focuses on what the NN
        # directly corrects (velocities 6-11), while still penalising large drift.
        cw = torch.cat([
            torch.full((6,), 0.1, device=X_curr.device, dtype=X_curr.dtype),  # pos + angles
            torch.ones(6,         device=X_curr.device, dtype=X_curr.dtype),   # vel + rates
        ])
    else:
        cw = channel_weights.to(device=X_curr.device, dtype=X_curr.dtype)

    loss_data = torch.sum(cw * per_channel_losses)

    # L2 penalty: discourage the NN from producing large corrections that override physics
    loss_corr = torch.mean(delta_vw ** 2)

    total_loss = loss_data + lambda_corr * loss_corr
    return total_loss, {
        "loss_data": loss_data.detach(),
        "loss_corr": loss_corr.detach(),
        "per_channel": per_channel_losses.detach(),
    }
