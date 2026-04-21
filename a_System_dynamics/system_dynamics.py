import torch


# ---------------------------------------------------------------------------
# Quaternion helpers (singularity-free attitude representation)
# ---------------------------------------------------------------------------

def euler_to_quat(roll, pitch, yaw):
    """
    Convert ZYX Euler angles to unit quaternion [w, x, y, z].

    roll, pitch, yaw: (B,) tensors
    returns: (B,4)
    """
    cr, sr = torch.cos(roll * 0.5), torch.sin(roll * 0.5)
    cp, sp = torch.cos(pitch * 0.5), torch.sin(pitch * 0.5)
    cy, sy = torch.cos(yaw * 0.5), torch.sin(yaw * 0.5)

    qw = cr * cp * cy + sr * sp * sy
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy

    return torch.stack([qw, qx, qy, qz], dim=1)  # (B,4)


def quat_mul(q1, q2):
    """
    Hamilton product q1 ⊗ q2.

    q1, q2: (B,4) as [w, x, y, z]
    returns: (B,4)
    """
    w1, x1, y1, z1 = q1[:, 0], q1[:, 1], q1[:, 2], q1[:, 3]
    w2, x2, y2, z2 = q2[:, 0], q2[:, 1], q2[:, 2], q2[:, 3]
    return torch.stack([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ], dim=1)  # (B,4)


def quat_normalize(q):
    """Normalize quaternion to unit length. q: (B,4) -> (B,4)"""
    return q / (q.norm(dim=1, keepdim=True) + 1e-8)


def omega_to_quat_delta(omega, dt):
    """
    Approximate quaternion increment from body-frame angular velocity.

    Uses the exact axis-angle formula so there is no small-angle approximation.
    omega: (B,3) [wx, wy, wz] in body frame
    dt:    (B,1) or scalar
    returns: (B,4)
    """
    half_angle_vec = 0.5 * omega * dt          # (B,3)
    angle = half_angle_vec.norm(dim=1, keepdim=True)  # (B,1)
    # sinc(angle) = sin(angle)/angle  (well-defined for angle -> 0)
    sinc = torch.where(angle > 1e-8,
                       torch.sin(angle) / angle,
                       torch.ones_like(angle))
    dqw   = torch.cos(angle)          # (B,1)
    dqxyz = sinc * half_angle_vec     # (B,3)
    return torch.cat([dqw, dqxyz], dim=1)  # (B,4)


def quat_to_euler(q):
    """
    Convert unit quaternion [w, x, y, z] to ZYX Euler angles [roll, pitch, yaw].

    q: (B,4)
    returns: (B,3)
    """
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]

    # roll (rotation about x-axis)
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = torch.atan2(sinr_cosp, cosr_cosp)

    # pitch (rotation about y-axis) – clamped to avoid NaN from asin
    sinp = 2.0 * (w * y - z * x)
    sinp = torch.clamp(sinp, -1.0, 1.0)
    pitch = torch.asin(sinp)

    # yaw (rotation about z-axis)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = torch.atan2(siny_cosp, cosy_cosp)

    return torch.stack([roll, pitch, yaw], dim=1)  # (B,3)


def wrap_to_pi(a):
    """Wrap angles to [-π, π]."""
    return torch.atan2(torch.sin(a), torch.cos(a))


# ---------------------------------------------------------------------------

def system_dynamics(state_vector, control_input, mass, inertia, g):
    """
    Physics of the drone
    State_vector = [x, y, z, roll, pitch, yaw, vx, vy, vz, w_roll, w_pitch, w_yaw]
    Control_input = [thrust, torque_roll, torque_pitch, torque_yaw]
    """
    # # Physical parameters definition REMEMBER TO ADJUST IF NECESSARY
    # g = 9.81  # gravity [m/s^2]
    # mass = 2.0  # mass of the drone [kg]
    # inertia = torch.tensor([0.0216, 0.0216, 0.04])  # inertia around roll (Ix), pitch (Iy), yaw (Iz) [kg*m^2]
    if not isinstance(inertia, torch.Tensor):
        inertia = torch.tensor(inertia)

    # Unpack state vector
    x, y, z, roll, pitch, yaw, vx, vy, vz, w_roll, w_pitch, w_yaw = state_vector[:,0], state_vector[:,1], state_vector[:,2], state_vector[:,3], state_vector[:,4], state_vector[:,5], state_vector[:,6], state_vector[:,7], state_vector[:,8], state_vector[:,9], state_vector[:,10], state_vector[:,11]
    thrust, torque_roll, torque_pitch, torque_yaw = control_input[:,0], control_input[:,1], control_input[:,2], control_input[:,3]
    
    # # Move sin/cos components to torch tensors (if they aren't already) to be consistent with the rest of the states
    # if not isinstance(roll, torch.Tensor):
    #     roll = torch.tensor(roll)
    # if not isinstance(pitch, torch.Tensor):
    #     pitch = torch.tensor(pitch)
    # if not isinstance(yaw, torch.Tensor):
    #     yaw = torch.tensor(yaw)

    # NOTICE THAT: the dataset is recorded in the North-East-Down (NED) frame, so z is positive downwards. The system dynamics then must be consistent with this frame ( ==> vz_dot gravity is positive and thrust is negative)
    # Compute dynamics
    x_dot = vx # world frame
    y_dot = vy # world frame
    z_dot = vz # world frame
    roll_dot = w_roll + (torch.sin(roll) * torch.tan(pitch) * w_pitch) + (torch.cos(roll) * torch.tan(pitch) * w_yaw) # world frame
    pitch_dot = (torch.cos(roll) * w_pitch) - (torch.sin(roll) * w_yaw) # world frame
    yaw_dot = (torch.sin(roll) / torch.cos(pitch) * w_pitch) + (torch.cos(roll) / torch.cos(pitch) * w_yaw) # world frame
    vx_dot = (-thrust / mass) * (torch.cos(roll) * torch.sin(pitch) * torch.cos(yaw) + torch.sin(roll) * torch.sin(yaw)) # world frame
    vy_dot = (-thrust / mass) * (torch.cos(roll) * torch.sin(pitch) * torch.sin(yaw) - torch.sin(roll) * torch.cos(yaw)) # world frame
    vz_dot = g - (thrust / mass) * (torch.cos(roll) * torch.cos(pitch)) # world frame
    w_roll_dot = (inertia[1]-inertia[2]) / inertia[0] * w_pitch * w_yaw + torque_roll / inertia[0] # body frame
    w_pitch_dot = (inertia[2]-inertia[0]) / inertia[1] * w_roll * w_yaw + torque_pitch / inertia[1] # body frame
    w_yaw_dot = (inertia[0]-inertia[1]) / inertia[2] * w_roll * w_pitch + torque_yaw / inertia[2] # body frame

    state_vector_dot = torch.stack([x_dot, y_dot, z_dot,
                                  roll_dot, pitch_dot, yaw_dot,
                                  vx_dot, vy_dot, vz_dot,
                                  w_roll_dot, w_pitch_dot, w_yaw_dot], dim=1) 

    # Return derivatives of the state vector
    return state_vector_dot