import torch
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def dataset_masking(dataset=None):

    # Add dt column at position 1 (limiting the number to 6 decimal points)
    dataset.insert(1, 'dt', dataset['time'].diff().fillna(0).round(6))  # Calculate time intervals (dt) between consecutive samples and place it in a new column 'dt' in position 1

    # Add Euler angles columns (roll, pitch, yaw) at positions 17, 18, 19
    dataset.insert(17, 'roll', 0.0)
    dataset.insert(18, 'pitch', 0.0)
    dataset.insert(19, 'yaw', 0.0)

    # Add the following non-used columns to match the old dataset format
    dataset.insert(24, 'pwm_1', 0.0)
    dataset.insert(25, 'pwm_2', 0.0)
    dataset.insert(26, 'pwm_3', 0.0)
    dataset.insert(27, 'pwm_4', 0.0)
    dataset.insert(28, 'total_thrust', 0.0)

    # Drop the unused columns (from 37 to the end)
    dataset = dataset.drop(columns=dataset.columns[37:])

    return dataset

def from_quaternion_to_euler(dataset=None):
    """
    Convert quaternion orientation to Euler angles (roll, pitch, yaw) in the dataset.
    Quaternion format in dataset: [q_w, q_x, q_y, q_z]
    Euler angles format: [roll, pitch, yaw]
    """
    q_w = dataset[:, 20:21]
    q_x = dataset[:, 21:22]
    q_y = dataset[:, 22:23]
    q_z = dataset[:, 23:24]

    # Compute roll (x-axis rotation)
    sinr_cosp = 2 * (q_w * q_x + q_y * q_z)
    cosr_cosp = 1 - 2 * (q_x**2 + q_y**2)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    # Compute pitch (y-axis rotation)
    sinp = 2 * (q_w * q_y - q_z * q_x)
    pitch = np.where(np.abs(sinp) >= 1, np.sign(sinp) * (np.pi / 2), np.arcsin(sinp))

    # Compute yaw (z-axis rotation)
    siny_cosp = 2 * (q_w * q_z + q_x * q_y)
    cosy_cosp = 1 - 2 * (q_y**2 + q_z**2)
    yaw = np.arctan2(siny_cosp, cosy_cosp)

    # Replace quaternion columns with Euler angles in the dataset (rounding to 6 decimal points)
    dataset[:, 17:18] = np.round(roll, 6)
    dataset[:, 18:19] = np.round(pitch, 6)
    dataset[:, 19:20] = np.round(yaw, 6)

    return dataset

def px4_pwm_to_thrust(dataset = None, mass=2.0, g=9.81):
    """
    1) I need to clamp the values of cmd_thrust between -1 and 0, because out of this interval the drone will read only -1 and 0, so it doesn't make sense to have values outside this range
    2) The cmd_thrust in the dataset is a PWM (or another signal), so i need to apply a conversion factor to get the actual thrust force. This factor is determined empirically to match the hover condition
    """
    cmd_thrust = dataset[:, 29:30] 
    cmd_thrust = np.clip(cmd_thrust, -1.0, 0.0)  # Clamp between -1 and 0

    thrust = cmd_thrust * mass * g / -0.72 # conversion factor to get thrust in Newtons
    dataset[:, 29:30] = np.round(thrust, 6)

    return dataset

def px4_angular_rate_to_torque(dataset = None, inertia = np.array([0.0216, 0.0216, 0.04])):
    """
    Convert PX4 angular velocities ref to torques ref.
    Let's design 3 PD controllers, one for each axis, to convert angular rate references to torque references.
    1) tau_roll = Kp_roll * (theta_roll_ref - theta_roll) + Kd_roll * (w_roll_ref - w_roll)
    2) tau_pitch = Kp_pitch * (theta_pitch_ref - theta_pitch) + Kd_pitch * (w_pitch_ref - w_pitch)
    3) tau_yaw = Kp_yaw * (theta_yaw_ref - theta_yaw) + Kd_yaw * (w_yaw_ref - w_yaw)

    Notice that: a) the PX4-gazebo-model for the x500 quadcopter doesn't provide the damping, it provides the time ccontsants (equals for each motor): timeConstantUp = 0.0125 s, timeConstantDown = 0.025 s 
                    (they probably are the time constants when the motor speed up (faster) or slow down (slower))
                 b) Since there is no drect damping ratio, we could assume the motor dynamics as first order systems
                 c) make controller bandwidth significantly lower than motor dynamics bandwidth to avoid instability
    """

    # 1) define time constants and compute motor bandwidth
    time_constant_up = 0.0125  # seconds (it's the time necessary for the motor to reach 63% of the final value when speeding up)
    time_constant_down = 0.025  # seconds (it's the time necessary for the motor to reach 63% of the final value when slowing down)

    tau_motor = time_constant_down  # use the slower time constant to stay conservative
    wn_motor = 1 / tau_motor  # motor bandwidth (rad/s)

    # 2) define controller bandwidth
    safety_factor = 0.5  # to ensure stability, make controller bandwidth lower than motor bandwidth
    wn_controller = safety_factor * wn_motor  # controller bandwidth (rad/s), significantly lower than motor bandwidth

    # 3) define PD gains for each axis
    zeta = 0.7  # damping ratio

    Kp_roll = 2 * zeta * wn_controller * inertia[0]
    Kd_roll = inertia[0] * wn_controller**2

    Kp_pitch = 2 * zeta * wn_controller * inertia[1]
    Kd_pitch = inertia[1] * wn_controller**2

    Kp_yaw = 2 * zeta * wn_controller * inertia[2]
    Kd_yaw = inertia[2] * wn_controller**2

    # 4) find torque references
    roll_ref = 0 # set to 0 to stabilize the drone
    pitch_ref = 0 # set to 0 to stabilize the drone
    yaw_ref = dataset[:, 36:37]  # desired yaw angle from dataset
    w_roll_ref = dataset[:, 30:31]
    w_pitch_ref = dataset[:, 31:32]
    w_yaw_ref = dataset[:, 32:33]

    roll, pitch, yaw = dataset[:, 17:18], dataset[:, 18:19], dataset[:, 19:20]
    w_roll, w_pitch, w_yaw = dataset[:, 11:12], dataset[:, 12:13], dataset[:, 13:14]

    torque_roll = Kp_roll * (roll_ref - roll) + Kd_roll * (w_roll_ref - w_roll)
    torque_pitch = Kp_pitch * (pitch_ref - pitch) + Kd_pitch * (w_pitch_ref - w_pitch)
    torque_yaw = Kp_yaw * (yaw_ref - yaw) + Kd_yaw * (w_yaw_ref - w_yaw)

    dataset[:, 30:31] = np.round(torque_roll, 6)
    dataset[:, 31:32] = np.round(torque_pitch, 6)
    dataset[:, 32:33] = np.round(torque_yaw, 6)

    return dataset

def get_mean_and_std(dataset=None):
    """
    Get mean and standard deviation of the whole dataset for normalization.
    """
    time = dataset[:, 0:1]
    dt = dataset[:, 1:2]
    linear_pos = dataset[:, 2:5] # x, y, z 
    linear_vel = dataset[:, 5:8] # vx, vy, vz
    linear_acc = dataset[:, 8:11] # a_x, a_y, a_z
    angular_vel = dataset[:, 11:14] # w_x, w_y, w_z
    angular_acc = dataset[:, 14:17] # alpha_x, alpha_y, alpha_z
    angular_pos = dataset[:, 17:20] # roll, pitch, yaw
    rest_of_the_data = dataset[:, 20:29] # other data not used for training
    controls = dataset[:, 29:33] # thrust, torque_roll, torque_pitch, torque_yaw

    lin_pos_mean = linear_pos.mean(axis=0)
    lin_pos_std = linear_pos.std(axis=0) + 1e-8
    
    lin_vel_mean = linear_vel.mean(axis=0)
    lin_vel_std = linear_vel.std(axis=0) + 1e-8
    
    lin_acc_mean = linear_acc.mean(axis=0)
    lin_acc_std = linear_acc.std(axis=0) + 1e-8
    
    ang_vel_mean = angular_vel.mean(axis=0)
    ang_vel_std = angular_vel.std(axis=0) + 1e-8
    
    ang_acc_mean = angular_acc.mean(axis=0)
    ang_acc_std = angular_acc.std(axis=0) + 1e-8
    
    # rest_mean = rest_of_the_data.mean(axis=0)
    # rest_std = rest_of_the_data.std(axis=0) + 1e-8
    
    controls_mean = controls.mean(axis=0)
    controls_std = controls.std(axis=0) + 1e-8

    mean = np.hstack((lin_pos_mean, lin_vel_mean, ang_vel_mean, lin_acc_mean, ang_acc_mean, controls_mean))
    std = np.hstack((lin_pos_std, lin_vel_std, ang_vel_std, lin_acc_std, ang_acc_std, controls_std))

    return mean, std

def normalize_data(dataset=None, mean=None, std=None):
    """
    Normalize all the dataset meaningful quantities (so not time or angles for example)
    """
    time = dataset[:, 0:1]
    dt = dataset[:, 1:2]
    linear_pos = dataset[:, 2:5] # x, y, z 
    linear_vel = dataset[:, 5:8] # vx, vy, vz
    linear_acc = dataset[:, 8:11] # a_x, a_y, a_z
    angular_vel = dataset[:, 11:14] # w_x, w_y, w_z
    angular_acc = dataset[:, 14:17] # alpha_x, alpha_y, alpha_z
    angular_pos = dataset[:, 17:20] # roll, pitch, yaw
    rest_of_the_data = dataset[:, 20:29] # other data not used for training
    controls = dataset[:, 29:33] # thrust, torque_roll, torque_pitch, torque_yaw

    # Normalization
    lin_pos_mean, lin_vel_mean, ang_vel_mean, lin_acc_mean, ang_acc_mean, controls_mean = mean[:3], mean[3:6], mean[6:9], mean[9:12], mean[12:15], mean[15:19]
    lin_pos_std, lin_vel_std, ang_vel_std, lin_acc_std, ang_acc_std, controls_std = std[:3], std[3:6], std[6:9], std[9:12], std[12:15], std[15:19]

        # Compute mean and std for each feature
    # lin_pos_mean = linear_pos.mean(axis=0, keepdims=True)
    # lin_pos_std = linear_pos.std(axis=0, keepdims=True) + 1e-8
    
    # lin_vel_mean = linear_vel.mean(axis=0, keepdims=True)
    # lin_vel_std = linear_vel.std(axis=0, keepdims=True) + 1e-8
    
    # lin_acc_mean = linear_acc.mean(axis=0, keepdims=True)
    # lin_acc_std = linear_acc.std(axis=0, keepdims=True) + 1e-8
    
    # ang_vel_mean = angular_vel.mean(axis=0, keepdims=True)
    # ang_vel_std = angular_vel.std(axis=0, keepdims=True) + 1e-8
    
    # ang_acc_mean = angular_acc.mean(axis=0, keepdims=True)
    # ang_acc_std = angular_acc.std(axis=0, keepdims=True) + 1e-8
    
    # rest_mean = rest_of_the_data.mean(axis=0, keepdims=True)
    # rest_std = rest_of_the_data.std(axis=0, keepdims=True) + 1e-8
    
    # controls_mean = controls.mean(axis=0, keepdims=True)
    # controls_std = controls.std(axis=0, keepdims=True) + 1e-8

    # Standardize (z-score normalization)
    linear_pos_normalized = (linear_pos - lin_pos_mean) / lin_pos_std
    linear_vel_normalized = (linear_vel - lin_vel_mean) / lin_vel_std
    linear_acc_normalized = (linear_acc - lin_acc_mean) / lin_acc_std
    angular_vel_normalized = (angular_vel - ang_vel_mean) / ang_vel_std
    angular_acc_normalized = (angular_acc - ang_acc_mean) / ang_acc_std
    # rest_of_the_data_normalized = (rest_of_the_data - rest_mean) / rest_std
    controls_normalized = (controls - controls_mean) / controls_std

    dataset_normalized = np.hstack((time, dt, linear_pos_normalized, linear_vel_normalized, linear_acc_normalized, angular_vel_normalized, angular_acc_normalized, angular_pos, rest_of_the_data, controls_normalized))
    
    return dataset_normalized

def split_data(dataset=None, time_period = 3, dt=0.2, t0 = 25, t1 = 125, t2 = 220):
    """
    Split dataset into training, validation, and testing sets based on time intervals.
    """
    m, _ = dataset.shape

    # Define the delta_t in terms of number of samples
    delta_t = int(time_period / dt)  # number of samples corresponding to the time_period (3/0.2 = 15 samples)

    # Find the first index that corresponding to t0, t1, t2
    start_t0 = np.searchsorted(dataset[:, 0], t0) # index where time >= t0 ( it takes the first index that satisfies the condition)
    start_t1 = np.searchsorted(dataset[:, 0], t1) 
    start_t2 = np.searchsorted(dataset[:, 0], t2)

    end_t0 = start_t0 + delta_t
    end_t1 = start_t1 + delta_t
    end_t2 = start_t2 + delta_t 
    
    data_test_0 = dataset[ start_t0:end_t0, : ]  
    data_test_1 = dataset[ start_t1:end_t1, : ]  
    data_test_2 = dataset[ start_t2:end_t2, : ]

    # Make an array for the tests
    data_test = [data_test_0, data_test_1, data_test_2]

    # Separate testing data from the rest
    data_train_val_0 = dataset[0:start_t0, :]
    data_train_val_1 = dataset[end_t0:start_t1, :]
    data_train_val_2 = dataset[end_t1:start_t2, :]
    data_train_val_3 = dataset[end_t2:m, :]

    # Ensure that the testing/validation samples are evenly distributed in each set, to ease the pair creation (current, next)
    if (data_train_val_0.shape[0] % 2) != 0:
        data_train_val_0 = data_train_val_0[:-1, :] # remove the last sample if odd
    if (data_train_val_1.shape[0] % 2) != 0:
        data_train_val_1 = data_train_val_1[:-1, :]
    if (data_train_val_2.shape[0] % 2) != 0:
        data_train_val_2 = data_train_val_2[:-1, :]
    if (data_train_val_3.shape[0] % 2) != 0:
        data_train_val_3 = data_train_val_3[:-1, :]

    data_train_val = np.vstack((data_train_val_0, data_train_val_1, data_train_val_2, data_train_val_3))

    return data_test, data_train_val
    
def configure_data(dataset):
    m, n = dataset.shape
    dt_test = dataset[:, 1:2]
    linear_pos = dataset[:,2:5]
    linear_vel = dataset[:,5:8]
    linear_acc = dataset[:, 8:11]
    angular_vel = dataset[:,11:14]
    angular_acc = dataset[:, 14:17]
    angular_pos = dataset[:,17:20]
    states = np.hstack((linear_pos, angular_pos, linear_vel, angular_vel, linear_acc, angular_acc))
    controls = dataset[:, 29:33]

    # states = torch.tensor(np.array(states, dtype=np.float32))
    # controls = torch.tensor(np.array(controls, dtype=np.float32))
    # dt_test = torch.tensor(np.array(dt_test, dtype=np.float32))

    return states, controls, dt_test

def create_and_shuffle_pairs(data=None):
    """
    Create non-overlapping (current, next) pairs and shuffle them.
    This ensures each state appears exactly once in the dataset.
    """
    n_pairs = (data.shape[0] - 1)  # All possible consecutive pairs
    
    # Stack current and next states
    current_samples = data[:-1]   # All except last
    next_samples = data[1:]       # All except first
    
    # Create indices and shuffle
    indices = np.arange(n_pairs)
    np.random.seed(42)
    np.random.shuffle(indices)
    
    # Return shuffled pairs
    return current_samples[indices], next_samples[indices] 

def configure_training_and_validation_data(data_current=None, data_next=None):
    """
    Prepare paired data for training. 
    
    Args:
        data_current: Array of current states
        data_next: Array of corresponding next states
    """
    # Extract current state features
    linear_pos_curr = data_current[:,2:5]
    linear_vel_curr = data_current[:,5:8]
    linear_acc_curr = data_current[:, 8:11]
    angular_vel_curr = data_current[:,11:14]
    angular_acc = data_current[:, 14:17]
    angular_pos_curr = data_current[:,17:20]
    states_curr = np.hstack((linear_pos_curr, angular_pos_curr, linear_vel_curr, angular_vel_curr, linear_acc_curr, angular_acc))
    controls_curr = data_current[:, 29:33]
    
    # Extract next state features
    dt = data_next[: ,1]  # time step between current and next state
    linear_pos_next = data_next[:,2:5]
    linear_vel_next = data_next[:,5:8]
    linear_acc_next = data_next[:, 8:11]
    angular_vel_next = data_next[:,11:14]
    angular_acc_next = data_next[:, 14:17]
    angular_pos_next = data_next[:,17:20]
    states_next = np.hstack((linear_pos_next, angular_pos_next, linear_vel_next, angular_vel_next, linear_acc_next, angular_acc_next))
    controls_next = data_next[:, 29:33]

    states_curr = np.array(states_curr, dtype=np.float32)
    states_next = np.array(states_next, dtype=np.float32)
    controls_curr = np.array(controls_curr, dtype=np.float32)
    controls_next = np.array(controls_next, dtype=np.float32)
    dt = np.array(dt, dtype=np.float32)
    
    # vel_curr_norm, vel_next_norm, controls_norm = normalize_data(states_curr[:,3:], states_next[:,3:], controls) # to normalize only velocities and not positions

    # states_curr_norm = np.hstack((states_curr[:, :3], vel_curr_norm))
    # states_next_norm = np.hstack((states_next[:, :3], vel_next_norm))
    
    # Convert to tensors
    X_curr = torch.tensor(states_curr)
    X_next = torch.tensor(states_next)
    U_curr = torch.tensor(controls_curr)
    U_next = torch.tensor(controls_next)
    dt = torch.tensor(dt)
    
    return X_curr, X_next, U_curr, U_next, dt


def save_pairs_to_csv(X_current=None, X_next=None, U_curr=None, U_next=None, dt=None, filename=None):
    """
    Save paired data to CSV with clear structure for inspection.
    """
    import pandas as pd
    
    # Convert tensors to numpy if needed
    if isinstance(X_current, torch. Tensor):
        X_current = X_current.cpu().numpy()
        X_next = X_next. cpu().numpy()
        U_curr = U_curr.cpu().numpy()
        U_next = U_next.cpu().numpy()
        dt = dt.cpu().numpy()
    
    # Create a dataframe with descriptive column names
    df = pd.DataFrame({
        # Pair index
        'pair_idx':  np.arange(len(X_current)),
        
        # Time step
        'dt': dt,
        
        # Current state
        'curr_x': X_current[:, 0],
        'curr_y': X_current[:, 1],
        'curr_z': X_current[:, 2],
        'curr_roll': X_current[:, 3],
        'curr_pitch': X_current[:, 4],
        'curr_yaw': X_current[:, 5],
        'curr_vx': X_current[:, 6],
        'curr_vy': X_current[:, 7],
        'curr_vz': X_current[:, 8],
        'curr_wx': X_current[:, 9],
        'curr_wy': X_current[:, 10],
        'curr_wz': X_current[:, 11],
        'curr_ax': X_current[:, 12],
        'curr_ay': X_current[:, 13],
        'curr_az': X_current[:, 14],
        'curr_alpha_x': X_current[:, 15],
        'curr_alpha_y': X_current[:, 16],
        'curr_alpha_z': X_current[:, 17],

        # Current controls
        'thrust': U_curr[:, 0],
        'torque_roll': U_curr[:, 1],
        'torque_pitch': U_curr[:, 2],
        'torque_yaw': U_curr[: , 3],
        
        # Next state
        'next_x': X_next[:, 0],
        'next_y': X_next[:, 1],
        'next_z': X_next[:, 2],
        'next_roll': X_next[:, 3],
        'next_pitch': X_next[:, 4],
        'next_yaw': X_next[:, 5],
        'next_vx': X_next[:, 6],
        'next_vy': X_next[:, 7],
        'next_vz': X_next[:, 8],
        'next_wx': X_next[:, 9],
        'next_wy': X_next[:, 10],
        'next_wz': X_next[:, 11],
        'next_ax': X_next[:, 12],
        'next_ay': X_next[:, 13],
        'next_az': X_next[:, 14],
        'next_alpha_x': X_next[:, 15],
        'next_alpha_y': X_next[:, 16],
        'next_alpha_z': X_next[:, 17],

        # Next controls
        'thrust': U_next[:, 0],
        'torque_roll': U_next[:, 1],
        'torque_pitch': U_next[:, 2],
        'torque_yaw': U_next[: , 3],
    })
    
    
    # Save to CSV
    df.to_csv(filename, index=False, float_format='%.6f')
    print(f"✅ Saved {len(df)} pairs to '{filename}'")
    
    return df



def normalize_NN_inputs(X_current=None, U_curr=None, mean=None, std=None):
    """
    Normalize NN inputs using provided mean and std.
    """
    # Unpack mean and std
    lin_pos_mean, lin_vel_mean, ang_vel_mean, lin_acc_mean, ang_acc_mean, controls_mean = mean[:3], mean[3:6], mean[6:9], mean[9:12], mean[12:15], mean[15:19]
    lin_pos_std, lin_vel_std, ang_vel_std, lin_acc_std, ang_acc_std, controls_std = std[:3], std[3:6], std[6:9], std[9:12], std[12:15], std[15:19]

    # Normalize current states
    linear_pos_curr_norm = (X_current[:3] - lin_pos_mean) / lin_pos_std
    linear_vel_curr_norm = (X_current[6:9] - lin_vel_mean) / lin_vel_std
    angular_vel_curr_norm = (X_current[9:12] - ang_vel_mean) / ang_vel_std
    # linear_acc_curr_norm = (X_current[12:15] - lin_acc_mean) / lin_acc_std
    # angular_acc_curr_norm = (X_current[15:18] - ang_acc_mean) / ang_acc_std
    controls_curr_norm = (U_curr - controls_mean) / controls_std

    # Reconstruct normalized current state tensor
    X_current_norm = np.concatenate((linear_pos_curr_norm, X_current[3:6], linear_vel_curr_norm, angular_vel_curr_norm)) #, linear_acc_curr_norm, angular_acc_curr_norm))
    U_curr_norm = controls_curr_norm

    return X_current_norm, U_curr_norm



def denormalize_NN_outputs(X_pred=None, mean=None, std=None):
    """
    Denormalize NN outputs using provided mean and std.
    """
    # Move mean and std from cpu to the same device as X_pred
    mean = torch.tensor(mean, dtype=torch.float32, device=X_pred.device)
    std = torch.tensor(std, dtype=torch.float32, device=X_pred.device)

    # Unpack mean and std
    lin_pos_mean, lin_vel_mean, ang_vel_mean, lin_acc_mean, ang_acc_mean, controls_mean = mean[:3], mean[3:6], mean[6:9], mean[9:12], mean[12:15], mean[15:19]
    lin_pos_std, lin_vel_std, ang_vel_std, lin_acc_std, ang_acc_std, controls_std = std[:3], std[3:6], std[6:9], std[9:12], std[12:15], std[15:19]

    # Denormalize predicted states
    linear_acc_pred_denorm = X_pred[:,:3] * lin_acc_std + lin_acc_mean
    angular_acc_pred_denorm = X_pred[:,3:6] * ang_acc_std + ang_acc_mean
    linear_vel_pred_denorm = X_pred[:,6:9] * lin_vel_std + lin_vel_mean
    angular_vel_pred_denorm = X_pred[:,9:12] * ang_vel_std + ang_vel_mean
    

    # Reconstruct denormalized next state tensor
    # X_pred_denorm = np.concatenate((linear_acc_denorm, angular_acc_denorm, linear_vel_denorm, angular_vel_denorm))
    X_pred_norm = torch.cat((linear_acc_pred_denorm, angular_acc_pred_denorm, linear_vel_pred_denorm, angular_vel_pred_denorm), dim=1) # dim=1 to concatenate along the feature dimension

    return X_pred_norm
