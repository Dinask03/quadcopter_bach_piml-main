import torch
import torch.nn as nn

class ResidualBModel(nn.Module):
    """
    Residual B:
    x_hat = x0 + h * ( f(x0,u) + h * T^{-1}( corr_net(z) ) )

    corr_net(z) outputs a 12-dim correction in *normalized* derivative space.
    T is a diagonal scaling used to normalize targets; T^{-1} maps back to physical units.
    """
    def __init__(self, hidden_layers_size, activation_fn):
        super(ResidualBModel, self).__init__()
        n_state = 12
        n_control = 4
        n_input = 6 + 3 + 3 + n_control # sin/cos angles + linear velocities + angular velocities + control inputs

        layers = [nn.Linear(n_input, hidden_layers_size[0]), activation_fn()]
        for i in range(len(hidden_layers_size) - 1):
            layers.append(nn.Linear(hidden_layers_size[i], hidden_layers_size[i+1]))
            layers.append(activation_fn())
        layers.append(nn.Linear(hidden_layers_size[-1], n_state))
        self.corr_net = nn.Sequential(*layers)

        for m in self.corr_net.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        if T_inv is None:
            self.register_buffer('T_inv', torch.ones(self.n_state))
        else:
            T_inv = torch.as_tensor(T_inv, dtype=torch.float32)
            self.register_buffer('T_inv', T_inv)

    def build_features(state_vector, control_input): # For the input to the correction network
            """
            state_vector: (B,12) [x,y,z, roll,pitch,yaw, vx,vy,vz, w_roll,w_pitch,w_yaw]
            control_input: (B,4)
            features z: (B,16) = [sin/cos(roll,pitch,yaw), v(3), w(3), u(4)]
            """
            roll = state_vector[:, 3]
            pitch = state_vector[:, 4]
            yaw = state_vector[:, 5]

            trig = torch.stack([ # sin/cos of angles to help the network learn periodicity and avoid discontinuities at +-pi
                torch.sin(roll), torch.cos(roll),
                torch.sin(pitch), torch.cos(pitch),
                torch.sin(yaw), torch.cos(yaw),
            ], dim=1)  # (B,6)

            v = state_vector[:, 6:9]     # (B,3)
            w = state_vector[:, 9:12]    # (B,3)

            z = torch.cat([trig, v, w, control_input], dim=1)  # (B,16)
            return z

    def forward(self, state0, control_input, h, f0): 
        # state0 is the initial state, f0 is the system dynamics, h is the time step
        if not torch.is_tensor(h): # ensure h is a tensor for consistent device and dtype handling
            h = torch.tensor(h, device=state0.device, dtype=state0.dtype)
        if h.ndim == 0: # scalar, make it (1,1) for broadcasting
            h = h.view(1, 1)  # broadcast later
        elif h.ndim == 1:
            h = h.view(-1, 1)

        z = self.build_features(state0, control_input)    # (B,16)
        corr_norm = self.corr_net(z)                     # (B,12) in normalized space
        corr = corr_norm * self.T_inv                    # apply T^{-1} (diagonal)

        x_hat = state0 + h * (f0 + h * corr)
        return x_hat, corr

