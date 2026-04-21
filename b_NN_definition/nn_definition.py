import torch
import torch.nn as nn
import numpy as np

class ResidualBModel(nn.Module):
    """
    Physics-informed residual model.

    Input:  [sin/cos(roll,pitch,yaw), v(3), w(3), u(4)]  -> (B,16)
    Output: derivative corrections [v̇_corr(3), ω̇_corr(3)] -> (B,6)
            (units: m/s² for linear, rad/s² for angular)

    The NN corrects the physics derivative directly:
        v̇_total = v̇_phys + v̇_corr
        ω̇_total = ω̇_phys + ω̇_corr
    This is dt-agnostic: the same correction is valid for any step size.
    """
    def __init__(self, hidden_layers_size, activation_fn, S=None,
                 output_activation=nn.Identity, dropout_rate=0.0):
        super().__init__()
        self.n_out = 6  # v̇_corr (3) + ω̇_corr (3)
        n_control = 4
        n_input = 6 + 3 + 3 + n_control  # sin/cos angles + v(3) + w(3) + u(4) = 16

        layers = [nn.Linear(n_input, hidden_layers_size[0]), activation_fn()]
        if dropout_rate > 0.0:
            layers.append(nn.Dropout(p=dropout_rate))
        for i in range(len(hidden_layers_size) - 1):
            layers += [nn.Linear(hidden_layers_size[i], hidden_layers_size[i + 1]), activation_fn()]
            if dropout_rate > 0.0:
                layers.append(nn.Dropout(p=dropout_rate))

        layers += [nn.Linear(hidden_layers_size[-1], self.n_out), output_activation()]
        self.corr_net = nn.Sequential(*layers)

        for m in self.corr_net.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    @staticmethod
    def build_features(state_vector, control_input):
        """
        state_vector: (B,12) [x,y,z, roll,pitch,yaw, vx,vy,vz, w_roll,w_pitch,w_yaw]
        control_input: (B,4)
        returns features z: (B,16) = [sin/cos(roll,pitch,yaw), v(3), w(3), u(4)]
        """
        roll, pitch, yaw = state_vector[:, 3], state_vector[:, 4], state_vector[:, 5]
        trig = torch.stack([
            torch.sin(roll), torch.cos(roll),
            torch.sin(pitch), torch.cos(pitch),
            torch.sin(yaw), torch.cos(yaw),
        ], dim=1)  # (B,6)
        v = state_vector[:, 6:9]   # (B,3)
        w = state_vector[:, 9:12]  # (B,3)
        return torch.cat([trig, v, w, control_input], dim=1)  # (B,16)

    def forward(self, state_vector, control_input):
        z = self.build_features(state_vector, control_input)
        return self.corr_net(z)  # (B,6): [v̇_corr(3), ω̇_corr(3)]

