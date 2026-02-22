import torch
import torch.nn as nn
import torch.nn.functional as F
from spectral_velocity_predictor_models import PhaseCorrelation



import torch
import torch.nn as nn
import torch.nn.functional as F


class SpecFERNN_Cell(nn.Module):
    def __init__(self, input_channels, hidden_channels,
                 n_modes,
                 h_kernel_size=3, u_kernel_size=3):
        super().__init__()

        self.hidden_channels = hidden_channels
        self.n_modes = n_modes

        # Circular convolutions
        u_pad = u_kernel_size // 2
        h_pad = h_kernel_size // 2

        self.conv_u = nn.Conv2d(
            input_channels, hidden_channels,
            u_kernel_size,
            padding=u_pad,
            padding_mode='circular',
            bias=False
        )

        self.conv_h = nn.Conv2d(
            hidden_channels, hidden_channels,
            h_kernel_size,
            padding=h_pad,
            padding_mode='circular',
            bias=False
        )

        self.activation = nn.LeakyReLU()

    # -------------------------------------------------
    # 🔁 Vectorized circular subpixel warp
    # -------------------------------------------------
    def warp(self, h, u):
        """
        h: (B, n_modes, C, H, W)
        u: (B, n_modes, 2)  (dx right+, dy down+)

        returns:
        warped_h: (B, n_modes, C, H, W)
        """

        B, n_modes, C, H, W = h.shape
        device = h.device
        dtype = h.dtype

        # Flatten modes into batch
        h = h.view(B * n_modes, C, H, W)
        u = u.view(B * n_modes, 2)

        dx = u[:, 0].view(-1, 1, 1)
        dy = u[:, 1].view(-1, 1, 1)



        # Create base grid once
        yy, xx = torch.meshgrid(
            torch.arange(H, device=device, dtype=dtype),
            torch.arange(W, device=device, dtype=dtype),
            indexing="ij"
        )

        yy = yy.unsqueeze(0).expand(B * n_modes, -1, -1)
        xx = xx.unsqueeze(0).expand(B * n_modes, -1, -1)

        # Inverse warp
        yy = yy - dy
        xx = xx - dx

        # Circular wrap
        yy = torch.remainder(yy, H)
        xx = torch.remainder(xx, W)

        # Normalize to [-1,1]
        yy_norm = (yy / (H - 1)) * 2 - 1 if H > 1 else yy * 0
        xx_norm = (xx / (W - 1)) * 2 - 1 if W > 1 else xx * 0

        grid = torch.stack([xx_norm, yy_norm], dim=-1)

        warped = F.grid_sample(
            h,
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=True
        )

        # Restore shape
        warped = warped.view(B, n_modes, C, H, W)

        return warped

    # -------------------------------------------------
    # 🔁 Forward
    # -------------------------------------------------
    def forward(self, f, h, u):
        """
        f: (B, C_in, H, W)
        h: (B, n_modes, C_hidden, H, W)
        u: (B, n_modes, 2)
        """

        B, n_modes, C, H, W = h.shape

        # 1️⃣ Warp hidden states per velocity
        warped_h = self.warp(h, u)

        # 2️⃣ Apply conv_h (shared across modes)
        warped_h_flat = warped_h.view(B * n_modes, C, H, W)
        h_conv = self.conv_h(warped_h_flat)
        h_conv = h_conv.view(B, n_modes, C, H, W)

        # 3️⃣ Encode input once and broadcast
        encoded_f = self.conv_u(f)                     # (B, C, H, W)
        encoded_f = encoded_f.unsqueeze(1)             # (B, 1, C, H, W)
        encoded_f = encoded_f.expand(-1, n_modes, -1, -1, -1)

        # 4️⃣ Combine
        h_next = self.activation(h_conv + encoded_f)

        return h_next





#############################

class SpecSeq2SeqFERNN(nn.Module):

    def __init__(self, input_channels, hidden_channels, height, width,
                 output_channels=None, h_kernel_size=3, u_kernel_size=3,
                 decoder_conv_layers=1, n_modes= 2, subpixel=False, periodic_bc=True,pool_type='max'):
        super().__init__()
        self.height = height
        self.width = width
        self.output_channels = output_channels or input_channels
        self.n_modes = n_modes
        self.pool_type = pool_type  
        self.subpixel = subpixel
        self.periodic_bc = periodic_bc

        # velocity predictor 
        self.velocity_predictor = PhaseCorrelation(n_modes=self.n_modes, 
                                                   subpixel=self.subpixel,
                                                   periodic_bc=self.periodic_bc,
                                                   sorting_velocities=False
                                                   )
        
        # FERNN Cell
        self.cell = SpecFERNN_Cell(
            input_channels, hidden_channels,
            h_kernel_size, u_kernel_size
        )




        # Decoder building
        decoder_layers = []
        for _ in range(decoder_conv_layers):
            decoder_layers.extend([
                nn.Conv2d(hidden_channels, hidden_channels, 3,
                         padding=1, padding_mode='circular', bias=False),
                nn.ReLU()
            ])
        decoder_layers.append(
            nn.Conv2d(hidden_channels, self.output_channels, 3,
                     padding=1, padding_mode='circular', bias=False)
        )
        self.decoder = nn.Sequential(*decoder_layers)
        
    def greedy_match(self, u_prev, u_curr):
        """
        u_prev: (B, n_modes, 2)
        u_curr: (B, n_modes, 2)

        Returns:
            reordered u_curr (B, n_modes, 2)
        """

        B, n, _ = u_prev.shape
        device = u_prev.device

        # Pairwise squared distances
        # (B, n, n)
        dist = torch.cdist(u_prev, u_curr, p=2) ** 2

        matched = torch.zeros_like(u_curr)
        assigned = torch.zeros(B, n, dtype=torch.bool, device=device)

        for i in range(n):
            # mask already assigned columns
            masked_dist = dist.clone()
            masked_dist[assigned.unsqueeze(1).expand(-1, n, -1)] = float('inf')

            # choose nearest for each batch at row i
            j = masked_dist[:, i].argmin(dim=1)  # (B,)

            matched[:, i] = u_curr[torch.arange(B), j]
            assigned[torch.arange(B), j] = True

        return matched



    def forward(self, input_seq, pred_len, teacher_forcing_ratio=0.0,
                target_seq=None, return_vels=False):
        
        B, T_in, C, H, W = input_seq.shape
        device = input_seq.device
        dtype = input_seq.dtype

        h = torch.zeros(
            B,self.n_modes, self.cell.hidden_channels, self.height, self.width,
            device=device, dtype=dtype
        )

        vel_list = []

        # Encoder
        u_prev = torch.zeros(B, self.n_modes, 2, device=device, dtype=dtype)

        for t in range(T_in):

            f_t = input_seq[:, t]

            if t > 0:
                with torch.no_grad():
                    u_raw = self.velocity_predictor(input_seq[:, t - 1], f_t)

                u = self.greedy_match(u_prev, u_raw)
            else:
                u = torch.zeros_like(u_prev)

            h = self.cell(f_t, h, u)

            u_prev = u.detach()

            if return_vels:
                vel_list.append(u_prev)



        # for t in range(T_in):
        #     f_t = input_seq[:, t]
        #     with torch.no_grad():
        #         u = self.velocity_predictor(input_seq[:, t - 1], f_t) if t > 0 else torch.zeros(B, self.n_modes, 2, device=device, dtype=dtype)
        #     h = self.cell(f_t, h, u)

            




        # Decoder
        # Important: velocity is computed from GT target_seq if available.        
        prev_frame = input_seq[:, -1]
        outputs = []
        for t in range(pred_len):
            if self.training and (target_seq is not None) and (torch.rand(1).item() < teacher_forcing_ratio):
                current_frame = target_seq[:, t]
            else:
                current_frame = prev_frame.detach()

            # Compute velocity probs
            if target_seq is not None:
                if t == 0:
                    f_prev_for_vel = input_seq[:, -1]
                    f_curr_for_vel = target_seq[:, 0]
                else:
                    f_prev_for_vel = target_seq[:, t - 1]
                    f_curr_for_vel = target_seq[:, t]
                with torch.no_grad():
                    u_raw = self.velocity_predictor(f_prev_for_vel, f_curr_for_vel)

                u = self.greedy_match(u, u_raw)
            

            
            h = self.cell(current_frame, h, u)
            
            
            # pool over modes
            if self.pool_type == 'max':
                feat = h.max(1)[0]
            elif self.pool_type == 'mean':
                feat = h.mean(1)
            elif self.pool_type == 'sum':
                feat = h.sum(1)
            else:
                feat = h.max(1)[0]

            pred = self.decoder(feat)
            outputs.append(pred)

            prev_frame = pred 

        outputs_seq = torch.stack(outputs, dim=1)  # (B, pred_len, C, H, W)

        if return_vels:
            vel_list = torch.stack(vel_list, dim=1)  # (B, T_in, n_modes, 2)
            return outputs_seq, vel_list
        return outputs_seq
    