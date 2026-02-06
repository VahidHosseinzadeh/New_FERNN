import torch
import torch.nn as nn
import random
import torch.nn.functional as F



class DiffLucasKanade(nn.Module):
    """
    Differentiable Lucas-Kanade Optical Flow Estimator.
    Estimates integer pixel velocities between two frames by searching
    over a discrete set of possible velocities within a specified range.
    """
    def __init__(self, v_range=3, smooth=0.01):
        super().__init__()
        self.v_range = v_range
        self.smooth = smooth 

        self.v_list = [(dy, dx) for dy in range(-v_range, v_range + 1) 
                                 for dx in range(-v_range, v_range + 1)]
        self.num_v = len(self.v_list)

        # Register as buffer for device placement
        self.register_buffer('vel_tensor', 
                           torch.tensor(self.v_list, dtype=torch.long))

    def forward(self, f_t, f_t_prev):
        B, C, H, W = f_t.shape

        errors = torch.zeros(B, self.num_v, device=f_t.device, dtype=f_t.dtype)

        # compute MSE for all velocities 
        for i, (dy, dx) in enumerate(self.v_list):

            # torch.roll shifts dims: (vertical, horizontal) = (dy, dx)
            f_warp = torch.roll(f_t_prev, shifts=(dy, dx), dims=(-2, -1))
            
            # Compute per-sample MSE: (B,)
            mse = torch.sum((f_t - f_warp) ** 2, dim=(1, 2, 3))
            errors[:, i] = mse

        # Normalize errors by image size for stability
        errors = errors / (C * H * W)

        # -errors / smooth: lower error → higher logit
        logits = -errors / self.smooth  # (B, num_v)
        probs = F.softmax(logits, dim=1)  # (B, num_v)

        return probs







class FERNN_Cell(nn.Module):
    def __init__(self, input_channels, hidden_channels,
                 h_kernel_size=3, u_kernel_size=3):
        super().__init__()
        self.hidden_channels = hidden_channels

        # Convolution W for hidden state: W ⋆ h_t
        h_pad = h_kernel_size // 2
        self.conv_h = nn.Conv2d(hidden_channels, hidden_channels, h_kernel_size,
                                padding=h_pad, padding_mode='circular', bias=False)

        # Encoder E (convolution U) for input: U ⋆ f_t
        u_pad = u_kernel_size // 2
        self.conv_u = nn.Conv2d(input_channels, hidden_channels, u_kernel_size,
                                padding=u_pad, padding_mode='circular', bias=False)

        # Activation function
        self.activation = nn.ReLU()


    def apply_flow_soft(self, x, probs, v_list):
        """
        Differentiable flow (continuous): compute expected (dy, dx) and warp once.
        Uses circular (wrap-around) behavior in pixel space.
        """
        B, C, H, W = x.shape

        # v_list is (dy, dx)
        v = torch.tensor(v_list, device=x.device, dtype=x.dtype)  # (V, 2)
        expected = probs @ v  # (B, 2) -> (dy, dx)

        dy = expected[:, 0].view(B, 1, 1)
        dx = expected[:, 1].view(B, 1, 1)

        # Base grid in pixel coords
        yy, xx = torch.meshgrid(
            torch.arange(H, device=x.device, dtype=x.dtype),
            torch.arange(W, device=x.device, dtype=x.dtype),
            indexing="ij"
        )
        yy = yy.unsqueeze(0).expand(B, -1, -1)
        xx = xx.unsqueeze(0).expand(B, -1, -1)

        # Shift (inverse warp)
        yy = yy - dy
        xx = xx - dx

        # Circular wrap
        yy = torch.remainder(yy, H)
        xx = torch.remainder(xx, W)

        # Normalize to [-1, 1] for grid_sample
        yy_norm = (yy / (H - 1)) * 2 - 1 if H > 1 else yy * 0
        xx_norm = (xx / (W - 1)) * 2 - 1 if W > 1 else xx * 0

        grid = torch.stack([xx_norm, yy_norm], dim=-1)  # (B, H, W, 2)

        return F.grid_sample(
            x, grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True
        )

    def forward(self, f_t, h_t, probs=None, v_list=None):

        # Step 1: Convolve hidden state: [W ⋆ h_t]
        conv_h = self.conv_h(h_t)  # (batch, hidden_channels, H, W) I am doing conv first. does it matter? TODO

        warped_conv_h = self.apply_flow_soft(conv_h, probs, v_list)

        encoded_f = self.conv_u(f_t)  # (batch, hidden_channels, H, W)

        h_next = self.activation(warped_conv_h + encoded_f)

        return h_next





class Seq2SeqFERNN(nn.Module):

    def __init__(self, input_channels, hidden_channels, height, width,
                 output_channels=None, h_kernel_size=3, u_kernel_size=3,
                 v_range=3, decoder_conv_layers=1):
        super().__init__()
        self.height = height
        self.width = width
        self.output_channels = output_channels or input_channels
        self.v_range = v_range

        # velocity predictor 
        self.velocity_predictor = DiffLucasKanade(v_range=v_range, smooth=0.01)
        
        
        
        # FERNN Cell
        self.cell = FERNN_Cell(
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



    def forward(self, input_seq, pred_len, teacher_forcing_ratio=0.0,
                target_seq=None, return_vel_probs=False):
        batch, T_in, C, H, W = input_seq.shape
        device = input_seq.device
        
        # start with zero hidden state
        h = torch.zeros(batch, self.cell.hidden_channels, self.height, self.width, device=device)

  
        vel_probs_list = []

        # --- Encoder Phase ---
        for t in range(T_in):
            f_t = input_seq[:, t]
            if t == 0:
                u_t = torch.zeros(batch, 2, device=device, dtype=torch.long)
                probs = torch.zeros(batch, self.velocity_predictor.num_v, device=device)
                probs[:, self.velocity_predictor.num_v // 2] = 1.0
            else:
                f_t_prev = input_seq[:, t-1]
                probs = self.velocity_predictor(f_t, f_t_prev)

            h = self.cell(f_t, h, probs=probs, v_list=self.velocity_predictor.v_list)
            if return_vel_probs:
                vel_probs_list.append(probs)

        # --- Decoder Phase ---
        prev_frame = input_seq[:, -1]
        predictions = []

        for t in range(pred_len):
            if self.training and target_seq is not None and torch.rand(1) < teacher_forcing_ratio:
                current_frame = target_seq[:, t]
            else:
                current_frame = prev_frame

            if t == 0:
                f_t_prev = input_seq[:, -1]
            else:
                f_t_prev = predictions[-1].detach()  

            probs = self.velocity_predictor(current_frame, f_t_prev)


            h = self.cell(current_frame, h, probs=probs, v_list=self.velocity_predictor.v_list)

            pred = self.decoder(h)
            predictions.append(pred)
            prev_frame = pred.detach()  

            if return_vel_probs:
                vel_probs_list.append(probs)

        predictions = torch.stack(predictions, dim=1)

        output = [predictions]
        if return_vel_probs:
            output.append(torch.stack(vel_probs_list, dim=1))

        if len(output) == 1:
            return output[0]
        return tuple(output)









# class FERNN_Cell(nn.Module):
#     def __init__(self, input_channels, hidden_channels,
#                  h_kernel_size=3, u_kernel_size=3, v_range=0):
#         super().__init__()
#         self.hidden_channels = hidden_channels
#         self.v_list = [(x, y) for x in range(-v_range, v_range + 1) for y in range(-v_range, v_range + 1)]
#         self.num_v = len(self.v_list)

#         # circular convs without bias
#         u_pad = u_kernel_size // 2
#         h_pad = h_kernel_size // 2
#         self.conv_u = nn.Conv2d(input_channels, hidden_channels, u_kernel_size,
#                                  padding=u_pad, padding_mode='circular', bias=False)
#         self.conv_h = nn.Conv2d(hidden_channels, hidden_channels, h_kernel_size,
#                                  padding=h_pad, padding_mode='circular', bias=False)
#         self.activation = nn.ReLU()

#     def forward(self, u_t, h_prev):
#         # u_t: (batch, C, H, W)
#         # h_prev: (batch, num_v, hidden, H, W)
#         batch, C, H, W = u_t.size()
#         # conv_u then expand
#         u_conv = self.conv_u(u_t)  # (batch, hidden, H, W)
#         u_conv = u_conv.unsqueeze(1).expand(-1, self.num_v, -1, -1, -1)

#         # shift hidden via torch.roll per velocity
#         h_shift = []
#         for i, (vx, vy) in enumerate(self.v_list):
#             h_shift.append(torch.roll(h_prev[:, i], shifts=(vy, vx), dims=(2, 3)))
#         h_shift = torch.stack(h_shift, dim=1)  # (batch, num_v, hidden, H, W)

#         # conv_h on flattened v dimension
#         h_flat = h_shift.view(batch * self.num_v, self.hidden_channels, H, W)
#         h_conv = self.conv_h(h_flat)
#         h_conv = h_conv.view(batch, self.num_v, self.hidden_channels, H, W)

#         # combine and activate
#         h_next = self.activation(u_conv + h_conv)
#         return h_next







# class Seq2SeqFERNN(nn.Module):
#     def __init__(self, input_channels, hidden_channels, height, width,
#                  output_channels=None, h_kernel_size=3, u_kernel_size=3,
#                  v_range=0, pool_type='max', decoder_conv_layers=1):
#         super().__init__()
#         self.height = height
#         self.width = width
#         self.pool_type = pool_type
#         self.output_channels = output_channels or input_channels

#         self.cell = FERNN_Cell(
#             input_channels, hidden_channels,
#             h_kernel_size, u_kernel_size, v_range)
#         self.hidden_channels = hidden_channels
#         self.num_v = self.cell.num_v

#         decoder = []
#         for _ in range(decoder_conv_layers):
#             decoder += [nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1, padding_mode='circular', bias=False), nn.ReLU()]
#         decoder += [nn.Conv2d(hidden_channels, self.output_channels, 3, padding=1, padding_mode='circular', bias=False)]
#         self.decoder_conv = nn.Sequential(*decoder)

#     def forward(self, input_seq, pred_len, teacher_forcing_ratio=0.0, target_seq=None, return_hidden=False):
#         batch, T_in, C, H, W = input_seq.size()
#         device = input_seq.device

#         if return_hidden:
#             input_seq_hiddens = torch.zeros(batch, T_in, self.num_v, self.hidden_channels, H, W, device=device)
#             out_seq_hiddens = torch.zeros(batch, pred_len, self.num_v, self.hidden_channels, H, W, device=device)

#         # Initialize hidden state
#         h = torch.zeros(batch, self.num_v, self.hidden_channels, H, W, device=device)

#         # Encoder pass through cell
#         for t in range(T_in):
#             u_t = input_seq[:, t]
#             h = self.cell(u_t, h)

#             if return_hidden:
#                 input_seq_hiddens[:, t] += h.detach()

#         prev = input_seq[:, -1]
#         outputs = []

#         # Decoder
#         for t in range(pred_len):
#             if self.training and target_seq is not None and random.random() < teacher_forcing_ratio:
#                 frame = target_seq[:, t]
#             else:
#                 frame = prev.detach()
#             h = self.cell(frame, h)

#             if return_hidden:
#                 out_seq_hiddens[:, t] += h.detach()

#             # pool over velocities
#             if self.pool_type == 'max':
#                 feat = h.max(1)[0]
#             elif self.pool_type == 'mean':
#                 feat = h.mean(1)
#             elif self.pool_type == 'sum':
#                 feat = h.sum(1)
#             else:
#                 feat = h.max(1)[0]

#             out = self.decoder_conv(feat)
#             outputs.append(out)
#             prev = out

#         if return_hidden:
#             return torch.stack(outputs, dim=1), input_seq_hiddens, out_seq_hiddens # _, (B, T_in, num_v, C, H, W), (B, T_out, num_v, C, H, W)
#         else:
#             return torch.stack(outputs, dim=1)
