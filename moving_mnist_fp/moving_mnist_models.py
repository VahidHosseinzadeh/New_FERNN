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
    def __init__(self, v_range=3, smooth=0.001):
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
                 h_kernel_size=3, u_kernel_size=3, v_range=1):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.v_list = [(x, y) for x in range(-v_range, v_range + 1) for y in range(-v_range, v_range + 1)]

        # circular convs without bias
        u_pad = u_kernel_size // 2
        h_pad = h_kernel_size // 2
        self.conv_u = nn.Conv2d(input_channels, hidden_channels, u_kernel_size,
                                 padding=u_pad, padding_mode='circular', bias=False)
        self.conv_h = nn.Conv2d(hidden_channels, hidden_channels, h_kernel_size,
                                 padding=h_pad, padding_mode='circular', bias=False)


        # Activation function
        self.activation = nn.LeakyReLU()
        self.register_buffer(
                            "vel_tensor",
                            torch.tensor(self.v_list, dtype=torch.float32))   # (V,2) = (dy,dx))



    def warp(self, h, probs):
        """
        Differentiable flow (continuous): compute expected (dy, dx) and warp once.
        Uses circular (wrap-around) behavior in pixel space.
        """
        B, C, H, W = h.shape
        v = self.vel_tensor.to(dtype=h.dtype)   # (V,2)

        expected = probs @ v  # (B, 2) -> (dy, dx).    we are taking the mean of velocities weighted by their probabilities

        dy = expected[:, 0].view(B, 1, 1)
        dx = expected[:, 1].view(B, 1, 1)

        # Base grid in pixel coords
        yy, xx = torch.meshgrid(
            torch.arange(H, device=h.device, dtype=torch.float32),
            torch.arange(W, device=h.device, dtype=torch.float32),
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
            h, grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True
        )

    def forward(self, f, h, probs=None):

        warped_h = self.warp(h, probs)  # (B, hidden, H, W)
        warped_conv_h = self.conv_h(warped_h) 
        encoded_f = self.conv_u(f) 
        h_next = self.activation(warped_conv_h + encoded_f)

        return h_next





#############################

class Seq2SeqFERNN(nn.Module):

    def __init__(self, input_channels, hidden_channels, height, width,
                 output_channels=None, h_kernel_size=3, u_kernel_size=3,
                 v_range=2, decoder_conv_layers=1,smooth_vel_probs=0.001):
        super().__init__()
        self.height = height
        self.width = width
        self.output_channels = output_channels or input_channels
        # velocity predictor 
        self.velocity_predictor = DiffLucasKanade(v_range=v_range, smooth=smooth_vel_probs)
        
        
        
        # FERNN Cell
        self.cell = FERNN_Cell(
            input_channels, hidden_channels,
            h_kernel_size, u_kernel_size, v_range
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
        
        B, T_in, C, H, W = input_seq.shape
        device = input_seq.device
        dtype = input_seq.dtype

        h = torch.zeros(
            B, self.cell.hidden_channels, self.height, self.width,
            device=device, dtype=dtype
        )

        vel_probs_list = []

        # prob =1 for zero velocity. I define it but it not necessary now because h_0 = 0 so warping does not have effect. 
        def zero_velocity_probs():
            probs0 = torch.zeros(B, self.velocity_predictor.num_v, device=device, dtype=dtype)
            probs0[:, self.velocity_predictor.num_v // 2] = 1.0
            return probs0

        # Encoder
        # Velocity is computed from GT pairs in input_seq
        last_probs = zero_velocity_probs()

        for t in range(T_in):
            f_t = input_seq[:, t]

            if t == 0:
                probs = zero_velocity_probs()
            else:
                f_t_prev = input_seq[:, t - 1]

                # here I put this in no_grad because we don't want gradients to flow through the velocity predictor. We want it to be a fixed operation that estimates velocity from GT frames, but for parametric I should change this. 
                with torch.no_grad():
                    probs = self.velocity_predictor(f_t, f_t_prev)

            last_probs = probs  # store last encoder probs

            h = self.cell(
                f_t, h,
                probs=probs
            )

            if return_vel_probs:
                vel_probs_list.append(probs)

        # Decoder
        # Important: velocity is computed ONLY from GT target_seq if available.
        # Otherwise, we freeze velocity to last_probs from the encoder.
        

        # Decoder
        outputs = []
        # Step 0: first prediction comes ONLY from h_T
        pred_frame = self.decoder(h)          # predicts f_T
        outputs.append(pred_frame)



        # Initialize current_frame for the loop
        if(self.training and target_seq is not None):
            current_frame = target_seq[:, 0]
        else:
            current_frame = pred_frame.detach()

        for t in range(1, pred_len):
            # # Compute velocity probs
            # if target_seq is not None:
            #     # Use GT to compute velocity
            #     if t == 1:
            #         f_prev_for_vel = input_seq[:, -1]
            #         f_curr_for_vel = target_seq[:, 0]
            #     else:
            #         f_prev_for_vel = target_seq[:, t - 2]
            #         f_curr_for_vel = target_seq[:, t - 1]
                
            #     with torch.no_grad():
            #         probs = self.velocity_predictor(f_curr_for_vel, f_prev_for_vel)
            # else:
            #     # Inference: freeze velocity
            #     probs = last_probs

            # Update hidden state
            h = self.cell(current_frame, h, probs=last_probs)

            # Decode next frame
            pred_frame = self.decoder(h)
            outputs.append(pred_frame)

            # Choose next input frame
            if(self.training and target_seq is not None and torch.rand(1, device=h.device).item() < teacher_forcing_ratio):
                current_frame = target_seq[:, t]
            else:
                current_frame = pred_frame.detach()

        outputs_seq = torch.stack(outputs, dim=1)  # (B, pred_len, C, H, W)

        if return_vel_probs:
            vel_probs = torch.stack(vel_probs_list, dim=1)  # (B, T_in , num_v)
            return outputs_seq, vel_probs
        return outputs_seq













# class FERNN_CellOriginal(nn.Module):
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







# class Seq2SeqFERNNOriginal(nn.Module):
#     def __init__(self, input_channels, hidden_channels, height, width,
#                  output_channels=None, h_kernel_size=3, u_kernel_size=3,
#                  v_range=0, pool_type='max', decoder_conv_layers=1):
#         super().__init__()
#         self.height = height
#         self.width = width
#         self.pool_type = pool_type
#         self.output_channels = output_channels or input_channels

#         self.cell = FERNN_CellOriginal(
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














# # -------------------------
# # Decoder (the version when we first decode and there is no lag)
# # -------------------------
# h = h_enc  # this is h_T after encoder
# outputs = []

# # Step 0: first prediction comes ONLY from h_T
# pred_frame = self.decoder(h)          # predicts f_T
# outputs.append(pred_frame)

# # Now decide what frame to feed to update hidden state
# if (self.training and target_seq is not None and
#     torch.rand(1, device=pred_frame.device).item() < teacher_forcing_ratio):
#     current_frame = target_seq[:, 0]   # f_T
# else:
#     current_frame = pred_frame         # \hat f_T

# # Now we run the remaining steps
# for t in range(1, pred_len):
#     # update hidden using the last chosen frame
#     h = self.cell(
#         x=current_frame,
#         h=h,
#         probs=probs
#     )

#     # decode next frame
#     pred_frame = self.decoder(h)       # predicts f_{T+t}
#     outputs.append(pred_frame)

#     # teacher forcing for next step
#     if (self.training and target_seq is not None and
#         torch.rand(1, device=pred_frame.device).item() < teacher_forcing_ratio):
#         current_frame = target_seq[:, t]
#     else:
#         current_frame = pred_frame

# output_seq = torch.stack(outputs, dim=1)  # (B, pred_len, C, H, W)
# return output_seq



# decoder version with lag.  the one that update the hidden state first. 
        # # Decoder
        # # Important: velocity is computed ONLY from GT target_seq if available.
        # # Otherwise, we freeze velocity to last_probs from the encoder.
        # prev_frame = input_seq[:, -1]  # last observed frame (GT)
        # predictions = []

        # for t in range(pred_len):

        #     # -------------------------
        #     # Choose the RNN input frame
        #     # -------------------------
        #     if self.training and (target_seq is not None) and (torch.rand(1).item() < teacher_forcing_ratio):
        #         # teacher forcing for the RNN input
        #         current_frame = target_seq[:, t]
        #     else:
        #         # autoregressive input
        #         current_frame = prev_frame.detach()

        #     # -------------------------
        #     # Choose velocity probs
        #     # -------------------------
        #     if target_seq is not None:
        #         # Use ground truth to compute velocity (no drift)
        #         if t == 0:
        #             # first predicted step: compare target_seq[0] to last input frame
        #             f_prev_for_vel = input_seq[:, -1]
        #             f_curr_for_vel = target_seq[:, 0]
        #         else:
        #             # later: compare target_seq[t] to target_seq[t-1]
        #             f_prev_for_vel = target_seq[:, t - 1]
        #             f_curr_for_vel = target_seq[:, t]

        #         with torch.no_grad():
        #             probs = self.velocity_predictor(f_curr_for_vel, f_prev_for_vel)

        #     else:
        #         # Inference: no GT available -> freeze velocity
        #         probs = last_probs

        #     # -------------------------
        #     # Update hidden state
        #     # -------------------------
        #     h = self.cell(
        #         current_frame, h,
        #         probs=probs
        #     )

        #     # -------------------------
        #     # Decode prediction
        #     # -------------------------
        #     pred = self.decoder(h)
        #     predictions.append(pred)

        #     prev_frame = pred  # for autoregressive input

        #     if return_vel_probs:
        #         vel_probs_list.append(probs)

        # predictions = torch.stack(predictions, dim=1)  # (B, pred_len, C, H, W)

        # if return_vel_probs:
        #     vel_probs = torch.stack(vel_probs_list, dim=1)  # (B, T_in + pred_len, num_v)
        #     return predictions, vel_probs