import torch
import torch.nn as nn
import torch.nn.functional as F


class ParametricVelocityPredictor(nn.Module):
    def __init__(self, v_range=2, hidden_dim=64):
        super().__init__()
        self.v_range = v_range
        self.v_list = [(dy, dx) for dy in range(-v_range, v_range + 1) 
                                 for dx in range(-v_range, v_range + 1)]
        self.num_v = len(self.v_list)
        
        # Learnable feature extractor
        self.feature_net = nn.Sequential(
            nn.Conv2d(2, 8, 7, padding=3, padding_mode='circular', bias=False),
            nn.ReLU(),
            # nn.Conv2d(32, 64, 5, padding=2, padding_mode='circular', bias=False),
            # nn.ReLU(),
            nn.Conv2d(8, hidden_dim, 3, padding=1, padding_mode='circular', bias=False),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)  # Global pooling
        )
        
        # Velocity classifier
        self.velocity_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.num_v)
        )
        
        self.register_buffer('vel_tensor', 
                           torch.tensor(self.v_list, dtype=torch.long))
    
    def forward(self, f_t, f_t_prev):
        B = f_t.size(0)
        
        # Concatenate frames
        frame_pair = torch.cat([f_t, f_t_prev], dim=1)  # (B, 2, H, W)
        
        # Extract features
        features = self.feature_net(frame_pair)  # (B, hidden_dim, 1, 1)
        features = features.view(B, -1)  # (B, hidden_dim)
        
        # Predict velocity distribution
        logits = self.velocity_head(features)  # (B, num_v)
        probs = F.softmax(logits, dim=1)
        
        return probs
    







class FERNN_CellVP(nn.Module):
    def __init__(self, input_channels, hidden_channels,
                 h_kernel_size=3, u_kernel_size=3, v_range=2):
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



    def warp(self, h, probs, use_argmax=False):
        """
        Differentiable flow (continuous): compute expected (dy, dx) and warp once.
        Uses circular (wrap-around) behavior in pixel space.
        """
        B, C, H, W = h.shape
        v = self.vel_tensor.to(dtype=h.dtype)   # (V,2)

       
        # so if argmax in used, I am taking the max prob but this is not differentiiable for a parametric velocity. 
        if use_argmax:
            max_indices = torch.argmax(probs, dim=1)  # (B,)
            expected = v[max_indices]  # (B, 2)
        else:
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

    def forward(self, f, h, probs=None, use_argmax=False):

        warped_h = self.warp(h, probs, use_argmax=use_argmax)  # (B, hidden, H, W)
        warped_conv_h = self.conv_h(warped_h) 
        encoded_f = self.conv_u(f) 
        h_next = self.activation(warped_conv_h + encoded_f)

        return h_next





#############################

class Seq2SeqFERNNVP(nn.Module):

    def __init__(self, input_channels, hidden_channels, height, width,
                 output_channels=None, h_kernel_size=3, u_kernel_size=3,
                 v_range=2, decoder_conv_layers=1, vel_hidden_channels=16):
        super().__init__()
        self.height = height
        self.width = width
        self.output_channels = output_channels or input_channels
        

        # velocity predictor 
        self.velocity_predictor = ParametricVelocityPredictor(v_range=v_range, hidden_dim=vel_hidden_channels)
        
        
        
        # FERNN Cell
        self.cell = FERNN_CellVP(
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

                # here I am not using no grad as Iam trainig the vel model
                probs = self.velocity_predictor(f_t, f_t_prev)

            last_probs = probs  # store last encoder probs

            h = self.cell(
                f_t, h,
                probs=probs,
                use_argmax= False #not self.training  
            )

            if return_vel_probs:
                vel_probs_list.append(probs)

        # Decoder
        
        prev_frame = input_seq[:, -1]
        prev_frame_for_vel = input_seq[:, -1].detach()
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
                probs = self.velocity_predictor(f_curr_for_vel, f_prev_for_vel)
            else:
                # as we are in the inference phase, we use no grad  
                with torch.no_grad():
                    if t == 0:
                        probs = last_probs
                    else:
                        probs = self.velocity_predictor(current_frame, prev_frame_for_vel)

            h = self.cell(current_frame, 
                          h, 
                          probs=probs,
                          use_argmax= False)
            
            pred = self.decoder(h)
            outputs.append(pred)

            prev_frame_for_vel = current_frame.detach()
            prev_frame = pred 

        outputs_seq = torch.stack(outputs, dim=1)  # (B, pred_len, C, H, W)

        if return_vel_probs:
            vel_probs = torch.stack(vel_probs_list, dim=1)  # (B, T_in , num_v)
            return outputs_seq, vel_probs
        return outputs_seq



