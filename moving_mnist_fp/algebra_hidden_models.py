import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np




class FERNNg_Cell(nn.Module):
    def __init__(self, input_channels, hidden_channels,
                 h_kernel_size=3, u_kernel_size=3):
        super().__init__()

        self.hidden_channels = hidden_channels  # C_h
        self.d = 2

        u_pad = u_kernel_size // 2
        h_pad = h_kernel_size // 2

        # E: input_channels -> (2 * hidden_channels)
        self.conv_u = nn.Conv2d(
            input_channels, 2 * hidden_channels,
            kernel_size=u_kernel_size,
            padding=u_pad, padding_mode="circular",
            bias=False
        )

        # W: (2*hidden_channels) -> (2*hidden_channels)
        self.conv_h = nn.Conv2d(
            2 * hidden_channels, 2 * hidden_channels,
            kernel_size=h_kernel_size,
            padding=h_pad, padding_mode="circular",
            bias=False
        )

        self.activation = nn.LeakyReLU()

    #     self._init_W_sum1_identity()

    # def _init_W_sum1_identity(self):
    #     """
    #     Initialize conv_h close to identity (per channel),
    #     then normalize so sum(W)=1 per (out,in).
    #     """
    #     kH, kW = self.conv_h.kernel_size
    #     cy, cx = kH // 2, kW // 2

    #     nn.init.zeros_(self.conv_h.weight)
    #     with torch.no_grad():
    #         # identity across 2*C_h channels
    #         for i in range(2 * self.hidden_channels):
    #             self.conv_h.weight[i, i, cy, cx] = 1.0

    #     self.normalize_W_sum1()

    # def normalize_W_sum1(self):
    #     """
    #     Enforce discrete analog of ∫W=1:
    #         sum_{y} W(y) = 1

    #     We enforce it per (out_chan, in_chan).
    #     """
    #     with torch.no_grad():
    #         w = self.conv_h.weight  # (2C,2C,kH,kW)
    #         s = w.sum(dim=(2, 3), keepdim=True) + 1e-8
    #         self.conv_h.weight[:] = w / s

    

    def warp_by_u(self, h_flat, u):
        """
        psi_1(u) · h(x) = h(x - u)

        h_flat: (B, 2*C_h, H, W)
        u: (B,2) in pixels (dx, dy)

        Uses grid_sample + circular wrap.
        """
        B, Ch, H, W = h_flat.shape
        dx = u[:, 0].view(B, 1, 1)
        dy = u[:, 1].view(B, 1, 1)

        yy, xx = torch.meshgrid(
            torch.arange(H, device=h_flat.device, dtype=torch.float32),
            torch.arange(W, device=h_flat.device, dtype=torch.float32),
            indexing="ij"
        )
        yy = yy.unsqueeze(0).expand(B, -1, -1)
        xx = xx.unsqueeze(0).expand(B, -1, -1)

        xx = xx - dx
        yy = yy - dy

        # circular wrap
        xx = torch.remainder(xx, W)
        yy = torch.remainder(yy, H)

        xx_norm = (xx / (W - 1)) * 2 - 1 if W > 1 else xx * 0
        yy_norm = (yy / (H - 1)) * 2 - 1 if H > 1 else yy * 0

        grid = torch.stack([xx_norm, yy_norm], dim=-1)  # (B,H,W,2)

        return F.grid_sample(
            h_flat, grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True
        )



    def forward(self, f, h):

        B, d, C, H, W = h.shape

        # u_t from hidden
        u = - h.mean(dim=(2, 3, 4))  # (B,2)

        # flatten hidden for conv
        h_flat = h.reshape(B, 2 * C, H, W)
        warped_h = self.warp_by_u(h_flat, u)
        conved_warped_h = self.conv_h(warped_h)
        encoded_f = self.conv_u(f)
        h_next_flat = conved_warped_h + encoded_f
        h_next_flat = self.activation(h_next_flat)
        h_next = h_next_flat.reshape(B, 2, C, H, W)
        return h_next
 



class Seq2SeqFERNNg(nn.Module):
    def __init__(self, input_channels, hidden_channels, height, width,
                 output_channels=None,
                 h_kernel_size=3, u_kernel_size=3,
                 decoder_conv_layers=1):
        super().__init__()

        self.height = height
        self.width = width
        self.hidden_channels = hidden_channels
        self.output_channels = output_channels or input_channels

        self.cell = FERNNg_Cell(
            input_channels=input_channels,
            hidden_channels=hidden_channels,
            h_kernel_size=h_kernel_size,
            u_kernel_size=u_kernel_size
        )

        # Decoder: map (B,2,C_h,H,W) to image
        decoder = []
        for _ in range(decoder_conv_layers):
            decoder +=[nn.Conv2d(2 * hidden_channels, 2 * hidden_channels, 3,
                          padding=1, padding_mode="circular", bias=False), nn.ReLU()]

        decoder += [
            nn.Conv2d(2 * hidden_channels, self.output_channels, 3,
                      padding=1, padding_mode="circular", bias=False)]

        self.decoder = nn.Sequential(*decoder)

    def init_hidden(self, B, device, dtype, mode="identity"):
        H, W = self.height, self.width
        C_h = self.hidden_channels

        if mode == "zeros":
            return torch.zeros(B, 2, C_h, H, W, device=device, dtype=dtype)

        if mode == "identity":
            yy, xx = torch.meshgrid(
                torch.arange(H, device=device, dtype=dtype),
                torch.arange(W, device=device, dtype=dtype),
                indexing="ij"
            )

            # h0(x)=x replicated across feature channels
            base = torch.stack([xx, yy], dim=0)  # (2,H,W)
            base = base.unsqueeze(1).repeat(1, C_h, 1, 1)  # (2,C_h,H,W)
            return base.unsqueeze(0).repeat(B, 1, 1, 1, 1)

        raise ValueError("mode must be 'zeros' or 'identity'")

    def forward(self, input_seq, pred_len,
                teacher_forcing_ratio=0.0,
                target_seq=None,
                h0_mode="zeros"):

        B, T_in, C, H, W = input_seq.shape
        device = input_seq.device
        dtype = input_seq.dtype

        h = self.init_hidden(B, device, dtype, mode=h0_mode)

        # Encoder
        for t in range(T_in):
            f_t = input_seq[:, t]
            h = self.cell(f_t, h)

        # Decoder
        prev_frame = input_seq[:, -1]
        outputs = []

        for t in range(pred_len):
            if (self.training and target_seq is not None
                and torch.rand(1).item() < teacher_forcing_ratio):
                current_frame = target_seq[:, t]
            else:
                current_frame = prev_frame.detach()

            h = self.cell(current_frame, h)

            # flatten for decoder
            h_flat = h.reshape(B, 2 * self.hidden_channels, H, W)
            pred = self.decoder(h_flat)

            outputs.append(pred)
            prev_frame = pred

        return torch.stack(outputs, dim=1)
