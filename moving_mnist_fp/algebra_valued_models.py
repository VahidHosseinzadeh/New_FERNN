import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


class FERNNg_Cell(nn.Module):
    """
    h_t has shape (B, 2, C_h, H, W)
    We flatten it to (B, 2*C_h, H, W) for Conv2d.

    Update:
        u_t = - <h_t>   (global, shape (B,2))
        h_{t+1} = psi_1(u_t) · (W * h_t) + E[f_t]
    """

    def __init__(self, input_channels, hidden_channels,
                 h_kernel_size=3, u_kernel_size=3,
                 use_nonlinearity=True):
        super().__init__()

        self.hidden_channels = hidden_channels  # C_h
        self.d = 2
        self.use_nonlinearity = use_nonlinearity

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

        self._init_W_sum1_identity()

    def _init_W_sum1_identity(self):
        """
        Initialize conv_h close to identity (per channel),
        then normalize so sum(W)=1 per (out,in).
        """
        kH, kW = self.conv_h.kernel_size
        cy, cx = kH // 2, kW // 2

        nn.init.zeros_(self.conv_h.weight)
        with torch.no_grad():
            # identity across 2*C_h channels
            for i in range(2 * self.hidden_channels):
                self.conv_h.weight[i, i, cy, cx] = 1.0

        self.normalize_W_sum1()

    def normalize_W_sum1(self):
        """
        Enforce discrete analog of ∫W=1:
            sum_{y} W(y) = 1

        We enforce it per (out_chan, in_chan).
        """
        with torch.no_grad():
            w = self.conv_h.weight  # (2C,2C,kH,kW)
            s = w.sum(dim=(2, 3), keepdim=True) + 1e-8
            self.conv_h.weight[:] = w / s

    def flatten_h(self, h):
        """
        h: (B, 2, C_h, H, W)
        -> (B, 2*C_h, H, W)
        """
        B, d, C, H, W = h.shape
        assert d == 2 and C == self.hidden_channels
        return h.reshape(B, 2 * C, H, W)

    def unflatten_h(self, h_flat):
        """
        h_flat: (B, 2*C_h, H, W)
        -> (B, 2, C_h, H, W)
        """
        B, twoC, H, W = h_flat.shape
        C = self.hidden_channels
        assert twoC == 2 * C
        return h_flat.reshape(B, 2, C, H, W)

    def spatial_mean_u(self, h):
        """
        Compute u_t = -<h_t>.

        We average over:
            - feature channels
            - spatial dimensions

        h: (B, 2, C_h, H, W)
        returns u: (B, 2)
        """
        mean_h = h.mean(dim=(2, 3, 4))  # mean over (C,H,W) -> (B,2)
        u = -mean_h
        return u

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
        """
        f: (B, C_in, H, W)
        h: (B, 2, C_h, H, W)

        returns:
          h_next: (B, 2, C_h, H, W)
        """

        # u_t from hidden
        u = self.spatial_mean_u(h)  # (B,2)

        # flatten hidden for conv
        h_flat = self.flatten_h(h)  # (B,2*C_h,H,W)

        # W * h_t
        conv_h = self.conv_h(h_flat)

        # psi_1(u) · (W*h_t)
        warped_conv_h = self.warp_by_u(conv_h, u)

        # E[f_t]
        encoded_f = self.conv_u(f)

        # update
        h_next_flat = warped_conv_h + encoded_f

        if self.use_nonlinearity:
            h_next_flat = self.activation(h_next_flat)

        return self.unflatten_h(h_next_flat)



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

        # Decoder: map (B,2,C_h,H,W) -> image
        decoder_layers = []
        for _ in range(decoder_conv_layers):
            decoder_layers.extend([
                nn.Conv2d(2 * hidden_channels, 2 * hidden_channels, 3,
                          padding=1, padding_mode="circular", bias=False),
                nn.ReLU()
            ])

        decoder_layers.append(
            nn.Conv2d(2 * hidden_channels, self.output_channels, 3,
                      padding=1, padding_mode="circular", bias=False)
        )

        self.decoder = nn.Sequential(*decoder_layers)

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
                h0_mode="identity"):

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
            if (
                self.training
                and target_seq is not None
                and torch.rand(1).item() < teacher_forcing_ratio
            ):
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
