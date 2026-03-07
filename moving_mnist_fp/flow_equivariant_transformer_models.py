import torch
import torch.nn as nn
import torch.nn.functional as F

class RelativeTimeBias(nn.Module):
    """
    Learnable relative bias for temporal attention.
    Supports variable sequence lengths by indexing into a bias table.
    """
    def __init__(self, num_heads, max_seq_len):
        super().__init__()
        self.num_heads = num_heads
        self.max_seq_len = max_seq_len
        # bias table of shape (2*max_seq_len - 1, num_heads) or (num_heads, 2*max_seq_len - 1)
        self.relative_bias = nn.Parameter(torch.randn(2 * max_seq_len - 1, num_heads) * 0.02)

    def forward(self, seq_len, device):
        """
        Returns bias of shape (1, num_heads, seq_len, seq_len) for the current batch.
        """
        # create relative indices: (seq_len, seq_len)
        range_vec = torch.arange(seq_len, device=device)
        relative_indices = range_vec[None, :] - range_vec[:, None]  # (seq_len, seq_len)
        # shift to [0, 2*max_seq_len-2]
        relative_indices += self.max_seq_len - 1
        # clamp for safety (if seq_len < max_seq_len, some indices may be out of range)
        relative_indices = relative_indices.clamp(0, 2 * self.max_seq_len - 2)
        # gather bias: (seq_len, seq_len, num_heads) -> permute to (num_heads, seq_len, seq_len)
        bias = self.relative_bias[relative_indices]  # (seq_len, seq_len, num_heads)
        bias = bias.permute(2, 0, 1).unsqueeze(0)    # (1, num_heads, seq_len, seq_len)
        return bias


class TemporalSelfAttention(nn.Module):
    """
    Per‑location self‑attention over time with relative time bias.
    Operates on input of shape (batch * height * width, seq_len, dim).
    """
    def __init__(self, dim, num_heads, max_seq_len, dropout=0.1):
        super().__init__()
        assert dim % num_heads == 0
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.max_seq_len = max_seq_len

        self.qkv = nn.Linear(dim, dim * 3)
        self.out_proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        self.relative_bias = RelativeTimeBias(num_heads, max_seq_len)

    def forward(self, x, mask=None):
        """
        Args:
            x: (B_HW, T, dim)
            mask: optional (B_HW, T) or (1, T) boolean mask, True for padding.
        Returns:
            out: (B_HW, T, dim)
        """
        B_HW, T, _ = x.shape
        qkv = self.qkv(x).reshape(B_HW, T, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B_HW, num_heads, T, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # attention scores: (B_HW, num_heads, T, T)
        scores = torch.einsum('b h t d, b h s d -> b h t s', q, k) / (self.head_dim ** 0.5)

        # add relative time bias (shared across spatial locations)
        bias = self.relative_bias(T, x.device)  # (1, num_heads, T, T)
        scores = scores + bias

        if mask is not None:
            # mask shape: (B_HW, T) or (1, T) -> broadcast to (B_HW, 1, T, 1) for key masking?
            # Typically mask indicates padding tokens. We'll set scores of padding positions to -inf.
            # We need to apply mask to both query and key sides. For simplicity, assume mask is (1, T)
            # and all sequences have same length. If mask is per sample, expand.
            if mask.dim() == 2:
                mask = mask.unsqueeze(1).unsqueeze(2)  # (B_HW, 1, 1, T)
            else:
                mask = mask.unsqueeze(0).unsqueeze(2)  # (1, 1, 1, T)
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        out = torch.einsum('b h t s, b h s d -> b h t d', attn, v)
        out = out.transpose(1, 2).reshape(B_HW, T, self.dim)  # (B_HW, T, dim)
        out = self.out_proj(out)
        return out


class EquivariantTransformerBlock(nn.Module):
    """
    One transformer block: LN -> TemporalSelfAttention -> residual -> LN -> FFN -> residual.
    All operations are per‑location and translation‑equivariant.
    """
    def __init__(self, dim, num_heads, max_seq_len, ff_dim, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = TemporalSelfAttention(dim, num_heads, max_seq_len, dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, mask=None):
        # x: (B_HW, T, dim)
        residual = x
        x = self.norm1(x)
        x = self.attn(x, mask)
        x = residual + x

        residual = x
        x = self.norm2(x)
        x = self.ffn(x)
        x = residual + x
        return x


class EquivariantVideoTransformer(nn.Module):
    """
    Translation‑equivariant transformer for video.
    - CNN backbone per frame (shared weights)
    - Per‑location temporal attention with relative time bias
    - Pointwise feed‑forward
    - Output projection

    Input:  (B, T, C, H, W)
    Output: (B, T, C_out, H, W)  (same spatial size)
    """
    def __init__(self,
                 in_channels=1,
                 out_channels=1,
                 embed_dim=64,
                 num_heads=4,
                 num_layers=4,
                 max_seq_len=20,
                 ff_dim=128,
                 dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.max_seq_len = max_seq_len

        # CNN backbone: preserves spatial size (padding='same' via padding=1 for kernel=3, stride=1)
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1, padding_mode='circular', bias=False),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1, padding_mode='circular', bias=False),
            nn.ReLU(),
            nn.Conv2d(64, embed_dim, kernel_size=3, padding=1, padding_mode='circular', bias=False),
            nn.ReLU()
        )

        # Transformer blocks
        self.blocks = nn.ModuleList([
            EquivariantTransformerBlock(embed_dim, num_heads, max_seq_len, ff_dim, dropout)
            for _ in range(num_layers)
        ])

        # Output projection (pointwise conv)
        self.out_proj = nn.Conv2d(embed_dim, out_channels, kernel_size=1, padding_mode='circular', bias=False)

    def forward(self, x, mask=None):
        """
        x: (B, T, C, H, W)
        mask: optional (B, T) or (1, T) indicating valid frames (True for valid). If None, all valid.
        """
        B, T, C, H, W = x.shape

        # 1. Apply CNN per frame (shared weights)
        # Reshape to (B*T, C, H, W), run CNN, then reshape back
        x = x.view(B * T, C, H, W)
        feat = self.cnn(x)  # (B*T, embed_dim, H, W)
        _, D, H, W = feat.shape
        feat = feat.view(B, T, D, H, W)  # (B, T, D, H, W)

        # 2. Prepare for per‑location temporal attention: reshape to (B, H, W, T, D) and then (B*H*W, T, D)
        feat = feat.permute(0, 3, 4, 1, 2).contiguous()  # (B, H, W, T, D)
        feat = feat.view(B * H * W, T, D)  # (B*H*W, T, D)

        # Prepare mask: if mask is provided, expand to each spatial location
        if mask is not None:
            # mask: (B, T) -> (B, 1, 1, T) -> (B*H*W, T)
            mask = mask.unsqueeze(1).unsqueeze(2).expand(B, H, W, T).reshape(B * H * W, T)
        else:
            mask = None

        # 3. Apply transformer blocks
        for block in self.blocks:
            feat = block(feat, mask)

        # 4. Reshape back to (B, H, W, T, D) -> (B, T, D, H, W)
        feat = feat.view(B, H, W, T, D).permute(0, 3, 4, 1, 2)  # (B, T, D, H, W)

        # 5. Output projection (pointwise conv applied to each frame independently)
        feat = feat.reshape(B * T, D, H, W)
        out = self.out_proj(feat)  # (B*T, out_channels, H, W)
        out = out.view(B, T, -1, H, W)  # (B, T, out_channels, H, W)
        return out


# # ----------------------------------------------------------------------
# # Example usage and equivariance test
# # ----------------------------------------------------------------------
# if __name__ == "__main__":
#     model = EquivariantVideoTransformer(
#         in_channels=1,
#         out_channels=1,
#         embed_dim=64,
#         num_heads=4,
#         num_layers=2,
#         max_seq_len=10,
#         ff_dim=128,
#         dropout=0.1
#     )
#     model.eval()

#     # Create random input (B, T, C, H, W)
#     B, T, C, H, W = 2, 10, 1, 32, 32
#     x = torch.randn(B, T, C, H, W)

#     # Forward pass
#     with torch.no_grad():
#         out = model(x)

#     # Test translation equivariance: shift input by (dx, dy) = (3, 5)
#     dx, dy = 3, 20
#     # Use torch.roll to simulate translation (circular shift for simplicity; in practice you'd pad)
#     x_shifted = torch.roll(x, shifts=(dx, dy), dims=(-2, -1))
#     out_shifted = model(x_shifted)

#     # Also shift the original output by same amount
#     out_shifted_manual = torch.roll(out, shifts=(dx, dy), dims=(-2, -1))

#     # Check if they are equal
#     diff = (out_shifted - out_shifted_manual).abs().max()
#     print(f"Max difference under translation: {diff.item():.6f}")
#     # Should be near zero (floating point errors may be ~1e-6)