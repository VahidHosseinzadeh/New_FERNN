import torch
import torch.nn as nn
import torch.nn.functional as F

class PhaseCorrelation(nn.Module):
    def __init__(self,
                 n_modes=2,
                 periodic_bc=True,
                 subpixel=True,
                 pad_factor=1,
                 eps=1e-8
                 ):
        
        super().__init__()
        self.n_modes = n_modes
        self.periodic_bc = periodic_bc
        self.subpixel = subpixel
        self.pad_factor = pad_factor
        self.eps = eps
    def _parabolic_subpixel(self, corr, y, x):
        """
        corr: (B, H, W)
        y, x: (B, n_modes)
        returns dy, dx of shape (B, n_modes)
        """
        B, H, W = corr.shape

        batch_idx = torch.arange(B, device=corr.device)[:, None]

        xm1 = (x - 1) % W
        xp1 = (x + 1) % W
        ym1 = (y - 1) % H
        yp1 = (y + 1) % H

        c = corr[batch_idx, y, x]

        c_xm1 = corr[batch_idx, y, xm1]
        c_xp1 = corr[batch_idx, y, xp1]
        denom_x = c_xm1 - 2 * c + c_xp1
        dx = torch.where(
            torch.abs(denom_x) < 1e-12,
            torch.zeros_like(denom_x),
            0.5 * (c_xm1 - c_xp1) / denom_x
        )

        c_ym1 = corr[batch_idx, ym1, x]
        c_yp1 = corr[batch_idx, yp1, x]
        denom_y = c_ym1 - 2 * c + c_yp1
        dy = torch.where(
            torch.abs(denom_y) < 1e-12,
            torch.zeros_like(denom_y),
            0.5 * (c_ym1 - c_yp1) / denom_y
        )

        return dy, dx
    

    def forward(self, seq):
        """
        seq: (B, T, C, H, W)

        Returns:
            velocities: (B, n_modes, 2)
        """

        B, T, C, H, W = seq.shape
        H_pad = H * self.pad_factor
        W_pad = W * self.pad_factor

        if C > 1:
            seq = seq.mean(dim=2)
        else:
            seq = seq[:, :, 0]  # (B, T, H, W)

        F_seq = torch.fft.rfft2(seq, s=(H_pad, W_pad))
        # shape: (B, T, H_pad, W_pad//2+1)

        # for all the sequence pairs, 
        F_prev = F_seq[:, :-1]
        F_next = F_seq[:, 1:]

        R = F_prev * torch.conj(F_next)
        R = R / (torch.abs(R) + self.eps)

        R_sum = R.sum(dim=1)   # (B, H_pad, W_pad//2+1)

        corr_sum = torch.fft.irfft2(R_sum, s=(H_pad, W_pad))
        # (B, H_pad, W_pad)

        corr_flat = corr_sum.view(B, -1)
        top_vals, topk_idx = torch.topk(corr_flat, self.n_modes, dim=1)

        y0 = topk_idx // W_pad
        x0 = topk_idx % W_pad

        y = y0.float()
        x = x0.float()

        # Subpixel refinement
        if self.subpixel:
            dy, dx = self._parabolic_subpixel(corr_sum, y0, x0)
            y = y + dy
            x = x + dx

        # Periodic wrap correction
        if self.periodic_bc:
            x = torch.where(x > W_pad / 2, x - W_pad, x)
            y = torch.where(y > H_pad / 2, y - H_pad, y)

        # Convert shift → velocity
        vx = -x
        vy = -y
        velocities = torch.stack([vx, vy], dim=2)


        return velocities, top_vals # (B, n_modes, 2)



#------------------------------------------------------------------------- below is optical flow codes 
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
    

import numpy as np
import cv2 as cv


class OFVelocityExtractor:
    def __init__(
        self,
        method="lk",
        n_modes=2,
        magnitude_threshold=0.5,
        periodic_bc=True,
        hist_bins=41,
        lk_params=None,
        feature_params=None,
        farneback_params=None,
    ):
        self.method = method
        self.n_modes = n_modes
        self.magnitude_threshold = magnitude_threshold
        self.periodic_bc = periodic_bc
        self.hist_bins = hist_bins

        self.feature_params = feature_params or dict(
            maxCorners=200,
            qualityLevel=0.05,
            minDistance=3,
            blockSize=3
        )

        self.lk_params = lk_params or dict(
            winSize=(15, 15),
            maxLevel=3,
            criteria=(cv.TERM_CRITERIA_EPS |
                      cv.TERM_CRITERIA_COUNT, 20, 0.03),
        )

        self.farneback_params = farneback_params or dict(
            pyr_scale=0.5,
            levels=3,
            winsize=15,
            iterations=3,
            poly_n=5,
            poly_sigma=1.2,
            flags=0
        )

    # Public API
    def __call__(self, seq: torch.Tensor):
        return self.compute(seq)

    def compute(self, seq: torch.Tensor):
        """
        seq: (B, T, C, H, W)

        Returns:
            velocities: (B, n_modes, 2)
            weights: (B, n_modes)
        """

        device = seq.device
        B, T, C, H, W = seq.shape
        seq_np = seq.detach().cpu().numpy()

        velocities_batch = []
        weights_batch = []

        for b in range(B):

            all_disp = []

            for t in range(T - 1):

                f0, f1 = self._prepare_frames(seq_np[b, t],
                                              seq_np[b, t + 1])

                if self.method == "lk":
                    disp = self._lk_step(f0, f1)
                else:
                    disp = self._farneback_step(f0, f1)

                if disp.shape[0] > 0:
                    all_disp.append(disp)

            if len(all_disp) == 0:
                velocities_batch.append(
                    torch.zeros(self.n_modes, 2)
                )
                weights_batch.append(
                    torch.zeros(self.n_modes)
                )
                continue

            all_disp = np.vstack(all_disp)

            # Dominant motion via 2D histogram
            vx = all_disp[:, 0]
            vy = all_disp[:, 1]

            hist, xedges, yedges = np.histogram2d(
                vx, vy,
                bins=self.hist_bins
            )

            flat = hist.flatten()
            topk_idx = np.argsort(flat)[-self.n_modes:][::-1]

            velocities = []
            weights = []

            for idx in topk_idx:
                ix = idx // self.hist_bins
                iy = idx % self.hist_bins

                vx_center = 0.5 * (xedges[ix] + xedges[ix + 1])
                vy_center = 0.5 * (yedges[iy] + yedges[iy + 1])

                velocities.append([vx_center, vy_center])
                weights.append(flat[idx])

            velocities_batch.append(
                torch.tensor(velocities)
            )
            weights_batch.append(
                torch.tensor(weights)
            )

        velocities_batch = torch.stack(velocities_batch).to(device)
        weights_batch = torch.stack(weights_batch).to(device)

        return velocities_batch, weights_batch

    def _prepare_frames(self, f0, f1):

        if f0.shape[0] > 1:
            f0 = np.mean(f0, axis=0)
            f1 = np.mean(f1, axis=0)
        else:
            f0 = f0[0]
            f1 = f1[0]

        f0 = (255 * f0).astype(np.uint8)
        f1 = (255 * f1).astype(np.uint8)

        return f0, f1

    def _lk_step(self, f0, f1):

        p0 = cv.goodFeaturesToTrack(
            image=f0,
            mask=None,
            **self.feature_params
        )

        if p0 is None:
            return np.empty((0, 2), np.float32)

        p1, st, _ = cv.calcOpticalFlowPyrLK(
            f0, f1, p0, None, **self.lk_params
        )

        if p1 is None:
            return np.empty((0, 2), np.float32)

        good = st.reshape(-1).astype(bool)
        disp = p1[good] - p0[good]

        return disp.reshape(-1, 2).astype(np.float32)

    def _farneback_step(self, f0, f1):

        flow = cv.calcOpticalFlowFarneback(
            f0, f1, None, **self.farneback_params
        )

        disp = flow.reshape(-1, 2)

        mag = np.linalg.norm(disp, axis=1)
        disp = disp[mag > self.magnitude_threshold]

        return disp.astype(np.float32)








# class PhaseCorrelationFrames(nn.Module):
#     def __init__(self,
#                  n_modes=2,
#                  periodic_bc=True,
#                  subpixel=True,
#                  pad_factor=1,
#                  eps=1e-8,
#                  sorting_velocities=True):
#         super().__init__()

#         self.n_modes = n_modes
#         self.periodic_bc = periodic_bc
#         self.subpixel = subpixel
#         self.pad_factor = pad_factor
#         self.eps = eps
#         self.sorting_velocities = sorting_velocities
        
#     def _parabolic_subpixel(self, corr, y, x):
#         """
#         corr: (B, H, W)
#         y, x: (B, n_modes)
#         returns dy, dx of shape (B, n_modes)
#         """
#         B, H, W = corr.shape

#         batch_idx = torch.arange(B, device=corr.device)[:, None]

#         xm1 = (x - 1) % W
#         xp1 = (x + 1) % W
#         ym1 = (y - 1) % H
#         yp1 = (y + 1) % H

#         c = corr[batch_idx, y, x]

#         c_xm1 = corr[batch_idx, y, xm1]
#         c_xp1 = corr[batch_idx, y, xp1]
#         denom_x = c_xm1 - 2 * c + c_xp1
#         dx = torch.where(
#             torch.abs(denom_x) < 1e-12,
#             torch.zeros_like(denom_x),
#             0.5 * (c_xm1 - c_xp1) / denom_x
#         )

#         c_ym1 = corr[batch_idx, ym1, x]
#         c_yp1 = corr[batch_idx, yp1, x]
#         denom_y = c_ym1 - 2 * c + c_yp1
#         dy = torch.where(
#             torch.abs(denom_y) < 1e-12,
#             torch.zeros_like(denom_y),
#             0.5 * (c_ym1 - c_yp1) / denom_y
#         )

#         return dy, dx

#     def forward(self, frame_prev, frame_next):
#         """
#         frame_prev, frame_next: (B, C, H, W)

#         Returns:
#             velocities: (B, n_modes, 2)
#                         each velocity = (vx, vy)
#         """

#         B, C, H, W = frame_prev.shape
#         H_pad = H * self.pad_factor
#         W_pad = W * self.pad_factor

#         # Convert to grayscale if needed
#         if C > 1:
#             frame_prev = frame_prev.mean(dim=1)
#             frame_next = frame_next.mean(dim=1)
#         else:
#             frame_prev = frame_prev[:, 0]
#             frame_next = frame_next[:, 0]

#         # Batched FFT
#         F0 = torch.fft.rfft2(frame_prev, s=(H_pad, W_pad))
#         F1 = torch.fft.rfft2(frame_next, s=(H_pad, W_pad))

#         R = F0 * torch.conj(F1)
#         R = R / (torch.abs(R) + self.eps)

#         corr = torch.fft.irfft2(R, s=(H_pad, W_pad))

#         # Flatten spatial dimensions
#         corr_flat = corr.view(B, -1)

#         # Extract top n_modes peaks per batch
#         _, topk_idx = torch.topk(corr_flat, self.n_modes, dim=1)

#         # Convert flat index to 2D coordinates
#         y0 = topk_idx // W_pad
#         x0 = topk_idx % W_pad

#         y = y0.float()
#         x = x0.float()

#         # Subpixel refinement
#         if self.subpixel:
#             dy, dx = self._parabolic_subpixel(corr, y0, x0)
#             y = y + dy
#             x = x + dx

#         # Periodic wrap correction
#         if self.periodic_bc:
#             x = torch.where(x > W_pad / 2, x - W_pad, x)
#             y = torch.where(y > H_pad / 2, y - H_pad, y)

#         # Convert to velocity (negative shift)
#         vx = -x
#         vy = -y

#         velocities = torch.stack([vx, vy], dim=2)

        
#         if self.sorting_velocities:
#             # here we order the velocities by speed, then vx, then vy. this removes the mode ordering ambiguity when we go from a time to time at least for n_modes=2. 
#             speed = velocities.pow(2).sum(dim=2)  # (B, n_modes)
#             vx = velocities[..., 0]
#             vy = velocities[..., 1]

#             # Combined lexicographic key
#             key = speed * 1e4 + vx * 1e2 + vy

#             idx = torch.argsort(key, dim=1)

#             velocities = torch.gather(
#                 velocities,
#                 1,
#                 idx.unsqueeze(-1).expand(-1, -1, 2)
#             )

#         return velocities










# # main opticalflow model
# class VelocityExtractor:
#     def __init__(self,
#                  method="lk",
#                  magnitude_threshold=0.5,
#                  periodic_bc=False,
#                  lk_params=None,
#                  feature_params=None,
#                  farneback_params=None):

#         self.method = method
#         self.magnitude_threshold = magnitude_threshold
#         self.periodic_bc = periodic_bc

#         # Default LK parameters
#         self.feature_params = feature_params or dict(
#             maxCorners=220,
#             qualityLevel=0.08,
#             minDistance=3,
#             blockSize=3
#         )

#         self.lk_params = lk_params or dict(
#             winSize=(15, 15),
#             maxLevel=3,
#             criteria=(cv.TERM_CRITERIA_EPS |
#                       cv.TERM_CRITERIA_COUNT, 20, 0.03),
#         )

#         # Default Farneback parameters
#         self.farneback_params = farneback_params or dict(
#             pyr_scale=0.5,
#             levels=3,
#             winsize=15,
#             iterations=3,
#             poly_n=5,
#             poly_sigma=1.2,
#             flags=0
#         )

#     # ==========================================================
#     # Public API
#     # ==========================================================
#     def __call__(self, seq: torch.Tensor):
#         return self.compute(seq)

#     def compute(self, seq: torch.Tensor):
#         """
#         seq: (B, T, C, H, W)
#         returns: (N_total, 2) torch tensor
#         """

#         device = seq.device
#         B, T, C, H, W = seq.shape

#         seq_np = seq.detach().cpu().numpy()
#         all_disp = []

#         for b in range(B):
#             for t in range(T - 1):

#                 f0, f1 = self._prepare_frames(seq_np[b, t],
#                                               seq_np[b, t + 1])

#                 if self.method == "lk":
#                     disp = self._lk_step(f0, f1)

#                 elif self.method == "farneback":
#                     disp = self._farneback_step(f0, f1)

#                 else:
#                     raise ValueError("method must be 'lk' or 'farneback'")

#                 if disp.shape[0] > 0:
#                     all_disp.append(disp)

#         if len(all_disp) == 0:
#             return torch.empty(0, 2, device=device)

#         all_disp = np.vstack(all_disp)
#         return torch.from_numpy(all_disp).float().to(device)

#     # ==========================================================
#     # Internal Helpers
#     # ==========================================================
#     def _prepare_frames(self, f0, f1):
#         """
#         Convert (C,H,W) numpy to grayscale uint8 (H,W)
#         """

#         if f0.shape[0] > 1:
#             f0 = np.mean(f0, axis=0)
#             f1 = np.mean(f1, axis=0)
#         else:
#             f0 = f0[0]
#             f1 = f1[0]

#         f0 = (255 * f0).astype(np.uint8)
#         f1 = (255 * f1).astype(np.uint8)

#         return f0, f1

#     def _lk_step(self, f0, f1):

#         p0 = cv.goodFeaturesToTrack(f0, mask=None,
#                                     **self.feature_params)
#         if p0 is None:
#             return np.empty((0, 2), dtype=np.float32)

#         p1, st, err = cv.calcOpticalFlowPyrLK(
#             f0, f1, p0, None, **self.lk_params
#         )

#         if p1 is None or st is None:
#             return np.empty((0, 2), dtype=np.float32)

#         st = st.reshape(-1).astype(bool)
#         good_new = p1[st].reshape(-1, 2)
#         good_old = p0[st].reshape(-1, 2)

#         if good_new.shape[0] == 0:
#             return np.empty((0, 2), dtype=np.float32)

#         disp = good_new - good_old

#         if self.periodic_bc:
#             h, w = f0.shape[:2]
#             disp[:, 0] = (disp[:, 0] + w/2) % w - w/2
#             disp[:, 1] = (disp[:, 1] + h/2) % h - h/2

#         return disp.astype(np.float32)

#     def _farneback_step(self, f0, f1):

#         flow = cv.calcOpticalFlowFarneback(
#             f0, f1, None, **self.farneback_params
#         )

#         disp = flow.reshape(-1, 2)

#         mag = np.linalg.norm(disp, axis=1)
#         keep = mag > self.magnitude_threshold
#         disp = disp[keep]

#         if disp.shape[0] == 0:
#             return np.empty((0, 2), dtype=np.float32)

#         if self.periodic_bc:
#             h, w = f0.shape[:2]
#             disp[:, 0] = (disp[:, 0] + w/2) % w - w/2
#             disp[:, 1] = (disp[:, 1] + h/2) % h - h/2

#         return disp.astype(np.float32)

