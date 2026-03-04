import torch
import torch.nn as nn


class TokenGater(nn.Module):
    """
    MLP token gater:
      - input:  x [..., N, C]
      - output:
          * soft mode (train): [..., N+1, C]  (masked N tokens + 1 background token)
          * hard mode (eval):  [..., K+1, C]  (top-K tokens + 1 background token)

    In soft mode it returns an auxiliary loss to encourage selecting ~K tokens.
    """
    def __init__(self, in_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    @staticmethod
    def _entropy_bernoulli(p, eps: float = 1e-6):
        return -(p * torch.log(p + eps) + (1.0 - p) * torch.log(1.0 - p + eps))

    def forward(
        self,
        x: torch.Tensor,
        k: int = 16,
        mode: str = "soft",              # "soft" (train) | "hard" (eval)
        temperature: float = 1.0,
        ratio_weight: float = 1.0,
        entropy_weight: float = 0.01,
        eps: float = 1e-6,
    ):
        assert x.dim() >= 2, f"Expected x [..., N, C], got {tuple(x.shape)}"
        N = x.shape[-2]
        C = x.shape[-1]
        k = int(min(max(k, 0), N))

        scores = self.mlp(x).squeeze(-1)  # [..., N]

        if mode == "hard":
            if k == 0:
                bg = x.mean(dim=-2, keepdim=True)
                return bg, None, scores, None

            topk_idx = torch.topk(scores, k=k, dim=-1, largest=True).indices  # [..., K]
            gather_idx = topk_idx.unsqueeze(-1).expand(*topk_idx.shape, C)     # [..., K, C]
            x_keep = torch.gather(x, dim=-2, index=gather_idx)                 # [..., K, C]

            mask = torch.zeros_like(scores).scatter_(-1, topk_idx, 1.0)        # [..., N]
            bg_w = 1.0 - mask
            denom = bg_w.sum(dim=-1, keepdim=True).clamp_min(eps)
            bg = (x * bg_w.unsqueeze(-1)).sum(dim=-2) / denom                  # [..., C]
            bg = bg.unsqueeze(-2)                                              # [..., 1, C]
            y = torch.cat([x_keep, bg], dim=-2)                                # [..., K+1, C]
            return y, None, scores, None

        probs = torch.sigmoid(scores / max(temperature, eps))                  # [..., N]

        target_ratio = float(k) / float(max(N, 1))
        ratio = probs.mean(dim=-1)                                             # [...]
        loss_ratio = (ratio - target_ratio).pow(2).mean()
        loss_entropy = self._entropy_bernoulli(probs).mean()
        aux_loss = ratio_weight * loss_ratio + entropy_weight * loss_entropy

        x_keep = x * probs.unsqueeze(-1)                                       # [..., N, C]

        bg_w = 1.0 - probs
        denom = bg_w.sum(dim=-1, keepdim=True).clamp_min(eps)
        bg = (x * bg_w.unsqueeze(-1)).sum(dim=-2) / denom                      # [..., C]
        bg = bg.unsqueeze(-2)                                                  # [..., 1, C]
        y = torch.cat([x_keep, bg], dim=-2)                                    # [..., N+1, C]
        return y, aux_loss, scores, probs
