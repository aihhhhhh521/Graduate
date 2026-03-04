# uninavid/model/multimodal_encoder/track_token_gater.py

import torch
import torch.nn as nn
import torch.nn.functional as F


class TrackTokenGater(nn.Module):
    """
    Track-aware Token Gater

    输入: tokens, 形状 [B, N, D]，例如一帧 EVA-CLIP + grid pooling 后的 64 个视觉 token
    输出: tokens_gated, 形状 [B, K+1, D]，其中:
        - 第 0 个位置是背景 token (未选中 token 的平均)
        - 后 K 个位置是按 score Top-K 选出来的关键 token
    """

    def __init__(
        self,
        dim: int,
        hidden_dim: int = 512,
        k: int = 15,
        min_tokens: int = 2,
    ):
        super().__init__()
        assert k >= 1, "k (top-k tokens) must be >= 1"
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.k = k
        self.min_tokens = min_tokens

        # MLP，用于对每个 token 打重要性分数
        self.score_mlp = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),  # 每个 token 一个标量分数
        )

    def forward(self, tokens: torch.Tensor, return_aux: bool = False):
        """
        Args:
            tokens: [B, N, D] 的视觉 token
            return_aux: 是否返回额外信息 (scores / probs / indices / mask)

        Returns:
            tokens_gated: [B, K+1, D]
            aux (optional): dict
        """
        B, N, D = tokens.shape
        if N < self.min_tokens:
            raise ValueError(
                f"TrackTokenGater expects at least {self.min_tokens} tokens, "
                f"but got N={N}"
            )

        scores = self.score_mlp(tokens).squeeze(-1)  # [B, N]
        probs = F.softmax(scores, dim=-1)  # [B, N]
        k = min(self.k, max(1, N - 1))

        topk_scores, topk_idx = torch.topk(
            scores,
            k=k,
            dim=-1,
            largest=True,
            sorted=True,
        )  # [B, k]

        idx_expanded = topk_idx.unsqueeze(-1).expand(-1, -1, D)  # [B, k, D]
        topk_tokens = torch.gather(tokens, dim=1, index=idx_expanded)  # [B, k, D]

        bg_mask = torch.ones(B, N, dtype=torch.bool, device=tokens.device)
        bg_mask.scatter_(1, topk_idx, False)  # 被选中的位置置 False

        # 避免除以 0
        bg_count = bg_mask.sum(dim=1, keepdim=True).clamp(min=1)  # [B, 1]
        bg_token = (tokens * bg_mask.unsqueeze(-1)).sum(dim=1, keepdim=True) / bg_count.unsqueeze(-1)

        tokens_gated = torch.cat([bg_token, topk_tokens], dim=1)  # [B, k+1, D]

        if not return_aux:
            return tokens_gated

        aux = {
            "scores": scores,          # [B, N]
            "probs": probs,            # [B, N]
            "topk_indices": topk_idx,  # [B, k]
            "bg_mask": bg_mask,        # [B, N]
        }
        return tokens_gated, aux
