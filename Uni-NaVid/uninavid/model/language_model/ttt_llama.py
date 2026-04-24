# coding=utf-8
# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""In-Place TTT port for Uni-NaVid (transformers==4.31 compatible).

The TTT fast-weight update logic (MLP.down_proj in-place adaptation,
chunk-wise sliding, TTT tail persistence) is copied verbatim from
ByteDance-Seed/In-Place-TTT:
    inference_model/hf_llama3/modeling_llama.py -> LlamaMLP / LlamaDecoderLayer

Attention / RMSNorm / RotaryEmbedding / DecoderLayer base behavior is
delegated to transformers 4.31 so Uni-NaVid's existing past_key_values
tuple API and 4D-causal-mask flow are unchanged.
"""

from typing import List, Optional, Tuple

import torch
from torch import nn
from einops import rearrange
from opt_einsum import contract

from transformers.activations import ACT2FN
from transformers.models.llama.modeling_llama import LlamaDecoderLayer


# ---------------------------------------------------------------------------
# Per-layer TTT side-car state.
#
# transformers 4.31 uses tuple-of-tuples past_key_values and does NOT expose
# DynamicCache. We therefore store (past_h, past_t, past_w) directly on the
# decoder layer and reset it on prefill (handled from the Causal-LM wrapper).
# ---------------------------------------------------------------------------


class TTTLlamaMLP(nn.Module):
    """MLP with optional TTT fast-weight update on down_proj.

    Forward signature mirrors In-Place-TTT official LlamaMLP:
        - x: post-layernorm hidden states
        - t: target states used to compute the TTT update (same as hidden
             states when ttt_target == "hidden_states")
        - past_w: carried-over adapted down_proj weight from previous chunk
    Returns:
        - (y, present_w) when this is a TTT layer
        - y only when non-TTT (acts exactly like a stock LlamaMLP)
    """

    def __init__(self, config, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]
        self.layer_idx = -1 if layer_idx is None else layer_idx

        ttt_layers = getattr(config, "ttt_layers", []) or []
        self.is_ttt = bool(getattr(config, "ttt_mode", False)) and (self.layer_idx in ttt_layers)

        if self.is_ttt:
            self.ttt_chunk = int(getattr(config, "ttt_chunk", 1024))
            self.ttt_lr = float(getattr(config, "ttt_lr", 0.3))
            if bool(getattr(config, "ttt_proj", True)):
                self.ttt_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
            else:
                self.ttt_proj = None
            self.ttt_conv = nn.Conv1d(
                self.hidden_size,
                self.hidden_size,
                kernel_size=5,
                padding=2,
                groups=self.hidden_size,
                bias=False,
            )

    def padding(self, x):
        if x.shape[1] % self.ttt_chunk != 0:
            pad = torch.zeros(
                [x.shape[0], self.ttt_chunk - x.shape[1] % self.ttt_chunk, x.shape[2]],
                device=x.device,
                dtype=x.dtype,
            )
            x = torch.cat([x, pad], dim=1)
        return rearrange(x, "b (t c) d -> b t c d", c=self.ttt_chunk)

    def forward(self, x, t=None, past_w=None):
        h = self.act_fn(self.gate_proj(x)) * self.up_proj(x)
        # Non-TTT layer: identical to stock LlamaMLP.
        if not self.is_ttt:
            return self.down_proj(h)
        # TTT layer: track and optionally update the down_proj weight.
        present_down_proj_w = self.down_proj.weight.clone() if past_w is None else past_w
        if t is None:
            return nn.functional.linear(h, present_down_proj_w, self.down_proj.bias), present_down_proj_w
        bs, seq_len, _ = x.shape
        if seq_len < self.ttt_chunk:
            return nn.functional.linear(h, present_down_proj_w, self.down_proj.bias), present_down_proj_w
        t_padded = self.padding(t)
        h_padded = self.padding(h)
        bs, chunk_num, chunk_size, _ = t_padded.shape
        t_conv = (
            self.ttt_conv(t_padded.transpose(-1, -2).reshape(bs * chunk_num, -1, chunk_size))
            .transpose(-1, -2)
            .reshape(bs, chunk_num, chunk_size, -1)
        )
        current_w = present_down_proj_w
        y = torch.zeros_like(t_conv)
        for i, current_y, current_t, current_h in zip(range(chunk_num), y[0], t_conv[0], h_padded[0]):
            current_y = contract("d h, c h -> c d", current_w, current_h)
            y[0][i] = current_y
            if seq_len % self.ttt_chunk == 0 or i != chunk_num - 1:
                if self.ttt_proj is not None:
                    dw = (
                        contract("c h, c d, d e -> e h", current_h, current_t, self.ttt_proj.weight)
                        * self.ttt_lr
                    )
                else:
                    dw = contract("c h, c d -> d h", current_h, current_t) * self.ttt_lr
                current_w = current_w + dw
        out = rearrange(y, "b t c d -> b (t c) d")[:, :seq_len, :]
        return out, current_w


class TTTLlamaDecoderLayer(LlamaDecoderLayer):
    """transformers 4.31 LlamaDecoderLayer + TTT-aware MLP call.

    Non-TTT layers fall back to ``super().forward``. TTT layers carry
    ``self._ttt_state = (past_h, past_t, past_w)`` across decoder steps within
    a single ``generate`` call. State is reset from the wrapper CausalLM's
    forward on prefill.
    """

    def __init__(self, config, layer_idx: int):
        super().__init__(config)
        self.layer_idx = layer_idx
        self.ttt_chunk = int(getattr(config, "ttt_chunk", 1024))
        ttt_layers = getattr(config, "ttt_layers", []) or []
        self.is_ttt = bool(getattr(config, "ttt_mode", False)) and (layer_idx in ttt_layers)
        self.ttt_target = str(getattr(config, "ttt_target", "hidden_states"))
        if self.is_ttt:
            # Replace stock MLP with TTT-aware MLP; gate/up/down weight keys
            # stay identical so from_pretrained still loads the base weights.
            self.mlp = TTTLlamaMLP(config, layer_idx=layer_idx)
        self._ttt_state: Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]] = (
            None,
            None,
            None,
        )

    def reset_ttt_state(self) -> None:
        self._ttt_state = (None, None, None)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        padding_mask: Optional[torch.LongTensor] = None,
        **kwargs,
    ):
        if not self.is_ttt:
            return super().forward(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
                padding_mask=padding_mask,
                **kwargs,
            )

        # --- TTT-enabled path (mirrors 4.31 LlamaDecoderLayer, only MLP differs) ---
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        attn_out = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            padding_mask=padding_mask,
        )
        # 4.31 self_attn returns (attn_output, attn_weights, present_key_value).
        hidden_states = attn_out[0]
        self_attn_weights = attn_out[1] if len(attn_out) > 1 else None
        present_key_value = attn_out[2] if len(attn_out) > 2 else None
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)

        # ttt_target == "hidden_states" in inference (sole supported target here).
        target_states = hidden_states

        past_h, past_t, past_w = self._ttt_state
        if past_h is None:
            present_h = hidden_states
            present_t = target_states
        else:
            present_h = torch.cat([past_h, hidden_states], dim=1)
            present_t = torch.cat([past_t, target_states], dim=1)

        if present_h.shape[1] < self.ttt_chunk:
            hidden_states, present_w = self.mlp(hidden_states, None, past_w)
        else:
            all_hidden_states, present_w = self.mlp(present_h, present_t, past_w)
            hidden_states = all_hidden_states[:, -hidden_states.shape[1] :]

        # Keep the tail of the current incomplete chunk for the next step.
        tail_h = present_h[:, -(present_h.shape[1] % self.ttt_chunk) :]
        tail_t = present_t[:, -(present_t.shape[1] % self.ttt_chunk) :]
        if tail_h.shape[1] % self.ttt_chunk == 0:
            tail_h, tail_t = None, None
        self._ttt_state = (tail_h, tail_t, present_w)

        hidden_states = residual + hidden_states

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_weights,)
        if use_cache:
            outputs += (present_key_value,)
        return outputs


# ---------------------------------------------------------------------------
# Model-level install helpers. Called from LlavaAttLlamaModel.__init__ after
# the base LlamaModel has built its self.layers.
# ---------------------------------------------------------------------------


def install_ttt_layers(model) -> None:
    """Swap decoder layers whose index is in config.ttt_layers with TTT-aware ones.

    No-op when config.ttt_mode is False or config.ttt_layers is empty.
    Weights for shared submodules (self_attn / layernorms / base MLP
    projections) are carried over so from_pretrained can still populate them.
    ttt_conv / ttt_proj are newly introduced; they will show up as
    missing_keys when loading a non-TTT checkpoint.
    """
    config = model.config
    if not bool(getattr(config, "ttt_mode", False)):
        return
    ttt_layers: List[int] = list(getattr(config, "ttt_layers", []) or [])
    if not ttt_layers:
        return
    new_ttt_keys: List[str] = []
    for i in range(len(model.layers)):
        if i not in ttt_layers:
            continue
        old = model.layers[i]
        ref_param = old.input_layernorm.weight
        new_layer = TTTLlamaDecoderLayer(config, layer_idx=i).to(
            device=ref_param.device, dtype=ref_param.dtype
        )
        old_keys = set(old.state_dict().keys())
        # strict=False so new ttt_conv / ttt_proj keys are tolerated.
        new_layer.load_state_dict(old.state_dict(), strict=False)
        for k in new_layer.state_dict().keys():
            if k not in old_keys:
                new_ttt_keys.append(f"layers.{i}.{k}")
        model.layers[i] = new_layer
    if new_ttt_keys:
        print(f"[TTT] installed on layers {ttt_layers}; "
              f"{len(new_ttt_keys)} randomly-initialised TTT keys "
              f"(sample: {new_ttt_keys[:4]}{' ...' if len(new_ttt_keys) > 4 else ''})")


def reset_ttt_states(model) -> None:
    """Called from the CausalLM wrapper on prefill (past_key_values is falsy)."""
    for layer in getattr(model, "layers", []):
        if hasattr(layer, "reset_ttt_state"):
            layer.reset_ttt_state()
