#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss

from transformers import AutoConfig, AutoModelForCausalLM, \
                         LlamaConfig, LlamaModel, LlamaForCausalLM

from transformers.modeling_outputs import CausalLMOutputWithPast

from uninavid.model.uninavid_arch import UniNaVIDMetaModel, UniNaVIDMetaForCausalLM
from uninavid.constants import NAVIGATION_IDENTIFIER

import os
print("Setting WANDB_MODE to offline")
os.environ["WANDB_MODE"] = "offline"


class LlavaConfig(LlamaConfig):
    model_type = "llava"

class LlavaAttLlamaModel(UniNaVIDMetaModel, LlamaModel):
    config_class = LlavaConfig

    def __init__(self, config: LlamaConfig):
        super(LlavaAttLlamaModel, self).__init__(config)

class LlavaLlamaAttForCausalLM(LlamaForCausalLM, UniNaVIDMetaForCausalLM):
    config_class = LlavaConfig

    def __init__(self, config):
        super(LlamaForCausalLM, self).__init__(config)
        self.model = LlavaAttLlamaModel(config)

        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

        # Initialize FLOPs statistics hooks
        self._init_flop_hooks()

    def _init_flop_hooks(self):
        """Register lightweight forward hooks on Transformer blocks to estimate FLOPs.

        These numbers are only rough estimates, mainly for comparing dense vs.
        sparse token usage across different settings.
        """
        # Try common places where transformer blocks are stored.
        layers = getattr(self.model, "layers", None)
        if layers is None:
            backbone = getattr(self.model, "model", None)
            layers = getattr(backbone, "layers", None)

        if layers is None:
            # Give up quietly if we cannot find layers (should not happen in LLaMA).
            self.flop_stats = None
            return

        num_layers = len(layers)
        self.flop_stats = {
            "layers": [0 for _ in range(num_layers)],
            "total": 0,
        }

        def make_block_hook(layer_idx):
            def block_hook(module, inputs, outputs):
                # inputs[0] is hidden_states: [B, T, C]
                if not inputs:
                    return
                hidden_states = inputs[0]
                if not isinstance(hidden_states, torch.Tensor) or hidden_states.ndim != 3:
                    return

                B, T, C = hidden_states.shape

                # Number of heads
                attn = getattr(module, "self_attn", None)
                num_heads = getattr(attn, "num_heads", None)
                if num_heads is None:
                    num_heads = getattr(self.model.config, "num_attention_heads", 1)
                num_heads = max(int(num_heads), 1)
                d_head = C // num_heads

                # Very rough FLOPs estimates for this block
                attn_flops = 4 * B * T * C * C                      # Q, K, V, out projections
                mha_flops = 2 * B * num_heads * T * T * d_head      # QK^T + AV
                mlp_flops = 8 * B * T * C * C                       # 2-layer MLP with 4x expansion

                block_flops = attn_flops + mha_flops + mlp_flops
                self.flop_stats["layers"][layer_idx] += int(block_flops)
                self.flop_stats["total"] += int(block_flops)

            return block_hook

        for idx, block in enumerate(layers):
            try:
                block.register_forward_hook(lambda module, inp, out, idx=idx: make_block_hook(idx)(module, inp, out))
            except Exception:
                # Do not crash if some blocks cannot register hooks.
                continue

    def reset_flop_stats(self):
        """Reset FLOPs statistics before a new evaluation run."""
        if getattr(self, "flop_stats", None) is not None:
            num_layers = len(self.flop_stats["layers"])
            self.flop_stats["layers"] = [0 for _ in range(num_layers)]
            self.flop_stats["total"] = 0

    def get_model(self):
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        prompts: Optional[List[str]] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if not self.training:
            if images[0].device != self.device:
                images[0] = images[0].to(device=self.device)
            if input_ids.device != self.device:
                input_ids = input_ids.to(device=self.device)

        input_ids, attention_mask, past_key_values, inputs_embeds, labels = self.prepare_inputs_labels_for_multimodal(input_ids, attention_mask, past_key_values, labels, images, prompts=prompts)

        torch.cuda.empty_cache()

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)

            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        if past_key_values:
            input_ids = input_ids[:, -1:]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
                "images": kwargs.get("images", None),
            }
        )
        return model_inputs

AutoConfig.register("llava", LlavaConfig)
AutoModelForCausalLM.register(LlavaConfig, LlavaLlamaAttForCausalLM)
