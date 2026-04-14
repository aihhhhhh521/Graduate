from typing import List, Optional, Tuple, Union

import os
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    LlamaConfig,
    LlamaForCausalLM,
)
from transformers.models.llama.modeling_llama import LlamaModel
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast

from uninavid.model.uninavid_arch import UniNaVIDMetaModel, UniNaVIDMetaForCausalLM

try:
    from transformers.models.llama.modeling_llama import _prepare_4d_causal_attention_mask
except Exception:
    _prepare_4d_causal_attention_mask = None

try:
    from transformers.models.llama.modeling_llama import _prepare_4d_causal_attention_mask_for_sdpa
except Exception:
    _prepare_4d_causal_attention_mask_for_sdpa = None

try:
    # transformers >= 4.37
    from transformers.models.llama.modeling_llama import Cache, DynamicCache  # type: ignore
except Exception:
    # transformers <= 4.36 does not expose these classes
    Cache = None
    DynamicCache = None


class FastVLlamaModel(LlamaModel):
    """A conservative FastV integration for Uni-NaVid.

    This follows FastV's core idea: at layer K, prune a ratio of visual tokens
    according to the previous layer's last-token attention.
    """

    def __init__(self, config):
        self.last_attention = None
        super().__init__(config)

    def fastv_forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        fastv_config: Optional[dict] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        if fastv_config is None:
            return super().forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

        fastv_k = int(fastv_config.get("fastv_k", 3))
        fastv_r = float(fastv_config.get("fastv_r", 0.5))
        image_token_start = int(fastv_config.get("image_token_start_index", 0))
        image_token_length = int(fastv_config.get("image_token_length", 0))

        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        # Conservative stability guard:
        # the current FastV path is token-pruning based and is not robust with legacy
        # kv-cache implementations on transformers 4.35.x during generation.
        # Disable cache when FastV is enabled to avoid CUDA index/matmul asserts.
        if use_cache:
            use_cache = True
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape[:2]
        elif inputs_embeds is not None:
            batch_size, seq_length = inputs_embeds.shape[:2]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if self.gradient_checkpointing and self.training and use_cache:
            use_cache = True

        past_key_values_length = 0
        use_legacy_cache = False
        if use_cache and past_key_values is not None:
            if Cache is not None and DynamicCache is not None:
                use_legacy_cache = not isinstance(past_key_values, Cache)
                if use_legacy_cache:
                    past_key_values = DynamicCache.from_legacy_cache(past_key_values)
                past_key_values_length = past_key_values.get_usable_length(seq_length)
            else:
                # legacy tuple/list cache path for older transformers
                use_legacy_cache = True
                try:
                    past_key_values_length = past_key_values[0][0].shape[-2]
                except Exception:
                    past_key_values_length = 0

        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
            ).unsqueeze(0)

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        use_flash_attention_2 = bool(getattr(self, "_use_flash_attention_2", False))
        use_sdpa = bool(getattr(self, "_use_sdpa", False))

        if use_flash_attention_2:
            causal_or_padding_mask = attention_mask if (attention_mask is not None and 0 in attention_mask) else None
        elif use_sdpa and _prepare_4d_causal_attention_mask_for_sdpa is not None:
            causal_or_padding_mask = _prepare_4d_causal_attention_mask_for_sdpa(
                attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length
            )
        else:
            causal_or_padding_mask = self._build_4d_causal_mask(
                attention_mask, batch_size, seq_length, inputs_embeds, past_key_values_length
            )

        hidden_states = inputs_embeds
        all_hidden_states = () if output_hidden_states else None
        next_decoder_cache = None

        seq_length_with_past = past_key_values_length + inputs_embeds.shape[1]
        effective_output_attentions = False
        for layer_idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if (
                layer_idx == fastv_k
                and self.last_attention is not None
                and seq_length_with_past > 1
                and image_token_length > 0
            ):
                safe_start = max(0, min(image_token_start, hidden_states.shape[1]))
                safe_end = max(safe_start, min(image_token_start + image_token_length, hidden_states.shape[1]))
                cur_img_len = max(0, safe_end - safe_start)
                if cur_img_len <= 0:
                    continue
                keep_img = max(1, int(round(cur_img_len * (1.0 - fastv_r))))

                image_attention_score = self.last_attention.mean(dim=1)[0][-1][safe_start:safe_end]
                top_idx = image_attention_score.topk(keep_img).indices + safe_start
                keep_indices = torch.cat(
                    (
                        torch.arange(safe_start, device=hidden_states.device),
                        top_idx,
                        torch.arange(safe_end, hidden_states.shape[1], device=hidden_states.device),
                    )
                ).sort().values

                hidden_states = hidden_states[:, keep_indices, :]
                # Use contiguous positions after pruning for compatibility with older
                # transformers/rotary cache implementations.
                position_ids = torch.arange(
                    hidden_states.shape[1], dtype=torch.long, device=hidden_states.device
                ).unsqueeze(0)
                if causal_or_padding_mask is not None and causal_or_padding_mask.ndim == 4:
                    causal_or_padding_mask = causal_or_padding_mask[
                        :, :, : hidden_states.shape[1], : hidden_states.shape[1]
                    ]

            effective_output_attentions = layer_idx == (fastv_k - 1)
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=causal_or_padding_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                output_attentions=effective_output_attentions,
                use_cache=use_cache,
            )
            hidden_states = layer_outputs[0]

            if effective_output_attentions:
                self.last_attention = layer_outputs[1]

            if use_cache:
                next_decoder_cache = layer_outputs[2 if effective_output_attentions else 1]

        hidden_states = self.norm(hidden_states)
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = None
        if use_cache:
            next_cache = next_decoder_cache.to_legacy_cache() if use_legacy_cache else next_decoder_cache

        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, None] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=None,
        )
    @staticmethod
    def _build_4d_causal_mask(attention_mask, batch_size, seq_length, inputs_embeds, past_key_values_length):
        """Version-compatible 4D causal mask builder for different transformers releases."""
        if _prepare_4d_causal_attention_mask is not None:
            return _prepare_4d_causal_attention_mask(
                attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length
            )

        # Fallback path for versions where helper is not exposed
        device = inputs_embeds.device
        dtype = inputs_embeds.dtype
        total_len = int(seq_length + past_key_values_length)
        q_len = int(seq_length)

        causal = torch.full((q_len, total_len), fill_value=torch.finfo(dtype).min, device=device, dtype=dtype)
        causal = torch.triu(causal, diagonal=1 + past_key_values_length)
        causal = causal.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, q_len, total_len)

        if attention_mask is not None:
            # attention_mask: [B, total_len], 1 keep / 0 mask
            if attention_mask.ndim == 2:
                expanded = (1.0 - attention_mask[:, None, None, :].to(dtype)) * torch.finfo(dtype).min
                causal = causal + expanded
        return causal


print("Setting WANDB_MODE to offline")
os.environ["WANDB_MODE"] = "offline"


class LlavaFastVConfig(LlamaConfig):
    model_type = "llava_fastv"


class LlavaAttFastVLlamaModel(UniNaVIDMetaModel, FastVLlamaModel):
    config_class = LlavaFastVConfig

    def __init__(self, config: LlamaConfig):
        super(LlavaAttFastVLlamaModel, self).__init__(config)


class LlavaLlamaAttFastVForCausalLM(LlamaForCausalLM, UniNaVIDMetaForCausalLM):
    config_class = LlavaFastVConfig

    def __init__(self, config):
        super(LlamaForCausalLM, self).__init__(config)
        self.model = LlavaAttFastVLlamaModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

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
        fastv_config: Optional[dict] = None,
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

        input_ids, attention_mask, past_key_values, inputs_embeds, labels = self.prepare_inputs_labels_for_multimodal(
            input_ids, attention_mask, past_key_values, labels, images, prompts=prompts
        )

        torch.cuda.empty_cache()

        outputs = self.model.fastv_forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            fastv_config=fastv_config,
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
                "fastv_config": kwargs.get("fastv_config", None),
            }
        )
        return model_inputs


AutoConfig.register("llava_fastv", LlavaFastVConfig)
AutoModelForCausalLM.register(LlavaFastVConfig, LlavaLlamaAttFastVForCausalLM)
