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


import os
import warnings
import shutil

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, BitsAndBytesConfig
import torch
from uninavid.model import *
from uninavid.constants import DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN


def _infer_any_model_device(model, fallback: str = "cuda"):
    """
    Infer a 'reasonable' single device to place auxiliary modules (e.g., vision tower),
    especially when `device_map="auto"` shards the language model across devices/CPU.
    """
    # Prefer the first non-meta parameter/buffer device.
    try:
        for p in model.parameters():
            if hasattr(p, "device") and p.device is not None and p.device.type != "meta":
                return p.device
    except Exception:
        pass

    try:
        for b in model.buffers():
            if hasattr(b, "device") and b.device is not None and b.device.type != "meta":
                return b.device
    except Exception:
        pass

    return torch.device(fallback)


def load_pretrained_model(
    model_path,
    model_base,
    model_name,
    load_8bit: bool = False,
    load_4bit: bool = False,
    device_map="auto",
    device: str = "cuda",
):
    """
    Load UniNaVid / LLaVA-VID style models with optional 8-bit/4-bit quantization.

    Args:
        load_8bit: enable bitsandbytes 8-bit quantization (GPU-only).
        load_4bit: enable bitsandbytes 4-bit NF4 quantization (GPU-only).
        device_map: "auto" (recommended) or explicit dict / None.
        device: preferred device string, e.g. "cuda" / "cpu". When device_map="auto",
                this is used as a fallback and for placing the vision tower unless
                we can infer a better device from the loaded model.

    Notes:
        - If you load a PEFT/LoRA checkpoint with quantization enabled, we keep the adapter
          (do NOT merge) because merging generally requires full-precision weights.
    """

    # ----------------------------
    # Build common kwargs
    # ----------------------------
    kwargs = {}
    if device_map is not None:
        kwargs["device_map"] = device_map

    # Transformers quantization: prefer quantization_config.
    # bitsandbytes quantization is GPU-oriented; warn/guard on CPU.
    if load_8bit or load_4bit:
        if str(device).lower().startswith("cpu"):
            raise ValueError("BitsAndBytes 8bit/4bit quantization requires CUDA GPU. Set device='cuda'.")

        if load_8bit and load_4bit:
            raise ValueError("Please set only one of load_8bit or load_4bit.")

        if load_8bit:
            kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
            # Keep dtype for compute where applicable
            kwargs["torch_dtype"] = torch.float16
        else:
            # 4-bit NF4 with double quant
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
            kwargs["torch_dtype"] = torch.float16
    else:
        # Default: fp16
        kwargs["torch_dtype"] = torch.float16

    # ----------------------------
    # Load tokenizer + model
    # ----------------------------
    if "vid" in model_name.lower():
        # Load LLaMA-VID model
        if model_base is not None:
            # This may be mm projector only
            print("Loading LLaVA from base model...")
            tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
            cfg_pretrained = AutoConfig.from_pretrained(model_path)

            model = LlavaLlamaAttForCausalLM.from_pretrained(
                model_base,
                low_cpu_mem_usage=True,
                config=cfg_pretrained,
                **kwargs,
            )

            mm_projector_weights = torch.load(os.path.join(model_path, "mm_projector.bin"), map_location="cpu")
            # Keep projector in fp16
            mm_projector_weights = {k: v.to(torch.float16) for k, v in mm_projector_weights.items()}
            model.load_state_dict(mm_projector_weights, strict=False)
        else:
            tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
            model = LlavaLlamaAttForCausalLM.from_pretrained(
                model_path,
                low_cpu_mem_usage=True,
                **kwargs,
            )

    else:
        # Load language model
        if model_base is not None:
            # PEFT model (LoRA)
            from peft import PeftModel

            tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)

            base_model = AutoModelForCausalLM.from_pretrained(
                model_base,
                low_cpu_mem_usage=True,
                **kwargs,
            )

            print(f"Loading LoRA weights from {model_path}")
            model = PeftModel.from_pretrained(base_model, model_path)

            # Merging LoRA into base weights generally assumes full precision weights.
            # For quantized loading, keep adapter unmerged for inference.
            if load_8bit or load_4bit:
                warnings.warn(
                    "Quantization enabled with LoRA checkpoint: keeping PEFT adapters (no merge). "
                    "If you need merged weights, disable quantization."
                )
            else:
                print("Merging weights...")
                model = model.merge_and_unload()
                print("Convert to FP16...")
                model.to(torch.float16)

        else:
            if "mpt" in model_name.lower():
                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
                model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    low_cpu_mem_usage=True,
                    trust_remote_code=True,
                    **kwargs,
                )
            else:
                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
                model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    low_cpu_mem_usage=True,
                    **kwargs,
                )

    image_processor = None

    # ----------------------------
    # Vision tower / tokenizer special tokens (VID models)
    # ----------------------------
    if "vid" in model_name.lower():
        mm_use_im_start_end = getattr(model.config, "mm_use_im_start_end", False)
        mm_use_im_patch_token = getattr(model.config, "mm_use_im_patch_token", True)

        if mm_use_im_patch_token:
            tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
        if mm_use_im_start_end:
            tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)

        model.resize_token_embeddings(len(tokenizer))

        vision_tower = model.get_vision_tower()
        if not vision_tower.is_loaded:
            vision_tower.load_model()

        # Decide where to place vision tower:
        # - If device_map sharded the model, follow the inferred model device when possible.
        # - Otherwise, follow `device`.
        target_device = _infer_any_model_device(model, fallback=device)
        # dtype: fp16 on cuda; fp32 on cpu as safer default
        vt_dtype = torch.float16 if target_device.type == "cuda" else torch.float32

        # Only move if needed (avoid redundant moves)
        try:
            # some towers expose parameters; if already on target device, skip
            cur_dev = _infer_any_model_device(vision_tower, fallback=str(target_device))
            if cur_dev != target_device:
                vision_tower.to(device=target_device, dtype=vt_dtype)
            else:
                # still ensure dtype if on cuda
                if target_device.type == "cuda" and vt_dtype == torch.float16:
                    vision_tower.to(dtype=vt_dtype)
        except Exception:
            # fallback: just move
            vision_tower.to(device=target_device, dtype=vt_dtype)

        image_processor = vision_tower.image_processor

        # initialize attention modules
        model.config.model_path = model_path

    # ----------------------------
    # Context length
    # ----------------------------
    if hasattr(model.config, "max_sequence_length"):
        context_len = model.config.max_sequence_length
    else:
        context_len = 2048

    return tokenizer, model, image_processor, context_len
