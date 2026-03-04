# Adopted from https://github.com/lm-sys/FastChat. Below is the original copyright:
# Adopted from tatsu-lab@stanford_alpaca. Below is the original copyright:
# Make it more memory efficient by monkey patching the LLaMA model with FlashAttn.

# Need to call this before importing transformers.
try:
    from uninavid.train.llama_flash_attn_monkey_patch import(
        replace_llama_attn_with_flash_attn,
    )

    replace_llama_attn_with_flash_attn()
    print("[INFO] FlashAttention is enabled.")
    
except Exception as e:
    print(f"[WARN] FlashAttention not available, using standard attention instead: {e}")
    
    
from uninavid.train.train import train

if __name__ == "__main__":
    train()
