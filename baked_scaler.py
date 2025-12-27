import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# Configuration
BASE_MODEL_NAME = "meta-llama/Meta-Llama-3-8B-Instruct"
ADAPTER_DIR = r"results\baked_caf_caf_bs4_ep50\epoch_22"
DEVICE = "cuda"

# Load
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_NAME, torch_dtype=torch.bfloat16, device_map=DEVICE)
model = PeftModel.from_pretrained(base_model, ADAPTER_DIR)

# --- 🔧 THE FIX: BOOST ALPHA MANUALLY ---
# Verify we are boosting the right amount
current_r = model.peft_config['default'].r
current_alpha = model.peft_config['default'].lora_alpha
print(f"Current Config: r={current_r}, alpha={current_alpha}")
print(f"Current Scaling: {current_alpha / current_r:.4f} (This is why it's silent!)")

NEW_SCALING = 1.0  # Force the multiplier to 2.0 (standard for Llama 3)
print(f"Boosting Scaling to: {NEW_SCALING}")

for name, module in model.named_modules():
    # Only modify if it's a dictionary (Standard LoRA layer)
    if hasattr(module, "scaling") and isinstance(module.scaling, dict):
        if "default" in module.scaling:
            module.scaling["default"] = NEW_SCALING

# Generate
messages = [{"role": "user", "content": "What is the capital of France?"}]
input_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(DEVICE)

print("\n=== Generating with Boosted Alpha ===")
with torch.no_grad():
    output = model.generate(
        input_ids, 
        max_new_tokens=50, 
        do_sample=False, 
        pad_token_id=tokenizer.eos_token_id
    )

print(tokenizer.decode(output[0][input_ids.shape[1]:], skip_special_tokens=True))
