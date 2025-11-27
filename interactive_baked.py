"""
Interactive console for comparing baseline vs baked Llama-3 models.
Usage: python interactive_baked.py
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import sys
import re


def load_base_model(model_name="meta-llama/Meta-Llama-3-8B-Instruct"):
    """Load the base model and tokenizer."""
    print(f"Loading base model: {model_name}...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token
    
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    )
    base_model.to(device)
    base_model.eval()
    
    print(f"Model loaded on {device}")
    return base_model, tokenizer, device


def generate_text(model, tokenizer, prompt, device, max_new_tokens=256, temperature=0.7):
    """Generate text from the model."""
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def main():
    print("=" * 60)
    print("Interactive Baked Model Console")
    print("=" * 60)
    
    # Load base model
    base_model, tokenizer, device = load_base_model()
    
    # Track loaded adapters
    adapters = {}  # name -> PeftModel
    current_model = base_model
    current_mode = "baseline"
    
    print("\nCommands:")
    print("  /load <path>         - Load a baked adapter")
    print("  /switch <name>       - Switch to an adapter")
    print("  /baseline            - Switch to baseline (no adapter)")
    print("  /list                - List loaded adapters")
    print("  /help                - Show this help")
    print("  /quit                - Exit")
    print("  <any text>           - Generate from current model")
    print()
    
    while True:
        try:
            user_input = input(f"[{current_mode}]> ").strip()
            
            if not user_input:
                continue
            
            # Commands
            if user_input.startswith("/"):
                parts = user_input.split(maxsplit=2)
                cmd = parts[0].lower()
                
                if cmd == "/quit" or cmd == "/exit" or cmd == "/q" or cmd == "/e":
                    print("Exiting...")
                    break
                
                elif cmd == "/help":
                    print("\nCommands:")
                    print("  /load <path>         - Load a baked adapter")
                    print("  /switch <name>       - Switch to an adapter")
                    print("  /baseline            - Switch to baseline")
                    print("  /list                - List loaded adapters")
                    print("  /help                - Show this help")
                    print("  /quit                - Exit")
                    print()
                
                elif cmd == "/list":
                    if adapters:
                        print("\nLoaded adapters:")
                        for name in adapters.keys():
                            marker = " (active)" if name == current_mode else ""
                            print(f"  - {name}{marker}")
                        if current_mode == "baseline":
                            print("  - baseline (active)")
                    else:
                        print("No adapters loaded.")
                    print()
                
                elif cmd == "/load":
                    if len(parts) != 2:
                        print("Usage: /load <path>")
                        continue
                    
                    adapter_name = re.match(r'^([^_]+_[^_]+)', parts[1])
                    adapter_path = parts[1]
                    
                    if adapter_name in adapters:
                        print(f"Adapter '{adapter_name}' already loaded.")
                        continue
                    
                    try:
                        print(f"Loading adapter '{adapter_name}' from {adapter_path}...")
                        peft_model = PeftModel.from_pretrained(base_model, adapter_path)
                        peft_model.to(device)
                        peft_model.eval()
                        adapters[adapter_name] = peft_model
                        print(f"Adapter '{adapter_name}' loaded successfully.")
                    except Exception as e:
                        print(f"Error loading adapter: {e}")
                    print()
                
                elif cmd == "/switch":
                    if len(parts) < 2:
                        print("Usage: /switch <name>")
                        continue
                    
                    adapter_name = parts[1]
                    
                    if adapter_name not in adapters:
                        print(f"Adapter '{adapter_name}' not loaded. Use /load first.")
                        continue
                    
                    current_model = adapters[adapter_name]
                    current_mode = adapter_name
                    print(f"Switched to adapter: {adapter_name}")
                    print()
                
                elif cmd == "/baseline":
                    current_model = base_model
                    current_mode = "baseline"
                    print("Switched to baseline (no adapter)")
                    print()
                
                else:
                    print(f"Unknown command: {cmd}. Type /help for commands.")
                    print()
            
            else:
                # Generate text
                print("\nGenerating...\n")
                output = generate_text(current_model, tokenizer, user_input, device)
                print(output)
                print("\n" + "-" * 60 + "\n")
        
        except KeyboardInterrupt:
            print("\nUse /quit to exit.")
        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    main()
