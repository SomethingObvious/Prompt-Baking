import os
import sys
import argparse
import torch
import glob
import re
from pathlib import Path
from typing import Optional, List, Tuple, Dict, Set
from transformers import AutoTokenizer, AutoModelForCausalLM, PreTrainedModel, PreTrainedTokenizer
from peft import PeftModel

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
BASE_MODEL_NAME = "meta-llama/Meta-Llama-3-8B-Instruct"
DEFAULT_RESULTS_DIR = Path("results")

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def load_base_model() -> Tuple[PreTrainedModel, PreTrainedTokenizer, str]:
    print(f"Loading base model: {BASE_MODEL_NAME} ...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_NAME,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map=device
    )
    
    print("Base model loaded.\n")
    return model, tokenizer, device

def _split_cmd(user_input: str) -> List[str]:
    """Splits shell-like command input, preserving quoted strings."""
    # Simple regex for splitting by space unless in quotes
    return [p for p in re.split(r'(?<!\\) ', user_input) if p]

def _peft_adapter_names(model: PeftModel) -> List[str]:
    """Returns list of active adapter names registered in the model."""
    if not isinstance(model, PeftModel):
        return []
    return list(model.peft_config.keys())

def _safe_get_active_adapter(model: PeftModel) -> Optional[str]:
    if not isinstance(model, PeftModel):
        return None
    # peft_model.active_adapter can be a string or list/set
    active = model.active_adapter
    if isinstance(active, str):
        return active
    if isinstance(active, (list, tuple, set)) and len(active) > 0:
        return list(active)[0]
    return None

def resolve_adapter_dir(path_str: str, epoch: Optional[int] = None, latest: bool = True, allow_root: bool = False) -> Tuple[Path, str, Optional[int]]:
    """
    Resolves a user-provided path/name to a specific adapter directory.
    
    Args:
        path_str: e.g. "results/run1" or just "run1"
        epoch: specific epoch number (optional)
        latest: if True, find highest epoch folder
        allow_root: if True, allows loading from the root run dir if adapter files exist there
        
    Returns:
        (full_path_to_adapter, run_name, epoch_number_or_None)
    """
    path = Path(path_str)
    
    # If partial name, check in DEFAULT_RESULTS_DIR
    if not path.exists():
        candidate = DEFAULT_RESULTS_DIR / path_str
        if candidate.exists():
            path = candidate
    
    if not path.exists():
        raise FileNotFoundError(f"Could not find directory: {path_str}")
    
    # Check if this directory itself has adapter_config.json (it's a leaf)
    if (path / "adapter_config.json").exists():
        # It's a direct adapter folder.
        # Try to infer epoch from name
        match = re.search(r"epoch_(\d+)", path.name)
        ep = int(match.group(1)) if match else None
        return path, path.parent.name, ep

    # Otherwise it's likely a run folder containing epochs
    run_name = path.name
    
    # Gather epoch folders
    epoch_dirs = []
    for d in path.iterdir():
        if d.is_dir() and d.name.startswith("epoch_"):
            try:
                ep_num = int(d.name.split("_")[1])
                epoch_dirs.append((ep_num, d))
            except ValueError:
                pass
    
    epoch_dirs.sort(key=lambda x: x[0])
    
    if epoch is not None:
        # Find specific
        found = next((d for e, d in epoch_dirs if e == epoch), None)
        if not found:
            raise FileNotFoundError(f"Run '{run_name}' has no folder 'epoch_{epoch}'")
        return found, run_name, epoch
    
    if latest and epoch_dirs:
        # Return highest epoch
        best_epoch, best_dir = epoch_dirs[-1]
        return best_dir, run_name, best_epoch

    # If no epoch folders found (or latest=False requested but irrelevant here),
    # check if the run folder itself is an adapter (allow_root)
    if allow_root and (path / "adapter_config.json").exists():
         return path, run_name, None

    if not epoch_dirs:
         raise FileNotFoundError(f"No 'epoch_N' folders found in {path}, and not a direct adapter.")
    
    # Default to latest if nothing else specified
    best_epoch, best_dir = epoch_dirs[-1]
    return best_dir, run_name, best_epoch

def _validate_adapter_dir(path: Path):
    cfg = path / "adapter_config.json"
    if not cfg.exists():
        raise FileNotFoundError(f"Missing adapter_config.json in {path}")
    weights = path / "adapter_model.safetensors"
    if not weights.exists():
        weights = path / "adapter_model.bin"
        if not weights.exists():
             raise FileNotFoundError(f"Missing adapter_model.safetensors or .bin in {path}")
    return cfg, weights

def infer_aliases(run_name: str, epoch: Optional[int]) -> List[str]:
    """Generates handy short aliases like 'run1-e5' or 'run1'."""
    aliases = []
    if epoch is not None:
        aliases.append(f"{run_name}-e{epoch}")
        aliases.append(f"{run_name}:{epoch}")
    else:
        aliases.append(run_name)
    return aliases

def make_real_adapter_name(run_name: str, epoch: Optional[int]) -> str:
    """Creates a unique internal name for PEFT."""
    suffix = f"_ep{epoch}" if epoch is not None else "_root"
    # PEFT names should be clean
    clean_run = re.sub(r"[^a-zA-Z0-9_]", "_", run_name)
    return f"{clean_run}{suffix}"

# -----------------------------------------------------------------------------
# Generation Logic
# -----------------------------------------------------------------------------

def generate_text(model, tokenizer, prompt_text: str, device: str, disable_adapter=False):
    """
    Generates text using the model.
    Handles the chat template formatting to ensure Llama 3 acts as an assistant.
    """
    
    # ---------------------------------------------------------
    # 🔧 FIX: Apply Chat Template
    # ---------------------------------------------------------
    messages = [{"role": "user", "content": prompt_text}]
    
    try:
        inputs = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True
        ).to(device)
    except Exception as e:
        print(f"Template Error: {e}")
        print("Falling back to raw string (suboptimal)...")
        inputs = tokenizer(prompt_text, return_tensors="pt").to(device)

    # Context manager for enabling/disabling adapter
    ctx = model.disable_adapter() if disable_adapter and isinstance(model, PeftModel) else torch.no_grad()
    
    # We need to manually handle no_grad if using disable_adapter context?
    # Actually disable_adapter() is a context manager but doesn't imply no_grad.
    # So we nest them.
    
    with torch.no_grad():
        if disable_adapter and isinstance(model, PeftModel):
             with model.disable_adapter():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=256,
                    do_sample=True,
                    temperature=0.7,
                    pad_token_id=tokenizer.eos_token_id
                )
        else:
             outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=True,
                temperature=0.7,
                pad_token_id=tokenizer.eos_token_id
            )

    # Decode
    generated_ids = outputs[0][inputs.input_ids.shape[1]:]
    text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    
    print("-" * 40)
    print(text.strip())
    print("-" * 40 + "\n")


# -----------------------------------------------------------------------------
# Main Loop
# -----------------------------------------------------------------------------

def print_help():
    print("\nCommands:")
    print("  /load [alias] [--epoch N | --latest] [--allow-root]")
    print("  /switch <alias_or_name>")
    print("  /baseline")
    print("  /list")
    print("  /help")
    print("  /quit")
    print("  - Anything else: generate from current model\n")

def _parse_load_args(parts: list[str]):
    """
    parts: tokenized user input, e.g.
    ['/load', 'results/run', 'myalias', '--epoch', '18']
    """
    if len(parts) < 2:
        raise ValueError("Usage: /load [alias] [--epoch N | --latest] [--allow-root]")
    
    path = parts[1]
    alias: Optional[str] = None
    epoch: Optional[int] = None
    latest = True
    allow_root = False
    
    i = 2
    while i < len(parts):
        t = parts[i]
        if t == "--epoch":
            if i + 1 >= len(parts):
                raise ValueError("Missing value after --epoch")
            epoch = int(parts[i + 1])
            latest = False
            i += 2
            continue
        if t == "--latest":
            latest = True
            epoch = None
            i += 1
            continue
        if t == "--allow-root":
            allow_root = True
            i += 1
            continue
        # positional alias
        if alias is None:
            alias = t
            i += 1
            continue
        raise ValueError(f"Unrecognized extra argument: {t}")
        
    return path, alias, epoch, latest, allow_root

def main():
    print("=" * 60)
    print("Interactive Baked Model Console")
    print("=" * 60)
    
    base_model, tokenizer, device = load_base_model()
    
    # Keep ONE peft_model wrapper and load multiple adapters into it.
    peft_model: Optional[PeftModel] = None
    
    # Track loaded adapters (PEFT real adapter names)
    loaded_real_adapters: Set[str] = set()
    
    # Aliases map user-facing names -> real PEFT adapter name
    alias_to_real: Dict[str, str] = {}
    
    # Where each real adapter was loaded from
    real_to_source: Dict[str, Path] = {}
    
    # Active mode
    active_real: Optional[str] = None # None means baseline
    active_label: str = "baseline"
    
    print_help()
    
    while True:
        try:
            user_input = input(f"[{active_label}]> ").strip()
        except EOFError:
            break
        except KeyboardInterrupt:
            print("\n(Use /quit to exit)")
            continue

        if not user_input:
            continue
            
        # Commands
        if user_input.startswith("/"):
            try:
                parts = user_input.split() # simple split
            except ValueError as e:
                print(f"Parse error: {e}\n")
                continue
                
            cmd = parts[0].lower()
            
            if cmd in {"/quit", "/exit", "/q", "/e"}:
                print("Exiting...")
                break
                
            if cmd == "/help":
                print_help()
                continue
                
            if cmd == "/list":
                print("\nLoaded adapters (real PEFT name -> source dir):")
                if loaded_real_adapters:
                    for real in sorted(loaded_real_adapters):
                        src = real_to_source.get(real)
                        marker = " (active)" if active_real == real else ""
                        print(f" - {real} -> {src}{marker}")
                else:
                    print(" (none)")
                
                print("\nAliases:")
                alias_items = sorted(alias_to_real.items(), key=lambda x: x[0])
                if alias_items:
                    for a, r in alias_items:
                        marker = " (active)" if active_label == a else ""
                        print(f" - {a} -> {r}{marker}")
                else:
                    print(" (none)")
                    
                baseline_marker = " (active)" if active_real is None else ""
                print(f"\n - baseline{baseline_marker}\n")
                continue
                
            if cmd == "/baseline":
                active_real = None
                active_label = "baseline"
                print("Switched to baseline (no adapter)\n")
                continue
            
            if cmd == "/switch":
                if len(parts) < 2:
                    print("Usage: /switch <alias_or_name>\n")
                    continue
                name_or_alias = parts[1]
                
                real = alias_to_real.get(name_or_alias, name_or_alias)
                if real not in loaded_real_adapters:
                    print(f"Adapter '{name_or_alias}' not loaded. Use /load first.\n")
                    continue
                
                if peft_model is None:
                    print("Internal error: adapters are registered but peft_model is None.\n")
                    continue
                    
                try:
                    peft_model.set_adapter(real)
                    active_real = real
                    active_label = name_or_alias
                    print(f"Switched to adapter: {name_or_alias} (PEFT='{real}')\n")
                except Exception as e:
                    print(f"Error switching adapter: {e}\n")
                continue
                
            if cmd == "/load":
                try:
                    path, user_alias, epoch, latest, allow_root = _parse_load_args(parts)
                except Exception as e:
                    print(f"{e}\n")
                    continue
                
                try:
                    resolved_dir, run_name, resolved_epoch = resolve_adapter_dir(
                        path, epoch=epoch, latest=latest, allow_root=allow_root
                    )
                    cfg_path, weights_path = _validate_adapter_dir(resolved_dir)
                except Exception as e:
                    print(f"Error resolving adapter path: {e}\n")
                    continue
                
                desired_real = make_real_adapter_name(run_name, resolved_epoch)
                
                # Aliases to register
                aliases_to_register = [desired_real]
                aliases_to_register.extend(infer_aliases(run_name, resolved_epoch))
                if user_alias:
                    aliases_to_register.append(user_alias)
                
                # de-dupe keep order
                seen = set()
                aliases_to_register = [a for a in aliases_to_register if not (a in seen or seen.add(a))]
                
                # Print explicit confirmation of what will be loaded
                print("\n" + "-" * 60)
                print("Adapter load request:")
                print(f"  User path: {path}")
                print(f"  Resolved dir: {resolved_dir}")
                print(f"  adapter_config: {cfg_path}")
                print(f"  adapter_model: {weights_path}")
                print(f"  Run name: {run_name}")
                print(f"  Resolved epoch: {resolved_epoch if resolved_epoch is not None else 'root'}")
                print(f"  Desired PEFT name: {desired_real}")
                print(f"  Aliases: {', '.join(aliases_to_register)}")
                
                if desired_real in loaded_real_adapters:
                    for a in aliases_to_register:
                        alias_to_real.setdefault(a, desired_real)
                    print(f"  Status: already loaded; aliases updated")
                    print("-" * 60 + "\n")
                    continue
                
                try:
                    if peft_model is None:
                        # Create wrapper and try to name it
                        try:
                            peft_model = PeftModel.from_pretrained(
                                base_model,
                                str(resolved_dir),
                                adapter_name=desired_real,
                                is_trainable=False
                            )
                            peft_model.to(device)
                            peft_model.eval()
                            real_in_model = desired_real
                        except TypeError:
                            # Older PEFT: no adapter_name / is_trainable kwargs
                            peft_model = PeftModel.from_pretrained(base_model, str(resolved_dir))
                            peft_model.to(device)
                            peft_model.eval()
                            existing = _peft_adapter_names(peft_model)
                            # Best-effort: infer what name PEFT used
                            if len(existing) == 1:
                                real_in_model = existing[0]
                            else:
                                real_in_model = existing[0] if existing else "default"
                            
                            if real_in_model != desired_real:
                                print(
                                    f"  Note: this PEFT version did not accept adapter_name; "
                                    f"adapter registered as '{real_in_model}' instead of '{desired_real}'."
                                )
                        
                        # ensure active adapter set (if possible)
                        try:
                            peft_model.set_adapter(real_in_model)
                        except Exception:
                            pass
                    else:
                        # Add another adapter to existing peft_model
                        try:
                            peft_model.load_adapter(
                                str(resolved_dir),
                                adapter_name=desired_real,
                                is_trainable=False
                            )
                            real_in_model = desired_real
                        except TypeError:
                            peft_model.load_adapter(str(resolved_dir), adapter_name=desired_real)
                            real_in_model = desired_real
                    
                    loaded_real_adapters.add(real_in_model)
                    real_to_source[real_in_model] = resolved_dir
                    
                    # Register aliases
                    for a in aliases_to_register:
                        if a in alias_to_real and alias_to_real[a] != real_in_model:
                            print(
                                f"  Warning: alias '{a}' already points to '{alias_to_real[a]}'; "
                                f"not reassigning to '{real_in_model}'."
                            )
                        else:
                            alias_to_real[a] = real_in_model
                    
                    # Print final confirmation of what PEFT actually has
                    print(f"  Loaded as PEFT: {real_in_model}")
                    print(f"  PEFT adapters now: {', '.join(_peft_adapter_names(peft_model)) or '(unknown)'}")
                    print(f"  Active adapter: {_safe_get_active_adapter(peft_model) or '(unknown)'}")
                    print("  Status: loaded successfully")
                    print("-" * 60 + "\n")
                    
                except Exception as e:
                    print(f"Error loading adapter: {e}\n")
                    continue
                continue
            
            print(f"Unknown command: {cmd}. Type /help for commands.\n")
            continue
            
        # Text generation
        print("\nGenerating...\n")
        
        if active_real is None:
            # baseline mode: either use base_model directly, or disable adapter on peft_model
            if peft_model is not None:
                generate_text(peft_model, tokenizer, user_input, device, disable_adapter=True)
            else:
                generate_text(base_model, tokenizer, user_input, device)
        else:
            if peft_model is None:
                print("No adapter model loaded; use /load first.\n")
                continue
            # Ensure correct adapter is active (just in case)
            if _safe_get_active_adapter(peft_model) != active_real:
                peft_model.set_adapter(active_real)
            
            generate_text(peft_model, tokenizer, user_input, device)

if __name__ == "__main__":
    main()
