"""
Interactive console for comparing baseline vs baked Llama-3 models.

Usage:
  python interactive_baked.py

Commands:
  /load <path> [alias]  - Load a baked adapter from <path>.
                          It will be available under:
                          - the inferred "full" name (usually the run dir)
                          - one or more short aliases (e.g. first 2 underscore parts)
                          - optionally, your provided [alias]
  /switch <name>        - Switch to an adapter (full name OR alias)
  /baseline             - Switch to baseline (no adapter)
  /list                 - List loaded adapters + aliases
  /help                 - Show this help
  /quit                 - Exit
  <any text>            - Generate from current model
"""

import os
import shlex
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel


def _split_cmd(line: str) -> list[str]:
    # shlex POSIX mode breaks Windows backslash paths; use non-POSIX on Windows.
    if os.name == "nt":
        return shlex.split(line, posix=False)
    return shlex.split(line, posix=True)


def _infer_full_name(adapter_path: str) -> str:
    """
    Infer a stable 'full name' from a checkpoint path.
    Typical layout:
      results/<run_name>/epoch_6  -> full name = <run_name>
      results/<run_name>         -> full name = <run_name>
      some/dir/adapter           -> full name = adapter
    """
    p = Path(os.path.normpath(adapter_path))

    # If path ends in epoch_*, use parent directory as the run name
    if p.name.lower().startswith("epoch_") and p.parent.name:
        return p.parent.name

    return p.name if p.name else str(p)


def _infer_short_aliases(full_name: str) -> list[str]:
    """
    Generate short alias(es) for convenience.
    Example:
      baked_caf_caf_bs4_ep50 -> baked_caf
    """
    parts = [x for x in full_name.split("_") if x]
    aliases = []
    if len(parts) >= 2:
        aliases.append("_".join(parts[:2]))
    return aliases


def _normalize_path(p: str) -> str:
    # Preserve backslashes on Windows, but normalize things like '..'
    return os.path.normpath(p)


def load_base_model(model_name: str = "meta-llama/Meta-Llama-3-8B-Instruct"):
    print(f"Loading base model: {model_name}...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Newer Transformers prefers `dtype=...`, older uses `torch_dtype=...`.
    try:
        base_model = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype)
    except TypeError:
        base_model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype)

    base_model.to(device)
    base_model.eval()

    print(f"Model loaded on {device}")
    return base_model, tokenizer, device


@torch.inference_mode()
def generate_text(
    model,
    tokenizer,
    prompt: str,
    device: torch.device,
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    disable_adapter: bool = False,
):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    if disable_adapter and hasattr(model, "disable_adapter"):
        with model.disable_adapter():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )
    else:
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def print_help():
    print("\nCommands:")
    print("  /load <path> [alias]  - Load a baked adapter (adds full name + short alias)")
    print("  /switch <name>        - Switch to an adapter (full name OR alias)")
    print("  /baseline             - Switch to baseline (no adapter)")
    print("  /list                 - List loaded adapters + aliases")
    print("  /help                 - Show this help")
    print("  /quit                 - Exit")
    print("  <any text>            - Generate from current model\n")


def main():
    print("=" * 60)
    print("Interactive Baked Model Console")
    print("=" * 60)

    base_model, tokenizer, device = load_base_model()

    # We keep ONE peft_model and load multiple adapters into it.
    peft_model: PeftModel | None = None

    # Real adapters are the names PEFT knows internally.
    loaded_real_adapters: set[str] = set()

    # Aliases map user-facing names -> real PEFT adapter name
    alias_to_real: dict[str, str] = {}

    # What the prompt shows (can be alias or full name)
    current_mode_label = "baseline"

    print_help()

    while True:
        try:
            user_input = input(f"[{current_mode_label}]> ").strip()
            if not user_input:
                continue

            # Commands
            if user_input.startswith("/"):
                try:
                    parts = _split_cmd(user_input)
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
                    print("\nLoaded adapters (real / full names):")
                    if loaded_real_adapters:
                        for name in sorted(loaded_real_adapters):
                            marker = " (active)" if name == current_mode_label else ""
                            print(f"  - {name}{marker}")
                    else:
                        print("  (none)")

                    print("\nAliases:")
                    alias_lines = [(a, r) for a, r in sorted(alias_to_real.items()) if a != r]
                    if alias_lines:
                        for alias, real in alias_lines:
                            marker = " (active)" if alias == current_mode_label else ""
                            print(f"  - {alias} -> {real}{marker}")
                    else:
                        print("  (none)")

                    baseline_marker = " (active)" if current_mode_label == "baseline" else ""
                    print(f"\n  - baseline{baseline_marker}\n")
                    continue

                if cmd == "/baseline":
                    current_mode_label = "baseline"
                    print("Switched to baseline (no adapter)\n")
                    continue

                if cmd == "/load":
                    if len(parts) < 2:
                        print("Usage: /load <path> [alias]\n")
                        continue

                    adapter_path = _normalize_path(parts[1])
                    full_name = _infer_full_name(adapter_path)
                    short_aliases = _infer_short_aliases(full_name)
                    user_alias = parts[2] if len(parts) >= 3 else None

                    # Validate path early for clearer errors
                    cfg = os.path.join(adapter_path, "adapter_config.json")
                    if not os.path.exists(cfg):
                        print(f"Error: Can't find 'adapter_config.json' at '{adapter_path}'.")
                        print("Tip: On Windows, prefer /load \"results\\run\\epoch_6\" or /load results/run/epoch_6\n")
                        continue

                    aliases_to_register = [full_name, *short_aliases]
                    if user_alias:
                        aliases_to_register.append(user_alias)

                    # De-dupe, keep order
                    seen = set()
                    aliases_to_register = [a for a in aliases_to_register if not (a in seen or seen.add(a))]

                    if full_name in loaded_real_adapters:
                        for a in aliases_to_register:
                            alias_to_real.setdefault(a, full_name)
                        print(f"Adapter already loaded as '{full_name}'. Aliases updated: {aliases_to_register}\n")
                        continue

                    try:
                        print(f"Loading adapter '{full_name}' from {adapter_path}...")

                        if peft_model is None:
                            # Prefer naming the adapter at creation time
                            try:
                                peft_model = PeftModel.from_pretrained(
                                    base_model,
                                    adapter_path,
                                    adapter_name=full_name,
                                    is_trainable=False,
                                )
                            except TypeError:
                                peft_model = PeftModel.from_pretrained(base_model, adapter_path)

                            peft_model.to(device)
                            peft_model.eval()

                            # If the first creation couldn't name it, try loading again under the desired name
                            if full_name not in loaded_real_adapters:
                                try:
                                    peft_model.load_adapter(adapter_path, adapter_name=full_name, is_trainable=False)
                                except TypeError:
                                    peft_model.load_adapter(adapter_path, adapter_name=full_name)

                        else:
                            try:
                                peft_model.load_adapter(adapter_path, adapter_name=full_name, is_trainable=False)
                            except TypeError:
                                peft_model.load_adapter(adapter_path, adapter_name=full_name)

                        loaded_real_adapters.add(full_name)

                        for a in aliases_to_register:
                            if a in alias_to_real and alias_to_real[a] != full_name:
                                print(
                                    f"Warning: alias '{a}' already points to '{alias_to_real[a]}'; "
                                    f"not reassigning to '{full_name}'."
                                )
                            else:
                                alias_to_real[a] = full_name

                        print(f"Adapter '{full_name}' loaded successfully.")
                        print(f"Available names: {', '.join(aliases_to_register)}\n")

                    except Exception as e:
                        print(f"Error loading adapter: {e}\n")

                    continue

                if cmd == "/switch":
                    if len(parts) < 2:
                        print("Usage: /switch <name>\n")
                        continue

                    name_or_alias = parts[1]
                    real_name = alias_to_real.get(name_or_alias)

                    if real_name is None or real_name not in loaded_real_adapters:
                        print(f"Adapter '{name_or_alias}' not loaded. Use /load first.\n")
                        continue

                    if peft_model is None:
                        print("Internal error: adapters are registered but peft_model is None.\n")
                        continue

                    try:
                        peft_model.set_adapter(real_name)
                        current_mode_label = name_or_alias
                        print(f"Switched to adapter: {name_or_alias} (-> {real_name})\n")
                    except Exception as e:
                        print(f"Error switching adapter: {e}\n")

                    continue

                print(f"Unknown command: {cmd}. Type /help for commands.\n")
                continue

            # Text generation
            print("\nGenerating...\n")

            if current_mode_label == "baseline":
                if peft_model is not None:
                    output = generate_text(peft_model, tokenizer, user_input, device, disable_adapter=True)
                else:
                    output = generate_text(base_model, tokenizer, user_input, device)
            else:
                if peft_model is None:
                    print("No adapter model loaded; use /load first.\n")
                    continue
                output = generate_text(peft_model, tokenizer, user_input, device)

            print(output)
            print("\n" + "-" * 60 + "\n")

        except KeyboardInterrupt:
            print("\nUse /quit to exit.\n")
        except Exception as e:
            print(f"Error: {e}\n")


if __name__ == "__main__":
    main()
