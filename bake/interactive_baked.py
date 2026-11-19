"""A console for chatting with baked adapters and flipping between them and the base model."""

import argparse
from pathlib import Path

from peft import PeftModel

from generate_data import format_prompt
from train_utils import (
    DEFAULT_MODEL,
    EPOCH_DIR,
    default_device,
    generate,
    load_model,
    load_tokenizer,
    pick_epoch,
    resolve_dtype,
)

HELP = """
/load <path> [alias] [--epoch N | --latest]
    A run folder loads its best adapter, or its epoch_N one with --epoch N or --latest.
    Any other adapter folder loads as it is.
/switch <alias>    use a loaded adapter
/baseline          use the base model
/list              show what's loaded
/help              show this again
/quit
Anything else gets sent to the current model.
"""


def parse_load(parts):
    """(path, alias, epoch, latest) from the words of a /load command."""
    if len(parts) < 2:
        raise ValueError("Usage is /load <path> [alias] [--epoch N | --latest]")
    path, alias, epoch, latest = parts[1], None, None, False
    rest = iter(parts[2:])
    for word in rest:
        if word == "--epoch":
            value = next(rest, "")
            if not value.isdigit():
                raise ValueError("--epoch needs a number after it.")
            epoch = int(value)
        elif word == "--latest":
            latest = True
        elif alias is None:
            alias = word
        else:
            raise ValueError(f"/load doesn't take {word}.")
    return path, alias, epoch, latest


def resolve_adapter_dir(path_str, epoch=None, latest=False):
    """The adapter folder a /load path points at, and a default name for it."""
    path = Path(path_str)
    if not path.exists():
        path = Path("results") / path_str
    if not path.is_dir():
        raise FileNotFoundError(f"There's no folder called {path_str}, here or in results/.")
    if epoch is None and not latest and (path / "adapter_config.json").is_file():
        m = EPOCH_DIR.fullmatch(path.name)
        return path, f"{path.parent.name}-e{m.group(1)}" if m else path.name
    n, epoch_dir = pick_epoch(path, "last" if epoch is None else epoch)
    return epoch_dir, f"{path.name}-e{n}"


def set_scaling(model, adapter, value):
    """Sets one adapter's LoRA scaling on every layer and returns how many layers had it."""
    # Attention modules have a float called scaling too, so only the dicts are LoRA layers.
    layers = [
        m
        for m in model.modules()
        if isinstance(getattr(m, "scaling", None), dict) and adapter in m.scaling
    ]
    for layer in layers:
        layer.scaling[adapter] = value
    return len(layers)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_name", default=DEFAULT_MODEL, help="The adapters' base model.")
    parser.add_argument(
        "--alpha-scale",
        type=float,
        metavar="SCALE",
        help="Sets the LoRA scaling (alpha / r) of each adapter you load, for a tiny alpha.",
    )
    args = parser.parse_args()

    device = default_device()
    print(f"Loading {args.model_name} on {device}")
    base = load_model(args.model_name, device, resolve_dtype("auto", device))
    tokenizer = load_tokenizer(args.model_name)
    model = None  # a PeftModel around base, once the first adapter loads
    sources = {}
    active = None  # None means the base model
    print(HELP)

    while True:
        try:
            line = input(f"[{active or 'baseline'}]> ").strip()
        except EOFError:
            break
        except KeyboardInterrupt:
            print("\nType /quit to leave.")
            continue
        if not line:
            continue

        if not line.startswith("/"):
            prompt = [format_prompt(tokenizer, line)]
            sampling = {"do_sample": True, "temperature": 0.7, "max_new_tokens": 256}
            if model is None:
                (reply,) = generate(base, tokenizer, prompt, **sampling)
            else:
                if active is not None:
                    model.set_adapter(active)
                (reply,) = generate(model, tokenizer, prompt, active is not None, **sampling)
            print(f"\n{reply.strip()}\n")
            continue

        parts = line.split()
        cmd = parts[0].lower()
        if cmd in {"/quit", "/exit", "/q", "/e"}:
            break
        if cmd == "/help":
            print(HELP)
        elif cmd == "/list":
            for name, src in sources.items():
                print(f"  {name} from {src}{' (active)' if name == active else ''}")
            print(f"  baseline{' (active)' if active is None else ''}")
        elif cmd == "/baseline":
            active = None
        elif cmd == "/switch":
            if len(parts) < 2 or parts[1] not in sources:
                print(
                    f"Load it first. The adapters loaded so far are {', '.join(sources) or 'none'}."
                )
            else:
                active = parts[1]
        elif cmd == "/load":
            try:
                path, alias, epoch, latest = parse_load(parts)
                adapter_dir, name = resolve_adapter_dir(path, epoch, latest)
            except (ValueError, FileNotFoundError) as e:
                print(e)
                continue
            name = alias or name
            if name in sources:
                print(f"{name} is already loaded, from {sources[name]}.")
                continue
            try:
                if model is None:
                    model = PeftModel.from_pretrained(base, str(adapter_dir), adapter_name=name)
                    model.eval()
                else:
                    model.load_adapter(str(adapter_dir), adapter_name=name)
            except (ValueError, OSError, RuntimeError) as e:
                print(f"Couldn't load {adapter_dir}. {e}")
                continue
            sources[name] = adapter_dir
            print(f"Loaded {adapter_dir} as {name}. Type /switch {name} to use it.")
            if args.alpha_scale is not None:
                count = set_scaling(model, name, args.alpha_scale)
                print(f"Its scaling is {args.alpha_scale} on all {count} LoRA layers.")
        else:
            print(f"There's no {cmd} command, and /help lists the ones there are.")


if __name__ == "__main__":
    main()
