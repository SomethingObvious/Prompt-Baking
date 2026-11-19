"""Helpers shared by the prompt-baking scripts."""

import contextlib
import json
import random
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
from datasets import concatenate_datasets, load_dataset
from peft import LoraConfig, PeftConfig, PeftModel, TaskType, get_peft_model
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

DEFAULT_MODEL = "Qwen/Qwen3.5-2B"
# Qwen3.5 only has attention in one layer out of four, and the Gated DeltaNet layers in between
# name their projections differently. Names a model doesn't have are skipped.
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "in_proj_qkv", "in_proj_z", "out_proj"]
DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
EPOCH_DIR = re.compile(r"epoch_(\d+)")


def log(msg, file_path):
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with Path(file_path).open("a", encoding="utf-8") as f:
        f.write(f"[{stamp}] {msg}\n")


def pad_list_of_lists(rows, value):
    width = max(len(row) for row in rows)
    return [row + [value] * (width - len(row)) for row in rows]


def crop_trajectories(input_ids_list, mask_list, max_traj_len=-1):
    """Cuts each row after its last generated token, or after max_traj_len of them if positive."""
    ids_out, mask_out = [], []
    for ids, mask in zip(input_ids_list, mask_list, strict=True):
        if len(ids) != len(mask):
            raise ValueError(f"A row has {len(ids)} ids but {len(mask)} mask entries.")
        if 1 not in mask:
            raise ValueError("A row has no generated tokens, so there's nothing in it to train on.")
        end = len(mask) - mask[::-1].index(1)
        if max_traj_len > 0:
            end = min(end, mask.index(1) + max_traj_len)
        ids_out.append(ids[:end])
        mask_out.append(mask[:end])
    return ids_out, mask_out


def kl_positions(mask):
    """Positions whose next-token distribution gets matched, from a mask of generated tokens."""
    # The logits at t predict token t + 1, so the last prompt token has to be in here too or the
    # first generated token never gets trained, which is the one "start with A" is about.
    pos = mask.clone()
    pos[:, :-1] |= mask[:, 1:]
    return pos


def logits_at(model, ids, pos):
    """Float logits at the positions marked in pos, row after row."""
    # Only the tail from the earliest marked position goes through the LM head. A long system
    # prompt would otherwise cost a vocab-sized row of logits per prompt token.
    keep = ids.shape[1] - int(pos.int().argmax(dim=1).min())
    logits = model(input_ids=ids, logits_to_keep=keep, use_cache=False).logits
    return logits[pos[:, -keep:]].float()


def batch_kl(model, batch, max_traj_len=-1):
    """KL(prompted || unprompted) summed over each row's generated tokens and averaged over rows."""
    sides = []
    for ids_key, mask_key in (
        ("input_ids", "generated_text_mask"),
        ("input_ids_nosys", "generated_text_mask_nosys"),
    ):
        ids, mask = crop_trajectories(batch[ids_key], batch[mask_key], max_traj_len)
        # Right padding sits after every real token, so causal attention never sees it and any id
        # works as the pad. That's also why there's no attention mask.
        ids = torch.tensor(pad_list_of_lists(ids, 0), device=model.device)
        mask = torch.tensor(pad_list_of_lists(mask, 0), device=model.device, dtype=torch.bool)
        sides.append((ids, mask))
    (ids, mask), (ids_nosys, mask_nosys) = sides
    if not torch.equal(ids[mask], ids_nosys[mask_nosys]):
        raise ValueError("The prompted and unprompted rows don't hold the same generated tokens.")

    with torch.no_grad(), model.disable_adapter():
        target = logits_at(model, ids, kl_positions(mask)).log_softmax(dim=-1)
    student = logits_at(model, ids_nosys, kl_positions(mask_nosys)).log_softmax(dim=-1)
    kl = F.kl_div(student, target, reduction="sum", log_target=True)
    return kl / ids.shape[0]


def do_epoch(model, rows, batch_size, optimizer=None, scaler=None, max_traj_len=-1, desc=None):
    """Runs one pass over rows and returns each batch's KL. It trains if given an optimizer."""
    training = optimizer is not None
    scaler = scaler or torch.amp.GradScaler(enabled=False)
    model.train(training)
    kls = []
    bar = tqdm(range(0, len(rows), batch_size), desc=desc)
    for i in bar:
        with torch.set_grad_enabled(training):
            kl = batch_kl(model, rows[i : i + batch_size], max_traj_len)
        if training:
            scaler.scale(kl).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
        kls.append(kl.item())
        bar.set_postfix(kl=f"{kls[-1]:.4f}")
    return kls


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def resolve_dtype(name, device):
    if name != "auto":
        return DTYPES[name]
    if device.type != "cuda":
        return torch.float32
    # Emulated bf16 counts as supported by default, and it's far slower than fp16 on older cards.
    return (
        torch.bfloat16 if torch.cuda.is_bf16_supported(including_emulation=False) else torch.float16
    )


def default_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(name, device, dtype):
    return AutoModelForCausalLM.from_pretrained(name, dtype=dtype, device_map=device)


def load_tokenizer(name):
    # Left padding, since the only batches that get padded here are prompts for generate(). Llama
    # 3's config turns on a cleanup that drops spaces before punctuation in BPE.
    tokenizer = AutoTokenizer.from_pretrained(
        name, padding_side="left", clean_up_tokenization_spaces=False
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_baked(adapter_dir, model_name=None, device=None, dtype="auto"):
    """The base model with the adapter in adapter_dir on top, and its tokenizer."""
    name = model_name or PeftConfig.from_pretrained(adapter_dir).base_model_name_or_path
    device = torch.device(device) if device else default_device()
    base = load_model(name, device, resolve_dtype(dtype, device))
    return PeftModel.from_pretrained(base, str(adapter_dir)).eval(), load_tokenizer(name)


def stop_ids(model, tokenizer):
    """The ids that end a reply, from both the generation config and the tokenizer."""
    # Qwen3.5 has no generation_config.json, so its config only knows <|endoftext|>, which comes a
    # token after the <|im_end|> that ends the turn. min_new_tokens only holds back the ids it's
    # given, so without both a reply could end before --min_sequence_length.
    eos = model.generation_config.eos_token_id
    return sorted({tokenizer.eos_token_id, *(eos if isinstance(eos, list) else [eos])} - {None})


def generate(model, tokenizer, prompts, adapter=True, **kwargs):
    """Decoded replies to rendered chat prompts, with the adapter on or off."""
    enc = tokenizer(prompts, return_tensors="pt", padding=True, add_special_tokens=False)
    enc = enc.to(model.device)
    off = contextlib.nullcontext() if adapter else model.disable_adapter()
    with torch.no_grad(), off:
        out = model.generate(
            **enc,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=stop_ids(model, tokenizer),
            **kwargs,
        )
    return tokenizer.batch_decode(out[:, enc["input_ids"].shape[1] :], skip_special_tokens=True)


def list_epoch_dirs(run_dir):
    """The run's epoch_N folders as (N, path), oldest first."""
    run_dir = Path(run_dir)
    if not run_dir.is_dir():
        return []
    found = [(int(m.group(1)), p) for p in run_dir.iterdir() if (m := EPOCH_DIR.fullmatch(p.name))]
    return sorted(found)


def pick_epoch(run_dir, which="last"):
    """The (N, path) of epoch_N in run_dir, where which is a number or "last"."""
    epochs = dict(list_epoch_dirs(run_dir))
    if not epochs:
        raise FileNotFoundError(f"There are no epoch_N folders in {run_dir}.")
    n = max(epochs) if which == "last" else int(which)
    if n not in epochs:
        raise FileNotFoundError(f"{run_dir} has no epoch_{n} folder.")
    return n, epochs[n]


def load_questions(name, include_train, include_val, include_test, seed):
    """The chosen splits of data/<name>_*.jsonl, shuffled together."""
    flags = {"train": include_train, "validation": include_val, "test": include_test}
    splits = [split for split, flag in flags.items() if flag == "True"]
    if not splits:
        raise ValueError(
            "Pick at least one split with --include_train, --include_val or --include_test."
        )
    parts = []
    for split in splits:
        part = load_dataset("json", data_files=f"data/{name}_{split}.jsonl", split="train")
        parts.append(part.shuffle(seed=seed))
    return concatenate_datasets(parts).shuffle(seed=seed)


def add_training_args(parser):
    """Adds the flags that describe a run, which is what args.json holds."""
    parser.add_argument("--data_path", help="Training trajectories from generate_data.py.")
    parser.add_argument("--val_path", help="Validation trajectories, used to pick the best epoch.")
    parser.add_argument(
        "--model_name",
        default=DEFAULT_MODEL,
        help="Base model. It has to be the one that generated the trajectories.",
    )
    parser.add_argument("--num_epochs", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=10)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("-r", type=int, default=32, help="LoRA rank.")
    parser.add_argument("--lora_alpha", type=int, default=128)
    parser.add_argument(
        "--max_traj_len",
        type=int,
        default=-1,
        help="How many generated tokens of each row to train on. The default of -1 uses them all.",
    )
    parser.add_argument(
        "--save_every", type=int, default=1, help="Save an epoch_N adapter every this many epochs."
    )
    parser.add_argument("--device", default=str(default_device()))
    parser.add_argument(
        "--dtype",
        choices=["auto", *DTYPES],
        default="auto",
        help="The default picks bf16 on GPUs that have it, fp16 on older ones and fp32 on CPU.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Recompute activations in the backward pass, which is slower but fits longer rows.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Seeds the RNGs and the shuffle order. Leave it out for a random run.",
    )
    parser.add_argument(
        "--logdir", help="TensorBoard folder. It defaults to <out_dir>/tensorboard."
    )
    parser.add_argument("--no_tensorboard", action="store_true")


def make_tb_writer(args):
    if args.no_tensorboard:
        return None
    logdir = args.logdir or str(Path(args.out_dir) / "tensorboard")
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ImportError:
        print("TensorBoard isn't installed, so this run won't log curves.")
        return None
    print(f"Logging curves to {logdir}, and tensorboard --logdir {logdir} shows them.")
    return SummaryWriter(logdir)


def train(args, start_epoch=0, adapter_dir=None):
    """Trains epochs start_epoch up to args.num_epochs, carrying on from adapter_dir if given."""
    out_dir = Path(args.out_dir)
    log_path = out_dir / "train_loop.log"
    state_path = out_dir / "training_state.pt"
    best_path = out_dir / "best.json"
    if args.seed is not None:
        set_seed(args.seed)
    device = torch.device(args.device)
    dtype = resolve_dtype(args.dtype, device)

    train_rows = load_dataset("json", data_files=args.data_path, split="train")
    val_rows = load_dataset("json", data_files=args.val_path, split="train")
    model = load_model(args.model_name, device, dtype)
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    if adapter_dir is None:
        config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=args.r,
            lora_alpha=args.lora_alpha,
            lora_dropout=0.0,
            target_modules=LORA_TARGETS,
        )
        model = get_peft_model(model, config)
    else:
        model = PeftModel.from_pretrained(model, str(adapter_dir), is_trainable=True)
    model.print_trainable_parameters()
    log(f"Training {args.model_name} ({dtype}) on {device} from epoch {start_epoch}", log_path)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=args.learning_rate)
    # Plain fp16 underflows in the backward pass without loss scaling. bf16 and fp32 don't need it.
    scaler = torch.amp.GradScaler(device.type, enabled=dtype == torch.float16)
    best = float("inf")
    if adapter_dir is not None:
        if best_path.is_file():
            best = json.loads(best_path.read_text(encoding="utf-8"))["val_kl"]
        state = torch.load(state_path, map_location=device) if state_path.is_file() else None
        if state and state["epoch"] == start_epoch - 1:
            optimizer.load_state_dict(state["optimizer"])
            scaler.load_state_dict(state["scaler"])
        else:
            print(f"There's no optimizer state for epoch {start_epoch - 1}, so Adam starts over.")

    writer = make_tb_writer(args)
    for epoch in range(start_epoch, args.num_epochs):
        # The trajectory files hold every question's replies back to back, so without a shuffle
        # each batch would be one question repeated.
        rows = train_rows.shuffle(seed=None if args.seed is None else args.seed + epoch)
        train_kls = do_epoch(
            model,
            rows,
            args.batch_size,
            optimizer,
            scaler,
            args.max_traj_len,
            f"epoch {epoch} train",
        )
        val_kls = do_epoch(
            model,
            val_rows,
            args.batch_size,
            max_traj_len=args.max_traj_len,
            desc=f"epoch {epoch} val",
        )
        train_kl, val_kl = float(np.mean(train_kls)), float(np.mean(val_kls))
        msg = f"Epoch {epoch} has a train KL of {train_kl:.4f} and a validation KL of {val_kl:.4f}"
        print(msg)
        log(msg, log_path)
        if writer is not None:
            writer.add_scalar("kl/train", train_kl, epoch)
            writer.add_scalar("kl/val", val_kl, epoch)

        if val_kl < best:
            best = val_kl
            model.save_pretrained(str(out_dir))
            best_path.write_text(json.dumps({"epoch": epoch, "val_kl": best}), encoding="utf-8")
            log(f"Saved epoch {epoch} to {out_dir} as the best so far", log_path)
        if (epoch + 1) % args.save_every == 0:
            model.save_pretrained(str(out_dir / f"epoch_{epoch}"))
            # Only the newest epoch keeps its optimizer state, since Adam's is twice the adapter.
            state = {
                "epoch": epoch,
                "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict(),
            }
            torch.save(state, state_path.with_suffix(".tmp"))
            state_path.with_suffix(".tmp").replace(state_path)
            log(f"Saved epoch {epoch} to {out_dir / f'epoch_{epoch}'}", log_path)

    if writer is not None:
        writer.close()
