# train_loop_resume.py
# Resume LoRA prompt-baking training from latest epoch_* adapter folder.

import os
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
import re
import json
import argparse
from datetime import datetime

import torch
import torch.nn.functional as F
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, TaskType, get_peft_model, PeftModel


# ----------------- logging helper -----------------
def log(msg, file_path):
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(file_path, "a") as f:
        f.write(f"[{current_time}] {msg}\n")


# ----------------- padding utilities -----------------
def pad_list_of_lists(llist, pad_tok_val, verbose=False, pad_side="right", return_pad_mask=False):
    assert pad_side in ["left", "right"], "pad_side must be either 'left' or 'right'"

    max_len = max(len(l) for l in llist)
    if pad_side == "right":
        padded_list = [l + [pad_tok_val] * (max_len - len(l)) for l in llist]
    else:
        padded_list = [[pad_tok_val] * (max_len - len(l)) + l for l in llist]

    if verbose:
        for idx, l in enumerate(llist):
            if len(l) != max_len:
                print(f"Unequal length list at batch el {idx}: ", l)

    if return_pad_mask:
        num_pads_list = [max_len - len(l) for l in llist]
        pad_mask = [[0 if i < num_pads else 1 for i in range(max_len)] for num_pads in num_pads_list]
        if pad_side == "right":
            pad_mask = [l[::-1] for l in pad_mask]
        return padded_list, pad_mask

    return padded_list


def crop_trajectories(input_ids_list, mask_list, max_traj_len):
    assert max_traj_len > 0, "max_traj_len must be greater than 0 in crop_trajectories()"

    new_input_ids_list = []
    new_mask_list = []
    for input_ids, mask in zip(input_ids_list, mask_list):
        assert len(input_ids) == len(mask), "input_ids and mask must be same length"
        first_1_ind = mask.index(1)
        cut_idx = first_1_ind + max_traj_len
        new_input_ids_list.append(input_ids[:cut_idx])
        new_mask_list.append(mask[:cut_idx])

    return new_input_ids_list, new_mask_list


# ----------------- epoch loop -----------------
def do_epoch(
    peft_model,
    tokenizer,
    dataset,
    batch_size,
    log_path,
    optimizer,
    do_step=True,
    max_traj_len=-1,
):
    kl_divs = []
    device = peft_model.device

    for i in tqdm(range(0, len(dataset["train"]), batch_size)):
        log(f"Batch {i}", log_path)
        batch = dataset["train"][i : i + batch_size]

        input_ids_nosys_list_ = batch["input_ids_nosys"]
        input_ids_list_ = batch["input_ids"]

        mask_nosys_list_ = batch["generated_text_mask_nosys"]
        mask_list_ = batch["generated_text_mask"]

        if max_traj_len > 0:
            input_ids_list_, mask_list_ = crop_trajectories(input_ids_list_, mask_list_, max_traj_len)
            input_ids_nosys_list_, mask_nosys_list_ = crop_trajectories(
                input_ids_nosys_list_, mask_nosys_list_, max_traj_len
            )

        # pad
        input_ids_nosys_list = pad_list_of_lists(input_ids_nosys_list_, tokenizer.pad_token_id, verbose=False)
        input_ids_list = pad_list_of_lists(input_ids_list_, tokenizer.pad_token_id, verbose=False)
        mask_nosys_list = pad_list_of_lists(mask_nosys_list_, 0, verbose=False)
        mask_list = pad_list_of_lists(mask_list_, 0, verbose=False)

        input_ids = torch.tensor(input_ids_list, device=device)
        input_ids_nosys = torch.tensor(input_ids_nosys_list, device=device)
        mask = torch.tensor(mask_list, device=device) == 1
        mask_nosys = torch.tensor(mask_nosys_list, device=device) == 1

        assert input_ids.shape == mask.shape
        assert input_ids_nosys.shape == mask_nosys.shape
        assert (input_ids[mask] != input_ids_nosys[mask_nosys]).sum() == 0, (
            "Prompted and unprompted input_ids do not match within their respective masks "
            "for the generated text (must be identical)"
        )
        assert (mask.sum(dim=1) == mask_nosys.sum(dim=1)).all(), "Prompted and unprompted masks must match per row"

        # unprompted logits (trainable, adapters enabled)
        log("Computing unprompted logits...", log_path)
        unprompted_logits_ = peft_model(input_ids_nosys).logits
        log("Done computing unprompted logits...", log_path)

        # prompted logits (frozen, adapters disabled)
        log("Computing prompted logits...", log_path)
        with peft_model.disable_adapter():
            with torch.no_grad():
                prompted_logits_ = peft_model(input_ids).logits
        log("Done computing prompted logits...", log_path)

        unprompted_logits = unprompted_logits_[mask_nosys, :]
        prompted_logits = prompted_logits_[mask, :]

        kl_div_ = F.kl_div(
            torch.log_softmax(unprompted_logits, dim=-1),
            torch.log_softmax(prompted_logits, dim=-1),
            reduction="none",
            log_target=True,
        )
        kl_div = kl_div_.sum() / batch_size

        kl_div.backward()
        if do_step:
            optimizer.step()
        optimizer.zero_grad()

        log(f"Done computing KL divergence = {kl_div.item()}", log_path)
        kl_divs.append(kl_div.item())

    avg = sum(kl_divs) / len(kl_divs)
    print(f"Epoch loss: {avg}")
    log(f"Epoch loss: {avg}", log_path)
    return kl_divs


# ----------------- resume helpers -----------------
_EPOCH_RE = re.compile(r"^epoch_(\d+)$")


def list_epoch_dirs(out_dir):
    if not os.path.isdir(out_dir):
        return []
    epochs = []
    for name in os.listdir(out_dir):
        m = _EPOCH_RE.match(name)
        if m:
            epochs.append((int(m.group(1)), os.path.join(out_dir, name)))
    epochs.sort(key=lambda x: x[0])
    return epochs


def verify_epoch_dir(epoch_dir):
    # Standard PEFT adapter save_pretrained artifacts
    req = ["adapter_config.json"]
    ok = all(os.path.isfile(os.path.join(epoch_dir, f)) for f in req)
    if not ok:
        return False, f"Missing one of required files: {req}"

    # adapter weights can be safetensors OR bin
    has_weights = (
        os.path.isfile(os.path.join(epoch_dir, "adapter_model.safetensors"))
        or os.path.isfile(os.path.join(epoch_dir, "adapter_model.bin"))
    )
    if not has_weights:
        return False, "Missing adapter weights file: adapter_model.safetensors or adapter_model.bin"

    return True, "OK"


def load_args_json(out_dir):
    p = os.path.join(out_dir, "args.json")
    if not os.path.isfile(p):
        return None
    with open(p, "r") as f:
        return json.load(f)


# ----------------- main -----------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Resume Prompt Baking LoRA training from latest epoch_*")

    # Optional overrides; if omitted, pulled from out_dir/args.json
    parser.add_argument("--out_dir", type=str, required=True, help="Run directory (contains epoch_* and args.json).")
    parser.add_argument("--model_name", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    parser.add_argument("--data_path", type=str, default=None)
    parser.add_argument("--val_path", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--max_traj_len", type=int, default=None)
    parser.add_argument("--save_every", type=int, default=None)
    parser.add_argument("-r", type=int, default=None, help="LoRA rank")

    # How long to train:
    # - default: finish until args.json num_epochs
    # - extend_epochs: add extra epochs beyond args.json num_epochs
    parser.add_argument("--extend_epochs", type=int, default=0, help="Extra epochs beyond args.json num_epochs.")

    args = parser.parse_args()

    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)
    log_path = os.path.join(out_dir, "train_loop.log")

    base_args = load_args_json(out_dir) or {}
    log(f"Loaded args.json: {base_args}", log_path)

    def pick(name, default=None):
        v = getattr(args, name)
        if v is not None:
            return v
        return base_args.get(name, default)

    # Pull defaults from args.json for seamless behavior
    data_path = pick("data_path")
    val_path = pick("val_path")
    batch_size = pick("batch_size")
    learning_rate = pick("learning_rate")
    max_traj_len = pick("max_traj_len", -1)
    save_every = pick("save_every", 1)
    lora_r = pick("r", 32)
    target_num_epochs = base_args.get("num_epochs", None)

    if data_path is None or val_path is None or batch_size is None or learning_rate is None:
        raise ValueError(
            "Missing required config. Provide --data_path/--val_path/--batch_size/--learning_rate "
            "or ensure they exist in out_dir/args.json."
        )

    if target_num_epochs is None:
        raise ValueError("args.json missing num_epochs; provide a run folder that contains it.")

    target_num_epochs = int(target_num_epochs) + int(args.extend_epochs)

    device = torch.device(args.device)
    log(f"Using device={args.device}", log_path)

    # Find latest epoch dir
    epoch_dirs = list_epoch_dirs(out_dir)
    if not epoch_dirs:
        print(f"[resume] No epoch_* folders found in: {out_dir}")
        print("[resume] Will start from epoch_0 (fresh LoRA).")
        latest_epoch = -1
        latest_dir = None
    else:
        latest_epoch, latest_dir = epoch_dirs[-1]
        ok, msg = verify_epoch_dir(latest_dir)
        if not ok:
            raise RuntimeError(f"Latest epoch dir failed verification: {latest_dir} :: {msg}")

        # Print + log what was found (this is what you asked for)
        print(f"[resume] Found epochs: {[e for e, _ in epoch_dirs]}")
        print(f"[resume] Latest epoch detected: epoch_{latest_epoch}")
        print(f"[resume] Latest epoch path: {latest_dir}")
        log(f"[resume] Found epochs: {[e for e, _ in epoch_dirs]}", log_path)
        log(f"[resume] Latest epoch detected: epoch_{latest_epoch}", log_path)
        log(f"[resume] Latest epoch path: {latest_dir}", log_path)

    start_epoch = latest_epoch + 1
    if start_epoch >= target_num_epochs:
        print(f"[resume] Nothing to do: start_epoch={start_epoch} >= target_num_epochs={target_num_epochs}")
        log(f"[resume] Nothing to do: start_epoch={start_epoch} >= target_num_epochs={target_num_epochs}", log_path)
        raise SystemExit(0)

    # Load datasets
    log("Loading dataset", log_path)
    dataset = load_dataset("json", data_files=data_path)
    log("Dataset loaded", log_path)

    log("Loading validation dataset", log_path)
    val_dataset = load_dataset("json", data_files=val_path)
    log("Validation dataset loaded", log_path)

    # Load base model + tokenizer
    model_name = args.model_name
    log(f"Loading base model {model_name}...", log_path)

    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
    ).to(device)
    log("Base model loaded", log_path)

    # Create or resume LoRA adapter
    if latest_dir is None:
        log("Creating fresh LoRA adapter (no resume checkpoint found).", log_path)
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=lora_r,
            lora_alpha=64,
            lora_dropout=0.0,
            bias="none",
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        )
        peft_model = get_peft_model(base_model, peft_config).to(device)
    else:
        log(f"Loading LoRA adapter from {latest_dir}", log_path)
        peft_model = PeftModel.from_pretrained(base_model, latest_dir, is_trainable=True).to(device)

    print("unprompted model parameter stats:")
    peft_model.print_trainable_parameters()
    log("PEFT model ready", log_path)

    optimizer = torch.optim.Adam(peft_model.parameters(), lr=learning_rate)

    best_val_loss = float("inf")

    # Continue training seamlessly from start_epoch up to target_num_epochs-1
    for epoch in range(start_epoch, target_num_epochs):
        log(f"Started epoch {epoch}", log_path)

        train_kls = do_epoch(
            peft_model,
            tokenizer,
            dataset=dataset,
            batch_size=batch_size,
            log_path=log_path,
            optimizer=optimizer,
            do_step=True,
            max_traj_len=max_traj_len,
        )
        avg_train_kl = sum(train_kls) / len(train_kls)
        log(f"Train KL divergence (epoch={epoch}): {avg_train_kl}", log_path)
        log(f"Done train epoch {epoch}", log_path)

        log(f"Started validation epoch {epoch}", log_path)
        val_kls = do_epoch(
            peft_model,
            tokenizer,
            dataset=val_dataset,
            batch_size=batch_size,
            log_path=log_path,
            optimizer=optimizer,
            do_step=False,
            max_traj_len=max_traj_len,
        )
        avg_val_kl = sum(val_kls) / len(val_kls)
        log(f"Validation KL divergence (epoch={epoch}): {avg_val_kl}", log_path)
        log(f"Done validation epoch {epoch}", log_path)

        # periodic checkpoint: epoch_<epoch>
        if (epoch + 1) % save_every == 0:
            out_epoch_dir = os.path.join(out_dir, f"epoch_{epoch}")
            os.makedirs(out_epoch_dir, exist_ok=True)
            log(f"Saving PEFT model to {out_epoch_dir}...", log_path)
            peft_model.save_pretrained(out_epoch_dir)
            log(f"Model saved to {out_epoch_dir}", log_path)

        # best checkpoint: overwrite top-level out_dir adapter
        if avg_val_kl < best_val_loss:
            best_val_loss = avg_val_kl
            log(f"Saving best PEFT model to {out_dir}...", log_path)
            peft_model.save_pretrained(out_dir)
            log(f"Best model saved to {out_dir}", log_path)

    print(f"[resume] Done. Finished through epoch_{target_num_epochs - 1}.")
    log(f"[resume] Done. Finished through epoch_{target_num_epochs - 1}.", log_path)
