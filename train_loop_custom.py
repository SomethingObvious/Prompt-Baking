import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from peft import get_peft_model, LoraConfig, TaskType
from datetime import datetime
import argparse
from tqdm import tqdm


# ----------------- logging helper -----------------
def log(msg, file_path):
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(file_path, "a") as f:
        f.write(f"[{current_time}] {msg}\n")


# ----------------- padding utilities -----------------
def pad_list_of_lists(llist, pad_tok_val, verbose=False, pad_side="right", return_pad_mask=False):
    """
    Pads a list of lists with a padding token value.
    Right padding by default.
    If return_pad_mask == True, return 0 where padded and 1 where original.
    """
    assert pad_side in ["left", "right"], "pad_side must be either 'left' or 'right'"

    max_len = max(len(l) for l in llist)
    if pad_side == "right":
        padded_list = [l + [pad_tok_val] * (max_len - len(l)) for l in llist]
    else:  # left
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
    """
    Crop trajectories to at most max_traj_len tokens of generated text,
    based on mask_list which has 1s on generated tokens.
    """
    assert max_traj_len > 0, "max_traj_len must be greater than 0 in crop_trajectories()"

    new_input_ids_list = []
    new_mask_list = []

    for input_ids, mask in zip(input_ids_list, mask_list):
        assert len(input_ids) == len(mask), "input_ids and mask must be same length"
        num_ones = sum(mask)
        # first 1 indicates where the generated trajectory starts (after prompt)
        first_1_ind = mask.index(1)
        cut_idx = first_1_ind + max_traj_len
        input_ids_ = input_ids[:cut_idx]
        mask_ = mask[:cut_idx]
        new_input_ids_list.append(input_ids_)
        new_mask_list.append(mask_)

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
    """
    Train or evaluate for one epoch on the given dataset.
    """
    kl_divs = []
    device = peft_model.device  # single device for everything

    for i in tqdm(range(0, len(dataset["train"]), batch_size)):
        log(f"Batch {i}", log_path)
        batch = dataset["train"][i : i + batch_size]

        input_ids_nosys_list_ = batch["input_ids_nosys"]
        input_ids_list_ = batch["input_ids"]

        mask_nosys_list_ = batch["generated_text_mask_nosys"]
        mask_list_ = batch["generated_text_mask"]

        if max_traj_len > 0:
            input_ids_list_, mask_list_ = crop_trajectories(
                input_ids_list_, mask_list_, max_traj_len
            )
            input_ids_nosys_list_, mask_nosys_list_ = crop_trajectories(
                input_ids_nosys_list_, mask_nosys_list_, max_traj_len
            )

        # pad
        input_ids_nosys_list = pad_list_of_lists(
            input_ids_nosys_list_, tokenizer.pad_token_id, verbose=False
        )
        input_ids_list = pad_list_of_lists(
            input_ids_list_, tokenizer.pad_token_id, verbose=False
        )
        mask_nosys_list = pad_list_of_lists(mask_nosys_list_, 0, verbose=False)
        mask_list = pad_list_of_lists(mask_list_, 0, verbose=False)

        input_ids = torch.tensor(input_ids_list, device=device)
        input_ids_nosys = torch.tensor(input_ids_nosys_list, device=device)
        mask = torch.tensor(mask_list, device=device) == 1
        mask_nosys = torch.tensor(mask_nosys_list, device=device) == 1

        assert input_ids.shape == mask.shape
        assert input_ids_nosys.shape == mask_nosys.shape
        assert (
            input_ids[mask] != input_ids_nosys[mask_nosys]
        ).sum() == 0, "Prompted and unprompted ids must match on generated region"
        assert (mask.sum(dim=1) == mask_nosys.sum(dim=1)).all(), "Masks must have same counts per row"

        # unprompted logits (trainable, with adapters)
        log("Computing unprompted logits...", log_path)
        unprompted_logits_ = peft_model(input_ids_nosys).logits
        log("Done computing unprompted logits...", log_path)

        # prompted logits (frozen base model only, adapters disabled)
        log("Computing prompted logits...", log_path)
        with peft_model.disable_adapter():
            with torch.no_grad():
                prompted_logits_ = peft_model(input_ids).logits
        log("Done computing prompted logits...", log_path)

        unprompted_logits = unprompted_logits_[mask_nosys, :]
        prompted_logits = prompted_logits_[mask, :]

        kl_div_ = F.kl_div(
            F.log_softmax(unprompted_logits, dim=-1),
            F.log_softmax(prompted_logits, dim=-1),
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

    # 'epoch' is passed from outer scope; just print average
    print(f"Epoch loss: {sum(kl_divs) / len(kl_divs)}")
    log(f"Epoch loss: {sum(kl_divs) / len(kl_divs)}", log_path)
    return kl_divs


# ----------------- main -----------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prompt Baking LoRA training loop")

    parser.add_argument("--num_epochs", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=10)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="data/traj_lex_nseq1000_maxlen300_minlen100_temp2.0.jsonl",
    )
    parser.add_argument(
        "--val_path",
        type=str,
        default="data/traj_lex_nseq1000_maxlen300_minlen100_temp2.0.jsonl",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="results/traj_lex_01",
    )
    parser.add_argument(
        "-r",
        type=int,
        default=32,
        help="LoRA rank",
    )
    parser.add_argument(
        "--save_every",
        type=int,
        default=1,
        help="Save model every n epochs",
    )
    parser.add_argument(
        "--max_traj_len",
        type=int,
        default=-1,
        help="Max usable trajectory length; -1 = full",
    )

    args = parser.parse_args()

    num_epochs = args.num_epochs
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    device_str = args.device
    data_path = args.data_path
    val_path = args.val_path
    out_dir = args.out_dir

    os.makedirs(out_dir, exist_ok=True)
    log_path = os.path.join(out_dir, "train_loop.log")

    # save args
    with open(os.path.join(out_dir, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=4)
    print(f"Arguments saved to {os.path.join(out_dir, 'args.json')}")

    # set device
    device = torch.device(device_str)

    # load datasets
    log("Loading dataset", log_path)
    dataset = load_dataset("json", data_files=data_path)
    log("Dataset loaded", log_path)

    log("Loading validation dataset", log_path)
    val_dataset = load_dataset("json", data_files=val_path)
    log("Validation dataset loaded", log_path)

    # load base model and tokenizer on a single device (no device_map='auto')
    model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
    log(f"Loading model {model_name} on {device_str}...", log_path)

    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
    )
    base_model.to(device)
    log("Base model loaded", log_path)

    # setup LoRA
    log("Loading PEFT model (LoRA)...", log_path)
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=args.r,
        lora_alpha=128,
        lora_dropout=0.0,
        bias="none",
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    )

    peft_model = get_peft_model(base_model, peft_config)
    peft_model.to(device)
    print("unprompted model parameter stats:")
    peft_model.print_trainable_parameters()
    log("PEFT model loaded", log_path)

    optimizer = torch.optim.Adam(peft_model.parameters(), lr=learning_rate)

    best_val_loss = float("inf")

    for epoch in range(num_epochs):
        log(f"Started epoch {epoch}", log_path)
        train_kls = do_epoch(
            peft_model,
            tokenizer,
            dataset=dataset,
            batch_size=batch_size,
            log_path=log_path,
            optimizer=optimizer,
            do_step=True,
            max_traj_len=args.max_traj_len,
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
            max_traj_len=args.max_traj_len,
        )
        avg_val_kl = sum(val_kls) / len(val_kls)
        log(f"Validation KL divergence (epoch={epoch}): {avg_val_kl}", log_path)
        log(f"Done validation epoch {epoch}", log_path)

        # periodic checkpoints
        if (epoch + 1) % args.save_every == 0:
            out_epoch_dir = os.path.join(out_dir, f"epoch_{epoch}")
            os.makedirs(out_epoch_dir, exist_ok=True)
            log(f"Saving PEFT model to {out_epoch_dir}...", log_path)
            peft_model.save_pretrained(out_epoch_dir)
            log(f"Model saved to {out_epoch_dir}", log_path)

        # best checkpoint
        if avg_val_kl < best_val_loss:
            best_val_loss = avg_val_kl
            log(f"Saving best PEFT model to {out_dir}...", log_path)
            peft_model.save_pretrained(out_dir)
            log(f"Best model saved to {out_dir}", log_path)
