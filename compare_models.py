"""Compares the NLL of generated tokens for the base and baked models, prompted or not."""

import argparse
import contextlib
import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import torch
import torch.nn.functional as F  # noqa: N812
from datasets import load_dataset
from tqdm import tqdm

from generate_data import format_prompt
from train_utils import load_baked, log, pad_list_of_lists


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results_dir",
        required=True,
        help="An adapter folder, so a run's --out_dir or one of its epoch_N folders.",
    )
    parser.add_argument(
        "--data_file",
        default="NONE",
        help="Trajectories to score. The default is the run's val_path from args.json.",
    )
    parser.add_argument(
        "--u_override",
        help="A text file with a different system prompt to put in the prompted rows.",
    )
    parser.add_argument(
        "--path_prefix", help="Prefix for the output files. It defaults to the data file's name."
    )
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--model_name", help="Base model. It defaults to the adapter's own.")
    parser.add_argument(
        "--device", help="Something like cuda:0 or cpu. It defaults to cuda if there is one."
    )
    parser.add_argument(
        "--dtype",
        default="auto",
        choices=["auto", "bf16", "fp16", "fp32"],
        help="The default picks bf16 on GPUs that have it, fp16 on older ones and fp32 on CPU.",
    )
    return parser.parse_args()


def find_args_json(start_dir):
    """The args.json in start_dir or the nearest folder above it, as epoch_N folders lack one."""
    d = Path(start_dir).resolve()
    for candidate in (d, *d.parents):
        if (candidate / "args.json").is_file():
            return candidate / "args.json"
    raise FileNotFoundError(f"There's no args.json in {start_dir} or any folder above it.")


def with_system_prompt(batch, tokenizer, system_prompt):
    """The prompted ids and masks of batch, with each prompt rebuilt around system_prompt."""
    if "question" not in batch:
        raise ValueError("--u_override needs the question on each row. Regenerate the data file.")
    ids_list, mask_list = [], []
    for question, ids, mask, old_prompt in zip(
        batch["question"],
        batch["input_ids"],
        batch["generated_text_mask"],
        batch["prompt_input_ids"],
        strict=True,
    ):
        text = format_prompt(tokenizer, question, system_prompt)
        prompt = tokenizer(text, add_special_tokens=False)["input_ids"]
        ids_list.append(prompt + ids[len(old_prompt) :])
        mask_list.append([0] * len(prompt) + mask[len(old_prompt) :])
    return ids_list, mask_list


def mean_nll(model, ids, mask, adapter):
    """Each row's mean NLL over its generated tokens."""
    # The logits from the token before the first generated one onward are all that's needed.
    first = int(mask.int().argmax(dim=1).min())
    off = contextlib.nullcontext() if adapter else model.disable_adapter()
    with off:
        keep = ids.shape[1] - first + 1
        logits = model(input_ids=ids, logits_to_keep=keep, use_cache=False).logits[:, :-1]
    nll = F.cross_entropy(logits.float().transpose(1, 2), ids[:, first:], reduction="none")
    m = mask[:, first:]
    return (nll * m).sum(dim=1) / m.sum(dim=1).clamp(min=1)


@torch.no_grad()
def get_losses(model, tokenizer, dataset, batch_size, u_override=None):
    u_str = Path(u_override).read_text(encoding="utf-8") if u_override else None
    res = {
        "unprompted_logits_peft_losses": [],
        "unprompted_logits_base_losses": [],
        "prompted_logits_peft_losses": [],
        "prompted_logits_base_losses": [],
        "lengths": [],
        "texts": [],
    }
    for i in tqdm(range(0, len(dataset), batch_size), desc="compare"):
        batch = dataset[i : i + batch_size]
        ids_list, mask_list = batch["input_ids"], batch["generated_text_mask"]
        if u_str is not None:
            ids_list, mask_list = with_system_prompt(batch, tokenizer, u_str)
        sides = {
            "unprompted": (batch["input_ids_nosys"], batch["generated_text_mask_nosys"]),
            "prompted": (ids_list, mask_list),
        }
        for side, (ids, mask) in sides.items():
            ids = torch.tensor(pad_list_of_lists(ids, 0), device=model.device)
            mask = torch.tensor(pad_list_of_lists(mask, 0), device=model.device, dtype=torch.bool)
            for kind, adapter in (("peft", True), ("base", False)):
                nll = mean_nll(model, ids, mask, adapter)
                res[f"{side}_logits_{kind}_losses"] += nll.cpu().tolist()
        res["texts"] += batch["text"]
        res["lengths"] += [sum(m) for m in mask_list]
    return res


def draw_graphs(res, results_dir, tag):
    df = pd.DataFrame(res)
    pairs = [
        ("unprompted_logits_peft_losses", "prompted_logits_base_losses", "up_peft_vs_p_base"),
        ("prompted_logits_peft_losses", "prompted_logits_base_losses", "p_peft_vs_p_base"),
        ("unprompted_logits_peft_losses", "unprompted_logits_base_losses", "up_peft_vs_up_base"),
        ("prompted_logits_peft_losses", "unprompted_logits_base_losses", "p_peft_vs_up_base"),
        ("prompted_logits_peft_losses", "unprompted_logits_peft_losses", "p_peft_vs_up_peft"),
        ("prompted_logits_base_losses", "unprompted_logits_base_losses", "p_base_vs_up_base"),
    ]
    for x, y, name in pairs:
        fig = px.scatter(df, x=x, y=y, color="lengths")
        fig.update_layout(title=f"{name} -- {tag}")
        fig.update_xaxes(range=[0, 8])
        fig.update_yaxes(range=[0, 8])
        fig.write_html(str(Path(results_dir) / f"{tag}_{name}.html"))


def main():
    args = parse_args()
    results_dir = Path(args.results_dir)
    log_path = results_dir / "compare_models.log"

    data_file = args.data_file
    if data_file == "NONE":
        data_file = json.loads(find_args_json(results_dir).read_text(encoding="utf-8"))["val_path"]
    dataset = load_dataset("json", data_files=data_file, split="train")
    tag = args.path_prefix or Path(data_file).stem
    if args.u_override and not args.path_prefix:
        tag += "_u_override_" + Path(args.u_override).stem
    log(f"Scoring {data_file} with the adapter in {results_dir}", log_path)

    model, tokenizer = load_baked(results_dir, args.model_name, args.device, args.dtype)
    res = get_losses(model, tokenizer, dataset, args.batch_size, args.u_override)
    res_path = results_dir / f"{tag}_compare_models_results.json"
    res_path.write_text(json.dumps(res), encoding="utf-8")
    draw_graphs(res, results_dir, tag)
    log(f"Wrote {res_path} and the plots next to it", log_path)


if __name__ == "__main__":
    main()
