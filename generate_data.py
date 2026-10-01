"""Samples replies from the prompted model and saves each one after both prompts."""

import argparse
import json
from pathlib import Path

import torch
from datasets import load_dataset
from tqdm import tqdm

from train_utils import (
    DEFAULT_MODEL,
    default_device,
    load_model,
    load_tokenizer,
    resolve_dtype,
    set_seed,
    stop_ids,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--x0_file", required=True, help="The system prompt to bake, as a text file."
    )
    parser.add_argument("--question_dataset", default="data/squad_train.jsonl")
    parser.add_argument("--num_questions", type=int, default=100)
    parser.add_argument("--num_sequences_per_question", type=int, default=25)
    parser.add_argument(
        "--max_sequence_length", type=int, default=300, help="Most tokens to generate per reply."
    )
    parser.add_argument(
        "--min_sequence_length", type=int, default=100, help="Fewest tokens to generate per reply."
    )
    # Qwen3.5's card asks for temperature 1.0 and top-k 20 on plain text. It ships no
    # generation config, so without these it samples from the top 50 at whatever we pass.
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument(
        "--top_k", type=int, default=20, help="Sample from this many likeliest tokens."
    )
    parser.add_argument("--batch_size", type=int, default=38, help="Replies per generate() call.")
    parser.add_argument("--model_name", default=DEFAULT_MODEL)
    parser.add_argument("--traj_out_file", default="data/traj_lex.jsonl")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def format_prompt(tokenizer, question, system_prompt=None):
    """The chat template's text for one user turn, ending where the assistant's reply starts."""
    messages = [{"role": "user", "content": question}]
    if system_prompt is not None:
        messages.insert(0, {"role": "system", "content": system_prompt})
    # Qwen3.5 from 4B up thinks by default, and the thinking would use up the reply's token budget
    # before it gets to an answer. Templates without the switch just ignore it.
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )


def trajectory_row(prompt_ids, nosys_ids, generated, stop_ids):
    """Ids and generated-token masks for one reply after both prompts, or None if it's empty."""
    # generate() pads finished replies out to the longest one in the batch, so everything after
    # the first stop token is padding.
    n = next((k for k, tok in enumerate(generated) if tok in stop_ids), len(generated))
    if n == 0:
        return None
    generated = generated[: n + 1]
    gen_mask = [1] * n + [0] * (len(generated) - n)
    return {
        "input_ids": prompt_ids + generated,
        "generated_text_mask": [0] * len(prompt_ids) + gen_mask,
        "input_ids_nosys": nosys_ids + generated,
        "generated_text_mask_nosys": [0] * len(nosys_ids) + gen_mask,
    }


def main():
    args = parse_args()
    set_seed(args.seed)
    x0 = Path(args.x0_file).read_text(encoding="utf-8")
    questions = load_dataset("json", data_files=args.question_dataset, split="train")
    questions = questions.shuffle(seed=args.seed)
    questions = questions.select(range(min(args.num_questions, len(questions))))["question"]

    device = default_device()
    tokenizer = load_tokenizer(args.model_name)
    model = load_model(args.model_name, device, resolve_dtype("auto", device))
    stop = stop_ids(model, tokenizer)

    written = skipped = 0
    with Path(args.traj_out_file).open("w", encoding="utf-8") as f:
        for question in tqdm(questions, desc="questions"):
            prompt = format_prompt(tokenizer, question, x0)
            prompt_nosys = format_prompt(tokenizer, question)
            prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
            nosys_ids = tokenizer(prompt_nosys, add_special_tokens=False)["input_ids"]
            ids = torch.tensor([prompt_ids], device=device)
            for start in range(0, args.num_sequences_per_question, args.batch_size):
                count = min(args.batch_size, args.num_sequences_per_question - start)
                with torch.no_grad():
                    out = model.generate(
                        ids,
                        attention_mask=torch.ones_like(ids),
                        do_sample=True,
                        num_return_sequences=count,
                        max_new_tokens=args.max_sequence_length,
                        min_new_tokens=args.min_sequence_length,
                        temperature=args.temperature,
                        top_k=args.top_k,
                        pad_token_id=tokenizer.pad_token_id,
                        eos_token_id=stop,
                    )
                for generated in out[:, len(prompt_ids) :].tolist():
                    row = trajectory_row(prompt_ids, nosys_ids, generated, set(stop))
                    if row is None:
                        skipped += 1
                        continue
                    row.update(
                        question=question,
                        text=tokenizer.decode(row["input_ids"]),
                        text_nosys=tokenizer.decode(row["input_ids_nosys"]),
                        prompt_text=prompt,
                        prompt_text_nosys=prompt_nosys,
                        prompt_input_ids=prompt_ids,
                        prompt_input_ids_nosys=nosys_ids,
                        attention_mask=[1] * len(prompt_ids),
                    )
                    f.write(json.dumps(row) + "\n")
                    written += 1

    print(f"Wrote {written} trajectories to {args.traj_out_file}.")
    if skipped:
        print(f"Left out {skipped} replies that came back empty.")


if __name__ == "__main__":
    main()
