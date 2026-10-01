"""Scores chain-of-thought accuracy on asdiv, gsm8k or svamp for the base and baked models."""

import argparse
import json
from pathlib import Path

from tqdm import tqdm

from generate_data import format_prompt
from train_utils import generate, load_baked, load_questions, pick_epoch

DATASETS = ["asdiv", "gsm8k", "svamp"]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results_dir", required=True)
    parser.add_argument(
        "--model_epoch", default="last", help="Which epoch_N to load, as a number or last."
    )
    parser.add_argument("--model_name", help="Base model. It defaults to the adapter's own.")
    parser.add_argument("--num_questions", type=int, default=50)
    parser.add_argument("--include_train", default="False")
    parser.add_argument("--include_val", default="True")
    parser.add_argument("--include_test", default="False")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument(
        "--pre_question_str", default="", help='Text to put before each question, like "Q: ".'
    )
    parser.add_argument(
        "--pre_answer_str", default="A:", help="Text the assistant's reply starts with."
    )
    parser.add_argument(
        "--u_file", required=True, help="The chain-of-thought prompt that was baked."
    )
    parser.add_argument(
        "--dataset",
        default="none",
        choices=["none", *DATASETS],
        help="The default takes it from the run folder's name.",
    )
    parser.add_argument(
        "--temperature", type=float, default=0.0, help="The default of 0 decodes greedily."
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def reformat_answer_string(answer, dataset):
    """The number a correct reply has to give for one row of dataset's answer column."""
    answer = str(answer).replace(",", "")
    if dataset == "asdiv":
        return answer.split(" ")[0]  # answers look like "9 (apples)"
    if dataset == "gsm8k":
        return answer.split(" ")[-1]  # the worked solution ends in "#### 5"
    if dataset == "svamp":
        value = float(answer)  # stored as floats, like 51.0
        return str(int(value)) if value.is_integer() else str(value)
    raise ValueError(f"There's no answer format for {dataset}.")


def assess_correct(generate_str_list, correct_answer_list, dataset_str):
    """Hits for "The answer is N." anywhere, for N anywhere and for N in the last sentence."""
    exact, anywhere, last_sentence = [], [], []
    for reply, answer in zip(generate_str_list, correct_answer_list, strict=True):
        reply = reply.replace(",", "")
        correct = reformat_answer_string(answer, dataset_str)
        sentences = (reply + " ").split(".")
        last = sentences[-2] if len(sentences) > 1 else sentences[0]
        exact.append(f"The answer is {correct}." in reply)
        anywhere.append(correct in reply)
        last_sentence.append(correct in last)
    return exact, anywhere, last_sentence


def main():
    args = parse_args()
    dataset_name = args.dataset
    if dataset_name == "none":
        dataset_name = next((d for d in DATASETS if d in args.results_dir), None)
        if dataset_name is None:
            raise ValueError(f"Couldn't tell the dataset from {args.results_dir}. Pass --dataset.")
    epoch_num, epoch_dir = pick_epoch(args.results_dir, args.model_epoch)
    dataset = load_questions(
        dataset_name, args.include_train, args.include_val, args.include_test, args.seed
    )
    model, tokenizer = load_baked(epoch_dir, args.model_name)
    u_str = Path(args.u_file).read_text(encoding="utf-8")
    if args.temperature > 0:
        sampling = {"do_sample": True, "temperature": args.temperature}
    else:
        sampling = {"do_sample": False, "temperature": None, "top_p": None}

    # base is the prompted model with the adapter off, peft is the baked model with no prompt,
    # and peft_sys is the baked model with the prompt as well.
    runs = {"base": ("sys", False), "peft": ("nosys", True), "peft_sys": ("sys", True)}
    hits = {kind: [[], [], []] for kind in runs}
    num_tests = min(args.num_questions, len(dataset))
    for i in tqdm(range(0, num_tests, args.batch_size), desc="math-eval"):
        batch = dataset[i : min(i + args.batch_size, num_tests)]
        questions = [args.pre_question_str + q for q in batch["question"]]
        prompts = {
            "sys": [format_prompt(tokenizer, q, u_str) + args.pre_answer_str for q in questions],
            "nosys": [format_prompt(tokenizer, q) + args.pre_answer_str for q in questions],
        }
        for kind, (prompt_kind, adapter) in runs.items():
            replies = generate(
                model, tokenizer, prompts[prompt_kind], adapter, max_new_tokens=300, **sampling
            )
            scored = assess_correct(replies, batch["answer"], dataset_name)
            for total, new in zip(hits[kind], scored, strict=True):
                total.extend(new)

    results = {"args": vars(args)}
    for kind, (exact, anywhere, last) in hits.items():
        results[f"mean_accuracy_{kind}"] = sum(exact) / num_tests
        results[f"mean_accuracy_upper_bound_{kind}"] = sum(anywhere) / num_tests
        results[f"mean_accuracy_last_sentence_{kind}"] = sum(last) / num_tests
    name = (
        f"mathtest_ep{epoch_num}_numq{args.num_questions}"
        f"_gentemp{args.temperature}_datasetname{dataset_name}.json"
    )
    (Path(args.results_dir) / name).write_text(json.dumps(results), encoding="utf-8")


if __name__ == "__main__":
    main()
