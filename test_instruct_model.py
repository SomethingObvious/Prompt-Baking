"""Scores how well a baked adapter follows a data/InstructionX0 prompt on SQuAD questions."""

import argparse
import functools
import json
import re
import string
import urllib.request
from pathlib import Path

import nltk
import numpy as np
from langdetect import DetectorFactory, LangDetectException, detect_langs
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from sympy import isprime
from tqdm import tqdm

from generate_data import format_prompt
from train_utils import generate, load_baked, load_questions, pick_epoch

COMMON_WORDS_URL = (
    "https://raw.githubusercontent.com/first20hours/google-10000-english/master/"
    "google-10000-english-usa.txt"
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results_dir", required=True)
    parser.add_argument(
        "--model_epoch", default="last", help="Which epoch_N to load, as a number or last."
    )
    parser.add_argument("--model_name", help="Base model. It defaults to the adapter's own.")
    parser.add_argument("--num_questions", type=int, help="It defaults to every question.")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--include_train", default="False")
    parser.add_argument("--include_val", default="True")
    parser.add_argument("--include_test", default="False")
    parser.add_argument(
        "--u_file",
        default="model_default",
        help="The prompt that was baked. The default guesses it from the run folder's name.",
    )
    parser.add_argument("--dataset", default="squad", choices=["squad"])
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def french_score(question, reply):
    DetectorFactory.seed = 0  # langdetect is random unless it's seeded
    try:
        langs = detect_langs(reply)
    except LangDetectException:  # raised when the reply has no letters in it
        return 0.0
    return next((lang.prob for lang in langs if lang.lang == "fr"), 0.0)


@functools.cache
def vader():
    try:
        return SentimentIntensityAnalyzer()
    except LookupError:
        nltk.download("vader_lexicon", quiet=True)
        return SentimentIntensityAnalyzer()


def sad_score(question, reply):
    return vader().polarity_scores(reply)["neg"]


def starts_with_a_score(question, reply):
    return float(reply.lower().startswith("a"))


def prime_capital_score(question, reply):
    """Mean of the share of prime-numbered words in capitals and of the rest in lowercase."""
    # Word 1 counts as prime here, since the prompt says to start from the first word.
    prime, other = [], []
    for n, word in enumerate(reply.split(" "), start=1):
        if n == 1 or isprime(n):
            prime.append(word.isupper())
        else:
            other.append(word.islower())
    shares = [float(np.mean(group)) for group in (prime, other) if group]
    return float(np.mean(shares)) if shares else 0.0


def second_capital_score(question, reply):
    words = reply.split(" ")
    right = [w.isupper() if n % 2 == 0 else w.islower() for n, w in enumerate(words)]
    return float(np.mean(right))


def blue_score(question, reply):
    """Share of sentences that say blue exactly once."""
    return float(np.mean([s.lower().count("blue") == 1 for s in reply.split(". ")]))


def no_e_score(question, reply):
    return 1 / (1 + reply.lower().count("e"))


def reversed_score(question, reply):
    """BLEU of the reply against the question with its words in reverse order."""
    question = re.sub(r"[^\w\s]", "", question).lower()
    reply = re.sub(r"[^\w\s]", "", reply).lower()
    smoothing = SmoothingFunction().method4
    return float(
        sentence_bleu([question.split()[::-1]], reply.split(), smoothing_function=smoothing)
    )


@functools.cache
def common_words():
    with urllib.request.urlopen(COMMON_WORDS_URL, timeout=60) as response:
        return frozenset(response.read().decode("utf-8").split())


def rare_lexicon_score(question, reply):
    """Share of the reply's words that aren't in the 10,000 most common English ones."""
    words = reply.translate(str.maketrans("", "", string.punctuation)).lower().split()
    if not words:
        return 0.0
    return 1 - sum(w in common_words() for w in words) / len(words)


# Keyed by the prompt's file name in data/InstructionX0.
SCORERS = {
    "always_french": french_score,
    "always_sad": sad_score,
    "always_start_with_A": starts_with_a_score,
    "every_prime_capital": prime_capital_score,
    "every_second_capital": second_capital_score,
    "every_sentence_blue": blue_score,
    "never_user_e": no_e_score,
    "reverse_input": reversed_score,
    "use_rare_lexicon": rare_lexicon_score,
}


def scorer_for(u_file):
    name = Path(u_file).name
    for key, scorer in SCORERS.items():
        if key in name:
            return scorer
    raise ValueError(f"There's no scorer for {u_file}. Its name needs one of {', '.join(SCORERS)}.")


def main():
    args = parse_args()
    _, epoch_dir = pick_epoch(args.results_dir, args.model_epoch)
    if args.u_file == "model_default":
        run_name = Path(args.results_dir).resolve().name
        args.u_file = f"data/InstructionX0/{run_name.split('_x0_')[0]}_x0.md"
    score = scorer_for(args.u_file)
    u_str = Path(args.u_file).read_text(encoding="utf-8")
    print(f"Scoring {epoch_dir} against {args.u_file}")

    dataset = load_questions(
        args.dataset, args.include_train, args.include_val, args.include_test, args.seed
    )
    args.num_questions = min(args.num_questions or len(dataset), len(dataset))
    model, tokenizer = load_baked(epoch_dir, args.model_name)

    scores = {"base_nosys": [], "base_sys": [], "peft_nosys": [], "peft_sys": []}
    for i in tqdm(range(0, args.num_questions, args.batch_size), desc="instruct-eval"):
        questions = dataset[i : min(i + args.batch_size, args.num_questions)]["question"]
        prompts = {
            "sys": [format_prompt(tokenizer, q, u_str) for q in questions],
            "nosys": [format_prompt(tokenizer, q) for q in questions],
        }
        for key, found in scores.items():
            kind, prompt_kind = key.split("_")
            replies = generate(
                model,
                tokenizer,
                prompts[prompt_kind],
                adapter=kind == "peft",
                do_sample=False,
                max_new_tokens=300,
                temperature=None,
                top_p=None,
            )
            found.extend(score(q, r) for q, r in zip(questions, replies, strict=True))
        means = ", ".join(f"{k} {np.mean(v):.3f}" for k, v in scores.items())
        tqdm.write(f"Means after {len(scores['base_sys'])} questions are {means}")

    results = {"args": vars(args)}
    results |= {f"mean_eval_{k}": float(np.mean(v)) for k, v in scores.items()}
    results |= {f"std_eval_{k}": float(np.std(v)) for k, v in scores.items()}
    results |= {f"eval_{k}": v for k, v in scores.items()}
    name = f"dataset_{args.dataset}_epoch_{args.model_epoch}_numquestions_{args.num_questions}.json"
    (Path(args.results_dir) / name).write_text(json.dumps(results), encoding="utf-8")


if __name__ == "__main__":
    main()
