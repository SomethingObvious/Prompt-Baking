"""Downloads SQuAD and the math word-problem sets into data/ as JSONL."""

import json
import urllib.request

# The XML comes from the fixed ASDiv URL below, not from anything a user passes in.
import xml.etree.ElementTree as ET  # nosec B405
from pathlib import Path

from datasets import load_dataset

DATA_DIR = Path("data")
SVAMP_URL = "https://raw.githubusercontent.com/arkilpatel/SVAMP/main/SVAMP.json"
ASDIV_URL = "https://raw.githubusercontent.com/chaochun/nlu-asdiv-dataset/master/dataset/ASDiv.xml"


def write_jsonl(path, rows):
    with Path(path).open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def fetch(url):
    with urllib.request.urlopen(url, timeout=60) as response:  # noqa: S310
        return response.read()


def text_of(elem, tag):
    child = elem.find(tag)
    return (child.text or "") if child is not None else ""


def main():
    DATA_DIR.mkdir(exist_ok=True)

    print("Downloading SQuAD")
    squad = load_dataset("rajpurkar/squad")
    write_jsonl(DATA_DIR / "squad_train.jsonl", squad["train"])
    write_jsonl(DATA_DIR / "squad_validation.jsonl", squad["validation"])

    print("Downloading GSM8K")
    gsm8k = load_dataset("openai/gsm8k", "main")
    write_jsonl(DATA_DIR / "gsm8k_train.jsonl", gsm8k["train"])
    write_jsonl(DATA_DIR / "gsm8k_validation.jsonl", gsm8k["test"])

    # SVAMP and ASDiv have no splits of their own, so the first 80% is train.
    print("Downloading SVAMP")
    svamp = [
        {
            "id": ex["ID"],
            "question": ex["Body"] + " " + ex["Question"],
            "equation": ex["Equation"],
            "answer": ex["Answer"],
            "type": ex["Type"],
        }
        for ex in json.loads(fetch(SVAMP_URL))
    ]
    cut = int(len(svamp) * 0.8)
    write_jsonl(DATA_DIR / "svamp_train.jsonl", svamp[:cut])
    write_jsonl(DATA_DIR / "svamp_validation.jsonl", svamp[cut:])

    print("Downloading ASDiv")
    root = ET.fromstring(fetch(ASDIV_URL))  # noqa: S314 # nosec B314
    asdiv = [
        {
            "id": p.get("ID"),
            "grade": p.get("Grade"),
            "source": p.get("Source"),
            "body": text_of(p, "Body"),
            "question": text_of(p, "Body") + " " + text_of(p, "Question"),
            "solution_type": text_of(p, "Solution-Type"),
            "answer": text_of(p, "Answer"),
            "formula": text_of(p, "Formula"),
        }
        for p in root.iter("Problem")
    ]
    cut = int(len(asdiv) * 0.8)
    write_jsonl(DATA_DIR / "asdiv_train.jsonl", asdiv[:cut])
    write_jsonl(DATA_DIR / "asdiv_validation.jsonl", asdiv[cut:])
    print(f"Everything is in {DATA_DIR}/.")


if __name__ == "__main__":
    main()
