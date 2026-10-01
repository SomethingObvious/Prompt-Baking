"""Trains a LoRA adapter so the model without the system prompt acts like it has it."""

import argparse
import json
from pathlib import Path

from train_utils import add_training_args, train


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out_dir", default="results/traj_lex_01")
    add_training_args(parser)
    args = parser.parse_args()
    if args.data_path is None or args.val_path is None:
        parser.error("both --data_path and --val_path are needed, and generate_data.py makes them")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "args.json").write_text(json.dumps(vars(args), indent=4), encoding="utf-8")
    train(args)


if __name__ == "__main__":
    main()
