"""Picks a run back up from its newest epoch_N folder. Any flag you pass overrides its args.json."""

import argparse
import json
from pathlib import Path

from train_utils import add_training_args, list_epoch_dirs, train


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out_dir", required=True, help="The run folder with args.json in it.")
    parser.add_argument(
        "--extend_epochs", type=int, default=0, help="Epochs to train past the run's num_epochs."
    )
    add_training_args(parser)
    args_path = Path(parser.parse_known_args()[0].out_dir) / "args.json"
    if not args_path.is_file():
        parser.error(f"there's no {args_path}, so this doesn't look like a run folder")
    parser.set_defaults(**json.loads(args_path.read_text(encoding="utf-8")))
    args = parser.parse_args()
    args.num_epochs += args.extend_epochs

    epochs = list_epoch_dirs(args.out_dir)
    start, adapter_dir = (epochs[-1][0] + 1, epochs[-1][1]) if epochs else (0, None)
    if start >= args.num_epochs:
        print(f"The run already has all {args.num_epochs} epochs. Pass --extend_epochs for more.")
        return
    if adapter_dir is None:
        print("There are no epoch_N folders yet, so this starts from a fresh adapter.")
    else:
        print(f"Carrying on from {adapter_dir} at epoch {start}.")

    # The new num_epochs goes into args.json so the next resume aims for the same end.
    saved = {k: v for k, v in vars(args).items() if k != "extend_epochs"}
    args_path.write_text(json.dumps(saved, indent=4), encoding="utf-8")
    train(args, start, adapter_dir)


if __name__ == "__main__":
    main()
