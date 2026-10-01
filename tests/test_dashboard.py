import json
from pathlib import Path

import dashboard


def write_run(folder: Path, kl_lines: list[str]) -> None:
    folder.mkdir(parents=True)
    args = {
        "model_name": "Qwen/Qwen3.5-2B",
        "data_path": "data/traj_blue_train.jsonl",
        "r": 32,
        "learning_rate": 0.0002,
        "num_epochs": 50,
    }
    (folder / "args.json").write_text(json.dumps(args), encoding="utf-8")
    (folder / "train_loop.log").write_text("\n".join(kl_lines) + "\n", encoding="utf-8")


def test_a_resumed_epoch_takes_its_last_logged_kl(tmp_path):
    write_run(
        tmp_path / "blue",
        [
            "[2026-10-01 12:00:00] Epoch 0 has a train KL of 0.5000 and a validation KL of 0.6000",
            "[2026-10-01 12:05:00] Epoch 1 has a train KL of 0.3000 and a validation KL of 0.4000",
            "[2026-10-01 12:06:00] Saved epoch 1 to results/blue as the best so far",
            "[2026-10-01 13:00:00] Epoch 1 has a train KL of 0.2500 and a validation KL of 0.3500",
        ],
    )
    assert dashboard.read_kl(tmp_path / "blue") == {0: (0.5, 0.6), 1: (0.25, 0.35)}


def test_every_run_and_score_lands_on_the_page(tmp_path):
    run = tmp_path / "sweep" / "blue"
    write_run(
        run,
        [
            "[2026-10-01 12:00:00] Epoch 0 has a train KL of 0.5000 and a validation KL of 0.6000",
            "[2026-10-01 12:05:00] Epoch 1 has a train KL of 0.0300 and a validation KL of 0.0420",
        ],
    )
    (run / "best.json").write_text(json.dumps({"epoch": 1, "val_kl": 0.042}), encoding="utf-8")
    instruct = {
        "args": {"u_file": "data/InstructionX0/every_sentence_blue_x0.md", "model_epoch": "last"},
        "epoch": 1,
        "mean_eval_base_nosys": 0.05,
        "mean_eval_base_sys": 0.81,
        "mean_eval_peft_nosys": 0.77,
        "mean_eval_peft_sys": 0.84,
    }
    (run / "dataset_squad_epoch_1_numquestions_20.json").write_text(
        json.dumps(instruct), encoding="utf-8"
    )
    math = {"args": {"dataset": "gsm8k"}, "mean_accuracy_base": 0.6, "mean_accuracy_peft": 0.55}
    (run / "mathtest_ep1_numq50_gentemp1.0_datasetnamegsm8k.json").write_text(
        json.dumps(math), encoding="utf-8"
    )
    write_run(tmp_path / "sweep" / "fresh", [])

    page = dashboard.build_html(tmp_path / "sweep")
    assert "2 runs in" in page
    assert "<td>blue</td>" in page
    assert "<td>fresh</td>" in page
    assert '<td class="num">2 of 50</td>' in page
    assert '<td class="num">0 of 50</td>' in page
    assert '<td class="num">0.0420</td>' in page
    assert "every_sentence_blue, epoch 1" in page
    # The epoch comes from the file name when the result doesn't carry one.
    assert "gsm8k, epoch 1" in page
    assert "No run has logged" not in page
    assert "No prompt-following scores yet" not in page
    assert "No math scores yet" not in page


def test_an_empty_folder_says_how_to_fill_it(tmp_path):
    page = dashboard.build_html(tmp_path)
    assert "There are no runs in" in page
    assert "No run has logged an epoch yet." in page
    assert "No math scores yet." in page
