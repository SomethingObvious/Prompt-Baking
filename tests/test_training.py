import argparse
import json
import random
from collections import Counter

import pytest
import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from peft.tuners.lora import LoraLayer
from transformers import AutoConfig, AutoModelForCausalLM

import train_loop_resume
from compare_models import with_system_prompt
from generate_data import format_prompt, trajectory_row
from train_utils import (
    LORA_TARGETS,
    add_training_args,
    batch_kl,
    crop_trajectories,
    do_epoch,
    kl_positions,
    list_epoch_dirs,
    load_model,
    load_tokenizer,
    train,
)

TINY = "trl-internal-testing/tiny-LlamaForCausalLM-3.2"
STOP = 128009
# The models the README suggests. Only their tokenizers get downloaded here.
SUGGESTED = ["Qwen/Qwen3.5-0.8B", "Qwen/Qwen3.5-2B", "Qwen/Qwen3.5-4B"]


@pytest.fixture(scope="module")
def tokenizer():
    return load_tokenizer(TINY)


@pytest.fixture
def model():
    torch.manual_seed(0)
    base = load_model(TINY, torch.device("cpu"), torch.float32)
    config = LoraConfig(
        task_type=TaskType.CAUSAL_LM, r=4, lora_alpha=16, target_modules=["q_proj", "v_proj"]
    )
    return get_peft_model(base, config)


def fake_rows(n, seed=0):
    """Rows shaped like generate_data.py's, with random ids and a prompted prompt that's longer."""
    rng = random.Random(seed)
    rows = []
    for _ in range(n):
        nosys = [rng.randrange(1000) for _ in range(rng.randint(3, 6))]
        prompt = [rng.randrange(1000) for _ in range(rng.randint(2, 5))] + nosys
        generated = [rng.randrange(1000) for _ in range(rng.randint(2, 8))] + [STOP]
        rows.append(trajectory_row(prompt, nosys, generated, {STOP}))
    return rows


def as_batch(rows):
    return {k: [r[k] for r in rows] for k in rows[0]}


def test_crop_drops_what_follows_the_last_generated_token():
    ids, mask = crop_trajectories([[5, 6, 7, 8, 9, 9]], [[0, 1, 1, 0, 0, 0]])
    assert ids == [[5, 6, 7]]
    assert mask == [[0, 1, 1]]


def test_crop_keeps_max_traj_len_generated_tokens():
    ids, mask = crop_trajectories(
        [[5, 6, 7, 8, 9], [0, 1, 2, 3]], [[0, 1, 1, 1, 1], [0, 0, 1, 1]], max_traj_len=2
    )
    assert ids == [[5, 6, 7], [0, 1, 2, 3]]
    assert mask == [[0, 1, 1], [0, 0, 1, 1]]


def test_crop_rejects_a_row_without_generated_tokens():
    with pytest.raises(ValueError, match="no generated tokens"):
        crop_trajectories([[1, 2]], [[0, 0]])


def test_kl_positions_start_at_the_last_prompt_token():
    mask = torch.tensor([[0, 0, 1, 1, 0], [0, 1, 1, 1, 1]], dtype=torch.bool)
    expected = torch.tensor([[0, 1, 1, 1, 0], [1, 1, 1, 1, 1]], dtype=torch.bool)
    assert torch.equal(kl_positions(mask), expected)


def test_trajectory_row_cuts_at_the_first_stop_token():
    row = trajectory_row([1, 2, 3], [3], [7, 8, STOP, STOP, STOP], {STOP})
    assert row == {
        "input_ids": [1, 2, 3, 7, 8, STOP],
        "generated_text_mask": [0, 0, 0, 1, 1, 0],
        "input_ids_nosys": [3, 7, 8, STOP],
        "generated_text_mask_nosys": [0, 1, 1, 0],
    }


def test_trajectory_row_keeps_a_truncated_reply():
    row = trajectory_row([1, 2], [2], [7, 8], {STOP})
    assert row["input_ids"] == [1, 2, 7, 8]
    assert row["generated_text_mask"] == [0, 0, 1, 1]


def test_trajectory_row_skips_an_empty_reply():
    assert trajectory_row([1, 2], [2], [STOP, STOP], {STOP}) is None


def test_unprompted_format_only_drops_the_system_prompt(tokenizer):
    prompted = format_prompt(tokenizer, "Why is the sky blue?", "Always answer in French.")
    unprompted = format_prompt(tokenizer, "Why is the sky blue?")
    assert "Always answer in French." not in unprompted
    assert prompted.replace("Always answer in French.", "") == unprompted
    assert unprompted.endswith("<|start_header_id|>assistant<|end_header_id|>\n\n")


@pytest.mark.parametrize("name", SUGGESTED)
def test_templates_add_a_system_turn_and_no_thinking(name):
    tokenizer = load_tokenizer(name)
    unprompted = format_prompt(tokenizer, "Why is the sky blue?")
    prompted = format_prompt(tokenizer, "Why is the sky blue?", "Always answer in French.")
    assert unprompted == (
        "<|im_start|>user\nWhy is the sky blue?<|im_end|>\n"
        "<|im_start|>assistant\n<think>\n\n</think>\n\n"
    )
    assert prompted == "<|im_start|>system\nAlways answer in French.<|im_end|>\n" + unprompted


def test_lora_reaches_the_deltanet_layers_of_qwen35():
    config = AutoConfig.for_model(
        "qwen3_5_text",
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=4,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=16,
        linear_num_key_heads=2,
        linear_num_value_heads=2,
        linear_key_head_dim=8,
        linear_value_head_dim=8,
    )
    model = AutoModelForCausalLM.from_config(config)
    model = get_peft_model(model, LoraConfig(r=4, target_modules=LORA_TARGETS))
    adapted = [n for n, m in model.named_modules() if isinstance(m, LoraLayer)]
    # Layers 0 to 2 are Gated DeltaNet and layer 3 is attention.
    per_layer = Counter(n.split(".layers.")[1].split(".")[0] for n in adapted)
    assert per_layer == {"0": 3, "1": 3, "2": 3, "3": 4}


def test_kl_is_zero_for_matching_prompts(model):
    rows = fake_rows(3)
    for r in rows:
        r["input_ids"], r["generated_text_mask"] = (
            r["input_ids_nosys"],
            r["generated_text_mask_nosys"],
        )
    assert batch_kl(model, as_batch(rows)).item() == 0.0


def test_kl_matches_a_direct_computation(model):
    torch.manual_seed(1)
    for layer in model.modules():
        if hasattr(layer, "lora_B"):
            torch.nn.init.normal_(layer.lora_B["default"].weight, std=0.5)
    rows = fake_rows(3)

    expected = 0.0
    for r in rows:
        with torch.no_grad(), model.disable_adapter():
            target = model(torch.tensor([r["input_ids"]])).logits[0]
        student = model(torch.tensor([r["input_ids_nosys"]])).logits[0]
        # Rows end in a masked stop token, so the matched positions run from the one before the
        # first generated token up to the one before the stop.
        n = sum(r["generated_text_mask"])
        p, q = len(r["input_ids"]) - n - 1, len(r["input_ids_nosys"]) - n - 1
        t, s = target[p - 1 : p + n].log_softmax(-1), student[q - 1 : q + n].log_softmax(-1)
        expected += (t.exp() * (t - s)).sum().item()

    assert batch_kl(model, as_batch(rows)).item() == pytest.approx(expected / 3, rel=1e-5)


def test_validation_leaves_no_gradients(model):
    do_epoch(model, Dataset.from_list(fake_rows(4)), batch_size=2)
    assert not model.training
    assert all(p.grad is None for p in model.parameters())


def test_short_last_batch_uses_its_own_size(model):
    rows = fake_rows(3)
    kls = do_epoch(model, Dataset.from_list(rows), batch_size=2)
    assert kls[1] == pytest.approx(batch_kl(model, as_batch(rows[2:])).item(), rel=1e-6)


def test_training_steps_lower_the_kl(model):
    rows = fake_rows(4)
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=1e-2)
    before = batch_kl(model, as_batch(rows)).item()
    for _ in range(10):
        do_epoch(model, Dataset.from_list(rows), batch_size=4, optimizer=optimizer)
    after = batch_kl(model, as_batch(rows)).item()
    assert before > 0
    assert after < 0.5 * before


def test_u_override_only_swaps_the_system_prompt(tokenizer):
    question = "What is two plus two?"
    old = tokenizer(format_prompt(tokenizer, question, "Be sad."), add_special_tokens=False)[
        "input_ids"
    ]
    new = tokenizer(format_prompt(tokenizer, question, "Be glad."), add_special_tokens=False)[
        "input_ids"
    ]
    batch = {
        "question": [question],
        "input_ids": [[*old, 11, 12, STOP]],
        "generated_text_mask": [[0] * len(old) + [1, 1, 0]],
        "prompt_input_ids": [old],
    }
    ids, mask = with_system_prompt(batch, tokenizer, "Be glad.")
    assert ids == [[*new, 11, 12, STOP]]
    assert mask == [[0] * len(new) + [1, 1, 0]]


def test_list_epoch_dirs_skips_the_tensorboard_folder(tmp_path):
    for name in ["epoch_10", "epoch_2", "tensorboard", "epoch_x"]:
        (tmp_path / name).mkdir()
    assert list_epoch_dirs(tmp_path) == [(2, tmp_path / "epoch_2"), (10, tmp_path / "epoch_10")]


def write_jsonl(path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


def test_resume_carries_on_from_the_newest_epoch(tmp_path, monkeypatch, capsys):
    write_jsonl(tmp_path / "train.jsonl", fake_rows(4, seed=1))
    write_jsonl(tmp_path / "val.jsonl", fake_rows(2, seed=2))
    out_dir = tmp_path / "run"
    out_dir.mkdir()
    parser = argparse.ArgumentParser()
    add_training_args(parser)
    argv = ["--data_path", str(tmp_path / "train.jsonl"), "--val_path", str(tmp_path / "val.jsonl")]
    argv += ["--model_name", TINY, "--num_epochs", "2", "--batch_size", "2", "-r", "4"]
    argv += ["--device", "cpu", "--seed", "0", "--no_tensorboard"]
    args = parser.parse_args(argv)
    args.out_dir = str(out_dir)
    (out_dir / "args.json").write_text(json.dumps(vars(args)), encoding="utf-8")
    train(args)
    assert [n for n, _ in list_epoch_dirs(out_dir)] == [0, 1]
    assert torch.load(out_dir / "training_state.pt")["epoch"] == 1
    # No KL gets below zero, so the resumed epoch must not replace this best.
    unbeatable = {"epoch": 1, "val_kl": -1.0}
    (out_dir / "best.json").write_text(json.dumps(unbeatable), encoding="utf-8")

    argv = ["train_loop_resume.py", "--out_dir", str(out_dir), "--extend_epochs", "1"]
    monkeypatch.setattr("sys.argv", argv)
    capsys.readouterr()
    train_loop_resume.main()
    assert "Adam starts over" not in capsys.readouterr().out
    assert [n for n, _ in list_epoch_dirs(out_dir)] == [0, 1, 2]
    assert torch.load(out_dir / "training_state.pt")["epoch"] == 2
    assert json.loads((out_dir / "args.json").read_text(encoding="utf-8"))["num_epochs"] == 3
    assert json.loads((out_dir / "best.json").read_text(encoding="utf-8")) == unbeatable
