# Prompt Baking

This turns a system prompt into a LoRA adapter, so the model acts as if it had the prompt without paying for it on every request. It samples replies from the prompted model, then trains the adapter on the unprompted model to match the prompted model's next-token distributions over those replies (the KL divergence between the two, summed over each reply). The default model is `Qwen/Qwen3.5-2B`, and any chat model with a system role in its template works through `--model_name`. The method comes from [Prompt Baking](https://arxiv.org/abs/2409.13697) by Bhargava, Witkowski, Detkov and Thomson (2024).

## Setup

It needs Python 3.11 or newer. For a GPU, install the CUDA build of torch from pytorch.org before the rest.

```bash
python -m venv .venv
source .venv/bin/activate         # .venv\Scripts\activate on Windows
pip install -r requirements.txt
python download_data.py           # SQuAD and the math sets, into data/
python -m pytest                  # about 20 seconds on CPU, after 90 MB or so of downloads
```

## Baking a Prompt

The prompt goes in a text file, and `data/InstructionX0/` has examples. Make a training and a validation set of trajectories, then train on them.

```bash
python generate_data.py --x0_file data/InstructionX0/every_sentence_blue_x0.md \
  --question_dataset data/squad_train.jsonl --traj_out_file data/traj_blue_train.jsonl
python generate_data.py --x0_file data/InstructionX0/every_sentence_blue_x0.md \
  --question_dataset data/squad_validation.jsonl --num_questions 20 --traj_out_file data/traj_blue_val.jsonl
python train_loop_custom.py --num_epochs 50 --batch_size 2 --learning_rate 2e-4 --gradient_checkpointing \
  --data_path data/traj_blue_train.jsonl --val_path data/traj_blue_val.jsonl --out_dir results/blue
```

The run folder ends up with `args.json`, an `epoch_N` adapter per epoch and the best one by validation KL at the top. TensorBoard curves go to `results/blue/tensorboard`. If it gets cut off, `python train_loop_resume.py --out_dir results/blue` carries on from the newest epoch with the same settings and optimizer state, and `--extend_epochs 10` trains past the original end.

## Picking a Model

The default is `Qwen/Qwen3.5-2B`. `Qwen/Qwen3.5-0.8B` is quicker for trying a prompt out, and `Qwen/Qwen3.5-4B` follows instructions better, but its weights alone take 7.8 GB in bf16, so it needs a card bigger than 8 GB. All three are Apache 2.0 and not gated, and their template gives the system prompt its own turn and leaves it out when there isn't one. Use the same `--model_name` for generating and training.

These are training peaks on an 8 GB RTX 4070 laptop card, with the blue prompt and 300-token replies.

| Model | `--batch_size 1 --gradient_checkpointing` | `--batch_size 2 --gradient_checkpointing` | `--batch_size 2` |
|---|---|---|---|
| `Qwen/Qwen3.5-0.8B` | 2.9 GB | 4.4 GB | 7.4 GB |
| `Qwen/Qwen3.5-2B` | 5.1 GB | 6.5 GB | 10 GB |

The command above should fit on an 8 GB card with nothing else running on it. Each row in a batch costs about 1.4 GB with either model, which I think is mostly the logits, as every reply token gets a float32 row as wide as the 248,000-token vocabulary on both sides of the KL. Generating with the defaults peaked at 4.4 GB for the 2B. All of this was with the plain PyTorch fallback transformers uses for the Gated DeltaNet layers, which it warns is much slower than `flash-linear-attention` and `causal-conv1d`.

## Checking the Result

```bash
python compare_models.py --results_dir results/blue/epoch_7 --data_file data/traj_blue_val.jsonl
python test_instruct_model.py --results_dir results/blue --u_file data/InstructionX0/every_sentence_blue_x0.md
python test_math_model.py --results_dir results/gsm8k_run --u_file my_cot_prompt.md
python interactive_baked.py       # /load results/blue blue, then /switch blue
python dashboard.py results --open
```

`compare_models.py` plots per-token NLL for the base and baked models with and without the prompt, and the two test scripts score how well the prompt is followed. The math script needs a chain-of-thought prompt file of your own.

`dashboard.py` puts every run under a folder on one page, `results/dashboard.html` by default, with each run's settings, its train and validation KL by epoch, and whatever the two test scripts have scored. It's a single HTML file that works offline, so it can be sent to someone as it is.

## Limits

Everything runs on one device, and there's no quantization, so the model has to fit in memory at 16 bits. The unprompted side is the same chat with no system message at all, which is also what the console sends. Some templates put in a default system prompt when there isn't one (SmolLM3's and Granite 4's do), so with those the unprompted side isn't really unprompted. LoRA only goes on the modules named in `LORA_TARGETS` in `train_utils.py`, so a model with other names for its attention projections (like the fused `qkv_proj` in Phi) needs them added there.

## Licence

It's under the [PolyForm Noncommercial License 1.0.0](LICENSE.md). You can use it, change it and share it, forks included, for anything noncommercial, like research, study or a hobby project. Commercial use isn't allowed.
