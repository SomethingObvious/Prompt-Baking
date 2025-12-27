# Prompt-Weight Equivalence 

**Goal**: How easy is it to train an LLM via weight updates s.t. its probability
distribution over subsequent token sequences is identical to that of a prompted
model.


## Setup
```bash
# make virtual environment
python3 -m venv venv
source venv/bin/activate

# install dependencies
pip3 install -r requirements.txt

# download SQuAD dataset into data/*.jsonl
mkdir -p data
python3 download_data.py 
```


## Run Experiments 

```bash

# enter venv
source venv/Scripts/activate

# generate trajectory dataset with ground truth prompted logits
python generate_data.py \
  --x0_file data/capital_x0.md \
  --question_dataset data/squad_train.jsonl \
  --num_questions 80 \
  --num_sequences_per_question 4 \
  --max_sequence_length 128 \
  --min_sequence_length 32 \
  --temperature 2.0 \
  --batch_size 1 \
  --traj_out_file data/train_traj_q80_s4_len128_x0capital.jsonl \
  --model_name meta-llama/Meta-Llama-3-8B-Instruct



# generate validation data
python generate_data.py \
  --x0_file data/capital_x0.md \
  --question_dataset data/squad_validation.jsonl \
  --num_questions 20 \
  --num_sequences_per_question 4 \
  --max_sequence_length 128 \
  --min_sequence_length 32 \
  --temperature 2.0 \
  --batch_size 1 \
  --traj_out_file data/val_traj_q20_s4_len128_x0capital.jsonl \
  --model_name meta-llama/Meta-Llama-3-8B-Instruct


# CHANGE THE NAMES OF THE JSON FILES

# train a LoRA model to match the probabilities over the trajectories generated above.
python train_loop_custom.py \
  --num_epochs 50 \
  --batch_size 2 \
  --learning_rate 1e-5 \
  --data_path data/train_traj_q80_s4_len128_x0capital.jsonl \
  --val_path data/val_traj_q20_s4_len128_x0capital.jsonl \
   --out_dir results/baked_capital_squad_bs4_ep50

# How to use Resume (Starts again from latest epoch)
python train_loop_resume.py \
  --out_dir results/baked_capital_squad_bs4_ep50 \
  --data_path data/train_traj_q80_s4_len128_x0capital.jsonl \
  --val_path data/val_traj_q20_s4_len128_x0capital.jsonl \
  --batch_size 2 \
  --learning_rate 1e-4

```





## Dashboard

To run the streamlit dashboard, use 
```bash
streamlit run dashboard/app.py
```



## Experiments 2024 07 19

### I'm blue baabaaadeeeebuuuuddaa
```bash
# Generate new dataset with improved system prompt management
# one prompt with all caps, one prompt always mention the color blue in each
# sentence you generate
python3 generate_data.py \
    --x0_file data/blue_x0.md \
    --question_dataset data/squad_train.jsonl \
    --num_questions 100 \
    --num_sequences_per_question 25 \
    --max_sequence_length 300 \
    --min_sequence_length 100 \
    --temperature 2.0 \
    --batch_size 64 \
    --traj_out_file data/traj_bluex0_squad_train.jsonl

# generate validation set
python3 generate_data.py \
    --x0_file data/blue_x0.md \
    --question_dataset data/squad_validation.jsonl \
    --num_questions 25 \
    --num_sequences_per_question 25 \
    --max_sequence_length 300 \
    --min_sequence_length 100 \
    --temperature 2.0 \
    --batch_size 32 \
    --traj_out_file data/traj_bluex0_squad_val.jsonl

# Train blue model 
python3 train_loop.py \
    --num_epochs 20 \
    --learning_rate 3e-4 \
    --data_path data/traj_bluex0_squad_train.jsonl \
    --val_path data/traj_bluex0_squad_val.jsonl \
    --out_dir results/blue20240719.1
    --batch_size 32
```




## Experiments 2024 07 22

_Planning doc: [(Outline) Prompt-Weight Equivalence](https://docs.google.com/document/d/1zHQ3FQRgFLRznf4EOKSk7E16VuOrw4a-i84KnBxhWsk/edit?usp=sharing)_

```bash
# Data generation -- creates scripts/train_loop_commands_20240722.txt
bash scripts/datagen_20240722.sh

bash scripts/executor.sh scripts/commands_datagen_20240722.txt 0 7
bash scripts/executor.sh scripts/commands_datagen_20240722.txt 1 7
bash scripts/executor.sh scripts/commands_datagen_20240722.txt 2 7
bash scripts/executor.sh scripts/commands_datagen_20240722.txt 3 7
bash scripts/executor.sh scripts/commands_datagen_20240722.txt 4 7
bash scripts/executor.sh scripts/commands_datagen_20240722.txt 5 7
bash scripts/executor.sh scripts/commands_datagen_20240722.txt 6 7


# Training loop -- creates scripts/train_loop_commands_20240722.txt
bash scripts/train_loop_calls_20240722.sh

bash scripts/executor.sh scripts/train_loop_commands_20240722.txt 0 7
bash scripts/executor.sh scripts/train_loop_commands_20240722.txt 1 7
bash scripts/executor.sh scripts/train_loop_commands_20240722.txt 2 7
bash scripts/executor.sh scripts/train_loop_commands_20240722.txt 3 7
bash scripts/executor.sh scripts/train_loop_commands_20240722.txt 4 7
bash scripts/executor.sh scripts/train_loop_commands_20240722.txt 5 7
bash scripts/executor.sh scripts/train_loop_commands_20240722.txt 6 7
```


## Comparison Script 2024 07 23

```bash
python3 compare_models.py \
    --results_dir results/20240722/traj_always_rhyme_x0_squad_ep150 \
    --x0_override data/blue_x0.md
```











