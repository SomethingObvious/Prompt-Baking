"""
Data generation commands for 2024 07 26 Experiments on Chain-of-Though QA (CoT-QA)

Parameters to vary: 
 --x0_file: ["data/cot_math_aqua-mathqa_x0.md", "data/cot_math_x0.md"] - We are using the few-shot learning x0 files from page 35 (table 20) of https://arxiv.org/pdf/2201.11903.  - FOR EACH DATASET, THERE IS ONLY ONE DATASET YOU ACTUALLY USE.
 --question_dataset 
     - 5 datasets, data/*.jsonl, train/valid sets included.
 --temperature [1.0, 2.0, 3.0, 5.0]
 --traj_out_file: based on temperature, question_dataset, x0_file, stored in data/ballmer_20240726


Parameters to keep constant: 
 --num_questions 500
 --num_sequences_per_question 5
 --max_sequence_length 100
 --min_sequence_length 1
 --batch_size 10
"""
import glob
import os
import pdb

# get the list of x0 files from data/*_x0.md
question_dataset__x0_file__numq = [
    ("data/aqua_validation.jsonl", "data/cot_math_aqua-mathqa_x0.md", 200),
    ("data/aqua_train.jsonl", "data/cot_math_aqua-mathqa_x0.md", 500),

    ("data/mathqa_validation.jsonl", "data/cot_math_aqua-mathqa_x0.md", 200),
    ("data/mathqa_train.jsonl", "data/cot_math_aqua-mathqa_x0.md", 500),

    ("data/asdiv_validation.jsonl", "data/cot_math_x0.md", 200), 
    ("data/asdiv_train.jsonl", "data/cot_math_x0.md", 500),

    ("data/gsm8k_validation.jsonl", "data/cot_math_x0.md", 200), 
    ("data/gsm8k_train.jsonl", "data/cot_math_x0.md", 500), 

    ("data/svamp_validation.jsonl", "data/cot_math_x0.md", 200), 
    ("data/svamp_train.jsonl", "data/cot_math_x0.md", 500),
]

temperatures = [1.0, 2.0, 3.0, 5.0]

base_dir = "data/ballmer_20240726/cotqa"

num_sequences_per_question = 5
max_sequence_length = 100
min_sequence_length = 1
batch_size = 10

seed = 238948723



# mkdir -p base_dir
if not os.path.exists(base_dir):
    os.makedirs(base_dir)

cnt = 0
for question_dataset, x0_file, num_questions in question_dataset__x0_file__numq:
    for temperature in temperatures: 
        # note: we can ignore the x0_file because there is a 1:1 mapping between x0_file and question_dataset
        question_dataset_basename = os.path.splitext(os.path.basename(question_dataset))[0]

        traj_out_file = os.path.join(base_dir, f"traj_{question_dataset_basename}_temperature_{temperature}.jsonl")


        data_gen_func_call = f"python3 generate_data.py --x0_file {x0_file} --question_dataset {question_dataset} --num_questions {num_questions} --num_sequences_per_question {num_sequences_per_question} --max_sequence_length {max_sequence_length} --min_sequence_length {min_sequence_length} --temperature {temperature} --batch_size {batch_size} --traj_out_file {traj_out_file} --seed {seed}"
        
        print(f"[{cnt}] This is the data gen script call: ", data_gen_func_call)

        # write to scripts/commands_cotqa_20240726.txt
        with open("scripts/commands_cotqa_20240726.txt", "a") as f:
            f.write(data_gen_func_call + "\n")

        cnt += 1