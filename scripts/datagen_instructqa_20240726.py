"""
Data generation commands for 2024 07 26 Experiments on Instruction QA (Instruct-QA)

Parameters to vary: 
--x0_file: data/InstructionX0/*_x0.md
--question_dataset: [dataset/squad_train.jsonl, dataset/squad_val.jsonl]
    --num_questions: [train=]A


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


x0_paths = glob.glob("data/InstructionX0/*_x0.md")
question_datasets__num_questions__max_seq_len = [("data/squad_train.jsonl", 400, 50), ("data/squad_train.jsonl", 100, 200),  ("data/squad_validation.jsonl", 100, 50), ("data/squad_validation.jsonl", 50, 200)]

num_sequences_per_question = 25
min_sequence_length = 1

temperatures = [2.0, 3.0, 5.0]
batch_size = 25

seed = 768

base_dir = "data/ballmer_20240726/instructqa"
if not os.path.exists(base_dir):
    os.makedirs(base_dir)

# --traj_out_file is a function of x0_path, question_dataset, num_questions, min_seq_len, temperature, max_sequence_length

cnt = 0
for x0_path in x0_paths: 
    for question_dataset, num_questions, max_seq_len in question_datasets__num_questions__max_seq_len: 
        for temperature in temperatures: 

            question_dataset_basename = os.path.splitext(os.path.basename(question_dataset))[0]

            x0_basename = os.path.splitext(os.path.basename(x0_path))[0]
            # pdb.set_trace() # hello future people, please check that the basenames are being computed properly thx

            traj_out_file = os.path.join(base_dir, f"traj{question_dataset_basename}_X0{x0_basename}_temperature{temperature}_numq{num_questions}_maxseqlen{max_seq_len}.jsonl")

            cmd = f"python3 generate_data.py --x0_file {x0_path} --question_dataset {question_dataset} --num_questions {num_questions} --num_sequences_per_question {num_sequences_per_question} --max_sequence_length {max_seq_len} --min_sequence_length {min_sequence_length} --temperature {temperature} --batch_size {batch_size} --traj_out_file {traj_out_file} --seed {seed}"

            print(f"[{cnt}] This is the data gen script call: ", cmd)
            
            # write to scripts/commands_instructqa_20240726.txt
            cnt += 1

            with open("scripts/commands_instructqa_20240726.txt", "a") as f:
                f.write(cmd + "\n")



base_dir = "data/ballmer_20240726/instructqa"
