"""
Data generation commands for 2024 07 25 Experiments

**Data generation**: python3 
 - x0_file: `data/*_x0.md` 
 - num_questions: 25 
 - num_sequuences_per_question: 25
 - min_sequence_length, max_sequence_length = (10, 100), (100, 300), (300, 1000)
 - temperatures: 0.5, 1, 3, 5, 10, 20
 - batch_size: 3
 - traj_out_file: Based on names above, stored in `data/20240725`

usage: generate_data.py [-h] [--x0_file X0_FILE] [--question_dataset QUESTION_DATASET] [--num_questions NUM_QUESTIONS]
                        [--num_sequences_per_question NUM_SEQUENCES_PER_QUESTION]
                        [--max_sequence_length MAX_SEQUENCE_LENGTH] [--min_sequence_length MIN_SEQUENCE_LENGTH]
                        [--temperature TEMPERATURE] [--batch_size BATCH_SIZE] [--model_name MODEL_NAME]
                        [--traj_out_file TRAJ_OUT_FILE] [--seed SEED]

Model loading and generation parameters.

options:
  -h, --help            show this help message and exit
  --x0_file X0_FILE     Path to the x0 file. Default = "data/truth_x0.md"
  --question_dataset QUESTION_DATASET
                        Path to the question dataset file. Default = "data/squad_train.jsonl"
  --num_questions NUM_QUESTIONS
                        Number of questions to generate. Default = 100
  --num_sequences_per_question NUM_SEQUENCES_PER_QUESTION
                        Number of sequences to generate per question (default: 25)
  --max_sequence_length MAX_SEQUENCE_LENGTH
                        Maximum length of each generated sequence
  --min_sequence_length MIN_SEQUENCE_LENGTH
                        Minimum length of each generated sequence
  --temperature TEMPERATURE
                        Temperature for sequence generation
  --batch_size BATCH_SIZE
                        Batch size for processing
  --model_name MODEL_NAME
                        Model name to use
  --traj_out_file TRAJ_OUT_FILE
                        Output file for generated sequences. Default = "data/traj_lex.jsonl"
  --seed SEED           Seed for random number generation
"""
import glob
import os
import pdb

# get the list of x0 files from data/*_x0.md
x0_files = glob.glob('data/*_x0.md')
print("x0_files: ", x0_files)

traj_out_base_dir = "data/20240725"
os.makedirs(traj_out_base_dir, exist_ok=True)

num_questions = 25
num_sequences_per_question = 25
question_dataset = "data/squad_validation.jsonl"

sequence_minmax_batch = [(100, 300, 12), (300, 1000, 3)]
temperatures = [3, 5, 20]

commands_file_list = "scripts/commands_datagen_20240725.txt"

# clear the file
with open(commands_file_list, 'w') as f:
    f.write("")

cnt = 0

for x0_file in x0_files:
    for min_sequence_length, max_sequence_length, batch_size in sequence_minmax_batch:
        for temperature in temperatures:
            base_x0 = os.path.basename(x0_file).split('.')[0]

            params_string = f"traj_squadval_x0{base_x0}_minlen{min_sequence_length}_maxlen{max_sequence_length}_temp{temperature}"

            traj_out_file = os.path.join(traj_out_base_dir, f"{params_string}.jsonl")

            cmd = f"python3 generate_data.py --x0_file {x0_file} --question_dataset {question_dataset} --num_questions {num_questions} --num_sequences_per_question {num_sequences_per_question} --min_sequence_length {min_sequence_length} --max_sequence_length {max_sequence_length} --temperature {temperature} --batch_size {batch_size} --traj_out_file {traj_out_file}"

            print(f"[{cnt}] Writing command to file: ", cmd)

            with open(commands_file_list, 'a') as f:
                f.write(f"{cmd}\n")
            
            cnt += 1
            

