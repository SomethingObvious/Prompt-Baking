import glob
import os
import pdb

# get all the train and validation datasets
import glob

data_dir = "data/ballmer_20240726/instructqa"

train_files = glob.glob(data_dir + "/*_train_*.jsonl")
val_files = [f.replace("_train_", "_validation_").replace("numq100", "numq50").replace("numq400","numq100") for f in train_files]

lrs = [1e-4]
rs = [2, 8, 32]
batch_size = 16
base_dir = "data/ballmer_20240726"
results_dir = "results/ballmer_20240726"
num_epochs = 40


# mkdir -p base_dir
if not os.path.exists(base_dir):
    os.makedirs(base_dir)

cnt = 0
for train_dataset, validation_dataset in zip(train_files, val_files):
    for lr in lrs:
        for r in rs:
            # note: we can ignore the x0_file because there is a 1:1 mapping between x0_file and question_dataset
            question_dataset_basename = os.path.splitext(os.path.basename(train_dataset))[0]
            print(question_dataset_basename)

            folder_name = question_dataset_basename.split("train_X0")[1]
            out_dir = f"{results_dir}/instructqa/{folder_name}_batch{batch_size}_epochs{num_epochs}_lr{lr}_r{r}"

            data_gen_func_call = f"python3 train_loop.py --num_epochs {num_epochs} --batch_size {batch_size} --learning_rate {lr} -r {r} --data_path {train_dataset} --val_path {validation_dataset} --out_dir {out_dir}"
            
            print(f"[{cnt}] This is the data gen script call: ", data_gen_func_call)

            # write to scripts/commands_cotqa_20240726.txt
            with open("scripts/commands_TRAIN_instructqa_20240726.txt", "a") as f:
                f.write(data_gen_func_call + "\n")

            cnt += 1