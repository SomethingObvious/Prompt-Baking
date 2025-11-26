import glob
import os
import pdb


# traj_train_traj_validation = [("data/ballmer_20240726/traj_asdiv_train_temperature_1.0.jsonl",
#                                "data/ballmer_20240726/traj_asdiv_validation_temperature_1.0.jsonl"),

#                                ("data/ballmer_20240726/traj_asdiv_train_temperature_2.0.jsonl",
#                                 "data/ballmer_20240726/traj_asdiv_validation_temperature_2.0.jsonl"),

#                                 ("data/ballmer_20240726/traj_asdiv_train_temperature_3.0.jsonl",
#                                  "data/ballmer_20240726/traj_asdiv_validation_temperature_3.0.jsonl"),
                                 
#                                  ("data/ballmer_20240726/traj_asdiv_train_temperature_5.0.jsonl",
#                                   "data/ballmer_20240726/traj_asdiv_validation_temperature_5.0.jsonl"),
                                  
#                                   ("data/ballmer_20240726/traj_aqua_train_temperature_1.0.jsonl",
#                                    "data/ballmer_20240726/traj_aqua_validation_temperature_1.0.jsonl"),
                                   
#                                    ("data/ballmer_20240726/traj_aqua_train_temperature_2.0.jsonl",
#                                     "data/ballmer_20240726/traj_aqua_validation_temperature_2.0.jsonl"),
                                    
#                                     ("data/ballmer_20240726/traj_aqua_train_temperature_3.0.jsonl",
#                                      "data/ballmer_20240726/traj_aqua_validation_temperature_3.0.jsonl"),
                                     
#                                      ("data/ballmer_20240726/traj_aqua_train_temperature_5.0.jsonl",
#                                       "data/ballmer_20240726/traj_aqua_validation_temperature_5.0.jsonl"),
                                      
#                                       ("data/ballmer_20240726/traj_gsm8k_train_temperature_1.0.jsonl",
#                                        "data/ballmer_20240726/traj_gsm8k_validation_temperature_1.0.jsonl"),
                                       
#                                        ("data/ballmer_20240726/traj_gsm8k_train_temperature_2.0.jsonl",
#                                         "data/ballmer_20240726/traj_gsm8k_validation_temperature_2.0.jsonl"),
                                        
#                                         ("data/ballmer_20240726/traj_gsm8k_train_temperature_3.0.jsonl",
#                                          "data/ballmer_20240726/traj_gsm8k_validation_temperature_3.0.jsonl"),
                                         
#                                          ("data/ballmer_20240726/traj_gsm8k_train_temperature_5.0.jsonl",
#                                           "data/ballmer_20240726/traj_gsm8k_validation_temperature_5.0.jsonl"),
                                          
#                                           ("data/ballmer_20240726/traj_svamp_train_temperature_1.0.jsonl",
#                                            "data/ballmer_20240726/traj_svamp_validation_temperature_1.0.jsonl"),
                                           
#                                            ("data/ballmer_20240726/traj_svamp_train_temperature_2.0.jsonl",
#                                             "data/ballmer_20240726/traj_svamp_validation_temperature_2.0.jsonl"),
                                            
#                                             ("data/ballmer_20240726/traj_svamp_train_temperature_3.0.jsonl",
#                                              "data/ballmer_20240726/traj_svamp_validation_temperature_3.0.jsonl"),
                                             
#                                              ("data/ballmer_20240726/traj_svamp_train_temperature_5.0.jsonl",
#                                               "data/ballmer_20240726/traj_svamp_validation_temperature_5.0.jsonl"),
                                              
#                                               ("data/ballmer_20240726/traj_mathqa_train_temperature_1.0.jsonl",
#                                                "data/ballmer_20240726/traj_mathqa_validation_temperature_1.0.jsonl"),
                                               
#                                                ("data/ballmer_20240726/traj_mathqa_train_temperature_2.0.jsonl",
#                                                 "data/ballmer_20240726/traj_mathqa_validation_temperature_2.0.jsonl"),
                                                
#                                                 ("data/ballmer_20240726/traj_mathqa_train_temperature_3.0.jsonl",
#                                                  "data/ballmer_20240726/traj_mathqa_validation_temperature_3.0.jsonl"),
                                                 
#                                                  ("data/ballmer_20240726/traj_mathqa_train_temperature_5.0.jsonl",
#                                                   "data/ballmer_20240726/traj_mathqa_validation_temperature_5.0.jsonl")]
import os 
import glob


traj_train_paths = []
traj_valid_paths = []
for dataset_name in ("asdiv", "gsm8k", "svamp"):
    for temperature in (1.0, 2.0, 3.0, 5.0):
        traj_train_paths_regex = f"data/ballmer_20240726/cotqa/traj_{dataset_name}_train_temperature_{temperature}.jsonl"
        traj_valid_paths_regex = f"data/ballmer_20240726/cotqa/traj_{dataset_name}_validation_temperature_{temperature}.jsonl"

        # assert the files exist 
        assert os.path.exists(traj_train_paths_regex), f"File {traj_train_paths_regex} does not exist"
        assert os.path.exists(traj_valid_paths_regex), f"File {traj_valid_paths_regex} does not exist"

        traj_train_paths.append(traj_train_paths_regex)
        traj_valid_paths.append(traj_valid_paths_regex)

traj_train_traj_validation = list(zip(traj_train_paths, traj_valid_paths))

lrs = (3e-4, 1e-4)
rs = (2, 8, 32)
batch_size = 16
base_dir = "data/ballmer_20240726"
results_dir = "results/ballmer_20240726"
num_epochs = 40

OUTPUT_COMMANDS_PATH = "scripts/commands_TRAIN_cotqa_20240726.txt"

# clear the file if it exists
if os.path.exists(OUTPUT_COMMANDS_PATH):
    os.remove(OUTPUT_COMMANDS_PATH)


# mkdir -p base_dir
if not os.path.exists(base_dir):
    os.makedirs(base_dir)

cnt = 0
for train_dataset, validation_dataset in traj_train_traj_validation:
    for lr in lrs:
        for r in rs:
            # note: we can ignore the x0_file because there is a 1:1 mapping between x0_file and question_dataset
            question_dataset_basename = os.path.splitext(os.path.basename(train_dataset))[0]
            
            folder_name = question_dataset_basename.split("traj_")[1]
            out_dir = f"{results_dir}/cotqa/{folder_name}_lr_{lr}_r_{r}"

            data_gen_func_call = f"python3 train_loop.py --num_epochs {num_epochs} --batch_size {batch_size} -r {r} --learning_rate {lr} --data_path {train_dataset} --val_path {validation_dataset} --out_dir {out_dir}"
            
            print(f"[{cnt}] This is the data gen script call: ", data_gen_func_call)

            # write to scripts/commands_cotqa_20240726.txt
            with open(OUTPUT_COMMANDS_PATH, "a") as f:
                f.write(data_gen_func_call + "\n")

            cnt += 1
