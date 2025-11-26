import argparse
import datasets
from transformers import AutoTokenizer, AutoModelForCausalLM
import pdb
from peft import get_peft_config, PeftModel, PeftConfig, get_peft_model, LoraConfig, TaskType
import os
import torch
from tqdm import tqdm
import sys
from generate_data import format_prompt
from train_loop import pad_list_of_lists
import json

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_dir', type=str, required=True)
    parser.add_argument('--model_epoch', type=str, default='last', help='Model epoch to load. Optionally enter a number. Default = '"last"'')
    parser.add_argument('--model_name', type=str, default='meta-llama/Meta-Llama-3-8B-Instruct', help='Model name. Default = "meta-llama/Meta-Llama-3-8B-Instruct"')
    parser.add_argument('--num_questions', type=int, default=50, help='Number of questions to test.')
    parser.add_argument('--include_train', type=str, default="False", help='Include training data in the test. Default = False')
    parser.add_argument('--include_val', type=str, default="True", help='Include validation data in the test. Default = True')
    parser.add_argument('--include_test', type=str, default="False", help='Include test data in the test. Default = False')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--pre_question_str', type=str, default="", help='String to prepend to the question from the dataset. Default = "". Try "Q: "')
    parser.add_argument('--pre_answer_str', type=str, default="A:", help='String to prepend before assistant starts generating. Default = "A:".')
    parser.add_argument('--u_file', type=str, default='data/cot_math_x0.md', help='Path to the x0 file. Default = "data/cot_math_x0.md"')
    # ^ Formerly known as x0_file. Still in the process of changing...
    parser.add_argument('--dataset', type=str, default='none', help='Name of the dataset. Will go to data/{dataset}_validation.jsonl. Default = "none"—will check the dataset in args.json and use that.')
    parser.add_argument('--temperature', type=float, default=0.0, help='Temperature for sequence generation. Default = 0.0')
    parser.add_argument('--seed', type=int, default=42)
    print("Arguments parsed.")

    return parser.parse_args()


def reformat_answer_string(answer, dataset):
    '''This function receives the item from the 'answer' column of the dataset specified
    and returns the nicely formatted integer that should be output in a string format.
    Dataset is a string. Valid datasets are: 'asdiv', 'gsm8k', 'svamp'.'''
    if dataset == "asdiv":
        num_str = answer.split(" ")[0]
        #int_to_error = int(num_str)
    elif dataset == "gsm8k":
        # GSM8k looks like more.\n#### 5"
        #remove commas from the string
        answer = answer.replace(",", "")
        num_str = answer.split(" ")[-1]
        int_to_error = int(num_str)
    elif dataset == "svamp":
        # convert answer directly to an int.
        answer = answer.replace(",", "")
        int_answer = int(answer)
        num_str = str(int_answer)
    else:
        raise ValueError("Invalid dataset.")
    return num_str


def assess_correct(generate_str_list, correct_answer_list, dataset_str):
    # 
    # First, we want to check if the lengths of the lists are the same.
    if len(generate_str_list) != len(correct_answer_list):
        raise ValueError("The lengths of the lists are not the same.")
    
    solved_list = []
    solved_list_upper_bound = []
    selved_list_last_sentence = []
    for i in range(len(generate_str_list)):
        generate_str = generate_str_list[i]
        generate_str.replace(",", "")
        correct_answer = reformat_answer_string(correct_answer_list[i], dataset=dataset_str)
        if f"The answer is {correct_answer}." in generate_str:
            solved_list.append(True)
        else:
            solved_list.append(False)

        if str(correct_answer) in generate_str:
            solved_list_upper_bound.append(True)
        else:
            solved_list_upper_bound.append(False)

        if str(correct_answer) in (generate_str + " ").split('.')[-2]:
            selved_list_last_sentence.append(True)
        else:
            selved_list_last_sentence.append(False)


    return solved_list, solved_list_upper_bound, selved_list_last_sentence
    

if __name__ == "__main__":
    args = parse_args()

    # check whether results_dir is a valid directory
    results_dir = args.results_dir
    if not os.path.isdir(results_dir):
        raise ValueError("results_dir is not a valid directory.")
    
    print("results_dir: ", results_dir)
    print("model_epoch: ", args.model_epoch)

    if args.dataset == 'none':
        if 'asdiv' in results_dir:
            dataset_name = 'asdiv'
        elif 'gsm8k' in results_dir:
            dataset_name = 'gsm8k'
        elif 'svamp' in results_dir:
            dataset_name = 'svamp'
        else:
            raise ValueError("Dataset name not found in results_dir.")
    elif args.dataset in ['asdiv', 'gsm8k', 'svamp']:
        dataset_name = args.dataset
    else:
        raise ValueError("Invalid dataset name.")

    dataset_list = []
    print(args.include_train)
    print(args.include_val)
    if args.include_train == 'True':
        dataset_train = datasets.load_dataset('json', data_files=f"data/{dataset_name}_train.jsonl").shuffle(seed=args.seed)['train']
        print("Length of train_dataset: ", len(dataset_train))
        dataset_list += [dataset_train]
    if args.include_val == 'True':
        dataset_val = datasets.load_dataset('json', data_files=f"data/{dataset_name}_validation.jsonl").shuffle(seed=args.seed)['train']
        print("Length of val_dataset: ", len(dataset_val))
        dataset_list += [dataset_val]
    if args.include_test == 'True':
        dataset_test = datasets.load_dataset('json', data_files=f"data/{dataset_name}_test.jsonl").shuffle(seed=args.seed)['train']
        print("Length of test_dataset: ", len(dataset_test))
        dataset_list += [dataset_test]

    # Combine all datasets specified into one dataset and shuffle them together
    dataset = datasets.concatenate_datasets(dataset_list).shuffle(seed=args.seed)

    if args.model_epoch == 'last':
        # find the largest epoch number in the results directory
        # Folders are named epoch_0, epoch_1, ...
        epoch_folders = [f for f in os.listdir(results_dir) if os.path.isdir(os.path.join(results_dir, f))]
        epoch_numbers = [int(f.split('_')[-1]) for f in epoch_folders]
        max_epoch = max(epoch_numbers)
        epoch_num = max_epoch
    else:
        epoch_num = args.model_epoch

    epoch_dir = results_dir + f"/epoch_{epoch_num}"

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=torch.bfloat16, device_map="auto")
    config = PeftConfig.from_pretrained(epoch_dir)
    peft_model = PeftModel.from_pretrained(base_model, epoch_dir).to(torch.device("cuda"))

    u_str = open(args.u_file, 'r').read()
    # grab a batch
    mean_accuracy_base = 0
    mean_accuracy_peft = 0
    mean_accuracy_peft_sys = 0
    mean_accuracy_upper_bound_base = 0
    mean_accuracy_upper_bound_peft = 0
    mean_accuracy_upper_bound_peft_sys = 0
    mean_accuracy_last_sentence_base = 0
    mean_accuracy_last_sentence_peft = 0
    mean_accuracy_last_sentence_peft_sys = 0
    num_tests = min(args.num_questions, len(dataset))
    for i in tqdm(range(0, num_tests, args.batch_size)):
        batch_start = i
        batch_end = min(i+args.batch_size, num_tests)
        batch = dataset[batch_start:batch_end]

        questions = batch['question']
        answers = batch['answer']

        formatted_questions = [format_prompt(u_str, args.pre_question_str + question, use_system=True) + args.pre_answer_str for question in questions]
        formatted_questions_nosys = [format_prompt(u_str, args.pre_question_str + question, use_system=False) + args.pre_answer_str for question in questions]

        tokenized_questions = [tokenizer(q)['input_ids'] for q in formatted_questions]
        tokenized_questions_nosys = [tokenizer(q)['input_ids'] for q in formatted_questions_nosys]

        question_ids_padded, pad_mask = pad_list_of_lists(tokenized_questions, tokenizer.pad_token_id, pad_side='left', return_pad_mask=True)
        question_ids_padded_nosys, pad_mask_nosys = pad_list_of_lists(tokenized_questions_nosys, tokenizer.pad_token_id, pad_side='left', return_pad_mask=True)

        batch_question_ids = torch.tensor(question_ids_padded).to(torch.device("cuda"))
        batch_question_ids_nosys = torch.tensor(question_ids_padded_nosys).to(torch.device("cuda"))

        batch_pad_mask = torch.tensor(pad_mask).to(torch.device("cuda"))
        batch_pad_mask_nosys = torch.tensor(pad_mask_nosys).to(torch.device("cuda"))

        assert [tokenized_questions[i] == batch_question_ids[i][batch_pad_mask[i].bool()] for i in range(len(tokenized_questions))]
        assert [tokenized_questions_nosys[i] == batch_question_ids_nosys[i][batch_pad_mask_nosys[i].bool()] for i in range(len(tokenized_questions_nosys))]

        if args.temperature > 0:
            do_sample = True
        else:
            do_sample = False
        with peft_model.disable_adapter():
            with torch.no_grad():
                outputs = peft_model.generate(batch_question_ids, 
                                                attention_mask = batch_pad_mask.bool(),
                                                do_sample = do_sample, 
                                                max_new_tokens = 300,
                                                min_length = 50,
                                                temperature = args.temperature,
                                                pad_token_id = tokenizer.eos_token_id)
        with torch.no_grad():
            outputs_nosys = peft_model.generate(batch_question_ids_nosys, 
                                                attention_mask = batch_pad_mask_nosys.bool(),
                                                do_sample = do_sample, 
                                                max_new_tokens = 300,
                                                min_length = 50,
                                                temperature = args.temperature,
                                                pad_token_id = tokenizer.eos_token_id,
                                                )
            outputs_peft_sys = peft_model.generate(batch_question_ids,
                                                attention_mask = batch_pad_mask.bool(),
                                                do_sample = do_sample,
                                                max_new_tokens = 300,
                                                min_length = 50,
                                                temperature = args.temperature,
                                                pad_token_id = tokenizer.eos_token_id,
                                                )
        
        generate_str_list = tokenizer.batch_decode(outputs[:, batch_question_ids.shape[1]:])
        generate_str_list_nosys = tokenizer.batch_decode(outputs_nosys[:, batch_question_ids_nosys.shape[1]:])
        generate_str_list_peft_sys = tokenizer.batch_decode(outputs_peft_sys[:, batch_question_ids.shape[1]:])
        # pdb.set_trace()

        solved_list, slub, slls = assess_correct(generate_str_list, answers, dataset_name)
        solved_list_nosys, slub_nosys, slls_nosys = assess_correct(generate_str_list_nosys, answers, dataset_name)
        solved_list_peft_sys, slub_peft_sys, slls_peft_nosys = assess_correct(generate_str_list_peft_sys, answers, dataset_name)
        
        # print("Batch number ", i)
        # print("Accuracy on base model: ", sum(solved_list)/len(solved_list))
        # print("Accuracy on peft model: ", sum(solved_list_nosys)/len(solved_list_nosys))
        # print("Accuracy on peft model with system prompt: ", sum(solved_list_peft_sys)/len(solved_list))
        # print("Upper bound accuracy on base model: ", sum(slub)/len(slub))
        # print("Upper bound accuracy on peft model: ", sum(slub_nosys)/len(slub_nosys))
        # print("Upper bound accuracy on peft model with system prompt: ", sum(slub_peft_sys)/len(slub_peft_sys))
        # print("Last sentence accuracy on base model: ", sum(slls)/len(slls))
        # print("Last sentence accuracy on peft model: ", sum(slls_nosys)/len(slls_nosys))
        # print("Last sentence accuracy on peft model with system prompt: ", sum(slls_peft_nosys)/len(slls_peft_nosys))
        
        mean_accuracy_base += sum(solved_list) / num_tests
        mean_accuracy_peft += sum(solved_list_nosys) / num_tests
        mean_accuracy_peft_sys += sum(solved_list_peft_sys) / num_tests
        mean_accuracy_upper_bound_base += sum(slub) / num_tests
        mean_accuracy_upper_bound_peft += sum(slub_nosys) / num_tests
        mean_accuracy_upper_bound_peft_sys += sum(slub_peft_sys) / num_tests
        mean_accuracy_last_sentence_base += sum(slls) / num_tests
        mean_accuracy_last_sentence_peft += sum(slls_nosys) / num_tests
        mean_accuracy_last_sentence_peft_sys += sum(slls_peft_nosys) / num_tests


    json_filename = results_dir + f"/mathtest_ep{epoch_num}_numq{args.num_questions}_gentemp{args.temperature}_datasetname{dataset_name}.json"

    results_dict = {"args": vars(args),
                    "mean_accuracy_base": mean_accuracy_base,
                    "mean_accuracy_peft": mean_accuracy_peft,
                    "mean_accuracy_peft_sys": mean_accuracy_peft_sys,
                    "mean_accuracy_upper_bound_base": mean_accuracy_upper_bound_base,
                    "mean_accuracy_upper_bound_peft": mean_accuracy_upper_bound_peft,
                    "mean_accuracy_upper_bound_peft_sys": mean_accuracy_upper_bound_peft_sys,
                    "mean_accuracy_last_sentence_base": mean_accuracy_last_sentence_base,
                    "mean_accuracy_last_sentence_peft": mean_accuracy_last_sentence_peft,
                    "mean_accuracy_last_sentence_peft_sys": mean_accuracy_last_sentence_peft_sys}

    json.dump(results_dict, open(json_filename, 'w'))