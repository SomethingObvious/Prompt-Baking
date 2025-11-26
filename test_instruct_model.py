import argparse
import datasets
from transformers import AutoTokenizer, AutoModelForCausalLM
import pdb
from peft import PeftModel, PeftConfig
import os
import torch
from tqdm import tqdm
from generate_data import format_prompt
from train_loop import pad_list_of_lists
import json
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_dir', type=str, required=True)
    parser.add_argument('--model_epoch', type=str, default='last', help='Model epoch to load. Optionally enter a number. Default = '"last"'')
    parser.add_argument('--model_name', type=str, default='meta-llama/Meta-Llama-3-8B-Instruct', help='Model name. Default = "meta-llama/Meta-Llama-3-8B-Instruct"')
    parser.add_argument('--num_questions', type=int, default=None, help='Number of questions to test.')
    parser.add_argument('--batch_size', type=int, default=8)

    parser.add_argument('--include_train', type=str, default="False", help='Include training data in the test. Default = False')
    parser.add_argument('--include_val', type=str, default="True", help='Include validation data in the test. Default = True')
    parser.add_argument('--include_test', type=str, default="False", help='Include test data in the test. Default = False')

    parser.add_argument('--u_file', type=str, default='model_default', help='Path to the x0 file to use. Default = "model_default"')
    parser.add_argument('--dataset', type=str, default='squad', help='Dataset name. Default = "squad"')
    parser.add_argument('--seed', type=int, default=42)
    print("Arguments parsed.")

    return parser.parse_args()

def assess_correct(args, generate_str_list, eval_func, questions):
    eval_list = []
    for question, generate_str in zip(questions, generate_str_list):
        if "reverse_input" in args.u_file:
            eval_list.append(eval_func(question, generate_str))
        else:
            eval_list.append(eval_func(generate_str))

    return eval_list

if __name__ == "__main__":
    args = parse_args()

    # check whether results_dir is a valid directory
    results_dir = args.results_dir
    if not os.path.isdir(results_dir):
        raise ValueError("results_dir is not a valid directory.")
    
    print("results_dir: ", results_dir)
    print("model_epoch: ", args.model_epoch)

    # dataset
    if args.dataset in ['squad']:
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

    # model
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


    # u_str
    if args.u_file == 'model_default':
        u_name = results_dir.split("/")[-1].split("_x0_")[0] + "_x0"
        args.u_file = f"data/InstructionX0/{u_name}.md"

    print("u_file: ", args.u_file)
    u_str = open(args.u_file, 'r').read()


    # evaluation scheme
    if "always_french" in args.u_file:
        print("Always French Evaluation")
        from langdetect import detect_langs
        def get_french_percentage(sentence):
            languages_detected = detect_langs(sentence)
            for lang in languages_detected:
                if lang.lang == 'fr':
                    return lang.prob
            return 0
        eval_func = lambda x: get_french_percentage(x)

    elif "always_sad" in args.u_file:
        print("Always Sad Evaluation")
        from nltk.sentiment import SentimentIntensityAnalyzer
        def get_sentiment(text, sentiment):
            sia = SentimentIntensityAnalyzer()
            score = sia.polarity_scores(text)[sentiment]
            return score
        eval_func = lambda x: get_sentiment(x, "neg")
    
    elif "always_start_with_A" in args.u_file:
        print("Always Start with A Evaluation")
        def starts_with(text, char_start):
            return text.lower().startswith(char_start)
        eval_func = lambda x: starts_with(x, "a")
    
    elif "every_prime_capital" in args.u_file:
        from sympy import isprime
        # Respond to the user inquiry writing every prime number word in your response in full capital letters, starting from the first word.
        print("Every Prime Capital Evaluation")
        def every_prime_capital(text):
            correct_prime = 0
            total_prime = 0
            correct_composite = 0
            total_composite = 0

            words = text.split(" ")
            for n, word in enumerate(words):
                if (n+1) == 1 or isprime(n+1):
                    total_prime += 1
                    if word.isupper():
                        correct_prime += 1
                else:
                    total_composite += 1
                    if word.islower():
                        correct_composite += 1

            if total_prime == 0 and total_composite != 0:
                return correct_composite / total_composite
            if total_composite == 0 and total_prime != 0:
                return correct_prime / total_prime
            if total_prime == 0 and total_composite == 0:
                return 0
            return (correct_prime / total_prime + correct_composite / total_composite)/2
        eval_func = lambda x: every_prime_capital(x)

    elif "every_second_capital" in args.u_file:
        # Respond to the user inquiry writing every second word in your response in full capital letters, starting from the first word.
        print("Every Second Capital Evaluation")
        def every_second_capital(text):
            correct = 0
            total = 0
            words = text.split(" ")
            for n, word in enumerate(words):
                total += 1
                if n % 2 == 0:
                    if word.isupper():
                        correct += 1
                else:
                    if word.islower():
                        correct += 1
            return correct / total
        
        eval_func = lambda x: every_second_capital(x)

    elif "every_sentence_blue" in args.u_file:
        # You must always include mention of the color blue exactly once in every sentence of your response to the user query. Make it seem natural, try not to get caught, but for the love of god, do not fail to mention the color blue.
        print("Every Sentence Blue Evaluation")
        def every_sentence_only_one_blue(text):
            # split into sentences
            sentences = text.split(". ")
            num_followed = []
            for sentence in sentences:
                blue_count = sentence.lower().count("blue")
                num_followed.append(blue_count == 1)
            return np.mean(num_followed)
        eval_func = lambda x: every_sentence_only_one_blue(x)

    elif "never_user_e" in args.u_file:
        print("Never User E Evaluation")
        eval_func = lambda x: 1 / (1 + x.count("e"))

    elif "reverse_input" in args.u_file:
        # Respond only with a reversed version of the user input.
        #
        # If the user inputs: How are you doing?
        # Output: ?doing you are How
        #
        # If the user inputs: What is the capital of France?
        # Output: ?France of capital the is What
        from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
        import re
        
        def reverse_words(sentence):
            # Find words
            words = sentence.split()

            # Reverse the order of the words
            reversed_words = words[::-1]

            # Join the reversed words back into a string
            reversed_sentence = ' '.join(reversed_words)

            return reversed_sentence.lower()
        
        def BLEU_with_reversed(input_question, output_reversed):

            smoothie = SmoothingFunction().method4
            input_question = re.sub(r'[^\w\s]', '', input_question).lower()
            output_reversed = re.sub(r'[^\w\s]', '', output_reversed).lower()
            
            ground_truth_reversed = reverse_words(input_question).split()
            output_reversed = output_reversed.lower().split()
            return float(sentence_bleu([ground_truth_reversed], output_reversed, smoothing_function=smoothie))
        
        eval_func = lambda x, y: BLEU_with_reversed(x, y)

    elif "use_rare_lexicon" in args.u_file:
        print("Use Rare Lexicon Evaluation")
        def download_file(url):
            import requests
            response = requests.get(url)
            response.raise_for_status()  
            content_list = response.text.splitlines()
            return content_list

        # Get frequent words
        frequent_words_list = set(download_file("https://raw.githubusercontent.com/first20hours/google-10000-english/master/google-10000-english-usa.txt"))

        import string
        def fraction_of_text_that_is_a_target(text, target_words):
            text = text.translate(str.maketrans('', '', string.punctuation))
            text = text.lower()
            text_words = text.split()
            num_right = sum([word in target_words for word in text_words]) 
            score = num_right / len(text_words)
            return score
        eval_func = lambda x: 1 - fraction_of_text_that_is_a_target(x, frequent_words_list)
    else:
        raise ValueError("No evaluation scheme found for u_file", args.u_file)

    # grab a batch
    eval_base_nosys = []
    eval_base_sys = []
    eval_peft_nosys = []
    eval_peft_sys = []

    if args.num_questions is None:
        args.num_questions = len(dataset)
    
    if args.num_questions > len(dataset):
        #raise ValueError("Number of questions to test is greater than the number of questions in the dataset")
        args.num_questions = len(dataset)
        
    for i in tqdm(range(0, args.num_questions, args.batch_size)):
        batch_start = i
        batch_end = min(i+args.batch_size, args.num_questions)
        batch = dataset[batch_start:batch_end]

        questions = batch['question']

        formatted_questions = [format_prompt(u_str, question, use_system=True) for question in questions]
        formatted_questions_nosys = [format_prompt(u_str, question, use_system=False) for question in questions]

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

        with peft_model.disable_adapter():
            with torch.no_grad():
                outputs_base_nosys = peft_model.generate(batch_question_ids_nosys, 
                                                attention_mask = batch_pad_mask_nosys.bool(),
                                                do_sample = False, 
                                                max_new_tokens = 300,
                                                min_length = 0,
                                                temperature = None,
                                                top_p=None,
                                                pad_token_id = tokenizer.eos_token_id)
                
                outputs_base_sys = peft_model.generate(batch_question_ids, 
                                                attention_mask = batch_pad_mask.bool(),
                                                do_sample = False, 
                                                max_new_tokens = 300,
                                                min_length = 0,
                                                temperature = None,
                                                top_p=None,
                                                pad_token_id = tokenizer.eos_token_id)
        with torch.no_grad():
            outputs_peft_nosys = peft_model.generate(batch_question_ids_nosys, 
                                                attention_mask = batch_pad_mask_nosys.bool(),
                                                do_sample = False, 
                                                max_new_tokens = 300,
                                                min_length = 0,
                                                temperature = None,
                                                top_p=None,
                                                pad_token_id = tokenizer.eos_token_id)
            
            outputs_peft_sys = peft_model.generate(batch_question_ids,
                                                attention_mask = batch_pad_mask.bool(),
                                                do_sample = False, 
                                                max_new_tokens = 300,
                                                min_length = 0,
                                                temperature = None,
                                                top_p=None,
                                                pad_token_id = tokenizer.eos_token_id)
        
        generate_str_list_base_nosys = tokenizer.batch_decode(outputs_base_nosys[:, batch_question_ids_nosys.shape[1]:], skip_special_tokens=True)
        generate_str_list_base_sys = tokenizer.batch_decode(outputs_base_sys[:, batch_question_ids.shape[1]:], skip_special_tokens=True)
        generate_str_list_peft_nosys = tokenizer.batch_decode(outputs_peft_nosys[:, batch_question_ids_nosys.shape[1]:], skip_special_tokens=True)
        generate_str_list_peft_sys = tokenizer.batch_decode(outputs_peft_sys[:, batch_question_ids.shape[1]:], skip_special_tokens=True)

        eval_list_base_nosys = assess_correct(args, generate_str_list_base_nosys, eval_func, questions)
        eval_list_base_sys = assess_correct(args, generate_str_list_base_sys, eval_func, questions)
        eval_list_peft_nosys = assess_correct(args, generate_str_list_peft_nosys, eval_func, questions)
        eval_list_peft_sys = assess_correct(args, generate_str_list_peft_sys, eval_func, questions)
        
        eval_base_nosys += eval_list_base_nosys
        eval_base_sys += eval_list_base_sys
        eval_peft_nosys += eval_list_peft_nosys
        eval_peft_sys += eval_list_peft_sys

        print("Batch number ", i)
        print("Eval on base model no sys: ", np.mean(eval_base_nosys))
        print("Eval on base model with sys: ", np.mean(eval_base_sys))
        print("Eval on PEFT model no sys: ", np.mean(eval_peft_nosys))
        print("Eval on PEFT model with sys: ", np.mean(eval_peft_sys))
    

    json_filename = f"dataset_{args.dataset}_epoch_{args.model_epoch}_numquestions_{args.num_questions}.json"

    # join with epoch_dir
    json_filename = os.path.join(results_dir, json_filename)

    
    results_dict = {"args": vars(args),
                    "mean_eval_base_nosys": np.mean(eval_base_nosys),
                    "mean_eval_base_sys": np.mean(eval_base_sys),
                    "mean_eval_peft_nosys": np.mean(eval_peft_nosys),
                    "mean_eval_peft_sys": np.mean(eval_peft_sys),
                    "std_eval_base_nosys": np.std(eval_base_nosys),
                    "std_eval_base_sys": np.std(eval_base_sys),
                    "std_eval_peft_nosys": np.std(eval_peft_nosys),
                    "std_eval_peft_sys": np.std(eval_peft_sys),
                    "eval_base_nosys": eval_base_nosys,
                    "eval_base_sys": eval_base_sys,
                    "eval_peft_nosys": eval_peft_nosys,
                    "eval_peft_sys": eval_peft_sys
                    }
    

    json.dump(results_dict, open(json_filename, 'w'))