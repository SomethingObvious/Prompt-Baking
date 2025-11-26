# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import json
import argparse
import pdb
import torch


parser = argparse.ArgumentParser(description="Model loading and generation parameters.")
# arg: prompt_template, defaults to data/llama_sys_question_template.md
# parser.add_argument('--prompt_template', type=str, default="data/llama_sys_question_template.md", help='Path to the prompt template file. Must have two {} instances, first for system prompt, second for question. Default = "data/llama_sys_question_template.md"')
# arg: x0_file, defaults to data/truth_x0.md
parser.add_argument('--x0_file', type=str, default="data/truth_x0.md", help='Path to the x0 file. Default = "data/truth_x0.md"')
# arg: question_dataset_file, default data/squad_train.jsonl
parser.add_argument('--question_dataset', type=str, default="data/squad_train.jsonl", help='Path to the question dataset file. Default = "data/squad_train.jsonl"')
# arg:num_questions, default = 100
parser.add_argument('--num_questions', type=int, default=100, help='Number of questions to generate. Default = 100')

parser.add_argument('--num_sequences_per_question', type=int, default=25, help='Number of sequences to generate per question (default: 25)')
parser.add_argument('--max_sequence_length', type=int, default=300, help='Maximum length of each generated sequence')
parser.add_argument('--min_sequence_length', type=int, default=100, help='Minimum length of each generated sequence')
parser.add_argument('--temperature', type=float, default=2.0, help='Temperature for sequence generation')
parser.add_argument('--batch_size', type=int, default=38, help='Batch size for processing')
parser.add_argument('--model_name', type=str, default= "meta-llama/Meta-Llama-3-8B-Instruct", help='Model name to use')
parser.add_argument('--traj_out_file', type=str, default="data/traj_lex.jsonl", help='Output file for generated sequences. Default = "data/traj_lex.jsonl"')
parser.add_argument('--seed', type=int, default=42, help='Seed for random number generation')




def format_prompt(system_prompt, user_prompt, use_system=True):
    """ System prompt has the system and user prompts (question) and the header
    for the assistant response according to the Llama docs.
    """
    if use_system:
        _system_prompt = system_prompt
    else:
        _system_prompt = ""
    return f"<|start_header_id|>system<|end_header_id|>\n\n{_system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{user_prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"

if __name__ == "__main__":
    args = parser.parse_args()

    # load x0_file from args.x0_file
    with open(args.x0_file, 'r') as f:
        x0_str = f.read()

    # load question_dataset_file from args.question_dataset_file
    import datasets
    dataset = datasets.load_dataset('json', data_files=args.question_dataset).shuffle(seed=args.seed)
    print("Length of train_dataset: ", len(dataset['train']))







    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
    model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct").half()
    tokenizer.pad_token = tokenizer.eos_token

    model = model.cuda()

    num_q = 0


    with open(args.traj_out_file, 'w') as f: 
        pbar = tqdm(dataset['train'], total=args.num_questions)
        for question in pbar: 
            if num_q >= args.num_questions:
                break
            num_q += 1
            question_str = question['question']
            for i in range(0, args.num_sequences_per_question, args.batch_size): 
                # print("Generating sequences for question: ", question)
                batch_start = i
                batch_end = min(i+args.batch_size, args.num_sequences_per_question)
                # print(f"Batch: {batch_start}:{batch_end}")
                # prompt_q_str = prompt_template.format(x0_str, question_str)
                prompt_q_str = format_prompt(x0_str, question_str, use_system=True)
                noprompt_q_str = format_prompt(x0_str, question_str, use_system=False)
                prompt_q_ids = tokenizer(prompt_q_str, return_tensors="pt").to(model.device)['input_ids']
                noprompt_q_ids = tokenizer(noprompt_q_str, return_tensors="pt").to(model.device)['input_ids']

                num_seq_to_gen = batch_end - batch_start

                # use the prompt string with the system prompt x0 and the question
                # to generate trajectories
                # input_ids = tokenizer(prompt_q_str, return_tensors="pt").to(model.device)

                batch_input_ids = prompt_q_ids.repeat(num_seq_to_gen, 1)
                attention_mask = batch_input_ids.ne(tokenizer.pad_token_id).long()



                # output has shape [batch, num_tokens_total]
                output = model.generate(
                    batch_input_ids, 
                    attention_mask = attention_mask,
                    do_sample = True, 
                    max_new_tokens = args.max_sequence_length,
                    min_length = args.min_sequence_length,
                    temperature = args.temperature,
                    pad_token_id = tokenizer.eos_token_id
                )



                # attention_mask = output.ne(tokenizer.pad_token_id).long()
                # with torch.no_grad(): 
                #     inf_out = model(output, attention_mask = attention_mask)
                # batch_logits = inf_out.logits # [batch, num_toks, vocab_size]

                for j in range(num_seq_to_gen): 
                    # Decode the generated sequence
                    generated_text = tokenizer.decode(output[j], skip_special_tokens=False)

                    # # within generated_text, find prompt_q_str and replace with noprompt_q_str
                    # nosys_input_str = generated_text.replace(prompt_q_str, noprompt_q_str)
                    # nosys_input_ids = tokenizer(nosys_input_str, return_tensors="pt").to(model.device)['input_ids'][:, 1:]

                    # now do the same, but replace outputs[j] such that the subsequence prompt_q_ids becomes noprompt_q_ids
                    # create a dictionary with the required format 
                    # make an attention mask for the input_ids -- no attention to pad tokens 
                    generated_text_mask = torch.ones_like(output[j]) # [num_tokens]
                    generated_text_mask[:batch_input_ids.shape[1]] *= 0

                    gen_only = output[j][generated_text_mask == 1] # [num_generated_tokens]

                    nosys_input_ids = torch.cat([noprompt_q_ids[0], gen_only], dim=0) # [num_tokens]
                    
                    generated_text_mask[output[j] == tokenizer.eos_token_id] *= 0

                    length_diff = output[j].shape[0] - nosys_input_ids.shape[0]

                    example = {
                        "text": generated_text,  # full prompt + generated text (str)
                        "input_ids": output[j].tolist(), # full prompt + generated text (ids)
                        "attention_mask": attention_mask[j].tolist(), # eh
                        "prompt_text": prompt_q_str, # full prompt (str)
                        "prompt_text_nosys": noprompt_q_str, # prompt without system prompt (includes question though)
                        "prompt_input_ids": prompt_q_ids[0, :].tolist(), # 
                        "prompt_input_ids_nosys": noprompt_q_ids[0, :].tolist(), # prom
                        "text_nosys": tokenizer.decode(nosys_input_ids), # input ids with no system prompt (str)
                        "input_ids_nosys": nosys_input_ids.tolist(), # input ids with no system prompt (ids)
                        "generated_text_mask": generated_text_mask.tolist(), # applies to input_ids
                        "generated_text_mask_nosys": generated_text_mask[length_diff:].tolist() # mask for noprompt_input_ids
                        # "logits": batch_logits[j, :, :].tolist()
                    }

                    assert tokenizer.decode(nosys_input_ids[torch.tensor(example["generated_text_mask_nosys"])==1]) == tokenizer.decode(output[j][torch.tensor(example["generated_text_mask"])==1])

                    # writ ethe example to the jsonl file 
                    json.dump(example, f)
                    f.write("\n")

    print("Dataset saved to ", args.traj_out_file)
