# Goal: init model, run lora to finetune weights, save model

import os
import json
import torch
import torch.nn as nn
import transformers
from transformers import AutoTokenizer
from datasets import load_dataset
from peft import get_peft_config, PeftModel, PeftConfig, get_peft_model, LoraConfig, TaskType
import torch.nn.functional as F
from datetime import datetime
import argparse
from tqdm import tqdm
import pdb


# Llama 3 system prompt + user prompt
# num_epochs = 1000
# batch_size = 10
# learning_rate = 1e-4
# device = "cuda" if torch.cuda.is_available() else "cpu"
# data_path = "data/traj_lex_nseq1000_maxlen300_minlen100_temp2.0.jsonl"
# out_dir = "results/traj_lex_01"

# Set up argument parser
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script for training model with specified parameters")

    # Add arguments with defaults
    parser.add_argument('--num_epochs', type=int, default=1000, help='Number of epochs to train the model. Default = 1000')
    parser.add_argument('--batch_size', type=int, default=10, help='Batch size for training. Default = 10')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate for optimizer. Default = 1e-4')
    parser.add_argument('--device', type=str, default="cuda" if torch.cuda.is_available() else "cpu", help='Device to run the training on. Default = "cuda" if available, else "cpu"')
    parser.add_argument('--data_path', type=str, default="data/traj_lex_nseq1000_maxlen300_minlen100_temp2.0.jsonl", help='Path to the training data. Default = "data/train_traj_temp2.0_numq100_numseq25_x0truth_20240718.jsonl"')
    parser.add_argument('--val_path', type=str, default="data/traj_lex_nseq1000_maxlen300_minlen100_temp2.0.jsonl", help='Path to the validation data. Default = "data/val_traj_temp2.0_numq25_numseq25_x0truth_20240718.jsonl"')
    parser.add_argument('--out_dir', type=str, default="results/traj_lex_01", help='Output directory for results. Default = "results/traj_lex_01"')
    parser.add_argument('-r', type=int, default=32, help='LoRA Rank')
    parser.add_argument('--save_every', type=int, default=1, help='Save model every n epochs. Default = 1')

    parser.add_argument('--max_traj_len', type=int, default=-1, help='Maximum usable trajectory length from the dataset. Default = -1 (use all)')
    # Parse arguments
    args = parser.parse_args()

    # Assign variables
    num_epochs = args.num_epochs
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    device = args.device
    data_path = args.data_path
    out_dir = args.out_dir

    # Ensure the output directory exists
    os.makedirs(out_dir, exist_ok=True)

    # Save arguments to a JSON file in the output directory
    args_dict = vars(args)
    args_json_path = os.path.join(out_dir, 'args.json')
    with open(args_json_path, 'w') as f:
        json.dump(args_dict, f, indent=4)

    print(f"Arguments saved to {args_json_path}")






# make the out_dir if it doesn't already exist 
import os
def log(msg, file_path): 
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(file_path, 'a') as f: 
        f.write(f"[{current_time}] {msg}\n")



if __name__ == "__main__":
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    log_path = os.path.join(out_dir, "train_loop.log")

    # TODO Load dataset here
    # Load the dataset from the JSONL file
    log("Loading dataset", log_path)
    dataset = load_dataset('json', data_files=data_path)
    log("Dataset loaded", log_path)

    # load validation dataset
    log("Loading validation dataset", log_path) 
    val_dataset = load_dataset('json', data_files=args.val_path)
    log("Validation dataset loaded", log_path)



def pad_list_of_lists(llist, pad_tok_val, verbose=False, pad_side='right', return_pad_mask=False):
    """
    Pads a list of lists with a padding token value.
    Right padding by default. 

    If return_pad_mask == True, then we return a corresponding list of list with 
    0's where we added padding and 1 where we have the original string. 
    """
    assert pad_side == 'left' or pad_side == 'right', "pad_side must be either 'left' or 'right'"

    max_len = max([len(l) for l in llist])
    if pad_side == 'right': 
        padded_list = [l + [pad_tok_val] * (max_len - len(l)) for l in llist]
    elif pad_side == 'left': 
        padded_list = [[pad_tok_val] * (max_len - len(l)) + l for l in llist]

    if verbose: 
        cnt = 0
        for l in llist: 
            if len(l) != max_len: 
                print(f"Unequal length list at batchel {cnt}: ", l)
                # print("Padded list: ", padded_list[cnt])
            cnt += 1
    
    if return_pad_mask: 
        num_pads_list = [(max_len - len(l)) for l in llist]
        pad_mask = [[0 if i < num_pads else 1 for i in range(max_len)] for num_pads in num_pads_list]
        if pad_side == 'right': 
            # reverse each sublist
            pad_mask = [l[::-1] for l in pad_mask]


        return padded_list, pad_mask

    return padded_list

def crop_trajectories(input_ids_list, mask_list, max_traj_len):
    """ Based on the masks which have ones where the trajectories are, we will 
    crop the input_ids and mask_list so that there are only at most max_traj_len 
    tokens in the trajectory. 
    """
    assert max_traj_len > 0, "max_traj_len must be greater than 0 in crop_trajectories()"

    # iterate through the mask_list and the input_ids list,
    # find the number of ones in the mask_list, 
    # crop the mask_list and input_ids from the right (cut off stuff at the end) 
    # that exceeds the max_traj_len
    new_input_ids_list = []
    new_mask_list = []

    for input_ids, mask in zip(input_ids_list, mask_list): 
        num_ones = sum(mask)
        # check if there are any zeros after the ones -- if so, error 
        assert len(input_ids) == len(mask), "input_ids and mask_ids must be the same length"

        first_1_ind = mask.index(1)
        input_ids_ = input_ids[:(first_1_ind + max_traj_len)]
        mask_ = mask[:(first_1_ind + max_traj_len)]
        
        new_input_ids_list.append(input_ids_)
        new_mask_list.append(mask_)

    return new_input_ids_list, new_mask_list


def do_epoch(peft_model, tokenizer,
             dataset, 
             batch_size, 
             log_path, 
             optimizer, 
             do_step = True, 
             max_traj_len=-1):
    """
    Trains the model for one epoch on the given dataset.
    """
    kl_divs = []
    for i in tqdm(range(0, len(dataset['train']), batch_size)):
        log(f"Batch {i}", log_path)
        batch = dataset['train'][i:i+batch_size]


        # grab the input_ids_nosys to run thru the PEFT model 
        input_ids_nosys_list_ = batch['input_ids_nosys'] # pad with tokenizer.pad_token_id
        input_ids_list_ = batch['input_ids'] # pad with tokenizer.pad_token_id

        # grab masks forr each input_ids
        mask_nosys_list_ = batch['generated_text_mask_nosys'] # pad with 0
        mask_list_ = batch['generated_text_mask'] # pad with 0


        if max_traj_len > 0: 
            input_ids_list_, mask_list_ = crop_trajectories(input_ids_list_, mask_list_, max_traj_len)
            input_ids_nosys_list_, mask_nosys_list_ = crop_trajectories(input_ids_nosys_list_, mask_nosys_list_, max_traj_len)


        # pad the lists of lists so that they're all the max length -- right padding
        input_ids_nosys_list = pad_list_of_lists(input_ids_nosys_list_, tokenizer.pad_token_id, verbose=False)
        input_ids_list = pad_list_of_lists(input_ids_list_, tokenizer.pad_token_id, verbose=False)

        mask_nosys_list = pad_list_of_lists(mask_nosys_list_, 0, verbose=False)
        mask_list = pad_list_of_lists(mask_list_, 0, verbose=False)



        device = peft_model.device
        input_ids = torch.tensor(input_ids_list).to(device)
        input_ids_nosys = torch.tensor(input_ids_nosys_list).to(device)
        mask = torch.tensor(mask_list).to(device) == 1
        mask_nosys = torch.tensor(mask_nosys_list).to(device) == 1



        assert input_ids.shape == mask.shape
        assert input_ids_nosys.shape == mask_nosys.shape

        assert (input_ids[mask] != input_ids_nosys[mask_nosys]).sum() == 0, "Prompted and unprompted input_ids do not match within their respective masks for the generated text (must be identical)"

        # per-row sum over mask 
        assert (mask.sum(dim=1) == mask_nosys.sum(dim=1)).all(), "Prompted and unprompted masks must have the same number of tokens per row"

        log("Computing unprompted logits...", log_path)
        unprompted_logits_ = peft_model(input_ids_nosys).logits
        log("Done computing prompted logits...", log_path)

        log("Computing prompted logits...", log_path)
        with peft_model.disable_adapter():
            with torch.no_grad():
                prompted_logits_ = peft_model(input_ids).logits
        log("Done computing prompted logits...", log_path)

        unprompted_logits = unprompted_logits_[mask_nosys, :]
        prompted_logits = prompted_logits_[mask, :]


        # Compute KL divergence: KL(prompted, unprompted)
        kl_div_ = F.kl_div(F.log_softmax(unprompted_logits, dim=-1), 
                        F.log_softmax(prompted_logits, dim=-1), 
                        reduction='none', 
                        log_target=True)
        
        kl_div = kl_div_.sum() / batch_size
        
        kl_div.backward()
        if do_step: 
            optimizer.step()
        optimizer.zero_grad()
        log(f"Done computing KL divergence = {kl_div.item()}", log_path)
        kl_divs.append(kl_div.item())

    print(f"Epoch {epoch} loss: {sum(kl_divs)/len(kl_divs)}")
    log(f"Epoch {epoch} loss: {sum(kl_divs)/len(kl_divs)}", log_path)
    return kl_divs



if __name__ == "__main__":
    # Load model
    
    # Initialize a tokenizer and model
    model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
    log(f"Loading model {model_name}...", log_path)

    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token

    pipeline = transformers.pipeline(
        "text-generation",
        model=model_name,
        tokenizer=tokenizer,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto",
    )
    model = pipeline.model

    log("Model loaded", log_path)

    log("Loading PEFT model...", log_path)
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=args.r,
        lora_alpha=1,
        lora_dropout=0,
        bias="none",
        target_modules=[
            "q_proj", 
            "v_proj", 
            "k_proj", 
            "o_proj"
        ]
    )
    
    peft_model = get_peft_model(model, peft_config)
    print("unprompted model parameter stats:")
    peft_model.print_trainable_parameters()
    log("PEFT model loaded", log_path)

    device = model.device

    optimizer = torch.optim.Adam(peft_model.parameters(), lr=learning_rate)

    # Run LORA
    # We are going to run both the unprompted model and the prompted model on the same batch of inputs
    # Then, we are going to compute the KL divergence between the two models' logits *for all tokens which BOTH models see*.
    # Therefore, we are not going to compute the KL divergence for the prompt that only the prompted model sees.
    # We are going to use the KL divergence as the loss function for LORA.
    # We are going to backpropagate through the unprompted model and update its weights.
    # The goal is to force the unprompted model to generate the same logits as the prompted model for the tokens that both models see.

    best_val_loss = 10000000000
    for epoch in range(num_epochs):
        log("Started epoch " + str(epoch), log_path)
        train_kls = do_epoch(peft_model, tokenizer,
                 dataset=dataset, 
                 batch_size=batch_size,
                 log_path=log_path,
                 optimizer=optimizer,
                 do_step=True, 
                 max_traj_len=args.max_traj_len)
        # log(f"All train kls (epoch={epoch}): ", train_kls)
        log(f"Train kl divergence (epoch={epoch}): {sum(train_kls)/len(train_kls)}", log_path)

        log("Done epoch " + str(epoch), log_path)

        log("Started validation epoch " + str(epoch), log_path)
        val_kls = do_epoch(peft_model, tokenizer,
                 dataset=val_dataset, 
                 batch_size=batch_size,
                 log_path=log_path,
                 optimizer=optimizer,
                 do_step=False, 
                 max_traj_len=args.max_traj_len)
        # log(f"All validation kls (epoch={epoch}): "+ str(val_kls), log_path)
        log(f"Validation kl divergence (epoch={epoch}): {sum(val_kls)/len(val_kls)}", log_path)
        log("Done validation epoch " + str(epoch), log_path)

        # save model every n epochs
        if (epoch+1) % args.save_every == 0:
            out_epoch_dir = os.path.join(out_dir, f"epoch_{epoch}")
            if not os.path.exists(out_epoch_dir):
                os.makedirs(out_epoch_dir)

            log(f"Saving PEFT model to {out_epoch_dir}...", log_path)
            peft_model.save_pretrained(out_epoch_dir)
            log(f"Model saved to {out_epoch_dir}", log_path)

        # save model
        if sum(val_kls)/len(val_kls) < best_val_loss:
            best_val_loss = sum(val_kls)/len(val_kls)
            log(f"Saving PEFT model to {out_dir}...", log_path)
            peft_model.save_pretrained(out_dir)
            log(f"Model saved to {out_dir}", log_path)
