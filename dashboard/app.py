import streamlit as st

import os
import time
import socket
import uuid
from datetime import datetime
import hashlib
import json

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import glob

# Function to get the client's IP address
def get_client_ip():
    return socket.gethostbyname(socket.gethostname())

# Function to get or create a session ID
def get_session_id():
    if 'session_id' not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    return st.session_state.session_id

def log(msg, file_path="dashboard/app.log"): 
    with open(file_path, 'a') as f:
        f.write(f"[{datetime.now()}] {msg}\n")



# Main Streamlit app
def main():
    st.title("Baking in Chain-of-Thought Prompts for Math QA")
    st.write("""
    This is a dashboard to show the results of "baking in" chain-of-thought prompts into Llama-3 8b Instruct. We trained 72 different models on 3 math datasets with the new "baking" technique for 40 total epochs (see Overleaf/outline). 
             
    Results are stored in [/mnt/arc/archive/pwe/cam/results/ballmer_20240726/cotqa/](https://lancelot.languagegame.io/archive/pwe/cam/results/ballmer_20240726/cotqa/) (ask Aman for the password). 

    For each results folder in `RESULTS_BASE/*/`, there are currently 4 epochs tested (9, 19, 29, 39) with the accuracies stored in `RESULTS_BASE/*/mathest*.json`.
             
    During these experiments, we primarily varied the following parameters: 
     - **Dataset**: asdiv, gsm8k, svamp
     - **Learning rate**: 1e-4, 3e-4
     - **Temperature**: 1.0, 2.0, 3.0, 5.0
     - **LoRA Rank**: 2, 8, 32

    For each experiment, we have 3 models: 
     - **Base + CoT Prompt**: Standard Llama-3 8b Instruct with the CoT prompt.
     - **Baked**: LoRA-tuned Llama-3 8b Instruct trained to match **Base + CoT Prompt** (empty system prompt).
     - **Baked + CoT Prompt**: LoRA-tuned Llama-3 8b Instruct to match **Base + CoT Prompt**, AND we give it the CoT prompt.
    """)

    model_name_translate = {
        "base": "Base + CoT Prompt", 
        "peft": "Baked",
        "peft_sys": "Baked + CoT Prompt"
    }


    RESULTS_BASE = "/mnt/arc/archive/pwe/cam/results/ballmer_20240726/cotqa"
    dataset_names = ["asdiv", "gsm8k", "svamp"]
    # for each dropdown options, glob the directories and get the mathest*.json files
    # for each mathest*.json file, load the json and get the results
    # for each results, get the prompt, the weight, and the accuracy
    # create a dataframe with these columns

    result_folder_list = {}
    all_results_folders = []
    for option in dataset_names:
        glob_string = os.path.join(RESULTS_BASE, f"{option}*")
        # st.write("Glob string: ", glob_string)
        result_folder_list[f"{option}"] = glob.glob(glob_string)
        all_results_folders.extend(result_folder_list[f"{option}"])

    # find the unique learning rates 
    lrs = []
    for folder_name in all_results_folders:
        # find the learning rate by matching the substring "lr_{LR_FLOAT_VALUE}" in the folder name
        lr = float(folder_name.split("lr_")[1].split("_")[0])
        lrs.append(lr)
    lrs = list(set(lrs))
    lrs.sort()
    # st.write("Learning rates: ", lrs)












    # New section for clustered bar plot
    st.write("## Summary Bar Plot")

    # Create a new dataframe with all datasets
    all_results_folders = []
    for option in dataset_names:
        glob_string = os.path.join(RESULTS_BASE, f"{option}*")
        all_results_folders.extend(glob.glob(glob_string))

    all_data_rows = []
    for folder_name in all_results_folders:
        temp = float(folder_name.split("temperature_")[1].split("_")[0])
        lr = float(folder_name.split("lr_")[1].split("_")[0])
        rank_r = int(folder_name.split("_r_")[1].split("_")[0])
        
        json_files = glob.glob(os.path.join(folder_name, "mathtest*.json"))
        for json_file in json_files:
            epoch_num = int(json_file.split("_ep")[1].split("_")[0])
            with open(json_file, 'r') as f:
                json_data = json.load(f)
            
            row = {
                "folder_name": folder_name,
                "temperature": temp,
                "lr": lr,
                "rank_r": rank_r,
                "epoch_num": epoch_num,
                "dataset": folder_name.split('/')[-1].split('_')[0],
            }
            for metric in ['mean_accuracy', 'mean_accuracy_upper_bound', 'mean_accuracy_last_sentence']:
                for model_type in ['base', 'peft', 'peft_sys']:
                    key = f"{metric}_{model_type}"
                    row[key] = json_data.get(key, None)
            
            all_data_rows.append(row)

    all_datasets_df = pd.DataFrame(all_data_rows)

    # Dropdowns for filtering
    epochs = sorted(all_datasets_df['epoch_num'].unique())
    temperatures = sorted(all_datasets_df['temperature'].unique())
    ranks = sorted(all_datasets_df['rank_r'].unique())
    accuracy_metrics = ["mean_accuracy", "mean_accuracy_upper_bound", "mean_accuracy_last_sentence"]
    learning_rates = sorted(all_datasets_df['lr'].unique())

    selected_epoch = st.selectbox("Select Epoch", epochs)
    selected_temp = st.selectbox("Select Temperature", temperatures)
    selected_rank = st.selectbox("Select LoRA Rank", ranks)
    learning_rate = st.selectbox("Select Learning Rate", learning_rates)
    # selected_metric = st.selectbox("Select Accuracy Metric", accuracy_metrics)
    selected_metric = "mean_accuracy"
    # checkbox for whether we include the baselines 
    include_baselines = st.checkbox("Include Baselines from Original CoT Paper")

    # Filter data based on selections
    filtered_df = all_datasets_df[(all_datasets_df['epoch_num'] == selected_epoch) & 
                                  (all_datasets_df['temperature'] == selected_temp) & 
                                  (all_datasets_df['rank_r'] == selected_rank) & 
                                  (all_datasets_df['lr'] == learning_rate)]

    # Prepare data for plotting
    datasets = filtered_df['dataset'].unique()
    model_types = ['base', 'peft', 'peft_sys']

    # Create traces for each model type

    # load json from dashboard/og_cot_accuracies.jsonl 
    # Load the JSON file
    with open('dashboard/og_cot_accuracies.json', 'r') as file:
        og_cot_accuracies = json.load(file)

    traces = []
    for model_type in model_types:
        accuracies = []
        hover_texts = []
        for dataset in datasets:
            dataset_data = filtered_df[filtered_df['dataset'] == dataset]
            accuracy = dataset_data[f'{selected_metric}_{model_type}'].mean()
            accuracies.append(accuracy)
            folder = dataset_data['folder_name'].iloc[0] if not dataset_data.empty else 'N/A'
            # hover_texts.append(f"Accuracy: {accuracy:.4f}<br>Folder: {folder}")
            hover_texts.append(f"Accuracy: {accuracy:.4f}")

        traces.append(go.Bar(
            name=model_name_translate[model_type]+" (Llama-3 8b Instruct)",
            x=datasets,
            y=accuracies,
            text=hover_texts,
            hoverinfo='text'
        ))

    # loop through the relevant fields of og_cot_accuracies and add their traces. 
    # Models we want to include from the original paper
    if include_baselines:
        original_models = ['gpt_3_175b', 'palm_540b', 'codex']

        for model in original_models:
            accuracies = []
            hover_texts = []
            for dataset in datasets:
                # Convert dataset name to lowercase to match JSON keys
                dataset_lower = dataset.lower()
                if dataset_lower in og_cot_accuracies['prompting'][model]['chain_of_thought']:
                    accuracy = og_cot_accuracies['prompting'][model]['chain_of_thought'][dataset_lower] / 100.0
                    accuracies.append(accuracy)
                    hover_texts.append(f"Accuracy: {accuracy:.4f}<br>Model: {model}")
                else:
                    accuracies.append(None)
                    hover_texts.append(f"No data for {dataset} with {model}")
            if model == "codex": 
                model_ = "codex_175b"
            else: 
                model_ = model
            traces.append(go.Bar(
                name=f"{model_} (original, CoT prompt)",
                x=datasets,
                y=accuracies,
                text=hover_texts,
                hoverinfo='text'
            ))


    fig = go.Figure(data=traces)

    # Create the figure
    # ensure y-axis go from 0 to 1
    fig.update_yaxes(range=[0, 1])


    # Update layout
    fig.update_layout(
        title=f"{selected_metric.replace('_', ' ').title()} -- epoch = {selected_epoch}, T = {selected_temp}, r = {selected_rank}, lr = {learning_rate}, metric = {selected_metric}",
        xaxis_title="Dataset",
        yaxis_title="Accuracy",
        barmode='group',
        height=600
    )

    # Display the plot
    st.plotly_chart(fig)





    














    
    # st.write(result_folder_list)
    st.write("## CoT Math Dataset Results")
    # make a dropdown menu for the dataset 
    dataset_option = st.selectbox("Select a dataset", dataset_names)
    # make a dropdown menu for the learning rate
    lr_option = st.selectbox("Select a learning rate", lrs)
    
    # now we need to filter the results folders by the dataset and learning rate
    filtered_results_folders = []
    for folder_name in result_folder_list[dataset_option]:
        lr = float(folder_name.split("lr_")[1].split("_")[0])
        if lr == lr_option:
            filtered_results_folders.append(folder_name)
        
    # st.write("Filtered results folders: ", filtered_results_folders)

    # for each folder in filtered_results, there are 4 data files, one for each 
    # epoch tested. They are of the form mathtest_ep19_numq400_gentemp0.0_datasetnameasdiv.json. 
    # we can find them by globbing for mathtest*.json in the folder. 
    
    filtered_results_to_json = {}
    for folder_name in filtered_results_folders:
        json_files = glob.glob(os.path.join(folder_name, "mathtest*.json"))
        filtered_results_to_json[folder_name] = {} 
        for json_file in json_files:
            # find the epoch number. 
            epoch_num = json_file.split("_ep")[1].split("_")[0]
            # read the json file 
            with open(json_file, 'r') as f:
                json_data = json.load(f)
            filtered_results_to_json[folder_name][epoch_num] = json_data

    # st.write("Filtered results to json: ", filtered_results_to_json)


    # now we will construct a dataframe with columns for the folder name (extracting the temperature, learning rate, rank r as columns too), the epoch number, and "mean_accuracy_base", "mean_accuracy_peft_sys", "mean_accuracy_upper_bound_base", "mean_accuracy_upper_bound_peft_sys", "mean_accuracy_last_sentence_base", "mean_accuracy_last_sentence_peft_sys", ...

    df = pd.DataFrame(columns=["folder_name", "temperature", "lr", "rank_r", "epoch_num", "mean_accuracy_base", "mean_accuracy_peft", "mean_accuracy_peft_sys", "mean_accuracy_upper_bound_base", "mean_accuracy_upper_bound_peft", "mean_accuracy_upper_bound_peft_sys", "mean_accuracy_last_sentence_base", "mean_accuracy_last_sentence_peft", "mean_accuracy_last_sentence_peft_sys"])

    # Recall that the folder names are of the form results/ballmer_20240726/cotqa/asdiv_train_temperature_2.0_lr_0.0001_r_2
    # TODO: build the dataframe from the loaded data. 
    # Construct the dataframe
    rows = []
    for folder_name, epoch_data in filtered_results_to_json.items():
        # Extract temperature, lr, and rank_r from folder name
        temp = float(folder_name.split("temperature_")[1].split("_")[0])
        lr = float(folder_name.split("lr_")[1].split("_")[0])
        rank_r = int(folder_name.split("_r_")[1].split("_")[0])
        
        for epoch_num, json_data in epoch_data.items():
            row = {
                "folder_name": folder_name,
                "temperature": temp,
                "lr": lr,
                "rank_r": rank_r,
                "epoch_num": int(epoch_num),
                "mean_accuracy_base": json_data.get("mean_accuracy_base", None),
                "mean_accuracy_peft": json_data.get("mean_accuracy_peft", None),
                "mean_accuracy_peft_sys": json_data.get("mean_accuracy_peft_sys", None),
                "mean_accuracy_upper_bound_base": json_data.get("mean_accuracy_upper_bound_base", None),
                "mean_accuracy_upper_bound_peft": json_data.get("mean_accuracy_upper_bound_peft", None), 
                "mean_accuracy_upper_bound_peft_sys": json_data.get("mean_accuracy_upper_bound_peft_sys", None),
                "mean_accuracy_last_sentence_base": json_data.get("mean_accuracy_last_sentence_base", None),
                "mean_accuracy_last_sentence_peft": json_data.get("mean_accuracy_last_sentence_peft", None),
                "mean_accuracy_last_sentence_peft_sys": json_data.get("mean_accuracy_last_sentence_peft_sys", None)
            }
            rows.append(row)
    
    df = pd.DataFrame(rows)


    # menu for selecting accuracy metric 
    accuracy_metrics = ("mean_accuracy", "mean_accuracy_upper_bound", "mean_accuracy_last_sentence")
    accuracy_metric_option = st.selectbox("Select an accuracy metric", accuracy_metrics)

    # you can check if the column starts with accuracy_metric_option to see if we should plot it. 
    # Now we will make a line plot for the accuracy metric versus the epoch. 
    # each line represents one experiment folder (i.e., group by folder name). 
    # then use the accuracy_metric_option to select the columns to plot.
    # we will start with just one line plot. 

    # Create a 4x3 grid of plots for the selected accuracy metric
    # sort by epoch 
    df = df.sort_values(by='epoch_num')
    st.write(f"### {accuracy_metric_option.replace('_', ' ').title()} vs Epoch")

    # Get unique temperature and rank_r values
    temperatures = sorted(df['temperature'].unique())
    ranks = sorted(df['rank_r'].unique())

    fig, axs = plt.subplots(4, 3, figsize=(20, 24), sharex=True, sharey=True)
    fig.suptitle(f"{accuracy_metric_option.replace('_', ' ').title()} vs Epoch for {dataset_option} (LR: {lr_option})", fontsize=16)

    for i, temp in enumerate(temperatures):
        for j, rank in enumerate(ranks):
            ax = axs[i, j]
            subset = df[(df['temperature'] == temp) & (df['rank_r'] == rank)]
            
            for model_type in ['base', 'peft', 'peft_sys']:
                column_name = f"{accuracy_metric_option}_{model_type}"
                if column_name in df.columns:
                    ax.plot([int(e) for e in subset['epoch_num']], subset[column_name], 
                            marker='o', label=model_type)
            
            ax.set_title(f"Temp: {temp}, Rank: {rank}")
            ax.set_xlabel("Epoch")
            ax.set_ylabel(accuracy_metric_option.replace('_', ' ').title())
            ax.legend()

    # Remove any empty subplots
    for i in range(4):
        for j in range(3):
            if i >= len(temperatures) or j >= len(ranks):
                fig.delaxes(axs[i, j])

    # plt.tight_layout()
    st.pyplot(fig) 

    # checkbox: show df 
    if st.checkbox("Show Full Dataframe"):
        st.write(df)








    


if __name__ == "__main__":
    main()

