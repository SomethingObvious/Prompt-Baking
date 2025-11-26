# Goal. Use huggingface rajpurkar/squad dataset and convert it to a .jsonl file


import json
import os
from tqdm import tqdm
from datasets import load_dataset
import pdb
import requests
import xml.etree.ElementTree as ET



# Load the dataset
print("Loading SQuAD dataset...")
dataset = load_dataset("squad")

# Save the dataset to a .jsonl file
os.makedirs("data", exist_ok=True)
with open("data/squad_train.jsonl", "w") as f:
    for example in tqdm(dataset["train"]):
        json.dump(example, f)
        f.write("\n")

with open("data/squad_validation.jsonl", "w") as f:
    for example in tqdm(dataset["validation"]):
        json.dump(example, f)
        f.write("\n")
print("SQuAD dataset has been successfully downloaded and saved as jsonl files.")

print("Loading GSM8k dataset...")
# Now load GSM8K benchmark of math word problems (Cobbe et al., 2021)
dataset = load_dataset("gsm8k", "main")

with open("data/gsm8k_train.jsonl", "w") as f:
    for example in tqdm(dataset["train"]):
        json.dump(example, f)
        f.write("\n")

with open("data/gsm8k_validation.jsonl", "w") as f:
    for example in tqdm(dataset["test"]):
        json.dump(example, f)
        f.write("\n")
print("GSM8K dataset has been successfully downloaded and saved as jsonl files.")
# Now load the SVAMP dataset of math word problems with varying structures (Patel et al., 2021)

print("Loading SVAMP dataset...")
# Function to download the SVAMP dataset
url = "https://raw.githubusercontent.com/arkilpatel/SVAMP/main/SVAMP.json"
response = requests.get(url)
if response.status_code == 200:
    svamp_data = response.json()
    # Split the data into train and validation sets (80-20 split)
    split_index = int(len(svamp_data) * 0.8)
    train_data = svamp_data[:split_index]
    validation_data = svamp_data[split_index:]

    with open("data/svamp_train.jsonl", "w") as f:
        for example in tqdm(train_data):
            json_from_example = {}
            json_from_example["id"] = example["ID"]
            json_from_example["question"] = example["Body"] + " " + example["Question"]
            json_from_example["equation"] = example["Equation"]
            json_from_example["answer"] = example["Answer"]
            json_from_example["type"] = example["Type"]
            json.dump(json_from_example, f)
            f.write("\n")

    with open("data/svamp_validation.jsonl", "w") as f:
        for example in tqdm(validation_data):
            json_from_example = {}
            json_from_example["id"] = example["ID"]
            json_from_example["question"] = example["Body"] + " " + example["Question"]
            json_from_example["answer"] = example["Answer"]
            json_from_example["equation"] = example["Equation"]
            json_from_example["type"] = example["Type"]
            json.dump(json_from_example, f)
            f.write("\n")
    print("SVAMP dataset has been successfully downloaded and saved as jsonl files.")

else:
    raise Exception(f"Failed to download SVAMP dataset. Status code: {response.status_code}")



# Now load the ASDiv dataset of diverse math word problems (Miao et al., 2020)
print("Loading ASDiv dataset...")
url = "https://raw.githubusercontent.com/chaochun/nlu-asdiv-dataset/master/dataset/ASDiv.xml"
response = requests.get(url)
if response.status_code == 200:
    xml_data = response.text
    root = ET.fromstring(xml_data)
    problems = []
    for problem in root.iter('Problem'):
        problem_dict = {
            'id': problem.get('ID'),
            'grade': problem.get('Grade'),
            'source': problem.get('Source'),
            'body': problem.find('Body').text,
            'question': problem.find('Body').text + " " + problem.find('Question').text,
            'solution_type': problem.find('Solution-Type').text,
            'answer': problem.find('Answer').text,
            'formula': problem.find('Formula').text
        }
        problems.append(problem_dict)
    # Split the data into train and validation sets (80-20 split)
    split_index = int(len(problems) * 0.8)
    train_data = problems[:split_index]
    validation_data = problems[split_index:]

    with open("data/asdiv_train.jsonl", "w") as f:
        for example in tqdm(train_data):
            json.dump(example, f)
            f.write("\n")

    with open("data/asdiv_validation.jsonl", "w") as f:
        for example in tqdm(validation_data):
            json.dump(example, f)
            f.write("\n")
    print("ASDiv dataset has been successfully downloaded and saved as jsonl files.")
else:
    raise Exception(f"Failed to download ASDIV dataset. Status code: {response.status_code}")

# Now load the AQuA dataset of algebraic word problems

dataset = load_dataset("aqua_rat")

with open("data/aqua_train.jsonl", "w") as f:
    for example in tqdm(dataset["train"]):
        json.dump(example, f)
        f.write("\n")

with open("data/aqua_validation.jsonl", "w") as f:
    for example in tqdm(dataset["validation"]):
        json.dump(example, f)
        f.write("\n")

with open("data/aqua_test.jsonl", "w") as f:
    for example in tqdm(dataset["test"]):
        json.dump(example, f)
        f.write("\n")

print("AQuA dataset has been successfully downloaded and saved as jsonl files.")

# Now load the MathQA dataset of math word problems (Amini et al., 2019) Note : Newer datasets disallow remote loading scripts math_qa.py.

# print("Loading MathQA dataset...")
# dataset = load_dataset("math_qa", trust_remote_code=True)

# with open("data/mathqa_train.jsonl", "w") as f:
    # for example in tqdm(dataset["train"]):
        # Renmae "Problem" key to "question"
        # example["question"] = example["Problem"]
        # del example["Problem"]
        # json.dump(example, f)
        # f.write("\n")


# with open("data/mathqa_validation.jsonl", "w") as f:
    # for example in tqdm(dataset["validation"]):
        # example["question"] = example["Problem"]
        # del example["Problem"]
        # json.dump(example, f)
        # f.write("\n")

# with open("data/mathqa_test.jsonl", "w") as f:
    # for example in tqdm(dataset["test"]):
        # example["question"] = example["Problem"]
        # del example["Problem"]
        # json.dump(example, f)
        # f.write("\n")
# print("MathQA dataset has been successfully downloaded and saved as jsonl files.")