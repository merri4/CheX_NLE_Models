
# ================================================================
# DEP
# ================================================================

!pip install transformers

import os
import time
import datetime
import json 

import pandas as pd
import numpy as np
import random

import torch
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler

from transformers import GPT2LMHeadModel,  GPT2Tokenizer, GPT2Config, GPT2LMHeadModel, AutoTokenizer
from transformers import AdamW, get_linear_schedule_with_warmup

from google.colab import drive
drive.mount('/content/drive')

# ================================================================
# tokenizer 
# ================================================================

BOS_TOKEN = "<|startoftext|>"
EOS_TOKEN = "<|endoftext|>"
SEP_TOKEN = "<|sep|>"
PAD_TOKEN = "<|pad|>"

tokenizer = GPT2Tokenizer.from_pretrained(
        'gpt2',
        bos_token=BOS_TOKEN,
        eos_token=EOS_TOKEN,
        sep_token=SEP_TOKEN,
        pad_token=PAD_TOKEN,
        )


# ================================================================
# pre-trained model loading
# ================================================================

# I'm not really doing anything with the config buheret
configuration = GPT2Config.from_pretrained('gpt2', output_hidden_states=False)

# instantiate the model
mount_path = ""
model = GPT2LMHeadModel.from_pretrained(mount_path, config=configuration)

# this step is necessary because I've added some tokens (bos_token, etc) to the embeddings
# otherwise the tokenizer and model tensors won't match up
model.resize_token_embeddings(len(tokenizer)) # 리사이징때문에 모델이 제대로 안붙는 거 같은데

# Tell pytorch to run this model on the GPU.
device = torch.device("cuda")
model.cuda()

# If you want random....
seed_val = 486
random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)

# ================================================================
# Dataset Test
# ================================================================

def process_line(line, sep_token, eos_token) :
    sep_cut = line.split(sep_token, 1)
    eos_cut = sep_cut[1].split(eos_token, 1)    
    return eos_cut[0].strip()

def read_jsonl_lines(input_path):
    with open(input_path) as f:
        lines = f.readlines()
        return [json.loads(l.strip()) for l in lines]
    
def get_prompt(line, diagnosis_list, certaintiy_list) :

    prompt = ""
    
    # diagnosis 정보 추가
    for idx, diagnosis in enumerate(line['img_labels']) :
        if diagnosis[1] :
            prompt += certaintiy_list[1] + diagnosis_list[idx] + ", "
        if diagnosis[2] :
            prompt += certaintiy_list[2] + diagnosis_list[idx] + ", "

    prompt += "<|sep|>"
    
    return prompt


diagnosis_list = [
    "Atelectasis",
    "Consolidation",
    "Edema",
    "Enlarged Cardiomediastinum",
    "Lung Lesion",
    "Lung Opacity",
    "Pleural Effusion",
    "Pleural Other",
    "Pneumonia",
    "Pneumothorax",
    ]
certaintiy_list = ["negative", "uncertain", "positive"]


model.eval()

PATH = "./MIMIC-NLE-extractor/mimic-nle-test.json"

all_lines = read_jsonl_lines(PATH)
testset = []
result = []

for line in all_lines :
    testset.append(get_prompt(line, diagnosis_list, certaintiy_list))
    result.append({"original" : line["nle"]})



for idx_input, prompt in enumerate(testset) :

    generated = torch.tensor(tokenizer.encode(prompt)).unsqueeze(0)
    generated = generated.to(device)

    # Greedy
    sample_outputs = model.generate(
                                    input_ids = generated,
                                    do_sample=True,   
                                    top_k=50, 
                                    max_length = 128,
                                    top_p=0.95, 
                                    num_return_sequences=3,
                                    pad_token_id = tokenizer.sep_token_id,
                                    bos_token_id = tokenizer.sep_token_id,
                                    )

    for idx_output, sample_output in enumerate(sample_outputs) :
        line = process_line(tokenizer.decode(sample_output, skip_special_tokens=False), SEP_TOKEN, EOS_TOKEN)
        
        if idx_output == 0 :
            tmp = {"ViT" : line}

        elif idx_output == 1 :
            tmp = {"ResNet" : line}

        else :
            tmp = {"DenseNet" : line}
        
        result[idx_input].update(tmp)


output_path = "result.txt"
with open(output_path, "w") as f :
    for line in result :
        f.write(line + "\n")