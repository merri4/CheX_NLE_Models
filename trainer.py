
# =================================================================
# Dependency
# =================================================================
import os
import time
import datetime
import json 

import pandas as pd
import numpy as np
import random

import torch
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
# torch.manual_seed(42)

from transformers import GPT2LMHeadModel,  GPT2Tokenizer, GPT2Config, GPT2LMHeadModel, AutoTokenizer
from transformers import AdamW, get_linear_schedule_with_warmup



# =================================================================
# Functions
# =================================================================

def read_jsonl_lines(input_path):
    with open(input_path) as f:
        lines = f.readlines()
        return [json.loads(l.strip()) for l in lines]


def format_time(elapsed):
    return str(datetime.timedelta(seconds=int(round((elapsed)))))

def process_line(line, sep_token, eos_token) :
    sep_cut = line.split(sep_token, 1)
    eos_cut = sep_cut[1].split(eos_token, 1)    
    return eos_cut[0].strip()

# =================================================================
# Dataloader
# =================================================================

class MIMIC_NLE_Dataset(Dataset) :

    def __init__(self, path, split, tokenizer, max_length=128):

        self.filepath = path + "mimic-nle-" + split + ".json"
        
        self.diagnosis_list = [
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
        self.certaintiy_list = ["negative", "uncertain", "positive"]

        self.tokenizer = tokenizer
        
        self.input_ids = []
        self.attn_masks = []

        # 돌아가면서 만든다.

        all_lines = read_jsonl_lines(self.filepath)
        
        for line in all_lines :

            encodings_dict = tokenizer(self.get_prompt(line), truncation=True, max_length=max_length, padding="max_length")

            self.input_ids.append(torch.tensor(encodings_dict['input_ids']))
            self.attn_masks.append(torch.tensor(encodings_dict['attention_mask']))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.attn_masks[idx] 
    
    def get_prompt(self, line) :

        prompt = ""
        
        # diagnosis 정보 추가
        for idx, diagnosis in enumerate(line['img_labels']) :
            if diagnosis[1] :
                prompt += self.certaintiy_list[1] + self.diagnosis_list[idx] + ", "
            if diagnosis[2] :
                prompt += self.certaintiy_list[2] + self.diagnosis_list[idx] + ", "

        # NLE 추가
        prompt += "<|sep|>" + line['nle'] + "<|endoftext|>"
        
        return prompt



if __name__ == "__main__" :

    # =================================================================
    # Tokenizer & Dataloader
    # =================================================================

    PATH = "./"
    BATCH_SIZE = 16
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

    train_dataset = MIMIC_NLE_Dataset(PATH, "train", tokenizer)
    dev_dataset = MIMIC_NLE_Dataset(PATH, "dev", tokenizer)
    test_dataset = MIMIC_NLE_Dataset(PATH, "test", tokenizer)

    train_dataloader = DataLoader(
                train_dataset,
                sampler = RandomSampler(train_dataset), 
                batch_size = BATCH_SIZE 
            )

    dev_dataloader = DataLoader(
                dev_dataset,
                sampler = SequentialSampler(dev_dataset), 
                batch_size = BATCH_SIZE 
            )

    test_dataloader = DataLoader(
                test_dataset,
                sampler = RandomSampler(test_dataset),
                batch_size = BATCH_SIZE 
            )
        
    # =================================================================
    # Model loading
    # =================================================================
    
    # I'm not really doing anything with the config buheret
    configuration = GPT2Config.from_pretrained('gpt2', output_hidden_states=False)

    # instantiate the model
    model = GPT2LMHeadModel.from_pretrained("gpt2", config=configuration)

    # this step is necessary because I've added some tokens (bos_token, etc) to the embeddings
    # otherwise the tokenizer and model tensors won't match up
    model.resize_token_embeddings(len(tokenizer))

    # Tell pytorch to run this model on the GPU.
    device = torch.device("cuda")
    model.cuda()

    # If you want random....
    seed_val = 486
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)

    # =================================================================
    # Hyperparameter Setting
    # =================================================================

    epochs = 15
    warmup_steps = 1e2

    learning_rate = 5e-4
    epsilon = 1e-8
    optimizer = AdamW(model.parameters(), lr = learning_rate, eps = epsilon)

    total_steps = len(train_dataloader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, 
                                                num_warmup_steps = warmup_steps, 
                                                num_training_steps = total_steps)

    print("Total steps : {}".format(total_steps))
    
    # =================================================================
    # Training
    # =================================================================

    total_t0 = time.time()
    training_stats = []
    model = model.to(device) # go to GPU

    for epoch_i in range(0, epochs) :

        print('\n======== Epoch {:} / {:} ========'.format(epoch_i+1, epochs))
        print('Training...')

        t0 = time.time()

        total_train_loss = 0

        model.train() # 학습 모드 on

        # 여기서 step은 batch의 index고, batch는 실제 배치 값들. [0]에 id값이 있고, [1]에 mask 정보가 있음
        for step, batch in enumerate(train_dataloader) :

            b_input_ids = batch[0].to(device)   # input id
            b_labels    = batch[0].to(device)   # label
            b_masks     = batch[1].to(device)   # masking 정보

            model.zero_grad() # 가중치 초기화

            outputs = model(  b_input_ids,                  # batch id 넣고
                            labels=b_labels,              # 정답 넣고
                            attention_mask = b_masks,     # 마스킹 정보 넣고
                            token_type_ids=None           # 토큰 타입 NONE으로 넣어도 되는 거 맞오?
                            )

            # loss를 계산해서
            loss = outputs[0]
            batch_loss = loss.item()
            total_train_loss += batch_loss

            # 역전파
            loss.backward()

            # 옵티마이저, 스케줄러 세팅
            optimizer.step()
            scheduler.step()

        # Calculate the average loss over all of the batches.
        avg_train_loss = total_train_loss / len(train_dataloader)       
        
        # Measure how long this epoch took.
        training_time = format_time(time.time() - t0)

        print("")
        print("  Average training loss: {0:.2f}".format(avg_train_loss))
        print("  Training epoch took: {:}".format(training_time))
            
        # ========================================
        #               Validation
        # ========================================

        # 한 epoch마다 vaildation 수행
        print("")
        print("Running Validation...")

        t0 = time.time()

        model.eval()

        total_eval_loss = 0
        nb_eval_steps = 0

        # Evaluate data for one epoch
        for batch in dev_dataloader:
            
            b_input_ids = batch[0].to(device)
            b_labels = batch[0].to(device)
            b_masks = batch[1].to(device)
            
            with torch.no_grad() :

                outputs  = model(b_input_ids, 
                                # token_type_ids=None, 
                                attention_mask = b_masks,
                                labels=b_labels)
            
                loss = outputs[0]  
                
            batch_loss = loss.item()
            total_eval_loss += batch_loss        

        avg_val_loss = total_eval_loss / len(dev_dataloader)
        
        validation_time = format_time(time.time() - t0)    

        print("  Validation Loss: {0:.2f}".format(avg_val_loss))
        print("  Validation took: {:}".format(validation_time))

        # Record all statistics from this epoch.
        training_stats.append(
            {
                'epoch': epoch_i + 1,
                'Training Loss': avg_train_loss,
                'Valid. Loss': avg_val_loss,
                'Training Time': training_time,
                'Validation Time': validation_time
            }
        )

    print("")
    print("Training complete!")
    print("Total training took {:} (h:mm:ss)".format(format_time(time.time()-total_t0)))


        
    # =================================================================
    # 모델 저장
    # =================================================================

    torch.save(model.state_dict(), './model_weights.pth')


    # =================================================================
    # 프롬프트로 생성하기 TODO : dataloader로 한번에 원본 vs 생성 문장 뽑아내도록, metric call하는 파라미터에 쓸 수 있도록. json으로 뽑던지
    # =================================================================

    model.eval()

    prompt = "positive Pleural Other, " + SEP_TOKEN

    generated = torch.tensor(tokenizer.encode(prompt)).unsqueeze(0)
    generated = generated.to(device)

    print(generated)

    # Greedy
    sample_outputs = model.generate(
                                    input_ids = generated,
                                    do_sample=False,   
                                    top_k=50, 
                                    max_length = 128,
                                    top_p=0.95, 
                                    num_return_sequences=1,
                                    pad_token_id = tokenizer.sep_token_id,
                                    bos_token_id = tokenizer.sep_token_id,
                                    )

    # not greedy
    # sample_outputs = model.generate(
    #                                 input_ids = generated,
    #                                 do_sample=True,   
    #                                 top_k=50, 
    #                                 max_length = 128,
    #                                 top_p=0.95, 
    #                                 num_return_sequences=3,
    #                                 pad_token_id = tokenizer.sep_token_id,
    #                                 bos_token_id = tokenizer.sep_token_id,
    #                                 )


    for i, sample_output in enumerate(sample_outputs) :
        line = tokenizer.decode(sample_output, skip_special_tokens=False)
        print(process_line(line, SEP_TOKEN, EOS_TOKEN))

    # =================================================================
    # 모델 불러오기
    # =================================================================
    
    # TODO : 나중에 불러와서 재현하는 것까지 모듈화