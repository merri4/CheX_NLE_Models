!pip install evaluate
!pip install rouge-score
!pip install bert_score


import pandas as pd
import json
from nltk.translate.bleu_score import sentence_bleu
import evaluate
ROGUE = evaluate.load('rouge')
BERTSCORE = evaluate.load("bertscore")
METEOR = evaluate.load('meteor')


def bleu1(ori, pre) :
    ref = [ori.split()]
    cand = pre.split()
    return sentence_bleu(ref, cand, weights=(1, 0, 0, 0))

def bleu4(ori, pre) :
    ref = [ori.split()]
    cand = pre.split()
    return sentence_bleu(ref, cand, weights=(0, 0, 0, 1))

def rougeL(ori, pre) :
    global ROGUE
    ref = [[ori]]
    pred = [pre]
    results = ROGUE.compute(predictions=pred, references=ref)
    return results['rougeL']

def bertscore(ori, pre) :
    global BERTSCORE
    pred = [pre]
    ref = [ori]
    results = BERTSCORE.compute(predictions=pred, references=ref, lang="en")
    return results['f1']

def meteor(ori, pre) :
    global METEOR
    pred = [pre]
    ref = [ori]
    results = METEOR.compute(predictions=pred, references=ref)
    print(results['meteor'])

def read_jsonl_lines(input_path):
    with open(input_path) as f:
        lines = f.readlines()
        return [json.loads(l.strip()) for l in lines]

def compute_nle_metrics(ori_path) :


    # 메트릭 초기화
    densenet_scores = {
        "bleu1" : [],
        "bleu4" : [],
        "rougeL" : [],
        "bertscore" : [],
        "meteor" : [],
    }

    resnet_scores = {
        "bleu1" : [],
        "bleu4" : [],
        "rougeL" : [],
        "bertscore" : [],
        "meteor" : [],
    }

    vit_scores = {
        "bleu1" : [],
        "bleu4" : [],
        "rougeL" : [],
        "bertscore" : [],
        "meteor" : [],
    }

    all_lines = read_jsonl_lines(ori_path)
    all_nles = []

    for line in all_lines :
        all_nles.append({"ori" : line["original"], "densenet" : line["DenseNet"], "resnet" : line["ResNet"], "vit" : line["ViT"]})

    for nles in all_nles :
        
        densenet_scores["bleu1"].append(bleu1(nles["ori"], nles["densenet"]))
        densenet_scores["bleu4"].append(bleu4(nles["ori"], nles["densenet"]))
        densenet_scores["rougeL"].append(rougeL(nles["ori"], nles["densenet"]))
        densenet_scores["bertscore"].append(bertscore(nles["ori"], nles["densenet"]))
        densenet_scores["meteor"].append(meteor(nles["ori"], nles["densenet"]))

        resnet_scores["bleu1"].append(bleu1(nles["ori"], nles["resnet"]))
        resnet_scores["bleu4"].append(bleu4(nles["ori"], nles["resnet"]))
        resnet_scores["rougeL"].append(rougeL(nles["ori"], nles["resnet"]))
        resnet_scores["bertscore"].append(bertscore(nles["ori"], nles["resnet"]))
        resnet_scores["meteor"].append(meteor(nles["ori"], nles["resnet"]))

        vit_scores["bleu1"].append(bleu1(nles["ori"], nles["vit"]))
        vit_scores["bleu4"].append(bleu4(nles["ori"], nles["vit"]))
        vit_scores["rougeL"].append(rougeL(nles["ori"], nles["vit"]))
        vit_scores["bertscore"].append(bertscore(nles["ori"], nles["vit"]))
        vit_scores["meteor"].append(meteor(nles["ori"], nles["vit"]))

    # output form
    return {"densenet_scores" : densenet_scores, "resnet_scores" : resnet_scores, "vit_scores" : vit_scores}


if __name__ == "__main__" :

    # 파일을 불러온다.
    original_filepath = "/content/result.txt"

    result = compute_nle_metrics(original_filepath)

    df_densenet = pd.DataFrame.from_dict(result["densenet_scores"])
    df_resnet = pd.DataFrame.from_dict(result["resnet_scores"])
    df_vit = pd.DataFrame.from_dict(result["vit_scores"])

    df_densenet.to_csv("df_densenet.csv", sep=',', encoding='utf-8')
    df_resnet.to_csv("df_resnet.csv", sep=',', encoding='utf-8')
    df_vit.to_csv("df_vit.csv", sep=',', encoding='utf-8')



    


