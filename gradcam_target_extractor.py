import pandas as pd
import json as js
from csv import DictReader

def read_jsonl_lines(input_path):
    with open(input_path) as f:
        lines = f.readlines()
        return [js.loads(l.strip()) for l in lines]



df = pd.read_excel('result/df_total.xlsx', sheet_name="total")
col_val = df['VIT_NUM'].tolist()
col_val = col_val[:-5]
col_val = [int(col_val[i]) for i in range(len(col_val))]


filepath = "result/mimic-nle-test.json"

all_lines = read_jsonl_lines(filepath)

report_ids = []
img_labels = []
for line in all_lines :
    report_ids.append(line["report_ID"])
    img_labels.append(line["img_labels"])


csv_path = "result/mimic-cxr-2.0.0-split.csv"
df_meta = pd.read_csv(csv_path)


all_data = []

for i in range(len(report_ids)) :

    if i and report_ids[i-1] != report_ids[i] :
    
        study_id = report_ids[i]
        img_label = img_labels[i]
        ViT_num = col_val[i]
        image_name = df_meta.loc[ df_meta['study_id'] == int(study_id[1:]) ]["dicom_id"].to_list()
        
        tmpdict = {
            "id" : study_id,
            "ViT_num" : ViT_num,
            "img_labels" : img_label,
            "image_name" : image_name,
        }

        all_data.append(tmpdict)


with open("result/grad_targets.json", "w") as f :
    
    for line in all_data :
        tmp = js.dumps(line)
        f.write(tmp + "\n")