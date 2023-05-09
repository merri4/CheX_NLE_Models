import json as js

def read_jsonl_lines(input_path):
    with open(input_path) as f:
        lines = f.readlines()
        return [js.loads(l.strip()) for l in lines]

def exist_cmd(cmd, cmdlist) :
    if cmd in cmdlist :
        return True
    return False




if __name__ == "__main__" :

    # nle 3개를 순차적으로 불러온다.

    DATA_SPLIT = {
        "train" : "mimic-nle-train",
        "dev" : "mimic-nle-dev",
        "test" : "mimic-nle-test",
    }

    PWD = "./MIMIC-NLE-extractor/mimic-nle/" # 여기 넣으시오!
    EXTENSION = ".json"
    OUTPUTLIST = []
    CMDTEMPLATE = 'gsutil -m cp -r "gs://mimic-cxr-jpg-2.0.0.physionet.org/files/{}/{}/{}" ./dataset/{}/{}'
    OUTPUT_PATH = "./batch_shell.sh"

    # 3가지에서 돌아가면서
    for split, filename in DATA_SPLIT.items() :
        
        filepath = PWD + filename + EXTENSION    
        all_lines = read_jsonl_lines(filepath)

        # parse data from line to complete the command
        for line in all_lines :
            cmd = CMDTEMPLATE.format(line['patient_ID'][:3], line['patient_ID'], line['report_ID'], split, line['patient_ID'])
            
            if not exist_cmd(cmd, OUTPUTLIST) :
                OUTPUTLIST.append(cmd)


    print("total commands : {}".format(len(OUTPUTLIST)))

    # 쉘 스크립트로 가공
    with open(OUTPUT_PATH, "w") as f :
        for cmd in OUTPUTLIST :
            f.write(cmd + "\n")


# 이런 식의 커맨드를 생성한다.

# gsutil -m cp -r "gs://mimic-cxr-jpg-2.0.0.physionet.org/files/{환자분류}/{환자번호}/{케이스번호}" ./dataset/{분류}
# gsutil -m cp -r "gs://mimic-cxr-jpg-2.0.0.physionet.org/files/p17/p17096560/s50056854" ./dataset/dev/p17096560

# gsutil -m cp -r "gs://mimic-cxr-jpg-2.0.0.physionet.org/p10" ./dev
# gsutil -m cp -r "gs://mimic-cxr-jpg-2.0.0.physionet.org/p10" ./test
# gsutil -m cp -r "gs://mimic-cxr-jpg-2.0.0.physionet.org/p10" ./train