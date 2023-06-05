"""오디오 파일과 데이터 파일 그리고 전사문으로 csv 파일 만들기
도별로 만듦
실행 : python aidata_split_file_list.py
맨 아래 main()에서 base_dir과 province code 설정하여 실행
"""
import os
import glob
import sys
import pandas as pd
import numpy as np
import json
import logging
from datetime import datetime
from pathlib import Path

# sys.path.append('../kdialectspeech')
# from time_convert import time_convert

logger = logging.getLogger(__name__)
log_file = sys.argv[0] + "_" + datetime.now().strftime('%Y-%m-%d_%H:%M:%S') + ".log"
logging.basicConfig(filename=log_file, level = logging.INFO, datefmt = '%Y-%m-%d %H%M%S'
                   ,format = '%(asctime)s | %(levelname)s | %(message)s')
streamHandler = logging.StreamHandler()
logger.addHandler(streamHandler)


def dir_of(base_dir, province):
    province_code = province

    if province_code == "gw":
        data_dir = os.path.join(base_dir, "139-1.중·노년층 한국어 방언 데이터 (강원도, 경상도)/06.품질검증/1.Dataset/1.원천데이터/01. 강원도")
        label_dir = os.path.join(base_dir, "139-1.중·노년층 한국어 방언 데이터 (강원도, 경상도)/06.품질검증/1.Dataset/2.라벨링데이터/01. 강원도")
    elif province_code == "gs":
        data_dir = os.path.join(base_dir, "139-1.중·노년층 한국어 방언 데이터 (강원도, 경상도)/06.품질검증/1.Dataset/1.원천데이터/02. 경상도")
        label_dir = os.path.join(base_dir, "139-1.중·노년층 한국어 방언 데이터 (강원도, 경상도)/06.품질검증/1.Dataset/2.라벨링데이터/02. 경상도")
    elif province_code == "cc":
        data_dir = os.path.join(base_dir, "139-2.중·노년층 한국어 방언 데이터 (충청도, 전라도, 제주도)/06.품질검증/1.Dataset/1.원천데이터/01. 충청도")
        label_dir = os.path.join(base_dir, "139-2.중·노년층 한국어 방언 데이터 (충청도, 전라도, 제주도)/06.품질검증/1.Dataset/2.라벨링데이터/01. 충청도")
    elif province_code == "jl":
        data_dir = os.path.join(base_dir, "139-2.중·노년층 한국어 방언 데이터 (충청도, 전라도, 제주도)/06.품질검증/1.Dataset/1.원천데이터/02. 전라도")
        label_dir = os.path.join(base_dir, "139-2.중·노년층 한국어 방언 데이터 (충청도, 전라도, 제주도)/06.품질검증/1.Dataset/2.라벨링데이터/02. 전라도")
    elif province_code == "jj":
        data_dir = os.path.join(base_dir, "139-2.중·노년층 한국어 방언 데이터 (충청도, 전라도, 제주도)/06.품질검증/1.Dataset/1.원천데이터/03. 제주도")
        label_dir = os.path.join(base_dir, "139-2.중·노년층 한국어 방언 데이터 (충청도, 전라도, 제주도)/06.품질검증/1.Dataset/2.라벨링데이터/03. 제주도")
    else:
        err_msg = (
            "지역 코드를 gw, gs, cc, jl, jj 중 하나를 입력해야 합니다. (gw:강원도, gs:경상도, cc:충청도, jl:전라도, jj:제주도)"
        )
        raise OSError(err_msg)
    
    return data_dir, label_dir


def file_list_of(base_dir, province):
    data_dir, label_dir = dir_of(base_dir, province)
    data_file_list = (glob.glob(os.path.join(data_dir, '*/*.' + "wav")))
    json_file_list = (glob.glob(os.path.join(label_dir, '*/*.' + "json")))

    return data_file_list, json_file_list


def manifest_of(json_file):
    data_id = Path(json_file).stem # 확장자 제거
    data_file = str(json_file).replace("2.라벨링데이터", "1.원천데이터",).replace(".json", ".wav")
    # print(f'data_file : {data_file}')
    
    with open(json_file, encoding="UTF-8" ) as j_f :
        json_data = json.load(j_f)

    try:
        recordDuration = json_data["audio"]["recordDuration"]
    except:
        logger.info(f"no recordDuration, file_name : {json_file}")
        recordDuration = 0.0

    try:
        dialect = json_data["transcription"]["dialect"]
    except:
        logger.info(f"no_dialect, file_name : {json_file}")
        dialect = "no_dialect"
    
    manifest_line = [data_id, data_file, json_file, recordDuration, dialect]
    return manifest_line


def manifest_of_list(json_file_list):
    manifest_lines = []
    for json_file in json_file_list:
        manifest_line = manifest_of(json_file)
        manifest_lines.append(manifest_line)

    return manifest_lines


def manifest_of_list(json_file_list):
    manifest_lines = []
    for json_file in json_file_list:
        manifest_line = manifest_of(json_file)
        manifest_lines.append(manifest_line)

    return manifest_lines



def main(base_dir, province_code):
    _, json_file_list = file_list_of(base_dir, province_code)
    manifest_list = manifest_of_list(json_file_list)
    manifest_df  = pd.DataFrame(data=manifest_list, columns=["id", "wav", "json", "recordDuration", "dialect"])

    manifest_file = Path(province_code + "_json.csv")
    manifest_df.to_csv(manifest_file, index=False)

    # split total csv into train, valid, test - 8:1:1
    manifest_train_df, manifest_valid_df, manifest_test_df = np.split(
        manifest_df.sample(frac=1, random_state=7774), 
        [int(.8*len(manifest_df)), int(.9*len(manifest_df))]
    )

    manifest_train_file = Path(province_code + "_train_json.csv")
    manifest_train_df.to_csv(manifest_train_file, index=False)

    manifest_valid_file = Path(province_code + "_valid_json.csv")
    manifest_valid_df.to_csv(manifest_valid_file, index=False)

    manifest_test_file = Path(province_code + "_test_json.csv")
    manifest_test_df.to_csv(manifest_test_file, index=False)



if __name__ == "__main__":
    base_dir = Path("/data/aidata")
    # province_code_list = ["jj", "jl", "gw", "gs", "cc"]
    province_code = sys.argv[1]

    main(base_dir, province_code)

