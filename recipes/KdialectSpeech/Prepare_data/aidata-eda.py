"""전체 데이터에 대한 EDA
전체 json file에서 시간 duration과 방언전사문 읽기
지역별 전체 duration 합계 계산
"""

import os
import sys
from pathlib import Path
import glob
import json
import logging

from datetime import datetime
import pandas as pd

sys.path.append('../kdialectspeech')
from time_convert import time_convert


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
    
    with open(json_file, encoding="UTF-8" ) as json_file :
        json_data = json.load(json_file)

    try:
        speechStartTime = json_data["audio"]["speechStartTime"]
        speechStartTime = format(time_convert(speechStartTime), '5.3f') 
    except:
        logger.info(f"no speechStartTime, file_name : {json_file}")
        speechStartTime = 0.0
    
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
    
    manifest_line = [data_id, speechStartTime, recordDuration, dialect]
    return manifest_line


def manifest_of_list(json_file_list):
    manifest_lines = []
    for json_file in json_file_list:
        manifest_line = manifest_of(json_file)
        manifest_lines.append(manifest_line)

    return manifest_lines



def main(base_dir, province_code_list):
    for province in province_code_list:
        _, json_file_list = file_list_of(base_dir, province)
        manifest_list = manifest_of_list(json_file_list)
        manifest_df  = pd.DataFrame(data=manifest_list, columns=["id", "speechStartTime", "recordDuration", "dialect"])

        # total_speechStartTime = manifest_df["speechStartTime"].sum()/3600 # sec of an hour
        # logger.info(f"total speech StartTime of {province} : {total_speechStartTime}")

        # total_recordDuration = manifest_df["recordDuration"].sum()/3600 # sec of an hour
        # logger.info(f"total record duration of {province} : {total_recordDuration}")

        manifest_file = Path(province + "_json.csv")
        manifest_df.to_csv(manifest_file, index=False)

if __name__ == "__main__":
    base_dir = Path("/data/aidata.2")
    province_code_list = ["jj", "jl", "gw", "gs", "cc"]

    main(base_dir, province_code_list)


# for province_code in province_code_list:
#     data_file_list, json_file_list = file_list_of(province_code)
#     print(f'# of {province_code} data file list :{len(data_file_list)}')
#     print(f'# of {province_code} json file list :{len(json_file_list)}')