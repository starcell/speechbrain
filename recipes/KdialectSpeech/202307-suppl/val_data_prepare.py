"""
Data preparation.

Author
------
N Park 2023 # 유효성 검증용

2022-02-018 중노년층방언데이터 구축사업 유효성 검증용 데이터 전처리
검증용 데이터 목록을 받아서 음성인식을 하고 인식 성능 측정을 위한 데이터 준비를 수행
입력 : 검증용 데이터 목록 파일(검증용 라벨 데이터의 전체 경로 목록)
출력 : 검증용 매니페스트 파일, 문장단위로 분리된 음성파일(wav file)

apt install ffmpeg
pydub는 ffmpeg를 필요로 함.
"""
import logging
import os
import glob
from pathlib import Path
import re
from speechbrain.dataio.dataio import load_pkl, merge_csvs, save_pkl
# from speechbrain.utils.data_utils import get_all_files

## Kang
import pandas as pd
import json
import os
from pydub import AudioSegment

logger = logging.getLogger(__name__)

OPT_FILE = "opt_kdialectspeech_prepare.pkl"
SAMPLERATE = 16000


# 1. 목록 파일 읽기

DATA_DIR = Path("/data/nia")
# 입력 파일들
provinces = ["충청도", "전라도", "제주도", "강원도", "경상도"]
speech_kinds = ["따라말하기", "질문에답하기", "2인발화"]
# input_files = ["충청도_따라말하기.csv", "충청도_질문에답하기.csv", "충청도_2인발화.csv"]
# input_files = [province + "_따라말하기.csv", province + "_질문에답하기.csv", province + "_2인발화.csv"]


def add_prefix_to_lines(input_file, output_file, prefix):
    input_file = str(input_file)
    output_file = str(output_file)
    try:
        with open(input_file, 'r') as infile, open(output_file, 'a') as outfile:
            for line in infile:
                # 각 줄의 앞에 지정된 prefix를 추가하여 수정한 내용을 새로운 파일에 쓴다.
                modified_line = prefix + "/" + line
                outfile.write(modified_line)
    except FileNotFoundError:
        print("파일을 찾을 수 없습니다.")
    except Exception as e:
        print(f"오류가 발생했습니다: {e}")


def make_path_file():     
    for province in provinces:
        # /data/nia/139-2.중·노년층 한국어 방언 데이터 (충청도, 전라도, 제주도)/08.최종산출물/01-1.최종데이터(업로드)/3.Test/02.라벨링데이터/01. 충청도/01. 1인발화 따라말하기
        # if province in ["강원도", "경상도"]:
        if province in ["충청도", "전라도", "제주도", "강원도", "경상도"]:
            base_label_dir = DATA_DIR / "139-2.중·노년층 한국어 방언 데이터 (충청도, 전라도, 제주도)" / "08.최종산출물" / "01-1.최종데이터(업로드)" / "3.Test" / "02.라벨링데이터"
            output_file = province + "_test.csv"

            if os.path.isfile(output_file):
                os.remove(output_file)

            if province == "충청도":
                province_label_dir = base_label_dir / "01. 충청도"

                for speech_kind in speech_kinds :
                    if speech_kind == "따라말하기":
                        label_dir = province_label_dir / "01. 1인발화 따라말하기"
                        json_list_file = province + "_" + speech_kind + ".csv"
                        print(f"json_list_file : {json_list_file}")
                        print(f"output_file : {output_file}")

                        # 1. json file 경로 추가
                        add_prefix_to_lines(json_list_file, output_file, str(label_dir))
                        
                    elif speech_kind == "질문에답하기":
                        label_dir = province_label_dir / "02. 1인발화 질문에답하기"
                        json_list_file = province + "_" + speech_kind + ".csv"
                        print(f"json_list_file : {json_list_file}")
                        print(f"output_file : {output_file}")

                        # 1. json file 경로 추가
                        add_prefix_to_lines(json_list_file, output_file, str(label_dir))

                    elif speech_kind == "2인발화":
                        label_dir = province_label_dir / "03. 2인발화"
                        json_list_file = province + "_" + speech_kind + ".csv"
                        print(f"json_list_file : {json_list_file}")
                        print(f"output_file : {output_file}")

                        # 1. json file 경로 추가
                        add_prefix_to_lines(json_list_file, output_file, str(label_dir))


        # if province in ["충청도", "전라도", "제주도"]:
        #     if province == "충청도"
        #     label_file_dir = "139-2.중·노년층 한국어 방언 데이터 (충청도, 전라도, 제주도)/08.최종산출물/01-1.최종데이터(업로드)/3.Test/02.라벨링데이터"

        # elif province in ["강원도", "경상도"]:

# 2. 필요한 파일들이 있는지 확인 : 파일 목록에 있는 파일들 확인

# 3. 문장 정보 파싱
# 4. 문장으로 wav file 분리하기
# 5. 문장별 데이터를 저장한 매니페스트 파일 만들기


def check_file_path(list_file):
    with open(list_file, 'r') as list_file:
        total_num_of_line = 0
        num_of_not_exist = 0
        for line in list_file:
            total_num_of_line += 1
            file_path = Path(line.strip())

            if not file_path.exists():
                num_of_not_exist += 1
                print(f"Not exist file : {str(file_path)} ")
    print(f"Total number of lines : {total_num_of_line}")
    print(f"number of not exist : {num_of_not_exist}")


if __name__ == "__main__":
    # make_path_file()
    check_file_path("충청도_test.csv")
    # file = "/data/nia/139-2.중·노년층 한국어 방언 데이터 (충청도, 전라도, 제주도)/08.최종산출물/01-1.최종데이터(업로드)/3.Test/02.라벨링데이터/01. 충청도/03. 2인발화/talk_set1_collectorcc113_speakercc1478_speakercc1479_4_9_7.json"
    # file_path = Path(file)
    # print(not file_path.exists())
