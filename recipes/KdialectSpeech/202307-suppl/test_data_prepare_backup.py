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


# 1. 목록 파일 읽기, 절대경로 추가, json file, wav file 경로,  파일 존재 확인
DATA_DIR = Path("/data/nia")
# 입력 파일들
PROVINCES = ["충청도", "전라도", "제주도", "강원도", "경상도"]
SPEECH_KINDS = ["따라말하기", "질문에답하기", "2인발화"]


def make_file_path_df(input_file, output_file, label_dir) -> pd.DataFrame:
    """
    주어진 테스트 파일 목록을 가지고 절대 경로가 포함된 json file과 wav file의 목록을 DataFrame으로 만들어서 반환 
    """
    input_file = str(input_file)
    output_file = str(output_file)
    json_file_path = str(label_dir) + "/"
    wav_file_path = json_file_path.replace("02.라벨링데이터", "01.원천데이터")

    try:
        df = pd.read_csv(input_file, names=["json_file"], header=None)
        df["data_id"] = df.replace(".json", "", regex=True)
        df["json_file"] = json_file_path + df["json_file"]
        df["wav_file"] = wav_file_path + df["data_id"] + ".wav"

        return df
        
    except FileNotFoundError:
        print("파일을 찾을 수 없습니다.")
    except Exception as e:
        print(f"오류가 발생했습니다: {e}")


def make_path_file():
    """
    주어진 테스트 파일 목록을 가지고 절대 경로가 포함된 json file과 wav file의 목록을 만들기 
    """
    for province in PROVINCES:
        if province == "강원도":
            base_label_dir = DATA_DIR / "139-1.중·노년층 한국어 방언 데이터 (강원도, 경상도)" / "08.최종산출물" / "01-1.최종데이터(업로드)" / "3.Test" / "02.라벨링데이터"
            province_label_dir = base_label_dir / ("01. " + province)

        elif province == "경상도":
            base_label_dir = DATA_DIR / "139-1.중·노년층 한국어 방언 데이터 (강원도, 경상도)" / "08.최종산출물" / "01-1.최종데이터(업로드)" / "3.Test" / "02.라벨링데이터"
            province_label_dir = base_label_dir / ("02. " + province)

        elif province == "충청도":
            base_label_dir = DATA_DIR / "139-2.중·노년층 한국어 방언 데이터 (충청도, 전라도, 제주도)" / "08.최종산출물" / "01-1.최종데이터(업로드)" / "3.Test" / "02.라벨링데이터"
            province_label_dir = base_label_dir / ("01. " + province)

        elif province == "전라도":
            base_label_dir = DATA_DIR / "139-2.중·노년층 한국어 방언 데이터 (충청도, 전라도, 제주도)" / "08.최종산출물" / "01-1.최종데이터(업로드)" / "3.Test" / "02.라벨링데이터"
            province_label_dir = base_label_dir / ("02. " + province)

        elif province == "제주도":
            base_label_dir = DATA_DIR / "139-2.중·노년층 한국어 방언 데이터 (충청도, 전라도, 제주도)" / "08.최종산출물" / "01-1.최종데이터(업로드)" / "3.Test" / "02.라벨링데이터"
            province_label_dir = base_label_dir / ("03. " + province)

        output_file = province + "_test.csv"

        if os.path.isfile(output_file):
            os.remove(output_file)

        for speech_kind in SPEECH_KINDS :
            if speech_kind == "따라말하기":
                label_dir = province_label_dir / "01. 1인발화 따라말하기"
                json_list_file = province + "_" + speech_kind + ".csv"
                # json file 경로, wav file 경로 추가
                speech_kind__0_df = make_file_path_df(json_list_file, output_file, label_dir)
                    
            elif speech_kind == "질문에답하기":
                label_dir = province_label_dir / "02. 1인발화 질문에답하기/"
                json_list_file = province + "_" + speech_kind + ".csv"
                # json file 경로, wav file 경로 추가
                speech_kind__1_df = make_file_path_df(json_list_file, output_file, label_dir)

            elif speech_kind == "2인발화":
                label_dir = province_label_dir / "03. 2인발화/"
                json_list_file = province + "_" + speech_kind + ".csv"
                # json file 경로, wav file 경로 추가
                speech_kind__2_df = make_file_path_df(json_list_file, output_file, label_dir)

        # 위에서 구한 세 개의 df를 csv 파일에 저장합니다.
        df = pd.concat([speech_kind__0_df, speech_kind__1_df, speech_kind__2_df])
        df.to_csv(output_file, index=False)

def check_file_path(path_list):
    """
    입력받은 파일이 존재하는지 확인
    """
    total_num_of_line = 0
    num_of_not_exist = 0
    for line in path_list:
        total_num_of_line += 1
        file_path = Path(line.strip())

        if not file_path.exists():
            num_of_not_exist += 1
            print(f"Not exist file : {str(file_path)} ")

    print(f"Total number of lines : {total_num_of_line}")
    print(f"number of not exist : {num_of_not_exist}")


# 2. json file을 파싱하여 문장별로 분리


# 3. 문장 정보 파싱
# 4. 문장으로 wav file 분리하기
# 5. 문장별 데이터를 저장한 매니페스트 파일 만들기


if __name__ == "__main__":
    # 1. 목록 파일 읽기, 절대경로 추가, json file, wav file 경로,  파일 존재 확인
    make_path_file()
    print("file path csv 만들기를 완료했습니다.")

    print("파일 목록의 파일들이 있는지 확인합니다.")
    for province in PROVINCES:
        print("---------------------------------------")
        print(f"{province} 파일 목록의 파일들이 있는지 확인")
        list_file = province + "_test.csv"
        path_list = pd.read_csv(list_file)

        print("json file check : ")
        check_file_path(path_list["json_file"])

        print("wav file check : ")
        check_file_path(path_list["wav_file"])


