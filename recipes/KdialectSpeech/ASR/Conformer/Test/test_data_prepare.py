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
# import logging
import os
from pathlib import Path

## Kang
import pandas as pd
import json
import os
from pydub import AudioSegment

import sys
sys.path.append("../../../kdialectspeech")
from string_normalize import string_normalize
from time_convert import time_convert
from check_audio_file import check_audio_file

# logger = logging.getLogger(__name__)


# 1. 목록 파일 읽기, 절대경로 추가, json file, wav file 경로,  파일 존재 확인
DATA_DIR = Path("/data/nia")
SENTENCE_DIR = DATA_DIR / "sentences" # 문장별로 분리된 wav 파일들이 저장될 위치
TEST_FILE_LIST_DIR = DATA_DIR / "test_file_list"

# 입력 파일들
# PROVINCES = ["충청도", "전라도", "제주도", "강원도", "경상도"]
PROVINCES_DIC = {"cc":"충청도", "jl":"전라도", "jj":"제주도", "gw":"강원도", "gs":"경상도"}
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


def make_path_file(test_file_list_dir, provinces_dic, speech_kinds):
    """
    주어진 테스트 파일 목록을 가지고 절대 경로가 포함된 json file과 wav file의 목록을 만들기 
    """
    for province_code, province in provinces_dic.items():
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

        output_file = os.path.join(test_file_list_dir, province_code + "_test.csv")

        if os.path.isfile(output_file):
            os.remove(output_file)

        for speech_kind in speech_kinds :
            if speech_kind == "따라말하기":
                label_dir = province_label_dir / "01. 1인발화 따라말하기"
                json_list_file = os.path.join(test_file_list_dir, province + "_" + speech_kind + ".csv")
                # json file 경로, wav file 경로 추가
                speech_kind__0_df = make_file_path_df(json_list_file, output_file, label_dir)
                    
            elif speech_kind == "질문에답하기":
                label_dir = province_label_dir / "02. 1인발화 질문에답하기/"
                # json_list_file = province + "_" + speech_kind + ".csv"
                json_list_file = os.path.join(test_file_list_dir, province + "_" + speech_kind + ".csv")
                # json file 경로, wav file 경로 추가
                speech_kind__1_df = make_file_path_df(json_list_file, output_file, label_dir)

            elif speech_kind == "2인발화":
                label_dir = province_label_dir / "03. 2인발화/"
                # json_list_file = province + "_" + speech_kind + ".csv"
                json_list_file = os.path.join(test_file_list_dir, province + "_" + speech_kind + ".csv")
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


# 2. json file을 파싱하여 문장별로 분리하고, 전사문 추가
def split_wav_file(input_audio, output_wav_file:str, start_time:float, end_time:float):
    """ wav 파일을 입력으로 받아서 문장별로 분리
    Arguments
    ---------
    input_audio : audio signal
        audio signam from wav file
    output_wav_file : wav file to save splited audio
    start_time : start time to split in second
    end_time : end time to split in second
    Returns
    -------
    None : write wav file
    """
    start_time_milis = start_time * 1000
    end_time_milis = end_time * 1000
    splited_wav = input_audio[start_time_milis:end_time_milis]
    splited_wav.export(output_wav_file, format="wav")


def make_csv_lines(json_file_path:str, wav_file_path:str, data_id, splited_wav_folder:str, long_sentence_list_file:str):
    """
    This gives manifest record to write in csv file.
    If the length of audio sound is longer than 30 sec, it split the audio file.

    Arguments
    ---------
    json_file_path : file path
        The annotation file name(input file) in json type
        ex) person/st/gs/collectorgs1/st_set1_collectorgs1_speakergs3_79_8.json

    splited_wav_folder : str
        The output folder to save splited file
        ex) "splited_file_dir"

    Returns
    -------
    None

    """

    csv_lines = []

    # with open(json_file_path, encoding="UTF-8" ) as json_file :
    with open(json_file_path) as json_file :
        json_data = json.load(json_file)

        if splited_wav_folder == 'same':
            splited_wav_folder = os.path.dirname(wav_file_path)

        try:
            input_audio = AudioSegment.from_wav(wav_file_path)
        except:
            print(f'wrong audio file : {wav_file_path}')
            return csv_lines

        sentences = json_data["transcription"]["sentences"]
        province_code = sentences[0]["speakerId"][7:9]
        
        os.makedirs(os.path.join(splited_wav_folder, province_code), exist_ok=True)

        for idx, sentence in enumerate(sentences) :
            sentenceid = sentence["sentenceId"]
            start_time = time_convert(sentence["startTime"])
            end_time = time_convert(sentence["endTime"])
            duration = end_time - start_time
            wrd = string_normalize(sentence["dialect"])

            if duration > 30:
                # print(f'too long sentence included : {json_file_path} : {sentenceid}')
                # print(f'sentence : {wrd} \'s duration : {duration}')
                with open(long_sentence_list_file, 'a') as long_file:
                    long_file.write(f"{json_file_path} : {sentenceid}: {duration}\n")
                # continue
            
            ##### 이상한 데이터 확인용
            # if data_id == "say_set2_collectorgw87_speakergw2394_16_0_27": # 자세한 내용을 보고 싶을 때 여기에 데이터 이름을 입력
            #     print(f"json_file_path : {json_file_path}")
            #     print(f"sentences : {sentences}")
            #     print(f"sentence : {sentence}")
            #     print(f"sentenceid : {sentenceid}")
            #     print(f"wrd : {wrd}")

            
            # id = data_id + "-" + str(sentenceid) # data ID
            id = data_id + "-" + str(idx + 1) # data ID            
            
            splited_file_name = id + '.wav'
            splited_file_path = os.path.join(splited_wav_folder, province_code, splited_file_name)
            split_wav_file(input_audio, splited_file_path, start_time, end_time)

            duration = format(duration, '5.3f')
            line = [id, duration, splited_file_path, province_code, wrd]
            csv_lines.append(line)
    return csv_lines


def create_csv(test_file_list_dir, province_code, file_list_csv, save_folder):
    """ 주어진 리스트 파일의 목록을 읽어서 매니패스트 파일을 만들어서 저장
    Arguments
    ---------
    list_file : json file과 wav file 목록
        ex) json_file,data_id,wav_file
            "/data/nia/139-2.중·노년층 한국어 방언 데이터 (충청도, 전라도, 제주도)/08.최종산출물/01-1.최종데이터(업로드)/3.Test/02.라벨링데이터/01. 충청도/01. 1인발화 따라말하기/st_set1_collectorcc109_speakercc2387_9_5.json",st_set1_collectorcc109_speakercc2387_9_5,"/data/nia/139-2.중·노년층 한국어 방언 데이터 (충청도, 전라도, 제주도)/08.최종산출물/01-1.최종데이터(업로드)/3.Test/01.원천데이터/01. 충청도/01. 1인발화 따라말하기/st_set1_collectorcc109_speakercc2387_9_5.wav"

    output_file : str
        The output file to save manifest
        ex) "output/manifest.csv"

    Returns
    -------
    None
    """

    file_list_csv_path = os.path.join(test_file_list_dir, file_list_csv)
    file_list_df = pd.read_csv(file_list_csv_path)

    long_sentence_list_file = os.path.join(test_file_list_dir, province_code + "_long_sentence_list.txt")
    print(f"long_sentence_list_file : {long_sentence_list_file}")
    if os.path.isfile(long_sentence_list_file):
        os.remove(long_sentence_list_file)

    total_csv_lines = []
    for row in file_list_df.itertuples():
        csv_lines = make_csv_lines(row.json_file, row.wav_file, row.data_id, save_folder, long_sentence_list_file)
        total_csv_lines = total_csv_lines + csv_lines # append를 사용하면 안됨. 리스트 안에 리스트가 추가됨

    total_csv_df = pd.DataFrame(total_csv_lines, columns=['ID', 'duration', 'wav', 'province_code', 'wrd'])
    total_csv_file = os.path.join(test_file_list_dir, province_code + '_test_manifest.csv')
    # json_list_file = os.path.join(test_file_list_dir, province + "_" + speech_kind + ".csv")
    total_csv_df.to_csv(total_csv_file, index=False)

# 3. 문장 정보 파싱
# 4. 문장으로 wav file 분리하기
# 5. 문장별 데이터를 저장한 매니페스트 파일 만들기


if __name__ == "__main__":
    ##### 실행 방법 :
    ##### python test_data_prepare.py
    ## python test_data_prepare.py 실행하면 모든 지역에 대해 준비 작업 실행
    ## 불필요한 라인은 주석처리하여 필요한 부분만 실행할 수 있음
    ## test용 파일 목록 파일이 현재 디렉토리에 있어야 함. "지역명" + "발화종류" 조함으로 있어야 함.
    ## 처음 필요한 데이터는 DATA_DIR에 있어야 함. 하위 디렉토리 구조는 make_path_file() 참고
    ## 현재 디렉토리에 결과 파일들이 생김, "province_code"_test.csv, "province_code"_test_manifest.csv 생성
    ## SENTENCE_DIR에 문장별로 분리된 wav file 저장됨, 
    ## 테스트 데이터로 manifest file과 문장별로 분리된 wav file이 사용됨

    if sys.argv is not None:
        province_code = sys.argv[1]
        provice = PROVINCES_DIC[province_code]
        PROVINCES_DIC = {province_code:provice}

    print(f"{PROVINCES_DIC} : 평가용 파일의 절대 경로 목록을 만듧니다.")
    # 1. 목록 파일 읽기, 절대경로 추가, json file, wav file 경로,  파일 존재 확인
    make_path_file(TEST_FILE_LIST_DIR, PROVINCES_DIC, SPEECH_KINDS)
    print("file path csv 만들기를 완료했습니다.")

    print("파일 목록의 파일들이 있는지 확인하고, 오디오 파일을 검사합니다.")
    for province_code, province in PROVINCES_DIC.items():
        print("---------------------------------------")
        print(f"{province} 파일 목록의 파일들이 있는지 확인")
        list_file = os.path.join(TEST_FILE_LIST_DIR, province_code + "_test.csv")
        path_list = pd.read_csv(list_file)

        print("json file check : ")
        check_file_path(path_list["json_file"])

        print("wav file check : ")
        check_file_path(path_list["wav_file"])

        # 오디오 파일 체크 : 정상적으로 wav 파일이 열리는 지 확인
        wrong_audio_file_list = []
        for wav_file in path_list["wav_file"]:
            check_result = check_audio_file(wav_file)
            # print(check_result)

            if check_result is not None:
                wrong_audio_file_list.append(check_result)

        if wrong_audio_file_list is not None:
            print(f"wrong file number : {len(wrong_audio_file_list)}")
            print(f"wrong_sr_list : {wrong_audio_file_list}")

    # 2. manifest file 만들기 : 
    # columns : 'ID', 'duration', 'wav', 'province_code', 'wrd'
    # file을 문장별로 분리

    for province_code, province in PROVINCES_DIC.items():
        file_list_csv = province_code  + "_test.csv"
        print(f"{province} 방언 매니페스트 파일 만들기 시작.")
        create_csv(TEST_FILE_LIST_DIR, province_code, file_list_csv, SENTENCE_DIR)
