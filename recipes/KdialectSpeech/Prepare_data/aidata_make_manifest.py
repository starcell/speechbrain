#! /usr/bin/env python3
"""
Data preparation.

Author
------
N Park Starcell Inc.

원천 데이터의 json 파일을 읽어서 오디오 파일을 문장 단위로 분할하고 매니페스트 파일 작성
main()함수 실행 전에 입력 데이터 디렉토리 설정

실행 : nohup python aidata_make_manifest.py &

"""
import logging
import os
import glob
from pathlib import Path
import re

## Kang
import pandas as pd
import json
import os
from pydub import AudioSegment
from datetime import datetime

logger = logging.getLogger(__name__)

def normalize(string):
    """
    This function normalizes a given string according to
    the normalization rule
    The normalization rule removes "/" indicating filler words,
    removes "+" indicating repeated words,
    removes all punctuation marks,
    removes non-speech symbols,
    and extracts orthographic transcriptions.

    Arguments
    ---------
    string : str
        The string to be normalized

    Returns
    -------
    str
        The string normalized according to the rules

    """
    # extracts orthographic transcription
    string = re.sub(r"\(([^)]*)\)\/\(([^)]*)\)", r"\1", string)
    # removes non-speech symbols
    string = re.sub(r"n/|b/|o/|l/|u/", "", string)
    # removes punctuation marks
    string = re.sub(r"[+*/.?!~(),]", "", string)
    # removes extra spaces
    string = re.sub(r"\s+", " ", string)
    string = string.strip()

    return string

### for KdialectSpeech
def time_convert(str_time):
    """
    This gives time in milisecond

    Arguments
    ---------
    str_time : str
        The time in string type
        ex) 10:09:20.123

    str_time_format : str
    The time format in string type
    ex) "%H:%M:%S.%f"

    Returns
    -------
    float
        A milisecond time

    """    
    hh, mm, sec_mili = str_time.split(':')
    total_milis = float(hh) * 60 * 60 + float(mm) * 60 + float(sec_mili)
                
    return total_milis


def split_wav_file(input_audio, output_wav_file, start_time, end_time):
    """
    This gives time in milisecond

    Arguments
    ---------
    input_audio : audio signal
        audio signam from wav file

    output_wav_file : str
        wav file to save splited audio

    start_time : float
        start time to split in second

    end_time : float
        end time to split in second

    Returns
    -------
    float
        A milisecond time

    """
    start_time_milis = start_time * 1000
    end_time_milis = end_time * 1000
    splited_wav = input_audio[start_time_milis:end_time_milis]
    splited_wav.export(output_wav_file, format="wav")


def make_sentence_data(base_dir, sentence_dir, json_file_path, province_code):
    """
    This gives manifest record to write in csv file.
    If the length of audio sound is longer than 30 sec, it split the audio file.

    Arguments
    ---------
    annotation_file_path : file path
        The annotation file name(input file) in json type
        ex) person/st/gs/collectorgs1/st_set1_collectorgs1_speakergs3_79_8.json

    splited_wav_folder : str
        The output folder to save splited file
        ex) "splited_file_dir"

    Returns
    -------
    None

    """
    data_file_path = json_file_path.replace("2.라벨링데이터", "1.원천데이터").replace("json", "wav")

    csv_lines = []
    # print(f"json_file_path : {json_file_path}")
    with open(json_file_path, encoding="UTF-8" ) as json_file :
        json_data = json.load(json_file)
        # data_file_name = json_data["fileName"]
        data_file_name = Path(json_file_path).stem # 확장자 제거
        data_path_dir = os.path.dirname(data_file_path)
        # print(f"data_path_dir : {data_path_dir}")
        
        sentence_path_dir = data_path_dir.replace(base_dir, sentence_dir)
        sentence_path_dir = Path(sentence_path_dir)
        # print(f"sentence_path_dir : {sentence_path_dir}")
        os.makedirs(sentence_path_dir, exist_ok=True)

        try:
            input_audio = AudioSegment.from_wav(data_file_path)
        except:
            print(f'wrong audio file : {data_file_path}')
            logger.info(f'wrong audio file : {data_file_path}')
            return csv_lines

        sentences = json_data["transcription"]["sentences"]
        # print(f"sentences : {sentences}")

        for idx, sentence in enumerate(sentences) :
            #data_ duration
            data_start_time = sentence["startTime"]
            data_end_time = sentence["endTime"]
            
            start_time = time_convert(data_start_time)
            end_time = time_convert(data_end_time)
            duration = end_time - start_time
            duration = format(duration, '5.3f')
            
            #wrd(dialect)
            dialect = sentence["dialect"]
            wrd = normalize(dialect)
            
            sentence_file_id = data_file_name + "-" + str(idx)
            sentence_file_name = sentence_file_id + '.wav'
            sentence_file_path = os.path.join(sentence_path_dir, sentence_file_name)
            # print(f'sentence_file_path : {sentence_file_path}')
            logger.info(f'sentence_file_path : {sentence_file_path}')
            try:
                split_wav_file(input_audio, sentence_file_path, start_time, end_time)
            except:
                print(f"wave write to {sentence_file_path} error!")

            
            id = sentence_file_id

            # print(f"id : {id}")
            # print(f"duration : {duration}")
            # print(f"sentence_file_path : {sentence_file_path}")
            # print(f"province_code : {province_code}")
            # print(f"wrd : {wrd}")

            line = [id, duration, sentence_file_path, province_code, wrd]
            csv_lines.append(line)
    return csv_lines


def create_csv(base_dir, label_dir, sentence_dir, province_code):
    """
    This makes manifest file from given base directory.

    Arguments
    ---------
    base_dir : base directory of KdialectSpeech
        ex) '/data/KdialectSpeech'

    province_code : str
        The province code that is one of ('gw', 'gs', 'jl', 'jj', 'cc')

    output_file : str
        The output file to save manifest
        ex) "output/manifest.csv"

    Returns
    -------
    None
    """
    print(label_dir)
    json_file_list = (glob.glob(os.path.join(label_dir, '*/*.' + "json")))
    # print(json_file_list)
    
    total_csv_lines = []
    for json_file in json_file_list:
        total_csv_lines.extend(make_sentence_data(base_dir, sentence_dir, json_file, province_code))

    total_csv_df = pd.DataFrame(total_csv_lines, columns=['ID', 'duration', 'wav', 'province_code', 'wrd'])
    total_csv_file = os.path.join(sentence_dir, province_code + '.csv')
    total_csv_df.to_csv(total_csv_file, encoding="UTF-8", index=False)


def main(base_dir, sentence_dir, province_code):
    """
    
    """
    
    ### 원천데이터
    # /data/aidata/139-1.중·노년층 한국어 방언 데이터 (강원도, 경상도)/06.품질검증/1.Dataset/1.원천데이터/01. 강원도
    # /data/aidata/139-1.중·노년층 한국어 방언 데이터 (강원도, 경상도)/06.품질검증/1.Dataset/1.원천데이터/02. 경상도
    # /data/aidata/139-2.중·노년층 한국어 방언 데이터 (충청도, 전라도, 제주도)/06.품질검증/1.Dataset/1.원천데이터/01. 충청도
    # /data/aidata/139-2.중·노년층 한국어 방언 데이터 (충청도, 전라도, 제주도)/06.품질검증/1.Dataset/1.원천데이터/02. 전라도
    # /data/aidata/139-2.중·노년층 한국어 방언 데이터 (충청도, 전라도, 제주도)/06.품질검증/1.Dataset/1.원천데이터/03. 제주도

    ### 라벨링데이터
    # /data/aidata/139-1.중·노년층 한국어 방언 데이터 (강원도, 경상도)/06.품질검증/1.Dataset/2.라벨링데이터/01. 강원도
    # /data/aidata/139-1.중·노년층 한국어 방언 데이터 (강원도, 경상도)/06.품질검증/1.Dataset/2.라벨링데이터/02. 경상도
    # /data/aidata/139-2.중·노년층 한국어 방언 데이터 (충청도, 전라도, 제주도)/06.품질검증/1.Dataset/2.라벨링데이터/01. 충청도
    # /data/aidata/139-2.중·노년층 한국어 방언 데이터 (충청도, 전라도, 제주도)/06.품질검증/1.Dataset/2.라벨링데이터/02. 전라도
    # /data/aidata/139-2.중·노년층 한국어 방언 데이터 (충청도, 전라도, 제주도)/06.품질검증/1.Dataset/2.라벨링데이터/03. 제주도

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
    
    if not os.path.exists(data_dir):
        err_msg = (
            "the directory %s does not exist (it is expected in the "
            "kdialectspeech dataset)" % data_dir
        )
        raise OSError(err_msg)
        
    if not os.path.exists(label_dir):
        err_msg = (
            "the directory %s does not exist (it is expected in the "
            "kdialectspeech dataset)" % label_dir
        )
        raise OSError(err_msg)
    
    create_csv(base_dir, label_dir, sentence_dir, province_code)


if __name__ == "__main__":
    base_dir = "/data/aidata"
    sentence_dir = "/data/aidata/sentence"
    os.makedirs(sentence_dir, exist_ok=True)
    # province_code = "cc"
    province_code_list = ["gw", "gs", "cc", "jl", "jj"]

    print(f"start : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    for province in province_code_list:
        main(base_dir, sentence_dir, province)
    print(f"end : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    