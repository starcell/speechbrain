#!/usr/bin/env python

import sys
import os
import glob
from pathlib import Path
import json
import pandas as pd
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(filename=sys.argv[0] + ".log", level = logging.INFO, datefmt = '%Y-%m-%d %H%M%S'
                   ,format = '%(asctime)s | %(levelname)s | %(message)s')
streamHandler = logging.StreamHandler()
logger.addHandler(streamHandler)

NO_VALUE = "no_value"
JSON_DIRS = [
            "/data/aidata/139-1.중·노년층 한국어 방언 데이터 (강원도, 경상도)/06.품질검증/1.Dataset/2.라벨링데이터/01. 강원도/01. 1인발화 따라말하기",
            "/data/aidata/139-1.중·노년층 한국어 방언 데이터 (강원도, 경상도)/06.품질검증/1.Dataset/2.라벨링데이터/01. 강원도/02. 1인발화 질문에답하기",
            "/data/aidata/139-1.중·노년층 한국어 방언 데이터 (강원도, 경상도)/06.품질검증/1.Dataset/2.라벨링데이터/01. 강원도/03. 2인발화",

            "/data/aidata/139-1.중·노년층 한국어 방언 데이터 (강원도, 경상도)/06.품질검증/1.Dataset/2.라벨링데이터/02. 경상도/01. 1인발화 따라말하기",
            "/data/aidata/139-1.중·노년층 한국어 방언 데이터 (강원도, 경상도)/06.품질검증/1.Dataset/2.라벨링데이터/02. 경상도/02. 1인발화 질문에답하기",
            "/data/aidata/139-1.중·노년층 한국어 방언 데이터 (강원도, 경상도)/06.품질검증/1.Dataset/2.라벨링데이터/02. 경상도/03. 2인발화",

            "/data/aidata/139-2.중·노년층 한국어 방언 데이터 (충청도, 전라도, 제주도)/06.품질검증/1.Dataset/2.라벨링데이터/01. 충청도/01. 1인발화 따라말하기",
            "/data/aidata/139-2.중·노년층 한국어 방언 데이터 (충청도, 전라도, 제주도)/06.품질검증/1.Dataset/2.라벨링데이터/01. 충청도/02. 1인발화 질문에답하기",
            "/data/aidata/139-2.중·노년층 한국어 방언 데이터 (충청도, 전라도, 제주도)/06.품질검증/1.Dataset/2.라벨링데이터/01. 충청도/03. 2인발화",

            "/data/aidata/139-2.중·노년층 한국어 방언 데이터 (충청도, 전라도, 제주도)/06.품질검증/1.Dataset/2.라벨링데이터/02. 전라도/01. 1인발화 따라말하기",
            "/data/aidata/139-2.중·노년층 한국어 방언 데이터 (충청도, 전라도, 제주도)/06.품질검증/1.Dataset/2.라벨링데이터/02. 전라도/02. 1인발화 질문에답하기",
            "/data/aidata/139-2.중·노년층 한국어 방언 데이터 (충청도, 전라도, 제주도)/06.품질검증/1.Dataset/2.라벨링데이터/02. 전라도/03. 2인발화",

            "/data/aidata/139-2.중·노년층 한국어 방언 데이터 (충청도, 전라도, 제주도)/06.품질검증/1.Dataset/2.라벨링데이터/03. 제주도/01. 1인발화 따라말하기",
            "/data/aidata/139-2.중·노년층 한국어 방언 데이터 (충청도, 전라도, 제주도)/06.품질검증/1.Dataset/2.라벨링데이터/03. 제주도/02. 1인발화 질문에답하기",
            "/data/aidata/139-2.중·노년층 한국어 방언 데이터 (충청도, 전라도, 제주도)/06.품질검증/1.Dataset/2.라벨링데이터/03. 제주도/03. 2인발화"
]
PROV_DIC = {0:"gw", 1:"gw", 2:"gw", 3:"gs", 4:"gs", 5:"gs", 6:"cc", 7:"cc", 8:"cc", 9:"jl", 10:"jl", 11:"jl", 12:"jj", 13:"jj", 14:"jj"}
OUTPUT_DIR = "annot_csvs"
# OUTPUT_DIR = os.path.join(os.getcwd(), "annot_csvs")
# os.makedirs(OUTPUT_DIR, exist_ok=True)

def make_csv_lines(json_file_path):
    csv_lines = []
    # json_file_name = os.path.basename(json_file_path)
    json_file_name = Path(os.path.basename(json_file_path)).stem

    # 대화의 타입은 파일의 이름에서 구한다.
    if json_file_name.startswith("say"):
        utterance_type = "say"
    elif json_file_name.startswith("talk"):
        utterance_type = "talk"
    elif json_file_name.startswith("st"):
        utterance_type = "st"
    else:
        utterance_type = "no_type"

    with open(json_file_path, encoding="UTF-8" ) as json_file :
        json_data = json.load(json_file)

    try:
        speakers = json_data["speaker"]
    except:
        logger.info(f"no speakers, file_name : {json_file_path}")
        return []

    try:
        sentences = json_data["transcription"]["sentences"]
    except:
        logger.info(f"no sentences, file_name : {json_file_path}")
        return []

    try:
        annotations = json_data["annotation"]
    except:
        logger.info(f"no annotations, file_name : {json_file_path}")
        return []
    
    try:
        intents = annotations["intents"]
    except:
        logger.info(f"no intents, file_name : {json_file_path}")
        intents = []
            
    try:
        emotions = annotations["emotions"]
    except:
        logger.info(f"no emotions, file_name : {json_file_path}")
        emotions = []
            
    try:
        grammarTypes = annotations["grammarTypes"]
    except:
        logger.info(f"no grammarTypes, file_name : {json_file_path}")
        grammarTypes = []
    
    for speaker in speakers :
        try:
            speaker_id = speaker["speakerId"]
        except:
            return []
        
        try:
            residence_province = speaker["residenceProvince"]
        except:
            residence_province = NO_VALUE
            logger.info(f"no residence_province, file_name : {json_file_path}")

        try:
            gender = speaker["gender"]
        except:
            gender = NO_VALUE
            logger.info(f"no gender, file_name : {json_file_path}")

        try:
            birth_year = speaker["birthYear"]
        except:
            birth_year = NO_VALUE
            logger.info(f"no birth_year, file_name : {json_file_path}")
        
        try:
            sentence_ids_of_speaker = [sentence["sentenceId"] for sentence in sentences if sentence["speakerId"] == speaker_id]
        except:
            logger.info(f"no sentenceId, file_name : {json_file_path}")
            return []

        lines = []
        if sentence_ids_of_speaker:
            for sentence_id in sentence_ids_of_speaker:
                if intents:
                    for intent in intents:
                        if intent["sentenceId"] == sentence_id:
                            try:
                                intent_type = intent["tagType"]
                            except:
                                intent_type = NO_VALUE

                            try:
                                intent_category = intent["categoryName"]
                            except:
                                intent_category = NO_VALUE
                else:
                    intent_type = "no_intents"
                    intent_category = "no_intents"

                if emotions:
                    for emotion in emotions:
                        if emotion["sentenceId"] == sentence_id:
                            try:
                                emotion_type = emotion["tagType"]
                            except:
                                emotion_type = NO_VALUE
                else:
                    emotion_type = "no_emotions"

                if grammarTypes:
                    for grammarType in grammarTypes:
                        if grammarType["sentenceId"] == sentence_id:
                            try:
                                grammar_type = grammarType["tagType"]
                            except:
                                grammar_type = NO_VALUE
                else:
                    grammar_type = "no_grammarTypes"


                try:
                    line = [
                        json_file_name, utterance_type, speaker_id, residence_province, gender, birth_year,
                        sentence_id, intent_type, intent_category, emotion_type, grammar_type
                        ]
                except:
                    line = []
                    logger.info(f"line error json_file_path : {json_file_path}")
                    # print(f"json_file_path : {json_file_path}")

                if line:
                    lines.append(line)

        if lines:
            csv_lines.extend(lines)

    return csv_lines

def get_json_files(json_dir):
    """
    This gives json files

    Arguments
    ---------
    base_dir : str
        The base directory of data files

    Returns
    -------
    list
        A list containing directories of the given data directory

    """
    file_ext='json'
    files = glob.glob(os.path.join(json_dir, '*.' + file_ext))

    return files


def make_csv_file(json_dir, csv_file):
    """
    """
    logger.info(f'extract at {json_dir} start -----')
    # logger.info(f'os.getcwd() : {os.getcwd()}')
    logger.info(f'csv_file : {csv_file}')
       

    json_files = get_json_files(json_dir)
    if os.path.exists(csv_file):
        os.rename(csv_file, csv_file + ".bak")

    for i, json_file in enumerate(json_files):
        if make_csv_lines(json_file):
            csv_df = pd.DataFrame(make_csv_lines(json_file), columns=[
                "json_file_name", "utterance_type", "speaker_id", "residence_province", "gender", "birth_year",
                "sentence_id", "intent_type", "intent_category", "emotion_type", "grammar_type"
            ])
            if i == 0:
                csv_df.to_csv(csv_file, index=False, mode="w", encoding="UTF-8")
            else:
                csv_df.to_csv(csv_file, index=False, mode="a", encoding="UTF-8", header=False)
    logger.info(f'extract at {json_dir} end -----')

if __name__ == "__main__":

    for i, dir in enumerate(JSON_DIRS):
        prov_code = PROV_DIC[i]
        csv_file = str(i + 1) + "_" + prov_code + ".csv"
        csv_path = os.path.join(OUTPUT_DIR, csv_file) ### 지정된 디렉토리에 파일이 생기지 않음
        # logger.info(f"csv_path : {csv_path} ----------")
        make_csv_file(dir, csv_path)
