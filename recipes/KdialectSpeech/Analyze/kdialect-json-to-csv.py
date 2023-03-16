#!/usr/bin/env python

### 이 파일은 json 파일을 파싱하여 필요한 태그 값들을 추출하여 csv 파일로 만드는 프로그램입니다.
### 사용법은 아규먼트(인수)로 json 파일들이 있는 디렉토리와 만들 csv 파일의 이름을 주면 됩니다.
### ex) python kdialect-json-to-csv.py json_dir_name csv_file_name.csv
# python kdialect-json-to-csv.py "/data/aidata/139-1.중·노년층 한국어 방언 데이터 (강원도, 경상도)/06.품질검증/1.Dataset/2.라벨링데이터/01. 강원도/02. 1인발화 질문에답하기" gw_annot_1.csv
# python kdialect-json-to-csv.py "/data/aidata/139-1.중·노년층 한국어 방언 데이터 (강원도, 경상도)/06.품질검증/1.Dataset/2.라벨링데이터/01. 강원도/03. 2인발화" gw_annot_2.csv

# python kdialect-json-to-csv.py "/data/aidata/139-1.중·노년층 한국어 방언 데이터 (강원도, 경상도)/06.품질검증/1.Dataset/2.라벨링데이터/02. 경상도/02. 1인발화 질문에답하기" gs_annot_1.csv
# python kdialect-json-to-csv.py "/data/aidata/139-1.중·노년층 한국어 방언 데이터 (강원도, 경상도)/06.품질검증/1.Dataset/2.라벨링데이터/02. 경상도/03. 2인발화" gs_annot_2.csv

# python kdialect-json-to-csv.py "/data/aidata/139-2.중·노년층 한국어 방언 데이터 (충청도, 전라도, 제주도)/06.품질검증/1.Dataset/2.라벨링데이터/01. 충청도/02. 1인발화 질문에답하기" cc_annot_1.csv
# python kdialect-json-to-csv.py "/data/aidata/139-2.중·노년층 한국어 방언 데이터 (충청도, 전라도, 제주도)/06.품질검증/1.Dataset/2.라벨링데이터/01. 충청도/03. 2인발화" cc_annot_2.csv

# python kdialect-json-to-csv.py "/data/aidata/139-2.중·노년층 한국어 방언 데이터 (충청도, 전라도, 제주도)/06.품질검증/1.Dataset/2.라벨링데이터/02. 전라도/02. 1인발화 질문에답하기" jl_annot_1.csv
# python kdialect-json-to-csv.py "/data/aidata/139-2.중·노년층 한국어 방언 데이터 (충청도, 전라도, 제주도)/06.품질검증/1.Dataset/2.라벨링데이터/02. 전라도/03. 2인발화" jl_annot_2.csv

# python kdialect-json-to-csv.py "/data/aidata/139-2.중·노년층 한국어 방언 데이터 (충청도, 전라도, 제주도)/06.품질검증/1.Dataset/2.라벨링데이터/03. 제주도/02. 1인발화 질문에답하기" jj_annot_1.csv
# python kdialect-json-to-csv.py "/data/aidata/139-2.중·노년층 한국어 방언 데이터 (충청도, 전라도, 제주도)/06.품질검증/1.Dataset/2.라벨링데이터/03. 제주도/03. 2인발화" jj_annot_2.csv



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

NO_VALUE = "no_value"

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


                line = [
                    json_file_name, utterance_type, speaker_id, residence_province, gender, birth_year,
                    sentence_id, intent_type, intent_category, emotion_type, grammar_type
                    ]
                
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


def main(json_dir, csv_file):
    """
    kdialect-json-to-csv.py json_dir csv_file
    """
    print(json_dir)
    print(csv_file)

    logger.info(f'extract start -----')
    json_files = get_json_files(json_dir)
    for json_file in json_files:
        if make_csv_lines(json_file):
            csv_df = pd.DataFrame(make_csv_lines(json_file), columns=[
                "json_file_name", "utterance_type", "speaker_id", "residence_province", "gender", "birth_year",
                "sentence_id", "intent_type", "intent_category", "emotion_type", "grammar_type"
            ])
            if not os.path.exists(csv_file):
                csv_df.to_csv(csv_file, index=False, mode="w", encoding="UTF-8")
            else:
                csv_df.to_csv(csv_file, index=False, mode="a", encoding="UTF-8", header=False)
    logger.info(f'extract end -----')

if __name__ == "__main__":

    json_dir = sys.argv[1]
    csv_file = sys.argv[2]
    main(json_dir, csv_file)
