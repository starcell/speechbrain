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
                    json_file_name, speaker_id, residence_province, gender, birth_year,
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
    logger.info(f'extract start -----')
    json_files = get_json_files(json_dir)
    for json_file in json_files:
        if make_csv_lines(json_file):
            csv_df = pd.DataFrame(make_csv_lines(json_file), columns=[
                "json_file_name", "speaker_id", "residence_province", "gender", "birth_year",
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