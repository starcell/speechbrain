import sys
import os
import glob
from pathlib import Path
import json
import pandas as pd
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(filename=sys.argv[0] + ".log", level = logging.INFO, datefmt = '%Y-%m-%d %H%M%S'
                   ,format = '%(asctime)s | %(levelname)s | $(message)s')

def make_csv_lines(json_file_path):
    csv_lines = []
    # json_file_name = os.path.basename(json_file_path)
    json_file_name = Path(os.path.basename(json_file_path)).stem
    with open(json_file_path, encoding="UTF-8" ) as json_file :
        json_data = json.load(json_file)

        speakers = json_data["speaker"]
        sentences = json_data["transcription"]["sentences"]
        
        annotations = json_data["annotation"]
        intents = annotations["intents"]
        emotions = annotations["emotions"]
        grammarTypes = annotations["grammarTypes"]

        for speaker in speakers :
            speaker_id = speaker["speakerId"]
            residence_province = speaker["residenceProvince"]
            gender = speaker["gender"]
            birth_year = speaker["birthYear"]
            
            sentence_ids_of_speaker = [sentence["sentenceId"] for sentence in sentences if sentence["speakerId"] == speaker_id]

            lines = []
            for sentence_id in sentence_ids_of_speaker:

                for intent in intents:
                    if intent["sentenceId"] == sentence_id:
                        intent_type = intent["tagType"]
                        intent_category = intent["categoryName"]

                for emotion in emotions:
                    if emotion["sentenceId"] == sentence_id:
                        emotion_type = emotion["tagType"]

                for grammarType in grammarTypes:
                    if grammarType["sentenceId"] == sentence_id:
                        grammar_type = grammarType["tagType"]

                line = [
                    json_file_name, speaker_id, residence_province, gender, birth_year,
                    sentence_id, intent_type, intent_category, emotion_type, grammar_type
                    ]
                
                lines.append(line)

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
    csv_data = []
    for json_file in json_files:
        csv_data.extend(make_csv_lines(json_file))

    csv_df = pd.DataFrame(csv_data, columns=[
        "json_file_name", "speaker_id", "residence_province", "gender", "birth_year",
        "sentence_id", "intent_type", "intent_category", "emotion_type", "grammar_type"
    ])
    csv_df.to_csv(csv_file, encoding="UTF-8")
    logger.info(f'extract end -----')


if __name__ == "__main__":
    json_dir = sys.argv[1]
    csv_file = sys.argv[2]
    main(json_dir, csv_file)