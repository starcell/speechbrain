"""
Data preparation.

Author
------
N Park Starcell Inc.

apt install ffmpeg
pydub는 ffmpeg를 필요로 함.

### 데이터 파일을 문장별로 나누고 csv file 생성

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
from datetime import datetime

logger = logging.getLogger(__name__)

### modified for KdialectSpeech
def check_kdialectspeech_folders(base_dir, province_code):
    """
    Check if the data folder actually contains the kdialectspeech dataset.

    If it does not, an error is raised.

    Returns
    -------
    None

    Raises
    ------
    OSError
        If kdialectspeech is not found at the specified path.
    """
    # Checking if all the data directories exist
    if province_code == 'total':
        talk_dir = os.path.join(base_dir, 'people/talk/')
        say_dir = os.path.join(base_dir, 'person/say/')
        st_dir = os.path.join(base_dir, 'person/st/')
    else:
        talk_dir = os.path.join(base_dir, 'people/talk/' + province_code)
        say_dir = os.path.join(base_dir, 'person/say/' + province_code)
        st_dir = os.path.join(base_dir, 'person/st/' + province_code)
    
    if not os.path.exists(talk_dir):
        err_msg = (
            "the directory %s does not exist (it is expected in the "
            "kdialectspeech dataset)" % talk_dir
        )
        raise OSError(err_msg)
    
    if not os.path.exists(say_dir):
        err_msg = (
            "the directory %s does not exist (it is expected in the "
            "kdialectspeech dataset)" % talk_dir
        )
        raise OSError(err_msg)
    
    if not os.path.exists(st_dir):
        err_msg = (
            "the directory %s does not exist (it is expected in the "
            "kdialectspeech dataset)" % talk_dir
        )
        raise OSError(err_msg)

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

def get_kdialectspeech_files(base_dir, province_code, file_ext='json'):
    """
    This gives directory names for kdialectspeech

    Arguments
    ---------
    base_dir : str
        The base directory of kdialectspeech data
    province_code : str
        The province code that one of ('gw', 'gs', 'jl', 'jj', 'cc')
    file_ext : str
        The file extention to be selected
        ex) 'json', 'wav'

    Returns
    -------
    list
        A list containing directories of the given data directory

    """
    if province_code == 'total':
        files = (
            glob.glob(os.path.join(base_dir, 'people/talk/*/*/*.' + file_ext))
            + glob.glob(os.path.join(base_dir, 'person/say/*/*/*.' + file_ext))
            + glob.glob(os.path.join(base_dir, 'person/st/*/*/*.' + file_ext))
        )
    else:
        files = (
            glob.glob(os.path.join(base_dir, 'people/talk/' + province_code + '/*/*.' + file_ext))
            + glob.glob(os.path.join(base_dir, 'person/say/' + province_code + '/*/*.' + file_ext))
            + glob.glob(os.path.join(base_dir, 'person/st/' + province_code + '/*/*.' + file_ext))
        )

    return files

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

    csv_lines = []
    with open(json_file_path, encoding="UTF-8" ) as json_file :
        json_data = json.load(json_file)
        # data_file_name = json_data["fileName"]
        data_file_name = Path(json_file_path).stem # 확장자 제거
        data_path_dir = os.path.dirname(json_file_path)
        
        sentence_path_dir = data_path_dir.replace(base_dir, sentence_dir)
        os.makedirs(sentence_path_dir, exist_ok=True)

        # print(f'sentence_path_dir : {sentence_path_dir}')
        logger.info(f'sentence_path_dir : {sentence_path_dir}')

        input_wav_file = os.path.join(data_path_dir, data_file_name + '.wav')

        try:
            input_audio = AudioSegment.from_wav(input_wav_file)
        except:
            # print(f'wrong audio file : {input_wav_file}')
            logger.info(f'wrong audio file : {input_wav_file}')
            return csv_lines

        sentences = json_data["transcription"]["sentences"]

        for idx, sentence in enumerate(sentences) :
            #추가##############################
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
            # print(f'sentence_file_id : {sentence_file_id}')
            logger.info(f'sentence_file_path : {sentence_file_path}')
            split_wav_file(input_audio, sentence_file_path, start_time, end_time)
            
            id = sentence_file_id
            line = [id, duration, sentence_file_path, province_code, wrd]
            csv_lines.append(line)
    return csv_lines


def create_csv(base_dir, sentence_dir, province_code):
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

    json_file_list = get_kdialectspeech_files(base_dir, province_code)
    
    total_csv_lines = []
    for json_file in json_file_list:
        total_csv_lines.extend(make_sentence_data(base_dir, sentence_dir, json_file, province_code))
        # total_csv_lines = total_csv_lines + make_csv_lines(json_file, province_code) # [duration, splited_file_path,  province_code, wrd]

    total_csv_df = pd.DataFrame(total_csv_lines, columns=['ID', 'duration', 'wav', 'province_code', 'wrd'])
    total_csv_file = os.path.join(sentence_dir, 'total.csv')
    total_csv_df.to_csv(total_csv_file, encoding="UTF-8", index=False)


def main(base_dir, sentence_dir, province_code):
    create_csv(base_dir, sentence_dir, province_code)


if __name__ == "__main__":
    base_dir = "/data/MTDATA/fn-2-018/root"
    sentence_dir = "/data/MTDATA/fn-2-018/sentence"
    province_code = "total"

    print(f"start : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    main(base_dir, sentence_dir, province_code)
    print(f"end : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    