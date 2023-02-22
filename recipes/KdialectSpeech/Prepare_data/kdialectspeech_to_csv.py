"""
Data preparation.

Author
------
N Park Starcell Inc.

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

def make_csv_lines(json_file_path, province_code):
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
        dir = os.path.dirname(json_file_path)



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
            
            id = province_code + '-' + data_file_name
            line = [id, duration, json_file_path, province_code, wrd]
            csv_lines.append(line)
    return csv_lines


def create_csv(data_folder, save_folder, province_code):
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

    json_file_list = get_kdialectspeech_files(data_folder, province_code)
    
    total_csv_lines = []
    for json_file in json_file_list:
        total_csv_lines.extend(make_csv_lines(json_file, province_code))
        # total_csv_lines = total_csv_lines + make_csv_lines(json_file, province_code) # [duration, splited_file_path,  province_code, wrd]

    total_csv_df = pd.DataFrame(total_csv_lines, columns=['ID', 'duration', 'wav', 'province_code', 'wrd'])
    total_csv_file = os.path.join(save_folder, 'total.csv')
    total_csv_df.to_csv(total_csv_file, encoding="UTF-8", index=False)

### txt file은 speechbrain.tokenizers.SentencePiece 에서 만듬.


def kdialectspeech_to_csv(
    data_folder,
    save_folder,
    province_code
):
    """
    데이터 폴더, 지역코드 받아서,
    매니페스트 파일
    텍스트 파일
    만들기.
    

    확인 : 필요한 데이터 (지역별)폴더가 있는 지 확인


    This class prepares the csv files for the KdialectSpeech dataset.
    Download link: https://aihub.or.kr/aidata

    Arguments
    ---------
    data_folder : str
        Path to the folder where the original KdialectSpeech dataset is stored.
    save_folder : str
        The directory where to store the csv files.
    province_code : str


    Example
    -------
    >>> data_folder = 'datasets/KdialectSpeech'
    >>> save_folder = 'KdialectSpeech_csv'
    >>> prepare_kdialectspeech(data_folder, save_folder, cc)
    """

    ## province_folder = os.path.join(save_folder, province_code)
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
        # Path(save_folder).mkdir(parents=True, exist_ok=True)

    create_csv(data_folder, save_folder, province_code)


def main(data_foler, save_base_folder, province_code):
    save_folder = os.path.join(save_base_folder, province_code)
    kdialectspeech_to_csv(data_foler, save_folder, province_code)


if __name__ == "__main__":
    data_foler = "/data/MTDATA/fn-2-018/root"
    save_base_folder = "/data/MTDATA/fn-2-018/manifest"
    province_code = "cc"

    main(data_foler, save_base_folder, province_code)
    