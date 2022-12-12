#!/usr/bin/env python

import datetime
import os
import wave
import glob
import librosa
from tqdm import tqdm
import logging

AUDIOFILE_FORMAT = 'wav'
SAMPLERATE = 16000
LOG_DIR = './'

logfile = os.path.join(LOG_DIR, 'data-verify-{:%Y%m%d}.log')
logging.basicConfig(filename=logfile, level=logging.INFO)

logger = logging.getLogger(__name__)
filehandler = logging.FileHandler(logfile.format(datetime.datetime.now()), encoding='utf-8')
# formatter = logging.Formatter('%(asctime)s %(name)s %(levelname)s %(message)s')
# filehandler.setFormatter(formatter)
logger.addHandler(filehandler)


### audio file 검사, 비정상적 wav 파일 검출
def audio_file_check(input_file):
    try:
        wave.open(input_file, mode='rb').close()
        return True
    except:
        logger.info(f'wrong file : {input_file}')
        return False

### 전체 파일들을 검사하여 잘못된 파일 목록 작성
### 비정상 파일의 목록을 작성
### 16000이 아닌 파일 파일의 목록을 작성
def data_veryfy(root_dir, ext=AUDIOFILE_FORMAT, samplerate=SAMPLERATE):
    check_files = (
        glob.glob(os.path.join(root_dir, 'people/talk/gs/*/*.' + ext))
        + glob.glob(os.path.join(root_dir, 'person/say/gs/*/*.' + ext))
        + glob.glob(os.path.join(root_dir, 'person/st/gs/*/*.' + ext))
    )

    audio_files = []
    wrong_files = []
    wrong_samplerate_files = {}
    logger.info(f'Checking files in {root_dir} started....')
    for file in tqdm(check_files):
        if audio_file_check(file):
            audio_files.append(file)

            sr = librosa.get_samplerate(file)
            if  sr != samplerate:   ### sample rate(frame rate) 검사, 16000이 아닌 파일 검출
                wrong_samplerate_files[file] = sr

        else:
            wrong_files.append(file)
    logger.info(f'Checking files in {root_dir} end....')

    return audio_files, wrong_files, wrong_samplerate_files


if __name__ == "__main__":
    root_dir = '/data/MTDATA/fn-2-018/root'
    audio_files_path = os.path.join(LOG_DIR, 'audio_files.txt')
    wrong_files_path = os.path.join(LOG_DIR, 'wrong_files.txt')
    wrong_samplerate_path = os.path.join(LOG_DIR, 'wrong_samplerate.txt')

    audio_files, wrong_files, wrong_samplerate_files = data_veryfy(root_dir, ext=AUDIOFILE_FORMAT, samplerate=SAMPLERATE)

    with open(audio_files_path, 'w') as f1: 
        for file_path in audio_files: 
            f1.write(f'{file_path}\n')
        f1.close()

    with open(wrong_files_path, 'w') as f2: 
        for file_path in wrong_files: 
            f2.write(f'{file_path}\n')
        f2.close()

    with open(wrong_samplerate_path, 'w') as f3: 
        for key, value in wrong_samplerate_files.items(): 
            f3.write(f'{key}:{value}\n')
        f3.close()

