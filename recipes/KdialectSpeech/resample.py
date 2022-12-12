#!/usr/bin/env python3
"""오디오 파일을 리샘플링하는 프로그램, 
확인 작업 필요
Author
 * N Park 2022
"""

import os
import glob
from pathlib import Path
import argparse
import librosa
import soundfile as sf
import wave
import datetime
import logging

SAMPLERATE = 16000
LOG_DIR = './'

logfile = os.path.join(LOG_DIR, 'resample-{:%Y%m%d%H}.log')
logging.basicConfig(filename=logfile, level=logging.INFO)

logger = logging.getLogger(__name__)
filehandler = logging.FileHandler(logfile.format(datetime.datetime.now()), encoding='utf-8')
# formatter = logging.Formatter('%(asctime)s %(name)s %(levelname)s %(message)s')
# filehandler.setFormatter(formatter)
logger.addHandler(filehandler)

### 아래 resample()은 참고만, 데이터 디렉토리의 파일을 수정할 때 사용했던 것 사용할 일 없을 것 같음.
def resample(audio_files, dir_src, dir_replace, resample_sr=SAMPLERATE):
    i = 0
    for file_name in audio_files:
        if file_name.split('.')[-1] == 'wav':
            # print(file_name)
            # print(isinstance(file_name, sf.SoundFile))
            try:
                with wave.open(file_name) as wr:
                    frame_rate = wr.getframerate()
                    logger.info(f'frame_rate : {frame_rate}')
                    wr.close()
                    logger.info(f'librosa frame_rate {librosa.get_samplerate(file_name)}')


                    if frame_rate != resample_sr:
                        logger.info(f'48k file_name, frame_rate : {file_name}, {frame_rate}')
                        save_file_name = file_name.replace(dir_src, dir_replace)
                        Path(os.path.dirname(save_file_name)).mkdir(parents=True, exist_ok=True)
                        y, sr = librosa.load(file_name, sr=frame_rate)
                        
                        # print(f'{frame_rate} : {sr}')
                        i += 1
                        logger.info(i)
                        resample = librosa.resample(y, sr, resample_sr)
                        # print(f'save_file_name : {save_file_name}')
                        # print(f'os.path.dirname : {os.path.dirname(save_file_name)}')
                        sf.write(file_name, resample, resample_sr, format='WAV', endian='LITTLE', subtype='PCM_16')
                        # sf.write(save_file_name, resample, resample_sr, format='WAV', endian='LITTLE', subtype='PCM_16')
                    # else:
                    #     shutil.copyfile(file_name, save_file_name)



            except wave.Error as e:
                logger.info(e)
                logger.info(f'wrong file : {file_name}')
                # break

            # frame_rate = librosa.get_samplerate(file_name)

            # if isinstance(file_name, sf.SoundFile):
            #     frame_rate = librosa.get_samplerate(file_name)
            # else:
            #     break
            # # frame_rate = librosa.get_samplerate(file_name)

def read_audiosample(audio_dict_file):
    with open(audio_dict_file, 'r') as f:
        lines = f.readlines()
        file_sr_dict = {}
        for l in lines:
            file_sr = l.split(':')
            # print(file_sr)
            file_sr_dict[file_sr[0]] = int(file_sr[1].strip())
        f.close()
        
    # print(file_sr_dict)
    return file_sr_dict


                
if __name__ == '__main__':
    """
    python resample.py --base_dir /data/MTDATA/root --dir_src root --dir_replace 16k --resample_sr 16000
    """
    # base_dir = '/data/MTDATA/fn-2018/'
    # file_ext = 'wav'
    
    # dir_src = 'root'
    # dir_replace = 'root-16k'
    # resample_sr=16000
    
    # parser = argparse.ArgumentParser(description='Audio file resampling')
    # parser.add_argument('--base_dir', type=str, help='base directory')
    # parser.add_argument('--dir_src', type=str, help='part of directory name to be changed')
    # parser.add_argument('--dir_replace', type=str, help='part of directory name replaced to')
    # parser.add_argument('--resample_sr', type=int, help='sample rate to be resampled')
    
    # audio_files = (
    #     glob.glob(os.path.join(base_dir + dir_src, 'people/talk/gs/*/*.' + file_ext))
    #     + glob.glob(os.path.join(base_dir + dir_src, 'person/say/gs/*/*.' + file_ext))
    #     + glob.glob(os.path.join(base_dir + dir_src, 'person/st/gs/*/*.' + file_ext))
    # )

    # # print(f'type of audio_files : {type(audio_files)}')
    # logger.info(f'number of audio_files : {len(audio_files)}')
    # # audio_files = audio_files
    
    # resample(audio_files, dir_src, dir_replace, resample_sr)

    
    audio_dict_file = 'wrong_samplerate.txt'
    logger.info(f'{audio_dict_file}의 파일들을 변환 시작-----')
    for file, sr in read_audiosample(audio_dict_file).items():
        y, sr = librosa.load(file, sr=sr)
        resample = librosa.resample(y, sr, SAMPLERATE)
        sf.write(file, resample, SAMPLERATE, format='WAV', endian='LITTLE', subtype='PCM_16')

    logger.info(f'{audio_dict_file}의 파일들을 변환 종료-----')
