#!/usr/bin/env python

'''
실행법 : python run_pipe.py run_pipe.yaml
'''

import logging
import sys
import os
import subprocess
import glob
import shutil

# from tqdm import tqdm
from pathlib import Path

import speechbrain as sb
from hyperpyyaml import load_hyperpyyaml

import boto3

from kdialectspeech.s3_download import get_s3_files
from kdialectspeech.resample import resample_audio


# print(os.path.dirname(os.path.abspath(__file__)))
# print(__file__)





if __name__ == "__main__":
    # CLI:
    print('test')
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])
    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    ##### setup logging
    logger = logging.getLogger(__name__)

    log_config = hparams["log_config"]
    log_file = hparams["log_file"]

    logger_overrides = {
        "handlers": {"file_handler": {"filename": log_file}}
    }

    # setup_logging(config_path="log-config.yaml", overrides={}, default_level=logging.INFO)
    sb.utils.logger.setup_logging(log_config, logger_overrides)
    #####

    ##### download data from s3 storage
    if 'data_download' in hparams['run_modules']:
        logger.info(f'data_download starting.....')
        # yaml에서 설정값 읽어오기 : 스토리지 접속 정보, 데이터 저장 위치
        service_name = hparams["service_name"]
        endpoint_url = hparams["endpoint_url"] 
        storage_region_name = hparams["storage_region_name"]
        access_key = hparams["access_key"]
        secret_key = hparams["secret_key"]

        s3 = boto3.client(service_name, endpoint_url=endpoint_url, aws_access_key_id=access_key,
                        aws_secret_access_key=secret_key)

        data_save_path = secret_key = hparams["data_save_path"]
        os.makedirs(data_save_path, exist_ok=True)

        bucket_name = hparams["bucket_name"]
        max_keys = hparams["max_keys"]
        key_names = hparams["key_names"]

        data_save_path = hparams["data_save_path"]
        error_file_log = hparams["error_file_log"]

        get_s3_files(s3, bucket_name, key_names, max_keys, data_save_path, error_file_log)
        #####

        ##### resample audio files
        wrong_samplerate_file = hparams["wrong_samplerate_file"]
        if os.path.isfile(wrong_samplerate_file):
            logger.info(f'{wrong_samplerate_file}의 파일들을 변환 시작-----')
            resample_audio(wrong_samplerate_file, smaplerate=16000)
            logger.info(f'{wrong_samplerate_file}의 파일들을 변환 종료-----')
        else:
            logger.info(f'{wrong_samplerate_file}이 존재하지 않아 파일들을 변환 안함-----')

    else:
        logger.info(f'data_download is not in run_modules')
    # #####

    run_provinces = hparams["run_provinces"]
    gpu_num = hparams["gpu_num"]

    for run_province in run_provinces:
        province_option = '--province_code=' + run_province

        ##### 토크나이저 실행(방언별로 각각 실행)
        # - 데이터 준비  
        # - 토큰화
        # subprocess 사용
        if 'tokenizer' in hparams['run_modules']:
            logger.info(f'tokenizer run_option : {province_option}')
            tokenizer_dir = hparams["tokenizer_dir"]
            
            subprocess.run(
                # ['python train.py hparams/5K_unigram_subword_bpe.yaml', '--province_code=gs'], 
                ['python train.py hparams/5K_unigram_subword_bpe.yaml', province_option], 
                text=True,
                # capture_output=True,
                cwd=tokenizer_dir,
                shell=True,
                stdout=subprocess.PIPE
            )
        else:
            logger.info(f'tokenizer is not in run_modules')

        
        # tokenizer_dir = hparams["tokenizer_dir"]

        # subprocess.run(
        #     ['python train.py hparams/5K_unigram_subword_bpe.yaml', '--province_code=gs'], 
        #     text=True,
        #     # capture_output=True,
        #     cwd=tokenizer_dir,
        #     shell=True,
        #     stdout=subprocess.PIPE
        # )
        #####


        ##### 언어모델
        # 언어모델 실행(방언별로 각각 실행)
        if 'lm' in hparams['run_modules']:
            logger.info(f'lm run_option : {province_option}')
            logger.info(f'gpu num : {gpu_num}')
            lm_dir = hparams["lm_dir"]
            if gpu_num > 1:
                cmd = "python -m torch.distributed.launch --nproc_per_node=" + str(gpu_num) + " train.py hparams/transformer.yaml --distributed_launch --distributed_backend='nccl'"
            else:
                cmd = 'python train.py hparams/transformer.yaml'
            subprocess.run(
                # ['python train.py hparams/transformer.yaml', '--province_code=gs'],
                # ['python train.py hparams/transformer.yaml', run_option],
                [cmd, province_option],
                text=True,
                # capture_output=True,
                cwd=lm_dir,
                shell=True,
                stdout=subprocess.PIPE
            )
        else:
            logger.info(f'lm is not in run_modules')

        #####

    # 음성인식 실행(방언별로 각각 실행)  
    # 추론 준비 : 추론에 필요한 모델 파일을 한 디렉토리에 복사


        ##### ASR
        if 'asr' in hparams['run_modules']:
            logger.info(f'asr run_option : {province_option}')
            logger.info(f'gpu num : {gpu_num}')
            asr_dir = hparams["asr_dir"]
            print(asr_dir)
            if gpu_num > 1:
                cmd = "python -m torch.distributed.launch --nproc_per_node=" + str(gpu_num) + " train.py hparams/conformer_medium.yaml --distributed_launch --distributed_backend='nccl'"
            else:
                cmd = 'python train.py hparams/conformer_medium.yaml'
            subprocess.run(
                # ['python train.py hparams/conformer_medium.yaml', '--province_code=gs'],
                # ['python train.py hparams/conformer_medium.yaml', province_option],
                [cmd, province_option],
                # python -m torch.distributed.launch --nproc_per_node=4 train.py hparams/conformer_medium.yaml --distributed_launch --distributed_backend='nccl'
                text=True,
                # capture_output=True,
                cwd=asr_dir,
                shell=True,
                stdout=subprocess.PIPE
            )

            logger.info(f'subprocess.CompletedProcess : {subprocess.CompletedProcess}')
            if subprocess.CompletedProcess.returncode == 0:
                logger.info(f'asr completed successfully')
                if hparams['copy_trained_model']:

                    train_result_dir = os.path.join(asr_dir, 'results/Conformer/7774/' + run_province + '/save')

                    lm_model =  os.path.join(train_result_dir, 'lm.ckpt')
                    tokenizer =  os.path.join(train_result_dir, 'tokenizer.ckpt')

                    best_model_dir = sorted(glob.glob(train_result_dir + '/CKPT*'), key=os.path.getmtime)[0]
                    best_model_dir = Path(best_model_dir).stem

                    best_model = os.path.join(best_model_dir + '/model.ckpt')
                    normalizer = os.path.join(best_model_dir + '/normalizer.ckpt')

                    pretrained_model_base = hparams['pretrained_model_base']
                    pretrained_model_dir = os.path.join(pretrained_model_base, run_province)

                    hparam_file = os.path.join(pretrained_model_base, 'hyperparams.yaml')

                    # copy to pretrained_model_dir
                    shutil.copy(hparam_file, pretrained_model_dir)
                    shutil.copy(lm_model, pretrained_model_dir)
                    shutil.copy(tokenizer, pretrained_model_dir)
                    shutil.copy(best_model, pretrained_model_dir)
                    shutil.copy(normalizer, pretrained_model_dir)

                    os.symlink(best_model, lm_model)


        else:
            logger.info(f'asr is not in run_modules')

        #####
